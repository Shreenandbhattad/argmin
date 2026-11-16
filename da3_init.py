"""Dense depth prior: Depth Anything 3 (MPS) -> extra 3D points in COLMAP's coordinates.

SIFT finds almost nothing on blank walls, so the splat trainer starts with holes there.
DA3 predicts depth for every pixel; we align it (per chunk of frames) to COLMAP's camera
centres with a similarity transform and add the confident points as splat initialisation.

Run with the .venv-da3 interpreter:
  python da3_init.py <images_dir> <colmap_txt_dir> <out.npz> [--chunk 24] [--max-points 400000]
Exit code 0 even when skipped; the pipeline checks whether out.npz exists.
"""
import argparse, os, sys, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from pathlib import Path


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


def read_colmap_centres(txt: Path):
    out = {}
    lines = [l for l in (txt / "images.txt").read_text().split("\n") if not l.startswith("#")]
    for l in lines[::2]:
        p = l.split()
        if len(p) < 10:
            continue
        R = qvec2rot(list(map(float, p[1:5]))); t = np.array(list(map(float, p[5:8])))
        out[p[9]] = -R.T @ t
    return out


def umeyama(src, dst):
    """Similarity transform dst ~= s R src + t (least squares)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images"); ap.add_argument("colmap_txt"); ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--model", default="depth-anything/DA3-BASE")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from depth_anything_3.api import DepthAnything3

    centres = read_colmap_centres(Path(a.colmap_txt))
    names = sorted(n for n in centres if (Path(a.images) / n).exists())
    if len(names) < 8:
        print("SKIP: too few registered frames"); return
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(a.model).to(dev)

    # overlapping chunks: each aligned to COLMAP on its own, so DA3 drift can't accumulate
    step = max(4, a.chunk - 4)
    chunks = [names[i:i + a.chunk] for i in range(0, max(1, len(names) - 3), step)]
    pts_all, col_all, resid = [], [], []
    per_chunk = max(1000, a.max_points // len(chunks))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for ci, ch in enumerate(chunks):
        if len(ch) < 6:
            continue
        pred = model.inference([str(Path(a.images) / n) for n in ch])
        D = np.asarray(pred.depth); C = np.asarray(pred.conf) if pred.conf is not None else np.ones_like(D)
        E = np.asarray(pred.extrinsics); K = np.asarray(pred.intrinsics)
        if E.shape[1] == 3:  # 3x4 -> 4x4
            E = np.concatenate([E, np.tile(np.array([[[0, 0, 0, 1.0]]]), (len(E), 1, 1))], 1)
        R_wc = E[:, :3, :3]; t_wc = E[:, :3, 3]
        cen_da3 = np.einsum("nji,nj->ni", R_wc, -t_wc)       # C = -R^T t
        cen_col = np.stack([centres[n] for n in ch])
        s, R, t = umeyama(cen_da3, cen_col)
        err = np.linalg.norm((s * (R @ cen_da3.T)).T + t - cen_col, axis=1)
        extent = np.linalg.norm(cen_col.max(0) - cen_col.min(0)) + 1e-9
        resid.append(float(err.mean() / extent))
        if err.mean() / extent > 0.10:     # DA3 and COLMAP disagree on this chunk: don't trust it
            print(f"chunk {ci}: skipped (alignment residual {err.mean()/extent:.2f})"); continue
        thr = np.percentile(C, 40)
        for i, n in enumerate(ch):
            d = D[i]; h, w = d.shape
            gy, gx = np.gradient(d)
            edge = np.hypot(gx, gy) / (d + 1e-6) > 0.05      # drop depth-discontinuity pixels
            ok = (C[i] >= thr) & (~edge) & np.isfinite(d) & (d > 0)
            ys, xs = np.nonzero(ok)
            if len(ys) > per_chunk // len(ch):
                sel = rng.choice(len(ys), per_chunk // len(ch), replace=False); ys, xs = ys[sel], xs[sel]
            z = d[ys, xs]
            Ki = K[i]
            cam = np.stack([(xs - Ki[0, 2]) * z / Ki[0, 0], (ys - Ki[1, 2]) * z / Ki[1, 1], z], 1)
            world_da3 = (R_wc[i].T @ (cam - t_wc[i]).T).T
            pts_all.append((s * (R @ world_da3.T)).T + t)
            img = Image.open(Path(a.images) / n).convert("RGB").resize((w, h))
            col_all.append(np.asarray(img)[ys, xs])
        print(f"chunk {ci+1}/{len(chunks)} aligned, residual {resid[-1]:.3f}, {time.time()-t0:.0f}s", flush=True)
        if dev == "mps":
            torch.mps.empty_cache()

    if not pts_all:
        print("SKIP: no chunk aligned"); return
    P = np.concatenate(pts_all); Cc = np.concatenate(col_all)
    # voxel downsample so dense overlapping views don't make 10M identical points
    lo = np.percentile(P, 1, 0); hi = np.percentile(P, 99, 0)
    inb = np.all((P >= lo - (hi - lo)) & (P <= hi + (hi - lo)), 1)
    P, Cc = P[inb], Cc[inb]
    vox = np.linalg.norm(hi - lo) / 400.0
    key = np.floor((P - P.min(0)) / vox).astype(np.int64)
    _, idx = np.unique(key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791, return_index=True)
    P, Cc = P[idx], Cc[idx]
    if len(P) > a.max_points:
        sel = rng.choice(len(P), a.max_points, replace=False); P, Cc = P[sel], Cc[sel]
    np.savez(a.out, xyz=P.astype(np.float32), rgb=Cc.astype(np.uint8), residual=np.mean(resid))
    print(f"OK: {len(P):,} points, mean alignment residual {np.mean(resid):.3f}")


if __name__ == "__main__":
    main()
