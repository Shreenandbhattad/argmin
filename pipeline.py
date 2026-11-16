"""argmin pipeline: phone video -> camera poses -> 3D Gaussian splat -> compact scene file.

  video --ffmpeg--> sharpest frames --COLMAP (global SfM)--> camera poses --trainer--> splat.ply
        --splat-transform--> scene.spz (small enough for a phone)

Trainers:  msplat (Apple Silicon, Metal; default)  |  gsplat (NVIDIA GPU; set RS_TRAINER=gsplat)

Usage:  python pipeline.py scans/<name>/input.mov [--frames 180] [--iters 7000]
Progress is written to scans/<name>/status.json so the web UI can show it.
"""
import argparse, json, os, shutil, subprocess, sys, time

# Colab sets a notebook-only plotting backend that breaks the separate training Python; children inherit this.
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def log(scan: Path, stage: str, msg: str = "", pct: float | None = None, done=False, error=None):
    s = {"stage": stage, "msg": msg, "pct": pct, "done": done, "error": error, "t": time.time()}
    (scan / "status.json").write_text(json.dumps(s))
    print(f"[{stage}] {msg}", flush=True)


def run(cmd, scan: Path, logfile="log.txt"):
    with open(scan / logfile, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        r = subprocess.run(["nice", "-n", "10"] + list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed (see {scan/logfile})")


def sharpness(path: Path) -> float:
    g = Image.open(path).convert("L")
    g.thumbnail((480, 480))
    lap = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float(lap.var())


def extract_frames(video: Path, scan: Path, target: int, width: int):
    raw = scan / "raw_frames"
    shutil.rmtree(raw, ignore_errors=True); raw.mkdir()
    dur = float(subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]).strip())
    # oversample 3x, then keep the sharpest frame in each window (kills motion blur)
    fps = min(30.0, max(1.0, target * 3 / dur))
    run(["ffmpeg", "-y", "-i", video, "-vf", f"fps={fps:.3f},scale='min({width},iw)':-2",
         "-q:v", "2", raw / "f_%05d.jpg"], scan)
    files = sorted(raw.glob("*.jpg"))
    imgs = scan / "images"
    shutil.rmtree(imgs, ignore_errors=True); imgs.mkdir()
    win = max(1, len(files) // target)
    kept = 0
    for i in range(0, len(files), win):
        group = files[i:i + win]
        best = max(group, key=sharpness)
        shutil.copy(best, imgs / f"{kept:05d}.jpg"); kept += 1
    shutil.rmtree(raw)
    return kept, dur


GPU = os.environ.get("RS_GPU") == "1"      # set in the cloud, where COLMAP has CUDA
THREADS = "-1" if GPU else str(max(2, (os.cpu_count() or 8) // 2))   # on the Mac: half the cores, keeps it cool


def colmap(scan: Path, robust: bool = False):
    """robust=True uses ALIKED features + LightGlue matching: better on blank walls, but slow on CPU."""
    db, imgs, sparse = scan / "colmap.db", scan / "images", scan / "sparse"
    db.unlink(missing_ok=True); shutil.rmtree(sparse, ignore_errors=True); sparse.mkdir()
    log(scan, "features", "Finding feature points in each frame" + (" (robust mode)" if robust else ""), 20)
    feat = (["--FeatureExtraction.type", "ALIKED_N16ROT"] if robust
            else ["--SiftExtraction.max_num_features", "8192" if GPU else "4096"])
    run(["colmap", "feature_extractor", "--database_path", db, "--image_path", imgs,
         "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
         "--FeatureExtraction.use_gpu", "1" if GPU else "0", "--FeatureExtraction.num_threads", THREADS] + feat, scan)
    log(scan, "matching", "Matching neighbouring frames", 35)
    match = ["--FeatureMatching.type", "ALIKED_LIGHTGLUE"] if robust else []
    run(["colmap", "sequential_matcher", "--database_path", db,
         "--SequentialMatching.overlap", "10" if robust else "20", "--SequentialMatching.quadratic_overlap", "1",
         "--FeatureMatching.use_gpu", "1" if GPU else "0", "--FeatureMatching.num_threads", THREADS] + match, scan)
    log(scan, "poses", "Solving camera positions (global SfM)", 50)
    run(["colmap", "global_mapper", "--database_path", db, "--image_path", imgs, "--output_path", sparse], scan)
    models = sorted([p for p in sparse.iterdir() if p.is_dir()],
                    key=lambda p: (p / "images.bin").stat().st_size if (p / "images.bin").exists() else 0)
    if not models:
        raise RuntimeError("COLMAP could not reconstruct cameras - try a slower, overlapping video")
    best = models[-1]
    log(scan, "undistort", "Undistorting images", 60)
    out = scan / "dataset"
    shutil.rmtree(out, ignore_errors=True)
    run(["colmap", "image_undistorter", "--image_path", imgs, "--input_path", best,
         "--output_path", out, "--output_type", "COLMAP"], scan)
    # msplat expects sparse/0
    s = out / "sparse"
    if (s / "cameras.bin").exists():
        (s / "0").mkdir(exist_ok=True)
        for f in list(s.glob("*.bin")):
            f.rename(s / "0" / f.name)
    n = len(list((out / "images").glob("*")))
    total = len(list(imgs.glob("*.jpg")))
    return out, n, total


def add_depth_points(scan: Path, dataset: Path):
    """Seed the splat with Depth Anything 3 points so blank walls get geometry. Optional + safe:
    if DA3 isn't installed or its depth disagrees with COLMAP, the scene trains as before."""
    root = Path(__file__).resolve().parent
    py = root / ".venv-da3" / "bin" / "python"
    if not py.exists():
        return 0
    npz = scan / "da3_pts.npz"
    npz.unlink(missing_ok=True)
    log(scan, "depth", "Estimating depth for blank surfaces", 62)
    try:
        run([py, root / "da3_init.py", dataset / "images", scan / "sparse_txt", npz], scan)
    except RuntimeError:
        return 0
    if not npz.exists():
        return 0
    d = np.load(npz)
    xyz, rgb = d["xyz"], d["rgb"]
    txt = scan / "sparse_txt"
    ptxt = txt / "points3D.txt"
    ids = [int(l.split()[0]) for l in ptxt.read_text().splitlines() if l and l[0] != "#"]
    base = max(ids + [0]) + 1
    with open(ptxt, "a") as f:
        for i in range(len(xyz)):
            f.write(f"{base+i} {xyz[i,0]:.5f} {xyz[i,1]:.5f} {xyz[i,2]:.5f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]} 1.0\n")
    run(["colmap", "model_converter", "--input_path", txt,
         "--output_path", dataset / "sparse" / "0", "--output_type", "BIN"], scan)
    return len(xyz)


def train(scan: Path, dataset: Path, iters: int, downscales: int = 2):
    import msplat
    ds = msplat.load_dataset(str(dataset))
    cfg = msplat.TrainingConfig(iterations=iters, num_downscales=downscales)
    cfg.keep_crs = True  # stay in COLMAP coordinates so view.json poses line up
    tr = msplat.GaussianTrainer(ds, cfg)
    t0 = time.time()

    def cb(s):
        log(scan, "training", f"step {s.iteration}/{iters} · {s.splat_count:,} splats · {time.time()-t0:.0f}s",
            62 + 33 * s.iteration / iters)
    tr.train(cb, callback_every=100)
    ply = scan / "splat.ply"
    tr.export_ply(str(ply))
    return ply


def train_gsplat(scan: Path, dataset: Path, iters: int, downscales: int = 2) -> Path:
    """gsplat 1.4 reference trainer (NVIDIA GPU, prebuilt kernels). Keeps COLMAP coordinates so view.json lines up.
    Env: GSPLAT_DIR (repo with examples/), GSPLAT_PYTHON (its python), GSPLAT_SHIM (dir holding a fused_ssim.py)."""
    import re
    gdir = Path(os.environ.get("GSPLAT_DIR", "/content/gsplat"))
    py = os.environ.get("GSPLAT_PYTHON", sys.executable)
    out = scan / "gsplat_out"
    cmd = [py, str(gdir / "examples" / "simple_trainer.py"), "default",
           "--data_dir", str(dataset), "--data_factor", "1", "--result_dir", str(out),
           "--max_steps", str(iters), "--eval_steps", str(iters), "--save_steps", str(iters),
           "--disable_viewer", "--no-normalize-world-space", "--test_every", "20"]
    env = dict(os.environ)
    if os.environ.get("GSPLAT_SHIM"):
        env["PYTHONPATH"] = os.environ["GSPLAT_SHIM"] + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    log(scan, "training", "Training the 3D scene on the GPU", 62)
    proc = subprocess.Popen(cmd, cwd=str(gdir / "examples"), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    tail, last = b"", 0.0
    with open(scan / "log.txt", "ab") as lf:
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            lf.write(chunk); lf.flush()
            tail = (tail + chunk)[-600:]
            if time.time() - last > 4:
                m = re.findall(rb"(\d+)/(\d+) \[", tail)
                if m:
                    a_, b_ = int(m[-1][0]), int(m[-1][1])
                    if b_ >= iters // 2:
                        log(scan, "training", f"step {a_}/{b_} · {time.time()-t0:.0f}s", 62 + 33 * a_ / max(b_, 1))
                        last = time.time()
    if proc.wait() != 0:
        raise RuntimeError("GPU training failed (see log.txt)")
    ckpts = sorted((out / "ckpts").glob("ckpt_*_rank0.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise RuntimeError("Training finished but saved no checkpoint")
    final = scan / "splat.ply"
    run([py, Path(__file__).resolve().parent / "export_ckpt.py", ckpts[-1], final], scan)
    shutil.rmtree(out, ignore_errors=True)
    return final


def qvec2rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])


def export_view(scan: Path, dataset: Path):
    """Camera path + gravity 'up' + room centre, so the viewer opens upright, from where you stood."""
    txt = scan / "sparse_txt"; txt.mkdir(exist_ok=True)
    run(["colmap", "model_converter", "--input_path", dataset / "sparse" / "0",
         "--output_path", txt, "--output_type", "TXT"], scan)
    cams = []
    lines = [l for l in (txt / "images.txt").read_text().split("\n") if not l.startswith("#")]
    for l in lines[::2]:
        if not l.strip():
            continue
        p = l.split()
        R = qvec2rot(list(map(float, p[1:5]))); t = np.array(list(map(float, p[5:8])))
        c = -R.T @ t
        cams.append((p[9], c, R.T @ [0, 0, 1], R.T @ [0, -1, 0]))  # COLMAP: +z forward, -y up
    cams.sort(key=lambda x: x[0])
    ups = np.array([c[3] for c in cams]); up = ups.mean(0); up /= np.linalg.norm(up)
    pts = []
    for l in (txt / "points3D.txt").read_text().splitlines():
        if l and not l.startswith("#"):
            pts.append(list(map(float, l.split()[1:4])))
    pts = np.array(pts) if pts else np.array([c[1] for c in cams])
    centre = np.median(pts, 0)
    r = lambda v: [round(float(x), 5) for x in v]
    view = {"up": r(up), "center": r(centre),
            "cams": [{"p": r(c), "f": r(f), "u": r(u)} for _, c, f, u in cams]}
    (scan / "view.json").write_text(json.dumps(view))


def process_video(video: Path, frames=200, width=1600, iters=15000, downscales=0, no_depth=False, trainer=None):
    """The whole pipeline. `trainer(scan, dataset, iters, downscales) -> ply path` defaults to msplat (Mac)."""
    scan = video.parent
    t0 = time.time()
    try:
        log(scan, "frames", "Extracting sharpest frames", 5)
        n, dur = extract_frames(video, scan, frames, width)
        log(scan, "frames", f"{n} frames from {dur:.0f}s of video", 15)
        dataset, reg, n = colmap(scan)
        if reg < 0.85 * n:   # too many frames lost: retry once with the slower, sturdier features
            log(scan, "poses", f"Only {reg}/{n} frames placed. Retrying in robust mode", 20)
            prev = scan / "dataset_prev"
            shutil.rmtree(prev, ignore_errors=True); dataset.rename(prev)
            d2, reg2, n2 = colmap(scan, robust=True)
            if reg2 > reg:
                dataset, reg, n = d2, reg2, n2
                shutil.rmtree(prev, ignore_errors=True)
            else:   # robust mode didn't help: put the first result back
                shutil.rmtree(d2, ignore_errors=True); prev.rename(d2)
                dataset = d2
        log(scan, "poses", f"{reg}/{n} frames placed in 3D", 61)
        export_view(scan, dataset)
        extra = 0 if no_depth else add_depth_points(scan, dataset)
        ply = (trainer or train)(scan, dataset, iters, downscales)
        st = Path(__file__).resolve().parent / "node_modules" / ".bin" / "splat-transform"
        st = st if st.exists() else "splat-transform"
        log(scan, "cleanup", "Removing floating blobs", 95)
        clean = scan / "splat_clean.ply"
        try:
            run([st, "-w", ply, "-F", clean], scan)
            src = clean
        except RuntimeError:
            src = ply
        log(scan, "compress", "Compressing for the phone", 98)
        run([st, "-w", src, "--spz-version", "3", scan / "scene.spz"], scan)
        clean.unlink(missing_ok=True)
        meta = {"frames": n, "registered": reg, "depth_points": extra, "seconds": round(time.time() - t0)}
        (scan / "meta.json").write_text(json.dumps(meta))
        log(scan, "done", f"Done in {meta['seconds']}s ({reg}/{n} frames used)", 100, done=True)
    except Exception as e:
        log(scan, "error", str(e), None, error=str(e))
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--downscales", type=int, default=0,
                    help="progressive low-res warm-up levels; 0 trains at full resolution (sharper, slower)")
    ap.add_argument("--no-depth", action="store_true", help="skip the Depth Anything 3 step")
    a = ap.parse_args()
    try:
        trainer = train_gsplat if os.environ.get("RS_TRAINER") == "gsplat" else None
        process_video(a.video, a.frames, a.width, a.iters, a.downscales, a.no_depth, trainer)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
