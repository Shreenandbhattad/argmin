"""A/B experiment: train from COLMAP points alone vs COLMAP + Depth Anything 3 points; report held-out PSNR/SSIM.

Usage:  python tools/ab_depth_init.py scans/<name> <iterations> <downscales>
Needs scans/<name>/dataset (pipeline output) and scans/<name>/da3_pts.npz (from da3_init.py).
"""
import sys, shutil, subprocess, time, numpy as np
from pathlib import Path
import msplat
scan = Path(sys.argv[1]); iters = int(sys.argv[2]); down = int(sys.argv[3])

def add_points(src: Path, dst: Path, npz: Path):
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
    txt = dst / "sparse_txt"; txt.mkdir()
    subprocess.run(["colmap", "model_converter", "--input_path", dst/"sparse"/"0", "--output_path", txt, "--output_type", "TXT"], check=True, capture_output=True)
    d = np.load(npz); xyz, rgb = d["xyz"], d["rgb"]
    base = max([int(l.split()[0]) for l in (txt/"points3D.txt").read_text().splitlines() if l and l[0] != "#"] + [0]) + 1
    with open(txt/"points3D.txt", "a") as f:
        for i, (p, c) in enumerate(zip(xyz, rgb)):
            f.write(f"{base+i} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 1.0\n")
    subprocess.run(["colmap", "model_converter", "--input_path", txt, "--output_path", dst/"sparse"/"0", "--output_type", "BIN"], check=True, capture_output=True)

def run(ds_path, label):
    ds = msplat.load_dataset(str(ds_path), eval_mode=True)
    cfg = msplat.TrainingConfig(iterations=iters, num_downscales=down)
    tr = msplat.GaussianTrainer(ds, cfg); t = time.time()
    tr.train(lambda s: None, callback_every=1000)
    m = tr.evaluate()
    print(f"{label:10s} PSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}  train {time.time()-t:.0f}s", flush=True)

ds = scan / "dataset"
add_points(ds, scan / "dataset_da3", scan / "da3_pts.npz")
run(ds, "colmap"); run(scan / "dataset_da3", "colmap+DA3")
