"""Builds colab/room_splat.ipynb from the real project files (so the notebook never drifts)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["pipeline.py", "server.py", "export_ckpt.py", "web/index.html", "web/view.html"]
cells = []


def md(t):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)})


def code(t, name=None):
    meta = {}
    if name and "%%" in t:          # a cell magic must be the first line, so the title goes above
        md(f"### {name}")
    elif name:
        t = f"#@title {name}\n" + t.strip("\n")
        meta = {"cellView": "form"}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [],
                  "source": t.strip("\n").splitlines(True)})


md("""
# argmin on Colab (free GPU)

Film a room on your iPhone, upload it from Safari, and this notebook turns it into a 3D scene you can
walk around on your phone. **Your Mac only needs this tab open.**

1. **Runtime → Change runtime type → T4 GPU.**
2. Run the cells **top to bottom** (▶). Steps 1–4 install everything (about 10–15 minutes, once per session).
3. The last cell prints a link and a QR code. Open it on your iPhone.

Scenes live only in this session. Open a scene and use *Save* in your browser, or download from the Files panel on the left (`roomsplat/scans`).
""")

code('''
import subprocess, sys
out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
assert out.returncode == 0 and out.stdout.strip(), "No GPU. Use Runtime → Change runtime type → T4 GPU, then run again."
print("GPU OK")
''', "1. Check the GPU")

code('''
%%bash
set -e
cd /content
if [ ! -x colmap-env/bin/colmap ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
  ./bin/micromamba create -y -q -p /content/colmap-env -c conda-forge "colmap=4.2.1=cuda*"
fi
/content/colmap-env/bin/colmap help | head -2
which ffmpeg ffprobe
''', "2. Install COLMAP (GPU) - about 3 min")

code('''
%%bash
set -e
cd /content
pip install -q uv qrcode
if [ ! -x /content/gs-env/bin/python ]; then uv venv -q /content/gs-env --python 3.10; fi
P=/content/gs-env/bin/python
# PyTorch 2.4 + the PREBUILT gsplat 1.4 GPU code: no compiling (compiling took over an hour on Colab)
uv pip install -q -p $P "numpy<2" torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install -q -p $P "gsplat==1.4.0+pt24cu124" --extra-index-url https://docs.gsplat.studio/whl/pt24cu124 --index-strategy unsafe-best-match
[ -d gsplat ] || git clone -q --depth 1 --branch v1.4.0 https://github.com/nerfstudio-project/gsplat
uv pip install -q -p $P viser nerfview==0.0.2 "imageio[ffmpeg]" scipy scikit-learn tqdm "torchmetrics[image]" opencv-python \\
  "tyro>=0.8.8" Pillow tensorboard tensorly pyyaml matplotlib \\
  "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
# splat-transform shrinks the scene for the phone
if ! command -v splat-transform >/dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null 2>&1
  apt-get install -y -qq nodejs >/dev/null 2>&1
  npm install -g @playcanvas/splat-transform@3.11.2 >/dev/null 2>&1
fi
splat-transform --version 2>&1 | head -1
$P -c "import torch, gsplat; print('torch', torch.__version__, '| gsplat', gsplat.__version__, '| cuda', torch.cuda.is_available())"
''', "3. Install the trainer + compressor - about 4 min")

code('''
# The trainer wants a "fused_ssim" package that has to be compiled. This is a small drop-in written in plain PyTorch.
import os, subprocess, textwrap
os.makedirs("/content/gs-shim", exist_ok=True)
open("/content/gs-shim/fused_ssim.py", "w").write(textwrap.dedent("""
    import torch, torch.nn.functional as F
    def _win(C, dev, dt, size=11, sigma=1.5):
        x = torch.arange(size, device=dev, dtype=dt) - size // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        return (g[:, None] * g[None, :]).expand(C, 1, size, size).contiguous()
    def fused_ssim(img1, img2, padding="same", train=True):
        C = img1.shape[1]; w = _win(C, img1.device, img1.dtype); pad = 5 if padding == "same" else 0
        c = lambda t: F.conv2d(t, w, padding=pad, groups=C)
        mu1, mu2 = c(img1), c(img2)
        s11, s22, s12 = c(img1 * img1) - mu1 ** 2, c(img2 * img2) - mu2 ** 2, c(img1 * img2) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))).mean()
"""))
test = textwrap.dedent("""
    import sys; sys.path.insert(0, "/content/gs-shim")
    import torch
    from fused_ssim import fused_ssim
    from gsplat.rendering import rasterization
    m = torch.randn(100, 3, device="cuda", requires_grad=True); q = torch.nn.functional.normalize(torch.randn(100, 4, device="cuda"), dim=-1)
    s = torch.rand(100, 3, device="cuda") * 0.1 + 0.01; o = torch.rand(100, device="cuda"); c = torch.rand(100, 3, device="cuda")
    K = torch.tensor([[[100., 0, 64], [0, 100., 64], [0, 0, 1]]], device="cuda"); V = torch.eye(4, device="cuda")[None]; V[0, 2, 3] = 5
    img = rasterization(m, q, s, o, c, V, K, 128, 128)[0]
    (img.sum() + fused_ssim(img.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2).detach().flip(3), padding="valid")).backward()
    print("GPU rendering + training math work on", torch.cuda.get_device_name(0))
""")
r = subprocess.run(["/content/gs-env/bin/python", "-c", test], capture_output=True, text=True)
print(r.stdout.strip() or r.stderr[-2500:])
assert r.returncode == 0, "GPU check failed. Open an issue with the error above."
''', "4. Quick GPU check - under a minute")

code("import os\nos.makedirs('/content/roomsplat/web', exist_ok=True)\nos.makedirs('/content/roomsplat/scans', exist_ok=True)", "Prepare folders")
for f in FILES:
    body = (ROOT / f).read_text()
    code(f"%%writefile /content/roomsplat/{f}\n{body}")
md("*(The five cells above just write the app's files. Run them, nothing to change.)*")

code('''
# OPTIONAL: test the whole GPU pipeline on a sample (a stuffed fox, 50 photos). Takes about 5 minutes.
import os, subprocess, sys
os.environ.update(RS_GPU="1", RS_TRAINER="gsplat", GSPLAT_DIR="/content/gsplat", GSPLAT_PYTHON="/content/gs-env/bin/python", GSPLAT_SHIM="/content/gs-shim",
                  PATH="/content/colmap-env/bin:/usr/local/cuda/bin:" + os.environ["PATH"])
os.chdir("/content")
if not os.path.exists("ngp"):
    subprocess.run("git clone -q --depth 1 --filter=blob:none --sparse https://github.com/NVlabs/instant-ngp ngp && cd ngp && git sparse-checkout set data/nerf/fox", shell=True, check=True)
os.makedirs("/content/roomsplat/scans/sample-fox", exist_ok=True)
subprocess.run("ffmpeg -y -v error -framerate 5 -pattern_type glob -i '/content/ngp/data/nerf/fox/images/*.jpg' -c:v libx264 -pix_fmt yuv420p "
               "-vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' /content/roomsplat/scans/sample-fox/input.mp4", shell=True, check=True)
r = subprocess.run([sys.executable, "pipeline.py", "scans/sample-fox/input.mp4", "--frames", "50", "--iters", "7000"], cwd="/content/roomsplat")
print("exit", r.returncode)
print(open("/content/roomsplat/scans/sample-fox/status.json").read())
if r.returncode:
    print(open("/content/roomsplat/scans/sample-fox/log.txt").read()[-2500:])
''', "5. (Optional) Self-test on a sample scene")

code('''
import os, re, secrets, subprocess, sys, time, urllib.request, stat
from IPython.display import display

key = open("/content/rs_key").read() if os.path.exists("/content/rs_key") else secrets.token_urlsafe(16)
open("/content/rs_key", "w").write(key)
os.environ.update(RS_GPU="1", RS_TRAINER="gsplat", GSPLAT_DIR="/content/gsplat", GSPLAT_PYTHON="/content/gs-env/bin/python", GSPLAT_SHIM="/content/gs-shim", RS_KEY=key, PORT="8800",
                  PATH="/content/colmap-env/bin:/usr/local/cuda/bin:" + os.environ["PATH"])
subprocess.run("pkill -f '[r]oomsplat/server.py'; pkill cloudflared", shell=True)
subprocess.Popen([sys.executable, "/content/roomsplat/server.py"], cwd="/content/roomsplat",
                 stdout=open("/content/server.log", "w"), stderr=subprocess.STDOUT)

cf = "/content/cloudflared"
if not os.path.exists(cf):
    urllib.request.urlretrieve("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", cf)
    os.chmod(cf, os.stat(cf).st_mode | stat.S_IEXEC)
tun = subprocess.Popen([cf, "tunnel", "--url", "http://localhost:8800", "--no-autoupdate"],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
url, ready, t0 = None, False, time.time()
while time.time() - t0 < 90 and not (url and ready):      # the address exists before it is connected: wait for both
    line = tun.stdout.readline()
    if not line:
        break
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", line)
    if m:
        url = m.group(0)
    if "Registered tunnel connection" in line:
        ready = True
assert url, "Could not open a public link. Run this cell again."
import threading, urllib.error
threading.Thread(target=lambda: [None for _ in tun.stdout], daemon=True).start()   # keep the tunnel's log pipe drained
for _ in range(30):      # prove the link works from outside before showing it (401 = the server answered)
    try:
        urllib.request.urlopen(url, timeout=5); break
    except urllib.error.HTTPError:
        break
    except Exception:
        time.sleep(2)
else:
    print("Warning: the link is not reachable yet. Wait a minute and try it; if it still fails run this cell again.")
link = f"{url}/?key={key}"

import qrcode
display(qrcode.make(link).get_image().resize((260, 260)))
print("\\nOpen this on your iPhone (scan the QR or type it):\\n\\n" + link + "\\n")
print("The server keeps running in the background. Keep this Colab tab open (it can sleep if closed).")
''', "6. Start argmin (open the link on your iPhone)")

# RESET: build the all-in-one cell from step 6's own code, so the two can never drift apart
_start = next("".join(c["source"]) for c in cells if "6. Start argmin" in "".join(c["source"]))
_head = '''
import os, shutil, subprocess, time
# 1) kill every leftover process: server, running job, COLMAP, trainer, tunnel
subprocess.run("pkill -9 -f '[r]oomsplat/server.py'; pkill -9 -f '[r]oomsplat/pipeline.py'; "
               "pkill -9 -f '[s]imple_trainer.py'; pkill -9 colmap; pkill -9 splat-transform; pkill -9 cloudflared", shell=True)
time.sleep(2)
assert os.path.exists("/content/roomsplat/server.py"), "App files are missing. Run steps 1-4 and the file cells first."
# 2) wipe all scenes (this also empties the queue)
shutil.rmtree("/content/roomsplat/scans", ignore_errors=True)
os.makedirs("/content/roomsplat/scans", exist_ok=True)
os.environ["MPLBACKEND"] = "Agg"
print("Cleaned up. Starting fresh...")
'''
code(_head + _start.split("\n", 1)[1], "RESET: stop everything, clear all scenes, start fresh (use when stuck)")

code('''
import os, json, time, psutil

killed = []
for p in psutil.process_iter(["pid", "cmdline"]):
    cmd = p.info["cmdline"] or []
    if any(c.endswith("roomsplat/pipeline.py") for c in cmd):      # the job that is running now
        scan = next((c for c in cmd if "/scans/" in c or c.startswith("scans/")), None)
        for ch in p.children(recursive=True):                      # stop COLMAP / the trainer first
            try: ch.kill()
            except psutil.Error: pass
        try: p.kill()
        except psutil.Error: pass
        if scan:
            d = os.path.dirname(os.path.join("/content/roomsplat", scan) if not scan.startswith("/") else scan)
            open(os.path.join(d, "status.json"), "w").write(json.dumps(
                {"stage": "error", "msg": "Cancelled", "pct": None, "done": False, "error": "Cancelled", "t": time.time()}))
            killed.append(os.path.basename(d))
print("Cancelled:", killed if killed else "nothing was running. The next queued scene starts by itself.")
''', "7. Cancel the scene that is running now (the next queued one starts automatically)")

code('''
import pathlib, json
for p in sorted(pathlib.Path("/content/roomsplat/scans").glob("*/status.json"), key=lambda p: p.stat().st_mtime):
    try: print(p.parent.name.ljust(28), json.loads(p.read_text()).get("msg"))
    except Exception: print(p.parent.name)
''', "8. Show the status of all scenes")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "colab" / "room_splat.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote", len(cells), "cells")
