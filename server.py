"""Upload server: a phone sends a video (in small chunks) and it is turned into a 3D splat automatically.

Runs locally (macOS) or on Google Colab. Optional env:  RS_KEY=<secret>  (require ?key=... once per device)
                                                    PORT=8800
"""
import json, os, queue, re, shutil, socket, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
SCANS = ROOT / "scans"; SCANS.mkdir(exist_ok=True)
PORT = int(os.environ.get("PORT", 8800))
KEY = os.environ.get("RS_KEY", "")
jobs: "queue.Queue[tuple[Path, dict]]" = queue.Queue()
uploads: dict = {}          # upload id -> (video path, status path)
lock = threading.Lock()


def worker():
    while True:
        video, opts = jobs.get()
        cmd = [sys.executable, str(ROOT / "pipeline.py"), str(video),
               "--frames", str(opts.get("frames", 200)), "--iters", str(opts.get("iters", 15000)),
               "--downscales", str(opts.get("ds", 0))]
        if shutil.which("caffeinate"):      # macOS: keep the Mac awake while processing
            cmd = ["caffeinate", "-is"] + cmd
        with open(video.parent / "log.txt", "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def gate(self) -> bool:
        """True if the request may proceed. With RS_KEY set, a device must open ?key=... once."""
        if not KEY:
            return True
        u = urlparse(self.path)
        if parse_qs(u.query).get("key", [""])[0] == KEY:
            self.send_response(302)
            self.send_header("Set-Cookie", f"rs={KEY}; Max-Age=31536000; Path=/; HttpOnly; Secure; SameSite=Lax")
            self.send_header("Location", u.path or "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        if f"rs={KEY}" not in (self.headers.get("Cookie") or ""):
            self.send_response(401)
            msg = b"Open the full link (with ?key=...) once on this device."
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return False
        return True

    def do_GET(self):
        if not self.gate():
            return
        u = urlparse(self.path)
        if u.path == "/":
            self.path = "/web/index.html"
        elif u.path == "/view":
            self.path = "/web/view.html"
        elif u.path == "/api/scans":
            out = []
            for d in sorted(SCANS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if not d.is_dir():
                    continue
                st = d / "status.json"
                try:
                    s = json.loads(st.read_text())
                except Exception:
                    s = {"stage": "queued", "msg": "Waiting in queue", "pct": 0}
                out.append({"name": d.name, **s, "ready": (d / "scene.spz").exists()})
            return self.json(out)
        elif u.path.startswith("/scans/"):
            parts = u.path.split("/")
            if len(parts) != 4 or parts[3] not in ("scene.spz", "view.json", "splat.ply", "meta.json"):
                return self.send_error(404)
        elif not u.path.startswith("/web/"):
            return self.send_error(404)
        return super().do_GET()

    def do_POST(self):
        if not self.gate():
            return
        u = urlparse(self.path)
        if u.path != "/upload":
            return self.send_error(404)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        uid = re.sub(r"[^a-zA-Z0-9]", "", q.get("id", "x"))[:40] or "x"
        part, parts = int(q.get("part", 0)), int(q.get("parts", 1))
        with lock:
            if part == 0 or uid not in uploads:
                name = re.sub(r"[^a-zA-Z0-9_-]+", "-", q.get("name", "room")).strip("-")[:40] or "room"
                name = f"{time.strftime('%m%d-%H%M')}-{name}"
                ext = re.sub(r"[^a-z0-9]", "", q.get("ext", "mov").lower())[:5] or "mov"
                d = SCANS / name; d.mkdir(parents=True, exist_ok=True)
                video = d / f"input.{ext}"
                video.write_bytes(b"")
                uploads[uid] = video
            video = uploads[uid]
        remaining = int(self.headers.get("Content-Length", 0))
        with open(video, "ab") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk); remaining -= len(chunk)
        if part == parts - 1:
            (video.parent / "status.json").write_text(json.dumps({"stage": "queued", "msg": "Waiting in queue", "pct": 0}))
            jobs.put((video, {"frames": min(400, int(q.get("frames", 200))), "iters": min(30000, int(q.get("iters", 15000))),
                              "ds": min(3, int(q.get("ds", 0)))}))
            uploads.pop(uid, None)
        self.json({"ok": True, "name": video.parent.name, "done": part == parts - 1})


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    print(f"Open on your iPhone:  http://{lan_ip()}:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
