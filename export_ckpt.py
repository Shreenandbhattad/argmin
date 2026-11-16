"""gsplat checkpoint -> standard 3DGS .ply (what splat-transform and every splat viewer reads).
Usage: python export_ckpt.py <ckpt.pt> <out.ply>   (run with the gsplat python: needs torch)"""
import sys
import numpy as np
import torch

ck = torch.load(sys.argv[1], map_location="cpu")["splats"]
means = ck["means"].float().numpy()
scales = ck["scales"].float().numpy()                 # log-space, as the PLY format expects
quats = torch.nn.functional.normalize(ck["quats"].float(), dim=-1).numpy()   # w, x, y, z
opac = ck["opacities"].float().numpy()                # logit-space, as the PLY format expects
sh0 = ck["sh0"].float()                               # N, 1, 3
shN = ck["shN"].float()                               # N, K-1, 3
n = len(means)
f_dc = sh0.reshape(n, 3).numpy()
f_rest = shN.transpose(1, 2).reshape(n, -1).numpy()   # channel-major, like the original 3DGS
m = f_rest.shape[1]
data = np.concatenate([means, np.zeros((n, 3), np.float32), f_dc, f_rest, opac[:, None], scales, quats], 1).astype("<f4")
names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(m)] \
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
assert data.shape[1] == len(names)
hdr = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n + "".join(f"property float {k}\n" for k in names) + "end_header\n"
with open(sys.argv[2], "wb") as f:
    f.write(hdr.encode()); f.write(data.tobytes())
print(f"wrote {n:,} splats")
