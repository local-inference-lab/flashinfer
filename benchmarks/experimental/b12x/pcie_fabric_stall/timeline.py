"""Per-bucket stalled ranks (which GPU stalls when), absolute wall-clock times."""
import sys
import numpy as np
from analyze import load
path = sys.argv[1]
bucket = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
launch_ns = int(sys.argv[3]) if len(sys.argv) > 3 else 0
t_from = float(sys.argv[4]) if len(sys.argv) > 4 else 0
t_to = float(sys.argv[5]) if len(sys.argv) > 5 else 1e12
R, iters, kib, ctas, gpus, a = load(path)
S, Rd, E = a[:, 0], a[:, 1], a[:, 2]
off = np.median(Rd - Rd[0], axis=1).astype(np.int64)
Rd, E = Rd - off[:, None], E - off[:, None]
x = (E - Rd)[:, 16:] / 1e3
t = (Rd[0, 16:] - Rd[0, 16]) / 1e6
b = (t // bucket).astype(int)
nb = b.max() + 1
for k in range(nb):
    if k * bucket < t_from or k * bucket > t_to:
        continue
    m = b == k
    if not m.any():
        continue
    cnt = [(int((x[r, m] > 100).sum())) for r in range(R)]
    p50 = [float(np.median(x[r, m])) for r in range(R)]
    if sum(cnt) or len(sys.argv) > 4:
        print(f"{k*bucket:9.0f} ms  " + "  ".join(f"g{gpus[r]}:{cnt[r]:4d}/{p50[r]:6.1f}" for r in range(R)))
