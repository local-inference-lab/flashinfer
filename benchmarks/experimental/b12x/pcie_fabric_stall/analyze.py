"""Summarise p2pstall dumps: stall durations, which GPUs stall together, and time structure.

usage: analyze.py FILE.bin [--stall-us 100]
"""
import sys
import numpy as np

def load(path):
    raw = open(path, "rb").read()
    R, iters, kib, ctas = np.frombuffer(raw[:16], dtype=np.int32)
    gpus = np.frombuffer(raw[16:16 + 4 * R], dtype=np.int32)
    arr = np.frombuffer(raw[16 + 4 * R:], dtype=np.uint64).reshape(R, 3, iters).astype(np.int64)
    return int(R), int(iters), int(kib), int(ctas), gpus, arr

def main():
    path = sys.argv[1]
    thr = float(sys.argv[sys.argv.index("--stall-us") + 1]) if "--stall-us" in sys.argv else 100.0
    R, iters, kib, ctas, gpus, a = load(path)
    S, Rd, E = a[:, 0], a[:, 1], a[:, 2]
    # Align clocks on barrier exits.
    off = np.median(Rd - Rd[0], axis=1).astype(np.int64)
    S, Rd, E = S - off[:, None], Rd - off[:, None], E - off[:, None]
    w = 16
    xfer = (E - Rd)[:, w:] / 1e3
    bar = (Rd - S)[:, w:] / 1e3
    t0 = Rd[:, w:].min()
    t = (Rd[:, w:] - t0) / 1e6  # ms
    span = (E[:, -1].max() - t0) / 1e6
    print(f"{path}: R={R} gpus={list(gpus)} {kib} KiB ctas {ctas} iters {iters} span {span:.1f} ms")
    for r in range(R):
        x = xfer[r]
        print(f"  gpu {gpus[r]:2d} xfer p50 {np.percentile(x,50):7.2f} p99 {np.percentile(x,99):8.2f} p99.9 {np.percentile(x,99.9):8.1f} max {x.max():8.1f}"
              f" | bar p50 {np.percentile(bar[r],50):6.2f} max {bar[r].max():8.1f} | >{thr:.0f}us {int((x>thr).sum())}")
    st = xfer > thr
    any_st = st.any(axis=0)
    idx = np.nonzero(any_st)[0]
    if len(idx) == 0:
        print("  no stalls")
        return
    # Stall durations (per rank event) histogram.
    d = xfer[st]
    edges = [100, 200, 400, 600, 800, 1000, 1100, 1200, 1300, 1500, 2000, 3000, 5000, 1e9]
    h, _ = np.histogram(d, bins=edges)
    print("  stall duration histogram (us):", ", ".join(f"[{edges[i]:.0f},{edges[i+1]:.0f}) {h[i]}" for i in range(len(h)) if h[i]))
    # Which GPUs stall together (pattern of stalled ranks per iteration).
    from collections import Counter
    pat = Counter(tuple(int(gpus[r]) for r in range(R) if st[r, i]) for i in idx)
    print("  stalled-rank sets:", ", ".join(f"{k}:{v}" for k, v in pat.most_common(12)))
    # Episodes: stalled iterations closer than 1 ms belong to one episode.
    times = t.min(axis=0)[idx]
    ep_start = [times[0]]
    for a_, b_ in zip(times[:-1], times[1:]):
        if b_ - a_ > 1.0:
            ep_start.append(b_)
    ep_start = np.array(ep_start)
    print(f"  stalled iterations {len(idx)} ({100*len(idx)/xfer.shape[1]:.3f}%), episodes {len(ep_start)} over {span:.0f} ms")
    if len(ep_start) > 1:
        g = np.diff(ep_start)
        print(f"  episode spacing ms: p10 {np.percentile(g,10):.2f} p50 {np.percentile(g,50):.2f} p90 {np.percentile(g,90):.2f}")
    # Coarse timeline: stall count per 50 ms bucket.
    nb = int(span // 50) + 1
    cnt = np.bincount((times // 50).astype(int), minlength=nb)
    print("  stalls per 50 ms:", " ".join(str(c) for c in cnt))
    # Stall time share.
    lost = np.clip(xfer - np.median(xfer, axis=1, keepdims=True), 0, None)
    print("  time lost to stalls per rank (ms):", " ".join(f"g{gpus[r]} {lost[r][st[r]].sum()/1e3:.2f}" for r in range(R)),
          f"of {span:.0f} ms")

if __name__ == "__main__":
    main()
