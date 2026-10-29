# Summarize dumps-phased logs: per GPU set, size and pattern, the best CTA count by per-rank mean.
import glob, re, sys

rows = {}
for path in sorted(glob.glob(sys.argv[1] + "/g*-?.log")):
    gset = path.split("/")[-1][1:-6]
    cur = None
    for line in open(path):
        m = re.match(r"== (\S+)\s+(\d+) KiB ctas\s+(\d+)", line)
        if m:
            cur = (gset, int(m.group(2)), m.group(1), int(m.group(3)))
            rows[cur] = []
            continue
        if line.startswith("   gpu") and cur:
            w = line.split()
            rows[cur].append((float(w[12]), float(w[10]), int(w[-1]), float(w[4])))
best = {}
for (gset, kib, pat, ctas), v in rows.items():
    mean = sum(x[0] for x in v) / len(v)
    rec = (mean, max(x[1] for x in v), sum(x[2] for x in v), sum(x[3] for x in v) / len(v), ctas)
    k = (gset, kib, pat)
    if k not in best or rec[0] < best[k][0]:
        best[k] = rec
for gset in sorted({k[0] for k in best}):
    for kib in sorted({k[1] for k in best if k[0] == gset}):
        print(f"## gpus {gset} {kib} KiB")
        for (g, k, pat), r in sorted(best.items(), key=lambda kv: kv[1][0]):
            if g == gset and k == kib:
                print(f"  {pat:12s} ctas {r[4]:2d} mean {r[0]:7.2f} p50 {r[3]:7.2f} max {r[1]:8.1f} >1ms {r[2]}")
