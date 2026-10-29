"""Table of global per-operation cost (mean, p50, p99, max, ops >1 ms) from the final comparison."""
import csv, glob, re, sys
d = sys.argv[1]
rows = {}
for f in glob.glob(f"{d}/k*.csv"):
    lines = [l for l in open(f) if l.startswith(("pattern,", "pull", "push", "ring", "rsag"))]
    for r in csv.DictReader(lines):
        key = (int(r["kib"]), r["pattern"], int(r["ctas"]))
        rows.setdefault(key, []).append(r)
out = []
for (kib, pat, ctas), rs in sorted(rows.items()):
    n1 = max(int(r["n_1ms"]) for r in rs)
    out.append((kib, f"{pat}/{ctas}", float(rs[0]["global_mean"]), float(rs[0]["global_p50"]), float(rs[0]["global_p99"]),
                float(rs[0]["global_max"]), n1))
for f in sorted(glob.glob(f"{d}/nccl-*.log")):
    m = re.match(r".*/nccl-(\w+)-(\d+)\.log", f)
    kind, kib = m.group(1), int(m.group(2))
    pat = re.compile(r"op p50 +([\d.]+) p99 +([\d.]+) p99\.9 +([\d.]+) max +([\d.]+) mean +([\d.]+) us \| >100us (\d+) >1ms (\d+)")
    vals = [pat.search(l).groups() for l in open(f) if pat.search(l)]
    if not vals:
        out.append((kib, f"nccl_{kind}", float("nan"), 0, 0, 0, 0)); continue
    p50 = max(float(v[0]) for v in vals); p99 = max(float(v[1]) for v in vals); mx = max(float(v[3]) for v in vals)
    mean = max(float(v[4]) for v in vals); n1 = max(int(v[6]) for v in vals)
    out.append((kib, f"nccl_{kind}", mean, p50, p99, mx, n1))
print(f"{'KiB':>4} {'transport':<18} {'mean':>8} {'p50':>8} {'p99':>8} {'max':>8} {'>1ms':>5}")
for kib, name, mean, p50, p99, mx, n1 in sorted(out, key=lambda x: (x[0], x[2])):
    print(f"{kib:4d} {name:<18} {mean:8.2f} {p50:8.2f} {p99:8.2f} {mx:8.1f} {n1:5d}")
