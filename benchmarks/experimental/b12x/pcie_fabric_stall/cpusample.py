"""Log wall-clock times at which given PIDs consume CPU (10 ms sampling) until the stop file appears."""
import os, sys, time
out, stop = sys.argv[1], sys.argv[2]
pids = [int(p) for p in sys.argv[3:]]
def cpu(pid):
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])
    except OSError:
        return -1
last = {p: cpu(p) for p in pids}
with open(out, "w") as fh:
    while not os.path.exists(stop):
        time.sleep(0.01)
        now = time.time()
        for p in pids:
            c = cpu(p)
            if c != last[p]:
                fh.write(f"{now:.3f} {p} {c - last[p]}\n")
                last[p] = c
