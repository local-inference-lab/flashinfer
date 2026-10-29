"""Sample PCIe LnkSta/DevSta of given devices (config space reads only) and log changes until a stop file appears."""
import os, sys, time
out, stop = sys.argv[1], sys.argv[2]
devs = []
for spec in sys.argv[3:]:
    bdf, cap = spec.split("@")
    fd = os.open(f"/sys/bus/pci/devices/0000:{bdf}/config", os.O_RDONLY)
    devs.append((bdf, fd, int(cap, 16)))
def read(fd, cap):
    b = os.pread(fd, 0x14, cap)
    devsta = int.from_bytes(b[0x0a:0x0c], "little")
    lnksta = int.from_bytes(b[0x12:0x14], "little")
    return devsta, lnksta
last = {d[0]: None for d in devs}
n = 0
t_start = time.time()
with open(out, "w") as fh:
    while not os.path.exists(stop):
        now = time.time()
        for bdf, fd, cap in devs:
            v = read(fd, cap)
            if v != last[bdf]:
                ds, ls = v
                fh.write(f"{now:.6f} {bdf} devsta={ds:#06x} lnksta={ls:#06x} speed={ls & 0xf} width={(ls >> 4) & 0x3f} "
                         f"train={(ls >> 11) & 1} dllact={(ls >> 13) & 1} bwmgmt={(ls >> 14) & 1} abw={(ls >> 15) & 1}\n")
                last[bdf] = v
        n += 1
        time.sleep(0.0005)
    fh.write(f"# samples {n} over {time.time() - t_start:.1f} s\n")
