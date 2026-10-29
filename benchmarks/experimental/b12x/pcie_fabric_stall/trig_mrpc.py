"""Issue the same read-only Switchtec MRPC commands as gpu-switch-exporter (DIETEMP, LNKSTAT, PMON) on every chip."""
import glob, os, struct, sys, time
MAX_PORTS, BW_PER_CMD = 60, 18
def mrpc(fd, cmd, payload, outlen):
    os.write(fd, struct.pack("<I", cmd) + payload)
    buf = os.read(fd, 4 + outlen)
    ret = struct.unpack_from("<I", buf)[0]
    if ret:
        raise OSError("MRPC 0x%x failed: 0x%x" % (cmd, ret))
    return buf[4:]
def read(dev, which):
    fd = os.open(dev, os.O_RDWR)
    try:
        if "temp" in which:
            mrpc(fd, 0x04, struct.pack("<I", 4), 16)
        ports = []
        if "lnk" in which or "pmon" in which:
            raw = mrpc(fd, 0x1C, struct.pack("<Q", 0), MAX_PORTS * 12)
            for i in range(MAX_PORTS):
                phys, par, log, stk, cfgw, negw, usp, lr, _l, _r, _f = struct.unpack_from("<8BH2B", raw, i * 12)
                if (stk >> 4) > 8:
                    continue
                if lr >> 7:
                    ports.append(phys)
        if "pmon" in which:
            for i in range(0, len(ports), BW_PER_CMD):
                chunk = ports[i:i + BW_PER_CMD]
                req = struct.pack("<BB", 1, len(chunk)) + b"".join(struct.pack("<BB", p, 0) for p in chunk)
                mrpc(fd, 0x07, req, 56 * len(chunk))
    finally:
        os.close(fd)
which = sys.argv[1] if len(sys.argv) > 1 else "temp,lnk,pmon"
for dev in sorted(glob.glob("/dev/switchtec*")):
    t0 = time.time()
    read(dev, which)
    print(f"{t0:.6f} {time.time():.6f} {dev} {which}", flush=True)
