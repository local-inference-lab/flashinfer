"""Hold a PM QoS CPU latency request of 0 us (no deep C-states) while a command runs."""
import os, struct, subprocess, sys
fd = os.open("/dev/cpu_dma_latency", os.O_WRONLY)
os.write(fd, struct.pack("i", 0))
try:
    sys.exit(subprocess.call(sys.argv[1:]))
finally:
    os.close(fd)
