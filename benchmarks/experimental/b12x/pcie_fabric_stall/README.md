# PCIe fabric stall diagnostics (PoC tooling, not production code)

Standalone tools used to locate the TP4 collective slowdown on GPUs behind different CPU IO-die
quadrants (frank2: AMD EPYC 9575F, RTX PRO 6000, Switchtec Gen5 boards, IOMMU off). They are
independent of b12x and need only CUDA (and NCCL for `ncclstall`).

## Build

```sh
nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo -o p2pstall p2pstall.cu
N=<site-packages>/nvidia/nccl
nvcc -O3 -std=c++17 -arch=sm_120 -I$N/include -L$N/lib -l:libnccl.so.2 -Xlinker -rpath=$N/lib -o ncclstall ncclstall.cu
gcc -O2 -o hsmpq hsmpq.c        # read-only AMD HSMP queries (also hsmpget.c, hsmplat.c, fclksample.c)
```

## Tools

- `p2pstall`: one process, one persistent kernel per GPU, per-operation `%globaltimer` stamps (start,
  after a cross-GPU start barrier, end) with clock alignment. Patterns: `read_ring`, `read_all`,
  `read_incast`, `read_pair`, `write_ring`, `write_all`, `write_phased`, `write_incast`, `write_pair`,
  `pull`, `push`, `push_phased`, `ring`, `rsag_push`, `rsag_phased`, `rsag_pull`, `host_read`,
  `host_write`. `--dump PREFIX` writes raw stamps for `analyze.py` (stall histogram and episodes) and
  `timeline.py` (stalled GPUs per 50 ms bucket).
  ```sh
  CUDA_DEVICE_ORDER=PCI_BUS_ID ./p2pstall --gpus 1,5,9,13 --iters 400000 --ctas 16 --kib 80 \
    --patterns read_ring,pull,push_phased,rsag_phased --dump dumps/x
  python3 timeline.py dumps/x-read_ring-80k-c16.bin
  ```
- `ncclstall`: NCCL all-reduce in CUDA graphs with per-operation stamps. Single-thread multi-GPU needs
  `NCCL_RUNTIME_CONNECT=0`.
- Correlation probes: `lnksample.py` (PCIe link status every 0.5 ms), `cpusample.py`, `hold_latency.py`
  (`/dev/cpu_dma_latency`), `trig_nvml.py` / `trig_mrpc.py` (timed NVML and Switchtec calls),
  `hsmpq.c`, `hsmpget.c`, `hsmplat.c`, `fclksample.c` (read-only HSMP).
- `hsmpset_lclk.c`, `hsmpset_pm.c`, `run_lclk_test.sh`, `run_pm_test.sh`: runtime HSMP changes (LCLK DPM
  range, power efficiency mode, DF C-states) with read-back and restore. They change CPU power
  management; run only with the machine owner's approval.
- `results/`: transport comparison (`rank80-400.csv`) and the size sweep on 1,5,9,13.
