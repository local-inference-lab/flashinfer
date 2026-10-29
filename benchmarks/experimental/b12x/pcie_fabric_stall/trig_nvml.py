"""NVML queries like gpu-switch-exporter, every GPU once."""
import time
import pynvml as nv
nv.nvmlInit()
n = nv.nvmlDeviceGetCount()
t0 = time.time()
for i in range(n):
    h = nv.nvmlDeviceGetHandleByIndex(i)
    for fn, a in ((nv.nvmlDeviceGetTemperature, (0,)), (nv.nvmlDeviceGetPowerUsage, ()), (nv.nvmlDeviceGetUtilizationRates, ()),
                  (nv.nvmlDeviceGetMemoryInfo, ()), (nv.nvmlDeviceGetClockInfo, (0,)), (nv.nvmlDeviceGetClockInfo, (2,)),
                  (nv.nvmlDeviceGetPerformanceState, ()), (nv.nvmlDeviceGetCurrPcieLinkGeneration, ()),
                  (nv.nvmlDeviceGetCurrPcieLinkWidth, ()), (nv.nvmlDeviceGetCurrentClocksEventReasons, ()),
                  (nv.nvmlDeviceGetTotalEnergyConsumption, ()), (nv.nvmlDeviceGetPcieReplayCounter, ()),
                  (nv.nvmlDeviceGetTotalEccErrors, (0, 0)), (nv.nvmlDeviceGetFanSpeed_v2, (0,))):
        try:
            fn(h, *a)
        except nv.NVMLError:
            pass
print(f"{t0:.6f} {time.time():.6f} nvml {n} gpus", flush=True)
