"""SMContention must build for every architecture FlashInfer's CuTe DSL kernels target.

The race checks themselves run on whatever GPU is present; this keeps the
contention kernels from quietly depending on one architecture. Each target is
compiled (not launched) in a fresh interpreter with ``CUTE_DSL_ARCH`` set.
SM75 is absent because CuTe DSL has no Turing target.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

ARCHS = (
    "sm_80",
    "sm_86",
    "sm_89",
    "sm_90a",
    "sm_100a",
    "sm_103a",
    "sm_110a",
    "sm_120a",
    "sm_121a",
)

_COMPILE = """
import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Int32, Int64
from tests.experimental.b12x.race.noise import _Hog, _Skew

buf = from_dlpack(torch.zeros(1 << 16, dtype=torch.int64, device="cuda"), assumed_align=16)
sink = from_dlpack(torch.zeros(1, dtype=torch.float32, device="cuda"), assumed_align=4)
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
for threads, words in ((1024, 6000), (256, 256)):
    cute.compile(_Hog(threads, words), buf, Int32(1), Int64(1), Int64(1), Int32(0), stream)
for threads in (32, 96):
    cute.compile(_Skew(threads), sink, Int32(1), Int64(1), stream)
print("compiled")
"""


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize("arch", ARCHS)
def test_contention_kernels_compile_for(arch):
    _require_cuda()
    root = Path(__file__).resolve().parents[4]
    out = subprocess.run(
        [sys.executable, "-c", _COMPILE],
        cwd=root,
        env=dict(os.environ, CUTE_DSL_ARCH=arch),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert out.returncode == 0 and "compiled" in out.stdout, out.stderr[-2000:]


def test_contention_runs_on_this_gpu():
    _require_cuda()
    from .noise import SMContention

    noise = SMContention(torch.device("cuda", torch.cuda.current_device()))
    with noise:
        time.sleep(1.0)
    assert noise.launched > 0
