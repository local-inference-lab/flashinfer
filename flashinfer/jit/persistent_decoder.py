# SPDX-License-Identifier: Apache-2.0
"""NVCC module for the fixed-geometry tensor-parallel persistent decoder."""

from . import env as jit_env
from .core import gen_jit_spec


def gen_persistent_decoder_module():
    return gen_jit_spec(
        "persistent_decoder_sm120",
        [jit_env.FLASHINFER_CSRC_DIR / "persistent_decoder.cu"],
        extra_cuda_cflags=["-std=c++20", "-lineinfo"],
        use_fast_math=False,
    )
