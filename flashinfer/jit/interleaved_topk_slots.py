# SPDX-License-Identifier: Apache-2.0
"""Compile stable owner-local top-2048 slot selection for a sixteen-rank CUDA IPC channel."""

from functools import cache
import hashlib

from . import env as jit_env
from .core import gen_jit_spec


@cache
def source_id():
    """Identify the entry point and both device headers used by the collective."""
    source = jit_env.FLASHINFER_CSRC_DIR / "interleaved_topk_slots.cu"
    headers = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "comm"
    digest = hashlib.sha256()
    for path in (
        source,
        headers / "interleaved_topk_slots.cuh",
        headers / "interleaved_topk_cutoff.cuh",
    ):
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@cache
def gen_interleaved_topk_slots_module():
    return gen_jit_spec(
        "interleaved_topk_slots_" + source_id(),
        [jit_env.FLASHINFER_CSRC_DIR / "interleaved_topk_slots.cu"],
        use_fast_math=False,
    )
