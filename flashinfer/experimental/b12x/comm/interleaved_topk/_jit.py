# SPDX-License-Identifier: Apache-2.0
"""Compile stable owner-local top-2048 slot selection for a sixteen-rank CUDA IPC channel."""

from functools import cache
import hashlib
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2] / "csrc" / "interleaved_topk"
_HEADERS = _ROOT / "include" / "flashinfer" / "comm"
_SOURCES = (
    _ROOT / "interleaved_topk_slots.cu",
    _HEADERS / "interleaved_topk_slots.cuh",
    _HEADERS / "interleaved_topk_cutoff.cuh",
)


@cache
def sources_available():
    """Small release wheels may omit the optional experimental CUDA sources."""
    return all(path.is_file() for path in _SOURCES)


@cache
def source_id():
    """Identify the entry point and both device headers used by the collective."""
    if not sources_available():
        raise RuntimeError(
            "interleaved top-k requires a FlashInfer wheel containing experimental kernel sources"
        )
    digest = hashlib.sha256()
    for path in _SOURCES:
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@cache
def gen_interleaved_topk_slots_module():
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.core import gen_jit_spec

    return gen_jit_spec(
        "interleaved_topk_slots_" + source_id(),
        [_SOURCES[0]],
        extra_include_paths=[_ROOT / "include", jit_env.FLASHINFER_CSRC_DIR],
        use_fast_math=False,
    )
