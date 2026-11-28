# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 paged block-16 NVFP4 (4-or-6 MAE K, max/6 V) KV writer and attention.

Covers:
  - ``PagedNVFP4KVWriteKernel``: single-pass vectorized BF16 -> NVFP4 paged
    cache append (4-or-6 MAE K scales, max/6 V scales, capture-safe, negative
    slots skipped).
  - ``PagedNVFP4DecodeKernel`` (+ shared ``PagedDecodeMergeKernel``): split-KV
    warp-specialized decode/verify attention with row/head sinks, sliding
    windows, and dense relative-distance bias.
  - ``PagedNVFP4PrefillKernel``: FP8-MMA 2x-Q-reuse tiled chunked prefill
    attention with row/head sinks, sliding windows, and factored relative bias.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="paged_nvfp4",
    group="attention",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "PagedNVFP4Config",
        "PagedNVFP4Query",
        "Plan",
        "bind",
        "clear_caches",
        "is_supported",
        "plan",
        "run",
        "supports",
        "write_kv",
    ),
    dtypes=("nvfp4_4o6k",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/flashinfer",
        commit="local",
        paths=("flashinfer/experimental/b12x/attention/paged_nvfp4/",),
    ),
    test_path="tests/experimental/b12x/attention/test_paged_nvfp4.py",
    since="1.4.0",
    notes=(
        "SM120 CuTe-DSL paged block-16 NVFP4 (4-or-6 MAE K) writer, split-KV "
        "decode/verify, and FP8-MMA tiled prefill attention with row/head sinks "
        "and dense/factored relative bias."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        PagedNVFP4Config,
        PagedNVFP4Query,
        Plan,
        bind,
        clear_caches,
        is_supported,
        plan,
        run,
        supports,
        write_kv,
    )

install_lazy_api(globals(), META)
