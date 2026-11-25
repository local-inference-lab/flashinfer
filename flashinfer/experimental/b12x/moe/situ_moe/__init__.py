# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 CuTe-DSL fused route-weighted SiTU MoE for Inkling 2 Mini.

Implements the exact Inkling 2 Mini routed-expert numerical contract where route
weights multiply the SiTU-gated intermediate BEFORE intermediate BF16 rounding
and per-route quantization:
  - NVFP4 (L0-19): four-or-six MAE block-16 quantization, interleaved gate/up
    fused SiTU epilogue, and optional zero-scratch inline CSF scale decoding.
  - MXFP8 (L20-23): per-32 UE8M0 E4M3 quantization with native SM120
    ``kind::mxf8f6f4`` block-scaled tensor-core MMA and warp-specialized decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="situ_moe",
    group="moe",
    api_style="planned",
    entry_points=(
        "Binding",
        "Caps",
        "Plan",
        "SituMoeConfig",
        "SituMoeQuery",
        "bind",
        "clear_caches",
        "is_supported",
        "partition_csf_slab_offsets",
        "plan",
        "run",
        "supports",
    ),
    dtypes=("nvfp4", "mxfp8"),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/flashinfer",
        commit="local",
        paths=("flashinfer/experimental/b12x/moe/situ_moe/",),
    ),
    test_path="tests/experimental/b12x/moe/test_situ_moe.py",
    since="1.4.0",
    notes=(
        "SM120/SM121 CuTe-DSL fused route-weighted SiTU MoE with inline NVFP4 CSF "
        "scale decoding and native MXFP8 block-scaled MMA."
    ),
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        Binding,
        Caps,
        Plan,
        SituMoeConfig,
        SituMoeQuery,
        bind,
        clear_caches,
        is_supported,
        partition_csf_slab_offsets,
        plan,
        run,
        supports,
    )

install_lazy_api(globals(), META)
