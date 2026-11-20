"""Opt-in Trellis prefill launch shapes for the full-rotation W4A16 fused MoE.

Kept outside moe/_shared/kernels so that no CuTe kernel module's source closure (compile-cache fingerprint)
includes it: with the switch off, every W4A16 kernel compile key is unchanged by this module.
"""

from __future__ import annotations

import os


# Trellis prefill launch shapes (ported from tunamitom/flashinfer@2676f2beae6a +
# @14369ed04566). Default OFF. Selected only when
# B12X_W4A16_PREFILL_SHAPES=1 (read at plan/compile time) and only for launch
# token counts M >= 128, so every decode / verify / draft launch (exact-M
# plans for capacities <= 32) keeps the shipped heuristic by construction.
#   M >= 2560:       3 pipeline stages, 64-row route blocks, compact table
#   2048 <= M < 2560: untouched (all variants lost in their sweep)
#   1024 <= M < 2048: 32-row blocks, 2 stages, direct ("full") decode table
#   128 <= M < 1024:  16-row blocks, 3 stages, direct ("full") decode table
# Port note: the fork's wiring of the raced MoeDecodeConfig.w4a16_* knobs into
# the launch was NOT ported (it would activate autotuned-but-inert decode
# knobs with the switch off). B12X_W4A16_PREFILL_SHAPES_TILE=native also drops
# a forced B12X_TRELLIS_TILE_CONFIG for these (M >= 128) launches.
def trellis_prefill_launch_shape(m: int) -> dict | None:
    """Measured prefill launch shape for the Trellis fused kernel, or None."""
    if os.getenv("B12X_W4A16_PREFILL_SHAPES", "0") != "1":
        return None
    m = int(m)
    if m < 128:
        return None
    if m >= 2560:
        return {"block_size_m": 64, "pipeline_stages": 3, "trellis_decode_table": "compact"}
    if m >= 2048:
        return None
    if m >= 1024:
        return {"block_size_m": 32, "pipeline_stages": 2, "trellis_decode_table": "full"}
    return {"block_size_m": 16, "pipeline_stages": 3, "trellis_decode_table": "full"}


def trellis_prefill_native_tile(m: int) -> bool:
    """True when a prefill-shape launch should ignore the forced env tile."""
    return (
        trellis_prefill_launch_shape(m) is not None
        and os.getenv("B12X_W4A16_PREFILL_SHAPES_TILE", "") == "native"
    )

