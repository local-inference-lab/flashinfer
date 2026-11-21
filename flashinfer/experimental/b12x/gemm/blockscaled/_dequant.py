"""MXFP8 weight expansion for the large-M A16 lowering.

``mxfp8_a16_dequant`` plans expand the packed E4M3 weight with its UE8M0 block
scales into caller-provided BF16 scratch and run a BF16 GEMM. Every product of
an E4M3 value and a power-of-two scale is exact in BF16, so the result matches
A16 over the same weight values.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ._quantize import _scale_offset

DEQUANT_BLOCK_R = 4
DEQUANT_BLOCK_K = 512


@triton.jit
def _dequantize_mxfp8(
    W,
    S,
    OUT,
    N,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Expand E4M3 weights with UE8M0 block-32 scales into BF16 (1 scale load per 32 elements)."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    r_mask = rows < N
    k0 = tl.program_id(1) * BLOCK_K
    groups = tl.arange(0, BLOCK_K // 32)
    g = k0 // 32 + groups
    g_mask = g < (K // 32)
    e_u8 = tl.load(
        S + _scale_offset(rows[:, None], g[None, :], K // 32),
        r_mask[:, None] & g_mask[None, :],
        127,
    ).to(tl.uint32)
    scale = tl.where(
        e_u8 == 0,
        tl.full(e_u8.shape, 2.0**-127, tl.float32),
        tl.where(
            e_u8 == 255,
            tl.full(e_u8.shape, float("nan"), tl.float32),
            (e_u8 << 23).to(tl.float32, bitcast=True),
        ),
    )
    k_offs = k0 + groups[None, :, None] * 32 + tl.arange(0, 32)[None, None, :]
    m3 = r_mask[:, None, None] & (k_offs < K)
    w = tl.load(W + rows[:, None, None] * K + k_offs, m3, 0.0).to(tl.float32)
    tl.store(OUT + rows[:, None, None] * K + k_offs, (w * scale[:, :, None]).to(tl.bfloat16), m3)
