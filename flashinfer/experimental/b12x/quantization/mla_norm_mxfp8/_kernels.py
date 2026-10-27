# SPDX-License-Identifier: Apache-2.0
"""Split RMSNorm with a BF16 query seam before floored MXFP8 encoding."""

import triton
import triton.language as tl
from b12x.gemm.blockscaled._quantize import _scale_offset


@triton.jit
def mla_norm_mxfp8_kernel(
    X,
    QW,
    KW,
    QBF,
    KV,
    Q8,
    ROWS,
    S,
    Q_FEATURES: tl.constexpr,
    KV_FEATURES: tl.constexpr,
    EPSILON: tl.constexpr,
    MIN_AMAX: tl.constexpr,
    INCOMING_PDL: tl.constexpr,
    RELEASE_DEPENDENTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)
    if RELEASE_DEPENDENTS:
        # The dependent GEMM may load its immutable RHS now; it waits before
        # consuming Q8 and S, which are written below.
        tl.extra.cuda.gdc_launch_dependents()
    if task == 0:
        c = tl.arange(0, 2048)
        w = tl.load(QW + c, c < Q_FEATURES, 0).to(tl.float32)
        if INCOMING_PDL:
            tl.extra.cuda.gdc_wait()
        x = tl.load(X + row * (Q_FEATURES + KV_FEATURES) + c, c < Q_FEATURES, 0).to(
            tl.float32
        )
        variance = tl.sum(x * x, 0) / Q_FEATURES
        y = (x * tl.rsqrt(variance + EPSILON) * w).to(QBF.dtype.element_ty)
        tl.store(QBF + row * Q_FEATURES + c, y, c < Q_FEATURES)
        grouped = tl.reshape(y.to(tl.float32), (64, 32))
        amax = tl.maximum(tl.max(tl.abs(grouped), 1), MIN_AMAX)
        bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        exponent = (
            ((bits >> 23) & 255).to(tl.int32)
            - 127
            + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        )
        scale = ((exponent + 127).to(tl.uint32) << 23).to(tl.float32, bitcast=True)
        quant = tl.minimum(
            tl.maximum(tl.div_rn(grouped, scale[:, None]), -448.0), 448.0
        ).to(tl.float8e4nv)
        tl.store(Q8 + row * Q_FEATURES + c, tl.reshape(quant, (2048,)), c < Q_FEATURES)
        group = tl.arange(0, 64)
        byte = (exponent + 127).to(tl.uint8)
        groups: tl.constexpr = Q_FEATURES // 32
        tl.store(ROWS + row * groups + group, byte, group < groups)
        tl.store(S + _scale_offset(row, group, groups), byte, group < groups)
    else:
        kc = tl.arange(0, 512)
        kw = tl.load(KW + kc).to(tl.float32)
        if INCOMING_PDL:
            tl.extra.cuda.gdc_wait()
        kx = tl.load(X + row * (Q_FEATURES + KV_FEATURES) + Q_FEATURES + kc).to(
            tl.float32
        )
        ky = kx * tl.rsqrt(tl.sum(kx * kx, 0) / KV_FEATURES + EPSILON) * kw
        tl.store(KV + row * KV_FEATURES + kc, ky)
