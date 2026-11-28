# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared SM120/SM121 PTX primitives for fused route-weighted SiTU MoE."""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.math import div_rn_f32
from b12x._lib.intrinsics.memory import (
    create_l2_evict_first_policy,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    st_shared_u8,
    st_shared_v4_u32,
)

FP32_MAX = 3.4028234663852886e38


@dsl_user_op
def rcp_rn_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.rn.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def situ_activate_f32(
    gate_f32: Float32, up_f32: Float32, weight_f32: Float32, *, loc=None, ip=None
) -> Float32:
    """Exact Inkling-2-Mini route-weighted SiTU matching ``situ_experts._activate``.

    Inputs ``gate_f32`` and ``up_f32`` are rounded to BF16 first (matching the
    BF16 FC1 output boundary), then:
      ``sigmoid = 1 / (1 + exp2(-log2(e) * gate))``
      ``g = 4 * tanh_approx(gate / 4)``
      ``u = 25 * tanh_approx(up / 25)``
      ``out = ((g * sigmoid) * u) * weight``
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(gate_f32).ir_value(loc=loc, ip=ip),
                Float32(up_f32).ir_value(loc=loc, ip=ip),
                Float32(weight_f32).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 gb, ub;
                .reg .f32 g, u, t, sig, gt, ut, res;
                cvt.rn.bf16.f32 gb, $1;
                cvt.rn.bf16.f32 ub, $2;
                cvt.f32.bf16 g, gb;
                cvt.f32.bf16 u, ub;
                mul.rn.f32 t, g, 0fBFB8AA3B;
                ex2.approx.ftz.f32 t, t;
                add.rn.f32 t, t, 0f3F800000;
                rcp.approx.ftz.f32 sig, t;
                div.rn.f32 gt, g, 0f40800000;
                tanh.approx.f32 gt, gt;
                mul.rn.f32 gt, gt, 0f40800000;
                div.rn.f32 ut, u, 0f41C80000;
                tanh.approx.f32 ut, ut;
                mul.rn.f32 ut, ut, 0f41C80000;
                mul.rn.f32 res, gt, sig;
                mul.rn.f32 res, res, ut;
                mul.rn.f32 $0, res, $3;
            }
            """,
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_byte_to_f32(byte_u32: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert low 8 bits (unsigned E4M3FN byte) to FP32 via SM120 hardware cvt."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(byte_u32).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 p16, h0, h1;
                .reg .b32 h2;
                cvt.u16.u32 p16, $1;
                cvt.rn.f16x2.e4m3x2 h2, p16;
                mov.b32 {h0, h1}, h2;
                cvt.f32.f16 $0, h0;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_to_e4m3_byte(val: Float32, *, loc=None, ip=None) -> Uint32:
    """Round FP32 scalar to E4M3FN byte in low 8 bits of Uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out16;
                cvt.rn.satfinite.e4m3x2.f32 out16, $1, $1;
                cvt.u32.u16 $0, out16;
                and.b32 $0, $0, 255;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e2m1x2_pack_and_roundtrip(
    lo: Float32, hi: Float32, *, loc=None, ip=None
) -> Tuple[Uint32, Float32, Float32]:
    """Pack ``(lo, hi)`` to E2M1x2 byte and return ``(byte_u32, dec_lo_f32, dec_hi_f32)``."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, f32, f32)>"),
        [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 b;
            .reg .b32 h2;
            .reg .b16 h0, h1;
            cvt.rn.satfinite.e2m1x2.f32 b, $4, $3;
            cvt.u32.u8 $0, b;
            cvt.rn.f16x2.e2m1x2 h2, b;
            mov.b32 {h0, h1}, h2;
            cvt.f32.f16 $1, h0;
            cvt.f32.f16 $2, h1;
        }
        """,
        "=r,=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def ue8m0_byte_to_f32(exp_u32: Uint32, *, loc=None, ip=None) -> Float32:
    """Decode UE8M0 exponent byte (0 -> 2^-127, e -> 2^(e-127)) to FP32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(exp_u32).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p0;
                .reg .b32 bits;
                setp.eq.u32 p0, $1, 0;
                shl.b32 bits, $1, 23;
                selp.b32 bits, 0x00400000, bits, p0;
                mov.b32 $0, bits;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_u8(addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .u8 b;
                ld.shared.u8 b, [$1];
                cvt.u32.u8 $0, b;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .u8 b;
                ld.global.nc.u8 b, [$1];
                cvt.u32.u8 $0, b;
            }
            """,
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_nc_l2evict_v4_u32(
    addr: Int64, policy: cutlass.Uint64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """16B streaming global load with L2::evict_first cache policy (preserves L2 activations)."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(policy).ir_value(loc=loc, ip=ip),
        ],
        "ld.global.nc.L2::cache_hint.v4.u32 {$0, $1, $2, $3}, [$4], $5;",
        "=r,=r,=r,=r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def shfl_xor_f32(val: Float32, mask: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {int(mask)}, 0x1f, 0xffffffff;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_idx_f32(val: Float32, src_lane: int, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.idx.b32 $0, $1, {int(src_lane)}, 0x1f, 0xffffffff;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Native SM120 block-scaled MMA intrinsics are imported from b12x._lib.intrinsics.mma.


@cute.jit
def expand_csf_slab_to_smem(
    mFixed: cute.Tensor,  # flat u8 view of [E, SLABS, 16 * (1 + C//2)]
    mExc: cute.Tensor,  # flat u32 [num_exc] (partitioned by slab)
    mSlabOffs: cute.Tensor,  # [E * SLABS + 1] i32
    smem_scales_addr: Int32,  # SMEM destination [16, C] u8
    expert: Int32,
    slab: Int32,
    tid: Int32,
    threads: cutlass.Constexpr,
    slabs: cutlass.Constexpr,
    cols: cutlass.Constexpr,
):
    """Cooperatively decode one 16-row NVFP4 CSF slab directly into shared memory.

    Layout of one slab in ``mFixed``:
      - bytes ``0..15``: 16 per-row base values (``u8``)
      - bytes ``16..16 + 16*(cols//2) - 1``: row-major packed 4-bit offsets
        (lo nibble = even col, hi nibble = odd col)
    Exceptions in ``mExc[start:end]`` store ``(val << 24) | flat_in_expert``,
    where ``flat_in_expert - slab * 16 * cols`` is the exact byte offset in
    ``smem_scales[16, cols]``.
    """
    half_cols: cutlass.Constexpr = cols // 2
    slab_bytes: cutlass.Constexpr = 16 * (1 + half_cols)
    total_pairs: cutlass.Constexpr = 16 * half_cols
    slab_idx = expert * Int32(slabs) + slab
    slab_base_ptr = get_ptr_as_int64(mFixed, Int64(slab_idx) * Int64(slab_bytes))

    # Each thread unpacks ceil(total_pairs / threads) nibble pairs -> 2 u8 scales.
    iters: cutlass.Constexpr = (total_pairs + threads - 1) // threads
    for it in cutlass.range_constexpr(iters):
        idx = tid + Int32(it * threads)
        if idx < Int32(total_pairs):
            local_row = idx // Int32(half_cols)
            pair_col = idx - local_row * Int32(half_cols)
            base_u8 = ld_global_nc_u8(slab_base_ptr + Int64(local_row))
            packed = ld_global_nc_u8(slab_base_ptr + Int64(16 + idx))
            lo = (base_u8 + (packed & Uint32(0xF))) & Uint32(0xFF)
            hi = (base_u8 + ((packed >> Uint32(4)) & Uint32(0xF))) & Uint32(0xFF)
            dst_off = local_row * Int32(cols) + pair_col * Int32(2)
            st_shared_u8(smem_scales_addr + dst_off, lo.to(Uint8))
            st_shared_u8(smem_scales_addr + dst_off + Int32(1), hi.to(Uint8))

    cute.arch.barrier()

    # Apply pre-partitioned sparse exceptions for this slab (mean 1.8-5.8/slab).
    exc_start = mSlabOffs[slab_idx]
    exc_end = mSlabOffs[slab_idx + Int32(1)]
    slab_flat_base = slab * Int32(16 * cols)
    pos = exc_start + tid
    while pos < exc_end:
        rec = ld_global_nc_u32(get_ptr_as_int64(mExc, Int64(pos)))
        flat = (rec & Uint32(0xFFFFFF)).to(Int32) - slab_flat_base
        val = ((rec >> Uint32(24)) & Uint32(0xFF)).to(Uint8)
        st_shared_u8(smem_scales_addr + flat, val)
        pos = pos + Int32(threads)

    cute.arch.barrier()


@cute.jit
def load_dense_slab_to_smem(
    mScales: cute.Tensor,  # flat u8 view of [E, N, C]
    smem_scales_addr: Int32,  # SMEM destination [16, C] u8
    expert: Int32,
    slab: Int32,
    tid: Int32,
    threads: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    cols: cutlass.Constexpr,
):
    """Cooperatively load 16 rows x ``cols`` bytes of dense scales into SMEM."""
    total_words: cutlass.Constexpr = (16 * cols) // 4
    base_off = (Int64(expert) * Int64(rows) + Int64(slab) * Int64(16)) * Int64(cols)
    base_ptr = get_ptr_as_int64(mScales, base_off)
    iters: cutlass.Constexpr = (total_words + threads - 1) // threads
    for it in cutlass.range_constexpr(iters):
        w = tid + Int32(it * threads)
        if w < Int32(total_words):
            word = ld_global_nc_u32(base_ptr + Int64(w) * Int64(4))
            st_shared_u32(smem_scales_addr + w * Int32(4), word)
    cute.arch.barrier()


@dsl_user_op
def e4m3x4_dot_f32(a_u32: Uint32, b_u32: Uint32, *, loc=None, ip=None) -> Float32:
    """Dot product of 4 packed E4M3FN bytes with 4 packed E4M3FN bytes -> FP32.

    Converts each E4M3 lane to FP32 before multiplying so full-range E4M3 values
    (up to ``+/-448``, product ``200,704 > FP16_MAX``) never overflow FP16.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(a_u32).ir_value(loc=loc, ip=ip),
                Uint32(b_u32).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 a_lo, a_hi, b_lo, b_hi;
                .reg .b32 ah0, ah1, bh0, bh1;
                .reg .b16 a0, a1, a2, a3, b0, b1, b2, b3;
                .reg .f32 fa0, fa1, fa2, fa3, fb0, fb1, fb2, fb3, acc;
                mov.b32 {a_lo, a_hi}, $1;
                mov.b32 {b_lo, b_hi}, $2;
                cvt.rn.f16x2.e4m3x2 ah0, a_lo;
                cvt.rn.f16x2.e4m3x2 ah1, a_hi;
                cvt.rn.f16x2.e4m3x2 bh0, b_lo;
                cvt.rn.f16x2.e4m3x2 bh1, b_hi;
                mov.b32 {a0, a1}, ah0;
                mov.b32 {a2, a3}, ah1;
                mov.b32 {b0, b1}, bh0;
                mov.b32 {b2, b3}, bh1;
                cvt.f32.f16 fa0, a0;
                cvt.f32.f16 fa1, a1;
                cvt.f32.f16 fa2, a2;
                cvt.f32.f16 fa3, a3;
                cvt.f32.f16 fb0, b0;
                cvt.f32.f16 fb1, b1;
                cvt.f32.f16 fb2, b2;
                cvt.f32.f16 fb3, b3;
                mul.rn.f32 acc, fa0, fb0;
                fma.rn.f32 acc, fa1, fb1, acc;
                fma.rn.f32 acc, fa2, fb2, acc;
                fma.rn.f32 $0, fa3, fb3, acc;
            }
            """,
            "=f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
