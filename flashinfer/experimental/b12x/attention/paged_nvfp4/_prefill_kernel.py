# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 warp-specialized producer-dequant ldmatrix FP8-MMA prefill over paged NVFP4 (D=128).

Four dedicated producer/dequant warps (2 for K, 2 for V) load NVFP4 + E4M3 scales
directly from global memory via vectorized ``ld.global.nc.v4/u32`` (2 lanes/row for
128 B coalescing and 2x dequant throughput), dequantize once per stage into
bank-conflict-free XOR-swizzled E4M3 shared memory (``8 KiB``/stage), and signal
independent ``full_k`` / ``full_v`` mbarriers so V transfers overlap QK MMA and
online softmax.

Consumer warps (12 warps for ``group_size % 6 == 0`` covering 32 Q tokens x all 6 Q
heads in one CTA; else 8 warps) hold 16 Q rows per warp (``NUM_MMA_Q=1``, zero
register spills) and execute full-rate SM120 ``mma.sync.aligned.m16n8k32.f16.e4m3``
for both QK and per-tile PV (accumulating cross-tile ``O`` and online softmax in
FP32), fed directly by ``ldmatrix.m8n8.x4.b16`` (K) and ``ldmatrix.m16n16.x1.trans.b8`` (V).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics.conversion import cvt_f32x4_to_e4m3x4
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v2_u32,
    ld_global_nc_v4_u32,
    ldmatrix_m8n8x4_b16,
    ldmatrix_m16n16x1_trans_b8,
    shared_ptr_to_u32,
    st_shared_v4_u32,
    u32_as_f32,
)
from b12x.attention.paged.forward_paged import (
    _exit_thread,
    _exp2_approx_ftz_f32,
    _max_nan_f32,
)
from b12x.attention.paged_decode._kernel import LOG2_E

from ._math import _extract_e4m3_broadcast

WARP_Q_ROWS = 16
CTA_ROWS = 4 * WARP_Q_ROWS  # legacy alias preserved for imports
REL_RANK = 16
Q_FP8_MAX = 16.0
P_FP8_SCALE = 32.0
_STAGE_ROWS = 32
_STAGE_HALF_BYTES = _STAGE_ROWS * 128  # 4096 B per K or V E4M3 plane
_STAGE_BYTES = 2 * _STAGE_HALF_BYTES  # 8192 B per stage


def _make_prefill_e4m3_shared_storage_cls(num_stages: int):
    payload_bytes = num_stages * _STAGE_BYTES

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "full_k": cute.struct.MemRange[cutlass.Int64, num_stages],
        "full_v": cute.struct.MemRange[cutlass.Int64, num_stages],
        "empty_k": cute.struct.MemRange[cutlass.Int64, num_stages],
        "empty_v": cute.struct.MemRange[cutlass.Int64, num_stages],
        "payload": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, payload_bytes], 16
        ],
    }
    return cute.struct(SharedStorage)


@dsl_user_op
def _mma_m16n8k32_f16_e4m3(
    d0: Uint32,
    d1: Uint32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Full-rate SM120 FP8 E4M3 warp MMA ``m16n8k32`` with packed FP16x2 accumulators."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(x).ir_value(loc=loc, ip=ip)
            for x in (a0, a1, a2, a3, b0, b1, d0, d1)
        ],
        """
        mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16
        {$0, $1},
        {$2, $3, $4, $5},
        {$6, $7},
        {$0, $1};
        """,
        "=r,=r,r,r,r,r,r,r,0,1",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def _f16x2_to_f32x2(h2: Uint32, *, loc=None, ip=None):
    """Unpack packed IEEE FP16x2 ``u32`` into two FP32 values."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def _nvfp4_u32_to_e4m3x8(
    packed_u32: Uint32, sf_f16x2: Uint32, *, loc=None, ip=None
):
    """Decode 4 NVFP4 bytes (8 nibbles) scaled by ``sf_f16x2`` into 8 packed E4M3 bytes."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(packed_u32).ir_value(loc=loc, ip=ip),
            Uint32(sf_f16x2).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 tmp, h0, h1, h2, h3;
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 e0, e1, e2, e3;
            cvt.u8.u32 b0, $2;
            shr.u32 tmp, $2, 8;
            cvt.u8.u32 b1, tmp;
            shr.u32 tmp, $2, 16;
            cvt.u8.u32 b2, tmp;
            shr.u32 tmp, $2, 24;
            cvt.u8.u32 b3, tmp;
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.f16x2 h0, h0, $3;
            mul.f16x2 h1, h1, $3;
            mul.f16x2 h2, h2, $3;
            mul.f16x2 h3, h3, $3;
            cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
            mov.b32 $0, {e0, e1};
            mov.b32 $1, {e2, e3};
        }
        """,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    e0 = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    e1 = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(e0), Uint32(e1)


@cute.jit
def _bf16x2_dot_f32(a: Uint32, b: Uint32) -> Float32:
    a0 = u32_as_f32(a << Uint32(16))
    a1 = u32_as_f32(a & Uint32(0xFFFF0000))
    b0 = u32_as_f32(b << Uint32(16))
    b1 = u32_as_f32(b & Uint32(0xFFFF0000))
    return a0 * b0 + a1 * b1


@cute.jit
def _run_e4m3_dequant_producer_half(
    lane: Int32,
    warp_sub: Int32,  # 0 for stage rows 0..15, 1 for stage rows 16..31
    mCache: cute.Tensor,  # [pages, P, Hkv, 64] u8
    mScales: cute.Tensor,  # [pages, P, Hkv, 8] u8
    mPageTable: cute.Tensor,  # [B, W] i32
    request_idx: Int32,
    kv_head_idx: Int32,
    chunk_start: Int32,
    first_key: Int32,
    last_key_exclusive: Int32,
    ntiles: Int32,
    num_stages: cutlass.Constexpr,
    plane_offset_bytes: cutlass.Constexpr,
    payload_base_addr: Int32,
    full_mbar,
    empty_mbar,
    page_size: cutlass.Constexpr,
):
    """Load 16 NVFP4 rows across 1 warp (2 lanes/row = 64 cols/lane), dequant to E4M3 SMEM."""
    mCacheFlat = cute.flatten(mCache)
    mScalesFlat = cute.flatten(mScales)
    d_base_ptr = get_ptr_as_int64(mCacheFlat, Int64(0))
    s_base_ptr = get_ptr_as_int64(mScalesFlat, Int64(0))
    d_s0 = Int64(mCache.stride[0])
    d_s1 = Int64(mCache.stride[1])
    d_h_off = Int64(kv_head_idx) * Int64(mCache.stride[2])
    s_s0 = Int64(mScales.stride[0])
    s_s1 = Int64(mScales.stride[1])
    s_h_off = Int64(kv_head_idx) * Int64(mScales.stride[2])

    local_row = lane >> Int32(1)  # 0..15
    half = lane & Int32(1)  # 0 -> cols 0..63, 1 -> cols 64..127
    row_in_stage = warp_sub * Int32(16) + local_row
    row_swiz = row_in_stage & Int32(7)
    d_half_off = Int64(half) * Int64(32)
    s_half_off = Int64(half) * Int64(4)
    chunk_base = half * Int32(4)

    t = Int32(0)
    while t < ntiles:
        buf = t % Int32(num_stages)
        use = t // Int32(num_stages)
        if t >= Int32(num_stages):
            cute.arch.mbarrier_wait(empty_mbar + buf, phase=(use - Int32(1)) & Int32(1))

        row_abs = chunk_start + t * Int32(_STAGE_ROWS) + row_in_stage
        stage_base = (
            payload_base_addr
            + buf * Int32(_STAGE_BYTES)
            + Int32(plane_offset_bytes)
        )
        if row_abs >= first_key and row_abs < last_key_exclusive:
            page_idx = Int64(mPageTable[request_idx, row_abs // Int32(page_size)])
            slot = Int64(row_abs % Int32(page_size))
            d_off = page_idx * d_s0 + slot * d_s1 + d_h_off + d_half_off
            s_off = page_idx * s_s0 + slot * s_s1 + s_h_off + s_half_off
            sf = ld_global_nc_u32(s_base_ptr + s_off)
            w0, w1, w2, w3 = ld_global_nc_v4_u32(d_base_ptr + d_off)
            w4, w5, w6, w7 = ld_global_nc_v4_u32(d_base_ptr + d_off + Int64(16))
            words = (w0, w1, w2, w3, w4, w5, w6, w7)
            for c in cutlass.range_constexpr(4):
                sf_h2 = _extract_e4m3_broadcast(sf, c, "f16x2")
                e0, e1 = _nvfp4_u32_to_e4m3x8(words[2 * c], sf_h2)
                e2, e3 = _nvfp4_u32_to_e4m3x8(words[2 * c + 1], sf_h2)
                st_shared_v4_u32(
                    stage_base
                    + (row_in_stage * Int32(8) + ((chunk_base + Int32(c)) ^ row_swiz))
                    * Int32(16),
                    e0,
                    e1,
                    e2,
                    e3,
                )
        else:
            z = Uint32(0)
            for c in cutlass.range_constexpr(4):
                st_shared_v4_u32(
                    stage_base
                    + (row_in_stage * Int32(8) + ((chunk_base + Int32(c)) ^ row_swiz))
                    * Int32(16),
                    z,
                    z,
                    z,
                    z,
                )
        cute.arch.mbarrier_arrive(full_mbar + buf)
        t += Int32(1)


@cute.jit
def _qk_ldmatrix_fp8mma_f16(
    frag_S: cute.Tensor,
    q_fp8_regs: cute.Tensor,
    q_scale_log2_0: Float32,
    q_scale_log2_1: Float32,
    sK_base: Int32,
    lane: Int32,
    sub: cutlass.Constexpr,
):
    """Compute ``frag_S[sub, 0..7] = (Q_e4m3 @ K_e4m3[sub*16 : sub*16+16]^T) * q_scale_log2``."""
    mat = lane // Int32(8)
    k_row = Int32(sub * 16) + (mat & Int32(1)) * Int32(8) + (lane & Int32(7))
    chunk_off = mat >> Int32(1)
    k_swiz = k_row & Int32(7)
    h0 = Uint32(0)
    h1 = Uint32(0)
    h2 = Uint32(0)
    h3 = Uint32(0)
    for k32 in cutlass.range_constexpr(4):
        chunk = Int32(k32 * 2) + chunk_off
        r0, r1, r2, r3 = ldmatrix_m8n8x4_b16(
            sK_base + (k_row * Int32(8) + (chunk ^ k_swiz)) * Int32(16)
        )
        h0, h1 = _mma_m16n8k32_f16_e4m3(
            h0,
            h1,
            q_fp8_regs[k32, 0],
            q_fp8_regs[k32, 1],
            q_fp8_regs[k32, 2],
            q_fp8_regs[k32, 3],
            r0,
            r2,
        )
        h2, h3 = _mma_m16n8k32_f16_e4m3(
            h2,
            h3,
            q_fp8_regs[k32, 0],
            q_fp8_regs[k32, 1],
            q_fp8_regs[k32, 2],
            q_fp8_regs[k32, 3],
            r1,
            r3,
        )
    f0, f1 = _f16x2_to_f32x2(h0)
    f2, f3 = _f16x2_to_f32x2(h1)
    f4, f5 = _f16x2_to_f32x2(h2)
    f6, f7 = _f16x2_to_f32x2(h3)
    s0p = (q_scale_log2_0, q_scale_log2_0)
    s1p = (q_scale_log2_1, q_scale_log2_1)
    f0, f1 = cute.arch.mul_packed_f32x2((f0, f1), s0p)
    f4, f5 = cute.arch.mul_packed_f32x2((f4, f5), s0p)
    f2, f3 = cute.arch.mul_packed_f32x2((f2, f3), s1p)
    f6, f7 = cute.arch.mul_packed_f32x2((f6, f7), s1p)
    frag_S[sub, 0] = f0
    frag_S[sub, 1] = f1
    frag_S[sub, 2] = f2
    frag_S[sub, 3] = f3
    frag_S[sub, 4] = f4
    frag_S[sub, 5] = f5
    frag_S[sub, 6] = f6
    frag_S[sub, 7] = f7


@cute.jit
def _pv_ldmatrix_trans_fp8mma_f16(
    o_frag: cute.Tensor,
    p_fp8_regs: cute.Tensor,
    sV_base: Int32,
    lane: Int32,
):
    """Compute ``O_f32 += fp32(P_e4m3 @ V_e4m3)`` using full-rate FP16-accum per-tile MMA."""
    lane16 = lane & Int32(15)
    lane_row = (
        (lane16 & Int32(0x1))
        | ((lane16 & Int32(0x4)) >> Int32(1))
        | ((lane16 & Int32(0x8)) >> Int32(1))
        | ((lane16 & Int32(0x2)) << Int32(2))
    )
    v_swiz = lane_row & Int32(7)
    row_lo_off = lane_row * Int32(8)
    row_hi_off = (lane_row + Int32(16)) * Int32(8)
    p0 = p_fp8_regs[0]
    p1 = p_fp8_regs[1]
    p2 = p_fp8_regs[2]
    p3 = p_fp8_regs[3]
    z = Uint32(0)
    for mma_d in cutlass.range_constexpr(8):
        c_swiz = Int32(mma_d) ^ v_swiz
        b0_lo, b0_hi = ldmatrix_m16n16x1_trans_b8(
            sV_base + (row_lo_off + c_swiz) * Int32(16)
        )
        b1_lo, b1_hi = ldmatrix_m16n16x1_trans_b8(
            sV_base + (row_hi_off + c_swiz) * Int32(16)
        )
        h0, h1 = _mma_m16n8k32_f16_e4m3(z, z, p0, p1, p2, p3, b0_lo, b1_lo)
        h2, h3 = _mma_m16n8k32_f16_e4m3(z, z, p0, p1, p2, p3, b0_hi, b1_hi)
        f0, f1 = _f16x2_to_f32x2(h0)
        f2, f3 = _f16x2_to_f32x2(h1)
        f4, f5 = _f16x2_to_f32x2(h2)
        f6, f7 = _f16x2_to_f32x2(h3)
        o_frag[mma_d, 0] = o_frag[mma_d, 0] + f0
        o_frag[mma_d, 1] = o_frag[mma_d, 1] + f1
        o_frag[mma_d, 2] = o_frag[mma_d, 2] + f2
        o_frag[mma_d, 3] = o_frag[mma_d, 3] + f3
        o_frag[mma_d, 4] = o_frag[mma_d, 4] + f4
        o_frag[mma_d, 5] = o_frag[mma_d, 5] + f5
        o_frag[mma_d, 6] = o_frag[mma_d, 6] + f6
        o_frag[mma_d, 7] = o_frag[mma_d, 7] + f7


@cute.jit
def _add_factored_relative_bias_subtile(
    frag_S: cute.Tensor,
    mma_kv: cutlass.Constexpr,
    rel_q_regs: cute.Tensor,
    mRelProjBytes: cute.Tensor,
    proj_row_stride_bytes: Int64,
    row_pos: cute.Tensor,
    row_valid: cute.Tensor,
    key_abs_base: Int32,
    lane_pair_base: Int32,
    extent: Int32,
):
    """Add BF16-rounded factored relative bias in log2 units directly to ``frag_S``."""
    log2_e = Float32(LOG2_E)
    for reg_id in cutlass.range_constexpr(8):
        row_slot = (reg_id % 4) // 2
        key_abs = (
            key_abs_base + lane_pair_base + Int32(8 * (reg_id // 4) + (reg_id % 2))
        )
        distance = row_pos[row_slot] - key_abs
        if (
            row_valid[row_slot] != Int32(0)
            and distance >= Int32(0)
            and distance < extent
        ):
            p_base = Int64(distance) * proj_row_stride_bytes
            acc = Float32(0.0)
            for k2 in cutlass.range_constexpr(REL_RANK // 2):
                pw = ld_global_nc_u32(
                    get_ptr_as_int64(mRelProjBytes, p_base + Int64(k2 * 4))
                )
                acc = acc + _bf16x2_dot_f32(rel_q_regs[row_slot, k2], pw)
            bias_bf16 = acc.to(cutlass.BFloat16).to(Float32)
            frag_S[mma_kv, reg_id] = frag_S[mma_kv, reg_id] + bias_bf16 * log2_e


@cute.jit
def _mask_subtile_kv(
    frag_S: cute.Tensor,
    mma_kv: cutlass.Constexpr,
    key_offset: Int32,
    tile_base: Int32,
    tile_tokens: Int32,
    lane_pair_base: Int32,
    causal_hi: cute.Tensor,
    window_lo: cute.Tensor,
    has_window: cutlass.Constexpr,
):
    """Apply bounds, causal, and optional sliding-window mask to ``frag_S[mma_kv, :]``."""
    for reg_id in cutlass.range_constexpr(8):
        row_slot = (reg_id % 4) // 2
        key_rel = (
            key_offset + lane_pair_base + Int32(8 * (reg_id // 4) + (reg_id % 2))
        )
        key_abs = tile_base + key_rel
        masked = (key_rel >= tile_tokens) or (key_abs > causal_hi[row_slot])
        if const_expr(has_window):
            masked = masked or (key_abs < window_lo[row_slot])
        if masked:
            frag_S[mma_kv, reg_id] = Float32(-Float32.inf)


@cute.jit
def _update_mdo_pack_p_fp8_32(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_fp8_regs: cute.Tensor,
):
    """Log2-domain online softmax over 32 KV tokens with skip-rescale and E4M3 packing."""
    p_scale = Float32(P_FP8_SCALE)
    for row_slot in cutlass.range_constexpr(2):
        r0 = row_slot * 2 + 0
        r1 = row_slot * 2 + 1
        r4 = row_slot * 2 + 4
        r5 = row_slot * 2 + 5
        m_prev = Float32(m_frag[row_slot])
        m_new = Float32(m_prev)
        for sub in cutlass.range_constexpr(2):
            m_loc = _max_nan_f32(
                _max_nan_f32(s_frag[sub, r0], s_frag[sub, r1]),
                _max_nan_f32(s_frag[sub, r4], s_frag[sub, r5]),
            )
            m_new = _max_nan_f32(m_new, m_loc)
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
        m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

        has_mass = m_new != -Float32.inf
        if m_new > m_prev and m_prev != -Float32.inf:
            scale_term = _exp2_approx_ftz_f32(m_prev - m_new)
            d_frag[row_slot] = d_frag[row_slot] * scale_term
            sp = (scale_term, scale_term)
            for mma_d in cutlass.range_constexpr(8):
                o0, o1 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r0], o_frag[mma_d, r1]), sp
                )
                o4, o5 = cute.arch.mul_packed_f32x2(
                    (o_frag[mma_d, r4], o_frag[mma_d, r5]), sp
                )
                o_frag[mma_d, r0] = o0
                o_frag[mma_d, r1] = o1
                o_frag[mma_d, r4] = o4
                o_frag[mma_d, r5] = o5

        d_loc = Float32(0.0)
        for sub in cutlass.range_constexpr(2):
            p0 = cutlass.select_(
                has_mass, _exp2_approx_ftz_f32(s_frag[sub, r0] - m_new), Float32(0.0)
            )
            p1 = cutlass.select_(
                has_mass, _exp2_approx_ftz_f32(s_frag[sub, r1] - m_new), Float32(0.0)
            )
            p2 = cutlass.select_(
                has_mass, _exp2_approx_ftz_f32(s_frag[sub, r4] - m_new), Float32(0.0)
            )
            p3 = cutlass.select_(
                has_mass, _exp2_approx_ftz_f32(s_frag[sub, r5] - m_new), Float32(0.0)
            )
            d_loc = d_loc + ((p0 + p1) + (p2 + p3))
            p_fp8_regs[row_slot + 2 * sub] = cvt_f32x4_to_e4m3x4(
                p0 * p_scale,
                p1 * p_scale,
                p2 * p_scale,
                p3 * p_scale,
            )
        d_loc = d_loc + cute.arch.shuffle_sync_bfly(d_loc, offset=2)
        d_loc = d_loc + cute.arch.shuffle_sync_bfly(d_loc, offset=1)
        d_frag[row_slot] = d_frag[row_slot] + d_loc
        m_frag[row_slot] = m_new


class PagedNVFP4PrefillKernel:
    """Warp-specialized 4-producer ldmatrix full-rate FP8-MMA prefill over paged block-16 NVFP4."""

    def __init__(
        self,
        *,
        group_size: int,
        window_left: int = -1,
        has_sinks: bool = False,
        sink_layout: str = "head",
        stage_rows: int = 32,
        num_stages: int = 4,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
        page_size: int = 128,
        has_factored_relative_bias: bool = False,
    ):
        if sink_layout not in ("head", "row"):
            raise ValueError(f"unsupported sink layout {sink_layout!r}")
        if (head_dim_qk, head_dim_vo) != (128, 128):
            raise ValueError("paged_nvfp4 prefill supports 128/128 head dims")
        if stage_rows != 32 or page_size % stage_rows != 0:
            raise ValueError(
                f"FP8-MMA prefill requires stage_rows=32 dividing page_size, got {stage_rows}"
            )

        self.group_size = int(group_size)
        self.num_q_warps = 8
        self.q_heads_per_cta = max(
            h for h in (1, 2, 4, 8) if self.group_size % h == 0 and 8 % h == 0
        )
        self.warps_per_head = self.num_q_warps // self.q_heads_per_cta
        self.q_tokens_per_cta = self.warps_per_head * WARP_Q_ROWS
        self.head_groups_per_kv = self.group_size // self.q_heads_per_cta

        self.window_left = int(window_left)
        self.has_sinks = bool(has_sinks)
        self.sink_per_row = sink_layout == "row"
        self.stage_rows = 32
        self.num_stages = int(num_stages)
        self.head_dim_qk = 128
        self.head_dim_vo = 128
        self.page_size = int(page_size)
        self.has_factored_relative_bias = bool(has_factored_relative_bias)
        self.num_warps = self.num_q_warps + 4  # +2 K producer/dequant, +2 V producer/dequant

    def _get_shared_storage_cls(self):
        return _make_prefill_e4m3_shared_storage_cls(self.num_stages)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # [T, Hq, 128] bf16
        mKCache: cute.Tensor,  # [pages, P, Hkv, 64] u8
        mKScales: cute.Tensor,  # [pages, P, Hkv, 8] u8
        mVCache: cute.Tensor,  # [pages, P, Hkv, 64] u8
        mVScales: cute.Tensor,  # [pages, P, Hkv, 8] u8
        mPageTable: cute.Tensor,  # [B, W] i32
        mSeqLens: cute.Tensor,  # [B] i32
        mCuSeqlensQ: cute.Tensor,  # [B + 1] i32
        mO: cute.Tensor,  # [T, Hq, 128] bf16
        mSinks: cute.Tensor,  # [Hq] or [T, Hq] f32
        mRelFactors: cute.Tensor,  # [T, Hq, 16] bf16
        mRelProj: cute.Tensor,  # [extent, 16] bf16
        max_q_per_req: Int32,
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        batch = mSeqLens.shape[0]
        num_kv_heads = mKCache.shape[2]
        q_tiles = (max_q_per_req + Int32(self.q_tokens_per_cta - 1)) // Int32(
            self.q_tokens_per_cta
        )
        grid_y = num_kv_heads * self.head_groups_per_kv
        sm_scale_log2 = softmax_scale * Float32(LOG2_E)
        self.kernel(
            mQ,
            mKCache,
            mKScales,
            mVCache,
            mVScales,
            mPageTable,
            mSeqLens,
            mCuSeqlensQ,
            mO,
            mSinks,
            mRelFactors,
            mRelProj,
            sm_scale_log2,
        ).launch(
            grid=(q_tiles, grid_y, batch),
            block=[32, self.num_warps, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mKCache: cute.Tensor,
        mKScales: cute.Tensor,
        mVCache: cute.Tensor,
        mVScales: cute.Tensor,
        mPageTable: cute.Tensor,
        mSeqLens: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mSinks: cute.Tensor,
        mRelFactors: cute.Tensor,
        mRelProj: cute.Tensor,
        sm_scale_log2: Float32,
    ):
        STAGE = _STAGE_ROWS
        S = self.num_stages
        NQW = self.num_q_warps
        lane, warp, _ = cute.arch.thread_idx()
        bx, head_group_idx, request_idx = cute.arch.block_idx()
        # Reverse Q-tile scheduling puts longest causal tiles first to eliminate tail waves.
        q_tile_idx = (cute.arch.grid_dim()[0] - Int32(1)) - bx
        kv_head_idx = head_group_idx // Int32(self.head_groups_per_kv)
        head_group_sub = head_group_idx % Int32(self.head_groups_per_kv)
        cta_q_head_base = kv_head_idx * Int32(self.group_size) + head_group_sub * Int32(
            self.q_heads_per_cta
        )

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        seq_len = mSeqLens[request_idx]
        q_tile_base = q_tile_idx * Int32(self.q_tokens_per_cta)
        if q_tile_base >= qo_len or seq_len <= Int32(0):
            _exit_thread()
        context_len = seq_len - qo_len
        q_tile_len = cutlass.select_(
            q_tile_base + Int32(self.q_tokens_per_cta) < qo_len,
            Int32(self.q_tokens_per_cta),
            qo_len - q_tile_base,
        )

        max_q_pos = context_len + q_tile_base + q_tile_len - Int32(1)
        last_key_exclusive = cutlass.select_(
            max_q_pos + Int32(1) < seq_len, max_q_pos + Int32(1), seq_len
        )
        first_key = Int32(0)
        if const_expr(self.window_left >= 0):
            min_win = context_len + q_tile_base - Int32(self.window_left)
            first_key = cutlass.select_(min_win > Int32(0), min_win, Int32(0))
        tile_begin = first_key // Int32(STAGE)
        tile_end = (last_key_exclusive + Int32(STAGE - 1)) // Int32(STAGE)
        ntiles = tile_end - tile_begin

        tidx = lane + warp * Int32(32)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        full_k_mbar = storage.full_k.data_ptr()
        full_v_mbar = storage.full_v.data_ptr()
        empty_k_mbar = storage.empty_k.data_ptr()
        empty_v_mbar = storage.empty_v.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_k_mbar + tidx, Int32(64))
            cute.arch.mbarrier_init(full_v_mbar + tidx, Int32(64))
            cute.arch.mbarrier_init(empty_k_mbar + tidx, Int32(NQW))
            cute.arch.mbarrier_init(empty_v_mbar + tidx, Int32(NQW))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((S * _STAGE_BYTES,), stride=(1,))
        )
        payload_base_addr = shared_ptr_to_u32(payload_u8.iterator)
        chunk_start = tile_begin * Int32(STAGE)

        if warp >= Int32(NQW) and warp < Int32(NQW + 2):
            _run_e4m3_dequant_producer_half(
                lane,
                warp - Int32(NQW),
                mKCache,
                mKScales,
                mPageTable,
                request_idx,
                kv_head_idx,
                chunk_start,
                first_key,
                last_key_exclusive,
                ntiles,
                S,
                0,
                payload_base_addr,
                full_k_mbar,
                empty_k_mbar,
                self.page_size,
            )
        elif warp >= Int32(NQW + 2):
            _run_e4m3_dequant_producer_half(
                lane,
                warp - Int32(NQW + 2),
                mVCache,
                mVScales,
                mPageTable,
                request_idx,
                kv_head_idx,
                chunk_start,
                first_key,
                last_key_exclusive,
                ntiles,
                S,
                _STAGE_HALF_BYTES,
                payload_base_addr,
                full_v_mbar,
                empty_v_mbar,
                self.page_size,
            )
        else:
            lane_group = lane // Int32(4)
            lane_quad = lane % Int32(4)
            lane_pair_base = Int32(2) * lane_quad
            q_head_local = warp // Int32(self.warps_per_head)
            q_head_idx = cta_q_head_base + q_head_local
            warp_tok_in_cta = (warp % Int32(self.warps_per_head)) * Int32(WARP_Q_ROWS)
            warp_q_base = q_tile_base + warp_tok_in_cta

            o_frag = cute.make_rmem_tensor(
                cute.make_layout((8, 8), stride=(8, 1)), Float32
            )
            m_frag = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Float32)
            d_frag = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Float32)
            row_valid = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            q_local = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            row_pos = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            causal_hi = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            window_lo = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            q_scale_log2 = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Float32
            )

            for row_slot in cutlass.range_constexpr(2):
                ql = warp_q_base + lane_group + Int32(8 * row_slot)
                valid_r = ql < qo_len
                row_valid[row_slot] = Int32(valid_r)
                q_local[row_slot] = ql
                rp = context_len + ql
                row_pos[row_slot] = rp
                causal_hi[row_slot] = cutlass.select_(valid_r, rp, Int32(-1))
                window_lo[row_slot] = rp - Int32(self.window_left)

            causal_hi_min = context_len + warp_q_base
            window_lo_max = (
                context_len
                + warp_q_base
                + Int32(WARP_Q_ROWS - 1)
                - Int32(self.window_left)
            )

            # Per-row dynamic E4M3 quantization of Q[16, 128] scaled to [-Q_FP8_MAX, Q_FP8_MAX].
            q_fp8_regs = cute.make_rmem_tensor(
                cute.make_layout((4, 4), stride=(4, 1)), Uint32
            )
            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            q_tok_stride_b = Int64(mQ.stride[0]) * Int64(2)
            q_head_off_b = Int64(q_head_idx) * Int64(mQ.stride[1]) * Int64(2)
            q_bf_tmp = cute.make_rmem_tensor(
                cute.make_layout((4, 2, 2), stride=(4, 2, 1)), Uint32
            )
            for row_slot in cutlass.range_constexpr(2):
                row_b = Int64(q_start + q_local[row_slot]) * q_tok_stride_b + q_head_off_b
                amax = Float32(0.0)
                for k32 in cutlass.range_constexpr(4):
                    for half in cutlass.range_constexpr(2):
                        col = k32 * 32 + half * 16 + Int32(4) * lane_quad
                        w0, w1 = Uint32(0), Uint32(0)
                        if row_valid[row_slot] != Int32(0):
                            w0, w1 = ld_global_nc_v2_u32(
                                get_ptr_as_int64(mQBytes, row_b + Int64(col * Int32(2)))
                            )
                        q_bf_tmp[k32, half, 0] = w0
                        q_bf_tmp[k32, half, 1] = w1
                        f0 = fabs_f32(u32_as_f32(w0 << Uint32(16)))
                        f1 = fabs_f32(u32_as_f32(w0 & Uint32(0xFFFF0000)))
                        f2 = fabs_f32(u32_as_f32(w1 << Uint32(16)))
                        f3 = fabs_f32(u32_as_f32(w1 & Uint32(0xFFFF0000)))
                        amax = fmax_f32(
                            amax, fmax_f32(fmax_f32(f0, f1), fmax_f32(f2, f3))
                        )
                amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
                amax = fmax_f32(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
                nz = amax > Float32(0.0)
                q_row_scale = cutlass.select_(
                    nz, amax * Float32(1.0 / Q_FP8_MAX), Float32(1.0)
                )
                q_scale_log2[row_slot] = q_row_scale * sm_scale_log2
                inv_qs = cutlass.select_(nz, Float32(Q_FP8_MAX) / amax, Float32(0.0))
                for k32 in cutlass.range_constexpr(4):
                    for half in cutlass.range_constexpr(2):
                        w0, w1 = q_bf_tmp[k32, half, 0], q_bf_tmp[k32, half, 1]
                        q_fp8_regs[k32, row_slot + 2 * half] = cvt_f32x4_to_e4m3x4(
                            u32_as_f32(w0 << Uint32(16)) * inv_qs,
                            u32_as_f32(w0 & Uint32(0xFFFF0000)) * inv_qs,
                            u32_as_f32(w1 << Uint32(16)) * inv_qs,
                            u32_as_f32(w1 & Uint32(0xFFFF0000)) * inv_qs,
                        )

            rel_reg_cols = (REL_RANK // 2) if self.has_factored_relative_bias else 1
            rel_q_regs = cute.make_rmem_tensor(
                cute.make_layout((2, rel_reg_cols), stride=(rel_reg_cols, 1)),
                Uint32,
            )
            bias_extent = Int32(0)
            bias_key_lo = Int32(0)
            bias_key_hi = Int32(-1)
            proj_row_stride_b = Int64(0)
            mRelProjBytes = cute.flatten(cute.recast_tensor(mRelProj, cutlass.Uint8))
            if const_expr(self.has_factored_relative_bias):
                bias_extent = Int32(mRelProj.shape[0])
                bias_key_lo = context_len + warp_q_base - bias_extent + Int32(1)
                bias_key_hi = context_len + warp_q_base + Int32(WARP_Q_ROWS - 1)
                proj_row_stride_b = Int64(mRelProj.stride[0]) * Int64(2)
                mRelFacBytes = cute.flatten(
                    cute.recast_tensor(mRelFactors, cutlass.Uint8)
                )
                rf_tok_stride_b = Int64(mRelFactors.stride[0]) * Int64(2)
                rf_head_off_b = (
                    Int64(q_head_idx) * Int64(mRelFactors.stride[1]) * Int64(2)
                )
                for row_slot in cutlass.range_constexpr(2):
                    rb = (
                        Int64(q_start + q_local[row_slot]) * rf_tok_stride_b
                        + rf_head_off_b
                    )
                    for k2 in cutlass.range_constexpr(REL_RANK // 2):
                        v = Uint32(0)
                        if row_valid[row_slot] != Int32(0):
                            v = ld_global_nc_u32(
                                get_ptr_as_int64(mRelFacBytes, rb + Int64(k2 * 4))
                            )
                        rel_q_regs[row_slot, k2] = v

            for mma_d in cutlass.range_constexpr(8):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[row_slot] = Float32(-Float32.inf)
                d_frag[row_slot] = Float32(1.0)
                if const_expr(self.has_sinks):
                    if row_valid[row_slot] != Int32(0):
                        raw = (
                            Float32(mSinks[q_start + q_local[row_slot], q_head_idx])
                            if const_expr(self.sink_per_row)
                            else Float32(mSinks[q_head_idx])
                        )
                        sv = raw * Float32(LOG2_E)
                        m_frag[row_slot] = cutlass.select_(
                            raw == Float32.inf, -Float32.inf, sv
                        )

            frag_S = cute.make_rmem_tensor(
                cute.make_layout((2, 8), stride=(8, 1)), Float32
            )
            p_fp8_regs = cute.make_rmem_tensor(
                cute.make_layout((4,), stride=(1,)), Uint32
            )
            q_sl0 = q_scale_log2[0]
            q_sl1 = q_scale_log2[1]
            warp_max_key = context_len + warp_q_base + Int32(WARP_Q_ROWS - 1)

            t = Int32(0)
            while t < ntiles:
                buf = t % Int32(S)
                phase = (t // Int32(S)) & Int32(1)
                tile_base = chunk_start + t * Int32(STAGE)

                cute.arch.mbarrier_wait(full_k_mbar + buf, phase=phase)
                if tile_base <= warp_max_key:
                    tile_tokens = cutlass.select_(
                        tile_base + Int32(STAGE) < last_key_exclusive,
                        Int32(STAGE),
                        last_key_exclusive - tile_base,
                    )
                    sK_base = payload_base_addr + buf * Int32(_STAGE_BYTES)
                    sV_base = sK_base + Int32(_STAGE_HALF_BYTES)

                    tile_interior = (tile_tokens >= Int32(STAGE)) and (
                        tile_base + Int32(STAGE - 1) <= causal_hi_min
                    )
                    if const_expr(self.window_left >= 0):
                        tile_interior = tile_interior and (tile_base >= window_lo_max)

                    for sub in cutlass.range_constexpr(2):
                        if Int32(sub * 16) < tile_tokens:
                            _qk_ldmatrix_fp8mma_f16(
                                frag_S, q_fp8_regs, q_sl0, q_sl1, sK_base, lane, sub
                            )
                            sub_base = tile_base + Int32(sub * 16)
                            if const_expr(self.has_factored_relative_bias):
                                if (
                                    sub_base <= bias_key_hi
                                    and sub_base + Int32(15) >= bias_key_lo
                                ):
                                    _add_factored_relative_bias_subtile(
                                        frag_S,
                                        sub,
                                        rel_q_regs,
                                        mRelProjBytes,
                                        proj_row_stride_b,
                                        row_pos,
                                        row_valid,
                                        sub_base,
                                        lane_pair_base,
                                        bias_extent,
                                    )
                            if not tile_interior:
                                _mask_subtile_kv(
                                    frag_S,
                                    sub,
                                    Int32(sub * 16),
                                    tile_base,
                                    tile_tokens,
                                    lane_pair_base,
                                    causal_hi,
                                    window_lo,
                                    self.window_left >= 0,
                                )
                        else:
                            for r in cutlass.range_constexpr(8):
                                frag_S[sub, r] = Float32(-Float32.inf)

                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_k_mbar + buf)

                    _update_mdo_pack_p_fp8_32(
                        frag_S, o_frag, m_frag, d_frag, p_fp8_regs
                    )
                    cute.arch.mbarrier_wait(full_v_mbar + buf, phase=phase)
                    _pv_ldmatrix_trans_fp8mma_f16(o_frag, p_fp8_regs, sV_base, lane)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_v_mbar + buf)
                else:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_k_mbar + buf)
                    cute.arch.mbarrier_wait(full_v_mbar + buf, phase=phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(empty_v_mbar + buf)
                t += Int32(1)

            for row_slot in cutlass.range_constexpr(2):
                if row_valid[row_slot] != Int32(0):
                    q_row = q_start + q_local[row_slot]
                    m_val = m_frag[row_slot]
                    d_val = d_frag[row_slot]
                    inv_d = Float32(0.0)
                    if d_val > Float32(0.0) and m_val != -Float32.inf:
                        inv_d = Float32(1.0 / P_FP8_SCALE) / d_val
                    positive_sink = Int32(0)
                    if const_expr(self.has_sinks):
                        raw_sink = (
                            Float32(mSinks[q_row, q_head_idx])
                            if const_expr(self.sink_per_row)
                            else Float32(mSinks[q_head_idx])
                        )
                        positive_sink = Int32(raw_sink == Float32.inf)
                    final_scale = cutlass.select_(
                        positive_sink != Int32(0), Float32(0.0), inv_d
                    )
                    for mma_d in cutlass.range_constexpr(8):
                        c0 = mma_d * 16 + Int32(2) * lane_quad
                        c8 = c0 + Int32(8)
                        mO[q_row, q_head_idx, c0 + 0] = (
                            o_frag[mma_d, row_slot * 2 + 0] * final_scale
                        ).to(cutlass.BFloat16)
                        mO[q_row, q_head_idx, c0 + 1] = (
                            o_frag[mma_d, row_slot * 2 + 1] * final_scale
                        ).to(cutlass.BFloat16)
                        mO[q_row, q_head_idx, c8 + 0] = (
                            o_frag[mma_d, row_slot * 2 + 4] * final_scale
                        ).to(cutlass.BFloat16)
                        mO[q_row, q_head_idx, c8 + 1] = (
                            o_frag[mma_d, row_slot * 2 + 5] * final_scale
                        ).to(cutlass.BFloat16)


__all__ = ["CTA_ROWS", "PagedNVFP4PrefillKernel", "REL_RANK"]
