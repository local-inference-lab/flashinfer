# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared SM120 PTX 9.2 NVFP4 register unpack, MMA, and async staging helpers."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint16, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u64_shared_global,
    get_ptr_as_int64,
    ld_shared_u8_offset,
    ld_shared_u16_offset,
    ld_shared_v2_u32,
)
from b12x._lib.intrinsics.mma import bf16_mma_m16n16k16_f32, mma_m16n8k32_f32_e4m3
from b12x._lib.intrinsics.packed import bfloat2_mul
from b12x.attention.paged.forward_paged import _mask_pv_bf16_pair_windowed


def kv_stage_sizes(
    stage_rows: int, head_dim_qk: int, head_dim_vo: int
) -> tuple[int, int, int, int, int]:
    """Return ``(k_data, k_scale, v_data, v_scale, total)`` bytes per KV stage."""
    kd = stage_rows * (head_dim_qk // 2)
    ks = stage_rows * (head_dim_qk // 16)
    vd = stage_rows * (head_dim_vo // 2)
    vs = stage_rows * (head_dim_vo // 16)
    return kd, ks, vd, vs, kd + ks + vd + vs


def make_kv_shared_storage_cls(num_stages: int, payload_bytes: int):
    """Build the ``cute.struct`` shared-memory layout for producer/consumer KV staging."""

    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "full": cute.struct.MemRange[cutlass.Int64, num_stages],
        "empty": cute.struct.MemRange[cutlass.Int64, num_stages],
        "payload": cute.struct.Align[
            cute.struct.MemRange[cutlass.Uint8, payload_bytes], 16
        ],
    }
    return cute.struct(SharedStorage)


@dsl_user_op
def _cvt_e2m1x2_to_bf16x2_sm120(
    packed_u8_in_u32: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Convert low 8 bits (two E2M1 nibbles) to packed bf16x2 via PTX 9.2 SM120a."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed_u8_in_u32).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 q;
                cvt.u8.u32 q, $1;
                cvt.rn.bf16x2.e2m1x2 $0, q;
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
def _extract_e4m3_broadcast(
    sf_word: Uint32, byte_idx: int, dst_fmt: str = "bf16x2", *, loc=None, ip=None
) -> Uint32:
    """Extract byte ``byte_idx`` (0..3) from ``sf_word`` and broadcast to ``dst_fmt``."""
    sel = int(byte_idx) | (int(byte_idx) << 4)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(sf_word).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 pair;
                .reg .b16 sf;
                prmt.b32 pair, $1, $1, 0x00{sel:02x};
                cvt.u16.u32 sf, pair;
                cvt.rn.{dst_fmt}.e4m3x2 $0, sf;
            }}
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
def _combine_e4m3_pair(
    w0: Uint32, w1: Uint32, byte_idx: int, dst_fmt: str = "bf16x2", *, loc=None, ip=None
) -> Uint32:
    """Pick byte ``byte_idx`` from ``w0`` (low) and ``w1`` (high) and convert to ``dst_fmt``."""
    b = int(byte_idx)
    sel = b | ((b + 4) << 4)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(w0).ir_value(loc=loc, ip=ip), Uint32(w1).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 pair;
                .reg .b16 sf;
                prmt.b32 pair, $1, $2, 0x00{sel:02x};
                cvt.u16.u32 sf, pair;
                cvt.rn.{dst_fmt}.e4m3x2 $0, sf;
            }}
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pv_dequant_pair_sm120(
    raw_k0: Uint32,
    raw_k1: Uint32,
    nib_shift: Uint32,
    fac_bf2: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Extract two E2M1 nibbles (one from raw_k0, one from raw_k1), decode to bf16x2, multiply by fac_bf2."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(raw_k0).ir_value(loc=loc, ip=ip),
                Uint32(raw_k1).ir_value(loc=loc, ip=ip),
                Uint32(nib_shift).ir_value(loc=loc, ip=ip),
                Uint32(fac_bf2).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 n0, n1, packed, val;
                .reg .b8 q;
                shr.u32 n0, $1, $3;
                shr.u32 n1, $2, $3;
                and.b32 n0, n0, 0xf;
                shl.b32 n1, n1, 4;
                or.b32 packed, n0, n1;
                cvt.u8.u32 q, packed;
                cvt.rn.bf16x2.e2m1x2 val, q;
                mul.bf16x2 $0, val, $4;
            }
            """,
            "=r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nvfp4_u16_to_e4m3x4(
    packed_u16: Uint16, sf_f16x2: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Decode 2 NVFP4 bytes (4 nibbles) scaled by ``sf_f16x2`` into 4 packed E4M3 bytes.

    Note: ``cvt.rn.satfinite.e4m3x2.f16x2`` saturates dequantized values at ``+-448.0``.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint16(packed_u16).ir_value(loc=loc, ip=ip),
                Uint32(sf_f16x2).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 u, h0, h1;
                .reg .b8 b0, b1;
                .reg .b16 e0, e1;
                cvt.u32.u16 u, $1;
                cvt.u8.u32 b0, u;
                shr.u32 u, u, 8;
                cvt.u8.u32 b1, u;
                cvt.rn.f16x2.e2m1x2 h0, b0;
                cvt.rn.f16x2.e2m1x2 h1, b1;
                mul.f16x2 h0, h0, $2;
                mul.f16x2 h1, h1, $2;
                cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
                cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
                mov.b32 $0, {e0, e1};
            }
            """,
            "=r,h,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pv_dequant_quad_to_e4m3x4(
    raw_k0: Uint32,
    raw_k1: Uint32,
    raw_k8: Uint32,
    raw_k9: Uint32,
    nib_shift: Uint32,
    fac01_h2: Uint32,
    fac89_h2: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Extract 4 E2M1 nibbles (k0,k1,k8,k9), scale by f16x2 factors, and pack into 4 E4M3 bytes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(raw_k0).ir_value(loc=loc, ip=ip),
                Uint32(raw_k1).ir_value(loc=loc, ip=ip),
                Uint32(raw_k8).ir_value(loc=loc, ip=ip),
                Uint32(raw_k9).ir_value(loc=loc, ip=ip),
                Uint32(nib_shift).ir_value(loc=loc, ip=ip),
                Uint32(fac01_h2).ir_value(loc=loc, ip=ip),
                Uint32(fac89_h2).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 n0, n1, n8, n9, p01, p89, h01, h89;
                .reg .b8 q01, q89;
                .reg .b16 e01, e89;
                shr.u32 n0, $1, $5;
                shr.u32 n1, $2, $5;
                shr.u32 n8, $3, $5;
                shr.u32 n9, $4, $5;
                and.b32 n0, n0, 0xf;
                shl.b32 n1, n1, 4;
                or.b32 p01, n0, n1;
                and.b32 n8, n8, 0xf;
                shl.b32 n9, n9, 4;
                or.b32 p89, n8, n9;
                cvt.u8.u32 q01, p01;
                cvt.u8.u32 q89, p89;
                cvt.rn.f16x2.e2m1x2 h01, q01;
                cvt.rn.f16x2.e2m1x2 h89, q89;
                mul.f16x2 h01, h01, $6;
                mul.f16x2 h89, h89, $7;
                cvt.rn.satfinite.e4m3x2.f16x2 e01, h01;
                cvt.rn.satfinite.e4m3x2.f16x2 e89, h89;
                mov.b32 $0, {e01, e89};
            }
            """,
            "=r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _mask_e4m3x4_windowed(
    packed_e4m3: Uint32,
    k0: Int32,
    valid_rows: Int32,
    window_lo: Int32,
) -> Uint32:
    """Zero out-of-range E4M3 bytes for rows (k0, k0+1, k0+8, k0+9)."""
    k1 = k0 + Int32(1)
    k8 = k0 + Int32(8)
    k9 = k0 + Int32(9)
    m0 = cutlass.select_(
        (k0 < valid_rows) and (k0 >= window_lo), Uint32(0x000000FF), Uint32(0)
    )
    m1 = cutlass.select_(
        (k1 < valid_rows) and (k1 >= window_lo), Uint32(0x0000FF00), Uint32(0)
    )
    m8 = cutlass.select_(
        (k8 < valid_rows) and (k8 >= window_lo), Uint32(0x00FF0000), Uint32(0)
    )
    m9 = cutlass.select_(
        (k9 < valid_rows) and (k9 >= window_lo), Uint32(0xFF000000), Uint32(0)
    )
    return packed_e4m3 & (m0 | m1 | m8 | m9)


@cute.jit
def _cp_async_2d(
    lane: Int32,
    gmem_base: Int64,
    tok_stride: Int64,
    smem_base: Int32,
    token_offset: Int32,
    stage_rows: cutlass.Constexpr,
    row_bytes: cutlass.Constexpr,
    vec_bytes: cutlass.Constexpr,
):
    """Issue coalesced 2D ``cp.async`` copies of ``[stage_rows, row_bytes]``."""
    vecs_per_row: cutlass.Constexpr = row_bytes // vec_bytes
    total_vecs: cutlass.Constexpr = stage_rows * vecs_per_row
    for it in cutlass.range_constexpr((total_vecs + 31) // 32):
        idx = lane + Int32(it * 32)
        if idx < Int32(total_vecs):
            r = idx // Int32(vecs_per_row)
            c = idx - r * Int32(vecs_per_row)
            src = (
                gmem_base + Int64(token_offset + r) * tok_stride + Int64(c * vec_bytes)
            )
            dst = smem_base + r * Int32(row_bytes) + c * Int32(vec_bytes)
            if const_expr(vec_bytes == 16):
                cp_async4_shared_global(dst, src)
            else:
                cp_async_u64_shared_global(dst, src)


@cute.jit
def run_kv_producer_loop(
    lane: Int32,
    mKCache: cute.Tensor,
    mKScales: cute.Tensor,
    mVCache: cute.Tensor,
    mVScales: cute.Tensor,
    mPageTable: cute.Tensor,
    request_idx: Int32,
    kv_head_idx: Int32,
    chunk_start: Int32,
    ntiles: Int32,
    num_key_groups: Int32,
    stages_per_group: Int32,
    payload_base_addr: Int32,
    full_mbar: cute.Pointer,
    empty_mbar: cute.Pointer,
    page_size: cutlass.Constexpr,
    stage_rows: cutlass.Constexpr,
    head_dim_qk: cutlass.Constexpr,
    head_dim_vo: cutlass.Constexpr,
):
    """Warp-specialized producer loop streaming paged NVFP4 K/V + scales via ``cp.async``."""
    kd_b: cutlass.Constexpr = stage_rows * (head_dim_qk // 2)
    ks_b: cutlass.Constexpr = stage_rows * (head_dim_qk // 16)
    vd_b: cutlass.Constexpr = stage_rows * (head_dim_vo // 2)
    vs_b: cutlass.Constexpr = stage_rows * (head_dim_vo // 16)
    stage_b: cutlass.Constexpr = kd_b + ks_b + vd_b + vs_b

    k_base = get_ptr_as_int64(mKCache, 0) + Int64(kv_head_idx) * Int64(
        mKCache.stride[2]
    )
    ks_base = get_ptr_as_int64(mKScales, 0) + Int64(kv_head_idx) * Int64(
        mKScales.stride[2]
    )
    v_base = get_ptr_as_int64(mVCache, 0) + Int64(kv_head_idx) * Int64(
        mVCache.stride[2]
    )
    vs_base = get_ptr_as_int64(mVScales, 0) + Int64(kv_head_idx) * Int64(
        mVScales.stride[2]
    )
    k_ps, k_ts = Int64(mKCache.stride[0]), Int64(mKCache.stride[1])
    ks_ps, ks_ts = Int64(mKScales.stride[0]), Int64(mKScales.stride[1])
    v_ps, v_ts = Int64(mVCache.stride[0]), Int64(mVCache.stride[1])
    vs_ps, vs_ts = Int64(mVScales.stride[0]), Int64(mVScales.stride[1])

    pt_width = mPageTable.shape[1]
    pt_base = chunk_start // Int32(page_size)
    pt_idx = pt_base + lane
    pid_lane = Int32(0)
    if pt_idx < pt_width:
        pid_lane = mPageTable[request_idx, pt_idx]

    t = Int32(0)
    while t < ntiles:
        kg_t = t % num_key_groups
        j = t // num_key_groups
        buf = kg_t * stages_per_group + j % stages_per_group
        use = j // stages_per_group
        if use > Int32(0):
            cute.arch.mbarrier_wait(empty_mbar + buf, phase=(use - Int32(1)) & Int32(1))
        token_base = chunk_start + t * Int32(stage_rows)
        page_idx = token_base // Int32(page_size)
        if page_idx - pt_base >= Int32(32):
            pt_base = pt_base + Int32(32)
            pt_idx = pt_base + lane
            pid_lane = Int32(0)
            if pt_idx < pt_width:
                pid_lane = mPageTable[request_idx, pt_idx]
        page_id = cute.arch.shuffle_sync(pid_lane, page_idx - pt_base)
        tok_in_page = token_base - page_idx * Int32(page_size)
        sK_data = payload_base_addr + buf * Int32(stage_b)
        sK_scale = sK_data + Int32(kd_b)
        sV_data = sK_scale + Int32(ks_b)
        sV_scale = sV_data + Int32(vd_b)
        _cp_async_2d(
            lane,
            k_base + Int64(page_id) * k_ps,
            k_ts,
            sK_data,
            tok_in_page,
            stage_rows,
            head_dim_qk // 2,
            16,
        )
        _cp_async_2d(
            lane,
            ks_base + Int64(page_id) * ks_ps,
            ks_ts,
            sK_scale,
            tok_in_page,
            stage_rows,
            head_dim_qk // 16,
            8,
        )
        _cp_async_2d(
            lane,
            v_base + Int64(page_id) * v_ps,
            v_ts,
            sV_data,
            tok_in_page,
            stage_rows,
            head_dim_vo // 2,
            16,
        )
        _cp_async_2d(
            lane,
            vs_base + Int64(page_id) * vs_ps,
            vs_ts,
            sV_scale,
            tok_in_page,
            stage_rows,
            head_dim_vo // 16,
            8,
        )
        cute.arch.cp_async_mbarrier_arrive_noinc(full_mbar + buf)
        t += Int32(1)


@cute.jit
def qk_fp8mma_into_sfrag(
    frag_S: cute.Tensor,
    q_fp8_regs: cute.Tensor,  # [4, NUM_MMA_Q, 4] u32 (Dqk=128)
    sK_data_addr: Int32,
    sK_scale_addr: Int32,
    lane: Int32,
    key_base: Int32,
    mma_kv: cutlass.Constexpr = 0,
    num_mma_q: cutlass.Constexpr = 2,
):
    """S[num_mma_q*16, 16] += Q_e4m3 @ K_nvfp4->e4m3^T using SM120 FP8 m16n8k32 MMA (D=128)."""
    g = lane // Int32(4)
    t = lane % Int32(4)
    k_lo = key_base + g
    k_hi = k_lo + Int32(8)
    sf_lo_0, sf_lo_1 = ld_shared_v2_u32(sK_scale_addr + k_lo * Int32(8))
    sf_hi_0, sf_hi_1 = ld_shared_v2_u32(sK_scale_addr + k_hi * Int32(8))
    data_base_lo = sK_data_addr + k_lo * Int32(64) + Int32(2) * t
    data_base_hi = sK_data_addr + k_hi * Int32(64) + Int32(2) * t
    for k32 in cutlass.range_constexpr(4):
        off0: cutlass.Constexpr = k32 * 16
        off8: cutlass.Constexpr = off0 + 8
        s0: cutlass.Constexpr = k32 * 2
        s1: cutlass.Constexpr = s0 + 1
        sf_lo_a = _extract_e4m3_broadcast(
            sf_lo_0 if const_expr(s0 < 4) else sf_lo_1, s0 & 3, "f16x2"
        )
        sf_lo_b = _extract_e4m3_broadcast(
            sf_lo_0 if const_expr(s1 < 4) else sf_lo_1, s1 & 3, "f16x2"
        )
        sf_hi_a = _extract_e4m3_broadcast(
            sf_hi_0 if const_expr(s0 < 4) else sf_hi_1, s0 & 3, "f16x2"
        )
        sf_hi_b = _extract_e4m3_broadcast(
            sf_hi_0 if const_expr(s1 < 4) else sf_hi_1, s1 & 3, "f16x2"
        )
        b0 = _nvfp4_u16_to_e4m3x4(ld_shared_u16_offset(data_base_lo, off0), sf_lo_a)
        b1 = _nvfp4_u16_to_e4m3x4(ld_shared_u16_offset(data_base_lo, off8), sf_lo_b)
        b2 = _nvfp4_u16_to_e4m3x4(ld_shared_u16_offset(data_base_hi, off0), sf_hi_a)
        b3 = _nvfp4_u16_to_e4m3x4(ld_shared_u16_offset(data_base_hi, off8), sf_hi_b)
        for mq in cutlass.range_constexpr(num_mma_q):
            d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                frag_S[mq, mma_kv, 0],
                frag_S[mq, mma_kv, 1],
                frag_S[mq, mma_kv, 2],
                frag_S[mq, mma_kv, 3],
                q_fp8_regs[k32, mq, 0],
                q_fp8_regs[k32, mq, 1],
                q_fp8_regs[k32, mq, 2],
                q_fp8_regs[k32, mq, 3],
                b0,
                b1,
            )
            d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                frag_S[mq, mma_kv, 4],
                frag_S[mq, mma_kv, 5],
                frag_S[mq, mma_kv, 6],
                frag_S[mq, mma_kv, 7],
                q_fp8_regs[k32, mq, 0],
                q_fp8_regs[k32, mq, 1],
                q_fp8_regs[k32, mq, 2],
                q_fp8_regs[k32, mq, 3],
                b2,
                b3,
            )
            frag_S[mq, mma_kv, 0] = d0
            frag_S[mq, mma_kv, 1] = d1
            frag_S[mq, mma_kv, 2] = d2
            frag_S[mq, mma_kv, 3] = d3
            frag_S[mq, mma_kv, 4] = d4
            frag_S[mq, mma_kv, 5] = d5
            frag_S[mq, mma_kv, 6] = d6
            frag_S[mq, mma_kv, 7] = d7


@cute.jit
def pv_fp8mma_32_into_ofrag(
    o_frag: cute.Tensor,
    p_fp8_regs: cute.Tensor,  # [NUM_MMA_Q, 4] u32
    sV_data_addr: Int32,
    sV_scale_addr: Int32,
    lane: Int32,
    valid_rows: Int32,
    window_lo: Int32,
    num_mma_q: cutlass.Constexpr = 2,
    skip_mask: cutlass.Constexpr = False,
):
    """O[num_mma_q*16, 128] += P_e4m3[*, 32] @ V_nvfp4->e4m3[32, 128] via FP8 m16n8k32 MMA."""
    g = lane // Int32(4)
    t = lane % Int32(4)
    k0 = Int32(2) * t
    k1, k8, k9 = k0 + Int32(1), k0 + Int32(8), k0 + Int32(9)
    k16 = k0 + Int32(16)
    k17, k24, k25 = k16 + Int32(1), k16 + Int32(8), k16 + Int32(9)
    g_byte = g // Int32(2)
    nib_shift = Uint32((g & Int32(1)) * Int32(4))

    s0_0, s0_1 = ld_shared_v2_u32(sV_scale_addr + k0 * Int32(8))
    s1_0, s1_1 = ld_shared_v2_u32(sV_scale_addr + k1 * Int32(8))
    s8_0, s8_1 = ld_shared_v2_u32(sV_scale_addr + k8 * Int32(8))
    s9_0, s9_1 = ld_shared_v2_u32(sV_scale_addr + k9 * Int32(8))
    s16_0, s16_1 = ld_shared_v2_u32(sV_scale_addr + k16 * Int32(8))
    s17_0, s17_1 = ld_shared_v2_u32(sV_scale_addr + k17 * Int32(8))
    s24_0, s24_1 = ld_shared_v2_u32(sV_scale_addr + k24 * Int32(8))
    s25_0, s25_1 = ld_shared_v2_u32(sV_scale_addr + k25 * Int32(8))

    d0_base = sV_data_addr + k0 * Int32(64) + g_byte
    d1_base = sV_data_addr + k1 * Int32(64) + g_byte
    d8_base = sV_data_addr + k8 * Int32(64) + g_byte
    d9_base = sV_data_addr + k9 * Int32(64) + g_byte
    d16_base = sV_data_addr + k16 * Int32(64) + g_byte
    d17_base = sV_data_addr + k17 * Int32(64) + g_byte
    d24_base = sV_data_addr + k24 * Int32(64) + g_byte
    d25_base = sV_data_addr + k25 * Int32(64) + g_byte

    for mma_d in cutlass.range_constexpr(8):
        off_lo: cutlass.Constexpr = mma_d * 8
        off_hi: cutlass.Constexpr = off_lo + 4
        hi_w: cutlass.Constexpr = mma_d >= 4
        b_idx: cutlass.Constexpr = mma_d & 3
        f01 = _combine_e4m3_pair(
            s0_1 if hi_w else s0_0, s1_1 if hi_w else s1_0, b_idx, "f16x2"
        )
        f89 = _combine_e4m3_pair(
            s8_1 if hi_w else s8_0, s9_1 if hi_w else s9_0, b_idx, "f16x2"
        )
        f1617 = _combine_e4m3_pair(
            s16_1 if hi_w else s16_0, s17_1 if hi_w else s17_0, b_idx, "f16x2"
        )
        f2425 = _combine_e4m3_pair(
            s24_1 if hi_w else s24_0, s25_1 if hi_w else s25_0, b_idx, "f16x2"
        )

        b0_lo = _pv_dequant_quad_to_e4m3x4(
            ld_shared_u8_offset(d0_base, off_lo),
            ld_shared_u8_offset(d1_base, off_lo),
            ld_shared_u8_offset(d8_base, off_lo),
            ld_shared_u8_offset(d9_base, off_lo),
            nib_shift,
            f01,
            f89,
        )
        b1_lo = _pv_dequant_quad_to_e4m3x4(
            ld_shared_u8_offset(d16_base, off_lo),
            ld_shared_u8_offset(d17_base, off_lo),
            ld_shared_u8_offset(d24_base, off_lo),
            ld_shared_u8_offset(d25_base, off_lo),
            nib_shift,
            f1617,
            f2425,
        )
        b0_hi = _pv_dequant_quad_to_e4m3x4(
            ld_shared_u8_offset(d0_base, off_hi),
            ld_shared_u8_offset(d1_base, off_hi),
            ld_shared_u8_offset(d8_base, off_hi),
            ld_shared_u8_offset(d9_base, off_hi),
            nib_shift,
            f01,
            f89,
        )
        b1_hi = _pv_dequant_quad_to_e4m3x4(
            ld_shared_u8_offset(d16_base, off_hi),
            ld_shared_u8_offset(d17_base, off_hi),
            ld_shared_u8_offset(d24_base, off_hi),
            ld_shared_u8_offset(d25_base, off_hi),
            nib_shift,
            f1617,
            f2425,
        )
        if const_expr(not skip_mask):
            b0_lo = _mask_e4m3x4_windowed(b0_lo, k0, valid_rows, window_lo)
            b1_lo = _mask_e4m3x4_windowed(b1_lo, k16, valid_rows, window_lo)
            b0_hi = _mask_e4m3x4_windowed(b0_hi, k0, valid_rows, window_lo)
            b1_hi = _mask_e4m3x4_windowed(b1_hi, k16, valid_rows, window_lo)

        for mq in cutlass.range_constexpr(num_mma_q):
            d0, d1, d2, d3 = mma_m16n8k32_f32_e4m3(
                o_frag[mq, mma_d, 0],
                o_frag[mq, mma_d, 1],
                o_frag[mq, mma_d, 2],
                o_frag[mq, mma_d, 3],
                p_fp8_regs[mq, 0],
                p_fp8_regs[mq, 1],
                p_fp8_regs[mq, 2],
                p_fp8_regs[mq, 3],
                b0_lo,
                b1_lo,
            )
            d4, d5, d6, d7 = mma_m16n8k32_f32_e4m3(
                o_frag[mq, mma_d, 4],
                o_frag[mq, mma_d, 5],
                o_frag[mq, mma_d, 6],
                o_frag[mq, mma_d, 7],
                p_fp8_regs[mq, 0],
                p_fp8_regs[mq, 1],
                p_fp8_regs[mq, 2],
                p_fp8_regs[mq, 3],
                b0_hi,
                b1_hi,
            )
            o_frag[mq, mma_d, 0] = d0
            o_frag[mq, mma_d, 1] = d1
            o_frag[mq, mma_d, 2] = d2
            o_frag[mq, mma_d, 3] = d3
            o_frag[mq, mma_d, 4] = d4
            o_frag[mq, mma_d, 5] = d5
            o_frag[mq, mma_d, 6] = d6
            o_frag[mq, mma_d, 7] = d7


@cute.jit
def qk_nvfp4_into_sfrag(
    frag_S: cute.Tensor,
    q_regs: cute.Tensor,
    sK_data_addr: Int32,
    sK_scale_addr: Int32,
    lane: Int32,
    key_base: Int32,
    mma_kv: cutlass.Constexpr = 0,
    num_mma_q: cutlass.Constexpr = 1,
):
    """S[num_mma_q*16, 16] += Q @ K_nvfp4->bf16^T using SM120 BF16 m16n16k16 MMA (D=128)."""
    g = lane // Int32(4)
    t = lane % Int32(4)
    k_lo = key_base + g
    k_hi = k_lo + Int32(8)
    sf_lo_0, sf_lo_1 = ld_shared_v2_u32(sK_scale_addr + k_lo * Int32(8))
    sf_hi_0, sf_hi_1 = ld_shared_v2_u32(sK_scale_addr + k_hi * Int32(8))
    data_base_lo = sK_data_addr + k_lo * Int32(64) + t
    data_base_hi = sK_data_addr + k_hi * Int32(64) + t
    for s_idx in cutlass.range_constexpr(8):
        off0: cutlass.Constexpr = s_idx * 8
        off4: cutlass.Constexpr = off0 + 4
        sf_lo_bf2 = _extract_e4m3_broadcast(
            sf_lo_0 if const_expr(s_idx < 4) else sf_lo_1, s_idx & 3, "bf16x2"
        )
        sf_hi_bf2 = _extract_e4m3_broadcast(
            sf_hi_0 if const_expr(s_idx < 4) else sf_hi_1, s_idx & 3, "bf16x2"
        )
        b0 = bfloat2_mul(
            _cvt_e2m1x2_to_bf16x2_sm120(ld_shared_u8_offset(data_base_lo, off0)),
            sf_lo_bf2,
        )
        b1 = bfloat2_mul(
            _cvt_e2m1x2_to_bf16x2_sm120(ld_shared_u8_offset(data_base_lo, off4)),
            sf_lo_bf2,
        )
        b2 = bfloat2_mul(
            _cvt_e2m1x2_to_bf16x2_sm120(ld_shared_u8_offset(data_base_hi, off0)),
            sf_hi_bf2,
        )
        b3 = bfloat2_mul(
            _cvt_e2m1x2_to_bf16x2_sm120(ld_shared_u8_offset(data_base_hi, off4)),
            sf_hi_bf2,
        )
        for mq in cutlass.range_constexpr(num_mma_q):
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                frag_S[mq, mma_kv, 0],
                frag_S[mq, mma_kv, 1],
                frag_S[mq, mma_kv, 2],
                frag_S[mq, mma_kv, 3],
                frag_S[mq, mma_kv, 4],
                frag_S[mq, mma_kv, 5],
                frag_S[mq, mma_kv, 6],
                frag_S[mq, mma_kv, 7],
                q_regs[s_idx, mq, 0],
                q_regs[s_idx, mq, 1],
                q_regs[s_idx, mq, 2],
                q_regs[s_idx, mq, 3],
                b0,
                b1,
                b2,
                b3,
            )
            frag_S[mq, mma_kv, 0] = d0
            frag_S[mq, mma_kv, 1] = d1
            frag_S[mq, mma_kv, 2] = d2
            frag_S[mq, mma_kv, 3] = d3
            frag_S[mq, mma_kv, 4] = d4
            frag_S[mq, mma_kv, 5] = d5
            frag_S[mq, mma_kv, 6] = d6
            frag_S[mq, mma_kv, 7] = d7


@cute.jit
def pv_nvfp4_into_ofrag(
    o_frag: cute.Tensor,
    p_frag: cute.Tensor,
    sV_data_addr: Int32,
    sV_scale_addr: Int32,
    lane: Int32,
    key_base: Int32,
    valid_rows: Int32,
    window_lo: Int32,
    mma_kv: cutlass.Constexpr = 0,
    num_mma_q: cutlass.Constexpr = 1,
    skip_mask: cutlass.Constexpr = False,
):
    """O[num_mma_q*16, 128] += P @ V_nvfp4->bf16 using SM120 BF16 m16n16k16 MMA (D=128)."""
    g = lane // Int32(4)
    t = lane % Int32(4)
    k0 = key_base + Int32(2) * t
    k1, k8, k9 = k0 + Int32(1), k0 + Int32(8), k0 + Int32(9)
    g_byte = g // Int32(2)
    nib_shift = Uint32((g & Int32(1)) * Int32(4))

    sf0_w0, sf0_w1 = ld_shared_v2_u32(sV_scale_addr + k0 * Int32(8))
    sf1_w0, sf1_w1 = ld_shared_v2_u32(sV_scale_addr + k1 * Int32(8))
    sf8_w0, sf8_w1 = ld_shared_v2_u32(sV_scale_addr + k8 * Int32(8))
    sf9_w0, sf9_w1 = ld_shared_v2_u32(sV_scale_addr + k9 * Int32(8))
    d0_base = sV_data_addr + k0 * Int32(64) + g_byte
    d1_base = sV_data_addr + k1 * Int32(64) + g_byte
    d8_base = sV_data_addr + k8 * Int32(64) + g_byte
    d9_base = sV_data_addr + k9 * Int32(64) + g_byte

    for mma_d in cutlass.range_constexpr(8):
        off_lo: cutlass.Constexpr = mma_d * 8
        off_hi: cutlass.Constexpr = off_lo + 4
        hi_w: cutlass.Constexpr = mma_d >= 4
        b_idx: cutlass.Constexpr = mma_d & 3
        fac01 = _combine_e4m3_pair(
            sf0_w1 if hi_w else sf0_w0, sf1_w1 if hi_w else sf1_w0, b_idx, "bf16x2"
        )
        fac89 = _combine_e4m3_pair(
            sf8_w1 if hi_w else sf8_w0, sf9_w1 if hi_w else sf9_w0, b_idx, "bf16x2"
        )

        b0 = _pv_dequant_pair_sm120(
            ld_shared_u8_offset(d0_base, off_lo),
            ld_shared_u8_offset(d1_base, off_lo),
            nib_shift,
            fac01,
        )
        b1 = _pv_dequant_pair_sm120(
            ld_shared_u8_offset(d8_base, off_lo),
            ld_shared_u8_offset(d9_base, off_lo),
            nib_shift,
            fac89,
        )
        b2 = _pv_dequant_pair_sm120(
            ld_shared_u8_offset(d0_base, off_hi),
            ld_shared_u8_offset(d1_base, off_hi),
            nib_shift,
            fac01,
        )
        b3 = _pv_dequant_pair_sm120(
            ld_shared_u8_offset(d8_base, off_hi),
            ld_shared_u8_offset(d9_base, off_hi),
            nib_shift,
            fac89,
        )

        if const_expr(not skip_mask):
            b0 = _mask_pv_bf16_pair_windowed(b0, k0, valid_rows, window_lo)
            b1 = _mask_pv_bf16_pair_windowed(b1, k8, valid_rows, window_lo)
            b2 = _mask_pv_bf16_pair_windowed(b2, k0, valid_rows, window_lo)
            b3 = _mask_pv_bf16_pair_windowed(b3, k8, valid_rows, window_lo)

        for mq in cutlass.range_constexpr(num_mma_q):
            d0, d1, d2, d3, d4, d5, d6, d7 = bf16_mma_m16n16k16_f32(
                o_frag[mq, mma_d, 0],
                o_frag[mq, mma_d, 1],
                o_frag[mq, mma_d, 2],
                o_frag[mq, mma_d, 3],
                o_frag[mq, mma_d, 4],
                o_frag[mq, mma_d, 5],
                o_frag[mq, mma_d, 6],
                o_frag[mq, mma_d, 7],
                p_frag[mq, mma_kv, 0],
                p_frag[mq, mma_kv, 1],
                p_frag[mq, mma_kv, 2],
                p_frag[mq, mma_kv, 3],
                b0,
                b1,
                b2,
                b3,
            )
            o_frag[mq, mma_d, 0] = d0
            o_frag[mq, mma_d, 1] = d1
            o_frag[mq, mma_d, 2] = d2
            o_frag[mq, mma_d, 3] = d3
            o_frag[mq, mma_d, 4] = d4
            o_frag[mq, mma_d, 5] = d5
            o_frag[mq, mma_d, 6] = d6
            o_frag[mq, mma_d, 7] = d7
