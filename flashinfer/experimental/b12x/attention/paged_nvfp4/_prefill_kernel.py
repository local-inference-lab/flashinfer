# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 FP8-MMA 2x-Q-reuse GQA-grouped tiled prefill/extend over paged NVFP4 (D=128).

Uses SM120's 2x-throughput ``mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32``
for both QK and PV:
  - Per-row dynamic E4M3 quantization of Q (computed once per CTA before the KV loop).
  - In-register ``e2m1 * e4m3_scale -> e4m3`` (saturated at ``+-448.0``) shared
    across 2 Q tiles/warp.
  - Zero-shuffle ``P`` packing (``P * 448.0 -> e4m3x4``) with matching
    ``(k0, k1, k8, k9)`` permutation in ``V``, processing all 32 KV tokens per
    stage in one K=32 PV MMA.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics.conversion import cvt_f32x4_to_e4m3x4
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    shared_ptr_to_u32,
    u32_as_f32,
)
from b12x.attention.paged.forward_paged import (
    _exit_thread,
    _exp2_approx_ftz_f32,
    _max_nan_f32,
)
from b12x.attention.paged_decode._kernel import LOG2_E

from ._math import (
    kv_stage_sizes,
    make_kv_shared_storage_cls,
    pv_fp8mma_32_into_ofrag,
    qk_fp8mma_into_sfrag,
    run_kv_producer_loop,
)

NUM_Q_WARPS = 4
NUM_MMA_Q = 2
WARP_Q_ROWS = NUM_MMA_Q * 16  # 32 Q rows per consumer warp
CTA_ROWS = NUM_Q_WARPS * WARP_Q_ROWS  # 128 query rows per CTA
REL_RANK = 16
P_FP8_SCALE = 448.0


@cute.jit
def _bf16x2_dot_f32(a: Uint32, b: Uint32) -> Float32:
    a0 = u32_as_f32(a << Uint32(16))
    a1 = u32_as_f32(a & Uint32(0xFFFF0000))
    b0 = u32_as_f32(b << Uint32(16))
    b1 = u32_as_f32(b & Uint32(0xFFFF0000))
    return a0 * b0 + a1 * b1


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
    inv_sm_scale: Float32,
):
    """Add BF16-rounded factored relative bias ``dot(factor[q], proj[d]) * inv_sm_scale``."""
    for mq in cutlass.range_constexpr(NUM_MMA_Q):
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            key_abs = (
                key_abs_base + lane_pair_base + Int32(8 * (reg_id // 4) + (reg_id % 2))
            )
            distance = row_pos[mq, row_slot] - key_abs
            if (
                row_valid[mq, row_slot] != Int32(0)
                and distance >= Int32(0)
                and distance < extent
            ):
                p_base = Int64(distance) * proj_row_stride_bytes
                acc = Float32(0.0)
                for k2 in cutlass.range_constexpr(REL_RANK // 2):
                    pw = ld_global_nc_u32(
                        get_ptr_as_int64(mRelProjBytes, p_base + Int64(k2 * 4))
                    )
                    acc = acc + _bf16x2_dot_f32(rel_q_regs[mq, row_slot, k2], pw)
                bias_bf16 = acc.to(cutlass.BFloat16).to(Float32)
                frag_S[mq, mma_kv, reg_id] = (
                    frag_S[mq, mma_kv, reg_id] + bias_bf16 * inv_sm_scale
                )


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
    """Apply bounds, causal, and optional sliding-window mask to ``frag_S[:, mma_kv, :]``."""
    for mq in cutlass.range_constexpr(NUM_MMA_Q):
        for reg_id in cutlass.range_constexpr(8):
            row_slot = (reg_id % 4) // 2
            key_rel = (
                key_offset + lane_pair_base + Int32(8 * (reg_id // 4) + (reg_id % 2))
            )
            key_abs = tile_base + key_rel
            masked = (key_rel >= tile_tokens) or (key_abs > causal_hi[mq, row_slot])
            if const_expr(has_window):
                masked = masked or (key_abs < window_lo[mq, row_slot])
            if masked:
                frag_S[mq, mma_kv, reg_id] = Float32(-Float32.inf)


@cute.jit
def _update_mdo_pack_p_fp8_32(
    s_frag: cute.Tensor,
    o_frag: cute.Tensor,
    m_frag: cute.Tensor,
    d_frag: cute.Tensor,
    p_fp8_regs: cute.Tensor,
    sm_scale_log2: Float32,
):
    """Online softmax over 32 KV tokens and zero-shuffle E4M3 packing for K=32 PV MMA."""
    for mq in cutlass.range_constexpr(NUM_MMA_Q):
        for row_slot in cutlass.range_constexpr(2):
            m_prev = Float32(m_frag[mq, row_slot])
            m_new = Float32(m_prev)
            for sub in cutlass.range_constexpr(2):
                m_loc = _max_nan_f32(
                    _max_nan_f32(
                        s_frag[mq, sub, row_slot * 2 + 0],
                        s_frag[mq, sub, row_slot * 2 + 1],
                    ),
                    _max_nan_f32(
                        s_frag[mq, sub, row_slot * 2 + 4],
                        s_frag[mq, sub, row_slot * 2 + 5],
                    ),
                )
                m_new = _max_nan_f32(m_new, m_loc)
            m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=2))
            m_new = _max_nan_f32(m_new, cute.arch.shuffle_sync_bfly(m_new, offset=1))

            has_mass = m_new != -Float32.inf
            scale_term = cutlass.select_(
                has_mass,
                _exp2_approx_ftz_f32((m_prev - m_new) * sm_scale_log2),
                Float32(1.0),
            )
            d_frag[mq, row_slot] = d_frag[mq, row_slot] * scale_term
            for mma_d in cutlass.range_constexpr(8):
                o_frag[mq, mma_d, row_slot * 2 + 0] = (
                    o_frag[mq, mma_d, row_slot * 2 + 0] * scale_term
                )
                o_frag[mq, mma_d, row_slot * 2 + 1] = (
                    o_frag[mq, mma_d, row_slot * 2 + 1] * scale_term
                )
                o_frag[mq, mma_d, row_slot * 2 + 4] = (
                    o_frag[mq, mma_d, row_slot * 2 + 4] * scale_term
                )
                o_frag[mq, mma_d, row_slot * 2 + 5] = (
                    o_frag[mq, mma_d, row_slot * 2 + 5] * scale_term
                )

            d_loc = Float32(0.0)
            for sub in cutlass.range_constexpr(2):
                p0 = cutlass.select_(
                    has_mass,
                    _exp2_approx_ftz_f32(
                        (s_frag[mq, sub, row_slot * 2 + 0] - m_new) * sm_scale_log2
                    ),
                    Float32(0.0),
                )
                p1 = cutlass.select_(
                    has_mass,
                    _exp2_approx_ftz_f32(
                        (s_frag[mq, sub, row_slot * 2 + 1] - m_new) * sm_scale_log2
                    ),
                    Float32(0.0),
                )
                p2 = cutlass.select_(
                    has_mass,
                    _exp2_approx_ftz_f32(
                        (s_frag[mq, sub, row_slot * 2 + 4] - m_new) * sm_scale_log2
                    ),
                    Float32(0.0),
                )
                p3 = cutlass.select_(
                    has_mass,
                    _exp2_approx_ftz_f32(
                        (s_frag[mq, sub, row_slot * 2 + 5] - m_new) * sm_scale_log2
                    ),
                    Float32(0.0),
                )
                d_loc = d_loc + (p0 + p1 + p2 + p3)
                p_fp8_regs[mq, row_slot + 2 * sub] = cvt_f32x4_to_e4m3x4(
                    p0 * Float32(P_FP8_SCALE),
                    p1 * Float32(P_FP8_SCALE),
                    p2 * Float32(P_FP8_SCALE),
                    p3 * Float32(P_FP8_SCALE),
                )
            d_loc = d_loc + cute.arch.shuffle_sync_bfly(d_loc, offset=2)
            d_loc = d_loc + cute.arch.shuffle_sync_bfly(d_loc, offset=1)
            d_frag[mq, row_slot] = d_frag[mq, row_slot] + d_loc
            m_frag[mq, row_slot] = m_new


class PagedNVFP4PrefillKernel:
    """FP8-MMA 2x-Q-reuse GQA-grouped tiled prefill/extend over paged block-16 NVFP4."""

    def __init__(
        self,
        *,
        group_size: int,
        window_left: int = -1,
        has_sinks: bool = False,
        sink_layout: str = "head",
        stage_rows: int = 32,
        num_stages: int = 6,
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
        self.q_heads_per_cta = max(
            h for h in (1, 2, 4) if self.group_size % h == 0 and NUM_Q_WARPS % h == 0
        )
        self.warps_per_head = NUM_Q_WARPS // self.q_heads_per_cta
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
        self.num_warps = NUM_Q_WARPS + 1

        (
            self.k_data_stage_bytes,
            self.k_scale_stage_bytes,
            self.v_data_stage_bytes,
            self.v_scale_stage_bytes,
            self.stage_bytes,
        ) = kv_stage_sizes(self.stage_rows, self.head_dim_qk, self.head_dim_vo)
        self.shared_storage_bytes = self.num_stages * self.stage_bytes

    def _get_shared_storage_cls(self):
        return make_kv_shared_storage_cls(self.num_stages, self.shared_storage_bytes)

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
        inv_sm_scale = Float32(1.0) / softmax_scale
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
            inv_sm_scale,
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
        inv_sm_scale: Float32,
    ):
        STAGE = 32
        S = self.num_stages
        lane, warp, _ = cute.arch.thread_idx()
        q_tile_idx, head_group_idx, request_idx = cute.arch.block_idx()
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
        full_mbar = storage.full.data_ptr()
        empty_mbar = storage.empty.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_mbar + tidx, Int32(32))
            cute.arch.mbarrier_init(empty_mbar + tidx, Int32(NUM_Q_WARPS))
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((self.shared_storage_bytes,), stride=(1,))
        )
        payload_base_addr = shared_ptr_to_u32(payload_u8.iterator)
        chunk_start = tile_begin * Int32(STAGE)

        if warp == Int32(NUM_Q_WARPS):
            run_kv_producer_loop(
                lane,
                mKCache,
                mKScales,
                mVCache,
                mVScales,
                mPageTable,
                request_idx,
                kv_head_idx,
                chunk_start,
                ntiles,
                Int32(1),
                Int32(S),
                payload_base_addr,
                full_mbar,
                empty_mbar,
                self.page_size,
                STAGE,
                128,
                128,
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
                cute.make_layout((NUM_MMA_Q, 8, 8), stride=(64, 8, 1)), Float32
            )
            m_frag = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Float32
            )
            d_frag = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Float32
            )
            row_valid = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Int32
            )
            q_local = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Int32
            )
            row_pos = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Int32
            )
            causal_hi = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Int32
            )
            window_lo = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Int32
            )
            q_row_scale = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 2), stride=(2, 1)), Float32
            )

            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                for row_slot in cutlass.range_constexpr(2):
                    ql = warp_q_base + Int32(mq * 16) + lane_group + Int32(8 * row_slot)
                    valid_r = ql < qo_len
                    row_valid[mq, row_slot] = Int32(valid_r)
                    q_local[mq, row_slot] = ql
                    rp = context_len + ql
                    row_pos[mq, row_slot] = rp
                    causal_hi[mq, row_slot] = cutlass.select_(valid_r, rp, Int32(-1))
                    window_lo[mq, row_slot] = rp - Int32(self.window_left)

            warp_window_lo = (
                (context_len + warp_q_base - Int32(self.window_left))
                if const_expr(self.window_left >= 0)
                else Int32(0)
            )
            causal_hi_min = context_len + warp_q_base
            window_lo_max = (
                context_len
                + warp_q_base
                + Int32(WARP_Q_ROWS - 1)
                - Int32(self.window_left)
            )

            # Per-row dynamic E4M3 quantization of Q[NUM_MMA_Q*16, 128].
            q_fp8_regs = cute.make_rmem_tensor(
                cute.make_layout((4, NUM_MMA_Q, 4), stride=(NUM_MMA_Q * 4, 4, 1)),
                Uint32,
            )
            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            q_tok_stride_b = Int64(mQ.stride[0]) * Int64(2)
            q_head_off_b = Int64(q_head_idx) * Int64(mQ.stride[1]) * Int64(2)
            q_bf_tmp = cute.make_rmem_tensor(
                cute.make_layout((4, 2, 2), stride=(4, 2, 1)), Uint32
            )
            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                for row_slot in cutlass.range_constexpr(2):
                    row_b = (
                        Int64(q_start + q_local[mq, row_slot]) * q_tok_stride_b
                        + q_head_off_b
                    )
                    amax = Float32(0.0)
                    for k32 in cutlass.range_constexpr(4):
                        for half in cutlass.range_constexpr(2):
                            col = k32 * 32 + half * 16 + Int32(4) * lane_quad
                            w0, w1 = Uint32(0), Uint32(0)
                            if row_valid[mq, row_slot] != Int32(0):
                                w0 = ld_global_nc_u32(
                                    get_ptr_as_int64(
                                        mQBytes, row_b + Int64(col * Int32(2))
                                    )
                                )
                                w1 = ld_global_nc_u32(
                                    get_ptr_as_int64(
                                        mQBytes,
                                        row_b + Int64((col + Int32(2)) * Int32(2)),
                                    )
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
                    q_row_scale[mq, row_slot] = cutlass.select_(
                        nz, amax * Float32(1.0 / 448.0), Float32(1.0)
                    )
                    inv_qs = cutlass.select_(nz, Float32(448.0) / amax, Float32(0.0))
                    for k32 in cutlass.range_constexpr(4):
                        for half in cutlass.range_constexpr(2):
                            w0, w1 = q_bf_tmp[k32, half, 0], q_bf_tmp[k32, half, 1]
                            q_fp8_regs[k32, mq, row_slot + 2 * half] = (
                                cvt_f32x4_to_e4m3x4(
                                    u32_as_f32(w0 << Uint32(16)) * inv_qs,
                                    u32_as_f32(w0 & Uint32(0xFFFF0000)) * inv_qs,
                                    u32_as_f32(w1 << Uint32(16)) * inv_qs,
                                    u32_as_f32(w1 & Uint32(0xFFFF0000)) * inv_qs,
                                )
                            )

            rel_q_regs = cute.make_rmem_tensor(
                cute.make_layout(
                    (NUM_MMA_Q, 2, REL_RANK // 2),
                    stride=(2 * (REL_RANK // 2), REL_RANK // 2, 1),
                ),
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
                for mq in cutlass.range_constexpr(NUM_MMA_Q):
                    for row_slot in cutlass.range_constexpr(2):
                        rb = (
                            Int64(q_start + q_local[mq, row_slot]) * rf_tok_stride_b
                            + rf_head_off_b
                        )
                        for k2 in cutlass.range_constexpr(REL_RANK // 2):
                            v = Uint32(0)
                            if row_valid[mq, row_slot] != Int32(0):
                                v = ld_global_nc_u32(
                                    get_ptr_as_int64(mRelFacBytes, rb + Int64(k2 * 4))
                                )
                            rel_q_regs[mq, row_slot, k2] = v

            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                for mma_d in cutlass.range_constexpr(8):
                    for reg_id in cutlass.range_constexpr(8):
                        o_frag[mq, mma_d, reg_id] = Float32(0.0)
                for row_slot in cutlass.range_constexpr(2):
                    m_frag[mq, row_slot] = Float32(-Float32.inf)
                    d_frag[mq, row_slot] = Float32(1.0)
                    if const_expr(self.has_sinks):
                        if row_valid[mq, row_slot] != Int32(0):
                            raw = (
                                Float32(
                                    mSinks[q_start + q_local[mq, row_slot], q_head_idx]
                                )
                                if const_expr(self.sink_per_row)
                                else Float32(mSinks[q_head_idx])
                            )
                            sv = raw * inv_sm_scale
                            m_frag[mq, row_slot] = cutlass.select_(
                                sv == Float32.inf, -Float32.inf, sv
                            )

            frag_s_layout = cute.make_layout((NUM_MMA_Q, 2, 8), stride=(16, 8, 1))
            p_fp8_regs = cute.make_rmem_tensor(
                cute.make_layout((NUM_MMA_Q, 4), stride=(4, 1)), Uint32
            )
            warp_max_key = context_len + warp_q_base + Int32(WARP_Q_ROWS - 1)

            t = Int32(0)
            while t < ntiles:
                buf = t % Int32(S)
                use = t // Int32(S)
                cute.arch.mbarrier_wait(full_mbar + buf, phase=use & Int32(1))
                tile_base = chunk_start + t * Int32(STAGE)
                if tile_base <= warp_max_key:
                    tile_tokens = cutlass.select_(
                        tile_base + Int32(STAGE) < last_key_exclusive,
                        Int32(STAGE),
                        last_key_exclusive - tile_base,
                    )
                    sK_data = payload_base_addr + buf * Int32(self.stage_bytes)
                    sK_scale = sK_data + Int32(self.k_data_stage_bytes)
                    sV_data = sK_scale + Int32(self.k_scale_stage_bytes)
                    sV_scale = sV_data + Int32(self.v_data_stage_bytes)

                    tile_interior = (tile_tokens >= Int32(STAGE)) and (
                        tile_base + Int32(STAGE - 1) <= causal_hi_min
                    )
                    if const_expr(self.window_left >= 0):
                        tile_interior = tile_interior and (tile_base >= window_lo_max)

                    frag_S = cute.make_rmem_tensor(frag_s_layout, Float32)
                    for sub in cutlass.range_constexpr(2):
                        if Int32(sub * 16) < tile_tokens:
                            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                                for r in cutlass.range_constexpr(8):
                                    frag_S[mq, sub, r] = Float32(0.0)
                            qk_fp8mma_into_sfrag(
                                frag_S,
                                q_fp8_regs,
                                sK_data,
                                sK_scale,
                                lane,
                                Int32(sub * 16),
                                mma_kv=sub,
                                num_mma_q=NUM_MMA_Q,
                            )
                            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                                qs0, qs1 = q_row_scale[mq, 0], q_row_scale[mq, 1]
                                frag_S[mq, sub, 0] = frag_S[mq, sub, 0] * qs0
                                frag_S[mq, sub, 1] = frag_S[mq, sub, 1] * qs0
                                frag_S[mq, sub, 4] = frag_S[mq, sub, 4] * qs0
                                frag_S[mq, sub, 5] = frag_S[mq, sub, 5] * qs0
                                frag_S[mq, sub, 2] = frag_S[mq, sub, 2] * qs1
                                frag_S[mq, sub, 3] = frag_S[mq, sub, 3] * qs1
                                frag_S[mq, sub, 6] = frag_S[mq, sub, 6] * qs1
                                frag_S[mq, sub, 7] = frag_S[mq, sub, 7] * qs1
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
                                        inv_sm_scale,
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
                            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                                for r in cutlass.range_constexpr(8):
                                    frag_S[mq, sub, r] = Float32(-Float32.inf)

                    _update_mdo_pack_p_fp8_32(
                        frag_S, o_frag, m_frag, d_frag, p_fp8_regs, sm_scale_log2
                    )
                    if tile_interior:
                        pv_fp8mma_32_into_ofrag(
                            o_frag,
                            p_fp8_regs,
                            sV_data,
                            sV_scale,
                            lane,
                            tile_tokens,
                            warp_window_lo - tile_base,
                            num_mma_q=NUM_MMA_Q,
                            skip_mask=True,
                        )
                    else:
                        pv_fp8mma_32_into_ofrag(
                            o_frag,
                            p_fp8_regs,
                            sV_data,
                            sV_scale,
                            lane,
                            tile_tokens,
                            warp_window_lo - tile_base,
                            num_mma_q=NUM_MMA_Q,
                            skip_mask=False,
                        )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(empty_mbar + buf)
                t += Int32(1)

            for mq in cutlass.range_constexpr(NUM_MMA_Q):
                for row_slot in cutlass.range_constexpr(2):
                    if row_valid[mq, row_slot] != Int32(0):
                        q_row = q_start + q_local[mq, row_slot]
                        m_val = m_frag[mq, row_slot]
                        d_val = d_frag[mq, row_slot]
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
                                o_frag[mq, mma_d, row_slot * 2 + 0] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, q_head_idx, c0 + 1] = (
                                o_frag[mq, mma_d, row_slot * 2 + 1] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, q_head_idx, c8 + 0] = (
                                o_frag[mq, mma_d, row_slot * 2 + 4] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, q_head_idx, c8 + 1] = (
                                o_frag[mq, mma_d, row_slot * 2 + 5] * final_scale
                            ).to(cutlass.BFloat16)


__all__ = ["CTA_ROWS", "PagedNVFP4PrefillKernel", "REL_RANK"]
