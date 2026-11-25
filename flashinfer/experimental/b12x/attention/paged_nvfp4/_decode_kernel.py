# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 warp-specialized split-KV decode/verify attention over paged NVFP4 (D=128)."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    shared_ptr_to_u32,
    st_global_v2_f32,
)
from b12x.attention.paged.forward_paged import (
    _exit_thread,
    _literal_update_mdo_states_fp32_pack_p,
    _max_nan_f32,
)
from b12x.attention.paged_decode._kernel import (
    LOG2_E,
    SMEM_LIMIT,
    PagedDecodeMergeKernel,
    _add_relative_bias_subtile,
    _mask_subtile,
    default_max_splits,
)

from ._math import (
    kv_stage_sizes,
    make_kv_shared_storage_cls,
    pv_nvfp4_into_ofrag,
    qk_nvfp4_into_sfrag,
    run_kv_producer_loop,
)


class PagedNVFP4DecodeKernel:
    """Warp-specialized split-KV decode/verify over paged block-16 NVFP4 caches."""

    def __init__(
        self,
        *,
        group_size: int,
        max_q_per_req: int = 8,
        window_left: int = -1,
        has_sinks: bool = False,
        split_kv: bool = True,
        stage_rows: int = 16,
        num_stages: int = 12,
        num_consumer_warps: int = 8,
        min_tiles_per_split: int = 4,
        head_dim_qk: int = 128,
        head_dim_vo: int = 128,
        page_size: int = 128,
        head_splits: int = 1,
        has_relative_bias: bool = False,
        sink_layout: str = "head",
    ):
        if sink_layout not in ("head", "row"):
            raise ValueError(f"unsupported sink layout {sink_layout!r}")
        if (head_dim_qk, head_dim_vo) != (128, 128):
            raise ValueError("paged_nvfp4 decode supports 128/128 head dims")
        if stage_rows not in (16, 32) or page_size % stage_rows != 0:
            raise ValueError(f"unsupported stage_rows {stage_rows}")
        if head_splits < 1 or group_size % head_splits != 0:
            raise ValueError(f"unsupported head_splits {head_splits}")
        rows_cap = (group_size // head_splits) * max_q_per_req
        if (rows_cap + 15) // 16 > num_consumer_warps:
            raise ValueError(f"unsupported row capacity {rows_cap}")

        self.sink_per_row = sink_layout == "row"
        self.head_splits = int(head_splits)
        self.group_size = int(group_size)
        self.max_q_per_req = int(max_q_per_req)
        self.window_left = int(window_left)
        self.has_sinks = bool(has_sinks)
        self.has_relative_bias = bool(has_relative_bias)
        self.split_kv = bool(split_kv)
        self.sinks_in_kernel = self.has_sinks and not self.split_kv
        self.stage_rows = int(stage_rows)
        self.num_stages = int(num_stages)
        self.ncw = int(num_consumer_warps)
        self.num_warps = self.ncw + 1
        self.min_tiles_per_split = int(min_tiles_per_split)
        self.head_dim_qk = 128
        self.head_dim_vo = 128
        self.page_size = int(page_size)
        self.num_mma_d_qk = 8
        self.num_mma_d_vo = 8

        (
            self.k_data_stage_bytes,
            self.k_scale_stage_bytes,
            self.v_data_stage_bytes,
            self.v_scale_stage_bytes,
            self.stage_bytes,
        ) = kv_stage_sizes(self.stage_rows, self.head_dim_qk, self.head_dim_vo)
        ring_bytes = self.num_stages * self.stage_bytes
        combine_bytes = (
            self.ncw * self.num_mma_d_vo * 8 * 32 + 2 * self.ncw * 2 * 32
        ) * 4
        self.shared_storage_bytes = max(ring_bytes, combine_bytes)

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
        mOPart: cute.Tensor,  # [T, Hq, max_splits, 128] f32
        mLsePart: cute.Tensor,  # [T, Hq, max_splits] f32
        mSinks: cute.Tensor,  # [Hq] or [T, Hq] f32
        mRelBias: cute.Tensor,  # [T, Hq, extent] bf16
        softmax_scale: Float32,
        stream: cuda.CUstream,
    ):
        batch = mSeqLens.shape[0]
        num_kv_heads = mKCache.shape[2]
        grid_z = (
            mLsePart.shape[2] * self.head_splits if self.split_kv else self.head_splits
        )
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
            mOPart,
            mLsePart,
            mSinks,
            mRelBias,
            sm_scale_log2,
            inv_sm_scale,
        ).launch(
            grid=(batch, num_kv_heads, grid_z),
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
        mOPart: cute.Tensor,
        mLsePart: cute.Tensor,
        mSinks: cute.Tensor,
        mRelBias: cute.Tensor,
        sm_scale_log2: Float32,
        inv_sm_scale: Float32,
    ):
        STAGE = self.stage_rows
        S = self.num_stages
        NCW = self.ncw
        lane, warp, _ = cute.arch.thread_idx()
        request_idx, kv_head_idx, split_idx = cute.arch.block_idx()
        head_slice = split_idx
        num_splits = 1
        if const_expr(self.split_kv):
            _, _, num_splits = cute.arch.grid_dim()
            if const_expr(self.head_splits > 1):
                num_splits = num_splits // Int32(self.head_splits)
                head_slice = split_idx % Int32(self.head_splits)
                split_idx = split_idx // Int32(self.head_splits)

        q_start = mCuSeqlensQ[request_idx]
        qo_len = mCuSeqlensQ[request_idx + 1] - q_start
        seq_len = mSeqLens[request_idx]
        if qo_len <= Int32(0) or seq_len <= Int32(0):
            _exit_thread()
        context_len = seq_len - qo_len

        group_size = Int32(self.group_size)
        sub_group = Int32(self.group_size // self.head_splits)
        head_base = kv_head_idx * group_size
        if const_expr(self.head_splits > 1):
            head_base = head_base + head_slice * sub_group
        packed_rows = qo_len * sub_group

        first_key = Int32(0)
        if const_expr(self.window_left >= 0):
            first_key = cutlass.select_(
                context_len - Int32(self.window_left) > Int32(0),
                context_len - Int32(self.window_left),
                Int32(0),
            )
        tile_begin = first_key // STAGE
        tile_end = (seq_len + (STAGE - 1)) // STAGE
        my_tile_lo, my_tile_hi = tile_begin, tile_end
        if const_expr(self.split_kv):
            num_tiles = tile_end - tile_begin
            tiles_per_split = (num_tiles + num_splits - 1) // num_splits
            tiles_per_split = cutlass.select_(
                tiles_per_split < Int32(self.min_tiles_per_split),
                Int32(self.min_tiles_per_split),
                tiles_per_split,
            )
            my_tile_lo = tile_begin + split_idx * tiles_per_split
            my_tile_hi = cutlass.select_(
                my_tile_lo + tiles_per_split < tile_end,
                my_tile_lo + tiles_per_split,
                tile_end,
            )
            if my_tile_lo >= tile_end:
                _exit_thread()
        chunk_start = my_tile_lo * STAGE
        chunk_end = cutlass.select_(
            my_tile_hi * STAGE < seq_len, my_tile_hi * STAGE, seq_len
        )
        ntiles = my_tile_hi - my_tile_lo
        num_row_groups = (packed_rows + 15) // 16
        num_key_groups = Int32(NCW) // num_row_groups
        if num_key_groups > Int32(S):
            num_key_groups = Int32(S)
        stages_per_group = Int32(S) // num_key_groups
        tidx = lane + warp * Int32(32)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        full_mbar = storage.full.data_ptr()
        empty_mbar = storage.empty.data_ptr()
        if tidx < Int32(S):
            cute.arch.mbarrier_init(full_mbar + tidx, Int32(32))
            cute.arch.mbarrier_init(empty_mbar + tidx, num_row_groups)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        payload_u8 = storage.payload.get_tensor(
            cute.make_layout((self.shared_storage_bytes,), stride=(1,))
        )
        payload_base_addr = shared_ptr_to_u32(payload_u8.iterator)

        is_producer = warp == Int32(NCW)
        is_consumer = warp < num_row_groups * num_key_groups
        row_group = warp % num_row_groups
        key_group = warp // num_row_groups
        lane_group = lane // Int32(4)
        lane_quad = lane % Int32(4)
        lane_pair_base = Int32(2) * lane_quad

        o_frag = cute.make_rmem_tensor(
            cute.make_layout((1, 8, 8), stride=(64, 8, 1)), Float32
        )
        m_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        d_frag = cute.make_rmem_tensor(cute.make_layout((1, 2), stride=(2, 1)), Float32)
        row_valid = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_token = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
        q_head = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)

        if is_producer:
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
                num_key_groups,
                stages_per_group,
                payload_base_addr,
                full_mbar,
                empty_mbar,
                self.page_size,
                STAGE,
                128,
                128,
            )
        elif is_consumer:
            causal_hi = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Int32
            )
            window_lo = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Int32
            )
            warp_row_base = row_group * Int32(16)

            for row_slot in cutlass.range_constexpr(2):
                packed_row = warp_row_base + lane_group + Int32(8 * row_slot)
                valid_row = packed_row < packed_rows
                row_valid[row_slot] = Int32(valid_row)
                token_local = packed_row // sub_group
                q_token[row_slot] = token_local
                q_head[row_slot] = head_base + (packed_row - token_local * sub_group)
                causal_hi[row_slot] = cutlass.select_(
                    valid_row, context_len + token_local, Int32(-1)
                )
                window_lo[row_slot] = (
                    context_len + token_local - Int32(self.window_left)
                )

            warp_window_lo = (
                (context_len + warp_row_base // sub_group - Int32(self.window_left))
                if const_expr(self.window_left >= 0)
                else Int32(0)
            )

            row_pos = cute.make_rmem_tensor(cute.make_layout((2,), stride=(1,)), Int32)
            bias_row_byte = cute.make_rmem_tensor(
                cute.make_layout((2,), stride=(1,)), Int64
            )
            bias_extent = Int32(0)
            bias_key_lo = Int32(0)
            bias_key_hi = Int32(-1)
            if const_expr(self.has_relative_bias):
                bias_extent = Int32(mRelBias.shape[2])
                bias_key_lo = context_len - bias_extent + Int32(1)
                bias_key_hi = context_len + qo_len - Int32(1)
                for row_slot in cutlass.range_constexpr(2):
                    row_pos[row_slot] = context_len + q_token[row_slot]
                    bias_row_byte[row_slot] = (
                        Int64(q_start + q_token[row_slot]) * Int64(mRelBias.stride[0])
                        + Int64(q_head[row_slot]) * Int64(mRelBias.stride[1])
                    ) * Int64(2)
            mRelBiasBytes = cute.flatten(cute.recast_tensor(mRelBias, cutlass.Uint8))

            q_regs = cute.make_rmem_tensor(
                cute.make_layout((8, 1, 4), stride=(4, 4, 1)), Uint32
            )
            mQBytes = cute.flatten(cute.recast_tensor(mQ, cutlass.Uint8))
            q_token_stride_bytes = Int64(mQ.stride[0]) * Int64(2)
            q_head_stride_bytes = Int64(mQ.stride[1]) * Int64(2)
            for row_slot in cutlass.range_constexpr(2):
                row_byte = (
                    Int64(q_start + q_token[row_slot]) * q_token_stride_bytes
                    + Int64(q_head[row_slot]) * q_head_stride_bytes
                )
                for mma_d in cutlass.range_constexpr(8):
                    for half in cutlass.range_constexpr(2):
                        col = mma_d * 16 + half * 8 + Int32(2) * lane_quad
                        value = Uint32(0)
                        if row_valid[row_slot] != Int32(0):
                            value = ld_global_nc_u32(
                                get_ptr_as_int64(
                                    mQBytes, row_byte + Int64(col * Int32(2))
                                )
                            )
                        q_regs[mma_d, 0, row_slot + 2 * half] = value

            for mma_d in cutlass.range_constexpr(8):
                for reg_id in cutlass.range_constexpr(8):
                    o_frag[0, mma_d, reg_id] = Float32(0.0)
            for row_slot in cutlass.range_constexpr(2):
                m_frag[0, row_slot] = Float32(-Float32.inf)
                d_frag[0, row_slot] = Float32(1.0)
                if const_expr(self.sinks_in_kernel):
                    if row_valid[row_slot] != Int32(0) and key_group == Int32(0):
                        raw = (
                            Float32(
                                mSinks[q_start + q_token[row_slot], q_head[row_slot]]
                            )
                            if const_expr(self.sink_per_row)
                            else Float32(mSinks[q_head[row_slot]])
                        )
                        sv = raw * inv_sm_scale
                        m_frag[0, row_slot] = cutlass.select_(
                            sv == Float32.inf, -Float32.inf, sv
                        )

            frag_s_layout = cute.make_layout((1, 1, 8), stride=(8, 8, 1))
            p_frag = cute.make_rmem_tensor(
                cute.make_layout((1, 1, 4), stride=(4, 4, 1)), Uint32
            )
            causal_hi_min = context_len
            window_lo_max = context_len + qo_len - Int32(1) - Int32(self.window_left)

            t = key_group
            j = Int32(0)
            while t < ntiles:
                buf = key_group * stages_per_group + j % stages_per_group
                use = j // stages_per_group
                cute.arch.mbarrier_wait(full_mbar + buf, phase=use & Int32(1))
                tile_base = chunk_start + t * Int32(STAGE)
                tile_tokens = cutlass.select_(
                    tile_base + STAGE < chunk_end, Int32(STAGE), chunk_end - tile_base
                )
                sK_data = payload_base_addr + buf * Int32(self.stage_bytes)
                sK_scale = sK_data + Int32(self.k_data_stage_bytes)
                sV_data = sK_scale + Int32(self.k_scale_stage_bytes)
                sV_scale = sV_data + Int32(self.v_data_stage_bytes)

                for sub in cutlass.range_constexpr(STAGE // 16):
                    if Int32(sub * 16) < tile_tokens:
                        frag_S = cute.make_rmem_tensor(frag_s_layout, Float32)
                        frag_S.fill(0.0)
                        qk_nvfp4_into_sfrag(
                            frag_S, q_regs, sK_data, sK_scale, lane, Int32(sub * 16)
                        )
                        sub_base = tile_base + Int32(sub * 16)
                        if const_expr(self.has_relative_bias):
                            if (
                                sub_base <= bias_key_hi
                                and sub_base + Int32(15) >= bias_key_lo
                            ):
                                _add_relative_bias_subtile(
                                    frag_S,
                                    mRelBiasBytes,
                                    bias_row_byte,
                                    row_pos,
                                    row_valid,
                                    sub_base,
                                    lane_pair_base,
                                    bias_extent,
                                    inv_sm_scale,
                                )
                        interior = (tile_tokens >= Int32(sub * 16 + 16)) and (
                            sub_base + Int32(15) <= causal_hi_min
                        )
                        if const_expr(self.window_left >= 0):
                            interior = interior and sub_base >= window_lo_max
                        if not interior:
                            _mask_subtile(
                                frag_S,
                                Int32(sub * 16),
                                tile_base,
                                tile_tokens,
                                lane_pair_base,
                                causal_hi,
                                window_lo,
                                self.window_left >= 0,
                            )
                        _literal_update_mdo_states_fp32_pack_p(
                            frag_S,
                            o_frag,
                            m_frag,
                            d_frag,
                            p_frag,
                            sm_scale_log2,
                            1,
                            1,
                            8,
                        )
                        if interior:
                            pv_nvfp4_into_ofrag(
                                o_frag,
                                p_frag,
                                sV_data,
                                sV_scale,
                                lane,
                                Int32(sub * 16),
                                tile_tokens,
                                warp_window_lo - tile_base,
                                skip_mask=True,
                            )
                        else:
                            pv_nvfp4_into_ofrag(
                                o_frag,
                                p_frag,
                                sV_data,
                                sV_scale,
                                lane,
                                Int32(sub * 16),
                                tile_tokens,
                                warp_window_lo - tile_base,
                                skip_mask=False,
                            )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(empty_mbar + buf)
                t += num_key_groups
                j += Int32(1)

        # Combine consumer key-group partials through shared memory (reusing payload).
        cute.arch.sync_threads()
        o_words = 8 * 8 * 32
        comb_o_floats = NCW * o_words
        comb_md_floats = NCW * 2 * 32
        sComb = cute.make_tensor(
            cute.recast_ptr(payload_u8.iterator.align(16), dtype=Float32),
            cute.make_layout((comb_o_floats + 2 * comb_md_floats,), stride=(1,)),
        )
        if is_consumer and num_key_groups > Int32(1):
            for mma_d in cutlass.range_constexpr(8):
                for reg_id in cutlass.range_constexpr(8):
                    sComb[warp * o_words + (mma_d * 8 + reg_id) * 32 + lane] = o_frag[
                        0, mma_d, reg_id
                    ]
            for row_slot in cutlass.range_constexpr(2):
                sComb[comb_o_floats + (warp * 2 + row_slot) * 32 + lane] = m_frag[
                    0, row_slot
                ]
                sComb[
                    comb_o_floats + comb_md_floats + (warp * 2 + row_slot) * 32 + lane
                ] = d_frag[0, row_slot]
        cute.arch.sync_threads()

        if is_consumer and key_group == Int32(0):
            if num_key_groups > Int32(1):
                for row_slot in cutlass.range_constexpr(2):
                    m_max = m_frag[0, row_slot]
                    kg = Int32(1)
                    while kg < num_key_groups:
                        w_other = kg * num_row_groups + row_group
                        m_max = _max_nan_f32(
                            m_max,
                            sComb[comb_o_floats + (w_other * 2 + row_slot) * 32 + lane],
                        )
                        kg += Int32(1)
                    scale0 = Float32(0.0)
                    if m_max != -Float32.inf and m_frag[0, row_slot] != -Float32.inf:
                        scale0 = cute.math.exp2(
                            (m_frag[0, row_slot] - m_max) * sm_scale_log2, fastmath=True
                        )
                    d_sum = d_frag[0, row_slot] * scale0
                    for mma_d in cutlass.range_constexpr(8):
                        for r_off in cutlass.range_constexpr(4):
                            reg_id = (r_off // 2) * 4 + row_slot * 2 + (r_off % 2)
                            o_frag[0, mma_d, reg_id] = o_frag[0, mma_d, reg_id] * scale0
                    kg = Int32(1)
                    while kg < num_key_groups:
                        w_other = kg * num_row_groups + row_group
                        m_other = sComb[
                            comb_o_floats + (w_other * 2 + row_slot) * 32 + lane
                        ]
                        if m_max != -Float32.inf and m_other != -Float32.inf:
                            sc = cute.math.exp2(
                                (m_other - m_max) * sm_scale_log2, fastmath=True
                            )
                            d_sum = (
                                d_sum
                                + sComb[
                                    comb_o_floats
                                    + comb_md_floats
                                    + (w_other * 2 + row_slot) * 32
                                    + lane
                                ]
                                * sc
                            )
                            for mma_d in cutlass.range_constexpr(8):
                                for r_off in cutlass.range_constexpr(4):
                                    reg_id = (
                                        (r_off // 2) * 4 + row_slot * 2 + (r_off % 2)
                                    )
                                    o_frag[0, mma_d, reg_id] = (
                                        o_frag[0, mma_d, reg_id]
                                        + sComb[
                                            w_other * o_words
                                            + (mma_d * 8 + reg_id) * 32
                                            + lane
                                        ]
                                        * sc
                                    )
                        kg += Int32(1)
                    m_frag[0, row_slot] = m_max
                    d_frag[0, row_slot] = d_sum

            for row_slot in cutlass.range_constexpr(2):
                if row_valid[row_slot] != Int32(0):
                    q_row = q_start + q_token[row_slot]
                    head = q_head[row_slot]
                    m_val = m_frag[0, row_slot]
                    d_val = d_frag[0, row_slot]
                    inv_d = Float32(0.0)
                    if d_val > Float32(0.0) and m_val != -Float32.inf:
                        inv_d = Float32(1.0) / d_val
                    if const_expr(self.split_kv):
                        if lane_quad == Int32(0):
                            lse = Float32(-Float32.inf)
                            if d_val > Float32(0.0) and m_val != -Float32.inf:
                                lse = m_val * sm_scale_log2 + cute.math.log2(
                                    d_val, fastmath=True
                                )
                            mLsePart[q_row, head, split_idx] = lse
                        mOPartBytes = cute.flatten(
                            cute.recast_tensor(mOPart, cutlass.Uint8)
                        )
                        base_b = (
                            Int64(q_row) * Int64(mOPart.stride[0])
                            + Int64(head) * Int64(mOPart.stride[1])
                            + Int64(split_idx) * Int64(mOPart.stride[2])
                        ) * Int64(4)
                        for mma_d in cutlass.range_constexpr(8):
                            c0 = mma_d * 16 + Int32(2) * lane_quad
                            c8 = c0 + Int32(8)
                            st_global_v2_f32(
                                get_ptr_as_int64(
                                    mOPartBytes, base_b + Int64(c0 * Int32(4))
                                ),
                                o_frag[0, mma_d, row_slot * 2 + 0] * inv_d,
                                o_frag[0, mma_d, row_slot * 2 + 1] * inv_d,
                            )
                            st_global_v2_f32(
                                get_ptr_as_int64(
                                    mOPartBytes, base_b + Int64(c8 * Int32(4))
                                ),
                                o_frag[0, mma_d, row_slot * 2 + 4] * inv_d,
                                o_frag[0, mma_d, row_slot * 2 + 5] * inv_d,
                            )
                    else:
                        positive_sink = Int32(0)
                        if const_expr(self.sinks_in_kernel):
                            raw_sink = (
                                Float32(mSinks[q_row, head])
                                if const_expr(self.sink_per_row)
                                else Float32(mSinks[head])
                            )
                            positive_sink = Int32(raw_sink == Float32.inf)
                        final_scale = cutlass.select_(
                            positive_sink != Int32(0), Float32(0.0), inv_d
                        )
                        for mma_d in cutlass.range_constexpr(8):
                            c0 = mma_d * 16 + Int32(2) * lane_quad
                            c8 = c0 + Int32(8)
                            mO[q_row, head, c0 + 0] = (
                                o_frag[0, mma_d, row_slot * 2 + 0] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, head, c0 + 1] = (
                                o_frag[0, mma_d, row_slot * 2 + 1] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, head, c8 + 0] = (
                                o_frag[0, mma_d, row_slot * 2 + 4] * final_scale
                            ).to(cutlass.BFloat16)
                            mO[q_row, head, c8 + 1] = (
                                o_frag[0, mma_d, row_slot * 2 + 5] * final_scale
                            ).to(cutlass.BFloat16)


__all__ = [
    "PagedDecodeMergeKernel",
    "PagedNVFP4DecodeKernel",
    "SMEM_LIMIT",
    "default_max_splits",
]
