# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 CuTe-DSL NVFP4 decode MoE kernels with fused SiTU and inline CSF.

Key design properties:
  - One CTA owns one exact 16-row CSF slab of one active expert (8 interleaved
    ``(gate, up)`` pairs for FC1; 16 output channels for FC2).
  - When ``has_csf=True``, scales are decoded directly from ``fixed`` +
    pre-partitioned ``slab_exceptions`` into 1.15-2.30 KiB of shared memory
    (zero HBM scratch write/read, zero extra kernel launches, half scale traffic).
  - Weights are read ONCE from HBM per active expert and reused across all
    tokens routed to that expert (1..M) in registers, using coalesced 128B/64B
    ``ld.global.nc.v2.u32`` loads and SM120 ``cvt.rn.f16x2.e2m1x2`` +
    ``fma.rn.f16x2`` dot products.
  - FC1 fuses interleaved ``(gate, up)`` SiTU activation + route-weight multiply
    in-warp and writes BF16 ``inter[route, I]`` directly (zero ``[R, 2*I]`` HBM).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.conversion import fp4_decode_4bytes
from b12x._lib.intrinsics.dot import fp4_dot8_sum
from b12x._lib.intrinsics.memory import get_ptr_as_int64, shared_ptr_to_u32

from ._math import (
    e4m3_byte_to_f32,
    expand_csf_slab_to_smem,
    ld_global_nc_v2_u32,
    ld_shared_u8,
    load_dense_slab_to_smem,
    rcp_rn_f32,
    shfl_idx_f32,
    shfl_xor_f32,
    situ_activate_f32,
)


@cute.jit
def _stage_nvfp4_act_to_f16x2_smem(
    mAQ: cute.Tensor,  # [M_act, K//2] u8
    mAS: cute.Tensor,  # [M_act, K//16] u8
    act_row: Int32,
    smem_a_f16_addr: Int32,  # [K//2] u32 (each u32 = f16x2)
    smem_as_f32_addr: Int32,  # [K//16] f32
    tid: Int32,
    threads: cutlass.Constexpr,
    k_dim: cutlass.Constexpr,
):
    """Pre-decode one activation row's E2M1x2 -> f16x2 and E4M3 scales -> f32 into SMEM.

    All 16 weight rows in the CTA then reuse these SMEM activations with zero
    per-row activation decoding inside the K loop.
    """
    k_blocks: cutlass.Constexpr = k_dim // 16
    row_q_ptr = get_ptr_as_int64(mAQ, Int64(act_row) * Int64(k_dim // 2))
    iters: cutlass.Constexpr = (k_blocks + threads - 1) // threads
    for it in cutlass.range_constexpr(iters):
        blk = tid + Int32(it * threads)
        if blk < Int32(k_blocks):
            q_lo, q_hi = ld_global_nc_v2_u32(row_q_ptr + Int64(blk) * Int64(8))
            h0, h1, h2, h3 = fp4_decode_4bytes(q_lo)
            h4, h5, h6, h7 = fp4_decode_4bytes(q_hi)
            dst = smem_a_f16_addr + blk * Int32(32)
            st_shared_u32(dst + Int32(0), h0)
            st_shared_u32(dst + Int32(4), h1)
            st_shared_u32(dst + Int32(8), h2)
            st_shared_u32(dst + Int32(12), h3)
            st_shared_u32(dst + Int32(16), h4)
            st_shared_u32(dst + Int32(20), h5)
            st_shared_u32(dst + Int32(24), h6)
            st_shared_u32(dst + Int32(28), h7)
            s_u8 = mAS[act_row, blk].to(Uint32)
            st_shared_u32(smem_as_f32_addr + blk * Int32(4), s_u8)


class SituDecodeFC1NVFP4Kernel:
    """Warp-specialized NVFP4 FC1 + fused interleaved SiTU for decode (M <= 16).

    Grid ``(2*I // 16, max_slots, 1)``, block ``(256, 1, 1)`` (8 warps = 16
    half-warps = 1 exact 16-row CSF slab = 8 interleaved ``(gate, up)`` pairs).
    """

    def __init__(
        self,
        *,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        max_tokens: int,
        has_csf: bool,
    ):
        if hidden % 256 != 0 or intermediate % 8 != 0 or (2 * intermediate) % 16 != 0:
            raise ValueError(
                "FC1 decode requires hidden % 256 == 0 and intermediate % 8 == 0"
            )
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.n_rows = 2 * self.intermediate
        self.num_slabs = self.n_rows // 16
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        self.has_csf = bool(has_csf)
        self.k_blocks = self.hidden // 16
        self.k_iters = self.k_blocks // 16  # 16 lanes/row -> 128B coalesced line
        self.threads = 256
        # Tile up to 4 active tokens per expert in one HBM weight pass (covers >99% of C=1..16).
        self.token_chunk = min(4, self.max_tokens)

        self.scale_bytes = 16 * self.k_blocks
        self.act_f16_bytes_per_tok = self.hidden * 2  # K * 2B
        self.act_scale_bytes_per_tok = self.k_blocks * 4  # stored as u32 E4M3 byte
        self.act_stride_bytes = (
            self.act_f16_bytes_per_tok + self.act_scale_bytes_per_tok
        )
        self.smem_bytes = (
            (self.scale_bytes + 15) // 16
        ) * 16 + self.token_chunk * self.act_stride_bytes

    def _get_shared_storage_cls(self):
        nbytes = self.smem_bytes

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, nbytes], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mAQ: cute.Tensor,  # [M, H//2] u8
        mAS: cute.Tensor,  # [M, H//16] u8
        mAOuter: cute.Tensor,  # [M] f32
        mW13: cute.Tensor,  # flat u8 view of [E, 2*I, H//2]
        mS13: cute.Tensor,  # flat u8 view of dense [E, 2*I, H//16] OR CSF fixed
        mExc: cute.Tensor,  # [num_exc] u32
        mSlabOffs: cute.Tensor,  # [E * num_slabs + 1] i32
        mG13: cute.Tensor,  # [2 * E] f32
        mWeights: cute.Tensor,  # [R] f32
        mActiveExperts: cute.Tensor,  # [max_slots] i32
        mActiveCounts: cute.Tensor,  # [max_slots] i32
        mActiveRoutes: cute.Tensor,  # [max_slots, R] i32
        mInter: cute.Tensor,  # [R, I] bf16
        stream: cuda.CUstream,
    ):
        max_slots = mActiveExperts.shape[0]
        self.kernel(
            mAQ,
            mAS,
            mAOuter,
            mW13,
            mS13,
            mExc,
            mSlabOffs,
            mG13,
            mWeights,
            mActiveExperts,
            mActiveCounts,
            mActiveRoutes,
            mInter,
        ).launch(
            grid=(self.num_slabs, max_slots, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAQ: cute.Tensor,
        mAS: cute.Tensor,
        mAOuter: cute.Tensor,
        mW13: cute.Tensor,
        mS13: cute.Tensor,
        mExc: cute.Tensor,
        mSlabOffs: cute.Tensor,
        mG13: cute.Tensor,
        mWeights: cute.Tensor,
        mActiveExperts: cute.Tensor,
        mActiveCounts: cute.Tensor,
        mActiveRoutes: cute.Tensor,
        mInter: cute.Tensor,
    ):
        slab = Int32(cute.arch.block_idx()[0])
        slot = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        expert = mActiveExperts[slot]
        count = mActiveCounts[slot]

        if expert >= Int32(0) and count > Int32(0):
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            smem_base = shared_ptr_to_u32(storage.payload.data_ptr())
            scales_base = smem_base
            act_base = smem_base + Int32(((self.scale_bytes + 15) // 16) * 16)

            if const_expr(self.has_csf):
                expand_csf_slab_to_smem(
                    mS13,
                    mExc,
                    mSlabOffs,
                    scales_base,
                    expert,
                    slab,
                    tid,
                    self.threads,
                    self.num_slabs,
                    self.k_blocks,
                )
            else:
                load_dense_slab_to_smem(
                    mS13,
                    scales_base,
                    expert,
                    slab,
                    tid,
                    self.threads,
                    self.n_rows,
                    self.k_blocks,
                )

            hw = tid // Int32(16)  # 0..15 (row within slab)
            lane16 = tid % Int32(16)  # 0..15 (coalesced K-block lane)
            warp = tid // Int32(32)  # 0..7 (interleaved (gate, up) pair)
            lane = tid % Int32(32)
            row_in_expert = slab * Int32(16) + hw
            w_row_ptr = get_ptr_as_int64(
                mW13,
                (Int64(expert) * Int64(self.n_rows) + Int64(row_in_expert))
                * Int64(self.hidden // 2),
            )
            g_ratio = mG13[expert] * rcp_rn_f32(
                Float32(1536.0) * mG13[Int32(self.num_experts) + expert]
            )

            # Process routed tokens for this expert in chunks of self.token_chunk (reusing weights in regs!).
            t_base = Int32(0)
            while t_base < count:
                chunk_n = cutlass.min(count - t_base, Int32(self.token_chunk))

                # Stage activations for this chunk into SMEM once.
                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                        tok_c = route_c // Int32(self.top_k)
                        a_dst = act_base + Int32(ci * self.act_stride_bytes)
                        _stage_nvfp4_act_to_f16x2_smem(
                            mAQ,
                            mAS,
                            tok_c,
                            a_dst,
                            a_dst + Int32(self.act_f16_bytes_per_tok),
                            tid,
                            self.threads,
                            self.hidden,
                        )
                cute.arch.barrier()

                acc = cute.make_rmem_tensor(
                    cute.make_layout((self.token_chunk,)), Float32
                )
                for ci in cutlass.range_constexpr(self.token_chunk):
                    acc[ci] = Float32(0.0)

                # Single coalesced weight pass over HBM, reused across all chunk_n tokens.
                for ki in cutlass.range_constexpr(self.k_iters):
                    g = Int32(ki * 16) + lane16
                    w_lo, w_hi = ld_global_nc_v2_u32(w_row_ptr + Int64(g) * Int64(8))
                    sw_f = e4m3_byte_to_f32(
                        ld_shared_u8(scales_base + hw * Int32(self.k_blocks) + g)
                    )

                    for ci in cutlass.range_constexpr(self.token_chunk):
                        if Int32(ci) < chunk_n:
                            a_src = (
                                act_base
                                + Int32(ci * self.act_stride_bytes)
                                + g * Int32(32)
                            )
                            x0 = ld_shared_u32(a_src + Int32(0))
                            x1 = ld_shared_u32(a_src + Int32(4))
                            x2 = ld_shared_u32(a_src + Int32(8))
                            x3 = ld_shared_u32(a_src + Int32(12))
                            x4 = ld_shared_u32(a_src + Int32(16))
                            x5 = ld_shared_u32(a_src + Int32(20))
                            x6 = ld_shared_u32(a_src + Int32(24))
                            x7 = ld_shared_u32(a_src + Int32(28))
                            sa_u8 = ld_shared_u32(
                                act_base
                                + Int32(
                                    ci * self.act_stride_bytes
                                    + self.act_f16_bytes_per_tok
                                )
                                + g * Int32(4)
                            )
                            sa_f = e4m3_byte_to_f32(sa_u8)
                            dot = fp4_dot8_sum(
                                w_lo, w_hi, x0, x1, x2, x3, x4, x5, x6, x7
                            )
                            acc[ci] = acc[ci] + dot * (sa_f * sw_f)

                # Half-warp butterfly reduce -> fused SiTU + route-weight -> store BF16 inter[route, col].
                out_col = slab * Int32(8) + warp
                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        val = acc[ci]
                        val = val + shfl_xor_f32(val, 8)
                        val = val + shfl_xor_f32(val, 4)
                        val = val + shfl_xor_f32(val, 2)
                        val = val + shfl_xor_f32(val, 1)
                        route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                        tok_c = route_c // Int32(self.top_k)
                        scaled = val * (mAOuter[tok_c] * g_ratio)
                        # Lanes 0..15 hold gate, lanes 16..31 hold up -> exchange to lane 0!
                        up_val = shfl_idx_f32(scaled, 16)
                        if lane == Int32(0):
                            rw = mWeights[route_c]
                            act_val = situ_activate_f32(scaled, up_val, rw)
                            mInter[route_c, out_col] = act_val.to(cutlass.BFloat16)

                cute.arch.barrier()
                t_base = t_base + Int32(self.token_chunk)


class SituDecodeFC2NVFP4Kernel:
    """Warp-specialized NVFP4 FC2 with inline CSF for decode (M <= 16).

    Grid ``(H // 16, max_slots, 1)``, block ``(128, 1, 1)`` (16 8-lane groups =
    1 exact 16-row CSF slab; 8 lanes/row x 9 blocks/lane = 72 blocks for I=1152).
    """

    def __init__(
        self,
        *,
        hidden: int,
        intermediate: int,
        num_experts: int,
        max_tokens: int,
        has_csf: bool,
    ):
        if hidden % 16 != 0 or intermediate % 128 != 0:
            raise ValueError(
                "FC2 decode requires hidden % 16 == 0 and intermediate % 128 == 0"
            )
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.num_slabs = self.hidden // 16
        self.num_experts = int(num_experts)
        self.max_tokens = int(max_tokens)
        self.has_csf = bool(has_csf)
        self.k_blocks = self.intermediate // 16
        self.k_iters = self.k_blocks // 8  # 8 lanes/row -> 64B coalesced line
        self.threads = 128
        self.token_chunk = min(4, self.max_tokens)

        self.scale_bytes = 16 * self.k_blocks
        self.act_f16_bytes_per_tok = self.intermediate * 2
        self.act_scale_bytes_per_tok = self.k_blocks * 4
        self.act_stride_bytes = (
            self.act_f16_bytes_per_tok + self.act_scale_bytes_per_tok
        )
        self.smem_bytes = (
            (self.scale_bytes + 15) // 16
        ) * 16 + self.token_chunk * self.act_stride_bytes

    def _get_shared_storage_cls(self):
        nbytes = self.smem_bytes

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "payload": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, nbytes], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mAQ: cute.Tensor,  # [R, I//2] u8
        mAS: cute.Tensor,  # [R, I//16] u8
        mAOuter: cute.Tensor,  # [R] f32
        mW2: cute.Tensor,  # flat u8 view of [E, H, I//2]
        mS2: cute.Tensor,  # flat u8 view of dense [E, H, I//16] OR CSF fixed
        mExc: cute.Tensor,  # [num_exc] u32
        mSlabOffs: cute.Tensor,  # [E * num_slabs + 1] i32
        mG2: cute.Tensor,  # [2 * E] f32
        mActiveExperts: cute.Tensor,  # [max_slots] i32
        mActiveCounts: cute.Tensor,  # [max_slots] i32
        mActiveRoutes: cute.Tensor,  # [max_slots, R] i32
        mRouteOut: cute.Tensor,  # [R, H] bf16
        stream: cuda.CUstream,
    ):
        max_slots = mActiveExperts.shape[0]
        self.kernel(
            mAQ,
            mAS,
            mAOuter,
            mW2,
            mS2,
            mExc,
            mSlabOffs,
            mG2,
            mActiveExperts,
            mActiveCounts,
            mActiveRoutes,
            mRouteOut,
        ).launch(
            grid=(self.num_slabs, max_slots, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAQ: cute.Tensor,
        mAS: cute.Tensor,
        mAOuter: cute.Tensor,
        mW2: cute.Tensor,
        mS2: cute.Tensor,
        mExc: cute.Tensor,
        mSlabOffs: cute.Tensor,
        mG2: cute.Tensor,
        mActiveExperts: cute.Tensor,
        mActiveCounts: cute.Tensor,
        mActiveRoutes: cute.Tensor,
        mRouteOut: cute.Tensor,
    ):
        slab = Int32(cute.arch.block_idx()[0])
        slot = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        expert = mActiveExperts[slot]
        count = mActiveCounts[slot]

        if expert >= Int32(0) and count > Int32(0):
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            smem_base = shared_ptr_to_u32(storage.payload.data_ptr())
            scales_base = smem_base
            act_base = smem_base + Int32(((self.scale_bytes + 15) // 16) * 16)

            if const_expr(self.has_csf):
                expand_csf_slab_to_smem(
                    mS2,
                    mExc,
                    mSlabOffs,
                    scales_base,
                    expert,
                    slab,
                    tid,
                    self.threads,
                    self.num_slabs,
                    self.k_blocks,
                )
            else:
                load_dense_slab_to_smem(
                    mS2,
                    scales_base,
                    expert,
                    slab,
                    tid,
                    self.threads,
                    self.hidden,
                    self.k_blocks,
                )

            qw = tid // Int32(8)  # 0..15 (row within slab)
            lane8 = tid % Int32(8)  # 0..7 (coalesced K-block lane)
            out_row = slab * Int32(16) + qw
            w_row_ptr = get_ptr_as_int64(
                mW2,
                (Int64(expert) * Int64(self.hidden) + Int64(out_row))
                * Int64(self.intermediate // 2),
            )
            g_ratio = mG2[expert] * rcp_rn_f32(
                Float32(1536.0) * mG2[Int32(self.num_experts) + expert]
            )

            t_base = Int32(0)
            while t_base < count:
                chunk_n = cutlass.min(count - t_base, Int32(self.token_chunk))

                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                        a_dst = act_base + Int32(ci * self.act_stride_bytes)
                        _stage_nvfp4_act_to_f16x2_smem(
                            mAQ,
                            mAS,
                            route_c,
                            a_dst,
                            a_dst + Int32(self.act_f16_bytes_per_tok),
                            tid,
                            self.threads,
                            self.intermediate,
                        )
                cute.arch.barrier()

                acc = cute.make_rmem_tensor(
                    cute.make_layout((self.token_chunk,)), Float32
                )
                for ci in cutlass.range_constexpr(self.token_chunk):
                    acc[ci] = Float32(0.0)

                for ki in cutlass.range_constexpr(self.k_iters):
                    g = Int32(ki * 8) + lane8
                    w_lo, w_hi = ld_global_nc_v2_u32(w_row_ptr + Int64(g) * Int64(8))
                    sw_f = e4m3_byte_to_f32(
                        ld_shared_u8(scales_base + qw * Int32(self.k_blocks) + g)
                    )

                    for ci in cutlass.range_constexpr(self.token_chunk):
                        if Int32(ci) < chunk_n:
                            a_src = (
                                act_base
                                + Int32(ci * self.act_stride_bytes)
                                + g * Int32(32)
                            )
                            x0 = ld_shared_u32(a_src + Int32(0))
                            x1 = ld_shared_u32(a_src + Int32(4))
                            x2 = ld_shared_u32(a_src + Int32(8))
                            x3 = ld_shared_u32(a_src + Int32(12))
                            x4 = ld_shared_u32(a_src + Int32(16))
                            x5 = ld_shared_u32(a_src + Int32(20))
                            x6 = ld_shared_u32(a_src + Int32(24))
                            x7 = ld_shared_u32(a_src + Int32(28))
                            sa_u8 = ld_shared_u32(
                                act_base
                                + Int32(
                                    ci * self.act_stride_bytes
                                    + self.act_f16_bytes_per_tok
                                )
                                + g * Int32(4)
                            )
                            sa_f = e4m3_byte_to_f32(sa_u8)
                            dot = fp4_dot8_sum(
                                w_lo, w_hi, x0, x1, x2, x3, x4, x5, x6, x7
                            )
                            acc[ci] = acc[ci] + dot * (sa_f * sw_f)

                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        val = acc[ci]
                        val = val + shfl_xor_f32(val, 4)
                        val = val + shfl_xor_f32(val, 2)
                        val = val + shfl_xor_f32(val, 1)
                        if lane8 == Int32(0):
                            route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                            scaled = val * (mAOuter[route_c] * g_ratio)
                            mRouteOut[route_c, out_row] = scaled.to(cutlass.BFloat16)

                cute.arch.barrier()
                t_base = t_base + Int32(self.token_chunk)
