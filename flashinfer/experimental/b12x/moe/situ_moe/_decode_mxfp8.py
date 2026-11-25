# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 CuTe-DSL MXFP8 decode MoE kernels with fused interleaved SiTU.

Targets Inkling 2 Mini MXFP8 expert layers (L20-23, block-32 UE8M0 E4M3) for
decode batch sizes (M <= 16):
  - Coalesced 128B/64B ``ld.global.nc.v2.u32`` weight streaming with register
    reuse across up to 4 routed tokens per active expert.
  - Fused interleaved ``(gate, up)`` SiTU activation + route-weight multiply
    inside FC1 (eliminates ``[R, 2*I]`` HBM traffic and separate activate launch).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, const_expr

from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.memory import get_ptr_as_int64, shared_ptr_to_u32

from ._math import (
    e4m3x4_dot_f32,
    ld_global_nc_v2_u32,
    shfl_idx_f32,
    shfl_xor_f32,
    situ_activate_f32,
    ue8m0_byte_to_f32,
)


@cute.jit
def _stage_mxfp8_act_to_smem(
    mAQ: cute.Tensor,  # [M_act, K] u8 (E4M3FN bits)
    mAS: cute.Tensor,  # [M_act, K//32] u8 (UE8M0)
    act_row: Int32,
    smem_a_addr: Int32,  # [K] u8 (aligned 16B)
    smem_as_addr: Int32,  # [K//32] u32 (stores UE8M0 byte)
    tid: Int32,
    threads: cutlass.Constexpr,
    k_dim: cutlass.Constexpr,
):
    """Cooperatively stage one MXFP8 activation row and its UE8M0 scales into SMEM."""
    k_v2: cutlass.Constexpr = k_dim // 8
    k_blocks: cutlass.Constexpr = k_dim // 32
    row_q_ptr = get_ptr_as_int64(mAQ, Int64(act_row) * Int64(k_dim))
    iters_q: cutlass.Constexpr = (k_v2 + threads - 1) // threads
    for it in cutlass.range_constexpr(iters_q):
        idx = tid + Int32(it * threads)
        if idx < Int32(k_v2):
            v0, v1 = ld_global_nc_v2_u32(row_q_ptr + Int64(idx) * Int64(8))
            st_shared_u32(smem_a_addr + idx * Int32(8), v0)
            st_shared_u32(smem_a_addr + idx * Int32(8) + Int32(4), v1)
    iters_s: cutlass.Constexpr = (k_blocks + threads - 1) // threads
    for it in cutlass.range_constexpr(iters_s):
        blk = tid + Int32(it * threads)
        if blk < Int32(k_blocks):
            st_shared_u32(smem_as_addr + blk * Int32(4), mAS[act_row, blk].to(Uint32))


class SituDecodeFC1MXFP8Kernel:
    """Warp-specialized MXFP8 FC1 + fused interleaved SiTU for decode (M <= 16).

    Grid ``(2*I // 8, max_slots, 1)``, block ``(128, 1, 1)`` (4 warps = 8
    half-warps = 4 interleaved ``(gate, up)`` pairs per CTA; 16 lanes/row).
    """

    def __init__(
        self,
        *,
        hidden: int,
        intermediate: int,
        num_experts: int,
        top_k: int,
        max_tokens: int,
    ):
        if hidden % 128 != 0 or intermediate % 4 != 0:
            raise ValueError(
                "MXFP8 FC1 decode requires hidden % 128 == 0 and intermediate % 4 == 0"
            )
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.n_rows = 2 * self.intermediate
        self.num_tiles = self.n_rows // 8
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.max_tokens = int(max_tokens)
        self.k_blocks = self.hidden // 32
        self.k_iters = self.hidden // 128  # 16 lanes x 8B = 128B/iter = 4 blocks/iter
        self.threads = 128
        self.token_chunk = min(4, self.max_tokens)

        self.act_q_bytes = self.hidden
        self.act_s_bytes = self.k_blocks * 4
        self.act_stride_bytes = ((self.act_q_bytes + self.act_s_bytes + 15) // 16) * 16
        self.smem_bytes = self.token_chunk * self.act_stride_bytes

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
        mAQ: cute.Tensor,  # [M, H] u8 (E4M3FN)
        mAS: cute.Tensor,  # [M, H//32] u8 (UE8M0)
        mW13: cute.Tensor,  # flat u8 view of [E, 2*I, H]
        mS13: cute.Tensor,  # [E, 2*I, H//32] u8 (UE8M0)
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
            mW13,
            mS13,
            mWeights,
            mActiveExperts,
            mActiveCounts,
            mActiveRoutes,
            mInter,
        ).launch(
            grid=(self.num_tiles, max_slots, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAQ: cute.Tensor,
        mAS: cute.Tensor,
        mW13: cute.Tensor,
        mS13: cute.Tensor,
        mWeights: cute.Tensor,
        mActiveExperts: cute.Tensor,
        mActiveCounts: cute.Tensor,
        mActiveRoutes: cute.Tensor,
        mInter: cute.Tensor,
    ):
        tile = Int32(cute.arch.block_idx()[0])
        slot = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        expert = mActiveExperts[slot]
        count = mActiveCounts[slot]

        if expert >= Int32(0) and count > Int32(0):
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            act_base = shared_ptr_to_u32(storage.payload.data_ptr())

            hw = tid // Int32(16)
            lane16 = tid % Int32(16)
            warp = tid // Int32(32)
            lane = tid % Int32(32)
            row_in_expert = tile * Int32(8) + hw
            w_row_ptr = get_ptr_as_int64(
                mW13,
                (Int64(expert) * Int64(self.n_rows) + Int64(row_in_expert))
                * Int64(self.hidden),
            )
            blk_in_iter = lane16 >> Int32(2)

            t_base = Int32(0)
            while t_base < count:
                chunk_n = cutlass.min(count - t_base, Int32(self.token_chunk))

                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                        tok_c = route_c // Int32(self.top_k)
                        a_dst = act_base + Int32(ci * self.act_stride_bytes)
                        _stage_mxfp8_act_to_smem(
                            mAQ,
                            mAS,
                            tok_c,
                            a_dst,
                            a_dst + Int32(self.act_q_bytes),
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

                for ki in cutlass.range_constexpr(self.k_iters):
                    byte_off = (Int32(ki * 16) + lane16) * Int32(8)
                    blk = Int32(ki * 4) + blk_in_iter
                    w_lo, w_hi = ld_global_nc_v2_u32(w_row_ptr + Int64(byte_off))
                    sw_f = ue8m0_byte_to_f32(
                        mS13[expert, row_in_expert, blk].to(Uint32)
                    )

                    for ci in cutlass.range_constexpr(self.token_chunk):
                        if Int32(ci) < chunk_n:
                            c_base = act_base + Int32(ci * self.act_stride_bytes)
                            a_lo = ld_shared_u32(c_base + byte_off)
                            a_hi = ld_shared_u32(c_base + byte_off + Int32(4))
                            sa_f = ue8m0_byte_to_f32(
                                ld_shared_u32(
                                    c_base + Int32(self.act_q_bytes) + blk * Int32(4)
                                )
                            )
                            dot = e4m3x4_dot_f32(w_lo, a_lo) + e4m3x4_dot_f32(
                                w_hi, a_hi
                            )
                            acc[ci] = acc[ci] + dot * (sa_f * sw_f)

                out_col = tile * Int32(4) + warp
                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        val = acc[ci]
                        val = val + shfl_xor_f32(val, 8)
                        val = val + shfl_xor_f32(val, 4)
                        val = val + shfl_xor_f32(val, 2)
                        val = val + shfl_xor_f32(val, 1)
                        up_val = shfl_idx_f32(val, 16)
                        if lane == Int32(0):
                            route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                            rw = mWeights[route_c]
                            act_val = situ_activate_f32(val, up_val, rw)
                            mInter[route_c, out_col] = act_val.to(cutlass.BFloat16)

                cute.arch.barrier()
                t_base = t_base + Int32(self.token_chunk)


class SituDecodeFC2MXFP8Kernel:
    """Warp-specialized MXFP8 FC2 for decode (M <= 16).

    Grid ``(H // 16, max_slots, 1)``, block ``(128, 1, 1)`` (16 groups of 8 lanes;
    8 lanes x 8B = 64B coalesced load = 2 UE8M0 blocks per iteration).
    """

    def __init__(
        self,
        *,
        hidden: int,
        intermediate: int,
        num_experts: int,
        max_tokens: int,
    ):
        if hidden % 16 != 0 or intermediate % 64 != 0:
            raise ValueError(
                "MXFP8 FC2 decode requires hidden % 16 == 0 and intermediate % 64 == 0"
            )
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.num_tiles = self.hidden // 16
        self.num_experts = int(num_experts)
        self.max_tokens = int(max_tokens)
        self.k_blocks = self.intermediate // 32
        self.k_iters = self.intermediate // 64
        self.threads = 128
        self.token_chunk = min(4, self.max_tokens)

        self.act_q_bytes = self.intermediate
        self.act_s_bytes = self.k_blocks * 4
        self.act_stride_bytes = ((self.act_q_bytes + self.act_s_bytes + 15) // 16) * 16
        self.smem_bytes = self.token_chunk * self.act_stride_bytes

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
        mAQ: cute.Tensor,  # [R, I] u8 (E4M3FN)
        mAS: cute.Tensor,  # [R, I//32] u8 (UE8M0)
        mW2: cute.Tensor,  # flat u8 view of [E, H, I]
        mS2: cute.Tensor,  # [E, H, I//32] u8 (UE8M0)
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
            mW2,
            mS2,
            mActiveExperts,
            mActiveCounts,
            mActiveRoutes,
            mRouteOut,
        ).launch(
            grid=(self.num_tiles, max_slots, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAQ: cute.Tensor,
        mAS: cute.Tensor,
        mW2: cute.Tensor,
        mS2: cute.Tensor,
        mActiveExperts: cute.Tensor,
        mActiveCounts: cute.Tensor,
        mActiveRoutes: cute.Tensor,
        mRouteOut: cute.Tensor,
    ):
        tile = Int32(cute.arch.block_idx()[0])
        slot = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        expert = mActiveExperts[slot]
        count = mActiveCounts[slot]

        if expert >= Int32(0) and count > Int32(0):
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            act_base = shared_ptr_to_u32(storage.payload.data_ptr())

            qw = tid // Int32(8)
            lane8 = tid % Int32(8)
            out_row = tile * Int32(16) + qw
            w_row_ptr = get_ptr_as_int64(
                mW2,
                (Int64(expert) * Int64(self.hidden) + Int64(out_row))
                * Int64(self.intermediate),
            )
            blk_in_iter = lane8 >> Int32(2)

            t_base = Int32(0)
            while t_base < count:
                chunk_n = cutlass.min(count - t_base, Int32(self.token_chunk))

                for ci in cutlass.range_constexpr(self.token_chunk):
                    if Int32(ci) < chunk_n:
                        route_c = mActiveRoutes[slot, t_base + Int32(ci)]
                        a_dst = act_base + Int32(ci * self.act_stride_bytes)
                        _stage_mxfp8_act_to_smem(
                            mAQ,
                            mAS,
                            route_c,
                            a_dst,
                            a_dst + Int32(self.act_q_bytes),
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
                    byte_off = (Int32(ki * 8) + lane8) * Int32(8)
                    blk = Int32(ki * 2) + blk_in_iter
                    w_lo, w_hi = ld_global_nc_v2_u32(w_row_ptr + Int64(byte_off))
                    sw_f = ue8m0_byte_to_f32(mS2[expert, out_row, blk].to(Uint32))

                    for ci in cutlass.range_constexpr(self.token_chunk):
                        if Int32(ci) < chunk_n:
                            c_base = act_base + Int32(ci * self.act_stride_bytes)
                            a_lo = ld_shared_u32(c_base + byte_off)
                            a_hi = ld_shared_u32(c_base + byte_off + Int32(4))
                            sa_f = ue8m0_byte_to_f32(
                                ld_shared_u32(
                                    c_base + Int32(self.act_q_bytes) + blk * Int32(4)
                                )
                            )
                            dot = e4m3x4_dot_f32(w_lo, a_lo) + e4m3x4_dot_f32(
                                w_hi, a_hi
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
                            mRouteOut[route_c, out_row] = val.to(cutlass.BFloat16)

                cute.arch.barrier()
                t_base = t_base + Int32(self.token_chunk)
