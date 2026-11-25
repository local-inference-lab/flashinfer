# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 CuTe-DSL block-scaled MMA prefill kernels for route-weighted SiTU MoE.

Uses native SM120 tensor-core instructions directly:
  - NVFP4 (L0-19): ``nvfp4_mma_m16n8k64_f32_e2m1`` with inline CSF slab decode
    into shared memory (zero HBM scale scratch) and zero-shuffle fused
    interleaved ``(gate, up)`` SiTU + route-weight epilogue on FC1.
  - MXFP8 (L20-23): ``mxfp8_mma_m16n8k32_f32_e4m3`` (native ``kind::mxf8f6f4``
    tensor core, 2x BF16 MMA throughput) with fused SiTU epilogue on FC1.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32, const_expr

from b12x._lib.intrinsics.atomics import (
    atomic_add_global_i32,
    ld_shared_u32,
    scatter_add_bf16x2,
    st_shared_u32,
)
from b12x._lib.intrinsics.memory import (
    create_l2_evict_first_policy,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_u8,
    st_shared_v4_u32,
)
from b12x._lib.intrinsics.mma import (
    mxfp8_mma_m16n8k32_f32_e4m3,
    nvfp4_mma_m16n8k64_f32_e2m1,
)
from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2

from ._math import (
    ld_global_nc_l2evict_v4_u32,
    ld_global_nc_u8,
    rcp_rn_f32,
    situ_activate_f32,
)


class SituPrefillRouteClearKernel:
    """Zero ``counts[E]`` and initialize ``task_expert[max_tasks] = -1``."""

    def __init__(self, num_experts: int, max_tasks: int):
        self.num_experts = int(num_experts)
        self.max_tasks = int(max_tasks)
        self.n_total = max(self.num_experts, self.max_tasks)
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        mCounts: cute.Tensor,
        mTaskExpert: cute.Tensor,
        stream: cuda.CUstream,
    ):
        blocks = (self.n_total + self.threads - 1) // self.threads
        self.kernel(mCounts, mTaskExpert).launch(
            grid=(blocks, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mCounts: cute.Tensor, mTaskExpert: cute.Tensor):
        idx = Int32(cute.arch.block_idx()[0]) * Int32(self.threads) + Int32(
            cute.arch.thread_idx()[0]
        )
        if idx < Int32(self.num_experts):
            mCounts[idx] = Int32(0)
        if idx < Int32(self.max_tasks):
            mTaskExpert[idx] = Int32(-1)


class SituPrefillRoutePackKernel:
    """Group valid routes ``0..R-1`` into ``rows[expert, pos]`` using per-expert counters."""

    def __init__(self, num_experts: int, capacity: int):
        self.num_experts = int(num_experts)
        self.capacity = int(capacity)
        self.threads = 256

    @cute.jit
    def __call__(
        self,
        mIds: cute.Tensor,  # [R] i32
        mCounts: cute.Tensor,  # [E] i32
        mRows: cute.Tensor,  # [E, capacity] i32
        num_routes: Int32,
        stream: cuda.CUstream,
    ):
        blocks = (self.capacity + self.threads - 1) // self.threads
        self.kernel(mIds, mCounts, mRows, num_routes).launch(
            grid=(blocks, 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIds: cute.Tensor,
        mCounts: cute.Tensor,
        mRows: cute.Tensor,
        num_routes: Int32,
    ):
        r = Int32(cute.arch.block_idx()[0]) * Int32(self.threads) + Int32(
            cute.arch.thread_idx()[0]
        )
        if r < num_routes:
            e = mIds[r]
            if e >= Int32(0) and e < Int32(self.num_experts):
                cnt_ptr = get_ptr_as_int64(mCounts, Int64(e))
                pos = atomic_add_global_i32(cnt_ptr, Int32(1))
                mRows[e, pos] = r


class SituPrefillBuildTasksKernel:
    """Single-CTA task-table builder mapping ``counts[E]`` -> ``(task_expert, task_begin)``."""

    def __init__(self, num_experts: int, tile_m: int):
        self.num_experts = int(num_experts)
        self.tile_m = int(tile_m)
        self.threads = 128

    def _get_shared_storage_cls(self):
        E = self.num_experts

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "counts": cute.struct.MemRange[cutlass.Int32, E],
            "offsets": cute.struct.MemRange[cutlass.Int32, E],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mCounts: cute.Tensor,
        mTaskExpert: cute.Tensor,
        mTaskBegin: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(mCounts, mTaskExpert, mTaskBegin).launch(
            grid=(1, 1, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCounts: cute.Tensor,
        mTaskExpert: cute.Tensor,
        mTaskBegin: cute.Tensor,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        s_cnt = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
        s_off = storage.offsets.get_tensor(cute.make_layout((self.num_experts,)))

        e_iters: cutlass.Constexpr = (
            self.num_experts + self.threads - 1
        ) // self.threads
        for it in cutlass.range_constexpr(e_iters):
            e = tid + Int32(it * self.threads)
            if e < Int32(self.num_experts):
                s_cnt[e] = mCounts[e]
        cute.arch.barrier()

        if tid == Int32(0):
            running = Int32(0)
            for e in range(self.num_experts):
                s_off[e] = running
                c = s_cnt[e]
                if c > Int32(0):
                    running = running + (c + Int32(self.tile_m - 1)) // Int32(
                        self.tile_m
                    )
        cute.arch.barrier()

        for it in cutlass.range_constexpr(e_iters):
            e = tid + Int32(it * self.threads)
            if e < Int32(self.num_experts):
                c = s_cnt[e]
                if c > Int32(0):
                    base = s_off[e]
                    nt = (c + Int32(self.tile_m - 1)) // Int32(self.tile_m)
                    t = Int32(0)
                    while t < nt:
                        mTaskExpert[base + t] = e
                        mTaskBegin[base + t] = t * Int32(self.tile_m)
                        t = t + Int32(1)


class SituPrefillNVFP4Kernel:
    """SM120 native NVFP4 block-scaled MMA prefill GEMM with inline CSF & fused SiTU.

    Tile: ``BM in {16, 32, 64}``, ``BN = 128`` (8 x 16-row CSF slabs = 64 interleaved
    ``(gate, up)`` pairs on FC1), ``BK = 64`` (one ``m16n8k64`` step per K chunk).
    Block: 256 threads = 8 warps (1x8 for BM=16, 2x4 for BM=32/64).
    Uses 16B vectorized global/shared loads and 2-stage double-buffered SMEM
    with register prefetch overlapping global loads against tensor-core MMAs.
    """

    def __init__(
        self,
        *,
        n_dim: int,
        k_dim: int,
        num_experts: int,
        top_k: int,
        is_fc1: bool,
        has_csf: bool,
        tile_m: int = 64,
        fused_route_sum: bool = False,
    ):
        if n_dim % 128 != 0 or k_dim % 128 != 0:
            raise ValueError(
                f"NVFP4 prefill requires n_dim % 128 == 0 and k_dim % 128 == 0, got {n_dim}, {k_dim}"
            )
        if tile_m not in (16, 32, 64, 128):
            raise ValueError(f"tile_m must be 16, 32, 64, or 128, got {tile_m}")
        self.n_dim = int(n_dim)
        self.k_dim = int(k_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.is_fc1 = bool(is_fc1)
        self.has_csf = bool(has_csf)
        self.tile_m = int(tile_m)
        self.fused_route_sum = bool(fused_route_sum) and not self.is_fc1
        self.tile_n = 128
        self.tile_k = 128
        self.threads = 256
        self.k_blocks = self.k_dim // 16
        self.k_steps = self.k_dim // self.tile_k
        self.n_tiles = self.n_dim // self.tile_n
        self.num_slabs = self.n_dim // 16
        self.m_warps = 1 if self.tile_m == 16 else (4 if self.tile_m == 128 else 2)
        self.n_warps = 8 // self.m_warps
        self.m_frags = (self.tile_m // self.m_warps) // 16
        self.n_frags = (self.tile_n // self.n_warps) // 8

        # Double-buffered SMEM layout (all 16B aligned, BK=128 -> 64B/row = 2 full 32B sectors):
        #   s_routes[BM] i32
        #   stage 0 & 1: s_a[BM, 64] + s_b[128, 64] + s_sfa[BM, 8] (+ s_sfb[128, 8] if dense)
        #   s_csf_sfb[128, k_blocks] u8 (if has_csf, decoded once per CTA)
        self.off_routes = 0
        self.off_stage0 = ((self.tile_m * 4 + 15) // 16) * 16
        self.st_off_a = 0
        self.st_off_b = self.tile_m * 64
        self.st_off_sfa = self.st_off_b + self.tile_n * 64
        self.st_off_sfb = self.st_off_sfa + ((self.tile_m * 8 + 15) // 16) * 16
        stage_end = self.st_off_sfb + (0 if self.has_csf else self.tile_n * 8)
        self.stage_stride = ((stage_end + 15) // 16) * 16
        self.off_csf_sfb = self.off_stage0 + 2 * self.stage_stride
        csf_bytes = self.tile_n * self.k_blocks if self.has_csf else 0
        self.smem_bytes = ((self.off_csf_sfb + csf_bytes + 15) // 16) * 16

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
        mAQ: cute.Tensor,  # [M_act, K//2] u8
        mAS: cute.Tensor,  # [M_act, K//16] u8
        mAOuter: cute.Tensor,  # [M_act] f32
        mW: cute.Tensor,  # flat u8 view of [E, N, K//2]
        mS: cute.Tensor,  # flat u8 view of dense [E, N, K//16] OR CSF fixed
        mExc: cute.Tensor,  # [num_exc] u32
        mSlabOffs: cute.Tensor,  # [E * num_slabs + 1] i32
        mG: cute.Tensor,  # [2 * E] f32
        mWeights: cute.Tensor,  # [R] f32 (used only when is_fc1=True)
        mRows: cute.Tensor,  # [E, capacity] i32
        mCounts: cute.Tensor,  # [E] i32
        mTaskExpert: cute.Tensor,  # [max_tasks] i32
        mTaskBegin: cute.Tensor,  # [max_tasks] i32
        mOut: cute.Tensor,  # [R, N//2 if is_fc1 else N] bf16
        stream: cuda.CUstream,
    ):
        max_tasks = mTaskExpert.shape[0]
        self.kernel(
            mAQ,
            mAS,
            mAOuter,
            mW,
            mS,
            mExc,
            mSlabOffs,
            mG,
            mWeights,
            mRows,
            mCounts,
            mTaskExpert,
            mTaskBegin,
            mOut,
        ).launch(
            grid=(max_tasks, self.n_tiles, 1),
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
        mW: cute.Tensor,
        mS: cute.Tensor,
        mExc: cute.Tensor,
        mSlabOffs: cute.Tensor,
        mG: cute.Tensor,
        mWeights: cute.Tensor,
        mRows: cute.Tensor,
        mCounts: cute.Tensor,
        mTaskExpert: cute.Tensor,
        mTaskBegin: cute.Tensor,
        mOut: cute.Tensor,
    ):
        task = Int32(cute.arch.block_idx()[0])
        n_tile = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])

        expert = mTaskExpert[task]
        if expert >= Int32(0):
            begin = mTaskBegin[task]
            count = mCounts[expert]
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            base_s = shared_ptr_to_u32(storage.payload.data_ptr())
            s_routes = base_s + Int32(self.off_routes)
            s_st0 = base_s + Int32(self.off_stage0)
            s_csf_sfb = base_s + Int32(self.off_csf_sfb)

            # Stage route indices for this BM tile.
            if tid < Int32(self.tile_m):
                pos = begin + tid
                r_val = Int32(-1)
                if pos < count:
                    r_val = mRows[expert, pos]
                st_shared_u32(s_routes + tid * Int32(4), r_val.to(Uint32))

            n_base = n_tile * Int32(self.tile_n)
            # If CSF enabled, decode all 8 16-row slabs for this BN=128 tile once into SMEM!
            if const_expr(self.has_csf):
                slab0 = n_tile * Int32(8)
                half_cols: cutlass.Constexpr = self.k_blocks // 2
                slab_bytes: cutlass.Constexpr = 16 * (1 + half_cols)
                pairs_per_tile: cutlass.Constexpr = 128 * half_cols
                first_slab_idx = expert * Int32(self.num_slabs) + slab0
                fixed_base_ptr = get_ptr_as_int64(
                    mS, Int64(first_slab_idx) * Int64(slab_bytes)
                )
                iters_csf: cutlass.Constexpr = (
                    pairs_per_tile + self.threads - 1
                ) // self.threads
                for it in cutlass.range_constexpr(iters_csf):
                    idx = tid + Int32(it * self.threads)
                    if idx < Int32(pairs_per_tile):
                        local_slab = idx // Int32(16 * half_cols)
                        rem = idx - local_slab * Int32(16 * half_cols)
                        local_row = rem // Int32(half_cols)
                        pair_col = rem - local_row * Int32(half_cols)
                        sp = fixed_base_ptr + Int64(local_slab) * Int64(slab_bytes)
                        base_u8 = ld_global_nc_u8(sp + Int64(local_row))
                        packed = ld_global_nc_u8(sp + Int64(16 + rem))
                        lo = (base_u8 + (packed & Uint32(0xF))) & Uint32(0xFF)
                        hi = (base_u8 + ((packed >> Uint32(4)) & Uint32(0xF))) & Uint32(
                            0xFF
                        )
                        dst_off = (local_slab * Int32(16) + local_row) * Int32(
                            self.k_blocks
                        ) + pair_col * Int32(2)
                        st_shared_u8(s_csf_sfb + dst_off, lo.to(Uint8))
                        st_shared_u8(s_csf_sfb + dst_off + Int32(1), hi.to(Uint8))
                cute.arch.barrier()

                exc_start = mSlabOffs[first_slab_idx]
                exc_end = mSlabOffs[first_slab_idx + Int32(8)]
                tile_flat_base = slab0 * Int32(16 * self.k_blocks)
                pos = exc_start + tid
                while pos < exc_end:
                    rec = ld_global_nc_u32(get_ptr_as_int64(mExc, Int64(pos)))
                    flat = (rec & Uint32(0xFFFFFF)).to(Int32) - tile_flat_base
                    val = ((rec >> Uint32(24)) & Uint32(0xFF)).to(Uint8)
                    st_shared_u8(s_csf_sfb + flat, val)
                    pos = pos + Int32(self.threads)
            cute.arch.barrier()

            warp = tid // Int32(32)
            lane = tid % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            warp_m = warp // Int32(self.n_warps)
            warp_n = warp % Int32(self.n_warps)
            sf_a_subrow = q + ((c & Int32(1)) << Int32(3))

            acc = cute.make_rmem_tensor(
                cute.make_layout((self.m_frags, self.n_frags, 4)), Float32
            )
            for mf in cutlass.range_constexpr(self.m_frags):
                for nf in cutlass.range_constexpr(self.n_frags):
                    for d in cutlass.range_constexpr(4):
                        acc[mf, nf, d] = Float32(0.0)

            w_tile_base = get_ptr_as_int64(
                mW,
                (Int64(expert) * Int64(self.n_dim) + Int64(n_base))
                * Int64(self.k_dim // 2),
            )
            s_tile_base = get_ptr_as_int64(
                mS,
                (Int64(expert) * Int64(self.n_dim) + Int64(n_base))
                * Int64(self.k_blocks),
            )
            aq_base = get_ptr_as_int64(mAQ, Int64(0))
            as_base = get_ptr_as_int64(mAS, Int64(0))

            # Precompute thread addresses for 16B vectorized loads (BK=128 -> 64B/row):
            #   A[BM, 64B] = 4*BM vectors (<= 256 for BM<=64, exact 2 vecs/thread for BM=128)
            #   B[128, 64B] = 512 vectors (exact 2 vectors per thread: rows b0_row and b0_row + 64!)
            #   SFA[BM, 8B] = 2*BM u32 (<= 256 even for BM=128)
            #   SFB[128, 8B] = 256 u32 (exact 1 u32 per thread!)
            l2_policy = create_l2_evict_first_policy()
            a_vecs: cutlass.Constexpr = self.tile_m * 4
            a_row = tid >> Int32(2)
            a_col16 = (tid & Int32(3)) * Int32(16)
            a_route = Int32(-1)
            a_ar = Int32(0)
            if tid < Int32(a_vecs):
                a_route = ld_shared_u32(s_routes + a_row * Int32(4)).to(Int32)
                if a_route >= Int32(0):
                    a_ar = (
                        a_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else a_route
                    )
            a_row_ptr = aq_base + Int64(a_ar) * Int64(self.k_dim // 2) + Int64(a_col16)

            a1_route = Int32(-1)
            a1_ar = Int32(0)
            if const_expr(self.tile_m == 128):
                a1_route = ld_shared_u32(
                    s_routes + (a_row + Int32(64)) * Int32(4)
                ).to(Int32)
                if a1_route >= Int32(0):
                    a1_ar = (
                        a1_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else a1_route
                    )
            a1_row_ptr = aq_base + Int64(a1_ar) * Int64(self.k_dim // 2) + Int64(a_col16)

            sfa_words: cutlass.Constexpr = self.tile_m * 2
            sfa_row = tid >> Int32(1)
            sfa_col4 = (tid & Int32(1)) * Int32(4)
            sfa_route = Int32(-1)
            sfa_ar = Int32(0)
            if tid < Int32(sfa_words):
                sfa_route = ld_shared_u32(s_routes + sfa_row * Int32(4)).to(Int32)
                if sfa_route >= Int32(0):
                    sfa_ar = (
                        sfa_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else sfa_route
                    )
            sfa_row_ptr = (
                as_base + Int64(sfa_ar) * Int64(self.k_blocks) + Int64(sfa_col4)
            )

            b0_row = tid >> Int32(2)
            b1_row = b0_row + Int32(64)
            b_col16 = (tid & Int32(3)) * Int32(16)
            b0_ptr = (
                w_tile_base + Int64(b0_row) * Int64(self.k_dim // 2) + Int64(b_col16)
            )
            b1_ptr = (
                w_tile_base + Int64(b1_row) * Int64(self.k_dim // 2) + Int64(b_col16)
            )
            sfb_row = tid >> Int32(1)
            sfb_col4 = (tid & Int32(1)) * Int32(4)
            sfb_row_ptr = (
                s_tile_base + Int64(sfb_row) * Int64(self.k_blocks) + Int64(sfb_col4)
            )

            # Prologue: load ks = 0 into stage 0.
            if tid < Int32(a_vecs):
                av0 = Uint32(0)
                av1 = Uint32(0)
                av2 = Uint32(0)
                av3 = Uint32(0)
                if a_route >= Int32(0):
                    av0, av1, av2, av3 = ld_global_nc_v4_u32(a_row_ptr)
                st_shared_v4_u32(
                    s_st0 + Int32(self.st_off_a) + tid * Int32(16), av0, av1, av2, av3
                )
            if const_expr(self.tile_m == 128):
                av4 = Uint32(0)
                av5 = Uint32(0)
                av6 = Uint32(0)
                av7 = Uint32(0)
                if a1_route >= Int32(0):
                    av4, av5, av6, av7 = ld_global_nc_v4_u32(a1_row_ptr)
                st_shared_v4_u32(
                    s_st0 + Int32(self.st_off_a) + (tid + Int32(256)) * Int32(16),
                    av4,
                    av5,
                    av6,
                    av7,
                )
            if tid < Int32(sfa_words):
                sv = Uint32(0)
                if sfa_route >= Int32(0):
                    sv = ld_global_nc_u32(sfa_row_ptr)
                st_shared_u32(s_st0 + Int32(self.st_off_sfa) + tid * Int32(4), sv)
            bv0, bv1, bv2, bv3 = ld_global_nc_l2evict_v4_u32(b0_ptr, l2_policy)
            st_shared_v4_u32(
                s_st0 + Int32(self.st_off_b) + tid * Int32(16), bv0, bv1, bv2, bv3
            )
            bv4, bv5, bv6, bv7 = ld_global_nc_l2evict_v4_u32(b1_ptr, l2_policy)
            st_shared_v4_u32(
                s_st0 + Int32(self.st_off_b) + (tid + Int32(256)) * Int32(16),
                bv4,
                bv5,
                bv6,
                bv7,
            )
            if const_expr(not self.has_csf):
                st_shared_u32(
                    s_st0 + Int32(self.st_off_sfb) + tid * Int32(4),
                    ld_global_nc_u32(sfb_row_ptr),
                )
            cute.arch.barrier()

            # Runtime K-loop (zero instruction-cache bloat / zero register spilling!).
            for ks in range(self.k_steps):
                cur_base = s_st0 + (ks & Int32(1)) * Int32(self.stage_stride)
                nxt_base = s_st0 + ((ks + Int32(1)) & Int32(1)) * Int32(
                    self.stage_stride
                )
                s_a = cur_base + Int32(self.st_off_a)
                s_b = cur_base + Int32(self.st_off_b)
                s_sfa = cur_base + Int32(self.st_off_sfa)
                s_sfb = cur_base + Int32(self.st_off_sfb)

                # Issue non-blocking global loads for ks + 1 into registers!
                pa0 = Uint32(0)
                pa1 = Uint32(0)
                pa2 = Uint32(0)
                pa3 = Uint32(0)
                pa4 = Uint32(0)
                pa5 = Uint32(0)
                pa6 = Uint32(0)
                pa7 = Uint32(0)
                psfa = Uint32(0)
                pb0 = Uint32(0)
                pb1 = Uint32(0)
                pb2 = Uint32(0)
                pb3 = Uint32(0)
                pb4 = Uint32(0)
                pb5 = Uint32(0)
                pb6 = Uint32(0)
                pb7 = Uint32(0)
                psfb = Uint32(0)
                has_next = ks + Int32(1) < Int32(self.k_steps)
                if has_next:
                    nk_byte = Int64(ks + Int32(1)) * Int64(64)
                    nk_blk = Int64(ks + Int32(1)) * Int64(8)
                    if tid < Int32(a_vecs) and a_route >= Int32(0):
                        pa0, pa1, pa2, pa3 = ld_global_nc_v4_u32(a_row_ptr + nk_byte)
                    if const_expr(self.tile_m == 128):
                        if a1_route >= Int32(0):
                            pa4, pa5, pa6, pa7 = ld_global_nc_v4_u32(
                                a1_row_ptr + nk_byte
                            )
                    if tid < Int32(sfa_words) and sfa_route >= Int32(0):
                        psfa = ld_global_nc_u32(sfa_row_ptr + nk_blk)
                    pb0, pb1, pb2, pb3 = ld_global_nc_l2evict_v4_u32(
                        b0_ptr + nk_byte, l2_policy
                    )
                    pb4, pb5, pb6, pb7 = ld_global_nc_l2evict_v4_u32(
                        b1_ptr + nk_byte, l2_policy
                    )
                    if const_expr(not self.has_csf):
                        psfb = ld_global_nc_u32(sfb_row_ptr + nk_blk)

                # Execute 2 k=64 MMA sub-steps on cur SMEM while ks+1 global loads are in flight.
                for kb in cutlass.range_constexpr(2):
                    k_sub = Int32(kb * 32)
                    sf_sub = Int32(kb * 4)
                    a_frag0 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag1 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag2 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag3 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    sfa_r = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )

                    for mf in cutlass.range_constexpr(self.m_frags):
                        m0 = warp_m * Int32(self.m_frags * 16) + Int32(mf * 16) + q
                        m1 = m0 + Int32(8)
                        a_frag0[mf] = ld_shared_u32(
                            s_a + m0 * Int32(64) + k_sub + c * Int32(4)
                        )
                        a_frag1[mf] = ld_shared_u32(
                            s_a + m1 * Int32(64) + k_sub + c * Int32(4)
                        )
                        a_frag2[mf] = ld_shared_u32(
                            s_a + m0 * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )
                        a_frag3[mf] = ld_shared_u32(
                            s_a + m1 * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )
                        sfa_r[mf] = ld_shared_u32(
                            s_sfa
                            + (
                                warp_m * Int32(self.m_frags * 16)
                                + Int32(mf * 16)
                                + sf_a_subrow
                            )
                            * Int32(8)
                            + sf_sub
                        )

                    for nf in cutlass.range_constexpr(self.n_frags):
                        n_col = warp_n * Int32(self.n_frags * 8) + Int32(nf * 8) + q
                        b0 = ld_shared_u32(
                            s_b + n_col * Int32(64) + k_sub + c * Int32(4)
                        )
                        b1 = ld_shared_u32(
                            s_b + n_col * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )
                        if const_expr(self.has_csf):
                            sfb = ld_shared_u32(
                                s_csf_sfb
                                + n_col * Int32(self.k_blocks)
                                + ks * Int32(8)
                                + sf_sub
                            )
                        else:
                            sfb = ld_shared_u32(s_sfb + n_col * Int32(8) + sf_sub)

                        for mf in cutlass.range_constexpr(self.m_frags):
                            d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                                acc[mf, nf, 0],
                                acc[mf, nf, 1],
                                acc[mf, nf, 2],
                                acc[mf, nf, 3],
                                a_frag0[mf],
                                a_frag1[mf],
                                a_frag2[mf],
                                a_frag3[mf],
                                b0,
                                b1,
                                sfa_r[mf],
                                sfb,
                            )
                            acc[mf, nf, 0] = d0
                            acc[mf, nf, 1] = d1
                            acc[mf, nf, 2] = d2
                            acc[mf, nf, 3] = d3

                # Commit prefetched ks+1 registers into nxt SMEM buffer and sync once.
                if has_next:
                    if tid < Int32(a_vecs):
                        st_shared_v4_u32(
                            nxt_base + Int32(self.st_off_a) + tid * Int32(16),
                            pa0,
                            pa1,
                            pa2,
                            pa3,
                        )
                    if const_expr(self.tile_m == 128):
                        st_shared_v4_u32(
                            nxt_base
                            + Int32(self.st_off_a)
                            + (tid + Int32(256)) * Int32(16),
                            pa4,
                            pa5,
                            pa6,
                            pa7,
                        )
                    if tid < Int32(sfa_words):
                        st_shared_u32(
                            nxt_base + Int32(self.st_off_sfa) + tid * Int32(4), psfa
                        )
                    st_shared_v4_u32(
                        nxt_base + Int32(self.st_off_b) + tid * Int32(16),
                        pb0,
                        pb1,
                        pb2,
                        pb3,
                    )
                    st_shared_v4_u32(
                        nxt_base
                        + Int32(self.st_off_b)
                        + (tid + Int32(256)) * Int32(16),
                        pb4,
                        pb5,
                        pb6,
                        pb7,
                    )
                    if const_expr(not self.has_csf):
                        st_shared_u32(
                            nxt_base + Int32(self.st_off_sfb) + tid * Int32(4), psfb
                        )
                    cute.arch.barrier()

            # Epilogue: apply outer/global gain; if is_fc1, fuse interleaved (gate, up) SiTU + route weight;
            # if is_fc2 + fused_route_sum, scatter-add bf16x2 directly into mOut[tok, col0:col0+2]!
            g_ratio = mG[expert] * rcp_rn_f32(
                Float32(1536.0) * mG[Int32(self.num_experts) + expert]
            )
            out_base = get_ptr_as_int64(mOut, Int64(0))
            for mf in cutlass.range_constexpr(self.m_frags):
                m_top = warp_m * Int32(self.m_frags * 16) + Int32(mf * 16) + q
                m_bot = m_top + Int32(8)
                route_top = ld_shared_u32(s_routes + m_top * Int32(4)).to(Int32)
                route_bot = ld_shared_u32(s_routes + m_bot * Int32(4)).to(Int32)

                if route_top >= Int32(0):
                    ar_top = (
                        route_top // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else route_top
                    )
                    gain_top = mAOuter[ar_top] * g_ratio
                    rw_top = (
                        mWeights[route_top] if const_expr(self.is_fc1) else Float32(1.0)
                    )
                    dst_row_top = (
                        route_top // Int32(self.top_k)
                        if const_expr(self.fused_route_sum)
                        else route_top
                    )
                    for nf in cutlass.range_constexpr(self.n_frags):
                        col0 = (
                            n_base
                            + warp_n * Int32(self.n_frags * 8)
                            + Int32(nf * 8)
                            + c * Int32(2)
                        )
                        v0 = acc[mf, nf, 0] * gain_top
                        v1 = acc[mf, nf, 1] * gain_top
                        if const_expr(self.is_fc1):
                            y = situ_activate_f32(v0, v1, rw_top)
                            mOut[route_top, col0 >> Int32(1)] = y.to(cutlass.BFloat16)
                        else:
                            out_ptr = (
                                out_base
                                + (
                                    Int64(dst_row_top) * Int64(self.n_dim)
                                    + Int64(col0)
                                )
                                * Int64(2)
                            )
                            if const_expr(self.fused_route_sum):
                                scatter_add_bf16x2(out_ptr, v0, v1)
                            else:
                                st_global_u32(out_ptr, pack_f32x2_to_bfloat2(v0, v1))

                if route_bot >= Int32(0):
                    ar_bot = (
                        route_bot // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else route_bot
                    )
                    gain_bot = mAOuter[ar_bot] * g_ratio
                    rw_bot = (
                        mWeights[route_bot] if const_expr(self.is_fc1) else Float32(1.0)
                    )
                    dst_row_bot = (
                        route_bot // Int32(self.top_k)
                        if const_expr(self.fused_route_sum)
                        else route_bot
                    )
                    for nf in cutlass.range_constexpr(self.n_frags):
                        col0 = (
                            n_base
                            + warp_n * Int32(self.n_frags * 8)
                            + Int32(nf * 8)
                            + c * Int32(2)
                        )
                        v2 = acc[mf, nf, 2] * gain_bot
                        v3 = acc[mf, nf, 3] * gain_bot
                        if const_expr(self.is_fc1):
                            y = situ_activate_f32(v2, v3, rw_bot)
                            mOut[route_bot, col0 >> Int32(1)] = y.to(cutlass.BFloat16)
                        else:
                            out_ptr = (
                                out_base
                                + (
                                    Int64(dst_row_bot) * Int64(self.n_dim)
                                    + Int64(col0)
                                )
                                * Int64(2)
                            )
                            if const_expr(self.fused_route_sum):
                                scatter_add_bf16x2(out_ptr, v2, v3)
                            else:
                                st_global_u32(out_ptr, pack_f32x2_to_bfloat2(v2, v3))


class SituPrefillMXFP8Kernel:
    """SM120 native MXFP8 block-scaled MMA prefill GEMM with fused SiTU epilogue.

    Uses ``mxfp8_mma_m16n8k32_f32_e4m3`` (``kind::mxf8f6f4.block_scale.scale_vec::1X``),
    eliminating Triton's BF16 dequantization overhead for L20-23.
    Tile: ``BM in {16, 32, 64}``, ``BN = 128``, ``BK = 64`` (two k=32 MMA steps per K chunk),
    16B vectorized global/shared loads, and 2-stage double-buffered SMEM.
    """

    def __init__(
        self,
        *,
        n_dim: int,
        k_dim: int,
        num_experts: int,
        top_k: int,
        is_fc1: bool,
        tile_m: int = 64,
        fused_route_sum: bool = False,
    ):
        if n_dim % 128 != 0 or k_dim % 64 != 0:
            raise ValueError(
                f"MXFP8 prefill requires n_dim % 128 == 0 and k_dim % 64 == 0, got {n_dim}, {k_dim}"
            )
        if tile_m not in (16, 32, 64, 128):
            raise ValueError(f"tile_m must be 16, 32, 64, or 128, got {tile_m}")
        self.n_dim = int(n_dim)
        self.k_dim = int(k_dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.is_fc1 = bool(is_fc1)
        self.tile_m = int(tile_m)
        self.fused_route_sum = bool(fused_route_sum) and not self.is_fc1
        self.tile_n = 128
        self.tile_k = 64
        self.threads = 256
        self.k_blocks = self.k_dim // 32
        self.k_steps = self.k_dim // self.tile_k
        self.n_tiles = self.n_dim // self.tile_n
        self.m_warps = 1 if self.tile_m == 16 else (4 if self.tile_m == 128 else 2)
        self.n_warps = 8 // self.m_warps
        self.m_frags = (self.tile_m // self.m_warps) // 16
        self.n_frags = (self.tile_n // self.n_warps) // 8

        self.off_routes = 0
        self.off_stage0 = ((self.tile_m * 4 + 15) // 16) * 16
        self.st_off_a = 0
        self.st_off_b = self.tile_m * 64
        self.st_off_sfa = self.st_off_b + self.tile_n * 64
        self.st_off_sfb = self.st_off_sfa + ((self.tile_m * 4 + 15) // 16) * 16
        self.stage_stride = ((self.st_off_sfb + self.tile_n * 4 + 15) // 16) * 16
        self.smem_bytes = self.off_stage0 + 2 * self.stage_stride

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
        mAQ: cute.Tensor,  # [M_act, K] u8 (E4M3FN)
        mAS: cute.Tensor,  # [M_act, K//32] u8 (UE8M0)
        mW: cute.Tensor,  # flat u8 view of [E, N, K]
        mS: cute.Tensor,  # [E, N, K//32] u8 (UE8M0)
        mWeights: cute.Tensor,  # [R] f32
        mRows: cute.Tensor,  # [E, capacity] i32
        mCounts: cute.Tensor,  # [E] i32
        mTaskExpert: cute.Tensor,  # [max_tasks] i32
        mTaskBegin: cute.Tensor,  # [max_tasks] i32
        mOut: cute.Tensor,  # [R, N//2 if is_fc1 else N] bf16
        stream: cuda.CUstream,
    ):
        max_tasks = mTaskExpert.shape[0]
        self.kernel(
            mAQ,
            mAS,
            mW,
            mS,
            mWeights,
            mRows,
            mCounts,
            mTaskExpert,
            mTaskBegin,
            mOut,
        ).launch(
            grid=(max_tasks, self.n_tiles, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mAQ: cute.Tensor,
        mAS: cute.Tensor,
        mW: cute.Tensor,
        mS: cute.Tensor,
        mWeights: cute.Tensor,
        mRows: cute.Tensor,
        mCounts: cute.Tensor,
        mTaskExpert: cute.Tensor,
        mTaskBegin: cute.Tensor,
        mOut: cute.Tensor,
    ):
        task = Int32(cute.arch.block_idx()[0])
        n_tile = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])

        expert = mTaskExpert[task]
        if expert >= Int32(0):
            begin = mTaskBegin[task]
            count = mCounts[expert]
            smem = cutlass.memory.SmemAllocator()
            storage = smem.allocate(self._get_shared_storage_cls())
            base_s = shared_ptr_to_u32(storage.payload.data_ptr())
            s_routes = base_s + Int32(self.off_routes)
            s_st0 = base_s + Int32(self.off_stage0)

            if tid < Int32(self.tile_m):
                pos = begin + tid
                r_val = Int32(-1)
                if pos < count:
                    r_val = mRows[expert, pos]
                st_shared_u32(s_routes + tid * Int32(4), r_val.to(Uint32))
            cute.arch.barrier()

            n_base = n_tile * Int32(self.tile_n)
            warp = tid // Int32(32)
            lane = tid % Int32(32)
            q = lane // Int32(4)
            c = lane % Int32(4)
            warp_m = warp // Int32(self.n_warps)
            warp_n = warp % Int32(self.n_warps)
            sf_a_subrow = q + ((c & Int32(1)) << Int32(3))

            acc = cute.make_rmem_tensor(
                cute.make_layout((self.m_frags, self.n_frags, 4)), Float32
            )
            for mf in cutlass.range_constexpr(self.m_frags):
                for nf in cutlass.range_constexpr(self.n_frags):
                    for d in cutlass.range_constexpr(4):
                        acc[mf, nf, d] = Float32(0.0)

            w_tile_base = get_ptr_as_int64(
                mW,
                (Int64(expert) * Int64(self.n_dim) + Int64(n_base)) * Int64(self.k_dim),
            )
            aq_base = get_ptr_as_int64(mAQ, Int64(0))

            # 16B vectorized load coordinates:
            #   A[BM, 64B] = 4*BM vectors (<= 256 for BM<=64, exact 2 vecs/thread for BM=128)
            #   B[128, 64B] = 512 vectors (exact 2 vectors per thread: tid and tid + 256!)
            l2_policy = create_l2_evict_first_policy()
            a_vecs: cutlass.Constexpr = self.tile_m * 4
            a_row = tid >> Int32(2)
            a_col16 = (tid & Int32(3)) * Int32(16)
            a_route = Int32(-1)
            a_ar = Int32(0)
            if tid < Int32(a_vecs):
                a_route = ld_shared_u32(s_routes + a_row * Int32(4)).to(Int32)
                if a_route >= Int32(0):
                    a_ar = (
                        a_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else a_route
                    )
            a_row_ptr = aq_base + Int64(a_ar) * Int64(self.k_dim) + Int64(a_col16)

            a1_route = Int32(-1)
            a1_ar = Int32(0)
            if const_expr(self.tile_m == 128):
                a1_route = ld_shared_u32(
                    s_routes + (a_row + Int32(64)) * Int32(4)
                ).to(Int32)
                if a1_route >= Int32(0):
                    a1_ar = (
                        a1_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else a1_route
                    )
            a1_row_ptr = aq_base + Int64(a1_ar) * Int64(self.k_dim) + Int64(a_col16)

            sfa_route = Int32(-1)
            sfa_ar = Int32(0)
            if tid < Int32(self.tile_m):
                sfa_route = ld_shared_u32(s_routes + tid * Int32(4)).to(Int32)
                if sfa_route >= Int32(0):
                    sfa_ar = (
                        sfa_route // Int32(self.top_k)
                        if const_expr(self.is_fc1)
                        else sfa_route
                    )

            b0_row = tid >> Int32(2)
            b1_row = b0_row + Int32(64)
            b_col16 = (tid & Int32(3)) * Int32(16)
            b0_ptr = w_tile_base + Int64(b0_row) * Int64(self.k_dim) + Int64(b_col16)
            b1_ptr = w_tile_base + Int64(b1_row) * Int64(self.k_dim) + Int64(b_col16)

            # Prologue: load ks = 0 into stage 0.
            if tid < Int32(a_vecs):
                av0 = Uint32(0)
                av1 = Uint32(0)
                av2 = Uint32(0)
                av3 = Uint32(0)
                if a_route >= Int32(0):
                    av0, av1, av2, av3 = ld_global_nc_v4_u32(a_row_ptr)
                st_shared_v4_u32(
                    s_st0 + Int32(self.st_off_a) + tid * Int32(16), av0, av1, av2, av3
                )
            if const_expr(self.tile_m == 128):
                av4 = Uint32(0)
                av5 = Uint32(0)
                av6 = Uint32(0)
                av7 = Uint32(0)
                if a1_route >= Int32(0):
                    av4, av5, av6, av7 = ld_global_nc_v4_u32(a1_row_ptr)
                st_shared_v4_u32(
                    s_st0 + Int32(self.st_off_a) + (tid + Int32(256)) * Int32(16),
                    av4,
                    av5,
                    av6,
                    av7,
                )
            if tid < Int32(self.tile_m):
                sv = Uint32(0x7F7F7F7F)
                if sfa_route >= Int32(0):
                    sv = mAS[sfa_ar, Int32(0)].to(Uint32) | (
                        mAS[sfa_ar, Int32(1)].to(Uint32) << Uint32(8)
                    )
                st_shared_u32(s_st0 + Int32(self.st_off_sfa) + tid * Int32(4), sv)
            bv0, bv1, bv2, bv3 = ld_global_nc_l2evict_v4_u32(b0_ptr, l2_policy)
            st_shared_v4_u32(
                s_st0 + Int32(self.st_off_b) + tid * Int32(16), bv0, bv1, bv2, bv3
            )
            bv4, bv5, bv6, bv7 = ld_global_nc_l2evict_v4_u32(b1_ptr, l2_policy)
            st_shared_v4_u32(
                s_st0 + Int32(self.st_off_b) + (tid + Int32(256)) * Int32(16),
                bv4,
                bv5,
                bv6,
                bv7,
            )
            if tid < Int32(self.tile_n):
                n_row_g = n_base + tid
                sfb0 = mS[expert, n_row_g, Int32(0)].to(Uint32) | (
                    mS[expert, n_row_g, Int32(1)].to(Uint32) << Uint32(8)
                )
                st_shared_u32(s_st0 + Int32(self.st_off_sfb) + tid * Int32(4), sfb0)
            cute.arch.barrier()

            for ks in range(self.k_steps):
                cur_base = s_st0 + (ks & Int32(1)) * Int32(self.stage_stride)
                nxt_base = s_st0 + ((ks + Int32(1)) & Int32(1)) * Int32(
                    self.stage_stride
                )
                s_a = cur_base + Int32(self.st_off_a)
                s_b = cur_base + Int32(self.st_off_b)
                s_sfa = cur_base + Int32(self.st_off_sfa)
                s_sfb = cur_base + Int32(self.st_off_sfb)

                # Prefetch ks + 1 into registers while MMAs execute on cur SMEM.
                pa0 = Uint32(0)
                pa1 = Uint32(0)
                pa2 = Uint32(0)
                pa3 = Uint32(0)
                pa4 = Uint32(0)
                pa5 = Uint32(0)
                pa6 = Uint32(0)
                pa7 = Uint32(0)
                psfa = Uint32(0x7F7F7F7F)
                pb0 = Uint32(0)
                pb1 = Uint32(0)
                pb2 = Uint32(0)
                pb3 = Uint32(0)
                pb4 = Uint32(0)
                pb5 = Uint32(0)
                pb6 = Uint32(0)
                pb7 = Uint32(0)
                psfb = Uint32(0)
                has_next = ks + Int32(1) < Int32(self.k_steps)
                if has_next:
                    nk_byte = Int64(ks + Int32(1)) * Int64(64)
                    nblk0 = (ks + Int32(1)) * Int32(2)
                    nblk1 = nblk0 + Int32(1)
                    if tid < Int32(a_vecs) and a_route >= Int32(0):
                        pa0, pa1, pa2, pa3 = ld_global_nc_v4_u32(a_row_ptr + nk_byte)
                    if const_expr(self.tile_m == 128):
                        if a1_route >= Int32(0):
                            pa4, pa5, pa6, pa7 = ld_global_nc_v4_u32(
                                a1_row_ptr + nk_byte
                            )
                    if tid < Int32(self.tile_m) and sfa_route >= Int32(0):
                        psfa = mAS[sfa_ar, nblk0].to(Uint32) | (
                            mAS[sfa_ar, nblk1].to(Uint32) << Uint32(8)
                        )
                    pb0, pb1, pb2, pb3 = ld_global_nc_l2evict_v4_u32(
                        b0_ptr + nk_byte, l2_policy
                    )
                    pb4, pb5, pb6, pb7 = ld_global_nc_l2evict_v4_u32(
                        b1_ptr + nk_byte, l2_policy
                    )
                    if tid < Int32(self.tile_n):
                        n_row_g = n_base + tid
                        psfb = mS[expert, n_row_g, nblk0].to(Uint32) | (
                            mS[expert, n_row_g, nblk1].to(Uint32) << Uint32(8)
                        )

                sfa_r = cute.make_rmem_tensor(cute.make_layout((self.m_frags,)), Uint32)
                for mf in cutlass.range_constexpr(self.m_frags):
                    sfa_r[mf] = ld_shared_u32(
                        s_sfa
                        + (
                            warp_m * Int32(self.m_frags * 16)
                            + Int32(mf * 16)
                            + sf_a_subrow
                        )
                        * Int32(4)
                    )
                sfb_r = cute.make_rmem_tensor(cute.make_layout((self.n_frags,)), Uint32)
                for nf in cutlass.range_constexpr(self.n_frags):
                    n_col = warp_n * Int32(self.n_frags * 8) + Int32(nf * 8) + q
                    sfb_r[nf] = ld_shared_u32(s_sfb + n_col * Int32(4))

                for kb in cutlass.range_constexpr(2):
                    k_sub = Int32(kb * 32)
                    a_frag0 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag1 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag2 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    a_frag3 = cute.make_rmem_tensor(
                        cute.make_layout((self.m_frags,)), Uint32
                    )
                    for mf in cutlass.range_constexpr(self.m_frags):
                        m0 = warp_m * Int32(self.m_frags * 16) + Int32(mf * 16) + q
                        m1 = m0 + Int32(8)
                        a_frag0[mf] = ld_shared_u32(
                            s_a + m0 * Int32(64) + k_sub + c * Int32(4)
                        )
                        a_frag1[mf] = ld_shared_u32(
                            s_a + m1 * Int32(64) + k_sub + c * Int32(4)
                        )
                        a_frag2[mf] = ld_shared_u32(
                            s_a + m0 * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )
                        a_frag3[mf] = ld_shared_u32(
                            s_a + m1 * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )

                    for nf in cutlass.range_constexpr(self.n_frags):
                        n_col = warp_n * Int32(self.n_frags * 8) + Int32(nf * 8) + q
                        b0 = ld_shared_u32(
                            s_b + n_col * Int32(64) + k_sub + c * Int32(4)
                        )
                        b1 = ld_shared_u32(
                            s_b + n_col * Int32(64) + k_sub + Int32(16) + c * Int32(4)
                        )
                        for mf in cutlass.range_constexpr(self.m_frags):
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
                                acc[mf, nf, 0],
                                acc[mf, nf, 1],
                                acc[mf, nf, 2],
                                acc[mf, nf, 3],
                                a_frag0[mf],
                                a_frag1[mf],
                                a_frag2[mf],
                                a_frag3[mf],
                                b0,
                                b1,
                                sfa_r[mf],
                                sfb_r[nf],
                                bid_a=kb,
                                tid_a=0,
                                bid_b=kb,
                                tid_b=0,
                            )
                            acc[mf, nf, 0] = d0
                            acc[mf, nf, 1] = d1
                            acc[mf, nf, 2] = d2
                            acc[mf, nf, 3] = d3

                if has_next:
                    if tid < Int32(a_vecs):
                        st_shared_v4_u32(
                            nxt_base + Int32(self.st_off_a) + tid * Int32(16),
                            pa0,
                            pa1,
                            pa2,
                            pa3,
                        )
                    if const_expr(self.tile_m == 128):
                        st_shared_v4_u32(
                            nxt_base
                            + Int32(self.st_off_a)
                            + (tid + Int32(256)) * Int32(16),
                            pa4,
                            pa5,
                            pa6,
                            pa7,
                        )
                    if tid < Int32(self.tile_m):
                        st_shared_u32(
                            nxt_base + Int32(self.st_off_sfa) + tid * Int32(4), psfa
                        )
                    st_shared_v4_u32(
                        nxt_base + Int32(self.st_off_b) + tid * Int32(16),
                        pb0,
                        pb1,
                        pb2,
                        pb3,
                    )
                    st_shared_v4_u32(
                        nxt_base
                        + Int32(self.st_off_b)
                        + (tid + Int32(256)) * Int32(16),
                        pb4,
                        pb5,
                        pb6,
                        pb7,
                    )
                    if tid < Int32(self.tile_n):
                        st_shared_u32(
                            nxt_base + Int32(self.st_off_sfb) + tid * Int32(4), psfb
                        )
                    cute.arch.barrier()

            # Epilogue: fused interleaved (gate, up) SiTU on FC1, or fused scatter-add / packed BF16x2 store on FC2.
            out_base = get_ptr_as_int64(mOut, Int64(0))
            for mf in cutlass.range_constexpr(self.m_frags):
                m_top = warp_m * Int32(self.m_frags * 16) + Int32(mf * 16) + q
                m_bot = m_top + Int32(8)
                route_top = ld_shared_u32(s_routes + m_top * Int32(4)).to(Int32)
                route_bot = ld_shared_u32(s_routes + m_bot * Int32(4)).to(Int32)

                if route_top >= Int32(0):
                    rw_top = (
                        mWeights[route_top] if const_expr(self.is_fc1) else Float32(1.0)
                    )
                    dst_row_top = (
                        route_top // Int32(self.top_k)
                        if const_expr(self.fused_route_sum)
                        else route_top
                    )
                    for nf in cutlass.range_constexpr(self.n_frags):
                        col0 = (
                            n_base
                            + warp_n * Int32(self.n_frags * 8)
                            + Int32(nf * 8)
                            + c * Int32(2)
                        )
                        v0 = acc[mf, nf, 0]
                        v1 = acc[mf, nf, 1]
                        if const_expr(self.is_fc1):
                            y = situ_activate_f32(v0, v1, rw_top)
                            mOut[route_top, col0 >> Int32(1)] = y.to(cutlass.BFloat16)
                        else:
                            out_ptr = (
                                out_base
                                + (
                                    Int64(dst_row_top) * Int64(self.n_dim)
                                    + Int64(col0)
                                )
                                * Int64(2)
                            )
                            if const_expr(self.fused_route_sum):
                                scatter_add_bf16x2(out_ptr, v0, v1)
                            else:
                                st_global_u32(out_ptr, pack_f32x2_to_bfloat2(v0, v1))

                if route_bot >= Int32(0):
                    rw_bot = (
                        mWeights[route_bot] if const_expr(self.is_fc1) else Float32(1.0)
                    )
                    dst_row_bot = (
                        route_bot // Int32(self.top_k)
                        if const_expr(self.fused_route_sum)
                        else route_bot
                    )
                    for nf in cutlass.range_constexpr(self.n_frags):
                        col0 = (
                            n_base
                            + warp_n * Int32(self.n_frags * 8)
                            + Int32(nf * 8)
                            + c * Int32(2)
                        )
                        v2 = acc[mf, nf, 2]
                        v3 = acc[mf, nf, 3]
                        if const_expr(self.is_fc1):
                            y = situ_activate_f32(v2, v3, rw_bot)
                            mOut[route_bot, col0 >> Int32(1)] = y.to(cutlass.BFloat16)
                        else:
                            out_ptr = (
                                out_base
                                + (
                                    Int64(dst_row_bot) * Int64(self.n_dim)
                                    + Int64(col0)
                                )
                                * Int64(2)
                            )
                            if const_expr(self.fused_route_sum):
                                scatter_add_bf16x2(out_ptr, v2, v3)
                            else:
                                st_global_u32(out_ptr, pack_f32x2_to_bfloat2(v2, v3))
