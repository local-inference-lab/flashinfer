# SPDX-License-Identifier: Apache-2.0
"""Gather four Kimi latent and router pairs and select 16 experts in each row CTA."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.program_cache import program_cache
from b12x._lib.intrinsics.memory import (
    ld_global_cg_v4_u32,
    ld_global_v4_u32,
    st_global_v4_u32,
)
from b12x._lib.utils.device import current_cuda_stream
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._island_rs_cute import (
    _fence_sc_sys,
    _load_acquire_sys_u32,
)
from ._dcp_a2a_cute import (
    _DCPA2ABase,
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _MAX_BLOCKS,
    _SLOT_ALIGNMENT,
    _a2a_graph_epoch_arrive,
    _copy_16b,
    _u32_ptr,
    _u8_ptr,
    _kimi_key_expert,
    _kimi_pack_key,
    _kimi_warp_max_key,
    _f32_ptr,
    _i32_ptr,
)
from ._dcp_cute_common import (
    _FLAG_STRIDE,
    _MAX_RANKS,
    _store_relaxed_sys_u32,
)


@dsl_user_op
def _u32_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.bitcast(T.f32(), Uint32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


class GatherRouterTopK(_DCPA2ABase):
    def __init__(
        self,
        world_size: int,
        rank: int,
        threads: int,
        device_slot_selection: bool,
        rows: int = 4,
    ) -> None:
        super().__init__(world_size, rank, threads, device_slot_selection)
        if world_size != 16:
            raise ValueError(f"island row route requires 16 ranks, got {world_size}")
        if type(rows) is not int or not 1 <= rows <= _MAX_BLOCKS:
            raise ValueError(f"row route requires 1..{_MAX_BLOCKS} rows, got {rows}")
        self._rows = rows
        self._island = rank // 4
        self._lane = rank % 4

    @cute.jit
    def _pair_barrier(
        self, signals, peer: cutlass.Constexpr[int], bidx, tid
    ) -> None:
        """Strict 1-to-1 single-peer barrier using single-leader fence and ld.acquire.sys."""
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _fence_sc_sys()
            self_signal = signals[self._rank]
            self_ptr = (
                self_signal + Int64(bidx) * Int64(_MAX_RANKS) + Int64(peer)
            )
            value = Uint32(cute.arch.load(self_ptr, Uint32)) + Uint32(1)
            cute.arch.store(self_ptr, value)

            slot = Int64(value % Uint32(2))
            common_index = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                (slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            peer_index = common_index + Int64(self._rank * _FLAG_STRIDE)
            mine_index = common_index + Int64(peer * _FLAG_STRIDE)
            _store_relaxed_sys_u32(signals[peer] + peer_index, value)
            mine_addr = Int64((self_signal + mine_index).toint())
            observed = _load_acquire_sys_u32(mine_addr)
            while observed != value:
                observed = _load_acquire_sys_u32(mine_addr)
        cute.arch.sync_threads()

    @cute.jit
    def _sw4_barrier(self, signals, bidx, tid) -> None:
        """Intra-switch 4-GPU barrier using single-leader fence and ld.acquire.sys."""
        cute.arch.sync_threads()
        p0, p1, p2 = self._rank ^ 1, self._rank ^ 2, self._rank ^ 3
        if Int32(tid) == Int32(0):
            _fence_sc_sys()
            self_sig = signals[self._rank]
            base_cnt = self_sig + Int64(bidx) * Int64(_MAX_RANKS)

            g0 = Uint32(cute.arch.load(base_cnt + Int64(p0), Uint32)) + Uint32(1)
            cute.arch.store(base_cnt + Int64(p0), g0)
            s0 = Int64(g0 % Uint32(2))
            ci0 = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                (s0 * Int64(_MAX_BLOCKS) + Int64(bidx))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            _store_relaxed_sys_u32(
                signals[p0] + ci0 + Int64(self._rank * _FLAG_STRIDE), g0
            )

            g1 = Uint32(cute.arch.load(base_cnt + Int64(p1), Uint32)) + Uint32(1)
            cute.arch.store(base_cnt + Int64(p1), g1)
            s1 = Int64(g1 % Uint32(2))
            ci1 = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                (s1 * Int64(_MAX_BLOCKS) + Int64(bidx))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            _store_relaxed_sys_u32(
                signals[p1] + ci1 + Int64(self._rank * _FLAG_STRIDE), g1
            )

            g2 = Uint32(cute.arch.load(base_cnt + Int64(p2), Uint32)) + Uint32(1)
            cute.arch.store(base_cnt + Int64(p2), g2)
            s2 = Int64(g2 % Uint32(2))
            ci2 = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                (s2 * Int64(_MAX_BLOCKS) + Int64(bidx))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            _store_relaxed_sys_u32(
                signals[p2] + ci2 + Int64(self._rank * _FLAG_STRIDE), g2
            )

            a0 = Int64((self_sig + ci0 + Int64(p0 * _FLAG_STRIDE)).toint())
            o0 = _load_acquire_sys_u32(a0)
            while o0 != g0:
                o0 = _load_acquire_sys_u32(a0)
            a1 = Int64((self_sig + ci1 + Int64(p1 * _FLAG_STRIDE)).toint())
            o1 = _load_acquire_sys_u32(a1)
            while o1 != g1:
                o1 = _load_acquire_sys_u32(a1)
            a2 = Int64((self_sig + ci2 + Int64(p2 * _FLAG_STRIDE)).toint())
            o2 = _load_acquire_sys_u32(a2)
            while o2 != g2:
                o2 = _load_acquire_sys_u32(a2)
        cute.arch.sync_threads()

    @cute.jit
    def _group3_barrier(self, signals, peers, bidx, tid) -> None:
        """3-peer barrier (cross-switch or intra-switch) with single-leader fence."""
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _fence_sc_sys()
            self_sig = signals[self._rank]
            base_cnt = self_sig + Int64(bidx) * Int64(_MAX_RANKS)
            g = [Uint32(0), Uint32(0), Uint32(0)]
            ci = [Int64(0), Int64(0), Int64(0)]
            for idx in cutlass.range_constexpr(3):
                p = peers[idx]
                gv = Uint32(cute.arch.load(base_cnt + Int64(p), Uint32)) + Uint32(1)
                cute.arch.store(base_cnt + Int64(p), gv)
                sl = Int64(gv % Uint32(2))
                c = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                    (sl * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                )
                _store_relaxed_sys_u32(
                    signals[p] + c + Int64(self._rank * _FLAG_STRIDE), gv
                )
                g[idx] = gv
                ci[idx] = c
            for idx in cutlass.range_constexpr(3):
                p = peers[idx]
                addr = Int64((self_sig + ci[idx] + Int64(p * _FLAG_STRIDE)).toint())
                obs = _load_acquire_sys_u32(addr)
                while obs != g[idx]:
                    obs = _load_acquire_sys_u32(addr)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        first: cute.Pointer,
        second: cute.Pointer,
        out_first: cute.Pointer,
        out_second: cute.Pointer,
        bias: cute.Pointer,
        weights: cute.Pointer,
        ids: cute.Pointer,
        stages: tuple,
        signals: tuple,
        stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            first,
            second,
            out_first,
            out_second,
            bias,
            weights,
            ids,
            stages,
            signals,
            stride,
        ).launch(grid=(self._rows, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        first,
        second,
        out_first,
        out_second,
        bias,
        weights,
        ids,
        staging,
        signals,
        stride,
    ):
        tid, _, _ = cute.arch.thread_idx()
        first_packs, second_packs = Int32(28), Int32(14)
        slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            epoch = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot_offset = (
                Int64(epoch % Uint32(2)) * Int64(stride) * Int64(_SLOT_ALIGNMENT)
            )
        row, _, _ = cute.arch.block_idx()
        row_stage_words = Int64(row) * Int64(16 * 42 * 4)
        self_stage = (
            self._staging_words(staging[self._rank] + slot_offset) + row_stage_words
        )
        wlane, warp = tid % 32, tid // 32

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            logits_smem: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            unbiased_scores: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            warp_keys: cute.struct.Align[cute.struct.MemRange[Uint64, 128], 16]

        storage = smem_alloc.allocate(SharedStorage)
        logits_smem = storage.logits_smem.get_tensor(
            cute.make_layout((896,), stride=(1,))
        )
        unbiased_scores = storage.unbiased_scores.get_tensor(
            cute.make_layout((896,), stride=(1,))
        )
        warp_keys = storage.warp_keys.get_tensor(cute.make_layout((128,), stride=(1,)))

        # 2-phase Cross3-first (r^4,r^8,r^12: ONLY 3 shards cross uplink, overlapped
        # with own-shard intra-switch push) -> Switch-4 forward (3 cross shards over
        # 54.8 GB/s physical switch): 2 barriers, 0 PCIe reads, 0-bit exact.
        xs_peers = (self._rank ^ 4, self._rank ^ 8, self._rank ^ 12)
        sw_peers = (self._rank ^ 1, self._rank ^ 2, self._rank ^ 3)

        if warp == Int32(0):
            out_row = Int64(row) * Int64(16) + Int64(self._rank)
            if wlane < first_packs:
                vals = ld_global_v4_u32(
                    (first + Int64(row * first_packs + wlane) * 4).toint()
                )
                st_global_v4_u32(
                    (out_first + (out_row * first_packs + wlane) * 4).toint(), *vals
                )
            if wlane < second_packs:
                vals = ld_global_v4_u32(
                    (second + Int64(row * second_packs + wlane) * 4).toint()
                )
                st_global_v4_u32(
                    (out_second + (out_row * second_packs + wlane) * 4).toint(), *vals
                )
                base = Int32(self._rank * 56) + Int32(wlane * 4)
                logits_smem[base + 0] = _u32_to_f32(vals[0])
                logits_smem[base + 1] = _u32_to_f32(vals[1])
                logits_smem[base + 2] = _u32_to_f32(vals[2])
                logits_smem[base + 3] = _u32_to_f32(vals[3])
        for pi in cutlass.range_constexpr(3):
            if warp == Int32(pi + 1):
                peer = xs_peers[pi]
                dst_base = (
                    self._staging_words(staging[peer] + slot_offset)
                    + row_stage_words
                    + Int64(self._rank * 42 * 4)
                )
                if wlane < first_packs:
                    _copy_16b(
                        first + Int64(row * first_packs + wlane) * 4,
                        dst_base + Int64(wlane) * 4,
                    )
                if wlane < second_packs:
                    _copy_16b(
                        second + Int64(row * second_packs + wlane) * 4,
                        dst_base + Int64(first_packs + wlane) * 4,
                    )
            if warp == Int32(pi + 4):
                peer = sw_peers[pi]
                dst_base = (
                    self._staging_words(staging[peer] + slot_offset)
                    + row_stage_words
                    + Int64(self._rank * 42 * 4)
                )
                if wlane < first_packs:
                    _copy_16b(
                        first + Int64(row * first_packs + wlane) * 4,
                        dst_base + Int64(wlane) * 4,
                    )
                if wlane < second_packs:
                    _copy_16b(
                        second + Int64(row * second_packs + wlane) * 4,
                        dst_base + Int64(first_packs + wlane) * 4,
                    )
        self._group3_barrier(signals, xs_peers, row, tid)

        # Phase 2: Forward 3 cross-switch shards to 3 intra-switch peers (warps 0..8)
        # while warps 9..11 unpack those 3 shards in parallel.
        for pi in cutlass.range_constexpr(3):
            peer = sw_peers[pi]
            rem_sw = (
                self._staging_words(staging[peer] + slot_offset) + row_stage_words
            )
            for si in cutlass.range_constexpr(3):
                if warp == Int32(pi * 3 + si):
                    sr = xs_peers[si]
                    src_base = self_stage + Int64(sr * 42 * 4)
                    dst_base = rem_sw + Int64(sr * 42 * 4)
                    if wlane < first_packs:
                        vals = ld_global_cg_v4_u32(
                            (src_base + Int64(wlane) * 4).toint()
                        )
                        st_global_v4_u32(
                            (dst_base + Int64(wlane) * 4).toint(), *vals
                        )
                    if wlane < second_packs:
                        vals = ld_global_cg_v4_u32(
                            (src_base + Int64(first_packs + wlane) * 4).toint()
                        )
                        st_global_v4_u32(
                            (dst_base + Int64(first_packs + wlane) * 4).toint(), *vals
                        )
        for si in cutlass.range_constexpr(3):
            if warp == Int32(9 + si):
                peer = xs_peers[si]
                src_base = self_stage + Int64(peer * 42 * 4)
                out_row = Int64(row) * Int64(16) + Int64(peer)
                if wlane < first_packs:
                    vals = ld_global_cg_v4_u32((src_base + Int64(wlane) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * first_packs + wlane) * 4).toint(),
                        *vals,
                    )
                if wlane < second_packs:
                    vals = ld_global_cg_v4_u32(
                        (src_base + Int64(first_packs + wlane) * 4).toint()
                    )
                    st_global_v4_u32(
                        (out_second + (out_row * second_packs + wlane) * 4).toint(),
                        *vals,
                    )
                    base = Int32(peer * 56) + Int32(wlane * 4)
                    logits_smem[base + 0] = _u32_to_f32(vals[0])
                    logits_smem[base + 1] = _u32_to_f32(vals[1])
                    logits_smem[base + 2] = _u32_to_f32(vals[2])
                    logits_smem[base + 3] = _u32_to_f32(vals[3])
        self._group3_barrier(signals, sw_peers, row, tid)

        # Unpack remaining 12 received shards from local L2-coherent staging.
        rem_peers = tuple(
            p for p in range(16) if p not in (self._rank, xs_peers[0], xs_peers[1], xs_peers[2])
        )
        for idx in cutlass.range_constexpr(12):
            if warp == Int32(idx):
                peer = rem_peers[idx]
                src_base = self_stage + Int64(peer * 42 * 4)
                out_row = Int64(row) * Int64(16) + Int64(peer)
                if wlane < first_packs:
                    vals = ld_global_cg_v4_u32((src_base + Int64(wlane) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * first_packs + wlane) * 4).toint(),
                        *vals,
                    )
                if wlane < second_packs:
                    vals = ld_global_cg_v4_u32(
                        (src_base + Int64(first_packs + wlane) * 4).toint()
                    )
                    st_global_v4_u32(
                        (out_second + (out_row * second_packs + wlane) * 4).toint(),
                        *vals,
                    )
                    base = Int32(peer * 56) + Int32(wlane * 4)
                    logits_smem[base + 0] = _u32_to_f32(vals[0])
                    logits_smem[base + 1] = _u32_to_f32(vals[1])
                    logits_smem[base + 2] = _u32_to_f32(vals[2])
                    logits_smem[base + 3] = _u32_to_f32(vals[3])
        cute.arch.sync_threads()
        if cutlass.const_expr(self._device_slot_selection):
            if tid == 0:
                flags = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (flags + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (flags + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(self._rows),
                )

        self.select(logits_smem, unbiased_scores, warp_keys, bias, weights, ids)

    @cute.jit
    def select(
        self,
        logits_smem,
        unbiased_scores,
        warp_keys,
        correction_bias,
        output_weights,
        output_ids,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)
        if tidx < Int32(256):
            keys = cute.make_rmem_tensor((4,), Uint64)
            for item in cutlass.range_constexpr(4):
                expert = Int32(tidx) + Int32(item * 256)
                key = Uint64(0)
                if expert < Int32(896):
                    router_value = logits_smem[expert]
                    bias = cute.arch.load(correction_bias + Int64(expert), Float32)
                    unbiased = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-router_value, fastmath=True)
                    )
                    if not cute.math.isfinite(unbiased):
                        unbiased = Float32(0.0)
                    selection = unbiased + bias
                    if not cute.math.isfinite(selection):
                        if selection > Float32(0.0):
                            selection = Float32(float("inf"))
                        else:
                            selection = Float32(float("-inf"))
                    if selection == Float32(0.0):
                        selection = Float32(0.0)
                    key = _kimi_pack_key(selection, expert)
                    unbiased_scores[expert] = unbiased
                keys[item] = key

            # Packed keys are unique because they include the expert ID. Each warp
            # therefore removes exactly one candidate per selection without a CTA
            # barrier or an active-expert table in shared memory.
            for selected in cutlass.range_constexpr(16):
                # Score keys occupy the unsigned high half; load register elements
                # explicitly in that domain before reducing them.
                thread_key = Uint64(keys[0])
                for item in cutlass.range_constexpr(1, 4):
                    thread_key = cutlass.max(thread_key, Uint64(keys[item]))
                winner = _kimi_warp_max_key(thread_key)
                if lane == Int32(0):
                    warp_keys[warp * Int32(16) + Int32(selected)] = winner
                for item in cutlass.range_constexpr(4):
                    if keys[item] == winner:
                        keys[item] = Uint64(0)
        cute.arch.sync_threads()

        if warp == Int32(0):
            cursor = Int32(0)
            head = Uint64(0)
            if lane < Int32(8):
                head = warp_keys[lane * Int32(16)]
            selected_ids = cute.make_rmem_tensor((16,), Int32)
            selected_weights = cute.make_rmem_tensor((16,), Float32)
            for selected in cutlass.range_constexpr(16):
                winner = _kimi_warp_max_key(head)
                expert = _kimi_key_expert(winner)
                if lane == Int32(0):
                    selected_ids[selected] = expert
                    selected_weights[selected] = unbiased_scores[expert]
                owner_warp = (expert % Int32(256)) // Int32(32)
                if lane == owner_warp:
                    cursor += Int32(1)
                    head = Uint64(0)
                    if cursor < Int32(16):
                        head = warp_keys[lane * Int32(16) + cursor]
            if lane == Int32(0):
                weight_sum = Float32(0.0)
                for selected in cutlass.range_constexpr(16):
                    weight_sum += selected_weights[selected]
                denominator = Float32(1.0)
                if weight_sum > Float32(0.0):
                    denominator = weight_sum
                scale = Float32(1.0) / denominator
                output_offset = Int64(bidx) * Int64(16)
                for selected in cutlass.range_constexpr(16):
                    cute.arch.store(
                        output_weights + output_offset + Int64(selected),
                        selected_weights[selected] * scale,
                    )
                    cute.arch.store(
                        output_ids + output_offset + Int64(selected),
                        selected_ids[selected],
                    )


class GatherRouterTopKTP8(_DCPA2ABase):
    """Fused TP8 latent + router AllGather + Kimi TopK16 (1 CTA per row, 512 threads).

    On dual-switch 8-GPU PCIe topologies with Relaxed Ordering:
      - mode="direct" (1-hop XOR-balanced push): fastest for rows <= 16 (10.3-13.1 us)
      - mode="hier"   (2-hop Switch4 x Pair2):   fastest for rows > 16  (16.0 us @ 32 rows)
    """

    def __init__(
        self,
        rank: int,
        threads: int,
        device_slot_selection: bool,
        rows: int = 4,
        mode: str | None = None,
    ) -> None:
        super().__init__(8, rank, threads, device_slot_selection)
        if type(rows) is not int or not 1 <= rows <= _MAX_BLOCKS:
            raise ValueError(f"row route requires 1..{_MAX_BLOCKS} rows, got {rows}")
        self._rows = rows
        self._mode = ("direct" if rows <= 16 else "hier") if mode is None else mode

    @cute.jit
    def _barrier_peers(self, signals, peers, bidx, tid):
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _fence_sc_sys()
            self_sig = signals[self._rank]
            base_cnt = self_sig + Int64(bidx) * Int64(_MAX_RANKS)
            for i in cutlass.range_constexpr(len(peers)):
                p = peers[i]
                g = Uint32(cute.arch.load(base_cnt + Int64(p), Uint32)) + Uint32(1)
                cute.arch.store(base_cnt + Int64(p), g)
                slot = Int64(g % Uint32(2))
                ci = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                    (slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                )
                _store_relaxed_sys_u32(
                    signals[p] + ci + Int64(self._rank * _FLAG_STRIDE), g
                )
            for i in cutlass.range_constexpr(len(peers)):
                p = peers[i]
                g = Uint32(cute.arch.load(base_cnt + Int64(p), Uint32))
                slot = Int64(g % Uint32(2))
                ci = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                    (slot * Int64(_MAX_BLOCKS) + Int64(bidx))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                )
                addr = Int64((self_sig + ci + Int64(p * _FLAG_STRIDE)).toint())
                obs = _load_acquire_sys_u32(addr)
                while obs != g:
                    obs = _load_acquire_sys_u32(addr)
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        first: cute.Pointer,
        second: cute.Pointer,
        out_first: cute.Pointer,
        out_second: cute.Pointer,
        bias: cute.Pointer,
        weights: cute.Pointer,
        ids: cute.Pointer,
        stages: tuple,
        signals: tuple,
        stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            first, second, out_first, out_second, bias, weights, ids, stages, signals, stride
        ).launch(grid=(self._rows, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self, first, second, out_first, out_second, bias, weights, ids, staging, signals, stride
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        first_packs, second_packs = Int32(56), Int32(28)
        slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            epoch = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot_offset = (
                Int64(epoch % Uint32(2)) * Int64(stride) * Int64(_SLOT_ALIGNMENT)
            )
        row_stage_words = Int64(row) * Int64(8 * 84 * 4)
        self_stage = (
            self._staging_words(staging[self._rank] + slot_offset) + row_stage_words
        )
        wlane, warp = tid % 32, tid // 32

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            logits_smem: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            unbiased_scores: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            warp_keys: cute.struct.Align[cute.struct.MemRange[Uint64, 128], 16]

        storage = smem_alloc.allocate(SharedStorage)
        logits_smem = storage.logits_smem.get_tensor(cute.make_layout((896,), stride=(1,)))
        unbiased_scores = storage.unbiased_scores.get_tensor(
            cute.make_layout((896,), stride=(1,))
        )
        warp_keys = storage.warp_keys.get_tensor(cute.make_layout((128,), stride=(1,)))

        if cutlass.const_expr(self._mode == "direct"):
            peers = tuple(self._rank ^ d for d in range(1, 8))
            for pi in cutlass.range_constexpr(7):
                if warp == Int32(pi):
                    peer = peers[pi]
                    rem = (
                        self._staging_words(staging[peer] + slot_offset)
                        + row_stage_words
                    )
                    dst_base = rem + Int64(self._rank * 84 * 4)
                    _copy_16b(
                        first + Int64(row * 56 + wlane) * 4,
                        dst_base + Int64(wlane) * 4,
                    )
                    if wlane < Int32(24):
                        p2 = wlane + Int32(32)
                        _copy_16b(
                            first + Int64(row * 56 + p2) * 4,
                            dst_base + Int64(p2) * 4,
                        )
                    if wlane < second_packs:
                        _copy_16b(
                            second + Int64(row * 28 + wlane) * 4,
                            dst_base + Int64(56 + wlane) * 4,
                        )
            if warp == Int32(7):
                out_row = Int64(row) * Int64(8) + Int64(self._rank)
                _copy_16b(
                    first + Int64(row * 56 + wlane) * 4,
                    out_first + (out_row * 56 + wlane) * 4,
                )
                if wlane < Int32(24):
                    p2 = wlane + Int32(32)
                    _copy_16b(
                        first + Int64(row * 56 + p2) * 4,
                        out_first + (out_row * 56 + p2) * 4,
                    )
                if wlane < second_packs:
                    vals = ld_global_v4_u32(
                        (second + Int64(row * 28 + wlane) * 4).toint()
                    )
                    st_global_v4_u32(
                        (out_second + (out_row * 28 + wlane) * 4).toint(), *vals
                    )
                    base = Int32(self._rank * 112) + Int32(wlane * 4)
                    logits_smem[base + 0] = _u32_to_f32(vals[0])
                    logits_smem[base + 1] = _u32_to_f32(vals[1])
                    logits_smem[base + 2] = _u32_to_f32(vals[2])
                    logits_smem[base + 3] = _u32_to_f32(vals[3])

            self._barrier_peers(signals, peers, row, tid)

            for pi in cutlass.range_constexpr(7):
                if warp == Int32(pi):
                    peer = peers[pi]
                    src_base = self_stage + Int64(peer * 84 * 4)
                    out_row = Int64(row) * Int64(8) + Int64(peer)
                    v0 = ld_global_cg_v4_u32((src_base + Int64(wlane) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * 56 + wlane) * 4).toint(), *v0
                    )
                    if wlane < Int32(24):
                        p2 = wlane + Int32(32)
                        v1 = ld_global_cg_v4_u32((src_base + Int64(p2) * 4).toint())
                        st_global_v4_u32(
                            (out_first + (out_row * 56 + p2) * 4).toint(), *v1
                        )
                    if wlane < second_packs:
                        vals = ld_global_cg_v4_u32(
                            (src_base + Int64(56 + wlane) * 4).toint()
                        )
                        st_global_v4_u32(
                            (out_second + (out_row * 28 + wlane) * 4).toint(), *vals
                        )
                        base = Int32(peer * 112) + Int32(wlane * 4)
                        logits_smem[base + 0] = _u32_to_f32(vals[0])
                        logits_smem[base + 1] = _u32_to_f32(vals[1])
                        logits_smem[base + 2] = _u32_to_f32(vals[2])
                        logits_smem[base + 3] = _u32_to_f32(vals[3])
        else:
            p4 = self._rank ^ 4
            rem4 = self._staging_words(staging[p4] + slot_offset) + row_stage_words
            if warp == Int32(0):
                dst_base = rem4 + Int64(self._rank * 84 * 4)
                out_row = Int64(row) * Int64(8) + Int64(self._rank)
                v0 = ld_global_v4_u32((first + Int64(row * 56 + wlane) * 4).toint())
                st_global_v4_u32(
                    (out_first + (out_row * 56 + wlane) * 4).toint(), *v0
                )
                st_global_v4_u32((dst_base + Int64(wlane) * 4).toint(), *v0)
                if wlane < Int32(24):
                    p2 = wlane + Int32(32)
                    v1 = ld_global_v4_u32((first + Int64(row * 56 + p2) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * 56 + p2) * 4).toint(), *v1
                    )
                    st_global_v4_u32((dst_base + Int64(p2) * 4).toint(), *v1)
                if wlane < second_packs:
                    vals = ld_global_v4_u32(
                        (second + Int64(row * 28 + wlane) * 4).toint()
                    )
                    st_global_v4_u32(
                        (out_second + (out_row * 28 + wlane) * 4).toint(), *vals
                    )
                    st_global_v4_u32((dst_base + Int64(56 + wlane) * 4).toint(), *vals)
                    base = Int32(self._rank * 112) + Int32(wlane * 4)
                    logits_smem[base + 0] = _u32_to_f32(vals[0])
                    logits_smem[base + 1] = _u32_to_f32(vals[1])
                    logits_smem[base + 2] = _u32_to_f32(vals[2])
                    logits_smem[base + 3] = _u32_to_f32(vals[3])
            self._barrier_peers(signals, (p4,), row, tid)

            sw_peers = (self._rank ^ 1, self._rank ^ 2, self._rank ^ 3)
            owned = (self._rank, p4)
            for pi in cutlass.range_constexpr(3):
                peer = sw_peers[pi]
                rem_sw = (
                    self._staging_words(staging[peer] + slot_offset) + row_stage_words
                )
                for si in cutlass.range_constexpr(2):
                    if warp == Int32(pi * 2 + si):
                        sr = owned[si]
                        dst_base = rem_sw + Int64(sr * 84 * 4)
                        if cutlass.const_expr(sr == self._rank):
                            _copy_16b(
                                first + Int64(row * 56 + wlane) * 4,
                                dst_base + Int64(wlane) * 4,
                            )
                            if wlane < Int32(24):
                                p2 = wlane + Int32(32)
                                _copy_16b(
                                    first + Int64(row * 56 + p2) * 4,
                                    dst_base + Int64(p2) * 4,
                                )
                            if wlane < second_packs:
                                _copy_16b(
                                    second + Int64(row * 28 + wlane) * 4,
                                    dst_base + Int64(56 + wlane) * 4,
                                )
                        else:
                            src_base = self_stage + Int64(p4 * 84 * 4)
                            v0 = ld_global_cg_v4_u32(
                                (src_base + Int64(wlane) * 4).toint()
                            )
                            st_global_v4_u32((dst_base + Int64(wlane) * 4).toint(), *v0)
                            if wlane < Int32(24):
                                p2 = wlane + Int32(32)
                                v1 = ld_global_cg_v4_u32(
                                    (src_base + Int64(p2) * 4).toint()
                                )
                                st_global_v4_u32((dst_base + Int64(p2) * 4).toint(), *v1)
                            if wlane < second_packs:
                                vals = ld_global_cg_v4_u32(
                                    (src_base + Int64(56 + wlane) * 4).toint()
                                )
                                st_global_v4_u32(
                                    (dst_base + Int64(56 + wlane) * 4).toint(), *vals
                                )
            if warp == Int32(6):
                src_base = self_stage + Int64(p4 * 84 * 4)
                out_row = Int64(row) * Int64(8) + Int64(p4)
                v0 = ld_global_cg_v4_u32((src_base + Int64(wlane) * 4).toint())
                st_global_v4_u32(
                    (out_first + (out_row * 56 + wlane) * 4).toint(), *v0
                )
                if wlane < Int32(24):
                    p2 = wlane + Int32(32)
                    v1 = ld_global_cg_v4_u32((src_base + Int64(p2) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * 56 + p2) * 4).toint(), *v1
                    )
                if wlane < second_packs:
                    vals = ld_global_cg_v4_u32(
                        (src_base + Int64(56 + wlane) * 4).toint()
                    )
                    st_global_v4_u32(
                        (out_second + (out_row * 28 + wlane) * 4).toint(), *vals
                    )
                    base = Int32(p4 * 112) + Int32(wlane * 4)
                    logits_smem[base + 0] = _u32_to_f32(vals[0])
                    logits_smem[base + 1] = _u32_to_f32(vals[1])
                    logits_smem[base + 2] = _u32_to_f32(vals[2])
                    logits_smem[base + 3] = _u32_to_f32(vals[3])
            self._barrier_peers(signals, sw_peers, row, tid)

            rem_peers = tuple(p for p in range(8) if p not in (self._rank, p4))
            for idx in cutlass.range_constexpr(6):
                if warp == Int32(idx):
                    peer = rem_peers[idx]
                    src_base = self_stage + Int64(peer * 84 * 4)
                    out_row = Int64(row) * Int64(8) + Int64(peer)
                    v0 = ld_global_cg_v4_u32((src_base + Int64(wlane) * 4).toint())
                    st_global_v4_u32(
                        (out_first + (out_row * 56 + wlane) * 4).toint(), *v0
                    )
                    if wlane < Int32(24):
                        p2 = wlane + Int32(32)
                        v1 = ld_global_cg_v4_u32((src_base + Int64(p2) * 4).toint())
                        st_global_v4_u32(
                            (out_first + (out_row * 56 + p2) * 4).toint(), *v1
                        )
                    if wlane < second_packs:
                        vals = ld_global_cg_v4_u32(
                            (src_base + Int64(56 + wlane) * 4).toint()
                        )
                        st_global_v4_u32(
                            (out_second + (out_row * 28 + wlane) * 4).toint(), *vals
                        )
                        base = Int32(peer * 112) + Int32(wlane * 4)
                        logits_smem[base + 0] = _u32_to_f32(vals[0])
                        logits_smem[base + 1] = _u32_to_f32(vals[1])
                        logits_smem[base + 2] = _u32_to_f32(vals[2])
                        logits_smem[base + 3] = _u32_to_f32(vals[3])

        cute.arch.sync_threads()
        if cutlass.const_expr(self._device_slot_selection):
            if tid == 0:
                flags = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (flags + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (flags + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(self._rows),
                )

        GatherRouterTopK.select(
            self, logits_smem, unbiased_scores, warp_keys, bias, weights, ids
        )


@program_cache
def compile_row_route_tp8(rank, device_slot_selection, rows=4):
    rows = int(rows)
    launch = GatherRouterTopKTP8(rank, 512, device_slot_selection, rows=rows)
    words = _u32_ptr(16, align=4)
    raw = compile_kernel(
        launch,
        words,
        words,
        words,
        words,
        _f32_ptr(16),
        _f32_ptr(16),
        _i32_ptr(16),
        (_u8_ptr(16),) * 8,
        (words,) * 8,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.row_gather_kimi_topk_tp8",
            1,
            (rank, device_slot_selection, rows, launch._mode),
            labels=("rank", "device_slot_selection", "rows", "mode"),
        ),
    )

    def run(down, router, bias, latent, logits, weights, ids, stages, signals, delta):
        raw(
            _u32_ptr(down, align=4),
            _u32_ptr(router, align=4),
            _u32_ptr(latent, align=4),
            _u32_ptr(logits, align=4),
            _f32_ptr(bias),
            _f32_ptr(weights),
            _i32_ptr(ids),
            tuple(_u8_ptr(v) for v in stages),
            tuple(_u32_ptr(v, align=4) for v in signals),
            delta,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


@program_cache
def compile_row_route(rank, device_slot_selection, rows=4):
    rows = int(rows)
    launch = GatherRouterTopK(16, rank, 512, device_slot_selection, rows=rows)
    words = _u32_ptr(16, align=4)
    raw = compile_kernel(
        launch,
        words,
        words,
        words,
        words,
        _f32_ptr(16),
        _f32_ptr(16),
        _i32_ptr(16),
        (_u8_ptr(16),) * 16,
        (words,) * 16,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.row_gather_kimi_topk",
            10,
            (rank, device_slot_selection, rows),
            labels=("rank", "device_slot_selection", "rows"),
        ),
    )

    def run(down, router, bias, latent, logits, weights, ids, stages, signals, delta):
        raw(
            _u32_ptr(down, align=4),
            _u32_ptr(router, align=4),
            _u32_ptr(latent, align=4),
            _u32_ptr(logits, align=4),
            _f32_ptr(bias),
            _f32_ptr(weights),
            _i32_ptr(ids),
            tuple(_u8_ptr(v) for v in stages),
            tuple(_u32_ptr(v, align=4) for v in signals),
            delta,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


def compile_four_row_route(
    rank, device_slot_selection, fused, threads, rows=4, world_size=16
):
    """Compile the complete multi-row exchange, including selection and its visible logits."""
    rows = int(rows)
    world_size = int(world_size)
    if fused:
        if threads != 512:
            raise ValueError("one warp per remote rank requires a 512-thread CTA")
        if world_size == 8:
            return compile_row_route_tp8(rank, device_slot_selection, rows=rows)
        return compile_row_route(rank, device_slot_selection, rows=rows)
    from ._dcp_a2a_cute import _get_compiled_all_gather_pair, _get_compiled_kimi_topk16

    gather = _get_compiled_all_gather_pair(
        world_size, rank, threads, device_slot_selection, False
    )
    select = _get_compiled_kimi_topk16(256)
    first_packs = 3584 * 2 // (16 * world_size)
    second_packs = 896 * 4 // (16 * world_size)

    def run(down, router, bias, latent, logits, weights, ids, stages, signals, delta):
        gather(
            down,
            router,
            router,
            latent,
            logits,
            logits,
            stages,
            signals,
            rows,
            first_packs,
            second_packs,
            delta,
        )
        select(logits, bias, weights, ids, rows)

    return attach_programs(run, gather, select)
