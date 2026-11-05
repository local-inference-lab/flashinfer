"""One-destination-at-a-time byte gathers in one resident CTA cluster."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _a2a_graph_epoch_arrive,
    _copy_16b,
    _copy_16b_addr,
)
from ._dcp_cute_common import _membar_sys, block_pair_barrier
from ._joint_cluster_cute import _threadfence_sys
from ._runtime_gather_cute import SelectedGather


class PhasedGather(SelectedGather):
    def __init__(
        self, world, rank, row_words, threads, blocks, chunks, phase_peer_barrier
    ):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        if blocks != 8:
            raise ValueError("phased gather requires one complete eight-CTA cluster")
        self._phase_peer_barrier = phase_peer_barrier

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        selector: cute.Pointer,
        stages: tuple,
        signals: tuple,
        rows: Int32,
        stride: Int64,
        offset: Int64,
        counter: Int64,
        stream: cuda.CUstream,
    ):
        self.phased(
            source, output, selector, stages, signals, rows, stride, offset, counter
        ).launch(
            grid=(8, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(8, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def phased(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        # A partial cluster cannot participate in a cluster barrier. Prepared
        # profiles require eight CTAs; corrupt selectors touch no IPC state.
        if profile == Uint32(8):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            own_stage = self._staging_words(stages[self._rank] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(block) * Int32(self._warps_per_block) + Int32(tid) // Int32(
                32
            )
            step = Int32(8 * self._warps_per_block)
            local_packs = Int32(self.row_words // 4)
            tiles_per_rank = (local_packs + Int32(31)) // Int32(32)
            local_tiles = rows * tiles_per_rank
            for phase in cutlass.range_constexpr(1, self._world_size):
                target = (self._rank + phase) % self._world_size
                incoming = self._rank if self._rank < target else self._rank - 1
                remote = self._staging_words(stages[target] + bank)
                tile = first
                while tile < local_tiles:
                    row = tile // tiles_per_rank
                    pack = (tile - row * tiles_per_rank) * Int32(32) + lane
                    if pack < local_packs:
                        source_pack = Int64(row) * Int64(local_packs) + Int64(pack)
                        staged_pack = (
                            Int64(row) * Int64(self._world_size - 1) + Int64(incoming)
                        ) * Int64(local_packs) + Int64(pack)
                        _copy_16b(
                            source + source_pack * Int64(4),
                            remote + staged_pack * Int64(4),
                        )
                    tile += step
                # Every writer fences before the local cluster completes the
                # phase, including empty tails. A peer CTA can publish completion
                # only after all writers on its rank have reached that boundary.
                _threadfence_sys()
                cute.arch.cluster_arrive()
                cute.arch.cluster_wait()
                if cutlass.const_expr(self._phase_peer_barrier):
                    block_pair_barrier(
                        signals,
                        self_signal=signals[self._rank],
                        rank=self._rank,
                        world_size=self._world_size,
                        max_blocks=64,
                    )
            if cutlass.const_expr(not self._phase_peer_barrier):
                # Source-rank staging spans are disjoint and phases never read
                # received data. One peer barrier after all sends suffices for
                # visibility before assembly; local phase fences limit fan-out.
                block_pair_barrier(
                    signals,
                    self_signal=signals[self._rank],
                    rank=self._rank,
                    world_size=self._world_size,
                    max_blocks=64,
                )
            if Int32(tid) == Int32(0):
                _membar_sys()
            cute.arch.sync_threads()
            tiles_per_row = tiles_per_rank * Int32(self._world_size)
            tile = first
            while tile < rows * tiles_per_row:
                row = tile // tiles_per_row
                row_tile = tile - row * tiles_per_row
                source_rank = row_tile // tiles_per_rank
                rank_tile = row_tile - source_rank * tiles_per_rank
                pack = rank_tile * Int32(32) + lane
                if pack < local_packs:
                    source_pack = Int64(row) * Int64(local_packs) + Int64(pack)
                    output_pack = (
                        Int64(row) * Int64(self._world_size) + Int64(source_rank)
                    ) * Int64(local_packs) + Int64(pack)
                    address = Int64((source + source_pack * Int64(4)).toint())
                    for peer in cutlass.range_constexpr(self._world_size):
                        if source_rank == Int32(peer):
                            if cutlass.const_expr(peer != self._rank):
                                incoming = peer if peer < self._rank else peer - 1
                                staged_pack = (
                                    Int64(row) * Int64(self._world_size - 1)
                                    + Int64(incoming)
                                ) * Int64(local_packs) + Int64(pack)
                                address = Int64(
                                    (own_stage + staged_pack * Int64(4)).toint()
                                )
                    _copy_16b_addr(
                        address, Int64((output + output_pack * Int64(4)).toint())
                    )
                tile += step
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )
