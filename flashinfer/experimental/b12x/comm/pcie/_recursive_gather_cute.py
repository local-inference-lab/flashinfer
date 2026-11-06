"""Byte-exact TP16 gather through four pairwise doubling exchanges."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _a2a_graph_epoch_arrive,
    _copy_16b,
)
from ._dcp_cute_common import (
    _load_relaxed_sys_u32,
    _membar_sys,
    _store_relaxed_sys_u32,
)
from ._joint_cluster_cute import _threadfence_sys
from ._runtime_gather_cute import SelectedGather


class RecursiveGather(SelectedGather):
    def __init__(
        self,
        world,
        rank,
        row_words,
        threads,
        blocks,
        chunks,
        acknowledge_consumption=True,
    ):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        self.acknowledge_consumption = acknowledge_consumption
        if world != 16 or threads != 128 or blocks != 8:
            raise ValueError(
                "recursive gather requires TP16 with eight 128-thread CTAs"
            )

    @cute.jit
    def _flag(self, signal, sender, block, value):
        slot = Int64(value % Uint32(2))
        index = (
            Int64(64 * 16)
            + ((slot * Int64(64) + Int64(block)) * Int64(16 * 32))
            + Int64(sender * 32)
        )
        return signal + index

    @cute.jit
    def _pair_barrier(self, signals, peer: cutlass.Constexpr, block, tid):
        if Int32(tid) == Int32(0):
            own = signals[self._rank] + Int64(block * 16 + peer)
            generation = Uint32(cute.arch.load(own, Uint32)) + Uint32(1)
            cute.arch.store(own, generation)
            _store_relaxed_sys_u32(
                self._flag(signals[peer], self._rank, block, generation), generation
            )
            ready = self._flag(signals[self._rank], peer, block, generation)
            observed = _load_relaxed_sys_u32(ready)
            while observed != generation:
                observed = _load_relaxed_sys_u32(ready)
            _membar_sys()
        cute.arch.sync_threads()

    @cute.kernel
    def selected(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        if profile == Uint32(8):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            local = self._staging_words(stages[self._rank] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(block) * Int32(self._warps_per_block) + Int32(tid) // Int32(
                32
            )
            step = Int32(8 * self._warps_per_block)
            packs = Int32(self.row_words // 4)
            row_tiles = (packs + Int32(31)) // Int32(32)
            tiles = rows * row_tiles
            for phase in cutlass.range_constexpr(4):
                group = 1 << phase
                peer = self._rank ^ group
                own_begin = self._rank & ~(group - 1)
                peer_begin = peer & ~(group - 1)
                remote = self._staging_words(stages[peer] + bank)
                # A CTA owns the same row/column tiles in every rank segment.
                # It only forwards bytes that it completed in an earlier phase.
                for segment in cutlass.range_constexpr(group):
                    sending = own_begin + segment
                    destination = sending if sending < peer else sending - 1
                    tile = first
                    while tile < tiles:
                        row = tile // row_tiles
                        pack = (tile - row * row_tiles) * Int32(32) + lane
                        if pack < packs:
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(sending)
                            ) * Int64(packs) + Int64(pack)
                            if cutlass.const_expr(phase == 0):
                                source_pack = Int64(row) * Int64(packs) + Int64(pack)
                                words = ld_global_v4_u32(
                                    (source + source_pack * Int64(4)).toint()
                                )
                                st_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint(), *words
                                )
                            else:
                                words = ld_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint()
                                )
                            destination_pack = (
                                Int64(row) * Int64(15) + Int64(destination)
                            ) * Int64(packs) + Int64(pack)
                            st_global_v4_u32(
                                (remote + destination_pack * Int64(4)).toint(), *words
                            )
                        tile += step
                _threadfence_sys()
                cute.arch.sync_threads()
                self._pair_barrier(signals, peer, block, tid)
                for segment in cutlass.range_constexpr(group):
                    receiving = peer_begin + segment
                    origin = receiving if receiving < self._rank else receiving - 1
                    tile = first
                    while tile < tiles:
                        row = tile // row_tiles
                        pack = (tile - row * row_tiles) * Int32(32) + lane
                        if pack < packs:
                            incoming_pack = (
                                Int64(row) * Int64(15) + Int64(origin)
                            ) * Int64(packs) + Int64(pack)
                            out_pack = (
                                Int64(row) * Int64(16) + Int64(receiving)
                            ) * Int64(packs) + Int64(pack)
                            _copy_16b(
                                local + incoming_pack * Int64(4),
                                output + out_pack * Int64(4),
                            )
                        tile += step
                # With consumption acknowledgements, READY and ACK use separate
                # paired generations. Otherwise the two receive banks protect
                # staging: completing the intervening collective requires every
                # rank to enter it, hence finish this kernel before bank reuse.
                _threadfence_sys()
                cute.arch.sync_threads()
                if cutlass.const_expr(self.acknowledge_consumption):
                    self._pair_barrier(signals, peer, block, tid)
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )
