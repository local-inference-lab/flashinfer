"""Neighbor-only byte gather using the DCP channel's banks and paired flags."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _MAX_BLOCKS,
    _a2a_graph_epoch_arrive,
    _copy_16b,
)
from ._dcp_cute_common import (
    _FLAG_STRIDE,
    _MAX_RANKS,
    _load_relaxed_sys_u32,
    _membar_sys,
    _store_relaxed_sys_u32,
)
from ._joint_cluster_cute import _threadfence_sys
from ._runtime_gather_cute import SelectedGather


class RingGather(SelectedGather):
    def __init__(self, world, rank, row_words, threads, blocks, chunks):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        if world not in (8, 16) or threads != 128 or blocks != 8:
            raise ValueError(
                "ring gather requires DCP8 or DCP16 with eight 128-thread CTAs"
            )
        self.next = (rank + 1) % world
        self.previous = (rank - 1) % world

    @cute.jit
    def _flag(self, signal, sender, block, value):
        slot = Int64(value % Uint32(2))
        index = (
            Int64(_MAX_BLOCKS * _MAX_RANKS)
            + (
                (slot * Int64(_MAX_BLOCKS) + Int64(block))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            + Int64(sender * _FLAG_STRIDE)
        )
        return signal + index

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
            remote = self._staging_words(stages[self.next] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(block) * Int32(self._warps_per_block) + Int32(tid) // Int32(
                32
            )
            step = Int32(8 * self._warps_per_block)
            packs = Int32(self.row_words // 4)
            row_tiles = (packs + Int32(31)) // Int32(32)
            tiles = rows * row_tiles
            incoming, outgoing = Uint32(0), Uint32(0)
            for phase in cutlass.range_constexpr(self._world_size - 1):
                sending = (self._rank - phase) % self._world_size
                receiving = (self._rank - phase - 1) % self._world_size
                destination = sending if sending < self.next else sending - 1
                origin = receiving if receiving < self._rank else receiving - 1
                tile = first
                while tile < tiles:
                    row = tile // row_tiles
                    pack = (tile - row * row_tiles) * Int32(32) + lane
                    if pack < packs:
                        out_pack = (
                            Int64(row) * Int64(self._world_size) + Int64(sending)
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
                            # Acknowledged receive staging is never read again.
                            # Relays consume the completed local output segment.
                            words = ld_global_v4_u32(
                                (output + out_pack * Int64(4)).toint()
                            )
                        destination_pack = (
                            Int64(row) * Int64(self._world_size - 1)
                            + Int64(destination)
                        ) * Int64(packs) + Int64(pack)
                        st_global_v4_u32(
                            (remote + destination_pack * Int64(4)).toint(), *words
                        )
                    tile += step
                # Every CTA writer releases its complete segment set before the
                # control thread publishes ready to the outgoing neighbor.
                _threadfence_sys()
                cute.arch.sync_threads()
                if Int32(tid) == Int32(0):
                    own = signals[self._rank] + Int64(block) * Int64(16)
                    outgoing = Uint32(
                        cute.arch.load(own + Int64(self.next), Uint32)
                    ) + Uint32(1)
                    incoming = Uint32(
                        cute.arch.load(own + Int64(self.previous), Uint32)
                    ) + Uint32(1)
                    cute.arch.store(own + Int64(self.next), outgoing)
                    cute.arch.store(own + Int64(self.previous), incoming)
                    _store_relaxed_sys_u32(
                        self._flag(signals[self.next], self._rank, block, outgoing),
                        outgoing,
                    )
                    ready = self._flag(
                        signals[self._rank], self.previous, block, incoming
                    )
                    observed = _load_relaxed_sys_u32(ready)
                    while observed != incoming:
                        observed = _load_relaxed_sys_u32(ready)
                    _membar_sys()
                cute.arch.sync_threads()
                tile = first
                while tile < tiles:
                    row = tile // row_tiles
                    pack = (tile - row * row_tiles) * Int32(32) + lane
                    if pack < packs:
                        received_pack = (
                            Int64(row) * Int64(self._world_size - 1) + Int64(origin)
                        ) * Int64(packs) + Int64(pack)
                        out_pack = (
                            Int64(row) * Int64(self._world_size) + Int64(receiving)
                        ) * Int64(packs) + Int64(pack)
                        _copy_16b(
                            local + received_pack * Int64(4),
                            output + out_pack * Int64(4),
                        )
                    tile += step
                # Consume incoming staging into final output before acknowledging
                # it. Waiting for downstream ack bounds each flag to one live hop.
                _threadfence_sys()
                cute.arch.sync_threads()
                if Int32(tid) == Int32(0):
                    _store_relaxed_sys_u32(
                        self._flag(signals[self.previous], self._rank, block, incoming),
                        incoming,
                    )
                    acknowledged = self._flag(
                        signals[self._rank], self.next, block, outgoing
                    )
                    observed = _load_relaxed_sys_u32(acknowledged)
                    while observed != outgoing:
                        observed = _load_relaxed_sys_u32(acknowledged)
                cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )
