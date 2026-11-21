"""Topology-aware staggered 1-to-1 posted-push PCIe DCP4 column gather."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.intrinsics.memory import (
    ld_global_cg_v4_u32,
    ld_global_v4_u32,
    st_global_v4_u32,
)
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _MAX_BLOCKS,
    _a2a_graph_epoch_arrive,
)
from ._dcp_cute_common import (
    _FLAG_STRIDE,
    _MAX_RANKS,
    _load_relaxed_sys_u32,
    _store_relaxed_sys_u32,
    _threadfence_sys,
)
from ._runtime_gather_cute import SelectedGather


class SwitchPushGather(SelectedGather):
    """One-hop outer-staggered 1-to-1 posted push across four GPUs."""

    def __init__(self, world, rank, row_words, threads, blocks, chunks):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        if world != 4 or threads not in (128, 256, 512) or blocks != 8:
            raise ValueError(
                "switch push gather requires DCP4 with eight 128-, 256-, or 512-thread CTAs"
            )

    @cute.jit
    def _group4_barrier(self, signals, block, tid) -> None:
        if Int32(tid) < Int32(4):
            if Int32(tid) != Int32(self._rank):
                peer_rank = Int32(0)
                peer_signal_address = Int64(0)
                for coord in cutlass.range_constexpr(4):
                    if cutlass.const_expr(coord != self._rank):
                        if Int32(tid) == Int32(coord):
                            peer_rank = Int32(coord)
                            peer_signal_address = Int64(signals[coord].toint())
                peer_signal = cute.make_ptr(
                    Uint32,
                    peer_signal_address,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                self_signal = signals[self._rank]
                self_ptr = (
                    self_signal + Int64(block) * Int64(_MAX_RANKS) + Int64(peer_rank)
                )
                value = Uint32(cute.arch.load(self_ptr, Uint32)) + Uint32(1)
                cute.arch.store(self_ptr, value)

                slot = Int64(value % Uint32(2))
                common_index = Int64(_MAX_BLOCKS * _MAX_RANKS) + (
                    (slot * Int64(_MAX_BLOCKS) + Int64(block))
                    * Int64(_MAX_RANKS * _FLAG_STRIDE)
                )
                peer_index = common_index + Int64(self._rank * _FLAG_STRIDE)
                mine_index = common_index + Int64(peer_rank) * Int64(_FLAG_STRIDE)
                _store_relaxed_sys_u32(peer_signal + peer_index, value)
                observed = _load_relaxed_sys_u32(self_signal + mine_index)
                while observed != value:
                    observed = _load_relaxed_sys_u32(self_signal + mine_index)

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
            first = Int32(block) * Int32(self._threads) + Int32(tid)
            step = Int32(8 * self._threads)
            packs = Int32(self.row_words // 4)
            total = rows * packs

            # Outer-staggered 1-to-1 push: each pass writes contiguously to ONE
            # remote peer (d=1 also copies own shard to local output).
            for d in cutlass.range_constexpr(1, 4):
                peer = (self._rank + d) % 4
                dest_slot = self._rank if self._rank < peer else self._rank - 1
                remote = self._staging_words(stages[peer] + bank)
                index = first
                while index < total:
                    row = index // packs
                    pack = index - row * packs
                    words = ld_global_v4_u32((source + Int64(index) * Int64(4)).toint())
                    if cutlass.const_expr(d == 1):
                        out_pack = (
                            Int64(row) * Int64(4) + Int64(self._rank)
                        ) * Int64(packs) + Int64(pack)
                        st_global_v4_u32(
                            (output + out_pack * Int64(4)).toint(), *words
                        )
                    dst_pack = (
                        Int64(row) * Int64(3) + Int64(dest_slot)
                    ) * Int64(packs) + Int64(pack)
                    st_global_v4_u32((remote + dst_pack * Int64(4)).toint(), *words)
                    index += step

            _threadfence_sys()
            cute.arch.sync_threads()
            self._group4_barrier(signals, block, tid)
            cute.arch.sync_threads()

            for d in cutlass.range_constexpr(1, 4):
                peer = (self._rank + d) % 4
                origin = peer if peer < self._rank else peer - 1
                index = first
                while index < total:
                    row = index // packs
                    pack = index - row * packs
                    recv_pack = (
                        Int64(row) * Int64(3) + Int64(origin)
                    ) * Int64(packs) + Int64(pack)
                    out_pack = (
                        Int64(row) * Int64(4) + Int64(peer)
                    ) * Int64(packs) + Int64(pack)
                    words = ld_global_cg_v4_u32(
                        (local + recv_pack * Int64(4)).toint()
                    )
                    st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
                    index += step

            cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(8),
                )
