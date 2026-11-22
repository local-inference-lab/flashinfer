"""Topology-aware XOR-staggered 1-to-1 posted-push PCIe DCP4 column gather."""

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
    _load_acquire_sys_u32,
    _store_relaxed_sys_u32,
    _threadfence_sys,
)
from ._runtime_gather_cute import SelectedGather


class SwitchPushGather(SelectedGather):
    """One-hop XOR-staggered 1-to-1 posted push across four GPUs with overlapped local copy."""

    def __init__(self, world, rank, row_words, threads, blocks, chunks):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        if world != 4 or threads not in (128, 256, 512) or blocks not in (8, 16):
            raise ValueError(
                "switch push gather requires DCP4 with 8 or 16 128-, 256-, or 512-thread CTAs"
            )

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

    @cute.jit
    def _group4_arrive(self, signals, block, tid):
        cute.arch.sync_threads()
        p0, p1, p2 = (self._rank ^ 1, self._rank ^ 2, self._rank ^ 3)
        g0, g1, g2 = Uint32(0), Uint32(0), Uint32(0)
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            own0 = signals[self._rank] + Int64(block) * Int64(_MAX_RANKS) + Int64(p0)
            g0 = Uint32(cute.arch.load(own0, Uint32)) + Uint32(1)
            cute.arch.store(own0, g0)
            _store_relaxed_sys_u32(self._flag(signals[p0], self._rank, block, g0), g0)

            own1 = signals[self._rank] + Int64(block) * Int64(_MAX_RANKS) + Int64(p1)
            g1 = Uint32(cute.arch.load(own1, Uint32)) + Uint32(1)
            cute.arch.store(own1, g1)
            _store_relaxed_sys_u32(self._flag(signals[p1], self._rank, block, g1), g1)

            own2 = signals[self._rank] + Int64(block) * Int64(_MAX_RANKS) + Int64(p2)
            g2 = Uint32(cute.arch.load(own2, Uint32)) + Uint32(1)
            cute.arch.store(own2, g2)
            _store_relaxed_sys_u32(self._flag(signals[p2], self._rank, block, g2), g2)
        return g0, g1, g2

    @cute.jit
    def _group4_wait(self, signals, block, tid, g0, g1, g2) -> None:
        p0, p1, p2 = (self._rank ^ 1, self._rank ^ 2, self._rank ^ 3)
        if Int32(tid) == Int32(0):
            r0 = self._flag(signals[self._rank], p0, block, g0)
            o0 = _load_acquire_sys_u32(r0)
            while o0 != g0:
                o0 = _load_acquire_sys_u32(r0)

            r1 = self._flag(signals[self._rank], p1, block, g1)
            o1 = _load_acquire_sys_u32(r1)
            while o1 != g1:
                o1 = _load_acquire_sys_u32(r1)

            r2 = self._flag(signals[self._rank], p2, block, g2)
            o2 = _load_acquire_sys_u32(r2)
            while o2 != g2:
                o2 = _load_acquire_sys_u32(r2)
        cute.arch.sync_threads()

    @cute.kernel
    def selected(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        active = Int32(profile & Uint32(0xFFFF))
        if Int32(block) < active:
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            local = self._staging_words(stages[self._rank] + bank)
            first = Int32(block) * Int32(self._threads) + Int32(tid)
            step = active * Int32(self._threads)
            packs = Int32(self.row_words // 4)
            total = rows * packs

            # XOR-staggered 1-to-1 posted push: conflict-free full-duplex switch matching.
            for d in cutlass.range_constexpr(1, 4):
                peer = self._rank ^ d
                dest_slot = d - 1
                remote = self._staging_words(stages[peer] + bank)
                index = first
                while index < total:
                    row = index // packs
                    pack = index - row * packs
                    words = ld_global_v4_u32((source + Int64(index) * Int64(4)).toint())
                    dst_pack = (
                        Int64(row) * Int64(3) + Int64(dest_slot)
                    ) * Int64(packs) + Int64(pack)
                    st_global_v4_u32((remote + dst_pack * Int64(4)).toint(), *words)
                    index += step

            # Post remote arrival flags immediately, then overlap own-shard local copy while
            # remote posted writes traverse the PCIe switch.
            g0, g1, g2 = self._group4_arrive(signals, block, tid)

            index = first
            while index < total:
                row = index // packs
                pack = index - row * packs
                words = ld_global_v4_u32((source + Int64(index) * Int64(4)).toint())
                out_pack = (
                    Int64(row) * Int64(4) + Int64(self._rank)
                ) * Int64(packs) + Int64(pack)
                st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
                index += step

            self._group4_wait(signals, block, tid, g0, g1, g2)

            for d in cutlass.range_constexpr(1, 4):
                peer = self._rank ^ d
                origin = d - 1
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

            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(active),
                )
