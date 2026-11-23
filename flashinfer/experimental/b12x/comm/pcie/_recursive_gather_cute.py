"""Byte-exact power-of-two DCP gather through pairwise doubling exchanges."""

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
    _copy_16b,
)
from ._dcp_cute_common import (
    _FLAG_STRIDE,
    _MAX_RANKS,
    _load_acquire_sys_u32,
    _store_relaxed_sys_u32,
    _threadfence_sys,
)
from ._runtime_gather_cute import SelectedGather


@cute.jit
def _lr4_flag(signal, sender, block, slot, word_off: cutlass.Constexpr):
    index = (
        Int64(_MAX_BLOCKS * _MAX_RANKS)
        + (
            (Int64(slot) * Int64(_MAX_BLOCKS) + Int64(block))
            * Int64(_MAX_RANKS * _FLAG_STRIDE)
        )
        + Int64(sender * _FLAG_STRIDE + word_off)
    )
    return signal + index


@cute.jit
def _run_lane_ring4_sw4_tp16(
    op, source, output, stages, signals, rows, stride, block, tid
):
    """4-hop TP16 All-Gather with flat per-sender staging and XOR intra-switch order.

    Uses isolated _FLAG_STRIDE word offsets (+4, +5, +6, +7) on graph-epoch parity slot
    with monotonic tag = generation + 1, making it 100% safe alongside RingGather and
    RecursiveGather on the same DcpAllToAll channel across unbounded CUDA Graph replays.
    """
    island = op._rank // 4
    lane = op._rank % 4
    next_x = ((island + 1) % 4) * 4 + lane
    prev_x = ((island - 1) % 4) * 4 + lane
    lpeers = tuple(island * 4 + (lane ^ s) for s in (1, 2, 3))

    generation = ld_relaxed_gpu_u32(
        (signals[op._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
    )
    slot = generation % Uint32(2)
    tag = generation + Uint32(1)
    bank = Int64(slot) * stride

    local_stage = op._staging_words(stages[op._rank] + bank)
    next_x_stage = op._staging_words(stages[next_x] + bank)
    lp0_stage = op._staging_words(stages[lpeers[0]] + bank)
    lp1_stage = op._staging_words(stages[lpeers[1]] + bank)
    lp2_stage = op._staging_words(stages[lpeers[2]] + bank)

    packs = Int32(op.row_words // 4)
    total = rows * packs
    total64 = Int64(total)
    first = Int32(block) * Int32(op._threads) + Int32(tid)
    step = Int32(op.blocks * op._threads)

    dst_nx = op._rank if op._rank < next_x else op._rank - 1
    dst_l0 = op._rank if op._rank < lpeers[0] else op._rank - 1
    dst_l1 = op._rank if op._rank < lpeers[1] else op._rank - 1
    dst_l2 = op._rank if op._rank < lpeers[2] else op._rank - 1
    nx_base = next_x_stage + Int64(dst_nx) * total64 * Int64(4)
    l0_base = lp0_stage + Int64(dst_l0) * total64 * Int64(4)
    l1_base = lp1_stage + Int64(dst_l1) * total64 * Int64(4)
    l2_base = lp2_stage + Int64(dst_l2) * total64 * Int64(4)
    idx = first
    while idx < total:
        row = idx // packs
        pack = idx - row * packs
        off = Int64(idx) * Int64(4)
        words = ld_global_v4_u32((source + off).toint())
        out_p = (Int64(row) * Int64(16) + Int64(op._rank)) * Int64(packs) + Int64(pack)
        st_global_v4_u32((output + out_p * Int64(4)).toint(), *words)
        st_global_v4_u32((nx_base + off).toint(), *words)
        st_global_v4_u32((l0_base + off).toint(), *words)
        st_global_v4_u32((l1_base + off).toint(), *words)
        st_global_v4_u32((l2_base + off).toint(), *words)
        idx += step

    cute.arch.sync_threads()
    if Int32(tid) == Int32(0):
        _threadfence_sys()
        _store_relaxed_sys_u32(
            _lr4_flag(signals[next_x], op._rank, block, slot, 4), tag
        )

    for hop in cutlass.range_constexpr(3):
        recv_island = (island - hop - 1) % 4
        recv_rank = recv_island * 4 + lane
        orig_slot = recv_rank if recv_rank < op._rank else recv_rank - 1
        if Int32(tid) == Int32(0):
            f = _lr4_flag(signals[op._rank], prev_x, block, slot, 4 + hop)
            observed = _load_acquire_sys_u32(f)
            while Int32(observed - tag) < Int32(0):
                observed = _load_acquire_sys_u32(f)
        cute.arch.sync_threads()

        d_nx = recv_rank if recv_rank < next_x else recv_rank - 1
        d_l0 = recv_rank if recv_rank < lpeers[0] else recv_rank - 1
        d_l1 = recv_rank if recv_rank < lpeers[1] else recv_rank - 1
        d_l2 = recv_rank if recv_rank < lpeers[2] else recv_rank - 1
        in_base = local_stage + Int64(orig_slot) * total64 * Int64(4)
        h_nx = next_x_stage + Int64(d_nx) * total64 * Int64(4)
        h_l0 = lp0_stage + Int64(d_l0) * total64 * Int64(4)
        h_l1 = lp1_stage + Int64(d_l1) * total64 * Int64(4)
        h_l2 = lp2_stage + Int64(d_l2) * total64 * Int64(4)
        idx = first
        while idx < total:
            row = idx // packs
            pack = idx - row * packs
            off = Int64(idx) * Int64(4)
            words = ld_global_cg_v4_u32((in_base + off).toint())
            out_p = (Int64(row) * Int64(16) + Int64(recv_rank)) * Int64(packs) + Int64(
                pack
            )
            st_global_v4_u32((output + out_p * Int64(4)).toint(), *words)
            if cutlass.const_expr(hop < 2):
                st_global_v4_u32((h_nx + off).toint(), *words)
            st_global_v4_u32((h_l0 + off).toint(), *words)
            st_global_v4_u32((h_l1 + off).toint(), *words)
            st_global_v4_u32((h_l2 + off).toint(), *words)
            idx += step

        if cutlass.const_expr(hop < 2):
            cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _threadfence_sys()
                _store_relaxed_sys_u32(
                    _lr4_flag(signals[next_x], op._rank, block, slot, 5 + hop), tag
                )

    cute.arch.sync_threads()
    if Int32(tid) == Int32(0):
        _threadfence_sys()
        for p_i in cutlass.range_constexpr(3):
            _store_relaxed_sys_u32(
                _lr4_flag(signals[lpeers[p_i]], op._rank, block, slot, 7), tag
            )
        for p_i in cutlass.range_constexpr(3):
            f = _lr4_flag(signals[op._rank], lpeers[p_i], block, slot, 7)
            observed = _load_acquire_sys_u32(f)
            while Int32(observed - tag) < Int32(0):
                observed = _load_acquire_sys_u32(f)
    cute.arch.sync_threads()

    for p_i in cutlass.range_constexpr(3):
        pl = lpeers[p_i] % 4
        for isl in cutlass.range_constexpr(4):
            src_r = isl * 4 + pl
            s_slot = src_r if src_r < op._rank else src_r - 1
            in_base = local_stage + Int64(s_slot) * total64 * Int64(4)
            idx = first
            while idx < total:
                row = idx // packs
                pack = idx - row * packs
                words = ld_global_cg_v4_u32((in_base + Int64(idx) * Int64(4)).toint())
                out_p = (Int64(row) * Int64(16) + Int64(src_r)) * Int64(packs) + Int64(
                    pack
                )
                st_global_v4_u32((output + out_p * Int64(4)).toint(), *words)
                idx += step

    if Int32(tid) == Int32(0):
        _a2a_graph_epoch_arrive(
            (signals[op._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
            (signals[op._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
            Uint32(op.blocks),
        )


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
        outer_first=False,
        pix_last=False,
    ):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        self.acknowledge_consumption = acknowledge_consumption
        self.outer_first = outer_first
        self.pix_last = pix_last
        if outer_first and world != 16:
            raise ValueError("outer-first recursive doubling requires sixteen ranks")
        if pix_last:
            valid_blocks = (8, 16) if world == 8 else (8,)
            if (
                world not in (4, 8, 16)
                or threads not in (128, 256, 512)
                or blocks not in valid_blocks
            ):
                raise ValueError(
                    "pix-last recursive gather requires DCP4/8/16 with 8 (or 16 on DCP8) 128/256/512-thread CTAs"
                )
        elif world not in (8, 16) or threads != 128 or blocks != 8:
            raise ValueError(
                "recursive gather requires DCP8 or DCP16 with eight 128-thread CTAs"
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
    def _pair_barrier(self, signals, peer: cutlass.Constexpr, block, tid):
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            own = signals[self._rank] + Int64(block * 16 + peer)
            generation = Uint32(cute.arch.load(own, Uint32)) + Uint32(1)
            cute.arch.store(own, generation)
            _store_relaxed_sys_u32(
                self._flag(signals[peer], self._rank, block, generation), generation
            )
            ready = self._flag(signals[self._rank], peer, block, generation)
            observed = _load_acquire_sys_u32(ready)
            while observed != generation:
                observed = _load_acquire_sys_u32(ready)
        cute.arch.sync_threads()

    @cute.jit
    def _signal_arrive(self, signals, peer: cutlass.Constexpr, block, tid):
        cute.arch.sync_threads()
        gen = Uint32(0)
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            own = signals[self._rank] + Int64(block * 16 + peer)
            gen = Uint32(cute.arch.load(own, Uint32)) + Uint32(1)
            cute.arch.store(own, gen)
            _store_relaxed_sys_u32(
                self._flag(signals[peer], self._rank, block, gen), gen
            )
        return gen

    @cute.jit
    def _signal_wait(self, signals, peer: cutlass.Constexpr, block, tid, gen):
        if Int32(tid) == Int32(0):
            ready = self._flag(signals[self._rank], peer, block, gen)
            observed = _load_acquire_sys_u32(ready)
            while observed != gen:
                observed = _load_acquire_sys_u32(ready)
        cute.arch.sync_threads()

    @cute.jit
    def _switch4_barrier(self, signals, block, tid):
        cute.arch.sync_threads()
        p0, p1, p2 = (self._rank ^ 1, self._rank ^ 2, self._rank ^ 3)
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            own0 = signals[self._rank] + Int64(block * 16 + p0)
            g0 = Uint32(cute.arch.load(own0, Uint32)) + Uint32(1)
            cute.arch.store(own0, g0)
            _store_relaxed_sys_u32(self._flag(signals[p0], self._rank, block, g0), g0)

            own1 = signals[self._rank] + Int64(block * 16 + p1)
            g1 = Uint32(cute.arch.load(own1, Uint32)) + Uint32(1)
            cute.arch.store(own1, g1)
            _store_relaxed_sys_u32(self._flag(signals[p1], self._rank, block, g1), g1)

            own2 = signals[self._rank] + Int64(block * 16 + p2)
            g2 = Uint32(cute.arch.load(own2, Uint32)) + Uint32(1)
            cute.arch.store(own2, g2)
            _store_relaxed_sys_u32(self._flag(signals[p2], self._rank, block, g2), g2)

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
        if cutlass.const_expr(self._world_size == 8 and self.pix_last):
            active = Int32(profile & Uint32(0xFFFF))
            if Int32(block) < active:
                generation = ld_relaxed_gpu_u32(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
                )
                bank = Int64(generation % Uint32(2)) * stride
                local = self._staging_words(stages[self._rank] + bank)
                packs = Int32(self.row_words // 4)
                total = rows * packs
                total64 = Int64(total)
                first = Int32(block) * Int32(self._threads) + Int32(tid)
                step = active * Int32(self._threads)
                # Overlapped 2-barrier Pair-2 (r^4 cross-switch) x Switch-4 (r^1,r^2,r^3
                # 54.8 GB/s physical switch) with flat per-sender staging & single source read.
                partner = self._rank ^ 4
                sw0, sw1, sw2 = self._rank ^ 1, self._rank ^ 2, self._rank ^ 3
                rem_p4 = self._staging_words(stages[partner] + bank)
                rem_sw0 = self._staging_words(stages[sw0] + bank)
                rem_sw1 = self._staging_words(stages[sw1] + bank)
                rem_sw2 = self._staging_words(stages[sw2] + bank)
                s_p4 = self._rank if self._rank < partner else self._rank - 1
                s_sw0 = self._rank if self._rank < sw0 else self._rank - 1
                s_sw1 = self._rank if self._rank < sw1 else self._rank - 1
                s_sw2 = self._rank if self._rank < sw2 else self._rank - 1
                p4_base = rem_p4 + Int64(s_p4) * total64 * Int64(4)
                sw0_base = rem_sw0 + Int64(s_sw0) * total64 * Int64(4)
                sw1_base = rem_sw1 + Int64(s_sw1) * total64 * Int64(4)
                sw2_base = rem_sw2 + Int64(s_sw2) * total64 * Int64(4)
                idx = first
                while idx < total:
                    row = idx // packs
                    pack = idx - row * packs
                    off = Int64(idx) * Int64(4)
                    words = ld_global_v4_u32((source + off).toint())
                    out_p = (
                        Int64(row) * Int64(8) + Int64(self._rank)
                    ) * Int64(packs) + Int64(pack)
                    st_global_v4_u32((output + out_p * Int64(4)).toint(), *words)
                    st_global_v4_u32((p4_base + off).toint(), *words)
                    st_global_v4_u32((sw0_base + off).toint(), *words)
                    st_global_v4_u32((sw1_base + off).toint(), *words)
                    st_global_v4_u32((sw2_base + off).toint(), *words)
                    idx += step
                self._pair_barrier(signals, partner, block, tid)

                orig_p4 = partner if partner < self._rank else partner - 1
                in_p4 = local + Int64(orig_p4) * total64 * Int64(4)
                ps_sw0 = partner if partner < sw0 else partner - 1
                ps_sw1 = partner if partner < sw1 else partner - 1
                ps_sw2 = partner if partner < sw2 else partner - 1
                p4_sw0 = rem_sw0 + Int64(ps_sw0) * total64 * Int64(4)
                p4_sw1 = rem_sw1 + Int64(ps_sw1) * total64 * Int64(4)
                p4_sw2 = rem_sw2 + Int64(ps_sw2) * total64 * Int64(4)
                idx = first
                while idx < total:
                    row = idx // packs
                    pack = idx - row * packs
                    off = Int64(idx) * Int64(4)
                    w_rem = ld_global_cg_v4_u32((in_p4 + off).toint())
                    out_p = (
                        Int64(row) * Int64(8) + Int64(partner)
                    ) * Int64(packs) + Int64(pack)
                    st_global_v4_u32((output + out_p * Int64(4)).toint(), *w_rem)
                    st_global_v4_u32((p4_sw0 + off).toint(), *w_rem)
                    st_global_v4_u32((p4_sw1 + off).toint(), *w_rem)
                    st_global_v4_u32((p4_sw2 + off).toint(), *w_rem)
                    idx += step
                self._switch4_barrier(signals, block, tid)

                others = tuple(
                    r for r in range(8) if r != self._rank and r != partner
                )
                for oi in cutlass.range_constexpr(6):
                    src_r = others[oi]
                    slot_r = src_r if src_r < self._rank else src_r - 1
                    in_base = local + Int64(slot_r) * total64 * Int64(4)
                    idx = first
                    while idx < total:
                        row = idx // packs
                        pack = idx - row * packs
                        off = Int64(idx) * Int64(4)
                        words = ld_global_cg_v4_u32((in_base + off).toint())
                        out_p = (
                            Int64(row) * Int64(8) + Int64(src_r)
                        ) * Int64(packs) + Int64(pack)
                        st_global_v4_u32((output + out_p * Int64(4)).toint(), *words)
                        idx += step
                if Int32(tid) == Int32(0):
                    _a2a_graph_epoch_arrive(
                        (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                        (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                        Uint32(active),
                    )
        elif cutlass.const_expr(self._world_size == 16):
            if profile == Uint32(self.blocks):
                _run_lane_ring4_sw4_tp16(
                    self, source, output, stages, signals, rows, stride, block, tid
                )
        elif profile == Uint32(self.blocks):
            generation = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            bank = Int64(generation % Uint32(2)) * stride
            local = self._staging_words(stages[self._rank] + bank)
            lane = Int32(tid) % Int32(32)
            first = Int32(block) * Int32(self._warps_per_block) + Int32(tid) // Int32(
                32
            )
            step = Int32(self.blocks * self._warps_per_block)
            packs = Int32(self.row_words // 4)
            row_tiles = (packs + Int32(31)) // Int32(32)
            tiles = rows * row_tiles
            for phase in cutlass.range_constexpr(self._world_size.bit_length() - 1):
                    if cutlass.const_expr(self.pix_last):
                        # Cross EPYC root-port hops first; keep the largest final doubling
                        # (50% of all bytes) strictly on the intra-switch PIX link (r^1).
                        if cutlass.const_expr(self._world_size == 16):
                            distance = (4, 8, 2, 1)[phase]
                        elif cutlass.const_expr(self._world_size == 8):
                            distance = (4, 2, 1)[phase]
                        else:
                            distance = (2, 1)[phase]
                        group = 1 << phase
                    elif cutlass.const_expr(self._world_size == 8):
                        # Cross the two four-GPU switches first. Later exchanges
                        # stay within each switch while forwarding both sets of shards.
                        distance = (4, 1, 2)[phase]
                        group = (1, 2, 4)[phase]
                    elif cutlass.const_expr(self.outer_first):
                        # For four GPUs per switch, the first two rounds cross switches.
                        distance = (4, 8, 1, 2)[phase]
                        group = 1 << phase
                    else:
                        distance = 1 << phase
                        group = distance
                    peer = self._rank ^ distance
                    remote = self._staging_words(stages[peer] + bank)
                    # A CTA owns the same row/column tiles in every rank segment.
                    # It only forwards bytes that it completed in an earlier phase.
                    for segment in cutlass.range_constexpr(group):
                        if cutlass.const_expr(self.pix_last):
                            if cutlass.const_expr(self._world_size == 16):
                                mask = ((segment & 3) << 2) ^ ((segment & 4) >> 1)
                            elif cutlass.const_expr(self._world_size == 8):
                                mask = ((segment & 1) << 2) ^ (segment & 2)
                            else:
                                mask = (segment & 1) << 1
                            sending = self._rank ^ mask
                        elif cutlass.const_expr(self._world_size == 8):
                            if cutlass.const_expr(phase == 0):
                                sending = self._rank
                            elif cutlass.const_expr(phase == 1):
                                sending = self._rank if segment == 0 else self._rank ^ 4
                            else:
                                sending = self._rank ^ (segment & 1) ^ ((segment & 2) * 2)
                        elif cutlass.const_expr(self.outer_first):
                            sending = self._rank ^ ((segment & 3) << 2) ^ (segment >> 2)
                        else:
                            sending = (self._rank & ~(distance - 1)) + segment
                        destination = sending if sending < peer else sending - 1
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
                    self._pair_barrier(signals, peer, block, tid)
                    for segment in cutlass.range_constexpr(group):
                        if cutlass.const_expr(self.pix_last):
                            if cutlass.const_expr(self._world_size == 16):
                                mask = ((segment & 3) << 2) ^ ((segment & 4) >> 1)
                            elif cutlass.const_expr(self._world_size == 8):
                                mask = ((segment & 1) << 2) ^ (segment & 2)
                            else:
                                mask = (segment & 1) << 1
                            receiving = peer ^ mask
                        elif cutlass.const_expr(self._world_size == 8):
                            if cutlass.const_expr(phase == 0):
                                receiving = peer
                            elif cutlass.const_expr(phase == 1):
                                receiving = peer if segment == 0 else peer ^ 4
                            else:
                                receiving = peer ^ (segment & 1) ^ ((segment & 2) * 2)
                        elif cutlass.const_expr(self.outer_first):
                            receiving = peer ^ ((segment & 3) << 2) ^ (segment >> 2)
                        else:
                            receiving = (peer & ~(distance - 1)) + segment
                        origin = receiving if receiving < self._rank else receiving - 1
                        tile = first
                        while tile < tiles:
                            row = tile // row_tiles
                            pack = (tile - row * row_tiles) * Int32(32) + lane
                            if pack < packs:
                                incoming_pack = (
                                    Int64(row) * Int64(self._world_size - 1) + Int64(origin)
                                ) * Int64(packs) + Int64(pack)
                                out_pack = (
                                    Int64(row) * Int64(self._world_size) + Int64(receiving)
                                ) * Int64(packs) + Int64(pack)
                                words = ld_global_cg_v4_u32(
                                    (local + incoming_pack * Int64(4)).toint()
                                )
                                st_global_v4_u32(
                                    (output + out_pack * Int64(4)).toint(), *words
                                )
                            tile += step
                    # With consumption acknowledgements, READY and ACK use separate
                    # paired generations. Otherwise the two receive banks protect
                    # staging: completing the intervening collective requires every
                    # rank to enter it, hence finish this kernel before bank reuse.
                    if cutlass.const_expr(self.acknowledge_consumption):
                        self._pair_barrier(signals, peer, block, tid)
                    else:
                        cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _a2a_graph_epoch_arrive(
                    (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(self.blocks),
                )
