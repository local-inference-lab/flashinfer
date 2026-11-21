"""Neighbor-only byte gather using the DCP channel's banks and paired flags."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

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
from ._recursive_gather_cute import _run_lane_ring4_sw4_tp16


@dsl_user_op
def _invalid_count(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "trap;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class RingGather(SelectedGather):
    def __init__(self, world, rank, row_words, threads, blocks, chunks):
        super().__init__(world, rank, row_words, threads, blocks, chunks, "barrier")
        valid_blocks = (8, 16) if world == 16 else (8,)
        if world not in (2, 4, 8, 16) or threads not in (128, 512) or blocks not in valid_blocks:
            raise ValueError(
                "ring gather requires DCP2/4/8/16 with 8 (or 16 on DCP16) 128- or 512-thread CTAs"
            )
        self.next = (rank + 1) % world
        self.previous = (rank - 1) % world

    @cute.jit
    def _flag(self, signal, sender, block, value, ack: cutlass.Constexpr = False):
        slot = Int64(value % Uint32(2))
        index = (
            Int64(_MAX_BLOCKS * _MAX_RANKS)
            + (
                (slot * Int64(_MAX_BLOCKS) + Int64(block))
                * Int64(_MAX_RANKS * _FLAG_STRIDE)
            )
            + Int64(sender * _FLAG_STRIDE)
            + Int64(3 if self._world_size == 2 and ack else 0)
        )
        return signal + index

    @cute.jit
    def _run_ring(self, source, output, stages, signals, rows, stride, block, tid):
        generation = ld_relaxed_gpu_u32(
            (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
        )
        bank = Int64(generation % Uint32(2)) * stride
        local = self._staging_words(stages[self._rank] + bank)
        remote = self._staging_words(stages[self.next] + bank)
        first = Int32(block) * Int32(self._threads) + Int32(tid)
        step = Int32(self.blocks * self._threads)
        packs = Int32(self.row_words // 4)
        total = rows * packs
        destination = self._rank if self._rank < self.next else self._rank - 1
        index = first
        while index < total:
            row = index // packs
            pack = index - row * packs
            out_pack = (
                Int64(row) * Int64(self._world_size) + Int64(self._rank)
            ) * Int64(packs) + Int64(pack)
            destination_pack = (
                Int64(row) * Int64(self._world_size - 1) + Int64(destination)
            ) * Int64(packs) + Int64(pack)
            words = ld_global_v4_u32((source + Int64(index) * Int64(4)).toint())
            st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
            st_global_v4_u32((remote + destination_pack * Int64(4)).toint(), *words)
            index += step
        incoming, outgoing = Uint32(0), Uint32(0)
        for phase in cutlass.range_constexpr(self._world_size - 1):
            receiving = (self._rank - phase - 1) % self._world_size
            origin = receiving if receiving < self._rank else receiving - 1
            if cutlass.const_expr(phase == 0):
                # Source stores have no preceding receive-side release.
                cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                if cutlass.const_expr(phase == 0):
                    _threadfence_sys()
                own = signals[self._rank] + Int64(block) * Int64(16)
                outgoing = Uint32(
                    cute.arch.load(own + Int64(self.next), Uint32)
                ) + Uint32(1)
                incoming = Uint32(
                    cute.arch.load(own + Int64(self.previous), Uint32)
                ) + Uint32(1)
                if cutlass.const_expr(phase >= 2):
                    # Reusing a parity slot requires its earlier notification
                    # to have been consumed by the downstream CTA.
                    retired = outgoing - Uint32(2)
                    acknowledged = self._flag(
                        signals[self._rank], self.next, block, retired
                    )
                    observed = _load_acquire_sys_u32(acknowledged)
                    while observed != retired:
                        observed = _load_acquire_sys_u32(acknowledged)
                cute.arch.store(own + Int64(self.next), outgoing)
                cute.arch.store(own + Int64(self.previous), incoming)
                _store_relaxed_sys_u32(
                    self._flag(signals[self.next], self._rank, block, outgoing),
                    outgoing,
                )
                ready = self._flag(
                    signals[self._rank], self.previous, block, incoming
                )
                observed = _load_acquire_sys_u32(ready)
                while observed != incoming:
                    observed = _load_acquire_sys_u32(ready)
            cute.arch.sync_threads()
            index = first
            while index < total:
                row = index // packs
                pack = index - row * packs
                received_pack = (
                    Int64(row) * Int64(self._world_size - 1) + Int64(origin)
                ) * Int64(packs) + Int64(pack)
                out_pack = (
                    Int64(row) * Int64(self._world_size) + Int64(receiving)
                ) * Int64(packs) + Int64(pack)
                words = ld_global_cg_v4_u32((local + received_pack * Int64(4)).toint())
                st_global_v4_u32((output + out_pack * Int64(4)).toint(), *words)
                if cutlass.const_expr(phase < self._world_size - 2):
                    destination = (
                        receiving if receiving < self.next else receiving - 1
                    )
                    destination_pack = (
                        Int64(row) * Int64(self._world_size - 1)
                        + Int64(destination)
                    ) * Int64(packs) + Int64(pack)
                    st_global_v4_u32(
                        (remote + destination_pack * Int64(4)).toint(), *words
                    )
                index += step
            # Distinct origin slots retain forwarding bytes while two ready
            # notifications overlap. ACK permits matching parity slot reuse.
            cute.arch.sync_threads()
            if Int32(tid) == Int32(0):
                _threadfence_sys()
                _store_relaxed_sys_u32(
                    self._flag(signals[self.previous], self._rank, block, incoming),
                    incoming,
                )
            cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            # Ordered receives make the final ACK drain every notification.
            acknowledged = self._flag(
                signals[self._rank], self.next, block, outgoing
            )
            observed = _load_acquire_sys_u32(acknowledged)
            while observed != outgoing:
                observed = _load_acquire_sys_u32(acknowledged)
            _a2a_graph_epoch_arrive(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                Uint32(self.blocks),
            )

    @cute.kernel
    def selected(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        if profile == Uint32(self.blocks):
            if cutlass.const_expr(self._world_size == 16):
                _run_lane_ring4_sw4_tp16(
                    self, source, output, stages, signals, rows, stride, block, tid
                )
            else:
                self._run_ring(
                    source, output, stages, signals, rows, stride, block, tid
                )

    @cute.jit
    def _ready(self, signals, block, tid):
        incoming, outgoing = Uint32(0), Uint32(0)
        # Single-leader system fence after CTA sync releases all segment writes.
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            own = signals[self._rank] + Int64(block) * Int64(16)
            outgoing = Uint32(cute.arch.load(own + Int64(self.next), Uint32)) + Uint32(
                1
            )
            incoming = Uint32(
                cute.arch.load(own + Int64(self.previous), Uint32)
            ) + Uint32(1)
            cute.arch.store(own + Int64(self.next), outgoing)
            cute.arch.store(own + Int64(self.previous), incoming)
            _store_relaxed_sys_u32(
                self._flag(signals[self.next], self._rank, block, outgoing),
                outgoing,
            )
            ready = self._flag(signals[self._rank], self.previous, block, incoming)
            observed = _load_acquire_sys_u32(ready)
            while observed != incoming:
                observed = _load_acquire_sys_u32(ready)
        cute.arch.sync_threads()
        return incoming, outgoing

    @cute.jit
    def _acknowledge(self, signals, block, tid, incoming, outgoing):
        # Consume incoming staging into final output before acknowledging
        # it. Waiting for downstream ack bounds each flag to one live hop.
        cute.arch.sync_threads()
        if Int32(tid) == Int32(0):
            _threadfence_sys()
            _store_relaxed_sys_u32(
                self._flag(signals[self.previous], self._rank, block, incoming, True),
                incoming,
            )
            # Two ranks share a sender; skip ready and the two graph-epoch words.
            acknowledged = self._flag(
                signals[self._rank], self.next, block, outgoing, True
            )
            observed = _load_acquire_sys_u32(acknowledged)
            while observed != outgoing:
                observed = _load_acquire_sys_u32(acknowledged)
        cute.arch.sync_threads()


class VarlenRingGather(RingGather):
    """In-place byte rows streamed through bounded, CTA-owned neighbor banks.

    All ranks use the maximum count for the chunk loop, including one empty
    chunk. Each CTA sends, receives into output, and awaits acknowledgement
    before reusing its segment. Relays read output, never acknowledged staging.
    """

    def __init__(self, world, rank, capacity, bank_bytes):
        super().__init__(world, rank, capacity // 4, 128, 8, 0)
        self.capacity, self.bank_packs = capacity, bank_bytes // 16

    @cute.jit
    def __call__(
        self,
        output: cute.Pointer,
        counts: cute.Pointer,
        stages: tuple,
        signals: tuple,
        stride: Int64,
        stream,
    ):
        self.gather(output, counts, stages, signals, stride).launch(
            grid=(8, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def gather(self, output, counts, stages, signals, stride):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        maximum = Int64(0)
        for peer in cutlass.range_constexpr(self._world_size):
            count = cute.arch.load(counts + peer, Int64)
            if (
                count < Int64(0)
                or count > Int64(self.capacity)
                or count % Int64(16) != Int64(0)
            ):
                _invalid_count()
            maximum = cutlass.max(maximum, count)
        chunks = cutlass.max(
            Int64(1),
            (maximum + Int64(self.bank_packs * 16 - 1)) // Int64(self.bank_packs * 16),
        )
        generation = ld_relaxed_gpu_u32(
            (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
        )
        bank = Int64(generation % Uint32(2)) * stride
        local = self._staging_words(stages[self._rank] + bank)
        remote = self._staging_words(stages[self.next] + bank)
        for chunk in cutlass.range(chunks):
            base = chunk * Int64(self.bank_packs)
            for phase in cutlass.range_constexpr(self._world_size - 1):
                sending = (self._rank - phase) % self._world_size
                receiving = (self._rank - phase - 1) % self._world_size
                send_packs = cute.arch.load(counts + sending, Int64) // Int64(16)
                recv_packs = cute.arch.load(counts + receiving, Int64) // Int64(16)
                pack = Int64(block) * Int64(128) + Int64(tid)
                while pack < Int64(self.bank_packs) and base + pack < send_packs:
                    source = Int64(sending * self.row_words) + (base + pack) * Int64(4)
                    _copy_16b(output + source, remote + pack * Int64(4))
                    pack += Int64(1024)
                incoming, outgoing = self._ready(signals, block, tid)
                pack = Int64(block) * Int64(128) + Int64(tid)
                while pack < Int64(self.bank_packs) and base + pack < recv_packs:
                    destination = Int64(receiving * self.row_words) + (
                        base + pack
                    ) * Int64(4)
                    _copy_16b(local + pack * Int64(4), output + destination)
                    pack += Int64(1024)
                self._acknowledge(signals, block, tid, incoming, outgoing)
        if Int32(tid) == Int32(0):
            _a2a_graph_epoch_arrive(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                (signals[self._rank] + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                Uint32(8),
            )


@program_cache(scope="preparation")
def compile_varlen_ring(world, rank, capacity, bank_bytes):
    operation = VarlenRingGather(world, rank, capacity, bank_bytes)

    def pointer(dtype, address=16, alignment=16):
        return make_ptr(
            dtype, address, mem_space=cute.AddressSpace.gmem, assumed_align=alignment
        )

    raw = compile_kernel(
        operation,
        pointer(Uint32),
        pointer(Int64, alignment=8),
        tuple(pointer(Uint8) for _ in range(16)),
        tuple(pointer(Uint32, alignment=4) for _ in range(16)),
        Int64(256),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.varlen_ring", 1, (world, rank, capacity, bank_bytes)
        ),
    )

    def run(output, counts, stages, signals, stride):
        raw(
            pointer(Uint32, output),
            pointer(Int64, counts, 8),
            tuple(pointer(Uint8, p) for p in (tuple(stages) * 16)[:16]),
            tuple(pointer(Uint32, p, 4) for p in (tuple(signals) * 16)[:16]),
            Int64(stride),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
