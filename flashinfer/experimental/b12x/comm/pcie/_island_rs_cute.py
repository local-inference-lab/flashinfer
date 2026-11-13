"""CuTeDSL pull-based island reduce-scatter TP16 collective.

Rank ``island * 4 + lane`` owns one equal quarter of the vector. Each rank
pulls from six peers: the three other lanes of its four-rank island and the
same lane in the three other islands. This bounds peer degree at six while
distributing inter-island traffic across every rank. The three protocol rounds
move 2.25 times the BF16 payload per rank.
"""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence
from typing import Tuple

from b12x._lib.compile_plan import attach_programs
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
)
from b12x._lib.utils.pointers import make_ptr


@dsl_user_op
def _atomic_add_global_u32(
    address: Int64, value: Uint32, *, loc=None, ip=None
) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_acquire_sys_u32(address: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_release_sys_u32(address: Int64, value: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_sc_sys(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.sc.sys;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nanosleep(cycles: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Uint32(cycles).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _wait_for(address: Int64, generation: Uint32, cycles: int) -> None:
    observed = _load_acquire_sys_u32(address)
    # A later ready tag also suffices. Signed modular comparison preserves this
    # ordering across wrap; the two-slot protocol bounds producer advancement.
    while Int32(observed - generation) < Int32(0):
        if cutlass.const_expr(cycles > 0):
            _nanosleep(Uint32(cycles))
        observed = _load_acquire_sys_u32(address)


@cute.jit
def _flag_address(base: Int64, region: int, block: Int32, index: Int32) -> Int64:
    return (
        base
        + Int64(region)
        + (Int64(block) * Int64(_ISLAND_SIZE) + Int64(index)) * Int64(128)
    )


@dsl_user_op
def unpack_bf16x2(value: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(value).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def pack_f32x2_to_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Match two scalar ``__float2bfloat16`` conversions without saturation."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 blo, bhi;
                cvt.rn.bf16.f32 blo, $1;
                cvt.rn.bf16.f32 bhi, $2;
                mov.b32 $0, {blo, bhi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


_ISLAND_SIZE = 4
_MAX_ISLANDS = 4
_MAX_WORLD_SIZE = _ISLAND_SIZE * _MAX_ISLANDS

# Generation counters precede three regions with one cache line per CTA and peer.
MAX_BLOCKS = 64
_ACTIVE_EPOCH_OFFSET = MAX_BLOCKS * 4
_RS_ARRIVED = ((_ACTIVE_EPOCH_OFFSET + MAX_BLOCKS * 4 + 255) // 256) * 256
_XS_ARRIVED = _RS_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128
_AG_ARRIVED = _XS_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128
HEADER_BYTES = _AG_ARRIVED + MAX_BLOCKS * _ISLAND_SIZE * 128


def island_rs_peers(rank: int, world_size: int) -> tuple[int, ...]:
    """Slabs this rank maps: three island lanes, three same-lane partners."""

    if world_size != _MAX_WORLD_SIZE:
        raise ValueError(f"island reduce-scatter requires TP16, got TP{world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"invalid rank {rank} for TP{world_size}")
    island, lane = divmod(rank, _ISLAND_SIZE)
    lanes = {island * _ISLAND_SIZE + p for p in range(_ISLAND_SIZE)}
    partners = {j * _ISLAND_SIZE + lane for j in range(_MAX_ISLANDS)}
    return tuple(sorted((lanes | partners) - {rank}))


class _IslandRSLaunch:
    """Rank-specialized launcher; every peer index folds away at compile time."""

    def __init__(
        self,
        world_size: int,
        rank: int,
        *,
        threads: int = 128,
        wait_nanosleep_cycles: int = 24,
        cross_island_rsag: bool = False,
        local_push: bool = False,
    ) -> None:
        if world_size != _MAX_WORLD_SIZE:
            raise ValueError(f"unsupported world size {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"invalid rank {rank} for world size {world_size}")
        if not 32 <= threads <= 1024 or threads % 32:
            raise ValueError("threads must be a multiple of 32 in [32, 1024]")
        if type(cross_island_rsag) is not bool:
            raise TypeError("cross_island_rsag must be a boolean")
        if type(local_push) is not bool:
            raise TypeError("local_push must be a boolean")
        self._local_push = local_push
        self._cross_island_rsag = cross_island_rsag
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._threads = int(threads)
        self._wait_nanosleep_cycles = int(wait_nanosleep_cycles)
        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE

    @cute.jit
    def __call__(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
        blocks: Int32,
        active_blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
            input_ptr,
            output_ptr,
            stage_offset,
            part_offset,
            final_offset,
            quarter_capacity,
            elements,
            active_blocks,
        ).launch(
            grid=(blocks, 1, 1),
            block=[self._threads, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        slab0: cute.Pointer,
        slab1: cute.Pointer,
        slab2: cute.Pointer,
        slab3: cute.Pointer,
        slab4: cute.Pointer,
        slab5: cute.Pointer,
        slab6: cute.Pointer,
        slab7: cute.Pointer,
        slab8: cute.Pointer,
        slab9: cute.Pointer,
        slab10: cute.Pointer,
        slab11: cute.Pointer,
        slab12: cute.Pointer,
        slab13: cute.Pointer,
        slab14: cute.Pointer,
        slab15: cute.Pointer,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        stage_offset: Int64,
        part_offset: Int64,
        final_offset: Int64,
        quarter_capacity: Int64,
        elements: Int64,
        active_blocks: Int32,
    ) -> None:
        slabs = (
            slab0,
            slab1,
            slab2,
            slab3,
            slab4,
            slab5,
            slab6,
            slab7,
            slab8,
            slab9,
            slab10,
            slab11,
            slab12,
            slab13,
            slab14,
            slab15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        self_base = Int64(slabs[self._rank].toint())

        allocator = cutlass.utils.SmemAllocator()
        shared_generation = allocator.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_layout((2,)),
            byte_alignment=4,
        )
        if tidx == Int32(0):
            shared_generation[0] = _atomic_add_global_u32(
                self_base + Int64(bidx) * Int64(4), Uint32(1)
            ) + Uint32(1)
            epoch = Uint32(0)
            if bidx < active_blocks:
                epoch = _atomic_add_global_u32(
                    self_base + Int64(_ACTIVE_EPOCH_OFFSET) + Int64(bidx) * Int64(4),
                    Uint32(1),
                ) + Uint32(1)
            shared_generation[1] = epoch
        cute.arch.barrier()
        data_generation = Uint32(shared_generation[0])
        # Every physical CTA advances buffer parity. Communication epochs advance
        # only for participating CTAs, keeping their last flags within one epoch
        # even when a CTA has been inactive across a global counter rollover.
        if bidx < active_blocks:
            self._participate(
                slabs,
                input_ptr,
                output_ptr,
                stage_offset,
                part_offset,
                final_offset,
                quarter_capacity,
                elements,
                data_generation,
                Uint32(shared_generation[1]),
                self_base,
                tidx,
                bidx,
                active_blocks,
            )

    @cute.jit
    def _participate(
        self,
        slabs,
        input_ptr,
        output_ptr,
        stage_offset,
        part_offset,
        final_offset,
        quarter_capacity,
        elements,
        data_generation,
        generation,
        self_base,
        tidx,
        bidx,
        gdim,
    ):
        if data_generation % Uint32(2) != Uint32(0):
            stage_offset += quarter_capacity * Int64(_ISLAND_SIZE * 4)
            part_offset += quarter_capacity * Int64(4)
            final_offset += quarter_capacity * Int64(_ISLAND_SIZE * 4)

        # Everything below counts in bf16x2 words; callers guarantee an even
        # element count so there is no scalar tail to chase.
        pairs = elements // Int64(2)
        quarter = (pairs + Int64(_ISLAND_SIZE - 1)) // Int64(_ISLAND_SIZE)
        stride = Int64(gdim) * Int64(self._threads)
        start = Int64(bidx) * Int64(self._threads) + Int64(tidx)

        input_words = cute.recast_ptr(input_ptr.align(4), dtype=Uint32)
        output_words = cute.recast_ptr(output_ptr.align(4), dtype=Uint32)

        # Each rank reduces one quarter. Local push delivers that quarter's
        # four input contributions into its staging plane before the reduction.
        self_stage = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + stage_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        # A block publishes the same shifted residue class that the matching
        # block on each destination lane consumes after its per-block arrival
        # flag. This keeps data ownership aligned with synchronization even
        # when a quarter boundary is not divisible by the launch stride.
        for lane in cutlass.range_constexpr(_ISLAND_SIZE):
            begin = Int64(lane) * quarter
            stop = begin + quarter
            if stop > pairs:
                stop = pairs
            index = start
            if cutlass.const_expr(self._local_push):
                destination = self._island * _ISLAND_SIZE + lane
                destination_stage = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[destination].toint()) + stage_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                # In the non-RSAG push specialization, the owned contribution
                # is read directly from the input during reduction instead of
                # staging it locally and reading it back.
                if cutlass.const_expr(self._cross_island_rsag or lane != self._lane):
                    while index < stop - begin:
                        destination_stage[Int64(self._lane) * quarter + index] = (
                            input_words[begin + index]
                        )
                        index += stride
            else:
                while index < stop - begin:
                    self_stage[begin + index] = input_words[begin + index]
                    index += stride
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    peer = self._island * _ISLAND_SIZE + p
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[peer].toint()),
                            _RS_ARRIVED,
                            bidx,
                            Int32(self._lane),
                        ),
                        generation,
                    )
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    _wait_for(
                        _flag_address(self_base, _RS_ARRIVED, bidx, Int32(p)),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
        cute.arch.barrier()

        mine_begin = Int64(self._lane) * quarter
        mine_stop = mine_begin + quarter
        if mine_stop > pairs:
            mine_stop = pairs
        mine_length = mine_stop - mine_begin

        self_part = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + part_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        index = start
        # Read four independent stride positions before consuming peer values.
        # Each block retains the residue classes covered by its arrival flag.
        while index + Int64(3) * stride < mine_length:
            words = ()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                peer = self._island * _ISLAND_SIZE + p
                if cutlass.const_expr(self._local_push):
                    if cutlass.const_expr(
                        p == self._lane and not self._cross_island_rsag
                    ):
                        peer_stage = input_words
                        peer_begin = mine_begin
                    else:
                        peer_stage = self_stage
                        peer_begin = Int64(p) * quarter
                else:
                    peer_begin = mine_begin
                    peer_stage = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                for item in cutlass.range_constexpr(4):
                    words += (peer_stage[peer_begin + index + Int64(item) * stride],)
            for item in cutlass.range_constexpr(4):
                total_lo = Float32(0.0)
                total_hi = Float32(0.0)
                for p in cutlass.range_constexpr(_ISLAND_SIZE):
                    lo, hi = unpack_bf16x2(words[p * 4 + item])
                    total_lo += lo
                    total_hi += hi
                self_part[index + Int64(item) * stride] = pack_f32x2_to_bf16x2(
                    total_lo, total_hi
                )
            index += Int64(4) * stride
        while index < mine_length:
            total_lo = Float32(0.0)
            total_hi = Float32(0.0)
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                peer = self._island * _ISLAND_SIZE + p
                if cutlass.const_expr(self._local_push):
                    if cutlass.const_expr(
                        p == self._lane and not self._cross_island_rsag
                    ):
                        peer_stage = input_words
                        peer_begin = mine_begin
                    else:
                        peer_stage = self_stage
                        peer_begin = Int64(p) * quarter
                else:
                    peer_begin = mine_begin
                    peer_stage = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + stage_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                lo, hi = unpack_bf16x2(peer_stage[peer_begin + index])
                total_lo += lo
                total_hi += hi
            self_part[index] = pack_f32x2_to_bf16x2(total_lo, total_hi)
            index += stride
        cute.arch.barrier()

        # The XS mailbox publishes partial-ready (2*g), then final-ready (2*g+1)
        # for scattered stripes. A later tag also certifies the still-live partials.
        # Both modes sum island partials in the same order before BF16 conversion.
        if tidx == Int32(0):
            _fence_sc_sys()
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    partner = j * _ISLAND_SIZE + self._lane
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[partner].toint()),
                            _XS_ARRIVED,
                            bidx,
                            Int32(self._island),
                        ),
                        generation * Uint32(2),
                    )
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    _wait_for(
                        _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
                        generation * Uint32(2),
                        self._wait_nanosleep_cycles,
                    )
        cute.arch.barrier()

        self_final = cute.recast_ptr(
            cute.make_ptr(
                cutlass.BFloat16,
                self_base + final_offset,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ).align(4),
            dtype=Uint32,
        )
        # Distribute stride positions across islands without changing which
        # CTA owns their publication flags. The final plane already reserves a
        # full vector and can hold both partial and assembled final quarters.
        cross_stride = stride
        index = start
        if cutlass.const_expr(self._cross_island_rsag):
            index += Int64(self._island) * stride
            cross_stride *= Int64(_MAX_ISLANDS)
        # Read four independent cross_stride positions before consuming peer values.
        # Each block retains the residue classes covered by its arrival flag.
        while index + Int64(3) * cross_stride < mine_length:
            words = ()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                partner = p * _ISLAND_SIZE + self._lane
                partner_part = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[partner].toint()) + part_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                for item in cutlass.range_constexpr(4):
                    words += (partner_part[index + Int64(item) * cross_stride],)
            for item in cutlass.range_constexpr(4):
                total_lo = Float32(0.0)
                total_hi = Float32(0.0)
                for p in cutlass.range_constexpr(_ISLAND_SIZE):
                    lo, hi = unpack_bf16x2(words[p * 4 + item])
                    total_lo += lo
                    total_hi += hi
                value = pack_f32x2_to_bf16x2(total_lo, total_hi)
                if cutlass.const_expr(not self._local_push or self._cross_island_rsag):
                    self_final[mine_begin + index + Int64(item) * cross_stride] = value
                output_words[mine_begin + index + Int64(item) * cross_stride] = value
                if cutlass.const_expr(self._local_push and not self._cross_island_rsag):
                    for destination_lane in cutlass.range_constexpr(_ISLAND_SIZE):
                        if cutlass.const_expr(destination_lane != self._lane):
                            destination_final = cute.recast_ptr(
                                cute.make_ptr(
                                    cutlass.BFloat16,
                                    Int64(
                                        slabs[
                                            self._island * _ISLAND_SIZE
                                            + destination_lane
                                        ].toint()
                                    )
                                    + final_offset,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ).align(4),
                                dtype=Uint32,
                            )
                            destination_final[
                                mine_begin + index + Int64(item) * cross_stride
                            ] = value
            index += Int64(4) * cross_stride
        while index < mine_length:
            total_lo = Float32(0.0)
            total_hi = Float32(0.0)
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                partner = j * _ISLAND_SIZE + self._lane
                partner_part = cute.recast_ptr(
                    cute.make_ptr(
                        cutlass.BFloat16,
                        Int64(slabs[partner].toint()) + part_offset,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ).align(4),
                    dtype=Uint32,
                )
                lo, hi = unpack_bf16x2(partner_part[index])
                total_lo += lo
                total_hi += hi
            value = pack_f32x2_to_bf16x2(total_lo, total_hi)
            if cutlass.const_expr(not self._local_push or self._cross_island_rsag):
                self_final[mine_begin + index] = value
            output_words[mine_begin + index] = value
            if cutlass.const_expr(self._local_push and not self._cross_island_rsag):
                for destination_lane in cutlass.range_constexpr(_ISLAND_SIZE):
                    if cutlass.const_expr(destination_lane != self._lane):
                        destination_final = cute.recast_ptr(
                            cute.make_ptr(
                                cutlass.BFloat16,
                                Int64(
                                    slabs[
                                        self._island * _ISLAND_SIZE + destination_lane
                                    ].toint()
                                )
                                + final_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ).align(4),
                            dtype=Uint32,
                        )
                        destination_final[mine_begin + index] = value
            index += cross_stride
        cute.arch.barrier()

        if cutlass.const_expr(self._cross_island_rsag):
            if tidx == Int32(0):
                _fence_sc_sys()
                for j in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(j != self._island):
                        partner = j * _ISLAND_SIZE + self._lane
                        _store_release_sys_u32(
                            _flag_address(
                                Int64(slabs[partner].toint()),
                                _XS_ARRIVED,
                                bidx,
                                Int32(self._island),
                            ),
                            generation * Uint32(2) + Uint32(1),
                        )
                for j in cutlass.range_constexpr(_MAX_ISLANDS):
                    if cutlass.const_expr(j != self._island):
                        _wait_for(
                            _flag_address(self_base, _XS_ARRIVED, bidx, Int32(j)),
                            generation * Uint32(2) + Uint32(1),
                            self._wait_nanosleep_cycles,
                        )
            cute.arch.barrier()
            for j in cutlass.range_constexpr(_MAX_ISLANDS):
                if cutlass.const_expr(j != self._island):
                    partner = j * _ISLAND_SIZE + self._lane
                    partner_final = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[partner].toint()) + final_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                    index = start + Int64(j) * stride
                    while index < mine_length:
                        value = partner_final[mine_begin + index]
                        self_final[mine_begin + index] = value
                        output_words[mine_begin + index] = value
                        index += cross_stride
            cute.arch.barrier()

        # RSAG publishes completed quarters in a separate pass. The non-RSAG
        # push specialization above already forwarded each rounded value to all
        # three local peers directly from registers.
        if cutlass.const_expr(self._local_push and self._cross_island_rsag):
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    push_peer = self._island * _ISLAND_SIZE + p
                    destination_final = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[push_peer].toint()) + final_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                    index = start
                    while index < mine_length:
                        destination_final[mine_begin + index] = self_final[
                            mine_begin + index
                        ]
                        index += stride
            cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    peer = self._island * _ISLAND_SIZE + p
                    _store_release_sys_u32(
                        _flag_address(
                            Int64(slabs[peer].toint()),
                            _AG_ARRIVED,
                            bidx,
                            Int32(self._lane),
                        ),
                        generation,
                    )
            for p in cutlass.range_constexpr(_ISLAND_SIZE):
                if cutlass.const_expr(p != self._lane):
                    _wait_for(
                        _flag_address(self_base, _AG_ARRIVED, bidx, Int32(p)),
                        generation,
                        self._wait_nanosleep_cycles,
                    )
        cute.arch.barrier()

        for p in cutlass.range_constexpr(_ISLAND_SIZE):
            if cutlass.const_expr(p != self._lane):
                peer = self._island * _ISLAND_SIZE + p
                if cutlass.const_expr(self._local_push):
                    peer_final = self_final
                else:
                    peer_final = cute.recast_ptr(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            Int64(slabs[peer].toint()) + final_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ).align(4),
                        dtype=Uint32,
                    )
                begin = Int64(p) * quarter
                stop = begin + quarter
                if stop > pairs:
                    stop = pairs
                index = start
                while index < stop - begin:
                    output_words[begin + index] = peer_final[begin + index]
                    index += stride


@program_cache
def get_island_rs_launcher(
    world_size: int,
    rank: int,
    device_index: int,
    *,
    threads: int = 128,
    wait_nanosleep_cycles: int = 24,
    cross_island_rsag: bool = False,
    local_push: bool = False,
) -> Callable[..., None]:
    """Return the rank-specialized TP16 island reduce-scatter launcher."""

    del device_index  # retained in the process-local cache key
    launch = _IslandRSLaunch(
        world_size,
        rank,
        threads=threads,
        wait_nanosleep_cycles=wait_nanosleep_cycles,
        cross_island_rsag=cross_island_rsag,
        local_push=local_push,
    )
    cache_key = (
        int(world_size),
        int(rank),
        int(threads),
        int(wait_nanosleep_cycles),
        bool(cross_island_rsag),
        bool(local_push),
    )
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=launch, cache_key=cache_key
    )
    slab_placeholder = make_ptr(
        cutlass.Uint8,
        256,
        cute.AddressSpace.gmem,
        assumed_align=256,
    )
    raw = b12x_compile(
        launch,
        *(slab_placeholder for _ in range(_MAX_WORLD_SIZE)),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2),
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_rs.bf16",
            3,
            cache_key,
            labels=(
                "world_size",
                "rank",
                "threads",
                "wait_nanosleep_cycles",
                "cross_island_rsag",
                "local_push",
            ),
        ),
    )

    def run(
        slab_addresses: Sequence[int],
        input_address: int,
        output_address: int,
        stage_offset: int,
        part_offset: int,
        final_offset: int,
        quarter_capacity: int,
        elements: int,
        blocks: int,
        stream: object = None,
        active_blocks: int | None = None,
    ) -> None:
        """Launch the compiled island reduce-scatter with runtime arguments."""
        if len(slab_addresses) != world_size:
            raise ValueError(
                f"expected {world_size} slab addresses, got {len(slab_addresses)}"
            )
        if not 1 <= int(blocks) <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
        active_blocks = blocks if active_blocks is None else active_blocks
        if not 1 <= int(active_blocks) <= int(blocks):
            raise ValueError("active blocks must fit the fixed physical grid")
        raw(
            *(
                make_ptr(
                    cutlass.Uint8,
                    int(address),
                    cute.AddressSpace.gmem,
                    assumed_align=256,
                )
                for address in slab_addresses
            ),
            make_ptr(
                cutlass.BFloat16,
                input_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            make_ptr(
                cutlass.BFloat16,
                output_address,
                cute.AddressSpace.gmem,
                assumed_align=2,
            ),
            stage_offset,
            part_offset,
            final_offset,
            quarter_capacity,
            elements,
            blocks,
            active_blocks,
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


__all__ = ["HEADER_BYTES", "MAX_BLOCKS", "get_island_rs_launcher", "island_rs_peers"]
