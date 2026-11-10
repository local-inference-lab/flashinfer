"""CuTe DSL kernel for exact owner-sharded DCP top-k transport."""

from __future__ import annotations

from b12x._lib.program_cache import program_cache
from collections.abc import Callable, Sequence

from b12x._lib.compile_plan import attach_programs
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics.memory import (
    ld_global_nc_u32,
    ld_global_v4_u32,
    st_global_u32,
    st_global_v4_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._dcp_cute_common import block_pair_barrier, _membar_sys


_MAX_BLOCKS = 128
_MAX_PEERS = 16


@cute.jit
def _copy_16b(source: cute.Pointer, destination: cute.Pointer) -> None:
    values = ld_global_v4_u32(source.toint())
    st_global_v4_u32(destination.toint(), *values)


@cute.jit
def _candidate_count(local_counts: cute.Pointer, row: Int64, topk: Int32) -> Int32:
    count = Int32(ld_global_nc_u32((local_counts + row).toint()))
    if count < Int32(0):
        count = Int32(0)
    if count > topk:
        count = topk
    return count


@cute.jit
def _copy_candidate(
    local_indices: cute.Pointer,
    local_scores: cute.Pointer,
    source_pack: Int64,
    destination_indices: cute.Pointer,
    destination_scores: cute.Pointer,
    destination_pack: Int64,
) -> None:
    _copy_16b(
        local_indices + source_pack * Int64(4),
        destination_indices + destination_pack * Int64(4),
    )
    _copy_16b(
        local_scores + source_pack * Int64(4),
        destination_scores + destination_pack * Int64(4),
    )


class _TopKOwnerStageLaunch:
    def __init__(
        self,
        world_size: int,
        rank: int,
        topk: int,
        threads: int,
        length_aware: bool = False,
    ) -> None:
        self._world_size = int(world_size)
        self._rank = int(rank)
        self._topk = int(topk)
        self._threads = int(threads)
        self._length_aware = bool(length_aware)

    @cute.jit
    def __call__(
        self,
        local_indices: cute.Pointer,
        local_scores: cute.Pointer,
        local_counts: cute.Pointer,
        candidate0: cute.Pointer,
        candidate1: cute.Pointer,
        candidate2: cute.Pointer,
        candidate3: cute.Pointer,
        candidate4: cute.Pointer,
        candidate5: cute.Pointer,
        candidate6: cute.Pointer,
        candidate7: cute.Pointer,
        candidate8: cute.Pointer,
        candidate9: cute.Pointer,
        candidate10: cute.Pointer,
        candidate11: cute.Pointer,
        candidate12: cute.Pointer,
        candidate13: cute.Pointer,
        candidate14: cute.Pointer,
        candidate15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        rows: Int32,
        candidate_plane_elems: Int64,
        grid_x: Int32,
        wait_for_prior_consumer: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            local_indices,
            local_scores,
            local_counts,
            candidate0,
            candidate1,
            candidate2,
            candidate3,
            candidate4,
            candidate5,
            candidate6,
            candidate7,
            candidate8,
            candidate9,
            candidate10,
            candidate11,
            candidate12,
            candidate13,
            candidate14,
            candidate15,
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
            rows,
            candidate_plane_elems,
            wait_for_prior_consumer,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        local_indices: cute.Pointer,
        local_scores: cute.Pointer,
        local_counts: cute.Pointer,
        candidate0: cute.Pointer,
        candidate1: cute.Pointer,
        candidate2: cute.Pointer,
        candidate3: cute.Pointer,
        candidate4: cute.Pointer,
        candidate5: cute.Pointer,
        candidate6: cute.Pointer,
        candidate7: cute.Pointer,
        candidate8: cute.Pointer,
        candidate9: cute.Pointer,
        candidate10: cute.Pointer,
        candidate11: cute.Pointer,
        candidate12: cute.Pointer,
        candidate13: cute.Pointer,
        candidate14: cute.Pointer,
        candidate15: cute.Pointer,
        signal0: cute.Pointer,
        signal1: cute.Pointer,
        signal2: cute.Pointer,
        signal3: cute.Pointer,
        signal4: cute.Pointer,
        signal5: cute.Pointer,
        signal6: cute.Pointer,
        signal7: cute.Pointer,
        signal8: cute.Pointer,
        signal9: cute.Pointer,
        signal10: cute.Pointer,
        signal11: cute.Pointer,
        signal12: cute.Pointer,
        signal13: cute.Pointer,
        signal14: cute.Pointer,
        signal15: cute.Pointer,
        rows: Int32,
        candidate_plane_elems: Int64,
        wait_for_prior_consumer: Int32,
    ) -> None:
        candidates = (
            candidate0,
            candidate1,
            candidate2,
            candidate3,
            candidate4,
            candidate5,
            candidate6,
            candidate7,
            candidate8,
            candidate9,
            candidate10,
            candidate11,
            candidate12,
            candidate13,
            candidate14,
            candidate15,
        )
        signals = (
            signal0,
            signal1,
            signal2,
            signal3,
            signal4,
            signal5,
            signal6,
            signal7,
            signal8,
            signal9,
            signal10,
            signal11,
            signal12,
            signal13,
            signal14,
            signal15,
        )
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        owner_rows = (Int64(rows) + Int64(self._world_size - 1)) // Int64(
            self._world_size
        )
        packs_per_row = Int64(self._topk // 4)
        owner_packs = owner_rows * packs_per_row
        output_row_packs = Int64(self._world_size) * packs_per_row
        chunk = (owner_packs + Int64(gdim) - Int64(1)) // Int64(gdim)
        begin = Int64(bidx) * chunk
        end = begin + chunk
        if end > owner_packs:
            end = owner_packs

        if wait_for_prior_consumer != Int32(0):
            block_pair_barrier(
                signals,
                self_signal=signals[self._rank],
                rank=self._rank,
                world_size=self._world_size,
                max_blocks=_MAX_BLOCKS,
            )

        candidate_plane_words = candidate_plane_elems
        # Ragged partitions retain fixed owner-row capacity. Dummy rows are
        # initialized locally; producers neither read absent rows nor send them.
        padding_pack = begin + Int64(tidx)
        while padding_pack < end:
            owner_row = padding_pack // packs_per_row
            if Int64(self._rank) * owner_rows + owner_row >= Int64(rows):
                column_pack = padding_pack - owner_row * packs_per_row
                destination = candidates[self._rank]
                peer = Int32(0)
                while peer < Int32(self._world_size):
                    offset = (
                        owner_row * output_row_packs
                        + Int64(peer) * packs_per_row
                        + column_pack
                    )
                    st_global_v4_u32(
                        (destination + offset * Int64(4)).toint(),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                        Uint32(0xFFFFFFFF),
                    )
                    st_global_v4_u32(
                        (
                            destination + candidate_plane_words + offset * Int64(4)
                        ).toint(),
                        Uint32(0xFF800000),
                        Uint32(0xFF800000),
                        Uint32(0xFF800000),
                        Uint32(0xFF800000),
                    )
                    if cutlass.const_expr(self._length_aware):
                        if column_pack == Int64(0):
                            st_global_u32(
                                (
                                    destination
                                    + 2 * candidate_plane_words
                                    + owner_row * Int64(self._world_size)
                                    + Int64(peer)
                                ).toint(),
                                Uint32(0),
                            )
                    peer += Int32(1)
            padding_pack += Int64(self._threads)
        if cutlass.const_expr(self._world_size in (2, 4)):
            # Scalar runtime pointer selection costs TP2/TP4 an occupancy tier.
            # Retain their smaller proven specializations while keeping the
            # remaining path below faithful to CUDA's ``#pragma unroll 1``.
            for unrolled_step in cutlass.range_constexpr(self._world_size):
                unrolled_destination = (self._rank + unrolled_step) % self._world_size
                unrolled_destination_indices = candidates[unrolled_destination]
                unrolled_destination_scores = (
                    unrolled_destination_indices + candidate_plane_words
                )
                unrolled_input_offset = Int64(unrolled_destination) * owner_packs
                unrolled_pack = begin + Int64(tidx)
                while unrolled_pack < end:
                    unrolled_owner_row = unrolled_pack // packs_per_row
                    unrolled_column_pack = (
                        unrolled_pack - unrolled_owner_row * packs_per_row
                    )
                    unrolled_output_offset = (
                        unrolled_owner_row * output_row_packs
                        + Int64(self._rank) * packs_per_row
                        + unrolled_column_pack
                    )
                    source_row = (
                        Int64(unrolled_destination) * owner_rows + unrolled_owner_row
                    )
                    if cutlass.const_expr(self._length_aware):
                        if source_row < Int64(rows):
                            count = _candidate_count(
                                local_counts, source_row, Int32(self._topk)
                            )
                            if unrolled_column_pack == Int64(0):
                                st_global_u32(
                                    (
                                        unrolled_destination_indices
                                        + 2 * candidate_plane_words
                                        + unrolled_owner_row * Int64(self._world_size)
                                        + Int64(self._rank)
                                    ).toint(),
                                    Uint32(count),
                                )
                            if unrolled_column_pack * Int64(4) < Int64(count):
                                _copy_candidate(
                                    local_indices,
                                    local_scores,
                                    unrolled_input_offset + unrolled_pack,
                                    unrolled_destination_indices,
                                    unrolled_destination_scores,
                                    unrolled_output_offset,
                                )
                    else:
                        if source_row < Int64(rows):
                            _copy_candidate(
                                local_indices,
                                local_scores,
                                unrolled_input_offset + unrolled_pack,
                                unrolled_destination_indices,
                                unrolled_destination_scores,
                                unrolled_output_offset,
                            )
                    unrolled_pack += Int64(self._threads)
        else:
            # The candidate pointers remain scalar kernel parameters; select
            # their address at runtime to avoid cloning the payload loop for
            # each peer.
            runtime_step = Int32(0)
            while runtime_step < Int32(self._world_size):
                runtime_destination = (Int32(self._rank) + runtime_step) % Int32(
                    self._world_size
                )
                runtime_destination_address = Int64(candidate0.toint())
                for runtime_peer in cutlass.range_constexpr(1, self._world_size):
                    if runtime_destination == Int32(runtime_peer):
                        runtime_destination_address = Int64(
                            candidates[runtime_peer].toint()
                        )
                runtime_destination_indices = cute.make_ptr(
                    Uint32,
                    runtime_destination_address,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                runtime_destination_scores = (
                    runtime_destination_indices + candidate_plane_words
                )
                runtime_input_offset = Int64(runtime_destination) * owner_packs
                runtime_pack = begin + Int64(tidx)
                while runtime_pack < end:
                    runtime_owner_row = runtime_pack // packs_per_row
                    runtime_column_pack = (
                        runtime_pack - runtime_owner_row * packs_per_row
                    )
                    runtime_output_offset = (
                        runtime_owner_row * output_row_packs
                        + Int64(self._rank) * packs_per_row
                        + runtime_column_pack
                    )
                    source_row = (
                        Int64(runtime_destination) * owner_rows + runtime_owner_row
                    )
                    if cutlass.const_expr(self._length_aware):
                        if source_row < Int64(rows):
                            count = _candidate_count(
                                local_counts, source_row, Int32(self._topk)
                            )
                            if runtime_column_pack == Int64(0):
                                st_global_u32(
                                    (
                                        runtime_destination_indices
                                        + 2 * candidate_plane_words
                                        + runtime_owner_row * Int64(self._world_size)
                                        + Int64(self._rank)
                                    ).toint(),
                                    Uint32(count),
                                )
                            if runtime_column_pack * Int64(4) < Int64(count):
                                _copy_candidate(
                                    local_indices,
                                    local_scores,
                                    runtime_input_offset + runtime_pack,
                                    runtime_destination_indices,
                                    runtime_destination_scores,
                                    runtime_output_offset,
                                )
                    else:
                        if source_row < Int64(rows):
                            _copy_candidate(
                                local_indices,
                                local_scores,
                                runtime_input_offset + runtime_pack,
                                runtime_destination_indices,
                                runtime_destination_scores,
                                runtime_output_offset,
                            )
                    runtime_pack += Int64(self._threads)
                if cutlass.const_expr(self._world_size == 16):
                    if (Int64(rows) > owner_rows) & (
                        Int64(runtime_destination) * owner_rows < Int64(rows)
                    ):
                        # Order this CTA's peer batches before its warps target
                        # another GPU. Single-owner rows use the final peer barrier.
                        cute.arch.sync_threads()
                        if tidx == 0:
                            _membar_sys()
                        cute.arch.sync_threads()
                runtime_step += Int32(1)

        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )


def _pad_ptrs(values: Sequence[int], world_size: int) -> tuple[int, ...]:
    if len(values) != world_size:
        raise ValueError("pointer bundle does not match DCP world size")
    padded = [int(value) for value in values]
    padded.extend([padded[-1]] * (_MAX_PEERS - len(padded)))
    return tuple(padded)


@program_cache
def _get_compiled_topk_stage(
    world_size: int,
    rank: int,
    topk: int,
    threads: int,
    length_aware: bool = False,
) -> Callable:
    """Compile and retain one owner top-k staging launcher specialization."""
    launch = _TopKOwnerStageLaunch(world_size, rank, topk, threads, length_aware)
    key = (int(world_size), int(rank), int(topk), int(threads), bool(length_aware))
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=launch,
        cache_key=key,
    )
    placeholder = make_ptr(
        Uint32,
        16,
        cute.AddressSpace.gmem,
        assumed_align=4,
    )
    raw = b12x_compile(
        launch,
        placeholder,
        placeholder,
        placeholder,
        *(placeholder for _ in range(2 * _MAX_PEERS)),
        1,
        1,
        1,
        0,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp_topk.owner_stage",
            8,
            key,
            labels=("world_size", "rank", "topk", "threads", "length_aware"),
        ),
    )

    def run(
        local_indices_ptr: int,
        local_scores_ptr: int,
        local_counts_ptr: int | None,
        candidate_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        rows: int,
        candidate_plane_elems: int,
        blocks: int,
        wait_for_prior_consumer: bool,
    ) -> None:
        """Launch the compiled owner top-k staging kernel with runtime arguments."""
        if length_aware != (local_counts_ptr is not None):
            raise ValueError("candidate counts differ from the compiled DCP top-k mode")
        candidates = _pad_ptrs(candidate_ptrs, world_size)
        signals = _pad_ptrs(signal_ptrs, world_size)
        raw(
            make_ptr(
                Uint32,
                int(local_indices_ptr),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                Uint32,
                int(local_scores_ptr),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                Uint32,
                16 if local_counts_ptr is None else int(local_counts_ptr),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            *(
                make_ptr(
                    Uint32,
                    ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                for ptr in candidates
            ),
            *(
                make_ptr(
                    Uint32,
                    ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                for ptr in signals
            ),
            int(rows),
            int(candidate_plane_elems),
            int(blocks),
            int(bool(wait_for_prior_consumer)),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
