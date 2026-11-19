# SPDX-License-Identifier: Apache-2.0
"""Select only a context shard's winners of a stable, rank-major FP32 top-2048."""

from dataclasses import dataclass

import torch

from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    make_fixed_contract,
)

from .pcie_dcp_a2a import _SIGNAL_BYTES, PCIeDCPA2A


@dataclass(frozen=True, kw_only=True)
class InterleavedTopKQuery:
    rows: int
    world: int
    rank: int
    cache_pages: int


_CONTRACT = make_fixed_contract(
    component_id="comm.pcie.interleaved_topk_slots",
    query_type=InterleavedTopKQuery,
    backend="cuda",
)


def supports_channel(runtime):
    """Use flag-line padding only in an owned, zero-initialized sixteen-rank channel."""
    return (
        isinstance(runtime, PCIeDCPA2A)
        and not runtime._closed
        and runtime._legacy_ext_module is None
        and runtime.world_size == 16
        and runtime.max_batch_size >= 4
        and runtime._signal_bytes >= _SIGNAL_BYTES
        and runtime._slot_bytes >= 4 * (2048 + 64) * 8
    )


@dataclass
class _State:
    run_kernel: object
    runtime: object
    cache_pages: int

    def run(self, ids, scores, positions, tables, owners, slots, counts):
        runtime = self.runtime
        if runtime._closed:
            raise RuntimeError("interleaved top-k channel is closed")
        runtime._check_stream()
        if ids.device != runtime.device:
            raise ValueError(
                "interleaved top-k inputs must use the prepared IPC device"
            )
        if not runtime._device_slot_selection:
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        self.run_kernel(
            ids,
            scores,
            positions,
            tables,
            owners,
            slots,
            counts,
            runtime._staging_ptrs[bank],
            runtime._signal_ptrs,
            runtime._slot_bytes if bank == 0 else -runtime._slot_bytes,
            runtime.rank,
            self.cache_pages,
        )


def plan(runtime, *, cache_pages):
    """Prepare owner-local slots for rank-major stable top-2048 across 16 CUDA peers.

    Ranks must provide identical global positions and issue channel operations
    in the same order, serially with all other uses of the channel. IDs are
    shard-local token indices. Each rank provides four contiguous rows of
    2048 int32 IDs and FP32 scores. Up to 2048 local entries are visible per
    row, followed by (-1, -inf) padding. FP32 radix keys, rank and original
    input column determine ties. Causal and page filtering follow selection.
    The contiguous int32 owner and slot outputs retain each row's local input
    order, with -1 padding and an int32 count. Page tables map local groups
    of 128 tokens and may share one row across all four int64 positions.
    """
    if not supports_channel(runtime):
        raise ValueError(
            "interleaved top-k requires an owned 16-rank DCP channel with two 67584-byte banks"
        )
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError(
            "interleaved top-k cache pages must fit positive int32 token slots"
        )
    from flashinfer.jit.interleaved_topk_slots import (
        gen_interleaved_topk_slots_module,
        source_id,
    )

    query = InterleavedTopKQuery(
        rows=4, world=16, rank=runtime.rank, cache_pages=cache_pages
    )

    def materialize(selection, device):
        with torch.cuda.device(device.ordinal):
            return _State(
                gen_interleaved_topk_slots_module().build_and_load().run,
                runtime,
                cache_pages,
            )

    size = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=_CONTRACT,
        query=query,
        invocation=FrozenMapping({"program": source_id()}),
        _device=runtime.device,
        _compile_jobs=lambda config, device: (),
        _materialize=materialize,
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(("pcie-dcp-runtime", id(runtime)), size, size),
            )
        ),
    )
