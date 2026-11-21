# SPDX-License-Identifier: Apache-2.0
"""Select a context shard's winners of a stable, rank-major FP32 top-2048.

The selector borrows an existing PCIe exchange channel. FlashInfer JIT builds
the optional CUDA C++ entry point for the device during plan materialization.
"""

from dataclasses import dataclass

import torch

from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    make_fixed_contract,
)

from b12x.comm.pcie.pcie_dcp_a2a import _SIGNAL_BYTES, PCIeDCPA2A

from . import _jit as jit


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


def supports_channel(runtime, rows: int = 4):
    """Use flag-line padding only in an owned, zero-initialized 4/8/16-rank channel."""
    return (
        isinstance(runtime, PCIeDCPA2A)
        and not runtime._closed
        and runtime._legacy_ext_module is None
        and runtime.world_size in (4, 8, 16)
        and 1 <= rows <= 32
        and runtime.max_batch_size >= rows
        and runtime._signal_bytes >= _SIGNAL_BYTES
        and runtime._slot_bytes >= rows * (2048 + 64) * 8
        and jit.sources_available()
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


def plan(runtime, *, cache_pages, rows: int = 4):
    """Prepare owner-local slots for rank-major stable top-2048 across 4, 8, or 16 CUDA peers."""
    if not supports_channel(runtime, rows=rows):
        raise ValueError(
            "interleaved top-k requires an owned 4/8/16-rank DCP channel with sufficient "
            f"staging for {rows} rows and the experimental CUDA kernel sources"
        )
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError(
            "interleaved top-k cache pages must fit positive int32 token slots"
        )
    query = InterleavedTopKQuery(
        rows=rows, world=runtime.world_size, rank=runtime.rank, cache_pages=cache_pages
    )

    def materialize(selection, device):
        with torch.cuda.device(device.ordinal):
            return _State(
                jit.gen_interleaved_topk_slots_module().build_and_load().run,
                runtime,
                cache_pages,
            )

    size = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=_CONTRACT,
        query=query,
        invocation=FrozenMapping({"program": jit.source_id()}),
        _device=runtime.device,
        _compile_jobs=lambda config, device: (),
        _materialize=materialize,
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(("pcie-dcp-runtime", id(runtime)), size, size),
            )
        ),
    )
