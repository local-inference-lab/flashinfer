# SPDX-License-Identifier: Apache-2.0
"""Select a context shard's winners of a stable, rank-major FP32 top-2048.

The selector borrows an existing PCIe exchange channel and compiles a pure
CuTeDSL kernel through ``b12x._lib.compiler``.
"""

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.compile_pool import CompileJob
from b12x.comm.pcie.pcie_dcp_a2a import _SIGNAL_BYTES, PCIeDCPA2A
from b12x.preparation import (
    MemoryRequirements,
    PersistentMemory,
    Plan,
    make_fixed_contract,
)

from ._cute import (
    K,
    MAX_ROWS,
    STAGE_ITEMS,
    SUPPORTED_WORLDS,
    compile_interleaved_topk_slots,
)


@dataclass(frozen=True, kw_only=True)
class InterleavedTopKQuery:
    rows: int
    world: int
    rank: int
    cache_pages: int


_CONTRACT = make_fixed_contract(
    component_id="comm.pcie.interleaved_topk_slots",
    query_type=InterleavedTopKQuery,
    backend="cute",
)


def supports_channel(runtime, rows: int = 4):
    """Use flag-line padding only in an owned, zero-initialized 4/8/16-rank channel."""
    return (
        isinstance(runtime, PCIeDCPA2A)
        and not runtime._closed
        and runtime._legacy_ext_module is None
        and runtime.world_size in SUPPORTED_WORLDS
        and type(rows) is int
        and 1 <= rows <= MAX_ROWS
        and runtime.max_batch_size >= rows
        and runtime._signal_bytes >= _SIGNAL_BYTES
        and runtime._slot_bytes >= rows * STAGE_ITEMS * 8
    )


def _validate_run_inputs(
    runtime,
    cache_pages: int,
    ids: torch.Tensor,
    scores: torch.Tensor,
    positions: torch.Tensor,
    tables: torch.Tensor,
    owners: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    stages,
    signals,
    bank_stride: int,
) -> None:
    device = runtime.device
    for name, tensor, dtype, ndim, contiguous in (
        ("ids", ids, torch.int32, 2, True),
        ("scores", scores, torch.float32, 2, True),
        ("positions", positions, torch.int64, 1, True),
        ("tables", tables, torch.int32, 2, False),
        ("owners", owners, torch.int32, 2, True),
        ("slots", slots, torch.int32, 2, True),
        ("counts", counts, torch.int32, 1, True),
    ):
        if tensor.device != device:
            raise ValueError(
                "interleaved top-k inputs must use the prepared IPC device"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"interleaved top-k {name} must have dtype {dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"interleaved top-k {name} must be {ndim}D")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"interleaved top-k {name} must be contiguous")

    rows = int(ids.size(0))
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"interleaved top-k rows must be in 1..{MAX_ROWS}, got {rows}")
    if int(ids.size(1)) != K:
        raise ValueError(f"interleaved top-k ids width must be {K}, got {ids.size(1)}")
    if scores.shape != ids.shape or owners.shape != ids.shape or slots.shape != ids.shape:
        raise ValueError("interleaved top-k scores/owners/slots must match ids shape")
    if int(positions.size(0)) != rows or int(counts.size(0)) != rows:
        raise ValueError("interleaved top-k positions/counts length must match rows")
    if int(tables.size(0)) != rows or int(tables.stride(1)) != 1:
        raise ValueError("interleaved top-k tables must have shape (rows, cols) with stride(1)==1")
    if int(tables.size(1)) <= 0 or int(tables.stride(0)) < 0:
        raise ValueError("interleaved top-k tables must have positive columns and non-negative row stride")
    if not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError("interleaved top-k cache pages must fit positive int32 token slots")
    world = runtime.world_size
    if len(stages) != world or len(signals) != world:
        raise ValueError("interleaved top-k stage/signal pointer counts must match world size")
    if not 0 <= runtime.rank < world:
        raise ValueError("interleaved top-k rank out of range")
    min_bank = rows * STAGE_ITEMS * 8
    if abs(bank_stride) < min_bank:
        raise ValueError("interleaved top-k staging bank stride is smaller than row requirement")
    for ptr in (*stages, *signals):
        if not ptr or (int(ptr) % 8) != 0:
            raise ValueError("interleaved top-k IPC pointers must be non-null and 8-byte aligned")


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
        if not runtime._device_slot_selection:
            runtime._graph_base_slot = runtime._next_slot & 1
            runtime._device_slot_selection = True
        bank = runtime._graph_base_slot
        stages = runtime._staging_ptrs[bank]
        signals = runtime._signal_ptrs
        bank_stride = runtime._slot_bytes if bank == 0 else -runtime._slot_bytes
        _validate_run_inputs(
            runtime,
            self.cache_pages,
            ids,
            scores,
            positions,
            tables,
            owners,
            slots,
            counts,
            stages,
            signals,
            bank_stride,
        )
        with torch.cuda.device(runtime.device):
            self.run_kernel(
                ids,
                scores,
                positions,
                tables,
                owners,
                slots,
                counts,
                stages,
                signals,
                bank_stride,
                runtime.rank,
                self.cache_pages,
            )


def plan(runtime, *, cache_pages, rows: int = 4):
    """Prepare owner-local slots for rank-major stable top-2048 across 4, 8, or 16 CUDA peers."""
    if not supports_channel(runtime, rows=rows):
        raise ValueError(
            "interleaved top-k requires an owned 4/8/16-rank DCP channel with sufficient "
            f"staging for {rows} rows"
        )
    if type(cache_pages) is not int or not 0 < cache_pages <= (2**31 - 1) // 128:
        raise ValueError(
            "interleaved top-k cache pages must fit positive int32 token slots"
        )
    world = int(runtime.world_size)
    query = InterleavedTopKQuery(
        rows=rows, world=world, rank=runtime.rank, cache_pages=cache_pages
    )

    def materialize(selection, device):
        launcher = load_programs(compile_interleaved_topk_slots(world, device.ordinal))
        return _State(launcher, runtime, cache_pages)

    size = sum(buffer.nbytes for buffer in runtime._owned_buffers)
    return Plan(
        contract=_CONTRACT,
        query=query,
        _device=runtime.device,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.comm.interleaved_topk._cute:compile_interleaved_topk_slots",
                world,
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _materialize=materialize,
        _memory_requirements=lambda config, device: MemoryRequirements(
            persistent=(
                PersistentMemory(("pcie-dcp-runtime", id(runtime)), size, size),
            )
        ),
    )


__all__ = ["InterleavedTopKQuery", "plan", "supports_channel"]
