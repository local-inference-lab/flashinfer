"""TP16 BF16 island all-reduce with prepared SM and copy-engine transports."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._island_copy import copy_scratch_bytes
from ._island_rs_cute import (
    GROUP_TIMING_FIELDS,
    HEADER_BYTES,
    MAX_BLOCKS,
    island_rs_peers,
)
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import (
    PCIeOneshotAllReduce,
    _finish_collective_runtime_setup,
    _normalize_device,
    _run_collective_preallocation_setup,
)


SUPPORTED_WORLD_SIZES = (16,)
SUPPORTED_BLOCKS = (1, 2, 4, 8, 16, 32, 64)
_ISLAND_SIZE = 4
_ALIGNMENT = 256


def _validate_output_overlap(inp, out):
    # Exact alias keeps each CTA's writes within its consumed input slice.
    # Shifted overlap can overwrite another CTA's inputs before it reads them.
    if (
        out is not None
        and inp.device == out.device
        and inp.data_ptr() != out.data_ptr()
        and inp.data_ptr() < out.data_ptr() + out.nbytes
        and out.data_ptr() < inp.data_ptr() + inp.nbytes
    ):
        raise ValueError(
            "island reduce-scatter permits exact in-place output or disjoint storage, not partial overlap"
        )


def _validate_disjoint_output(inp, out):
    if (
        out is not None
        and inp.device == out.device
        and inp.data_ptr() < out.data_ptr() + out.nbytes
        and out.data_ptr() < inp.data_ptr() + inp.nbytes
    ):
        raise ValueError("island collective requires disjoint input and output storage")


def _align_up(value: int, alignment: int = _ALIGNMENT) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _threads_from_env() -> int:
    # Default CUDA thread count for the TP16 equal-quarter launch.
    raw = os.getenv("B12X_PCIE_ISLAND_RS_THREADS", "512")
    threads = int(raw)
    if not 32 <= threads <= 1024 or threads % 32:
        raise ValueError(
            "B12X_PCIE_ISLAND_RS_THREADS must be a multiple of 32 in [32, 1024]"
        )
    return threads


def _wait_nanosleep_cycles_from_env() -> int:
    cycles = int(os.getenv("B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES", "24"))
    if not 0 <= cycles <= 1024:
        raise ValueError("B12X_PCIE_ISLAND_RS_NANOSLEEP_CYCLES must be in [0, 1024]")
    return cycles


def _pick_blocks(elements: int) -> int:
    """Return the default launch geometry for a BF16 element count."""

    if elements <= 4096:
        return 8
    return 16


# Route every aligned multiple of 128 BF16 elements (including 1x3584 and 1x7168)
# through equal-quarter ownership so PCIe RO=1 all-push and fused V4 kernels apply.
CROSSOVER_ELEMENTS = 0
PREFERRED_ALIGNMENT_ELEMENTS = 128


class PCIeIslandRSAllReduce:
    """Equal-quarter BF16 TP16 all-reduce with no leader hotspot.

    Every rank owns one quarter of the vector and reaches six peers: the three
    other lanes of its four-GPU island, and the same lane in the three other
    islands. Peer degree is bounded at six, and no rank carries the whole
    vector on behalf of the others.
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_elements: int,
        blocks: Optional[int] = None,
        ipc: object = None,
    ) -> None:
        """Allocate an ordered channel with a fixed physical CUDA grid.

        Automatic grids use at most 64 CTAs and fit one CTA per SM on every
        participating device; preparation tunes their active participation.
        Explicit ``blocks`` retains full participation unless the prepared
        call also declares ``active_blocks``. ``ipc`` supplies the shared slab
        allocator used by the hierarchical runtime.
        """
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                f"island reduce-scatter requires TP16, got TP{self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("island reduce-scatter requires a CUDA device")
        if blocks is not None and blocks not in SUPPORTED_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        if max_elements <= 0 or max_elements % 2:
            raise ValueError("max_elements must be a positive even count")

        self.max_elements = int(max_elements)
        self.blocks_override = blocks is not None
        sms = _run_collective_preallocation_setup(
            owner="PCIe island reduce-scatter capability query",
            exchange_group=self.group,
            setup=lambda: int(
                torch.cuda.get_device_properties(self.device).multi_processor_count
            ),
        )
        limits = [None] * self.world_size
        dist.all_gather_object(limits, (sms, blocks), group=self.group)
        if len({requested for _, requested in limits}) != 1:
            raise ValueError("island ranks must request the same physical grid")
        resident_limit = min(int(sms) for sms, _ in limits)
        automatic = min(MAX_BLOCKS, 1 << (resident_limit.bit_length() - 1))
        self.blocks = automatic if blocks is None else int(blocks)
        if self.blocks > resident_limit:
            raise ValueError(
                "island physical grid must fit one CTA per SM on every rank"
            )
        self.threads = _threads_from_env()
        self.threads_override = "B12X_PCIE_ISLAND_RS_THREADS" in os.environ
        self.wait_nanosleep_cycles = _wait_nanosleep_cycles_from_env()

        max_pairs = self.max_elements // 2
        # Quarter stride in BF16x2 words. Every transient region has two slots
        # selected by the device generation so a rank cannot overwrite data a
        # slower peer is still consuming from the preceding invocation.
        self.quarter_capacity = _align_up(
            (max_pairs + _ISLAND_SIZE - 1) // _ISLAND_SIZE, 8
        )
        quarter_bytes = self.quarter_capacity * 4
        vector_bytes = quarter_bytes * _ISLAND_SIZE
        self.stage_offset = _align_up(HEADER_BYTES)
        self.part_offset = _align_up(self.stage_offset + 2 * vector_bytes)
        self.final_offset = _align_up(self.part_offset + 2 * quarter_bytes)
        self.slab_bytes = _align_up(
            self.final_offset + 2 * vector_bytes
        ) + copy_scratch_bytes(self.quarter_capacity)

        self._ipc = CudaRTLibrary() if ipc is None else ipc
        self._ipc.cudaSetDevice(self.device.index or 0)
        self._slab_ptrs: tuple[int, ...] = ()
        self._local_ptr = 0
        self._remote_ptrs: list[int] = []
        self._closed = False
        self._group_timing_records = None
        self._packet_transport = None
        self._mapped_peers = island_rs_peers(self.rank, self.world_size)

        shared = PCIeOneshotAllReduce._allocate_shared_buffer(
            exchange_group,
            self.slab_bytes,
            zero_fill=True,
            ipc=self._ipc,
            peer_ranks=self._mapped_peers,
        )
        self._local_ptr = shared.local_ptr
        self._remote_ptrs = list(shared.remote_ptrs)
        self._slab_ptrs = shared.peer_ptrs

        # The native launcher is resolved and retained by the prepared plan;
        # construction allocates only the actual communicator-owned slabs.
        init_error: BaseException | None = None

        def detach_shared_ownership() -> None:
            self._slab_ptrs = ()
            self._remote_ptrs.clear()
            self._local_ptr = 0

        _finish_collective_runtime_setup(
            owner="PCIe island reduce-scatter",
            exchange_group=exchange_group,
            ipc=self._ipc,
            shared=shared,
            local_error=init_error,
            detach_shared_ownership=detach_shared_ownership,
        )

    @property
    def mapped_peer_count(self) -> int:
        return len(self._remote_ptrs)

    @property
    def mapped_peers(self) -> tuple[int, ...]:
        """Group ranks whose CUDA IPC allocations are mapped by this rank."""

        return self._mapped_peers

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        return (
            not self._closed
            and inp.device == self.device
            and inp.dtype == torch.bfloat16
            and inp.is_contiguous()
            and 0 < inp.numel() <= self.max_elements
            and inp.numel() % 2 == 0
            and inp.data_ptr() % 4 == 0
        )

    def should_reduce_scatter(self, inp: torch.Tensor) -> bool:
        if not self.should_allreduce(inp) or inp.ndim < 1:
            return False
        if (
            inp.shape[0] % self.world_size != 0
            or inp.numel() % (self.world_size * 2) != 0
        ):
            return False
        shard_pairs = inp.numel() // (2 * self.world_size)
        return shard_pairs <= self.quarter_capacity // 4

    def should_all_gather(self, inp: torch.Tensor) -> bool:
        if (
            self._closed
            or inp.device != self.device
            or inp.dtype != torch.bfloat16
            or not inp.is_contiguous()
            or inp.ndim < 1
            or inp.numel() <= 0
            or inp.numel() % 2 != 0
            or inp.data_ptr() % 4 != 0
        ):
            return False
        total_elems = inp.numel() * self.world_size
        shard_pairs = inp.numel() // 2
        return (
            total_elems <= self.max_elements
            and shard_pairs <= self.quarter_capacity // 4
        )

    def should_row_to_col_all_to_all(self, inp: torch.Tensor) -> bool:
        if (
            self._closed
            or inp.device != self.device
            or inp.dtype != torch.bfloat16
            or not inp.is_contiguous()
            or inp.ndim != 2
            or inp.shape[0] <= 0
            or inp.shape[1] <= 0
            or inp.shape[1] % (self.world_size * 2) != 0
            or inp.data_ptr() % 4 != 0
        ):
            return False
        return inp.numel() * 2 <= 2 * self.quarter_capacity * 4

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        blocks: Optional[int] = None,
        active_blocks: Optional[int] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del channel_id
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy island reduce-scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if tuple(inp.shape) != tuple(state.query.call["inp"]["shape"]):
            raise ValueError("input shape differs from the prepared island plan")
        if out is None:
            out = torch.empty_like(inp)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or out.shape != inp.shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError("output must match input and be 4-byte aligned")
        _validate_output_overlap(inp, out)
        selected = int(state.query.call["blocks"])
        if blocks is not None and int(blocks) != selected:
            raise ValueError("blocks differs from the prepared island plan")
        if selected not in SUPPORTED_BLOCKS or selected > MAX_BLOCKS:
            raise ValueError(f"blocks must be one of {SUPPORTED_BLOCKS}")
        participating = int(state.query.call["active_blocks"])
        if active_blocks is not None and int(active_blocks) != participating:
            raise ValueError("active_blocks differs from the prepared island plan")
        return self._run_prepared(
            state.launcher(), inp, out, selected, stream, active_blocks=participating
        )

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del channel_id
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeIslandRSAllReduce.reduce_scatter":
            raise ValueError("plan is not prepared for island reduce_scatter")
        if not self.should_reduce_scatter(inp):
            raise ValueError(
                "input does not satisfy island reduce_scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if tuple(inp.shape) != tuple(state.query.call["inp"]["shape"]):
            raise ValueError("input shape differs from the prepared island plan")
        expected_shape = (inp.shape[0] // self.world_size, *inp.shape[1:])
        if out is None:
            out = torch.empty(expected_shape, dtype=inp.dtype, device=inp.device)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or tuple(out.shape) != expected_shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError(
                "reduce_scatter output must match shard shape and be 4-byte aligned"
            )
        _validate_disjoint_output(inp, out)
        return self._run_prepared(
            state.launcher(),
            inp,
            out,
            int(state.query.call["blocks"]),
            stream,
            active_blocks=int(state.query.call["active_blocks"]),
        )

    def all_gather(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del channel_id
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeIslandRSAllReduce.all_gather":
            raise ValueError("plan is not prepared for island all_gather")
        if not self.should_all_gather(inp):
            raise ValueError(
                "input does not satisfy island all_gather requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if tuple(inp.shape) != tuple(state.query.call["inp"]["shape"]):
            raise ValueError("input shape differs from the prepared island plan")
        expected_shape = (self.world_size, *inp.shape)
        if out is None:
            out = torch.empty(expected_shape, dtype=inp.dtype, device=inp.device)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or tuple(out.shape) != expected_shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError(
                "all_gather output must match [world, *shard] and be 4-byte aligned"
            )
        _validate_disjoint_output(inp, out)
        return self._run_prepared(
            state.launcher(),
            inp,
            out,
            int(state.query.call["blocks"]),
            stream,
            active_blocks=int(state.query.call["active_blocks"]),
        )

    def row_to_col_all_to_all(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
        stream: object = None,
        channel_id: Optional[str] = None,
    ) -> torch.Tensor:
        del channel_id
        state = require_prepared(plan, "comm.pcie", self.device)
        state.require_runtime(self)
        if state.query.surface != "PCIeIslandRSAllReduce.row_to_col_all_to_all":
            raise ValueError("plan is not prepared for island row_to_col_all_to_all")
        if not self.should_row_to_col_all_to_all(inp):
            raise ValueError(
                "input does not satisfy island row_to_col_all_to_all requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if tuple(inp.shape) != tuple(state.query.call["inp"]["shape"]):
            raise ValueError("input shape differs from the prepared island plan")
        st, full_c = inp.shape
        c = full_c // self.world_size
        expected_shape = (st * self.world_size, c)
        if out is None:
            out = torch.empty(expected_shape, dtype=inp.dtype, device=inp.device)
        if (
            out.dtype != inp.dtype
            or out.device != inp.device
            or tuple(out.shape) != expected_shape
            or not out.is_contiguous()
            or out.data_ptr() % 4 != 0
        ):
            raise ValueError(
                "row_to_col_all_to_all output must match [world * st, c] and be 4-byte aligned"
            )
        _validate_disjoint_output(inp, out)
        return self._run_prepared_a2a(
            state.launcher(),
            inp,
            out,
            st,
            c,
            int(state.query.call["blocks"]),
            stream,
            active_blocks=int(state.query.call["active_blocks"]),
        )

    def _run_prepared_a2a(
        self, launcher, inp, out, st, c, blocks, stream, *, active_blocks=None
    ):
        with torch.cuda.device(self.device):
            launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self.stage_offset,
                self.part_offset,
                self.final_offset,
                self.quarter_capacity,
                st,
                c,
                blocks,
                stream,
                active_blocks=active_blocks,
            )
        return out

    def packet_transport(self):
        from ._island_copy import PacketTransport

        if self._packet_transport is None:
            self._packet_transport = PacketTransport(self.device, self._ipc)
        return self._packet_transport

    def _run_prepared(self, launcher, inp, out, blocks, stream, *, active_blocks=None):
        """Invoke an already-prepared native launcher after caller validation."""
        records = self._group_timing_records
        with torch.cuda.device(self.device):
            launcher(
                self._slab_ptrs,
                inp.data_ptr(),
                out.data_ptr(),
                self.stage_offset,
                self.part_offset,
                self.final_offset,
                self.quarter_capacity,
                inp.numel(),
                blocks,
                stream,
                active_blocks=active_blocks,
                timing_address=0 if records is None else records.data_ptr(),
                timing_calls=0 if records is None else records.shape[0],
                timing_blocks=0 if records is None else records.shape[1],
                timing_slices=0 if records is None else records.shape[3],
            )
        return out

    def bind_group_timing(self, records: torch.Tensor) -> None:
        """Retain caller-owned diagnostic records shaped as calls, CTAs, groups, slices, fields."""
        if (
            records.device != self.device
            or records.dtype != torch.int64
            or not records.is_contiguous()
            or records.ndim != 5
            or records.shape[2] != 4
            or records.shape[4] != len(GROUP_TIMING_FIELDS)
            or not 1 <= records.shape[0] <= 0x7FFFFFFF
            or not 1 <= records.shape[1] <= self.blocks
            or not 1 <= records.shape[3] <= 0x7FFFFFFF
        ):
            raise ValueError(
                "group timing records require contiguous device int64 storage with shape [calls, CTAs, 4, slices, 16]"
            )
        self._group_timing_records = records

    def prepare_channels(self, channel_ids: Sequence[str]) -> None:
        """Accept owner names for the single ordered collective channel."""

        del channel_ids

    def for_stream(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ) -> "PCIeIslandRSAllReduce":
        # Binding is validated at launch through the explicit stream argument.
        # Owner names are accepted because this runtime has one serialized
        # channel rather than per-owner storage.
        del channel_id
        if stream is not None and not hasattr(stream, "cuda_stream"):
            int(stream)
        return self

    @contextmanager
    def capture(
        self,
        stream: object = None,
        *,
        channel_id: Optional[str] = None,
    ):
        self.for_stream(stream, channel_id=channel_id)
        if stream is None:
            yield self
        else:
            torch_stream = (
                stream
                if hasattr(stream, "cuda_stream")
                else torch.cuda.ExternalStream(int(stream), device=self.device)
            )
            with torch.cuda.stream(torch_stream):
                yield self

    def register_graph_buffers(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._group_timing_records = None
        dist.barrier(group=self.group)
        self._slab_ptrs = ()
        for ptr in self._remote_ptrs:
            self._ipc.cudaIpcCloseMemHandle(ptr)
        self._remote_ptrs.clear()
        dist.barrier(group=self.group)
        if self._local_ptr:
            self._ipc.cudaFree(self._local_ptr)
            self._local_ptr = 0
        dist.barrier(group=self.group)
        if self._packet_transport is not None:
            self._packet_transport.close()
            self._packet_transport = None


__all__ = [
    "CROSSOVER_ELEMENTS",
    "PCIeIslandRSAllReduce",
    "PREFERRED_ALIGNMENT_ELEMENTS",
    "SUPPORTED_WORLD_SIZES",
]
