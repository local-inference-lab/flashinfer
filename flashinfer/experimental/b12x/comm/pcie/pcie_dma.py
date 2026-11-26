"""CE-driven PCIe ring allreduce for prefill-size tensors.

NCCL's SM-copy transport sustains ~34 GB/s bus bandwidth on this fabric
while CE peer copies run at ~56 GB/s on every ring hop concurrently
(including the two root-complex crossings, which each own a partition
uplink per direction). This runtime drives a classic reduce-scatter +
all-gather ring where the data plane is CE copies and the SM only
synchronizes (monotonic flag kernels) and reduces, so captured graphs
replay without host patching.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager, suppress
from typing import Optional
from weakref import finalize

import cuda.bindings.driver as cuda
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from b12x.preparation.types import Plan, require_prepared

from ._dma_kernels import DmaKernels
from ._cuda_ipc import CudaRTLibrary
from .pcie_oneshot import PCIeOneshotAllReduce, _normalize_device

logger = logging.getLogger(__name__)

SUPPORTED_DTYPES = {
    torch.bfloat16: 0,
    torch.float16: 1,
    torch.float32: 2,
}
SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8, 10, 16)
FLAG_STRIDE = 128
FLAG_SLOTS = 256
_CROSS_FLAG_BASE = 64
_DONE_INTRA_SLOT = FLAG_SLOTS - 4
_DONE_CROSS_SLOT = FLAG_SLOTS - 3
MAX_PIECES = 8
SCRATCH_ALIGN = 256
FP8_QUANT_BLOCK = 128


def _create_high_priority_stream(device: torch.device) -> torch.cuda.ExternalStream:
    """Create a dedicated non-pooled high-priority CUDA stream."""
    with torch.cuda.device(device):
        pstatus, _least, greatest = cuda.cuCtxGetStreamPriorityRange()
        if pstatus == cuda.CUresult.CUDA_SUCCESS and greatest < 0:
            status, handle = cuda.cuStreamCreateWithPriority(
                cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, greatest
            )
        else:
            status, handle = cuda.cuStreamCreate(
                cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
            )
        if status != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"creating high-priority DMA stream failed: {status}")
        try:
            stream = torch.cuda.ExternalStream(int(handle), device=device)
        except BaseException:
            cuda.cuStreamDestroy(handle)
            raise

    def release() -> None:
        with torch.cuda.device(device):
            cuda.cuStreamDestroy(handle)

    stream._b12x_release = finalize(stream, release)
    stream._b12x_release.atexit = False
    return stream


def _eager_replay_capacities(
    max_bytes: int,
    min_bytes: int,
    itemsize: int,
    world_size: int,
    *,
    max_elements: int | None = None,
) -> tuple[int, ...]:
    """Fixed element capacities with eight-value shards on every supported ring."""
    multiple = world_size * 8
    elements = max_bytes // itemsize
    if max_elements is not None:
        if max_elements <= 0 or max_elements > elements:
            raise ValueError("DMA dtype capacity must fit the positive ring byte bound")
        elements = max_elements
    elements -= elements % multiple
    if elements <= 0:
        raise ValueError("DMA replay capacity is too small")
    size_bytes = 1 << (max(1 << 20, min_bytes) - 1).bit_length()
    capacities = []
    while size_bytes < elements * itemsize:
        capacity = size_bytes // itemsize
        capacity -= capacity % multiple
        if capacity > 0:
            capacities.append(capacity)
        size_bytes *= 2
    return (*capacities, elements)


def _fp8_mode() -> str:
    """Opt-in compressed wire transport mode.

    "ag" (also "1"): keep the saturated bf16 reduce-scatter ring and
    quantize only the allgather phase. Final values quantize exactly once
    at their owner and are forwarded verbatim around the ring, so the
    error cost is a single rounding while AG wire bytes halve.

    "ring": quantize every reduce-scatter hop and the allgather payload,
    keeping the saturated neighbor-only topology while halving both phases.

    "a2a": quantize-once all-to-all (two roundings, half the wire in both
    phases).

    "i8" / "i8_ring" / "i8_a2a": the matching topology with a symmetric
    signed-INT8 payload. Both codecs use the same layout: one payload byte
    per value plus one fp32 scale per 128 values.

    "mx" / "mx_ring" / "mx_a2a": standard MXFP8, with E4M3 payload values
    and one E8M0 scale per 32 values. Four scale bytes per 128 values retain
    the same wire footprint as the other compressed codecs.

    Every compressed mode materializes the locally owned reduced shard through
    the same wire payload as its peers. An all-reduce result must be
    rank-identical; retaining a pre-wire BF16 owner shard while peers dequantize
    that shard gives every TP rank a different replicated activation.
    """

    return _normalize_fp8_mode(os.getenv("B12X_PCIE_DMA_FP8", "0"))


def _normalize_fp8_mode(value: str | None) -> str:
    raw = (value or "").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return ""
    if raw in ("a2a", "ring"):
        return raw
    if raw in (
        "i8",
        "int8",
        "i8_ag",
        "i8-ag",
        "ag_i8",
        "int8_ag",
        "int8-ag",
    ):
        return "i8"
    if raw in ("i8_ring", "i8-ring", "int8_ring", "int8-ring", "ring_i8"):
        return "i8_ring"
    if raw in ("i8_a2a", "i8-a2a", "int8_a2a", "int8-a2a", "a2a_i8"):
        return "i8_a2a"
    if raw in (
        "mx",
        "mxfp8",
        "mx_ag",
        "mx-ag",
        "mxfp8_ag",
        "mxfp8-ag",
        "ag_mx",
    ):
        return "mx"
    if raw in (
        "mx_ring",
        "mx-ring",
        "mxfp8_ring",
        "mxfp8-ring",
        "ring_mx",
    ):
        return "mx_ring"
    if raw in (
        "mx_a2a",
        "mx-a2a",
        "mxfp8_a2a",
        "mxfp8-a2a",
        "a2a_mx",
    ):
        return "mx_a2a"
    if raw in ("ag", "1"):
        return "ag"
    raise ValueError(f"unrecognized PCIe DMA wire mode: {value!r}")


def _load_kernels(ipc: CudaRTLibrary) -> DmaKernels:
    """Construct an unprepared transport façade for this IPC channel."""

    return DmaKernels(ipc)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class PCIeDmaAllReduce:
    """Single-channel ring allreduce over IPC scratch buffers.

    A channel is a single ordered stream context; concurrent use from
    multiple CUDA streams needs separate channels (same contract as the
    oneshot runtime).
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device | int | str,
        max_bytes: int,
        ext_module=None,
        fp8: Optional[str] = None,
        ipc=None,
    ) -> None:
        # Kept only as a source-compatible keyword for callers that used to
        # inject the removed C++ extension.  Device work is always CuTe DSL.
        del ext_module
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        self.device = _normalize_device(device)
        if self.world_size not in SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "PCIe DMA all-reduce supports only the reviewed world sizes "
                f"{SUPPORTED_WORLD_SIZES}, got {self.world_size}"
            )
        if self.device.type != "cuda":
            raise ValueError("PCIe ring allreduce requires a CUDA device")
        self.max_bytes = int(max_bytes)
        # Callers may supply the shared-memory allocator, e.g. VMM storage
        # that an exporter retains at recorded addresses.
        if ipc is None:
            ipc = CudaRTLibrary()
            ipc.cudaSetDevice(self.device.index or 0)
        self._ipc = ipc
        self._kernels = _load_kernels(self._ipc)
        self._closed = False
        self._capture_context_depth = 0
        self._eager_replays: dict[
            torch.dtype,
            tuple[tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph], ...],
        ] = {}

        self.shard_capacity = _align_up(
            (self.max_bytes + self.world_size - 1) // self.world_size, SCRATCH_ALIGN
        )
        # Explicit argument wins over the environment so integrations can
        # plumb the mode through their own configuration.
        self._fp8 = _normalize_fp8_mode(fp8) if fp8 is not None else _fp8_mode()
        if self.world_size == 16 and self._fp8:
            raise ValueError(
                "TP16 4x4 PCIe DMA ring supports lossless BF16/FP16/FP32 only; "
                f"got compressed mode {self._fp8!r}"
            )
        self._owned_streams: list[torch.cuda.ExternalStream] = []
        self._local_accum: torch.Tensor | None = None
        flags_bytes = FLAG_SLOTS * FLAG_STRIDE
        if self.world_size == 16:
            self._island, self._lane = divmod(self.rank, 4)
            self._intra_nxt = 4 * self._island + (self._lane + 1) % 4
            self._intra_prv = 4 * self._island + (self._lane - 1) % 4
            self._cross_nxt = 4 * ((self._island + 1) % 4) + self._lane
            self._cross_prv = 4 * ((self._island - 1) % 4) + self._lane
            peer_ranks = (
                self._intra_nxt,
                self._intra_prv,
                self._cross_nxt,
                self._cross_prv,
            )
            counters_bytes = _align_up(3 * FLAG_SLOTS * 4, SCRATCH_ALIGN)
            header_bytes = _align_up(flags_bytes + counters_bytes, SCRATCH_ALIGN)
            slab_bytes = header_bytes + 16 * self.shard_capacity
            self._slab_nbytes = slab_bytes
            self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
                exchange_group,
                slab_bytes,
                zero_fill=True,
                ipc=self._ipc,
                peer_ranks=peer_ranks,
            )
            self._flags_base = list(self._slab.peer_ptrs)
            self._scratch_base = [
                ptr + header_bytes if ptr else 0 for ptr in self._slab.peer_ptrs
            ]
            # Monotonic counters live inside the zero-initialized IPC slab so
            # exported cores retain zero-init semantics with no extra regions.
            self._send_counters = None
            self._wait_counters = None
            self._wait_done = None
            self._send_counters_base = self._slab.local_ptr + flags_bytes
            self._wait_counters_base = self._send_counters_base + FLAG_SLOTS * 4
            self._wait_done_base = self._wait_counters_base + FLAG_SLOTS * 4
            # Local-only ping-pong accumulators for RS (never IPC-exported).
            self._local_accum = torch.empty(
                8 * self.shard_capacity, dtype=torch.uint8, device=self.device
            )
            self._copy_stream = _create_high_priority_stream(self.device)
            self._flag_stream = _create_high_priority_stream(self.device)
            self._owned_streams.extend((self._copy_stream, self._flag_stream))
            self._ag_copy_stream = None
            self._ag_flag_stream = None
            self._piece_events = [torch.cuda.Event() for _ in range(MAX_PIECES)]
            self._p2_piece_events = [torch.cuda.Event() for _ in range(MAX_PIECES)]
            self._copied_intra = [torch.cuda.Event() for _ in range(3 * MAX_PIECES)]
            self._copied_cross = [torch.cuda.Event() for _ in range(3 * MAX_PIECES)]
            self._copied_events = []
            self._input_ready = torch.cuda.Event()
            self._p1_done = torch.cuda.Event()
            self._ag_ready = None
            self._a2a_qdone = []
            self._a2a_ownq = []
        else:
            steps = 2 * (self.world_size - 1)
            slab_bytes = flags_bytes + steps * self.shard_capacity
            self._slab_nbytes = slab_bytes
            self._slab = PCIeOneshotAllReduce._allocate_shared_buffer(
                exchange_group, slab_bytes, zero_fill=True, ipc=self._ipc
            )
            self._flags_base = list(self._slab.peer_ptrs)
            self._scratch_base = [ptr + flags_bytes for ptr in self._slab.peer_ptrs]
            # Device-resident monotonic counters: one per flag slot for the
            # publisher role and one for the waiter role.
            self._send_counters = torch.zeros(
                FLAG_SLOTS, dtype=torch.int32, device=self.device
            )
            self._wait_counters = torch.zeros(
                FLAG_SLOTS, dtype=torch.int32, device=self.device
            )
            self._wait_done = torch.zeros(
                FLAG_SLOTS, dtype=torch.int32, device=self.device
            )
            self._send_counters_base = self._send_counters.data_ptr()
            self._wait_counters_base = self._wait_counters.data_ptr()
            self._wait_done_base = self._wait_done.data_ptr()
            self._copy_stream = torch.cuda.Stream(device=self.device)
            self._flag_stream = torch.cuda.Stream(device=self.device)
            # Separate CE/flag streams for the a2a broadcast phase so allgather
            # traffic overlaps reduce-scatter traffic instead of queueing
            # behind it.
            self._ag_copy_stream = torch.cuda.Stream(device=self.device)
            self._ag_flag_stream = torch.cuda.Stream(device=self.device)
            # Persistent cross-stream events: captured graphs keep references to
            # recorded events, so per-call temporaries must not be destroyed.
            self._piece_events = [torch.cuda.Event() for _ in range(MAX_PIECES)]
            self._copied_events = [
                torch.cuda.Event()
                for _ in range(2 * (self.world_size - 1) * MAX_PIECES)
            ]
            self._input_ready = torch.cuda.Event()
            self._ag_ready = torch.cuda.Event()
            self._a2a_qdone = [torch.cuda.Event() for _ in range(MAX_PIECES)]
            self._a2a_ownq = [torch.cuda.Event() for _ in range(MAX_PIECES)]
        self._fp8_stage = None
        self._fp8_stage_stride = 0
        if self._fp8:
            max_shard_elems = self.max_bytes // 2 // self.world_size
            stride = _align_up(
                max_shard_elems + max_shard_elems // FP8_QUANT_BLOCK * 4,
                SCRATCH_ALIGN,
            )
            self._fp8_stage = torch.empty(
                self.world_size * stride, dtype=torch.uint8, device=self.device
            )
            self._fp8_stage_stride = stride
        self._pieces_override = int(os.getenv("B12X_PCIE_DMA_PIECES", "0"))
        self._a2a_chunks_override = int(os.getenv("B12X_PCIE_DMA_A2A_CHUNKS", "0"))
        self.min_bytes = 0
        wire_modes = {
            "i8": "int8-ag",
            "i8_ring": "int8-ring",
            "i8_a2a": "int8-a2a",
            "mx": "mxfp8-ag",
            "mx_ring": "mxfp8-ring",
            "mx_a2a": "mxfp8-a2a",
        }
        self.wire_mode = wire_modes.get(
            self._fp8, f"fp8-{self._fp8}" if self._fp8 else "bf16"
        )
        logger.debug("[PCIe DMA allreduce] wire mode: %s", self.wire_mode)

    def _flag_ptr(self, rank: int, slot: int) -> int:
        return self._flags_base[rank] + slot * FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _send_counter_ptr(self, slot: int) -> int:
        return self._send_counters_base + slot * 4

    def _wait_counter_ptr(self, slot: int) -> int:
        return self._wait_counters_base + slot * 4

    def _wait_done_ptr(self, slot: int) -> int:
        return self._wait_done_base + slot * 4

    def _scratch_ptr(self, rank: int, step: int) -> int:
        return self._scratch_base[rank] + step * self.shard_capacity

    def _pick_pieces(self, shard_elems: int, shard_bytes: int) -> int:
        override = self._pieces_override
        # pieces=2 measured best at every size (deeper chunking pays an
        # extra wait+add launch chain per piece on the main stream).
        candidates = (override,) if 1 <= override <= MAX_PIECES else (2,)
        for pieces in candidates:
            if shard_elems % (pieces * 8) == 0 and shard_bytes // pieces >= 512 << 10:
                return pieces
        return 1

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        numel = inp.numel()
        if numel <= 0 or numel % (self.world_size * 8) != 0:
            return False
        size_bytes = numel * inp.element_size()
        if size_bytes < self.min_bytes:
            return False
        return inp.is_contiguous() and size_bytes <= self.max_bytes

    def should_all_gather(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        numel = inp.numel()
        if numel <= 0 or numel % 8 != 0:
            return False
        size_bytes = numel * inp.element_size()
        return inp.is_contiguous() and size_bytes <= self.shard_capacity

    def should_reduce_scatter(self, inp: torch.Tensor) -> bool:
        if self._closed or inp.device != self.device:
            return False
        if inp.dtype not in SUPPORTED_DTYPES:
            return False
        if inp.ndim < 1 or inp.shape[0] % self.world_size != 0:
            return False
        numel = inp.numel()
        if numel <= 0 or numel % (self.world_size * 8) != 0:
            return False
        size_bytes = numel * inp.element_size()
        return inp.is_contiguous() and size_bytes <= self.max_bytes

    def prepare_eager_replay(
        self, dtype: torch.dtype, *, max_elements: int | None = None
    ) -> None:
        """Precapture bounded-capacity lossless rings for later eager calls.

        Call collectively before serving. Powers-of-two capacities, capped at
        max_bytes, bound padding traffic without capturing on a request path.
        Graphs borrow one private input/output allocation per dtype. Live sizes
        select a covering graph, not a compilation or allocation cache entry.
        The private output cannot alias a retained result.
        max_elements bounds this dtype independently of the ring byte capacity,
        so preparing FP32 reductions does not double BF16 replay storage.
        """
        if self._closed or self._fp8:
            raise ValueError("eager replay requires an open lossless DMA ring")
        if dtype not in SUPPORTED_DTYPES:
            raise TypeError(f"unsupported DMA replay dtype: {dtype}")
        if max_elements is None:
            elements = self.max_bytes // dtype.itemsize
        else:
            if isinstance(max_elements, bool) or not isinstance(max_elements, int):
                raise TypeError("DMA replay max_elements must be an integer")
            if max_elements * dtype.itemsize > self.max_bytes:
                raise ValueError(
                    f"DMA replay max_elements={max_elements} for {dtype} "
                    f"exceeds max_bytes={self.max_bytes}"
                )
            elements = max_elements
        alignment = self.world_size * 8
        elements -= elements % alignment
        if elements <= 0:
            raise ValueError(
                f"DMA replay capacity must contain at least {alignment} elements"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare DMA eager replay before CUDA graph capture")
        capacities = _eager_replay_capacities(
            self.max_bytes,
            self.min_bytes,
            dtype.itemsize,
            self.world_size,
            max_elements=max_elements,
        )
        if dtype in self._eager_replays:
            if capacities[-1] != self._eager_replays[dtype][-1][0].numel():
                raise ValueError(
                    "DMA dtype replay capacity cannot change after preparation"
                )
            return
        elements = capacities[-1]
        with torch.cuda.device(self.device):
            source = torch.zeros(elements, dtype=dtype, device=self.device)
            result = torch.empty_like(source)
            replays = []
            for capacity in capacities:
                source_view, result_view = source[:capacity], result[:capacity]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._all_reduce_on_device(source_view, out=result_view)
                replays.append((source_view, result_view, graph))
        self._eager_replays[dtype] = tuple(replays)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run only the already-prepared fixed DMA channel."""

        state = require_prepared(plan, "comm.pcie", self.device)
        if not hasattr(state, "require_runtime") or not hasattr(state, "run"):
            raise TypeError("plan does not contain a prepared DMA channel")
        state.require_runtime(self)
        if out is None:
            out = torch.empty_like(inp)
        return state.run(inp, out=out)

    def all_gather(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Experimental pure Copy-Engine ring all-gather `[shape] -> [world_size, *shape]`.

        Experimental API: opt in by preparing a DMA plan for this channel via
        ``b12x.comm.pcie.plan(query, runtime=channel)`` and passing the
        materialized ``plan`` here.

        Limitations:
            - Supported dtypes: ``torch.bfloat16``, ``torch.float16``, ``torch.float32``.
            - ``inp`` (and ``out`` when provided) must be contiguous on this
              channel's CUDA device, with ``inp.numel()`` nonzero and divisible by 8.
            - ``inp.nbytes`` must not exceed ``self.shard_capacity``.
            - ``out`` must have shape ``(world_size, *inp.shape)`` and storage
              strictly disjoint from ``inp`` (in-place / aliased outputs are rejected).
            - Channel submissions must be serialized across invocations and graphs.
        """

        state = require_prepared(plan, "comm.pcie", self.device)
        if not hasattr(state, "require_runtime") or not hasattr(
            state, "run_all_gather"
        ):
            raise TypeError("plan does not contain a prepared DMA channel")
        state.require_runtime(self)
        if out is None:
            out = torch.empty(
                (self.world_size, *inp.shape), dtype=inp.dtype, device=inp.device
            )
        return state.run_all_gather(inp, out=out)

    def reduce_scatter(
        self,
        inp: torch.Tensor,
        *,
        plan: Plan,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Experimental Copy-Engine ring reduce-scatter `[world_size * rows, *tail] -> [rows, *tail]`.

        Runs ``world_size - 1`` Copy-Engine peer-copy and ``dma_add`` reduction
        hops across the channel's preallocated IPC scratch slots and writes
        this rank's summed shard directly into ``out`` without a trailing
        all-gather phase.

        Experimental API: opt in by preparing a DMA plan for this channel via
        ``b12x.comm.pcie.plan(query, runtime=channel)`` and passing the
        materialized ``plan`` here.

        Limitations:
            - Supported dtypes: ``torch.bfloat16``, ``torch.float16``, ``torch.float32``.
            - ``inp`` (and ``out`` when provided) must be contiguous on this
              channel's CUDA device, with ``inp.ndim >= 1``,
              ``inp.shape[0] % world_size == 0``, ``inp.numel()`` nonzero, and
              ``inp.numel() % (world_size * 8) == 0``.
            - ``inp.nbytes`` must not exceed ``self.max_bytes``.
            - ``out`` must have shape ``(inp.shape[0] // world_size, *inp.shape[1:])``
              and storage strictly disjoint from ``inp`` (in-place or aliased
              outputs are rejected).
            - Executes lossless ``dma_add`` ring reductions regardless of any
              compressed ``all_reduce`` wire mode configured on the channel.
            - Channel submissions must be serialized across invocations and graphs.
        """

        state = require_prepared(plan, "comm.pcie", self.device)
        if not hasattr(state, "require_runtime") or not hasattr(
            state, "run_reduce_scatter"
        ):
            raise TypeError("plan does not contain a prepared DMA channel")
        state.require_runtime(self)
        if not self.should_reduce_scatter(inp):
            raise ValueError(
                "input does not satisfy ring reduce_scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device})"
            )
        if out is None:
            out = torch.empty(
                (inp.shape[0] // self.world_size, *inp.shape[1:]),
                dtype=inp.dtype,
                device=self.device,
            )
        return state.run_reduce_scatter(inp, out=out)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        """Retain the ordered DMA channel while recording a CUDA graph.

        Persistent events join the auxiliary streams to the captured stream,
        and device counters advance during replay. Recording needs no counter
        reset or additional channel storage. Graphs sharing this owner must
        replay serially and drain before the owner closes.
        """
        if self._closed:
            raise RuntimeError("PCIe DMA channel is closed")
        if self._capture_context_depth:
            raise RuntimeError("overlapping PCIe DMA capture contexts are not allowed")
        if stream is not None and stream.device != self.device:
            raise ValueError(
                "PCIe DMA capture stream must belong to the channel device"
            )
        self._capture_context_depth = 1
        try:
            yield self
        finally:
            self._capture_context_depth = 0

    def _run_prepared(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor], state
    ) -> torch.Tensor:
        state.require_runtime(self)
        torch.cuda.set_device(self.device)
        if not self.should_allreduce(inp):
            raise ValueError(
                "input does not satisfy ring allreduce requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        if out is None:
            raise ValueError(
                "prepared DMA all_reduce requires a caller-owned output tensor"
            )
        if (
            out.shape != inp.shape
            or out.dtype != inp.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "output must match input shape/dtype/device and be contiguous"
            )
        replays = self._eager_replays.get(inp.dtype)
        if replays is not None and not torch.cuda.is_current_stream_capturing():
            elements = inp.numel()
            replay = next(
                (item for item in replays if item[0].numel() >= elements), None
            )
            if replay is not None:
                source, result, graph = replay
                source[:elements].copy_(inp.view(-1))
                graph.replay()
                out.view(-1).copy_(result[:elements])
                return out
        if self.world_size == 16:
            return self._run_prepared_all_reduce_4x4(inp, out=out)
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem

        compressed_eligible = (
            bool(self._fp8)
            and inp.dtype == torch.bfloat16
            and shard_elems % FP8_QUANT_BLOCK == 0
        )
        int8_wire = compressed_eligible and self._fp8.startswith("i8")
        mxfp8_wire = compressed_eligible and self._fp8.startswith("mx")
        wire_codec = "i8" if int8_wire else "mx" if mxfp8_wire else "e4m3"
        if compressed_eligible and self._fp8 in ("a2a", "i8_a2a", "mx_a2a"):
            return self._all_reduce_fp8(inp, out, shard_elems, wire_codec=wire_codec)
        compressed_ring = compressed_eligible and self._fp8 in (
            "ring",
            "i8_ring",
            "mx_ring",
        )
        compressed_ag = compressed_eligible and self._fp8 in (
            "ag",
            "ring",
            "i8",
            "i8_ring",
            "mx",
            "mx_ring",
        )
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_store = kernels.dma_dequant_store_i8
            dequantize_add_quant = kernels.dma_dequant_add_quant_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_store = kernels.dma_dequant_store_mx
            dequantize_add_quant = kernels.dma_dequant_add_quant_mx
        else:
            quantize = kernels.dma_quant
            dequantize_store = kernels.dma_dequant_store
            dequantize_add_quant = kernels.dma_dequant_add_quant

        base = out.data_ptr()

        # Sub-chunking with a dedicated copy stream keeps the copy engine
        # busy: the CE never waits for a flag round trip or an add because
        # sub-chunk c+1's copy overlaps sub-chunk c's wait+reduce. Deeper
        # chunking amortizes the flag round trip further as long as each
        # piece's copy time dominates the ~5us sub-step overhead.
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        if compressed_ag and (shard_elems // pieces) % FP8_QUANT_BLOCK != 0:
            pieces = 1
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        # Compressed slices are piece-contiguous: [payload][scales] per piece.
        piece_slice_bytes = piece_elems + piece_elems // FP8_QUANT_BLOCK * 4
        steps = 2 * (world - 1)

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream

        # No upfront out.copy_(inp): the first send of each chunk reads the
        # caller's input directly and every reduce-scatter add is a first
        # touch (out = inp + scratch), so the accumulation base folds into
        # the add instead of a full-size copy on the critical path.
        in_base = inp.data_ptr()
        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)

        def piece_ptr(chunk: int, piece: int) -> int:
            return base + chunk * shard_bytes + piece * piece_bytes

        def in_piece_ptr(chunk: int, piece: int) -> int:
            return in_base + chunk * shard_bytes + piece * piece_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        # Events gating each step's send on the previous step's reduce of
        # the same payload piece (persistent; re-recorded per step). Flag
        # kernels run on their own stream, gated per copy by copied[] events,
        # so the copy stream is pure back-to-back CE work: an SM kernel
        # between CE ops stalls the engine for the launch round trip, which
        # is what made deeper sub-chunking regress.
        add_done = self._piece_events
        flag_stream = self._flag_stream
        copied = self._copied_events
        flag_stream.wait_event(self._input_ready)

        def fp8_scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_slice_bytes

        stage = self._fp8_stage.data_ptr() if compressed_ag else 0

        def fp8_stage_piece(chunk: int, piece: int) -> int:
            return stage + chunk * self._fp8_stage_stride + piece * piece_slice_bytes

        for k in range(steps):
            reduce_phase = k < world - 1
            if reduce_phase:
                send_chunk = (rank - k) % world
                recv_chunk = (rank - k - 1) % world
            else:
                send_chunk = (rank + 1 - (k - (world - 1))) % world
                recv_chunk = (rank - (k - (world - 1))) % world
            compressed_reduce = compressed_ring and reduce_phase
            compressed_step = compressed_reduce or (compressed_ag and not reduce_phase)
            if compressed_step and k == world - 1:
                # The AG-only mode quantizes the fully reduced owner chunk
                # here.  The FP8 ring's fused final reduce hop already
                # emitted the same payload.  Both modes forward those bytes
                # verbatim, with no additional all-gather rounding.
                if not compressed_ring:
                    for p in range(pieces):
                        ag_stage = fp8_stage_piece(send_chunk, p)
                        quantize(
                            piece_ptr(send_chunk, p),
                            ag_stage,
                            ag_stage + piece_elems,
                            piece_elems,
                        )
                # Publish the payload before the local materialization so the
                # CE broadcast can overlap this read-only dequant kernel.
                self._ag_ready.record(main)
                # The owner used to retain its pre-wire BF16 shard while the
                # other ranks materialized this same shard from FP8.  That
                # violates the replicated-output contract of all-reduce and
                # lets the next TP layer consume rank-dependent activations.
                # Round-trip the owner through the exact forwarded payload so
                # all ranks receive bit-identical BF16 values for every shard.
                for p in range(pieces):
                    owner_stage = fp8_stage_piece(send_chunk, p)
                    dequantize_store(
                        piece_ptr(send_chunk, p),
                        owner_stage,
                        owner_stage + piece_elems,
                        piece_elems,
                    )
            for p in range(pieces):
                if compressed_reduce:
                    send_src = fp8_stage_piece(send_chunk, p)
                    if k == 0:
                        quantize(
                            in_piece_ptr(send_chunk, p),
                            send_src,
                            send_src + piece_elems,
                            piece_elems,
                        )
                        self._a2a_qdone[p].record(main)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                elif not compressed_step:
                    send_src = (
                        in_piece_ptr(send_chunk, p)
                        if k == 0
                        else piece_ptr(send_chunk, p)
                    )
                    send_bytes = piece_bytes
                    send_dst = scratch_piece(nxt, k, p)
                elif k == world - 1:
                    send_src = fp8_stage_piece(send_chunk, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                else:
                    send_src = fp8_scratch_piece(rank, k - 1, p)
                    send_bytes = piece_slice_bytes
                    send_dst = fp8_scratch_piece(nxt, k, p)
                with torch.cuda.stream(copy_stream):
                    if compressed_reduce:
                        copy_stream.wait_event(
                            self._a2a_qdone[p] if k == 0 else add_done[p]
                        )
                    elif compressed_step and k == world - 1:
                        copy_stream.wait_event(self._ag_ready)
                    elif k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, send_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                wait = (
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                    self._counter_ptr(self._wait_done, slot(k, p)),
                )
                # Always wait in a 1-CTA flag kernel before launching the multi-CTA
                # reduction kernel so overlapping compute on another stream is not
                # starved by 64 spin-waiting CTAs.
                kernels.dma_wait_flag(*wait[:2])
                if reduce_phase:
                    if compressed_reduce:
                        payload = fp8_scratch_piece(rank, k, p)
                        reduced = fp8_stage_piece(recv_chunk, p)
                        dequantize_add_quant(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            payload,
                            payload + piece_elems,
                            reduced,
                            reduced + piece_elems,
                            piece_elems,
                            k == world - 2,
                        )
                    else:
                        kernels.dma_add(
                            piece_ptr(recv_chunk, p),
                            in_piece_ptr(recv_chunk, p),
                            scratch_piece(rank, k, p),
                            piece_elems,
                            dtype_code,
                        )
                elif compressed_step:
                    payload = fp8_scratch_piece(rank, k, p)
                    # Forwarding reads the received FP8 payload verbatim, so
                    # it only depends on the receive flag, not on the local
                    # BF16 materialization below.  Publish readiness before
                    # dequantization to overlap the next hop's CE copy with
                    # this rank's read-only dequant/store.
                    add_done[p].record(main)
                    dequantize_store(
                        piece_ptr(recv_chunk, p),
                        payload,
                        payload + piece_elems,
                        piece_elems,
                    )
                else:
                    kernels.dma_copy(
                        piece_ptr(recv_chunk, p),
                        scratch_piece(rank, k, p),
                        piece_bytes,
                    )
                if reduce_phase or not compressed_step:
                    add_done[p].record(main)

        # Neighbor handshake so the next call (or graph replay) cannot
        # overwrite scratch a lagging neighbor still reads. The main stream
        # must also drain the copy and flag streams before the op is done.
        # Use a fixed slot below FLAG_SLOTS - 1 (reserved for priming) so
        # calls with different piece counts never alias done with data slots.
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = FLAG_SLOTS - 2
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def _run_prepared_all_gather(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor], state
    ) -> torch.Tensor:
        state.require_runtime(self)
        torch.cuda.set_device(self.device)
        if not self.should_all_gather(inp):
            raise ValueError(
                "input does not satisfy ring allgather requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype})"
            )
        expected_shape = (self.world_size, *inp.shape)
        if (
            out is None
            or tuple(out.shape) != expected_shape
            or out.dtype != inp.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"DMA all_gather output must be contiguous {expected_shape} {inp.dtype}"
            )
        in_bytes = inp.numel() * inp.element_size()
        if (
            inp.data_ptr() < out.data_ptr() + out.nbytes
            and out.data_ptr() < inp.data_ptr() + in_bytes
        ):
            raise ValueError(
                "DMA all_gather requires disjoint input and output storage"
            )
        if self.world_size == 16:
            return self._run_prepared_all_gather_4x4(inp, out=out)
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        elem = inp.element_size()
        shard_elems = inp.numel()
        shard_bytes = shard_elems * elem
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        steps = world - 1

        base = out.data_ptr()
        in_base = inp.data_ptr()
        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        add_done = self._piece_events
        copied = self._copied_events

        def piece_ptr(chunk: int, piece: int) -> int:
            return base + chunk * shard_bytes + piece * piece_bytes

        def in_piece_ptr(piece: int) -> int:
            return in_base + piece * piece_bytes

        def scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)
        # Copy local shard into output[rank] on main while hop 0 P2P runs on CE.
        kernels.dma_copy(base + rank * shard_bytes, in_base, shard_bytes)

        for k in range(steps):
            send_chunk = (rank - k) % world
            recv_chunk = (rank - k - 1) % world
            for p in range(pieces):
                send_src = in_piece_ptr(p) if k == 0 else piece_ptr(send_chunk, p)
                send_dst = scratch_piece(nxt, k, p)
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, piece_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                kernels.dma_copy(
                    piece_ptr(recv_chunk, p),
                    scratch_piece(rank, k, p),
                    piece_bytes,
                )
                add_done[p].record(main)

        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = FLAG_SLOTS - 2
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def _run_prepared_reduce_scatter(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor], state
    ) -> torch.Tensor:
        state.require_runtime(self)
        torch.cuda.set_device(self.device)
        if not self.should_reduce_scatter(inp):
            raise ValueError(
                "input does not satisfy ring reduce_scatter requirements "
                f"(shape={tuple(inp.shape)}, dtype={inp.dtype}, device={inp.device})"
            )
        expected_shape = (inp.shape[0] // self.world_size, *inp.shape[1:])
        if (
            out is None
            or tuple(out.shape) != expected_shape
            or out.dtype != inp.dtype
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"DMA reduce_scatter output must be contiguous {expected_shape} {inp.dtype} on {self.device}"
            )
        in_ptr = inp.data_ptr()
        out_ptr = out.data_ptr()
        in_bytes = inp.numel() * inp.element_size()
        out_bytes = out.numel() * out.element_size()
        if in_ptr < out_ptr + out_bytes and out_ptr < in_ptr + in_bytes:
            raise ValueError(
                "DMA reduce_scatter requires disjoint input and output storage"
            )
        if self.world_size == 16:
            return self._run_prepared_reduce_scatter_4x4(inp, out=out)
        kernels = self._kernels
        world = self.world_size
        rank = self.rank
        nxt = (rank + 1) % world
        prv = (rank - 1) % world
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // world
        shard_bytes = shard_elems * elem
        pieces = self._pick_pieces(shard_elems, shard_bytes)
        piece_elems = shard_elems // pieces
        piece_bytes = piece_elems * elem
        steps = world - 1

        in_base = in_ptr
        out_base = out_ptr
        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        add_done = self._piece_events
        copied = self._copied_events

        def in_piece_ptr(chunk: int, piece: int) -> int:
            return in_base + chunk * shard_bytes + piece * piece_bytes

        def out_piece_ptr(piece: int) -> int:
            return out_base + piece * piece_bytes

        def recv_scratch_piece(owner: int, step: int, piece: int) -> int:
            return self._scratch_ptr(owner, step) + piece * piece_bytes

        def accum_scratch_piece(owner: int, step: int, piece: int) -> int:
            # Slots [0 .. world - 2] receive peer payloads; slots
            # [world - 1 .. 2 * world - 4] hold intermediate partial sums when world > 2.
            return self._scratch_ptr(owner, (world - 1) + step) + piece * piece_bytes

        def slot(step: int, piece: int) -> int:
            return step * pieces + piece

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)

        for k in range(steps):
            # Offset starting chunk by -1 so the final hop (k = world - 2) reduces chunk rank.
            send_chunk = (rank - 1 - k) % world
            recv_chunk = (rank - 2 - k) % world
            final_step = k == steps - 1
            for p in range(pieces):
                send_src = (
                    in_piece_ptr(send_chunk, p)
                    if k == 0
                    else accum_scratch_piece(rank, k - 1, p)
                )
                send_dst = recv_scratch_piece(nxt, k, p)
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, piece_bytes)
                    copied[slot(k, p)].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied[slot(k, p)])
                    kernels.dma_set_flag(
                        self._flag_ptr(nxt, slot(k, p)),
                        self._counter_ptr(self._send_counters, slot(k, p)),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot(k, p)),
                    self._counter_ptr(self._wait_counters, slot(k, p)),
                )
                dst_ptr = (
                    out_piece_ptr(p) if final_step else accum_scratch_piece(rank, k, p)
                )
                kernels.dma_add(
                    dst_ptr,
                    in_piece_ptr(recv_chunk, p),
                    recv_scratch_piece(rank, k, p),
                    piece_elems,
                    dtype_code,
                )
                add_done[p].record(main)

        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        done = FLAG_SLOTS - 2
        kernels.dma_set_flag(
            self._flag_ptr(prv, done), self._counter_ptr(self._send_counters, done)
        )
        kernels.dma_wait_flag(
            self._flag_ptr(rank, done), self._counter_ptr(self._wait_counters, done)
        )
        return out

    def _finish_4x4_handshake(self, main: torch.cuda.Stream) -> None:
        kernels = self._kernels
        main.wait_stream(self._copy_stream)
        main.wait_stream(self._flag_stream)
        kernels.dma_set_flag(
            self._flag_ptr(self._intra_prv, _DONE_INTRA_SLOT),
            self._send_counter_ptr(_DONE_INTRA_SLOT),
        )
        kernels.dma_set_flag(
            self._flag_ptr(self._cross_prv, _DONE_CROSS_SLOT),
            self._send_counter_ptr(_DONE_CROSS_SLOT),
        )
        kernels.dma_wait_flag(
            self._flag_ptr(self.rank, _DONE_INTRA_SLOT),
            self._wait_counter_ptr(_DONE_INTRA_SLOT),
        )
        kernels.dma_wait_flag(
            self._flag_ptr(self.rank, _DONE_CROSS_SLOT),
            self._wait_counter_ptr(_DONE_CROSS_SLOT),
        )

    def _run_prepared_reduce_scatter_4x4(
        self,
        inp: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        out_ptr_override: int | None = None,
    ) -> torch.Tensor | None:
        kernels = self._kernels
        rank = self.rank
        island, lane = self._island, self._lane
        intra_nxt = self._intra_nxt
        cross_nxt = self._cross_nxt
        dtype_code = SUPPORTED_DTYPES[inp.dtype]
        elem = inp.element_size()
        shard_elems = inp.numel() // 16
        shard_bytes = shard_elems * elem
        half_bytes = 2 * shard_bytes
        in_base = inp.data_ptr()
        out_base = out.data_ptr() if out_ptr_override is None else int(out_ptr_override)
        accum_a = self._local_accum.data_ptr()
        accum_b = accum_a + 4 * self.shard_capacity

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        add_done = self._piece_events
        p2_add_done = self._p2_piece_events
        copied_intra = self._copied_intra
        copied_cross = self._copied_cross

        def in_shard_ptr(t_island: int, t_lane: int) -> int:
            return in_base + (4 * t_island + t_lane) * shard_bytes

        def p1_recv(owner: int, step: int) -> int:
            return self._scratch_base[owner] + step * 4 * self.shard_capacity

        def p2_recv(owner: int, step: int) -> int:
            return self._scratch_base[owner] + (12 + step) * self.shard_capacity

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)

        # Phase 1: Intra-4 ring RS (3 hops x 4*S packed contiguously as 2 pieces of 2*S).
        for k in range(3):
            send_lane = (lane - 1 - k) % 4
            recv_lane = (lane - 2 - k) % 4
            dst_buf = accum_b if k == 1 else accum_a
            for p in range(2):
                slot = k * 2 + p
                t0, t1 = 2 * p, 2 * p + 1
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(add_done[p])
                    if k == 0:
                        kernels.dma_copy(
                            p1_recv(intra_nxt, 0) + t0 * shard_bytes,
                            in_shard_ptr(t0, send_lane),
                            shard_bytes,
                        )
                        kernels.dma_copy(
                            p1_recv(intra_nxt, 0) + t1 * shard_bytes,
                            in_shard_ptr(t1, send_lane),
                            shard_bytes,
                        )
                    else:
                        src_buf = accum_a if k == 1 else accum_b
                        kernels.dma_copy(
                            p1_recv(intra_nxt, k) + p * half_bytes,
                            src_buf + p * half_bytes,
                            half_bytes,
                        )
                    copied_intra[slot].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied_intra[slot])
                    kernels.dma_set_flag(
                        self._flag_ptr(intra_nxt, slot),
                        self._send_counter_ptr(slot),
                    )
                if k == 2:
                    # Wait for k=1 CE read of accum_a[p] before overwriting with intra_sum.
                    main.wait_event(copied_intra[2 + p])
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._wait_counter_ptr(slot),
                )
                kernels.dma_add(
                    dst_buf + t0 * shard_bytes,
                    in_shard_ptr(t0, recv_lane),
                    p1_recv(rank, k) + t0 * shard_bytes,
                    shard_elems,
                    dtype_code,
                )
                kernels.dma_add(
                    dst_buf + t1 * shard_bytes,
                    in_shard_ptr(t1, recv_lane),
                    p1_recv(rank, k) + t1 * shard_bytes,
                    shard_elems,
                    dtype_code,
                )
                add_done[p].record(main)

        # Ensure Phase 1 k=2 CE reads of accum_b have completed before Phase 2 reuses accum_b.
        main.wait_event(copied_intra[4])
        main.wait_event(copied_intra[5])

        # Phase 2: Cross-4 ring RS (3 hops x 1*S -> direct write into out on k=2).
        p2 = 2 if (shard_elems % 16 == 0 and shard_bytes >= (512 << 10)) else 1
        if (
            1 <= self._pieces_override <= 2
            and shard_elems % (self._pieces_override * 8) == 0
        ):
            p2 = self._pieces_override
        p2_elems = shard_elems // p2
        p2_bytes = p2_elems * elem
        cross_acc0 = accum_b
        cross_acc1 = accum_b + self.shard_capacity

        for k in range(3):
            send_island = (island - 1 - k) % 4
            recv_island = (island - 2 - k) % 4
            final_step = k == 2
            for p in range(p2):
                ev_idx = k * p2 + p
                slot = _CROSS_FLAG_BASE + ev_idx
                if k == 0:
                    send_src = accum_a + send_island * shard_bytes + p * p2_bytes
                elif k == 1:
                    send_src = cross_acc0 + p * p2_bytes
                else:
                    send_src = cross_acc1 + p * p2_bytes
                send_dst = p2_recv(cross_nxt, k) + p * p2_bytes
                with torch.cuda.stream(copy_stream):
                    if k == 0:
                        if p == 0:
                            copy_stream.wait_event(add_done[send_island // 2])
                    else:
                        copy_stream.wait_event(p2_add_done[p])
                    kernels.dma_copy(send_dst, send_src, p2_bytes)
                    copied_cross[ev_idx].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied_cross[ev_idx])
                    kernels.dma_set_flag(
                        self._flag_ptr(cross_nxt, slot),
                        self._send_counter_ptr(slot),
                    )
                dst_ptr = (
                    out_base + p * p2_bytes
                    if final_step
                    else (cross_acc0 if k == 0 else cross_acc1) + p * p2_bytes
                )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._wait_counter_ptr(slot),
                )
                kernels.dma_add(
                    dst_ptr,
                    accum_a + recv_island * shard_bytes + p * p2_bytes,
                    p2_recv(rank, k) + p * p2_bytes,
                    p2_elems,
                    dtype_code,
                )
                p2_add_done[p].record(main)

        self._finish_4x4_handshake(main)
        return out

    def _run_prepared_all_gather_4x4(
        self,
        inp: torch.Tensor | None,
        *,
        out: torch.Tensor,
        in_ptr_override: int | None = None,
        shard_elems_override: int | None = None,
        elem_size_override: int | None = None,
    ) -> torch.Tensor:
        kernels = self._kernels
        rank = self.rank
        island, lane = self._island, self._lane
        intra_nxt = self._intra_nxt
        cross_nxt = self._cross_nxt
        elem = (
            inp.element_size()
            if elem_size_override is None
            else int(elem_size_override)
        )
        shard_elems = (
            inp.numel() if shard_elems_override is None else int(shard_elems_override)
        )
        shard_bytes = shard_elems * elem
        half_bytes = 2 * shard_bytes
        in_base = inp.data_ptr() if in_ptr_override is None else int(in_ptr_override)
        out_base = out.data_ptr()

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        p1_recv_done = self._p2_piece_events
        add_done = self._piece_events
        copied_cross = self._copied_cross
        copied_intra = self._copied_intra

        def out_shard_ptr(t_island: int, t_lane: int) -> int:
            return out_base + (4 * t_island + t_lane) * shard_bytes

        def p1_slot(owner: int, t_island: int) -> int:
            # Packed contiguously by shard_bytes within the first 4*shard_capacity region
            # so Phase 2 forwards [0..3] as two contiguous 2*shard_bytes pieces.
            return self._scratch_base[owner] + t_island * shard_bytes

        def p2_slot(owner: int, step: int) -> int:
            return self._scratch_base[owner] + (step + 1) * 4 * self.shard_capacity

        p1 = 2 if (shard_elems % 16 == 0 and shard_bytes >= (512 << 10)) else 1
        if (
            1 <= self._pieces_override <= 2
            and shard_elems % (self._pieces_override * 8) == 0
        ):
            p1 = self._pieces_override
        p1_elems = shard_elems // p1
        p1_bytes = p1_elems * elem

        self._input_ready.record(main)
        copy_stream.wait_event(self._input_ready)
        flag_stream.wait_event(self._input_ready)

        # Copy own shard into p1_slot[island] and out[rank] on main while Phase 1 hop 0 runs.
        kernels.dma_copy(p1_slot(rank, island), in_base, shard_bytes)
        kernels.dma_copy(out_shard_ptr(island, lane), in_base, shard_bytes)

        # Phase 1: Cross-4 ring AG (3 hops x 1*S).
        for k in range(3):
            send_island = (island - k) % 4
            recv_island = (island - k - 1) % 4
            for p in range(p1):
                ev_idx = k * p1 + p
                slot = _CROSS_FLAG_BASE + ev_idx
                send_src = (
                    in_base if k == 0 else p1_slot(rank, send_island)
                ) + p * p1_bytes
                send_dst = p1_slot(cross_nxt, send_island) + p * p1_bytes
                with torch.cuda.stream(copy_stream):
                    if k > 0:
                        copy_stream.wait_event(p1_recv_done[p])
                    kernels.dma_copy(send_dst, send_src, p1_bytes)
                    copied_cross[ev_idx].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied_cross[ev_idx])
                    kernels.dma_set_flag(
                        self._flag_ptr(cross_nxt, slot),
                        self._send_counter_ptr(slot),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._wait_counter_ptr(slot),
                )
                # Publish readiness before local copy into out so next hop forwards immediately.
                p1_recv_done[p].record(main)
                kernels.dma_copy(
                    out_shard_ptr(recv_island, lane) + p * p1_bytes,
                    p1_slot(rank, recv_island) + p * p1_bytes,
                    p1_bytes,
                )

        self._p1_done.record(main)

        # Phase 2: Intra-4 ring AG (3 hops x 4*S as 2 contiguous 2*S pieces).
        for k in range(3):
            recv_lane = (lane - k - 1) % 4
            for p in range(2):
                slot = k * 2 + p
                t0, t1 = 2 * p, 2 * p + 1
                send_src = (
                    p1_slot(rank, 0) if k == 0 else p2_slot(rank, k - 1)
                ) + p * half_bytes
                send_dst = p2_slot(intra_nxt, k) + p * half_bytes
                with torch.cuda.stream(copy_stream):
                    if k == 0:
                        if p == 0:
                            copy_stream.wait_event(self._p1_done)
                    else:
                        copy_stream.wait_event(add_done[p])
                    kernels.dma_copy(send_dst, send_src, half_bytes)
                    copied_intra[slot].record(copy_stream)
                with torch.cuda.stream(flag_stream):
                    flag_stream.wait_event(copied_intra[slot])
                    kernels.dma_set_flag(
                        self._flag_ptr(intra_nxt, slot),
                        self._send_counter_ptr(slot),
                    )
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._wait_counter_ptr(slot),
                )
                add_done[p].record(main)
                kernels.dma_copy(
                    out_shard_ptr(t0, recv_lane),
                    p2_slot(rank, k) + t0 * shard_bytes,
                    shard_bytes,
                )
                kernels.dma_copy(
                    out_shard_ptr(t1, recv_lane),
                    p2_slot(rank, k) + t1 * shard_bytes,
                    shard_bytes,
                )

        self._finish_4x4_handshake(main)
        return out

    def _run_prepared_all_reduce_4x4(
        self, inp: torch.Tensor, *, out: torch.Tensor
    ) -> torch.Tensor:
        # Slot 6 of _local_accum is unused by RS Phase 2 (which uses slots 0..5),
        # so RS writes the reduced shard there and AG gathers it into out cleanly.
        shard_ptr = self._local_accum.data_ptr() + 6 * self.shard_capacity
        self._run_prepared_reduce_scatter_4x4(inp, out_ptr_override=shard_ptr)
        self._run_prepared_all_gather_4x4(
            None,
            out=out,
            in_ptr_override=shard_ptr,
            shard_elems_override=inp.numel() // 16,
            elem_size_override=inp.element_size(),
        )
        return out

    def _prime_prepared(self, state) -> None:
        """Prime selected code using this channel's fixed IPC buffers only."""

        state.require_runtime(self)
        self._kernels.install(state.launchers)
        with torch.cuda.device(self.device):
            # Local probes must not overwrite an arriving peer flag or payload.
            # The final slot is beyond both transport layouts for supported worlds.
            slot = FLAG_SLOTS - 1
            flag = self._flag_ptr(self.rank, slot)
            self._kernels.dma_set_flag(flag, self._send_counter_ptr(slot))
            self._kernels.dma_wait_flag(flag, self._wait_counter_ptr(slot))
            scratch = flag + 16
            self._kernels.dma_copy(scratch, scratch, 16)
            for dtype_code, elems in ((0, 8), (1, 8), (2, 4)):
                self._kernels.dma_add(scratch, scratch, scratch, elems, dtype_code)
                self._kernels.dma_set_flag(flag, self._send_counter_ptr(slot))
                self._kernels.dma_add(
                    scratch,
                    scratch,
                    scratch,
                    elems,
                    dtype_code,
                    wait=(
                        flag,
                        self._wait_counter_ptr(slot),
                        self._wait_done_ptr(slot),
                    ),
                )
            if self._fp8:
                if self._fp8_stage is None or self._fp8_stage.numel() < 512:
                    raise RuntimeError(
                        "compressed DMA stage buffer is too small to prime"
                    )
                codec = (
                    "i8"
                    if self._fp8.startswith("i8")
                    else "mx"
                    if self._fp8.startswith("mx")
                    else "e4m3"
                )
                stage = self._fp8_stage.data_ptr()
                source, payload, scales = stage, stage + 256, stage + 384
                self._kernels._quant(codec, source, payload, scales, FP8_QUANT_BLOCK)
                self._kernels._dequant_store(
                    codec, source, payload, scales, FP8_QUANT_BLOCK
                )
                self._kernels._dequant_add_quant(
                    codec,
                    source,
                    source,
                    payload,
                    scales,
                    payload,
                    scales,
                    FP8_QUANT_BLOCK,
                    False,
                )
                self._kernels._dequant_add_quant(
                    codec,
                    source,
                    source,
                    payload,
                    scales,
                    payload,
                    scales,
                    FP8_QUANT_BLOCK,
                    True,
                )
                if self._fp8 == "a2a" or self._fp8.endswith("a2a"):
                    sources = [payload] * (self.world_size - 1)
                    self._kernels._dequant_accum(
                        codec,
                        source,
                        source,
                        sources,
                        [scales] * len(sources),
                        FP8_QUANT_BLOCK,
                    )
            torch.cuda.current_stream(self.device).synchronize()

    def _pick_a2a_chunks(self, shard_elems: int) -> int:
        override = self._a2a_chunks_override
        candidates = (override,) if 1 <= override <= MAX_PIECES else (4, 3, 2)
        for chunks in candidates:
            if (
                shard_elems % (chunks * FP8_QUANT_BLOCK) == 0
                and shard_elems // chunks >= 384 << 10
            ):
                return chunks
        return 1

    def _all_reduce_fp8(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        shard_elems: int,
        *,
        wire_codec: str = "e4m3",
    ) -> torch.Tensor:
        """Pipelined quantize-once compressed all-to-all.

        Slices are split into chunks; each chunk's quantize -> scatter ->
        fp32 dequant-accumulate -> quantize-once broadcast wave overlaps the
        next chunk's, with broadcast copies on their own CE stream so the
        two phases' wire time overlaps rather than queues.

        No end handshake is needed: a rank re-enters the op only after its
        stream finished, which required every peer's broadcast of every
        chunk and therefore every peer's accumulate and placement; peers'
        next-call writes are stream-ordered after that.
        """

        kernels = self._kernels
        if wire_codec == "i8":
            quantize = kernels.dma_quant_i8
            dequantize_accum = kernels.dma_dequant_accum_i8
            dequantize_store = kernels.dma_dequant_store_i8
        elif wire_codec == "mx":
            quantize = kernels.dma_quant_mx
            dequantize_accum = kernels.dma_dequant_accum_mx
            dequantize_store = kernels.dma_dequant_store_mx
        else:
            quantize = kernels.dma_quant
            dequantize_accum = kernels.dma_dequant_accum
            dequantize_store = kernels.dma_dequant_store
        world = self.world_size
        rank = self.rank
        shard_bytes = shard_elems * 2
        chunks = self._pick_a2a_chunks(shard_elems)
        chunk_elems = shard_elems // chunks
        chunk_bytes = chunk_elems * 2
        chunk_payload = chunk_elems
        chunk_slice = chunk_payload + chunk_elems // FP8_QUANT_BLOCK * 4
        in_base = inp.data_ptr()
        out_base = out.data_ptr()
        stage_base = self._fp8_stage.data_ptr()
        stride = self._fp8_stage_stride

        def stage_chunk(shard: int, c: int) -> int:
            return stage_base + shard * stride + c * chunk_slice

        def rs_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, srcpos) + c * chunk_slice

        def ag_chunk(owner: int, srcpos: int, c: int) -> int:
            return self._scratch_ptr(owner, (world - 1) + srcpos) + c * chunk_slice

        def rs_slot(srcpos: int, c: int) -> int:
            return srcpos * chunks + c

        def ag_slot(srcpos: int, c: int) -> int:
            return (world - 1) * chunks + srcpos * chunks + c

        main = torch.cuda.current_stream(self.device)
        copy_stream = self._copy_stream
        flag_stream = self._flag_stream
        ag_copy = self._ag_copy_stream
        ag_flag = self._ag_flag_stream
        copied = self._copied_events
        half = len(copied) // 2
        peers = [(rank + 1 + i) % world for i in range(world - 1)]
        pos_at = [(rank - j - 1) % world for j in peers]

        # Quantize all outgoing chunks up front (cheap kernels on main);
        # per-chunk events let the scatter start as soon as its chunk is
        # ready while later quants still run.
        for c in range(chunks):
            for j in peers:
                quantize(
                    in_base + j * shard_bytes + c * chunk_bytes,
                    stage_chunk(j, c),
                    stage_chunk(j, c) + chunk_payload,
                    chunk_elems,
                )
            self._a2a_qdone[c].record(main)

        # Scatter: reduce-scatter slices, chunk-pipelined.
        for c in range(chunks):
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(self._a2a_qdone[c])
                for i, j in enumerate(peers):
                    kernels.dma_copy(
                        rs_chunk(j, pos_at[i], c), stage_chunk(j, c), chunk_slice
                    )
                    copied[i * chunks + c].record(copy_stream)
            with torch.cuda.stream(flag_stream):
                for i, j in enumerate(peers):
                    flag_stream.wait_event(copied[i * chunks + c])
                    slot = rs_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Accumulate own shard chunk by chunk; broadcast each chunk as soon
        # as it is reduced and quantized (once).
        for c in range(chunks):
            for i in range(world - 1):
                slot = rs_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
            payloads = [rs_chunk(rank, i, c) for i in range(world - 1)]
            scales = [ptr + chunk_payload for ptr in payloads]
            own = rank * shard_bytes + c * chunk_bytes
            dequantize_accum(
                out_base + own, in_base + own, payloads, scales, chunk_elems
            )
            quantize(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            # Publish the payload first so its CE broadcast can overlap the
            # local read-only materialization below.
            self._a2a_ownq[c].record(main)
            # Peers materialize this reduced shard from the broadcast FP8
            # payload.  The owner must do the same or every rank enters the
            # next TP layer with a different replicated activation.
            dequantize_store(
                out_base + own,
                stage_chunk(rank, c),
                stage_chunk(rank, c) + chunk_payload,
                chunk_elems,
            )
            with torch.cuda.stream(ag_copy):
                ag_copy.wait_event(self._a2a_ownq[c])
                for i, j in enumerate(peers):
                    kernels.dma_copy(
                        ag_chunk(j, pos_at[i], c), stage_chunk(rank, c), chunk_slice
                    )
                    copied[half + i * chunks + c].record(ag_copy)
            with torch.cuda.stream(ag_flag):
                for i, j in enumerate(peers):
                    ag_flag.wait_event(copied[half + i * chunks + c])
                    slot = ag_slot(pos_at[i], c)
                    kernels.dma_set_flag(
                        self._flag_ptr(j, slot),
                        self._counter_ptr(self._send_counters, slot),
                    )

        # Place incoming reduced shards.
        for c in range(chunks):
            for i in range(world - 1):
                src = peers[i]
                slot = ag_slot(i, c)
                kernels.dma_wait_flag(
                    self._flag_ptr(rank, slot),
                    self._counter_ptr(self._wait_counters, slot),
                )
                payload = ag_chunk(rank, i, c)
                dequantize_store(
                    out_base + src * shard_bytes + c * chunk_bytes,
                    payload,
                    payload + chunk_payload,
                    chunk_elems,
                )
        main.wait_stream(copy_stream)
        main.wait_stream(flag_stream)
        main.wait_stream(ag_copy)
        main.wait_stream(ag_flag)
        return out

    def close(self) -> None:
        if self._closed:
            return
        if self._capture_context_depth:
            raise RuntimeError("cannot close PCIe DMA channel during capture")
        self._closed = True
        # Drain the main and four auxiliary streams before any rank unmaps a
        # peer allocation.  Every importer must unmap before its owner frees
        # the exported slab.
        torch.cuda.synchronize(self.device)
        for stream in self._owned_streams:
            with suppress(Exception):
                stream._b12x_release()
        self._owned_streams.clear()
        self._eager_replays.clear()
        dist.barrier(group=self.group)
        for ptr in self._slab.remote_ptrs:
            with suppress(Exception):
                self._ipc.cudaIpcCloseMemHandle(ptr)
        dist.barrier(group=self.group)
        with suppress(Exception):
            self._ipc.cudaFree(self._slab.local_ptr)
        dist.barrier(group=self.group)

    def __enter__(self) -> "PCIeDmaAllReduce":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __del__(self) -> None:
        # Distributed barriers are unsafe during asymmetric interpreter
        # teardown. Explicit/context-manager close owns coordinated release.
        return None


__all__ = ["PCIeDmaAllReduce"]
