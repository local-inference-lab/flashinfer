"""Host simulation of the TP4 pull all-reduce's block-pair ordering.

The pull all-reduce synchronizes through a block-pair barrier: block ``b`` of a
rank waits only for block ``b`` of every peer. A pack that a block reads from
a peer's slab must therefore be written earlier in the same launch by that
peer's block with the same index.

These tests execute the kernel body on the host for all four ranks, with the
CuTe intrinsics redirected to host tensors, using the launch arguments that a
prepared pull declaration produces. Blocks run one at a time, in ascending or
descending order: adversarial schedules that the block-pair protocol permits,
not an exhaustive search of interleavings. Every rank's output must equal FP32
accumulation in the documented rank order (owner first, then
``(owner + i) % 4``) rounded once to BF16, bit for bit. Eager launches
alternate host slots; captured launches replay with device-selected slots and
an advancing graph epoch.
"""

from __future__ import annotations

import contextlib
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import b12x.comm.pcie._twoshot_bf16_cute as kernels
import b12x.comm.pcie.pcie_twoshot_bf16 as twoshot_bf16
from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie import _twoshot_preparation as preparation
from b12x.comm.pcie._twoshot_cute import _GRAPH_EPOCH_OFFSET
from b12x.comm.pcie.pcie_twoshot import _pad_scalar_peer_ptrs
from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout

WORLD = 4
ROW_ELEMS = 8  # PCIeAllReduce transport rows hold one 16-byte pack.
MAX_ROWS = 131072  # The dispatcher's two-shot capacity, 2 MiB.
POISON = 0x7FC07FC0  # Two BF16 NaNs, so a pack read before staging is visible.
LAUNCHES_PER_MODE = 3


class _HostMemory:
    """Pack-granular loads and stores on host tensors that stand in for GPUs."""

    def __init__(self) -> None:
        self._regions: list[tuple[int, int, torch.Tensor, str]] = []

    def map(self, tensor: torch.Tensor, name: str) -> None:
        words = tensor.view(-1).view(torch.int32)
        start = tensor.data_ptr()
        self._regions.append((start, start + 4 * words.numel(), words, name))

    def _find(self, address: int) -> tuple[int, torch.Tensor, str]:
        for start, end, words, name in self._regions:
            if start <= address and address + 16 <= end:
                return start, words, name
        raise AssertionError(f"access outside every simulated buffer: {address:#x}")

    def describe(self, address: int) -> str:
        start, _, name = self._find(address)
        return f"{name}+{address - start:#x}"

    def is_slab(self, address: int) -> bool:
        return self._find(address)[2].startswith("slab")

    def load(self, address: int) -> tuple[int, ...]:
        assert address % 16 == 0, f"misaligned pack load at {address:#x}"
        start, words, _ = self._find(address)
        index = (address - start) // 4
        return tuple(word & 0xFFFFFFFF for word in words[index : index + 4].tolist())

    def store(self, address: int, values) -> None:
        assert address % 16 == 0, f"misaligned pack store at {address:#x}"
        start, words, _ = self._find(address)
        index = (address - start) // 4
        words[index : index + 4] = torch.tensor(
            [value - (1 << 32) if value >= 1 << 31 else value for value in values],
            dtype=torch.int32,
        )


def _unpack_bf16x2(word: int):
    halves = np.array([word & 0xFFFF, word >> 16], dtype=np.uint32) << np.uint32(16)
    low, high = halves.view(np.float32)
    return low, high


def _bf16_bits(value) -> int:
    """``cvt.rn.bf16.f32``: round to nearest even."""
    if np.isnan(value):
        return 0x7FC0
    bits = int(np.float32(value).view(np.uint32))
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def _pack_f32x2_to_bf16x2(low, high) -> int:
    return _bf16_bits(low) | (_bf16_bits(high) << 16)


class _Simulation:
    """Executes the kernel body for one thread of one rank at a time.

    Control flow and addresses do not depend on loaded data, so each thread
    runs once per barrier phase and only the operations of that phase take
    effect. That lets one block finish a phase on every rank while other
    blocks have not started.
    """

    def __init__(self) -> None:
        self.memory = _HostMemory()
        self.rank = self.block = self.thread = self.grid = 0
        self.phase = 0  # pair barriers passed by the running thread
        self.active = 0  # phase whose memory operations take effect
        self.capturing = False
        self.epochs: dict[int, int] = {}  # launch-start graph epoch by address
        self.accesses: list[tuple[int, int, int, str, int]] = []
        self.arrivals: list[tuple[int, int, int]] = []

    def load(self, address):
        if self.phase != self.active:
            return (0, 0, 0, 0)
        self.accesses.append((self.rank, self.block, self.phase, "ld", int(address)))
        return self.memory.load(int(address))

    def store(self, address, *values) -> None:
        if self.phase == self.active:
            self.accesses.append(
                (self.rank, self.block, self.phase, "st", int(address))
            )
            self.memory.store(int(address), values)

    def load_epoch(self, address) -> int:
        # Every CTA reads the epoch before the last CTA advances it.
        return self.epochs[int(address)]

    def arrive(self, epoch_address, arrived_address, blocks) -> None:
        if self.phase == self.active:
            self.arrivals.append((self.rank, self.block, int(blocks)))

    def pair_barrier(self, signals, rank) -> None:
        self.phase += 1


@pytest.fixture
def simulation(monkeypatch) -> _Simulation:
    sim = _Simulation()
    arch = SimpleNamespace(
        thread_idx=lambda: (sim.thread, 0, 0),
        block_idx=lambda: (sim.block, 0, 0),
        grid_dim=lambda: (sim.grid, 1, 1),
        barrier=lambda: None,
    )
    for name, value in {
        "cute": SimpleNamespace(
            arch=arch, make_rmem_tensor=lambda shape, dtype: [None] * shape[0]
        ),
        "cutlass": SimpleNamespace(
            const_expr=lambda value: value,
            range_constexpr=range,
            Float32=np.float32,
        ),
        "Int32": int,
        "Int64": int,
        "Uint32": int,
        "Float32": np.float32,
        "ld_global_nc_v4_u32": sim.load,
        "ld_global_v4_u32": sim.load,
        "st_global_v4_u32": sim.store,
        "unpack_bf16x2": _unpack_bf16x2,
        "pack_f32x2_to_bf16x2": _pack_f32x2_to_bf16x2,
        "ld_relaxed_gpu_u32": sim.load_epoch,
        "graph_epoch_arrive_serialized": sim.arrive,
    }.items():
        monkeypatch.setattr(kernels, name, value)
    monkeypatch.setattr(
        twoshot_bf16, "_is_current_stream_capturing", lambda _: sim.capturing
    )
    return sim


def _kernel_instance(sim, rank, device_slot_selection, slot_bias, threads):
    launch = kernels._TwoShotPullAllReduceLaunch(
        WORLD, rank, device_slot_selection, slot_bias, threads, ROW_ELEMS
    )
    cls = type(launch)
    for name in ("_select_address", "_accumulate_words", "_store_pack"):
        setattr(launch, name, types.MethodType(getattr(cls, name).__wrapped__, launch))
    launch._barrier = sim.pair_barrier
    return launch


def _pointer(address: int):
    return SimpleNamespace(toint=lambda: address)


def _kernel_launch(sim, threads, device_slot, bias, call):
    """Map a prepared launcher call onto the kernel's arguments."""
    payload, staging, signals, output, rank, reduced, slot_bytes, rows, grid = call
    args = (
        _pointer(payload),
        *(_pointer(address) for address in staging),
        *(_pointer(address) for address in signals),
        _pointer(output),
        rank,
        reduced,
        slot_bytes,
        rows,
    )
    return _kernel_instance(sim, rank, device_slot, bias, threads), args, grid


def _simulate(sim, launches, *, descending: bool) -> None:
    """Run one launch on every rank, one block index at a time."""
    (grid,) = {grid for _, _, grid in launches}
    kernel = kernels._TwoShotPullAllReduceLaunch.kernel.__wrapped__
    sim.accesses.clear()
    sim.arrivals.clear()
    for block in range(grid - 1, -1, -1) if descending else range(grid):
        for phase in range(3):
            for rank, (launch, args, _) in enumerate(launches):
                for thread in range(launch._threads):
                    sim.rank, sim.block, sim.thread = rank, block, thread
                    sim.grid, sim.phase, sim.active = grid, 0, phase
                    kernel(launch, *args)
                    assert sim.phase == 2, "every thread passes both pair barriers"


def _assert_block_pair_ordering(sim) -> None:
    """Every slab read has one producer: the same block index, a phase earlier."""
    producers = {}
    for rank, block, phase, kind, address in sim.accesses:
        if kind == "st":
            assert address not in producers, (
                f"{sim.memory.describe(address)} is written twice in one launch"
            )
            producers[address] = (rank, block, phase)
    unordered = []
    for rank, block, phase, kind, address in sim.accesses:
        if kind != "ld" or not sim.memory.is_slab(address):
            continue
        producer = producers.get(address)
        if producer is None or producer[1] != block or producer[2] >= phase:
            source = (
                "no block of this launch writes it"
                if producer is None
                else "written by rank {} block {} in phase {}".format(*producer)
            )
            unordered.append(
                f"rank {rank} block {block} phase {phase} reads "
                f"{sim.memory.describe(address)}, {source}"
            )
    assert not unordered, (
        f"{len(unordered)} slab reads are not ordered by the block-pair barrier; "
        + "; ".join(unordered[:3])
    )


def _fill(payloads, step: int) -> None:
    """Ordinary values plus cancelling pairs of large ones on half the elements.

    An FP32 sum of four ordinary BF16 values is almost always exact, which
    would hide the accumulation order. A large addend absorbs smaller ones
    until its partner cancels it, so the order decides which survive.
    """
    generator = torch.Generator().manual_seed(7919 * step + 13)
    shape = (WORLD, *payloads[0].shape)
    values = torch.randn(shape, generator=generator)
    values *= torch.exp2(torch.randint(-8, 9, shape, generator=generator).float())
    large = torch.exp2(torch.randint(24, 33, shape[1:], generator=generator).float())
    first = torch.randint(0, WORLD, shape[1:], generator=generator)
    second = (first + torch.randint(1, WORLD, shape[1:], generator=generator)) % WORLD
    paired = torch.rand(shape[1:], generator=generator) < 0.5
    ranks = torch.arange(WORLD).view(WORLD, *([1] * (len(shape) - 1)))
    values = torch.where(paired & (ranks == first), large, values)
    values = torch.where(paired & (ranks == second), -large, values)
    for payload, value in zip(payloads, values):
        payload.copy_(value)


def _reduce(payloads, order, *, round_each_add: bool = False):
    rows = payloads[0].shape[0]
    shard = rows // WORLD
    result = torch.empty_like(payloads[0])
    for owner in range(WORLD):
        owned = slice(owner * shard, (owner + 1) * shard)
        accumulator = torch.zeros((shard, ROW_ELEMS), dtype=torch.float32)
        for source in order(owner):
            accumulator += payloads[source][owned].float()
            if round_each_add:
                accumulator = accumulator.to(torch.bfloat16).float()
        result[owned] = accumulator.to(torch.bfloat16)
    return result


def _owner_cyclic(owner):
    return [(owner + offset) % WORLD for offset in range(WORLD)]


def _expected(payloads):
    """FP32 accumulation, owner first, then ``(owner + i) % 4``; one rounding."""
    return _reduce(payloads, _owner_cyclic)


def _assert_payload_discriminates(payloads, expected) -> None:
    """Rank-ascending order or per-add rounding must change some element."""
    assert not torch.equal(_reduce(payloads, lambda owner: range(WORLD)), expected)
    assert not torch.equal(
        _reduce(payloads, _owner_cyclic, round_each_add=True), expected
    )


# (message bytes, threads, block limit, packs per rank shard, grid)
CASES = (
    pytest.param(65_536, 512, 64, 1024, 2, id="64KiB-two-full-blocks"),
    pytest.param(65_600, 512, 64, 1025, 3, id="65600B-one-thread-last-block"),
    pytest.param(73_728, 512, 64, 1152, 3, id="72KiB-partial-last-block"),
    pytest.param(98_304, 512, 64, 1536, 3, id="96KiB-full-blocks"),
    pytest.param(65_600, 64, 4, 1025, 4, id="65600B-grid-stride-tail"),
)


@pytest.mark.parametrize(
    ("nbytes", "threads", "block_limit", "shard_packs", "grid"), CASES
)
def test_pull_all_reduce_orders_every_remote_read(
    simulation, nbytes, threads, block_limit, shard_packs, grid
) -> None:
    sim = simulation
    layout = _make_layout(MAX_ROWS, ROW_ELEMS, WORLD)
    rows = nbytes // (2 * ROW_ELEMS)
    slabs, payloads, outputs = [], [], []
    for rank in range(WORLD):
        slab = torch.full((layout.slab_bytes // 4,), POISON, dtype=torch.int32)
        slab[: layout.signal_bytes // 4] = 0
        payload = torch.empty((rows, ROW_ELEMS), dtype=torch.bfloat16)
        output = torch.empty_like(payload)
        sim.memory.map(slab, f"slab{rank}")
        sim.memory.map(payload, f"payload{rank}")
        sim.memory.map(output, f"output{rank}")
        slabs.append(slab)
        payloads.append(payload)
        outputs.append(output)
    bases = [slab.data_ptr() for slab in slabs]
    runtimes = [
        PCIeTwoShotBF16._from_prepared_factory(
            rank=rank,
            world_size=WORLD,
            device=torch.device("cpu"),
            signal_ptrs=_pad_scalar_peer_ptrs(bases, rank=rank, world_size=WORLD),
            staging_ptrs=tuple(
                _pad_scalar_peer_ptrs(
                    [
                        base + layout.signal_bytes + slot * layout.slot_bytes
                        for base in bases
                    ],
                    rank=rank,
                    world_size=WORLD,
                )
                for slot in (0, 1)
            ),
            owned_buffers=(),
            ipc=None,
            exchange_group=None,
            max_rows=MAX_ROWS,
            row_elems=ROW_ELEMS,
            pack_stride=layout.pack_stride,
            reduced_offset=layout.reduced_offset,
            slot_bytes=layout.slot_bytes,
        )
        for rank in range(WORLD)
    ]
    # The dispatcher declares two-shot candidates only for inputs it accepts.
    # 512 threads with block limit 64 is the twoshot_pull candidate's base
    # configuration; the tuner's block limit 32 gives the same grids here.
    assert all(runtime.accepts(payload) for runtime, payload in zip(runtimes, payloads))
    queries = [
        preparation.query_from_runtime(
            runtime,
            surface="PCIeTwoShotBF16.all_reduce",
            call={
                "inp": payload,
                "out": output,
                "transport": "pull",
                "threads": threads,
                "block_limit": block_limit,
            },
        )
        for runtime, payload, output in zip(runtimes, payloads, outputs)
    ]

    calls = []

    def launcher(device_slot, bias, operation):
        assert operation == "all_reduce"
        return lambda *call: calls.append((device_slot, bias, call))

    def prepare_launches():
        calls.clear()
        for runtime, query, payload, output in zip(
            runtimes, queries, payloads, outputs
        ):
            runtime._launch_prepared(
                payload,
                None,
                output,
                state=SimpleNamespace(query=query, launcher=launcher),
                threads=threads,
                block_limit=block_limit,
            )
        assert len(calls) == WORLD
        # Launch geometry: packs per rank shard (one pack per row) and grid.
        assert {call[-1][-2:] for call in calls} == {(shard_packs, grid)}
        return [_kernel_launch(sim, threads, *call) for call in calls]

    step = 0

    def run(launches, *, slot):
        nonlocal step
        _fill(payloads, step)
        expected = _expected(payloads)
        if step == 0:
            _assert_payload_discriminates(payloads, expected)
        for output in outputs:
            output.fill_(float("nan"))
        _simulate(sim, launches, descending=bool(step % 2))
        _assert_block_pair_ordering(sim)
        for rank, output in enumerate(outputs):
            assert torch.equal(output, expected), (rank, step)
        # Each rank stages into, and publishes from, only the selected slot.
        for rank, block, phase, kind, address in sim.accesses:
            if kind == "st" and sim.memory.is_slab(address):
                offset = address - bases[rank] - layout.signal_bytes
                assert offset // layout.slot_bytes == slot, (rank, step, slot)
        step += 1

    for launch_index in range(LAUNCHES_PER_MODE):
        run(prepare_launches(), slot=launch_index % 2)

    # A captured launch selects its slot from the device epoch: every CTA
    # arrives once per launch and the last one advances the epoch.
    sim.capturing = True
    with contextlib.ExitStack() as stack:
        for runtime in runtimes:
            stack.enter_context(runtime.capture())
        launches = prepare_launches()
    sim.capturing = False
    bias = LAUNCHES_PER_MODE & 1
    assert all(runtime._device_slot_bias == bias for runtime in runtimes)
    epochs = [base + _GRAPH_EPOCH_OFFSET for base in bases]
    sim.epochs = dict.fromkeys(epochs, 0)
    for replay in range(LAUNCHES_PER_MODE):
        run(launches, slot=(replay + bias) % 2)
        for rank, address in enumerate(epochs):
            arrived = sorted(block for r, block, _ in sim.arrivals if r == rank)
            assert arrived == list(range(grid))
            assert all(blocks == grid for r, _, blocks in sim.arrivals if r == rank)
            sim.epochs[address] += 1
