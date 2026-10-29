"""Prepare captured selection among supplied or measured joint collective profiles."""

from dataclasses import dataclass, field

import torch

from b12x.preparation import FinalistPolicy
from ._joint_profile import (
    JOINT_FAMILIES,
    JOINT_MODES,
    JOINT_STARTUP_SAMPLES as SAMPLES_PER_MODE,
    joint_modes,
    selected_profile,
)


def joint_finalist_policy():
    """Retain the best measured CTA count for each mode and compiled geometry."""
    return FinalistPolicy(
        family_fields=("implementation", "threads", "blocks", "submode"),
        per_family=1,
        max_total=5 * sum(len(modes) for modes in JOINT_FAMILIES.values()),
    )


@dataclass(frozen=True)
class JointStartupSpec:
    key: tuple
    profiles: tuple[tuple[int, int], ...]
    selector: torch.Tensor = field(compare=False, repr=False)
    state: object = field(compare=False, repr=False)
    inplace: bool
    rank: int


def joint_startup_spec(plan, *, profiles=None):
    """Describe supplied runtime profiles or measured finalists of a prepared launcher."""
    from ._joint_preparation import _JointExecutionState

    prepared = plan.prepared
    if prepared is None or not isinstance(prepared.state, _JointExecutionState):
        raise ValueError("startup tuning requires a prepared joint collective")
    state = prepared.state
    call = state.query.call
    family = call["transport"]
    geometry = (family, call["threads"], call["blocks"])
    if profiles is not None:
        profiles = tuple(profiles)
        if not profiles or any(
            not isinstance(profile, (tuple, list)) or len(profile) != 2
            for profile in profiles
        ):
            raise ValueError(
                "runtime profiles must be nonempty (mode, active CTA) pairs"
            )
        encoded = []
        for mode, active in profiles:
            selected_profile(
                family, call["blocks"], mode, active, world_size=state.query.world_size
            )
            encoded.append((JOINT_MODES[mode], active))
        if len(set(encoded)) != len(encoded):
            raise ValueError("runtime profiles must be distinct")
        profiles = tuple(encoded)
    else:
        measured = {}
        for record in prepared.finalists:
            assignment = record.assignment
            if (
                tuple(
                    assignment[name] for name in ("implementation", "threads", "blocks")
                )
                == geometry
            ):
                mode = assignment["submode"]
                active = assignment["active_blocks"]
                selected_profile(
                    family,
                    call["blocks"],
                    mode,
                    active,
                    world_size=state.query.world_size,
                )
                measured[mode] = (JOINT_MODES[mode], active)
        modes = joint_modes(family, state.query.world_size)
        if set(measured) != set(modes):
            raise ValueError(
                "startup tuning requires measured finalists for every mode of the selected geometry"
            )
        profiles = tuple(measured[mode] for mode in modes)
    inplace = bool(plan.query.inplace)
    import torch.distributed as dist

    members = tuple(dist.get_process_group_ranks(state.runtime.exchange_group))
    key = (
        "joint_startup",
        1,
        state.query.world_size,
        members,
        *geometry,
        tuple(state.query.setup["shape"]),
        call["dtype"],
        call["size_packs"],
        state.workspace.capacity_packs,
        inplace,
        profiles,
    )
    return JointStartupSpec(
        key, profiles, state.selector, state, inplace, state.query.rank
    )


class JointStartupTuner:
    """Captureable bounded race; retained owners must outlive its CUDA graph."""

    def __init__(self, specs, cache_policy):
        from ._benchmark import _ReductionInputs
        from ._joint_startup_cute import SCRATCH_BYTES, SIGNAL_BYTES, timing_launcher
        from .pcie_oneshot import (
            PCIeOneshotAllReduce,
            _require_collective_contract,
            _run_collective_preallocation_setup,
            _finish_collective_runtime_setup,
        )

        self.specs = tuple(specs)
        if not self.specs:
            raise ValueError("startup tuning requires at least one consumer")
        first = self.specs[0]
        state, runtime = first.state, first.state.runtime
        if any(
            spec.key != first.key
            or spec.rank != first.rank
            or spec.selector.device != first.selector.device
            or spec.state.runtime is not runtime
            for spec in self.specs
        ):
            raise ValueError(
                "startup consumers must share geometry, profiles, rank, and device"
            )
        if cache_policy not in ("cold", "warm"):
            raise ValueError("cache_policy must be cold or warm")
        self.key = first.key
        self.state, self.runtime = state, runtime
        self.cache_policy = cache_policy
        _require_collective_contract(
            owner="joint startup tuner",
            exchange_group=runtime.exchange_group,
            contract=(self.key, cache_policy),
        )

        def prepare_local():
            with torch.cuda.device(runtime.device):
                self.peers = torch.empty(
                    runtime.world_size, dtype=torch.uint64, device=runtime.device
                )
                self.status = torch.full(
                    (1,), -1, dtype=torch.int32, device=runtime.device
                )
                self.samples = torch.empty(
                    (len(first.profiles), SAMPLES_PER_MODE),
                    dtype=torch.int64,
                    device=runtime.device,
                )
                self.profiles = torch.tensor(
                    [(mode << 16) | active for mode, active in first.profiles],
                    dtype=torch.uint32,
                    device=runtime.device,
                )
                self.inputs = _ReductionInputs(state, inplace=first.inplace)
                self.mismatch = torch.empty_like(self.inputs.out, dtype=torch.bool)
                self.invalid = torch.zeros((), dtype=torch.int32, device=runtime.device)
                self.eviction = self.eviction_sum = None
                if cache_policy == "cold":
                    eviction = getattr(runtime, "_joint_startup_eviction", None)
                    if eviction is None:
                        l2 = torch.cuda.get_device_properties(
                            runtime.device
                        ).L2_cache_size
                        eviction = (
                            torch.ones(
                                max(2 * l2, 64 * 1024 * 1024) // 4,
                                dtype=torch.float32,
                                device=runtime.device,
                            ),
                            torch.empty((), dtype=torch.float32, device=runtime.device),
                        )
                        runtime._joint_startup_eviction = eviction
                    self.eviction, self.eviction_sum = eviction
                self.launchers = {
                    action: timing_launcher(
                        runtime.world_size,
                        runtime.rank,
                        action,
                        len(first.profiles),
                        runtime.device.index,
                    )
                    for action in ("reset", "validate", "begin", "end", "publish")
                }
            return True

        _run_collective_preallocation_setup(
            owner="joint startup tuner",
            exchange_group=runtime.exchange_group,
            setup=prepare_local,
        )
        with torch.cuda.device(runtime.device):
            shared = PCIeOneshotAllReduce._allocate_shared_buffer(
                runtime.exchange_group, SCRATCH_BYTES, zero_fill=True, ipc=runtime._ipc
            )
            local_error = None
            try:
                self.peers.copy_(
                    torch.tensor(
                        shared.peer_ptrs, dtype=torch.uint64, device=runtime.device
                    )
                )
            except Exception as error:
                local_error = error
            _finish_collective_runtime_setup(
                owner="joint startup tuner",
                exchange_group=runtime.exchange_group,
                ipc=runtime._ipc,
                shared=shared,
                local_error=local_error,
                detach_shared_ownership=lambda: None,
            )
            runtime._owned_buffers.append(shared)
            self.shared = shared
        self.local_address = shared.local_ptr + SIGNAL_BYTES
        self.owners = (
            self.specs,
            runtime,
            shared,
            self.inputs,
            self.peers,
            self.status,
            self.samples,
            self.profiles,
            self.mismatch,
            self.invalid,
            self.eviction,
            self.eviction_sum,
            self.launchers,
        )

    def memory_requirements(self):
        """Resident startup storage, with shared fixtures and eviction deduplicated."""
        from b12x.preparation import MemoryRequirements, PersistentMemory

        tensors = (
            self.inputs.source,
            self.inputs.out,
            *self.inputs.reference.seeds,
            *self.inputs.reference.expected,
            self.peers,
            self.status,
            self.samples,
            self.profiles,
            self.mismatch,
            self.invalid,
            self.eviction,
            self.eviction_sum,
        )
        entries = {}
        for tensor in tensors:
            if tensor is not None:
                storage = tensor.untyped_storage()
                key = (
                    "pcie-joint-startup-tensor",
                    str(tensor.device),
                    storage.data_ptr(),
                )
                entries[key] = PersistentMemory(key, storage.nbytes(), storage.nbytes())
        key = ("pcie-joint-startup-ipc", self.shared.local_ptr)
        entries[key] = PersistentMemory(key, self.shared.nbytes, self.shared.nbytes)
        return MemoryRequirements(persistent=tuple(entries.values()))

    def _timing(self, action, sample, selector):
        self.launchers[action](
            self.peers.data_ptr(),
            self.local_address,
            self.samples.data_ptr(),
            self.invalid.data_ptr(),
            self.profiles.data_ptr(),
            selector.data_ptr(),
            self.status.data_ptr(),
            sample,
        )

    def _collective(self):
        work = self.state.workspace
        self.state.launcher(
            work.peer_table.data_ptr(),
            work.signal_table.data_ptr(),
            work.shared.local_ptr,
            self.state.selector.data_ptr(),
            self.inputs.source.data_ptr(),
            self.inputs.out.data_ptr(),
            self.state.query.call["size_packs"],
            work.capacity_packs,
        )

    def _reset_validate(self):
        self._timing("reset", 0, self.state.selector)
        for mode, active in self.specs[0].profiles:
            self.state.selector.fill_((mode << 16) | active)
            for producer in range(3):
                self.inputs.produce(producer)
                self._collective()
                torch.ne(
                    self.inputs.out,
                    self.inputs.reference.expected[producer],
                    out=self.mismatch,
                )
                torch.sum(
                    self.mismatch,
                    dim=tuple(range(self.mismatch.ndim)),
                    dtype=torch.int32,
                    out=self.invalid,
                )
                self._timing("validate", 0, self.state.selector)

    def _sample(self, sample):
        if type(sample) is not int or not 0 <= sample < SAMPLES_PER_MODE:
            raise ValueError("startup sample index is outside the prepared schedule")
        profiles = self.specs[0].profiles
        for offset in range(len(profiles)):
            mode_index = (offset + sample // 3) % len(profiles)
            mode, active = profiles[mode_index]
            self.state.selector.fill_((mode << 16) | active)
            if self.eviction is not None:
                torch.sum(self.eviction, dim=0, out=self.eviction_sum)
            self.inputs.produce(sample % 3)
            position = mode_index * SAMPLES_PER_MODE + sample
            self._timing("begin", position, self.state.selector)
            for _ in range(self.inputs.repetitions):
                self._collective()
            self._timing("end", position, self.state.selector)

    def _publish(self):
        for spec in self.specs:
            self._timing("publish", 0, spec.selector)

    def run(self):
        """Enqueue one group's race without allocation or host selection."""
        run_joint_startup_tuners((self,))


def run_joint_startup_tuners(producers):
    """Interleave fixed sample passes across prepared groups on one rank.

    Each profile keeps 21 complete-chain samples and seven uses of each exact
    fixture. Other groups separate a group's samples in time; a single group
    has no such separation. Publication minimizes the sum of maximum-rank
    chain durations, with prepared profile order breaking exact ties.
    """
    import torch.distributed as dist

    producers = tuple(producers)
    if not producers or len({id(producer) for producer in producers}) != len(producers):
        raise ValueError("startup groups must be nonempty and distinct")
    first = producers[0]
    members = tuple(dist.get_process_group_ranks(first.runtime.exchange_group))
    for producer in producers:
        if (
            producer.runtime.device != first.runtime.device
            or producer.specs[0].rank != first.specs[0].rank
            or tuple(dist.get_process_group_ranks(producer.runtime.exchange_group))
            != members
        ):
            raise ValueError(
                "startup groups must share device and ordered rank membership"
            )
    with torch.cuda.device(first.runtime.device):
        for producer in producers:
            producer._reset_validate()
        for sample in range(SAMPLES_PER_MODE):
            for producer in producers:
                producer._sample(sample)
        for producer in producers:
            producer._publish()


def prepare_joint_startup_tuner(specs, *, cache_policy="cold"):
    """Collectively allocate and compile a startup producer before capture freeze."""
    producer = JointStartupTuner(specs, cache_policy)
    producer.run()
    torch.cuda.synchronize(producer.runtime.device)
    if producer.status.item() != 0:
        raise RuntimeError("joint startup fixture validation failed during preparation")
    for spec in producer.specs:
        call = spec.state.query.call
        spec.selector.fill_(
            selected_profile(
                call["transport"],
                call["blocks"],
                call["submode"],
                call["active_blocks"],
                world_size=spec.state.query.world_size,
            )
        )
    producer.status.fill_(-1)
    return producer
