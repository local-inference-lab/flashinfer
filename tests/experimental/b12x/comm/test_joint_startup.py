"""Prepared mode retention uses the established finalist cache contract."""

from b12x.comm.pcie import (
    joint_finalist_policy,
    joint_startup_spec,
    run_joint_startup_tuners,
)
from contextlib import nullcontext
import os
from types import SimpleNamespace
import pytest


def test_joint_finalist_policy_preserves_mode_and_compiled_geometry():
    policy = joint_finalist_policy()
    assert policy.family_fields == ("implementation", "threads", "blocks", "submode")
    assert policy.per_family == 1
    assert policy.max_total == 55


def test_unprepared_joint_startup_rejected():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="prepared joint"):
        joint_startup_spec(SimpleNamespace(prepared=None))


@pytest.fixture
def prepared_joint(monkeypatch):
    from types import SimpleNamespace

    import torch
    from b12x.comm.pcie._joint_preparation import _JointExecutionState

    state = _JointExecutionState.__new__(_JointExecutionState)
    state.query = SimpleNamespace(
        call=dict(
            transport="joint_bandwidth",
            threads=512,
            blocks=64,
            dtype="bfloat16",
            size_packs=65536,
        ),
        setup=dict(shape=(512, 1024)),
        world_size=4,
        rank=0,
    )
    state.workspace = SimpleNamespace(capacity_packs=65536)
    state.selector = torch.zeros(1, dtype=torch.uint32)
    state.runtime = SimpleNamespace(exchange_group=object())
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: list(range(state.query.world_size)),
    )
    return SimpleNamespace(
        prepared=SimpleNamespace(state=state, finalists=()),
        query=SimpleNamespace(inplace=False),
    )


def test_explicit_runtime_profiles_need_no_measured_finalists(prepared_joint):
    spec = joint_startup_spec(
        prepared_joint,
        profiles=(("bulk_push", 32), ("bulk_push", 64), ("bulk_pull", 8)),
    )
    assert spec.profiles == ((2, 32), (2, 64), (3, 8))
    assert spec.selector is prepared_joint.prepared.state.selector
    assert spec.state is prepared_joint.prepared.state
    other = joint_startup_spec(prepared_joint, profiles=(("bulk_push", 32),))
    assert spec.key != other.key
    with pytest.raises(ValueError, match="measured finalists"):
        joint_startup_spec(prepared_joint)


@pytest.mark.parametrize(
    "profiles",
    [
        (),
        (("bulk_push", 0),),
        (("bulk_push", 65),),
        (("bulk_push", True),),
        (("full", 1),),
        (("bulk_pull", 8), ("bulk_pull", 8)),
        ("bulk_pull",),
        (("bulk_pull",),),
    ],
)
def test_invalid_explicit_runtime_profiles(prepared_joint, profiles):
    with pytest.raises(ValueError):
        joint_startup_spec(prepared_joint, profiles=profiles)


def test_explicit_write_profiles_follow_rank_and_active_cta_admission(prepared_joint):
    spec = joint_startup_spec(
        prepared_joint, profiles=(("serial_push", 1), ("queued_push", 64))
    )
    assert spec.profiles == ((5, 1), (6, 64))
    with pytest.raises(ValueError, match="one active CTA"):
        joint_startup_spec(prepared_joint, profiles=(("serial_push", 2),))
    prepared_joint.prepared.state.query.world_size = 2
    for mode in ("serial_push", "queued_push"):
        with pytest.raises(ValueError, match="not admitted"):
            joint_startup_spec(prepared_joint, profiles=((mode, 1),))


def test_measured_profile_completeness_uses_admitted_modes(prepared_joint):
    prepared_joint.prepared.state.query.world_size = 2
    prepared_joint.prepared.finalists = tuple(
        SimpleNamespace(
            assignment=dict(
                implementation="joint_bandwidth",
                threads=512,
                blocks=64,
                submode=mode,
                active_blocks=8,
            )
        )
        for mode in ("payload_rsag", "bulk_push", "bulk_pull", "pull")
    )
    assert joint_startup_spec(prepared_joint).profiles == (
        (1, 8),
        (2, 8),
        (3, 8),
        (4, 8),
    )


def test_sample_passes_separate_groups_and_publish_after_every_sample(monkeypatch):
    import torch

    events = []

    def producer(name, device="cuda:0", rank=0, members=(0, 1, 2, 3)):
        return SimpleNamespace(
            runtime=SimpleNamespace(device=device, exchange_group=members),
            specs=(SimpleNamespace(rank=rank),),
            _reset_validate=lambda: events.append((name, "validate")),
            _sample=lambda sample: events.append((name, sample)),
            _publish=lambda: events.append((name, "publish")),
        )

    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda members: list(members)
    )
    first, second = producer("first"), producer("second")
    run_joint_startup_tuners((first, second))
    assert events == [
        ("first", "validate"),
        ("second", "validate"),
        *((name, sample) for sample in range(21) for name in ("first", "second")),
        ("first", "publish"),
        ("second", "publish"),
    ]
    for invalid in ((), (first, first)):
        with pytest.raises(ValueError, match="nonempty and distinct"):
            run_joint_startup_tuners(invalid)
    for other in (
        producer("other", device="cuda:1"),
        producer("other", rank=1),
        producer("other", members=(1, 0, 2, 3)),
    ):
        with pytest.raises(ValueError, match="ordered rank membership"):
            run_joint_startup_tuners((first, other))


@pytest.mark.parametrize("profile_count", (1, 2, 3, 4, 6))
def test_each_profile_samples_all_fixtures_with_balanced_positions(profile_count):
    from collections import Counter
    import torch
    from b12x.comm.pcie._joint_startup import JointStartupTuner

    producer = JointStartupTuner.__new__(JointStartupTuner)
    profiles = tuple((mode, 1) for mode in range(profile_count))
    selector = torch.zeros(1, dtype=torch.int32)
    producer.specs = (SimpleNamespace(profiles=profiles),)
    producer.state = SimpleNamespace(selector=selector)
    produced, events = [], []
    producer.inputs = SimpleNamespace(produce=produced.append, repetitions=3)
    producer.eviction, producer.eviction_sum = torch.ones(8), torch.empty(())
    producer._timing = lambda action, sample, selected: events.append(
        (action, sample, int(selected.item()) >> 16, produced[-1])
    )
    calls = []
    producer._collective = lambda: calls.append(int(selector.item()) >> 16)
    for sample in range(21):
        producer._sample(sample)
    begin = [event for event in events if event[0] == "begin"]
    assert len(calls) == 21 * profile_count * 3
    assert len(begin) == 21 * profile_count
    for mode in range(profile_count):
        own = [event for event in begin if event[2] == mode]
        assert {event[1] for event in own} == set(range(mode * 21, (mode + 1) * 21))
        assert Counter(event[3] for event in own) == {0: 7, 1: 7, 2: 7}
        for fixture in range(3):
            positions = Counter(
                index % profile_count
                for index, event in enumerate(begin)
                if event[2] == mode and event[3] == fixture
            )
            assert max(positions.values()) - min(positions.values()) <= 1
            assert set(positions) == set(range(profile_count))
    for invalid in (-1, 21, True):
        with pytest.raises(ValueError, match="sample index"):
            producer._sample(invalid)


def test_validation_checks_each_profile_against_every_exact_fixture():
    import torch
    from b12x.comm.pcie._joint_startup import JointStartupTuner

    producer = JointStartupTuner.__new__(JointStartupTuner)
    producer.state = SimpleNamespace(selector=torch.zeros(1, dtype=torch.int32))
    producer.specs = (SimpleNamespace(profiles=((0, 1), (1, 8), (4, 4))),)
    expected = tuple(torch.full((8,), fixture + 1.0) for fixture in range(3))
    output = torch.empty(8)
    produced, events = [], []
    producer.inputs = SimpleNamespace(
        produce=produced.append,
        out=output,
        reference=SimpleNamespace(expected=expected),
    )
    producer.mismatch = torch.empty(8, dtype=torch.bool)
    producer.invalid = torch.empty((), dtype=torch.int32)
    producer._collective = lambda: output.copy_(expected[produced[-1]])
    producer._timing = lambda action, sample, selector: events.append(
        (
            action,
            int(selector.item()) >> 16,
            None if action == "reset" else produced[-1],
            None if action == "reset" else int(producer.invalid.item()),
        )
    )
    producer._reset_validate()
    assert events[0][0] == "reset"
    assert events[1:] == [
        ("validate", mode, fixture, 0) for mode in (0, 1, 4) for fixture in range(3)
    ]


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_PCIE_TEST") != "1", reason="requires an assigned GPU"
)
def test_device_publication_scores_total_time_and_clears_stale_success():
    import torch
    from b12x.comm.pcie._joint_startup_cute import timing_launcher

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign CUDA devices explicitly"
    device = torch.cuda.current_device()
    peers = torch.zeros(2, dtype=torch.uint64, device=device)
    local = torch.zeros(4, dtype=torch.int64, device=device)
    samples = torch.ones((3, 21), dtype=torch.int64, device=device)
    invalid = torch.zeros(1, dtype=torch.int32, device=device)
    profiles = torch.tensor(
        [(1 << 16) | 1, (3 << 16) | 8, (4 << 16) | 8], dtype=torch.uint32, device=device
    )
    selector = torch.zeros(1, dtype=torch.uint32, device=device)
    status = torch.zeros(1, dtype=torch.int32, device=device)
    addresses = tuple(
        value.data_ptr()
        for value in (peers, local, samples, invalid, profiles, selector, status)
    )
    reset = timing_launcher(2, 0, "reset", 3, device)
    publish = timing_launcher(2, 0, "publish", 3, device)
    reset(*addresses, 0)
    torch.cuda.synchronize()
    assert status.item() == -1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reset(*addresses, 0)
        publish(*addresses, 0)
    stable = [[1] * 15 + [1000] * 6, [20] * 21, [20] * 21]
    for data, expected_status in (
        (stable, 0),
        ([stable[0], [0] + [20] * 20, stable[2]], 1),
        (stable, 0),
        ([stable[0], [-1] + [20] * 20, stable[2]], 1),
        ([[2**63 - 1] + [1] * 20, stable[1], stable[2]], 1),
    ):
        samples.copy_(torch.tensor(data, dtype=torch.int64))
        status.zero_()
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
        assert status.item() == expected_status
        if expected_status == 0:
            assert selector.item() == profiles[1].item()
    graph.reset()
