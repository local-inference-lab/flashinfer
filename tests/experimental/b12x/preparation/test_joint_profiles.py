"""CPU coverage for fixed joint launch geometry and runtime profile admission."""

from dataclasses import replace

import pytest

from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
from b12x.comm.pcie._joint_profile import (
    JOINT_FAMILIES,
    JOINT_MODES,
    JointMode,
    joint_modes,
    joint_slot_packs,
    profile_index,
    profile_records,
    selected_profile,
)
from b12x.preparation import FrozenMapping, autotune_effort


@pytest.mark.parametrize("family", JOINT_FAMILIES)
def test_write_profiles_validate_peer_count_and_serial_cta_limit(family):
    records = profile_records(family, 64, world_size=2)
    assert all(
        mode not in (JointMode.SERIAL_PUSH, JointMode.QUEUED_PUSH)
        for mode, _ in records
    )
    for mode in ("serial_push", "queued_push"):
        with pytest.raises(ValueError, match="peer count"):
            selected_profile(family, 64, mode, 1, world_size=2)
    for world in (3, 4, 6, 8, 10):
        assert (
            selected_profile(family, 64, "serial_push", 1, world_size=world)
            == (int(JointMode.SERIAL_PUSH) << 16) | 1
        )
        assert (
            selected_profile(family, 64, "queued_push", 64, world_size=world)
            == (int(JointMode.QUEUED_PUSH) << 16) | 64
        )
        with pytest.raises(ValueError, match="exactly one"):
            selected_profile(family, 64, "serial_push", 2, world_size=world)


@pytest.mark.parametrize("world", (0, 1, 5, 12, 16, True))
def test_joint_modes_reject_unsupported_peer_counts(world):
    with pytest.raises(ValueError, match="peer count"):
        joint_modes("joint_latency", world)


@pytest.mark.parametrize("family", JOINT_FAMILIES)
@pytest.mark.parametrize("world", (2, 3, 4))
def test_profile_table_selects_every_admitted_mode_and_grid(family, world):
    modes = joint_modes(family, world)
    records = profile_records(family, 64, world_size=world)
    assert len(records) == 64 * (len(modes) - (world > 2)) + (world > 2)
    for mode in modes:
        for blocks in (1,) if mode == "serial_push" else range(1, 65):
            selected = selected_profile(family, 64, mode, blocks, world_size=world)
            assert selected >> 16 == JOINT_MODES[mode]
            assert selected & 0xFFFF == blocks
            assert records[
                profile_index(family, 64, mode, blocks, world_size=world)
            ] == (
                JOINT_MODES[mode],
                blocks,
            )
    for count in (0, 65, True):
        with pytest.raises(ValueError):
            profile_index(family, 64, modes[0], count, world_size=world)
        with pytest.raises(ValueError):
            selected_profile(family, 64, modes[0], count, world_size=world)
    with pytest.raises(ValueError):
        profile_index(family, 64, "invalid", 1, world_size=world)
    with pytest.raises(ValueError):
        selected_profile(family, 64, "invalid", 1, world_size=world)


@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("capacity", (1, 257, 4099))
def test_bulk_gather_cannot_overlap_incoming_scatter(world, capacity):
    shard = (capacity + world - 1) // world
    incoming = (world + 1) * capacity
    gather = incoming + world * shard
    end = joint_slot_packs(capacity, world)
    assert gather >= incoming + world * shard
    assert end - gather == world * shard
    assert end >= (world + 1) * capacity + 2 * capacity


def _query(family, packs, world=2):
    call = dict(
        threads=256,
        blocks=64,
        active_blocks=16,
        submode=JOINT_FAMILIES[family][0],
        dtype="bfloat16",
        stage_input=True,
        size_packs=packs,
    )
    config = CollectiveConfig(
        implementation=family,
        threads=256,
        blocks=64,
        active_blocks=16,
        submode=call["submode"],
    )
    return CollectiveQuery(
        operation="all_reduce",
        world_size=world,
        rank=0,
        alternatives=FrozenMapping(
            {
                family: dict(
                    surface="OneshotAllReduce.all_reduce",
                    world_size=world,
                    rank=0,
                    topology="pcie_ipc",
                    setup={},
                    call=call,
                )
            }
        ),
        default=config,
    )


@pytest.mark.parametrize("family", JOINT_FAMILIES)
@pytest.mark.parametrize("world", (2, 3, 4))
def test_joint_tuning_keeps_admitted_modes_at_large_payloads(family, world):
    query = _query(family, 1 << 24, world)
    with autotune_effort("default"):
        candidates = tuple(
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        )
    assert {c.submode for c in candidates} == set(joint_modes(family, world))
    assert any(c.submode == "serial_push" for c in candidates) == (world > 2)
    assert all(c.active_blocks == 1 for c in candidates if c.submode == "serial_push")
    assert {c.threads for c in candidates} == {32, 64, 128, 256, 512}
    assert {c.blocks for c in candidates} == {64}
    assert {c.active_blocks for c in candidates} == {1, 2, 4, 8, 16, 32, 36, 64}
    assert TUNING.knobs[-2].binding.value == "runtime"
    assert TUNING.knobs[-1].binding.value == "runtime"
    with autotune_effort("off"):
        assert tuple(
            c for _, c in TUNING.iterate(TUNING.configure(query, device=None))
        ) == (query.default,)
    with pytest.raises(ValueError):
        TUNING.validate_config(query, replace(query.default, submode="invalid"), None)


def test_queued_push_races_the_tile_covering_grid():
    query = _query("joint_bandwidth", 2400, 4)
    with autotune_effort("default"):
        candidates = tuple(
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        )
    assert any(
        config.submode == "queued_push"
        and config.threads == 64
        and config.active_blocks == 3
        for config in candidates
    )
    assert any(
        config.submode == "queued_push"
        and config.threads == 32
        and config.active_blocks == 5
        for config in candidates
    )


def test_joint_and_standalone_candidates_share_compile_geometry():
    standalone = _query("joint_latency", 4099)
    payload = standalone.alternatives["joint_latency"].to_dict()
    payload["call"].pop("submode")
    payload["call"].pop("active_blocks")
    payload["call"].update(blocks=16)
    alternatives = {
        name: payload
        for name in (
            "pull",
            "tp2_remote_push",
            "tp2_remote_push_simple",
            "remote_push_ll",
            "remote_push_simple",
            "remote_push_full",
            "remote_push_rsag",
        )
    }
    for family in JOINT_FAMILIES:
        alternatives.update(_query(family, 4099).alternatives)
    query = replace(
        standalone,
        alternatives=FrozenMapping(alternatives),
        default=CollectiveConfig(implementation="pull", threads=256, blocks=16),
    )
    with autotune_effort("default"):
        configuration = TUNING.configure(query, device=None)
        candidates = tuple(TUNING.iterate(configuration))
    assert {config.implementation for _, config in candidates} == set(alternatives)
    joint = [
        (assignment, config)
        for assignment, config in candidates
        if config.implementation in JOINT_FAMILIES
    ]
    assert len(joint) >= 280
    assert (
        len(
            {
                configuration.space.compile_assignment(assignment)
                for assignment, _ in joint
            }
        )
        == 10
    )


def test_joint_workspace_does_not_change_native_resident_declaration():
    from types import SimpleNamespace

    import torch

    from b12x.comm.pcie._oneshot_preparation import _owned_resident_layout
    from b12x.comm.pcie.pcie_oneshot import _OwnedSharedBuffer

    native_buffer = _OwnedSharedBuffer(4096, (4096, 8192), (8192,), 4096)
    runtime = SimpleNamespace(
        rank_data=torch.empty(128, dtype=torch.uint8),
        world_size=2,
        _sentinel_supported=True,
        _ext=SimpleNamespace(meta_size=lambda: 256),
        _owned_buffers=[native_buffer],
    )
    native = SimpleNamespace(eager_buffer_bytes=1024, transport_policy=(False,) * 5)
    before = _owned_resident_layout(runtime, native)
    joint_buffer = _OwnedSharedBuffer(16384, (16384, 32768), (32768,), 16384)
    runtime._owned_buffers.append(joint_buffer)
    runtime._joint_workspace = SimpleNamespace(shared=joint_buffer)
    assert _owned_resident_layout(runtime, native) == before
    assert before[2] == 1
    runtime._owned_buffers.append(native_buffer)
    assert _owned_resident_layout(runtime, native) == (*before[:2], 2)


def test_materialization_workspace_lookup_never_provisions_a_channel():
    from types import SimpleNamespace

    from b12x.comm.pcie._joint_preparation import _workspace

    runtime = SimpleNamespace()
    with pytest.raises(RuntimeError, match="coordinated channel setup"):
        _workspace(runtime)
    prepared = object()
    runtime._joint_workspace = prepared
    assert _workspace(runtime) is prepared


def test_native_and_joint_plans_account_for_the_same_channel_workspace():
    from types import SimpleNamespace

    import torch

    from b12x.comm.pcie import _oneshot_preparation, _joint_preparation
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.comm.pcie.pcie_oneshot import _OwnedSharedBuffer

    native = SimpleNamespace(eager_buffer_bytes=1024, transport_policy=(False,) * 5)
    shared = _OwnedSharedBuffer(16384, (16384, 32768), (32768,), 16384)
    runtime = SimpleNamespace(
        rank_data=torch.empty(128, dtype=torch.uint8),
        world_size=2,
        _sentinel_supported=True,
        _ptr=1,
        device=torch.device("cpu"),
        _ext=SimpleNamespace(meta_size=lambda: 256, _state=lambda pointer: native),
        _owned_buffers=[_OwnedSharedBuffer(4096, (4096, 8192), (8192,), 4096), shared],
        _joint_workspace=SimpleNamespace(
            shared=shared,
            peer_table=torch.empty((2, 16), dtype=torch.uint64),
            signal_table=torch.empty(16, dtype=torch.uint64),
        ),
    )
    rank_bytes, slab_bytes, slabs = _oneshot_preparation._owned_resident_layout(
        runtime, native
    )
    joint_bytes = _oneshot_preparation._joint_workspace_nbytes(runtime)
    assert joint_bytes == 16384 + 384
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=2,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(dict(transport="pull")),
        setup=FrozenMapping(
            dict(
                rank_data_nbytes=rank_bytes,
                owned_slab_nbytes=slab_bytes,
                owned_slab_count=slabs,
                joint_workspace_nbytes=joint_bytes,
                joint_max_blocks=64,
            )
        ),
    )
    plain = _oneshot_preparation.plan(query, runtime=runtime)._memory_requirements(
        None, None
    )
    joint_query = replace(
        query,
        call=FrozenMapping(
            dict(
                transport="joint_latency",
                threads=256,
                blocks=64,
                submode="full",
                active_blocks=16,
                dtype="bfloat16",
                stage_input=True,
            )
        ),
    )
    joint = _joint_preparation.plan(joint_query, runtime=runtime).memory_requirements()
    assert plain.persistent[0] == joint.persistent[0]
    assert (
        plain.persistent[0].resident_nbytes
        == rank_bytes + slab_bytes * slabs + joint_bytes
    )
    assert (
        sum(item.resident_nbytes for item in joint.persistent)
        == plain.persistent[0].resident_nbytes + 4
    )


def test_external_channel_does_not_advertise_unprovisioned_joint_candidates(
    monkeypatch,
):
    from types import SimpleNamespace

    import torch

    from b12x.comm.pcie._collective_tuning import oneshot_alternatives
    from b12x.comm.pcie._oneshot_preparation import _joint_max_blocks
    from b12x.comm.pcie._tuning import PcieQuery

    runtime = SimpleNamespace(_sentinel_supported=True)
    assert _joint_max_blocks(runtime) == 0
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=2,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                transport="pull",
                dtype="bfloat16",
                threads=256,
                blocks=4,
                eager_threads=256,
                eager_blocks=4,
                size_packs=32,
                stage_input=True,
                remote_push_region_packs=0,
                payload_ready_region_packs=1024,
            )
        ),
        setup=FrozenMapping(
            dict(
                eager_buffer_bytes=16384,
                eager_storage_shards=3,
                variants=((False, 0), (True, 0), (True, 1)),
                joint_max_blocks=_joint_max_blocks(runtime),
            )
        ),
    )
    alternatives = oneshot_alternatives(query, runtime)
    assert "remote_push_full" in alternatives
    assert not set(JOINT_FAMILIES).intersection(alternatives)
    assert not hasattr(runtime, "_joint_workspace")
    runtime._joint_workspace = object()
    runtime.device = torch.device("cuda", 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=48),
    )
    provisioned = replace(
        query,
        setup=FrozenMapping(
            {**query.setup, "joint_max_blocks": _joint_max_blocks(runtime)}
        ),
    )
    candidates = oneshot_alternatives(provisioned, runtime)
    assert set(JOINT_FAMILIES).issubset(candidates)
    assert all(candidates[family][0].call["blocks"] == 48 for family in JOINT_FAMILIES)
