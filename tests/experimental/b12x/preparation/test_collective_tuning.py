"""Collective exchanges preserve identical candidate schedules across ranks."""

from dataclasses import replace

import pytest

from b12x.preparation import CollectiveTuningRequirement, FrozenMapping
from b12x.preparation._collective import exchange, _same
from b12x.preparation.types import CollectiveRequirement
from .test_session import request
from .test_race_elimination import _Timer, _race, host_timing
from b12x.preparation._measurement import measure_race_steps


def test_family_benchmark_is_used_by_default_and_caller_can_override():
    req = request(name="source")
    builtin, supplied = lambda state: None, lambda state: state
    plan = replace(req.plan, _benchmark_call=builtin)
    assert (
        plan.request(name="a", prepare_call=req.prepare_call).benchmark_call is builtin
    )
    assert (
        plan.request(
            name="b", prepare_call=req.prepare_call, benchmark_call=supplied
        ).benchmark_call
        is supplied
    )


def test_exchange_preserves_rank_order_and_rejects_different_catalogs():
    requirement = CollectiveRequirement("reduce", (2, 5))
    steps = exchange(requirement, "catalog", candidates=(1, 2))
    local = next(steps)
    assert local == CollectiveTuningRequirement(
        "reduce", (2, 5), "catalog", FrozenMapping({"candidates": (1, 2)})
    )
    with pytest.raises(StopIteration) as result:
        steps.send(
            (local, replace(local, payload=FrozenMapping({"candidates": (1, 3)})))
        )
    with pytest.raises(ValueError, match="candidates"):
        _same(result.value.value, "candidates")


def test_dcp_candidates_coalesce_effective_grids_and_preserve_overrides():
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        TUNING,
    )
    from b12x.preparation import autotune_effort

    native = dict(
        surface="DcpAllToAll.all_gather_heads",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=dict(threads=256, block_limit=16, shape=(1, 4, 512)),
        setup=dict(total_heads=16, threads_override=0, blocks_override=0),
    )
    query = CollectiveQuery(
        operation="all_gather_heads",
        world_size=4,
        rank=0,
        alternatives=FrozenMapping({"dcp": native}),
        default=CollectiveConfig(implementation="dcp", threads=256, blocks=16),
    )
    with autotune_effort("default"):
        values = list(TUNING.iterate(TUNING.configure(query, device=None)))
    assert len(values) == 3
    assert {config.threads for _, config in values} == {128, 256, 512}
    native["setup"].update(threads_override=256, blocks_override=16)
    query = replace(query, alternatives=FrozenMapping({"dcp": native}))
    with autotune_effort("default"):
        values = list(TUNING.iterate(TUNING.configure(query, device=None)))
    assert len(values) == 1
    assert values[0][1] == query.default


@pytest.mark.parametrize("shape", [(3, 1536), (513, 2560), (8,)])
@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("dtype", ("bfloat16", "float16"))
def test_plain_catalog_uses_storage_bounds_instead_of_shape_policy(shape, world, dtype):
    from math import prod
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        TUNING,
        oneshot_alternatives,
    )
    from b12x.comm.pcie._tuning import PcieQuery, TUNING as NATIVE
    from b12x.preparation import autotune_effort

    packs = prod(shape) // 8
    capacity = max(packs, 64)
    native = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=world,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                dtype=dtype,
                transport="pull",
                threads=256,
                blocks=8,
                eager_threads=256,
                eager_blocks=8,
                stage_input=True,
                size_packs=packs,
                remote_push_region_packs=capacity,
                payload_ready_region_packs=(1 + 2 * (world - 1)) * capacity,
            )
        ),
        setup=FrozenMapping(
            dict(
                shape=shape,
                variants=((False, 0), (True, 0), (True, 1)),
                eager_buffer_bytes=capacity * 16,
                eager_storage_shards=1 + 2 * (world - 1) + world,
            )
        ),
    )
    alternatives = oneshot_alternatives(native, object())
    expected = {"pull", "remote_push_ll", "remote_push_simple"}
    if world == 2:
        expected |= {"tp2_remote_push", "tp2_remote_push_simple"}
    if dtype == "bfloat16":
        expected |= {"remote_push_full", "remote_push_rsag"}
    assert set(alternatives) == expected
    query = CollectiveQuery(
        operation="all_reduce",
        world_size=world,
        rank=0,
        alternatives=FrozenMapping(
            {k: NATIVE.encode_query(v[0]) for k, v in alternatives.items()}
        ),
        default=CollectiveConfig(
            implementation="remote_push_ll", threads=128, blocks=36
        ),
    )
    with autotune_effort("default"):
        configs = [
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        ]
    assert {config.implementation for config in configs} == set(alternatives)
    assert all(1 <= config.blocks <= 36 for config in configs)
    assert (
        CollectiveConfig(implementation="remote_push_ll", threads=128, blocks=36)
        in configs
    )
    with pytest.raises(ValueError, match="parameter space"):
        TUNING.validate_config(
            query, CollectiveConfig(implementation="pull", threads=128, blocks=37), None
        )
    small = replace(
        native,
        call=FrozenMapping(
            {**native.call, "size_packs": 1, "remote_push_region_packs": 1}
        ),
        setup=FrozenMapping({**native.setup, "shape": (8,), "eager_buffer_bytes": 16}),
    )
    small_expected = {"pull", "remote_push_ll"}
    if world == 2:
        small_expected.add("tp2_remote_push")
    assert set(oneshot_alternatives(small, object())) == small_expected
    absent = replace(
        native, setup=FrozenMapping({**native.setup, "eager_storage_shards": 1})
    )
    assert set(oneshot_alternatives(absent, object())) == {"pull"}

    for default in {config.implementation: config for config in configs}.values():
        other = replace(query, default=default)
        with autotune_effort("default"):
            measured = {
                config
                for _, config in TUNING.iterate(TUNING.configure(other, device=None))
            }
        assert measured == set(configs)

    for world_size, rank in ((world, -1), (world, world), (5, 0)):
        with pytest.raises(ValueError, match="world size and rank"):
            TUNING.validate_query(
                replace(query, world_size=world_size, rank=rank), None
            )


def test_collective_signature_distinguishes_shapes_but_not_rank_ordinals():
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        _collective_signature,
    )

    def query(rank, rows):
        return CollectiveQuery(
            operation="all_reduce",
            world_size=4,
            rank=rank,
            default=CollectiveConfig(implementation="pull", threads=256, blocks=16),
            alternatives=FrozenMapping(
                {
                    "pull": dict(
                        rank=rank,
                        world_size=4,
                        call=dict(shape=(rows, 5120), dtype="bf16", device_index=rank),
                        setup=dict(device_index=rank),
                    )
                }
            ),
        )

    assert _collective_signature(query(0, 8)) == _collective_signature(query(3, 8))
    assert _collective_signature(query(0, 8)) != _collective_signature(query(0, 16))


def test_collective_samples_control_repetitions_and_elimination(
    host_timing, monkeypatch
):
    from b12x.preparation import _measurement

    monkeypatch.setattr(_measurement, "ROUND_BUDGET_US", 64)
    schedules, counts = [], []
    for rank in range(2):
        timers = (_Timer(1 if rank == 0 else 8), _Timer(4 if rank == 0 else 2))
        schedule = []

        def reduce(indices, turn, repetition, samples):
            schedule.append((indices, turn, repetition))
            yield "exchange"
            return tuple((8.0 if index == 0 else 4.0,) for index in indices)

        prepared = _race(timers)
        steps = measure_race_steps(
            prepared, device_ordinal=0, rounds=3, eliminate=True, sample_exchange=reduce
        )
        while True:
            try:
                next(steps)
            except StopIteration as result:
                assert result.value.latencies_us == (8.0, 4.0)
                break
        schedules.append(schedule)
        counts.append(tuple(timer.replays for timer in timers))
    assert schedules[0] == schedules[1]
    assert counts[0] == counts[1] == (8, 48)
