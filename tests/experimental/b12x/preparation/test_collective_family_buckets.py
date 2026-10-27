"""Collective calibration selects two families before per-request launch races."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.preparation import (
    CollectiveRequirement,
    FrozenMapping,
    ParameterSpace,
    PreparedCall,
    autotune_effort,
)
from b12x.preparation import _collective
from .test_defaults import Query, contract
from .test_session import _deterministic_timer, request, session


def declarations(rows):
    from b12x.preparation._efficiency import capture_autotune_effort

    tuning = replace(
        contract(),
        parameters=lambda query, device: ParameterSpace(
            knobs=contract().knobs, autotune_effort=capture_autotune_effort()
        ),
        collective_signature=lambda query: query.rows,
        collective_family_bucket=lambda query: (
            "width",
            {"upper": 8 if query.rows <= 8 else 16},
            query.rows,
        ),
    )
    result = []
    for row in rows:
        item = request(
            name=f"rows-{row}",
            tuning=tuning,
            collective=CollectiveRequirement(f"group/{row}", (0, 1)),
            benchmark=lambda state: PreparedCall(
                run=lambda: state.value, output=state.value, produce=lambda: None
            ),
        )
        result.append(
            replace(
                item,
                plan=replace(
                    item.plan, query=Query(row), _benchmark_call=item.benchmark_call
                ),
            )
        )
    return tuple(result)


def coordinate(progress):
    if progress.ready_collective_tuning is not None:
        value = progress.ready_collective_tuning
        return (value, value)
    if progress.ready_collectives:
        return progress.ready_collectives[0].key


@pytest.fixture
def host_races(monkeypatch):
    _deterministic_timer(monkeypatch)
    monkeypatch.setattr(_collective, "capture_benchmark_call", lambda call, **kw: call)


def test_calibration_precedes_requests_and_cached_calibration_reuses_shortlist(
    tmp_path, host_races
):
    for cached in (False, True):
        catalogs = []

        def record(progress):
            value = progress.ready_collective_tuning
            if value is not None and value.step == "catalog":
                catalogs.append(value.key)
            return coordinate(progress)

        with session(tmp_path) as engine:
            snapshots = []
            result = engine.prepare(
                declarations((3, 5, 11, 13)),
                coordinator=record,
                progress=snapshots.append,
            )
            assert catalogs == ["group/5", "group/13", "group/3", "group/11"]
            assert result.benchmarked_candidates == (0 if cached else 10)
            assert result.cache_hits == (4 if cached else 0)
            assert all(
                p.completed_candidates <= p.total_candidates
                for p in snapshots
                if p.total_candidates is not None
            )
            assert all(item.config.width == 2 for item in result.selections.values())
            assert len(engine._collective_family_buckets) == 2
            assert all(
                value[1] == (2, 1)
                for value in engine._collective_family_buckets.values()
            )
            later = engine.prepare(declarations((7,)), coordinator=coordinate)
            assert later.benchmarked_candidates == (0 if cached else 2)
            repeat = engine.prepare(declarations((5,)), coordinator=coordinate)
            assert repeat.benchmarked_candidates == 0
            assert repeat.cache_hits == 1


@pytest.mark.parametrize("effort,count", (("audit", 12), ("off", 0)))
def test_effort_preserves_full_audit_and_untuned_defaults(
    tmp_path, host_races, effort, count
):
    with autotune_effort(effort), session(tmp_path) as engine:
        result = engine.prepare(declarations((3, 5, 11, 13)), coordinator=coordinate)
        assert result.benchmarked_candidates == count
        assert not engine._collective_family_buckets


def test_cancelled_calibration_does_not_publish_partial_family_results(
    tmp_path, host_races
):
    def cancel(progress):
        value = progress.ready_collective_tuning
        if value is not None and value.step == "bound/1":
            value = replace(
                value, payload=FrozenMapping({**value.payload, "cancel": True})
            )
            return (value, value)
        return coordinate(progress)

    with session(tmp_path) as engine:
        result = engine.prepare(declarations((5,)), coordinator=cancel)
        assert result.selections["rows-5"].source == "default"
        assert not engine._collective_family_buckets


def test_custom_benchmark_does_not_share_builtin_calibration(tmp_path, host_races):
    items = tuple(
        replace(
            item,
            benchmark_call=lambda state: PreparedCall(
                run=lambda: state.value, output=state.value, produce=lambda: None
            ),
        )
        for item in declarations((3, 5))
    )
    with session(tmp_path) as engine:
        result = engine.prepare(items, coordinator=coordinate)
        assert result.benchmarked_candidates == 6
        assert not engine._collective_family_buckets


def test_plain_buckets_follow_bytes_and_preserve_collective_semantics():
    from b12x.comm.pcie._collective_tuning import _family_bucket

    def query(shape, **changes):
        payload = dict(
            surface="OneshotAllReduce.all_reduce",
            topology="pcie_ipc",
            call=dict(dtype="bfloat16", stage_input=True),
            setup=dict(shape=shape),
        )
        return SimpleNamespace(
            operation="all_reduce",
            world_size=changes.get("world_size", 4),
            inplace=changes.get("inplace", False),
            alternatives={"pull": payload},
        )

    first = _family_bucket(query((32, 5120)))
    assert first[1] == _family_bucket(query((36, 5120)))[1]
    assert first[1] == _family_bucket(query((16, 10240)))[1]
    assert first[1] != _family_bucket(query((40, 5120)))[1]
    assert first[1] != _family_bucket(query((32, 5120), inplace=True))[1]
    assert first[1] != _family_bucket(query((32, 5120), world_size=2))[1]
    assert first[1]["upper_bytes"] == 48 * 8192
    assert _family_bucket(query((160, 5120)))[1]["upper_bytes"] == 256 * 8192


def test_finalist_policy_preserves_full_families_and_cached_candidate_indices(
    tmp_path, host_races
):
    from b12x.preparation import FinalistPolicy

    items = tuple(
        replace(item, finalist_policy=FinalistPolicy(family_fields=("width",)))
        for item in declarations((3, 5))
    )
    original = None
    for cached in (False, True):
        with session(tmp_path) as engine:
            result = engine.prepare(items, coordinator=coordinate)
            assert result.benchmarked_candidates == (0 if cached else 6)
            assert result.cache_hits == (2 if cached else 0)
            assert not engine._collective_family_buckets
            for name, finalists in result.finalists.items():
                assert {item.selection.config.width for item in finalists} == {1, 2, 4}
                assert {item.candidate_index for item in finalists} == {0, 1, 2}
                assert "bucket_families" not in result.coverage[name]
            if cached:
                assert result.finalists == {
                    name: tuple(
                        replace(
                            item, selection=replace(item.selection, source="cached")
                        )
                        for item in finalists
                    )
                    for name, finalists in original.items()
                }
            else:
                original = result.finalists
