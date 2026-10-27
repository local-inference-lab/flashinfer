"""Completed races leave self-describing records for retrospective analysis."""

import pytest

from b12x.preparation import Knob, ParameterSpace, PreparedCall
from b12x.preparation._ledger import admission_levels
from b12x.tools import tuning_ledger

from .test_defaults import contract
from .test_session import _deterministic_timer, request, session


def test_admission_levels_follow_the_efficiency_tiers():
    space = ParameterSpace.create(
        (Knob(name="tile", values=(8, 16, 32, 64)),),
        predicates=(lambda p: p["tile"] != 8,),
        efficiency_predicates=(lambda p: p["tile"] <= 16,),
        autotune_effort="audit",
    )
    candidates = [(assignment, None) for assignment in space.configurations()]
    assert [assignment["tile"] for assignment, _ in candidates] == [16, 32, 64]
    levels, counts = admission_levels(space, candidates)
    assert levels == ["default", "audit", "audit"]
    assert counts == {"default": 1, "audit": 3}
    assert admission_levels(space, candidates, cartesian_count=10**7) == (
        ["default", "audit", "audit"],
        None,
    )


def _benchmark(state):
    return PreparedCall(run=lambda: state.value, produce=lambda: None)


def test_complete_race_appends_one_record_and_cached_restart_adds_none(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    with session(tmp_path) as engine:
        result = engine.prepare(
            (request(name="first", tuning=contract(), benchmark=_benchmark),)
        )
        assert result.selections["first"].source == "tuned"
        rows = engine._race_progress_cache().ledger_records()
        session_id = engine.session_id
    assert len(rows) == 1
    row = rows[0]
    assert (row["shard_index"], row["shard_count"], row["session"]) == (
        0,
        1,
        session_id,
    )
    record = row["record"]
    selection = result.selections["first"]
    assert record["component_id"] == "test.arithmetic"
    assert record["query"] == selection.query.to_dict()
    assert record["kind"] == "local"
    assert record["effort"] == "default"
    # The heuristic default (width 7) lies outside the candidate set.
    assert record["default_index"] is None
    assert [c["assignment"] for c in record["candidates"]] == [
        {"width": 1},
        {"width": 2},
        {"width": 4},
    ]
    assert all(c["admission"] == "default" for c in record["candidates"])
    assert record["coverage"]["legal_by_effort"] == {"default": 3, "audit": 3}
    assert record["coverage"]["measured_count"] == 3
    # The deterministic timer scores |output - 6| + 1 with output = width * rows.
    assert record["champion"]["index"] == 1
    assert selection.assignment.to_dict() == {"width": 2}
    assert record["candidates"][1]["latency_us"] == record["champion"]["latency_us"]
    assert all(c["local"] and not c["restored"] for c in record["candidates"])
    assert all(c["batch"] is not None for c in record["candidates"])
    assert record["batches"] and record["batches"][-1]["winner"] == 1
    assert record["protocol"]["rounds"] == 15 and record["protocol"]["samples"] == 2
    assert set(record["phase_seconds"]) >= {"compile", "prepare", "measure"}
    assert record["seconds"] >= 0
    assert row["key"] == record["key"]

    with session(tmp_path) as engine:
        result = engine.prepare(
            (request(name="again", tuning=contract(), benchmark=_benchmark),)
        )
        assert result.selections["again"].source == "cached"
        assert len(engine._race_progress_cache().ledger_records()) == 1

    loaded = tuning_ledger.load_records(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["_ledger"]["session"] == session_id
    assert loaded[0]["_ledger"]["selected"]["assignment"] == {"width": 2}
    analysis = tuning_ledger.analyze(loaded[0])
    assert analysis["winner_index"] == 1
    assert analysis["default_regret"] is None
    assert analysis["pruning_regret"] == {"default": 0.0}
    assert analysis["measured"] == 3
    assert "test.arithmetic" in tuning_ledger.summarize([analysis])
    assert "test.arithmetic" in tuning_ledger.describe_records([analysis])


def _record(*, key, effort, default_index, candidates, winner=None):
    """Minimal ledger record: candidates are (admission, latency_us, round_means)."""
    entries = []
    for index, (admission, latency, means) in enumerate(candidates):
        entries.append(
            {
                "index": index,
                "assignment": {"i": index},
                "config": {"i": index},
                "programs": [],
                "admission": admission,
                "local": True,
                "restored": False,
                "precompiled": index % 2 == 0,
                "batch": 1,
                "resident_bytes": 0,
                "prepare_seconds": 0.1,
                "latency_us": latency,
                "rounds": [
                    {
                        "round": turn,
                        "n": 4,
                        "repetitions": 2,
                        "mean_us": mean,
                        "stdev_us": 0.0,
                    }
                    for turn, mean in enumerate(means)
                ],
            }
        )
    measured = [c for c in entries if c["latency_us"] is not None]
    champion = (
        min(measured, key=lambda c: (c["latency_us"], c["index"]))["index"]
        if winner is None
        else winner
    )
    return {
        "ledger_version": 1,
        "kind": "local",
        "key": key,
        "request": "r",
        "component_id": "synthetic.op",
        "query": {"rows": 8},
        "effort": effort,
        "coverage": {"legal_by_effort": {"default": 2, "audit": 3}},
        "default_index": default_index,
        "champion": {"index": champion, "latency_us": entries[champion]["latency_us"]},
        "candidates": entries,
        "rejected": [],
        "rejected_count": 0,
        "batches": [],
        "phase_seconds": {"compile": 1.0, "prepare": 2.0, "measure": 3.0},
        "started_at": 0.0,
        "seconds": 6.0,
        "_ledger": {"shard_count": 1, "shard_index": 0, "session": key},
    }


def test_analysis_regrets_margins_and_noise_follow_the_definitions():
    record = _record(
        key="a",
        effort="audit",
        default_index=0,
        candidates=[
            ("default", 10.0, (10.0, 10.0, 10.0)),
            ("default", 9.0, (8.5, 9.0, 9.5)),
            ("audit", 8.0, (8.0, 8.0, 8.0)),
            ("audit", None, ()),
        ],
    )
    analysis = tuning_ledger.analyze(record)
    assert analysis["winner_index"] == 2
    assert analysis["default_regret"] == pytest.approx(0.25)
    assert analysis["pruning_regret"] == pytest.approx({"default": 0.125, "audit": 0.0})
    assert analysis["runner_up_ratio"] == pytest.approx(9 / 8)
    assert analysis["within"] == {"1": 1, "2": 1, "5": 1, "10": 1}
    assert analysis["noise_cv"] == pytest.approx(0.0)
    assert analysis["winner_cv"] == pytest.approx(0.0)
    assert analysis["compiled_candidates"] == 2
    assert analysis["measured"] == 3 and analysis["candidates"] == 4
    assert tuning_ledger.analyze({**record, "candidates": []}) is None


def test_replications_report_agreement_spread_and_cross_regret():
    first = _record(
        key="k",
        effort="default",
        default_index=None,
        candidates=[
            ("default", 10.0, ()),
            ("default", 10.5, ()),
            ("default", 20.0, ()),
        ],
    )
    second = _record(
        key="k",
        effort="default",
        default_index=None,
        candidates=[
            ("default", 10.4, ()),
            ("default", 10.0, ()),
            ("default", 22.0, ()),
        ],
    )
    rows = tuning_ledger.replications([first, second])
    assert len(rows) == 1
    row = rows[0]
    assert row["replications"] == 2
    assert row["distinct_winners"] == 2
    assert row["winner_agreement"] == pytest.approx(0.5)
    # Candidate 2 moves by 10%, candidates 0 and 1 by 4% and 5%.
    assert row["candidate_spread_p50"] == pytest.approx(0.05)
    assert row["candidate_spread_p90"] == pytest.approx(0.10)
    # Each run's winner is 4% or 5% behind the other run's best.
    assert row["cross_regret_p50"] == pytest.approx(0.045)
    assert row["cross_regret_max"] == pytest.approx(0.05)
    assert "synthetic.op" in tuning_ledger.describe_replications(rows)
    assert tuning_ledger.replications([first]) == []


def test_shard_records_merge_into_one_run_per_start_order():
    def shard(index, started, candidates, session):
        record = _record(
            key="tp", effort="audit", default_index=0, candidates=candidates
        )
        record["started_at"] = started
        record["_ledger"] = {
            "shard_count": 2,
            "shard_index": index,
            "session": session,
            "wall_time": started,
        }
        return record

    # Rank 0 measures even candidates, rank 1 odd ones; two complete runs and
    # a trailing lone shard.
    d, a = "default", "audit"
    records = [
        shard(0, 10.0, [(d, 10.0, ()), (d, None, ()), (a, 9.0, ())], "a0"),
        shard(1, 10.5, [(d, None, ()), (d, 8.0, ()), (a, None, ())], "a1"),
        shard(1, 20.5, [(d, None, ()), (d, 8.5, ()), (a, None, ())], "b1"),
        shard(0, 20.0, [(d, 10.0, ()), (d, None, ()), (a, 7.0, ())], "b0"),
        shard(0, 30.0, [(d, 10.0, ()), (d, None, ()), (a, 9.5, ())], "c0"),
    ]
    runs = tuning_ledger.merge_runs(records)
    assert [run["_ledger"].get("shards_merged") for run in runs] == [2, 2, 1]
    assert runs[0]["_ledger"]["sessions"] == ["a0", "a1"]
    first = tuning_ledger.analyze(runs[0])
    assert first["measured"] == 3 and first["winner_index"] == 1
    assert first["default_regret"] == pytest.approx(0.25)
    assert first["pruning_regret"] == pytest.approx({"default": 0.0, "audit": 0.0})
    assert runs[0]["phase_seconds"]["measure"] == pytest.approx(6.0)
    second = tuning_ledger.analyze(runs[1])
    assert second["winner_index"] == 2 and second["default_regret"] == pytest.approx(
        10 / 7 - 1
    )
    # Only complete runs replicate; the lone trailing shard is excluded.
    rows = tuning_ledger.replications(runs)
    assert len(rows) == 1 and rows[0]["replications"] == 2
    assert rows[0]["distinct_winners"] == 2
