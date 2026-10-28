"""Public contextual preparation retains ordinary plan and cache ownership."""

from dataclasses import replace
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from b12x.preparation import (
    CollectiveRequirement,
    ContextCandidate,
    ContextRaceResult,
    DetectedDevice,
    FinalistPolicy,
    FinalistRecord,
    FrozenMapping,
    MemoryRequirements,
    Plan,
    PreparationSession,
    PreparedCall,
    TuningRequirement,
)
from b12x.preparation._cache import SelectionCache
from .test_defaults import Config, Query, contract


def _plan():
    return Plan(
        contract=contract(values=(1, 2)),
        query=Query(3),
        override=Config(1),
        _compile_jobs=lambda config, device: (),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: SimpleNamespace(
            value=selection.config.width
        ),
    )


def _call():
    return PreparedCall(run=lambda: None, produce=lambda: None)


def test_finalist_policy_requires_legal_family_and_bounded_request():
    with pytest.raises(ValueError, match="distinct"):
        FinalistPolicy(family_fields=("tile", "tile"))
    with pytest.raises(ValueError, match="cover one family"):
        FinalistPolicy(family_fields=("tile",), per_family=2, max_total=1)
    with pytest.raises(ValueError, match="unpinned"):
        _plan().request(
            name="pinned",
            prepare_call=lambda state: _call(),
            benchmark_call=lambda state: _call(),
            finalist_policy=FinalistPolicy(family_fields=("width",)),
        )
    unpinned = replace(_plan(), override=None)
    collective = unpinned.request(
        name="collective",
        prepare_call=lambda state: _call(),
        benchmark_call=lambda state: _call(),
        collective=CollectiveRequirement("c", (0, 1)),
        finalist_policy=FinalistPolicy(family_fields=("width",)),
    )
    assert collective.finalist_policy.per_family == 1
    with pytest.raises(ValueError, match="one winner per family"):
        unpinned.request(
            name="collective",
            prepare_call=lambda state: _call(),
            benchmark_call=lambda state: _call(),
            collective=CollectiveRequirement("c", (0, 1)),
            finalist_policy=FinalistPolicy(family_fields=("width",), per_family=2),
        )
    policy = FinalistPolicy(family_fields=("width",))
    empty_shard = TuningRequirement(
        "race", (0, 1), None, None, None, finalist_policy=policy
    )
    assert empty_shard.finalists == ()
    with pytest.raises(ValueError, match="require their family policy"):
        TuningRequirement(
            "race",
            (0, 1),
            FrozenMapping({"width": 1}),
            1.0,
            0,
            finalists=(
                FinalistRecord(
                    assignment=FrozenMapping({"width": 1}),
                    family=FrozenMapping({"width": 1}),
                    candidate_index=0,
                    latency_us=1.0,
                ),
            ),
        )


def test_plan_alternative_identity_is_public_and_excludes_only_override():
    original = _plan()
    alternative = replace(original, override=Config(2))
    assert original.same_declaration_except_override(alternative)
    assert not original.same_declaration_except_override(
        replace(alternative, invocation=FrozenMapping({"mode": "other"}))
    )
    assert not original.same_declaration_except_override(
        replace(alternative, dependencies=("other",))
    )
    assert not original.same_declaration_except_override(_plan())


def test_selection_cache_persists_finalists_separately_from_winner(tmp_path):
    identity = {"schema_version": 23, "tuning_cache_version": 1}
    finalist = FinalistRecord(
        assignment=FrozenMapping({"width": 1}),
        family=FrozenMapping({"width": 1}),
        candidate_index=0,
        latency_us=2.5,
    )
    cache = SelectionCache(tmp_path, identity)
    cache.save(
        "context-enabled",
        assignment={"width": 2},
        config={"width": 2},
        coverage={"measured_count": 2},
        programs=(),
        finalists=(finalist,),
    )
    cache.save(
        "ordinary",
        assignment={"width": 2},
        config={"width": 2},
        coverage={"measured_count": 2},
        programs=(),
    )
    restored = SelectionCache(tmp_path, identity)
    assert restored.get("context-enabled")["assignment"]["width"] == 2
    assert restored.get("context-enabled")["finalists"][0]["candidate_index"] == 0
    assert "finalists" not in restored.get("ordinary")


def test_context_race_checks_catalog_and_live_plan_ownership():
    plan = _plan()
    candidate = ContextCandidate(name="one", plans=(plan,), call=_call())
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False
    ) as session:
        session.prepare((plan.request(name="p", prepare_call=lambda state: _call()),))
        with pytest.raises(ValueError, match="names must be unique"):
            session.race_contexts((candidate, candidate))
        with pytest.raises(ValueError, match="every rank"):
            session.race_contexts(
                (candidate,),
                collective_ranks=(0, 1),
                collective_exchange=lambda step, payload: (payload,),
            )
        session.release(plan)
        with pytest.raises(ValueError, match="not owned"):
            session.race_contexts((candidate,))


def test_context_result_is_immutable_and_separate_from_plan():
    result = ContextRaceResult(
        winner="paired",
        latencies_us={"paired": 3.0},
        selections={"paired": ()},
        cache_policy="cold",
    )
    assert result.winner == "paired"
    with pytest.raises(TypeError):
        result.latencies_us["paired"] = 1.0


@pytest.mark.parametrize(
    "field,value",
    (
        ("rounds", 2),
        ("samples", 3),
        ("producer_counts", (2,)),
        ("cache_policy", "warm"),
        ("ranks", (0, 2)),
    ),
)
def test_context_collective_rejects_measurement_mismatch_before_replay(field, value):
    plan = _plan()
    calls = []
    candidate = ContextCandidate(
        name="one",
        plans=(plan,),
        call=PreparedCall(
            run=lambda: calls.append("run"),
            produce=lambda: calls.append("produce"),
        ),
    )

    def exchange(step, payload):
        assert step == "catalog"
        peer = dict(payload)
        peer[field] = value
        return payload, peer

    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False, cache_policy="cold"
    ) as session:
        session.prepare((plan.request(name="p", prepare_call=lambda state: _call()),))
        with pytest.raises(ValueError, match=field):
            session.race_contexts(
                (candidate,),
                collective_ranks=(0, 1),
                collective_exchange=exchange,
                rounds=1,
                samples=1,
            )
    assert calls == []


def test_context_cold_eviction_follows_state_restoration(monkeypatch):
    from b12x.preparation import _measurement
    import torch

    events = []

    class Event:
        def __init__(self, **kwargs):
            self.name = "start" if not hasattr(Event, "first") else "end"
            Event.first = True

        def record(self, stream=None):
            events.append(self.name)

    class Gate:
        @contextmanager
        def hold(self, stream):
            events.append("gate")
            yield

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: object())
    monkeypatch.setattr(_measurement, "_collection_paused", nullcontext)
    call = PreparedCall(
        run=lambda: events.append("run"),
        reset=lambda: events.append("reset"),
        produce=lambda: events.append("produce"),
    )
    timer = _measurement._TimedCall(
        call,
        lambda: events.append("evict"),
        1,
        Gate(),
        eviction_after_produce=True,
    )
    events.clear()  # Timer initialization records its event pair once.
    timer.replay()
    assert events == ["reset", "produce", "evict", "gate", "start", "run", "end"]


@pytest.mark.parametrize("fail", (False, True))
def test_context_race_restores_then_synchronizes_before_closing_owners(
    monkeypatch, fail
):
    from b12x.preparation import _measurement

    events = []
    original = _plan()
    alternative = replace(original, override=Config(2))

    def call(name):
        return PreparedCall(
            run=lambda: events.append(f"run:{name}"),
            produce=lambda: events.append(f"produce:{name}"),
            restore=lambda: events.append(f"restore:{name}"),
            close=lambda: events.append(f"close:{name}"),
        )

    def prepare(calls, **kwargs):
        if False:
            yield
        return SimpleNamespace(close=lambda: events.append("race-close"))

    def measure(race, **kwargs):
        if False:
            yield
        if fail:
            raise RuntimeError("measurement failed")
        return _measurement.RaceMeasurements((2.0, 1.0), 0)

    monkeypatch.setattr(_measurement, "no_compilation", nullcontext)
    monkeypatch.setattr(_measurement, "_l2_flush_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False
    ) as session:
        session.prepare(
            (
                original.request(name="a", prepare_call=lambda state: _call()),
                alternative.request(name="b", prepare_call=lambda state: _call()),
            )
        )
        monkeypatch.setattr(session, "_synchronize", lambda: events.append("sync"))
        candidates = (
            ContextCandidate(name="a", plans=(original,), call=call("a")),
            ContextCandidate(name="b", plans=(alternative,), call=call("b")),
        )
        if fail:
            with pytest.raises(RuntimeError, match="measurement failed"):
                session.race_contexts(candidates)
        else:
            result = session.race_contexts(candidates)
            assert result.winner == "b"
            assert original.selection.config == Config(1)
            assert alternative.selection.config == Config(2)
        tail = events[events.index("race-close") :]
        assert tail == [
            "race-close",
            "restore:a",
            "restore:b",
            "sync",
            "close:a",
            "close:b",
        ]


def test_context_collective_uses_max_rank_samples_and_ordered_steps(monkeypatch):
    from b12x.preparation import _measurement

    original = _plan()
    alternative = replace(original, override=Config(2))
    observed = []

    def prepare(calls, **kwargs):
        if False:
            yield
        return SimpleNamespace(close=lambda: None)

    def measure(race, **kwargs):
        yield from kwargs["launch_exchange"]((0, 1), 0, 0)
        maxima = yield from kwargs["sample_exchange"](
            (0, 1), 0, 0, ((1.0, 1.0), (2.0, 2.0))
        )
        observed.append(maxima)
        return _measurement.RaceMeasurements(
            tuple(sum(group) / len(group) for group in maxima), 0
        )

    def exchange(step, payload):
        if step == "samples":
            slower = dict(payload)
            slower["groups"] = ((5.0, 5.0), (2.0, 2.0))
            return payload, slower
        return payload, payload

    monkeypatch.setattr(_measurement, "no_compilation", nullcontext)
    monkeypatch.setattr(_measurement, "_l2_flush_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)
    with PreparationSession(
        device=DetectedDevice(None, None), autotune=False
    ) as session:
        session.prepare(
            (
                original.request(name="a", prepare_call=lambda state: _call()),
                alternative.request(name="b", prepare_call=lambda state: _call()),
            )
        )
        candidates = (
            ContextCandidate(name="a", plans=(original,), call=_call()),
            ContextCandidate(name="b", plans=(alternative,), call=_call()),
        )
        result = session.race_contexts(
            candidates, collective_exchange=exchange, collective_ranks=(0, 1)
        )

        def wrong_turn(step, payload):
            if step == "launch":
                other = dict(payload)
                other["turn"] += 1
                return payload, other
            return payload, payload

        with pytest.raises(ValueError, match="different orders"):
            session.race_contexts(
                candidates, collective_exchange=wrong_turn, collective_ranks=(0, 1)
            )
    assert observed == [((5.0, 5.0), (2.0, 2.0))]
    assert result.winner == "b"  # Rank zero alone would choose a.
