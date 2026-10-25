"""Preparation pipelines discovery and compilation ahead of candidate races."""

import os
from types import SimpleNamespace

from b12x._lib.compile_plan import ProgramKey
from b12x._lib.compile_pool import CompilationPlan, CompileJob
from b12x.preparation import MemoryRequirements, Plan, PreparedCall
from b12x.preparation import session as session_module
from b12x.preparation._cache import SelectionCache

from .test_defaults import Query, contract
from .test_session import _deterministic_timer, session


class Pool:
    """Optional work stays queued; required work completes immediately."""

    def __init__(self, available):
        self.available = available
        self.submissions = []

    def submit_plans(self, plans, *, required=True):
        for plan in plans:
            self.submissions.append((plan.programs, required))
            if required:
                self.available.update(plan.programs)


def _program(tag, width):
    return ProgramKey("cute", f"{tag}-{width}")


def _request(tag, rows, prepared):
    def benchmark(state):
        prepared.append((tag, state.width))
        return PreparedCall(run=lambda: state.value, produce=lambda: None)

    plan = Plan(
        contract=contract(),
        query=Query(rows),
        _compile_jobs=lambda config, device: (
            CompileJob.create("test.pipeline:compile", tag, config.width),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=lambda selection, device: SimpleNamespace(
            width=selection.config.width, value=selection.config.width * 3
        ),
    )
    return plan.request(
        name=tag,
        prepare_call=lambda state: PreparedCall(run=lambda: None),
        benchmark_call=benchmark,
    )


def test_lookahead_ready_first_cold_promotion_and_one_selection_write(
    tmp_path, monkeypatch
):
    _deterministic_timer(monkeypatch)
    available = {_program("first", 4)}
    pool = Pool(available)
    monkeypatch.setattr(
        session_module,
        "describe_compilation",
        lambda job: CompilationPlan(job, (_program(*job.args),)),
    )
    monkeypatch.setattr(
        session_module, "compiled_program_available", available.__contains__
    )
    writes = []
    real_replace = os.replace

    def replace(source, target):
        if str(target).endswith(".json"):
            writes.append(target)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    prepared = []
    first, second = _request("first", 3, prepared), _request("second", 5, prepared)
    with session(tmp_path, compile_workers=2) as engine:
        monkeypatch.setattr(engine, "_compiler", lambda: pool)
        result = engine.prepare((first, second))

    # Job start queues both obligations' candidates, in race order, before the
    # first race makes anything required.
    optional = [programs for programs, required in pool.submissions if not required]
    first_required = next(
        index for index, (_, required) in enumerate(pool.submissions) if required
    )
    head = [programs for programs, _ in pool.submissions[:first_required]]
    assert head == [
        (_program("first", 1),),
        (_program("first", 2),),
        (_program("second", 1),),
        (_program("second", 2),),
        (_program("second", 4),),
    ]
    assert (_program("first", 4),) not in optional
    # The precompiled candidate is prepared first; the cold tail is promoted
    # as one block before its candidates are prepared.
    assert [width for tag, width in prepared if tag == "first"] == [4, 1, 2]
    promoted = {programs for programs, required in pool.submissions if required}
    assert {(_program("first", 1),), (_program("first", 2),)} <= promoted
    assert result.selections["first"].source == "tuned"
    assert result.selections["second"].source == "tuned"
    assert result.selections["first"].config.width == 2
    # Selections reach the file in one write per job, and persist.
    assert len(writes) == 1
    reloaded = SelectionCache(tmp_path, engine._cache.identity)
    assert len(reloaded.records) == 2
