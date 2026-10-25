"""Program discovery preserves identities and shares bounded compiler admission."""

import pickle
from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x._lib import compile_pool
from b12x._lib.compile_plan import ProgramKey
from b12x._lib.compile_pool import CompileJob, CompilePool

from .test_compile_admission import Activity
from .test_compile_pool_fake_indexing import _run_offline_worker
from .test_defaults import contract
from .test_session import request, session


def finish(iterator):
    while True:
        try:
            next(iterator)
        except StopIteration as done:
            return done.value


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("complete", [False, True])
def test_discovery_pipeline_bounds_open_locks(tmp_path, monkeypatch, workers, complete):
    resource = pytest.importorskip("resource")
    from b12x._lib.compile_pool import CompilationPlan
    from b12x.preparation import DetectedDevice
    from b12x.preparation import session as session_module
    from .test_defaults import Query

    pending = True

    class Discovery:
        def describe(self, jobs):
            return tuple(jobs)

        def descriptions(self, jobs):
            return None if pending else tuple(CompilationPlan(job, ()) for job in jobs)

    monkeypatch.setattr(
        session_module, "discovery_key", lambda rows, target: str(rows[0][2][0].args[0])
    )
    items = []
    for index in range(256):
        item = request(name=str(index), tuning=contract())
        items.append(
            replace(
                item,
                plan=replace(
                    item.plan,
                    query=Query(index + 1),
                    _compile_jobs=lambda config, device, index=index: (
                        CompileJob.create("test.factory:compile", index, config.width),
                    ),
                ),
            )
        )

    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    with session(tmp_path, compile_workers=workers) as engine:
        job = engine.begin(items)
        job._pipeline = finish(job._configure(session_module._coalesce_requests(items)))
        job._selections = {}
        cache = engine._race_progress_cache()
        engine.device = DetectedDevice(0, None)
        monkeypatch.setattr(engine, "_compiler", lambda: Discovery())
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(limits[0], 128), limits[1]))
            finish(job._pump_pipeline(None))
            assert len(job._discovery_locks) == workers
            # Required discovery progresses even when speculative discovery is full.
            current = job._pipeline[-1]
            steps = job._describe_candidates(current)
            assert next(steps) == "compile"
            assert len(job._discovery_locks) == workers + 1
            pending = False
            finish(steps)
            if complete:
                while job._pipeline:
                    finish(job._pump_pipeline(None))
                assert all(
                    cache.load_discovery(str(index)) is not None for index in range(256)
                )
                assert not job._discovery_locks
            else:
                assert cache.load_discovery("0") is None
            job.close()
            for index in range(256):
                with cache.discovery_lock(str(index)).acquire(timeout=0):
                    pass
        finally:
            resource.setrlimit(resource.RLIMIT_NOFILE, limits)
            engine.device = DetectedDevice(None, None)


@pytest.mark.parametrize("cancel", [False, True])
def test_session_counts_before_discovery_and_can_cancel_pending_metadata(
    tmp_path, monkeypatch, cancel
):
    from b12x.preparation.device import DetectedDevice
    from b12x.preparation.session import _coalesce_requests
    from b12x._lib.compile_pool import CompilationPlan

    item = request(name="first", tuning=contract())
    item = replace(
        item,
        plan=replace(
            item.plan,
            _compile_jobs=lambda config, device: (
                CompileJob.create("test.factory:compile", config.width),
            ),
        ),
    )
    calls = []
    pending = True

    class Discovery:
        def describe(self, jobs):
            calls.append(tuple(jobs))
            return tuple(jobs)

        def descriptions(self, jobs):
            assert not engine._stop.is_set(), "cancelled discovery was polled"
            if pending:
                return None
            return tuple(
                CompilationPlan(job, (ProgramKey("cute", str(job.args[0])),))
                for job in jobs
            )

    with session(tmp_path, compile_workers=2) as engine:
        job = engine.begin((item,))
        obligations = finish(job._configure(_coalesce_requests((item,))))
        assert calls == []
        assert job._planned_requests == 1
        assert job._total_candidates == 3
        engine.device = DetectedDevice(0, None)
        monkeypatch.setattr(engine, "_compiler", lambda: Discovery())
        steps = job._describe_candidates(obligations[0])
        assert next(steps) == "compile"
        assert len(calls) == 1 and len(calls[0]) == 3
        assert job._progress(False, True, (), False).phase == "planning programs"
        if cancel:
            engine.cancel_tuning()
        pending = False
        finish(steps)
        if cancel:
            assert obligations[0].candidate_program_identity is None
        else:
            assert job._candidates_described == 3
            assert obligations[0].candidate_program_identity is not None
            duplicate = replace(
                obligations[0],
                compiled={},
                programs={},
                candidate_program_identity=None,
            )
            finish(job._describe_candidates(duplicate))
            assert len(calls) == 1
            assert (
                duplicate.candidate_program_identity
                == obligations[0].candidate_program_identity
            )
        engine.device = DetectedDevice(None, None)


class Workers:
    def __init__(self):
        self._pool = (SimpleNamespace(pid=1, exitcode=None),)
        self.calls = []

    def apply_async(self, function, args, callback, error_callback):
        self.calls.append((function, args, callback, error_callback))

    def terminate(self):
        pass

    def join(self):
        pass


def pool(monkeypatch, workers):
    context = SimpleNamespace(
        Array=lambda kind, values: Activity(values), Pool=lambda **kwargs: workers
    )
    monkeypatch.setattr(
        compile_pool.multiprocessing, "get_context", lambda kind: context
    )
    return CompilePool(
        device_ordinal=0,
        compute_capability=(12, 0),
        device_uuid="test",
        product_name="test",
        sm_count=1,
        max_shared_memory_per_block=1024,
        max_shared_memory_per_multiprocessor=2048,
        workers=2,
    )


def test_discovery_deduplicates_jobs_and_preserves_submission_order(monkeypatch):
    workers = Workers()
    engine = pool(monkeypatch, workers)
    jobs = tuple(CompileJob.create("test.factory:compile", i) for i in range(80))
    keys = engine.describe((*jobs, jobs[0]))
    assert len(workers.calls) == 2
    assert engine.descriptions(keys) is None
    assert engine.pending
    assert engine.describe(jobs) == keys[:-1]
    assert len(workers.calls) == 2
    # Complete the second batch first; returned plans still follow the input.
    for index in (1, 0, 2):
        function, args, complete, _ = workers.calls[index]
        assert function is compile_pool._describe_jobs
        complete(
            tuple(
                (payload, (ProgramKey("cute", str(pickle.loads(payload).args[0])),))
                for payload in args[0]
            )
        )
    plans = engine.descriptions(keys)
    assert tuple(plan.job for plan in plans) == (*jobs, jobs[0])
    assert plans[0] is plans[-1]
    assert len(workers.calls) == 3
    assert not engine.pending
    engine.close(terminate=True)


def test_discovery_failure_and_worker_exit_are_reported(monkeypatch):
    workers = Workers()
    engine = pool(monkeypatch, workers)
    keys = engine.describe((CompileJob.create("test.factory:compile"),))
    workers.calls[0][3](RuntimeError("metadata failed"))
    with pytest.raises(RuntimeError, match="metadata failed"):
        engine.descriptions(keys)
    engine.close(terminate=True)
    workers = Workers()
    engine = pool(monkeypatch, workers)
    keys = engine.describe((CompileJob.create("test.factory:compile"),))
    workers._pool[0].exitcode = 1
    with pytest.raises(RuntimeError, match="worker exited"):
        engine.descriptions(keys)
    engine.close(terminate=True)


def test_real_mhc_discovery_matches_serial_without_cuda_or_compilation():
    _run_offline_worker(
        """
        from b12x.norm import mhc
        from b12x.norm.mhc import _preparation
        from b12x.preparation import autotune_effort
        from b12x.preparation.device import detect_device
        from b12x._lib.compile_plan import evict_planning_artifacts
        from b12x._lib.compile_pool import CompilePool, describe_compilation
        from b12x._lib.program_cache import PreparationProgramCache, evict_unretained

        initialize_worker()
        device = detect_device("cuda:0")
        jobs = []
        for operation in ("pre", "post_pre"):
            with autotune_effort("audit"):
                plan = mhc.plan(mhc.Caps(device="cuda:0", max_tokens=3584, hidden_size=5120),
                    invocation=dict(operation=operation, has_norm_weight=True, lagged_mix=True,
                        expanded_residual=operation == "pre", rms_eps=1e-20, hc_eps=1e-6,
                        norm_eps=1e-20, sinkhorn_iters=20))
            choices = list(plan.contract.choices(plan.query, device.identity))
            selected = choices[:5] + choices[::max(1, len(choices) // 48)]
            for _, config in selected:
                jobs.extend(plan._compile_jobs(config, device))
        expected = []
        # Each fresh scope constructs its own FakeTensorMode and operands.
        for job in jobs:
            scope = PreparationProgramCache()
            with scope.activate():
                expected.append(describe_compilation(job))
            scope.clear()
        evict_planning_artifacts()
        # A shared scope must preserve each exact identity while reusing operands.
        scope = PreparationProgramCache()
        with scope.activate():
            shared = [describe_compilation(job) for job in jobs]
            assert scope._caches[_preparation._fake_operands]._misses == 2
            evict_unretained(p for description in shared for p in description.programs)
        assert shared == expected
        scope.clear()
        evict_planning_artifacts()
        with CompilePool(device_ordinal=0, compute_capability=(12, 1),
                         device_uuid="synthetic", product_name="GB10", sm_count=48,
                         max_shared_memory_per_block=232448,
                         max_shared_memory_per_multiprocessor=232448, workers=2) as engine:
            assert list(engine.plan(jobs)) == expected
            assert not engine.pending
            summary = engine.summary()
            assert summary.cute_compilations == summary.triton_compilations == 0
        assert not torch.cuda.is_initialized()
        """
    )


def test_worker_can_compile_after_discovery_without_deferred_cache_entries():
    _run_offline_worker(
        """
        from tempfile import TemporaryDirectory
        import os
        from b12x.norm import mhc
        from b12x.preparation.device import detect_device
        from b12x._lib.compile_plan import compiled_program_available
        from b12x._lib.compile_pool import CompilePool

        initialize_worker()
        device = detect_device("cuda:0")
        plan = mhc.plan(mhc.Caps(device="cuda:0", max_tokens=1, hidden_size=5120),
                        invocation=dict(operation="collapse"))
        config = plan.contract.default_config(plan.query, device.identity)
        jobs = tuple(plan._compile_jobs(config, device))
        with TemporaryDirectory() as cache:
            os.environ["B12X_COMPILE_CACHE_DIR"] = cache
            with CompilePool(device_ordinal=0, compute_capability=(12, 1),
                             device_uuid="synthetic", product_name="GB10", sm_count=48,
                             max_shared_memory_per_block=232448,
                             max_shared_memory_per_multiprocessor=232448, workers=1) as engine:
                descriptions = engine.plan(jobs)
                assert all(not compiled_program_available(p)
                           for description in descriptions for p in description.programs)
                programs = engine.submit_plans(descriptions)
                engine.wait(programs)
                assert all(compiled_program_available(p) for p in programs)
                assert engine.summary().cute_compilations > 0
        assert not torch.cuda.is_initialized()
        """
    )
