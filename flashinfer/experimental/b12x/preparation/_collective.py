"""Group-coordinated candidate races using the preparation sampler."""

from __future__ import annotations

import math
import time
from dataclasses import replace

from ._cache import digest
from ._efficiency import AutotuneEffort
from ._ledger import RaceLedger
from ._measurement import capture_benchmark_call, no_compilation
from .types import (
    CollectiveTuningRequirement,
    FinalistRecord,
    FrozenMapping,
    call_scope,
)


class _Cancelled(Exception):
    pass


def family_bucket(obligation):
    request = obligation.request
    contract = request.plan.contract
    callback = contract.collective_family_bucket
    if (
        request.collective is None
        or callback is None
        or request.finalist_policy is not None
    ):
        return None
    space = obligation.configuration.space
    if (
        space is None
        or space.autotune_effort is not AutotuneEffort.DEFAULT
        or obligation.configuration.pinned is not None
        or request.dependencies
        or request.benchmark_call is not request.plan._benchmark_call
    ):
        return None
    bucket = callback(request.plan.query)
    if bucket is None:
        return None
    field, identity, size = bucket
    if len({assignment[field] for assignment, _ in obligation.candidates}) <= 2:
        return None
    return (
        field,
        dict(
            contract=(
                contract.component_id,
                contract.query_schema_version,
                contract.config_schema_version,
                contract.semantic_version,
                contract.candidate_contract_version,
            ),
            invocation=request.plan.invocation.to_dict(),
            bucket=identity,
        ),
        size,
    )


def calibration_order(obligations):
    """Race the largest declared payload in each bucket before its other requests."""
    representatives = {}
    for index, obligation in enumerate(obligations):
        if obligation.ready or obligation.selection is not None:
            continue
        bucket = family_bucket(obligation)
        if bucket is None:
            continue
        field, identity, size = bucket
        key = digest(
            (
                obligation.request.collective.ranks,
                field,
                identity,
            )
        )
        previous = representatives.get(key)
        if previous is None or size > previous[0]:
            representatives[key] = (size, index)
    first = {index for _, index in representatives.values()}
    return [item for index, item in enumerate(obligations) if index in first] + [
        item for index, item in enumerate(obligations) if index not in first
    ]


def exchange(requirement, step, **payload):
    peers = yield CollectiveTuningRequirement(
        requirement.key, requirement.ranks, step, FrozenMapping(payload)
    )
    return tuple(item.payload for item in peers)


def _same(peers, field):
    value = peers[0][field]
    if any(peer[field] != value for peer in peers[1:]):
        raise ValueError(f"collective tuning ranks disagree on {field}")
    return value


def _load(job, obligation, record):
    contract = obligation.request.plan.contract
    configuration = obligation.configuration
    assignment = FrozenMapping(record["assignment"])
    configuration.space.validate(assignment)
    config = contract._lower(configuration.query, configuration.device, assignment)
    if contract.config_payload(config) != record["config"]:
        raise ValueError("collective cached assignment differs from its configuration")
    obligation.finalists = job._validate_finalists(
        obligation, record.get("finalists", ())
    )
    obligation.coverage = dict(record["coverage"])
    job._cache_hits += 1
    return job._selection(obligation, config, "cached", assignment)


def select(job, obligation, selections):
    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    if contract.collective_signature is None:
        raise ValueError(
            "collective tuning requires a group-consistent query signature"
        )
    bucket = family_bucket(obligation)
    candidates = tuple(
        (assignment.to_dict(), contract.config_payload(config).to_dict())
        for assignment, config in obligation.candidates
    )
    policy = obligation.request.finalist_policy
    policy_payload = (
        None
        if policy is None
        else (policy.family_fields, policy.per_family, policy.max_total)
    )
    peers = yield from exchange(
        requirement,
        "catalog",
        finalist_policy=policy_payload,
        candidates=candidates,
        signature=contract.collective_signature(obligation.request.plan.query),
        local_key=job._choice_key(obligation, selections),
        device=job.session.device.uuid,
        cache_only=job.session.cache_only,
        rounds=job.session.rounds,
        samples=job.session.samples,
        cancel=job.session._stop.is_set(),
        bucket=bucket,
    )
    for field in (
        "signature",
        "candidates",
        "cache_only",
        "rounds",
        "samples",
        "finalist_policy",
        "bucket",
    ):
        _same(peers, field)
    if any(peer["cancel"] for peer in peers):
        return job._selection(obligation, obligation.configuration.default, "default")
    bucket_key = None
    calibration = None
    if bucket is not None:
        family_field, identity, _ = bucket
        bucket_key = digest(
            (
                identity,
                family_field,
                tuple(peer["device"] for peer in peers),
                requirement.ranks,
            )
        )
        calibration = job.session._collective_family_buckets.get(bucket_key)
    cache_identity = {
        "measurement": (
            "collective_max_stream_gated_graph_v5_finalists"
            if policy is not None
            else "collective_max_stream_gated_graph_v4_token_scaled"
        ),
        "repetitions": {"reduction_token_target": 100, "default": 1},
        "rounds": job.session.rounds,
        "samples": job.session.samples,
        "members": tuple((peer["device"], peer["local_key"]) for peer in peers),
    }
    if bucket is not None:
        cache_identity["family_bucket"] = (1, bucket)
        cache_identity["family_calibration"] = None
    obligation.key = digest(cache_identity)
    if calibration is not None and obligation.key != calibration[0]:
        obligation.key = digest({**cache_identity, "family_calibration": calibration})
    cache = job.session._selection_cache()
    peers = yield from exchange(
        requirement, "cache", record=cache.get(obligation.key), calibration=calibration
    )
    _same(peers, "calibration")
    record = next(
        (peer["record"] for peer in peers if peer["record"] is not None), None
    )
    if record is not None and (policy is None or record.get("finalists")):
        selection = _load(job, obligation, record)
        if (
            calibration is not None
            and selection.assignment[family_field] not in calibration[1]
        ):
            raise ValueError(
                "cached collective selection is outside its calibrated families"
            )
        if bucket_key is not None and calibration is None:
            job.session._collective_family_buckets[bucket_key] = (
                obligation.key,
                tuple(obligation.coverage["bucket_families"]),
            )
        return selection
    if job.session.cache_only:
        raise LookupError(
            f"no completed collective selection for {obligation.request.name}"
        )
    if obligation.request.benchmark_call is None:
        raise ValueError("collective tuning requires a representative benchmark call")
    if calibration is not None:
        _, families = calibration
        before = len(obligation.candidates)
        obligation.candidates = [
            (assignment, config)
            for assignment, config in obligation.candidates
            if assignment[family_field] in families
        ]
        if not obligation.candidates:
            raise ValueError(
                "collective bucket families have no eligible configurations"
            )
        obligation.coverage["bucket_pruned_count"] = before - len(obligation.candidates)
        obligation.coverage["bucket_families"] = families
    try:
        selection = yield from race(
            job,
            obligation,
            family_field=bucket[0]
            if bucket is not None and calibration is None
            else None,
        )
        if bucket_key is not None and calibration is None:
            job.session._collective_family_buckets[bucket_key] = (
                obligation.key,
                tuple(obligation.coverage["bucket_families"]),
            )
        return selection
    except _Cancelled:
        return job._selection(obligation, obligation.configuration.default, "default")


def race(job, obligation, *, family_field=None):
    from .session import _CallGuard, _Trial

    requirement = obligation.request.collective
    contract = obligation.request.plan.contract
    champion = None
    live = []
    measurements = {}
    policy = obligation.request.finalist_policy

    def family_of(trial):
        return FrozenMapping(
            {name: trial.assignment[name] for name in policy.family_fields}
        )

    job._candidate_sharded = False
    job._candidate_count = len(obligation.candidates)
    ledger = RaceLedger(
        job=job,
        obligation=obligation,
        kind="collective",
        shard_index=0,
        shard_count=1,
        local_indices=range(len(obligation.candidates)),
        restored={},
    )

    def sample_exchange(indices, turn, repetition, groups):
        peers = yield from exchange(
            requirement,
            f"samples/{job._batch_index}/{turn}/{repetition}",
            indices=indices,
            samples=groups,
            cancel=job.session._stop.is_set(),
        )
        _same(peers, "indices")
        if any(peer["cancel"] for peer in peers):
            raise _Cancelled
        if any(
            len(peer["samples"]) != len(groups)
            or any(
                len(values) != len(expected)
                for values, expected in zip(peer["samples"], groups, strict=True)
            )
            for peer in peers
        ):
            raise ValueError("collective ranks measured different sample counts")
        if any(
            not math.isfinite(value) or value <= 0
            for peer in peers
            for values in peer["samples"]
            for value in values
        ):
            raise ValueError("collective rank produced an invalid latency")
        result = tuple(
            tuple(
                max(peer["samples"][i][j] for peer in peers) for j in range(len(values))
            )
            for i, values in enumerate(groups)
        )
        if any(
            not math.isfinite(value) or value <= 0 for row in result for value in row
        ):
            raise ValueError("collective race produced an invalid latency")
        return result

    def launch_exchange(indices, turn, repetition):
        peers = yield from exchange(
            requirement,
            f"queued/{job._batch_index}/{turn}/{repetition}",
            indices=indices,
        )
        _same(peers, "indices")

    def prepare_trial(index, assignment, config):
        job._phase = "preparing candidates"
        ledger.note_precompiled(
            index,
            obligation.programs.get(contract.config_payload(config), frozenset()),
        )
        programs = yield from job._compile(obligation, config, required=True)
        yield from job._wait_programs(programs, obligation)
        compile_assignment = obligation.configuration.space.compile_assignment(
            assignment
        )
        previous = obligation.compile_assignments.setdefault(
            compile_assignment, programs
        )
        if previous != programs:
            raise ValueError("collective runtime parameter changed actual compile keys")
        selected = job._selection(obligation, config, "tuned", assignment)
        state, call, guard, retained = job._call(
            obligation,
            selected,
            obligation.request.benchmark_call,
            synchronize=False,
            prime=False,
        )
        trial = _Trial(index, assignment, config, call, guard, retained, state)
        live.append(trial)
        job._temporaries.remove(guard.finish)
        peers = yield from exchange(
            requirement,
            f"bound/{index}",
            cancel=job.session._stop.is_set(),
            producers=len(call.benchmark_producers) or 1,
            has_producer=call.produce is not None,
            repetitions=call.benchmark_repetitions,
        )
        _same(peers, "producers")
        repetitions = _same(peers, "repetitions")
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("benchmark repetitions must be a positive integer")
        if not all(peer["has_producer"] for peer in peers):
            raise ValueError(
                "collective benchmark requires an input producer on every rank"
            )
        if any(peer["cancel"] for peer in peers):
            raise _Cancelled
        with job.session._gpu_scope(), no_compilation():
            # Every rank has bound its operands before collective priming starts.
            for producer in call.benchmark_producers or (call.produce,):
                if producer is None:
                    raise ValueError("collective benchmark requires an input producer")
                with call_scope():
                    if call.reset is not None:
                        call.reset()
                    producer()
                    call.invoke()
                job.session._synchronize()
                if call.validate is not None:
                    call.validate()
            captured = capture_benchmark_call(
                replace(call, restore=guard.restore, close=guard.close, validate=None),
                repetitions=repetitions,
            )
            trial.call = captured
            trial.guard = _CallGuard(captured)
        yield from exchange(requirement, f"validated/{index}")
        return trial

    try:
        for index, (assignment, config) in enumerate(obligation.candidates):
            trial_started = time.perf_counter()
            trial = yield from prepare_trial(index, assignment, config)
            repetitions = trial.call.benchmark_repetitions
            job._candidates_prepared += 1
            job._batch_index += 1
            ledger.prepared(
                trial,
                seconds=time.perf_counter() - trial_started,
                batch=job._batch_index,
            )
            trials = ([] if champion is None else [champion]) + [trial]
            if any(item.call.benchmark_repetitions != repetitions for item in trials):
                raise ValueError(
                    "collective candidates require the same repetition count"
                )
            job._batch_candidates = len(trials)
            measure_started = time.perf_counter()
            measurement = yield from job._measure(
                trials,
                champion=len(trials) == 2,
                adaptive_collective_pair=len(trials) == 2,
                repetitions=repetitions,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
                sample_observer=ledger.observer(trials),
            )
            best = min(
                range(len(trials)),
                key=lambda i: (measurement.latencies_us[i], trials[i].index),
            )
            measurements.update(
                (item.index, latency)
                for item, latency in zip(trials, measurement.latencies_us, strict=True)
            )
            ledger.batch(
                job._batch_index,
                trials,
                measurement,
                champion=champion,
                winner=trials[best].index,
                seconds=time.perf_counter() - measure_started,
            )
            for position, item in enumerate(trials):
                if position != best:
                    item.close()
                    live.remove(item)
            champion = trials[best]
            champion.latency_us = measurement.latencies_us[best]
            job._benchmarked += 1
            job._timing.record(
                "collective_batch_end",
                request=obligation.request.name,
                candidates=[item.index for item in trials],
                latencies_us=measurement.latencies_us,
                winner=champion.index,
            )
        if champion is None:
            raise ValueError("collective race has no eligible candidates")
        if policy is not None:
            nominees = {}
            for index, latency in measurements.items():
                assignment, _ = obligation.candidates[index]
                family = FrozenMapping(
                    {name: assignment[name] for name in policy.family_fields}
                )
                previous = nominees.get(family)
                if previous is None or (latency, index) < previous:
                    nominees[family] = (latency, index)
            nominees[family_of(champion)] = (champion.latency_us, champion.index)
            if len(nominees) > policy.max_total:
                raise ValueError("finalist family coverage exceeds max_total")
            trials = []
            for index in sorted(index for _, index in nominees.values()):
                if index == champion.index:
                    trial = champion
                else:
                    assignment, config = obligation.candidates[index]
                    trial = yield from prepare_trial(index, assignment, config)
                trials.append(trial)
            job._batch_index += 1
            measurement = yield from job._measure(
                trials,
                champion=False,
                eliminate=False,
                adaptive_repeats=False,
                repetitions=trials[0].call.benchmark_repetitions,
                sample_exchange=sample_exchange,
                launch_exchange=launch_exchange,
            )
            obligation.finalists = job._validate_finalists(
                obligation,
                tuple(
                    FinalistRecord(
                        assignment=FrozenMapping(item.assignment),
                        family=family_of(item),
                        candidate_index=item.index,
                        latency_us=float(latency),
                    )
                    for item, latency in zip(
                        trials, measurement.latencies_us, strict=True
                    )
                ),
            )
            obligation.coverage["finalist_count"] = len(obligation.finalists)
        peers = yield from exchange(
            requirement, "selected", assignment=champion.assignment
        )
        _same(peers, "assignment")
        obligation.coverage["measured_count"] = len(measurements)
        if family_field is not None:
            best = {}
            for index, latency in measurements.items():
                family = obligation.candidates[index][0][family_field]
                best[family] = min(best.get(family, math.inf), latency)
            winner = champion.assignment[family_field]
            families = sorted(
                best, key=lambda family: (family != winner, best[family], family)
            )
            obligation.coverage["bucket_families"] = tuple(families[:2])
        obligation.cache_pending = True
        if obligation.key is not None:
            ledger.commit(
                job.session._race_progress_cache,
                measurements=measurements,
                champion_index=champion.index,
                rejected_count=0,
            )
        return job._selection(obligation, champion.config, "tuned", champion.assignment)
    finally:
        job.session._synchronize()
        for trial in reversed(live):
            trial.close()
