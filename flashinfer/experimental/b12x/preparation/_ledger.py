"""Per-race tuning records for retrospective analysis.

Every completed race appends one record to the ``ledger_v1`` table of the
race-progress database that sits next to the selection cache. A record holds
the component, encoded query, invocation, pin and dependencies; the effort and
coverage; the measurement protocol; every candidate with its assignment,
lowered configuration, program keys, the lowest autotune effort that admits it,
and its per-round timings; the batches that compared candidates; the rejected
candidates; and the wall seconds of each phase. Records are append-only, so
repeated tunings of one query accumulate as replications of the same race.

``b12x.tools.tuning_ledger`` reads the records and reports default regret,
pruning regret per effort level, winner margins, round-to-round noise, and
replication agreement.
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import replace

from ._efficiency import AutotuneEffort
from ._measurement import ELIMINATION_MARGIN, ROUND_BUDGET_US, SURVIVOR_ROUNDS
from .tuning import _identity

logger = logging.getLogger("b12x.preparation")

LEDGER_VERSION = 1
# Legal counts per effort enumerate the whole product once per effort; larger
# products record admission levels only.
LEGAL_COUNT_CARTESIAN_LIMIT = 20_000
EFFORT_LEVELS = (
    AutotuneEffort.LOW,
    AutotuneEffort.MEDIUM,
    AutotuneEffort.HIGH,
    AutotuneEffort.MAX,
)


def admission_levels(space, candidates, *, cartesian_count=None):
    """Lowest effort admitting each candidate, and the legal count per effort.

    Admission evaluates the space's predicates at each effort for the
    candidate's own assignment. Legal counts are assignments passing every
    predicate active at that effort, before lowering and equivalence
    deduplication; they are ``None`` when the cartesian product exceeds
    ``LEGAL_COUNT_CARTESIAN_LIMIT``.
    """
    variants = {
        effort: replace(
            space, autotune_effort=effort, exhaustive=effort is AutotuneEffort.MAX
        )
        for effort in EFFORT_LEVELS
    }
    levels = []
    for assignment, _ in candidates:
        level = None
        for effort in EFFORT_LEVELS:
            if variants[effort]._passes_predicates(assignment):
                level = effort.value
                break
        levels.append(level)
    counts = None
    if cartesian_count is None or cartesian_count <= LEGAL_COUNT_CARTESIAN_LIMIT:
        counts = {
            effort.value: sum(1 for _ in variants[effort].configurations())
            for effort in EFFORT_LEVELS
        }
    return levels, counts


class RaceLedger:
    """Collect one race's candidates, batches, phase times, and timing samples."""

    def __init__(
        self,
        *,
        job,
        obligation,
        kind,
        shard_index,
        shard_count,
        local_indices,
        restored,
    ):
        self.job = job
        self.obligation = obligation
        self.kind = kind
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.local_indices = list(local_indices)
        self.restored = dict(restored)
        self.started_wall = time.time()
        self.started = time.perf_counter()
        self.phase_seconds = defaultdict(float)
        self.trials = {}
        self.precompiled = {}
        self.rejected = []
        self.batches = []
        self._rounds = defaultdict(dict)

    def add_phase(self, name, seconds):
        self.phase_seconds[name] += seconds

    def note_precompiled(self, index, programs):
        """Record whether every program of a candidate existed before its compile."""
        self.precompiled[index] = self.job._programs_available(programs)

    def prepared(self, trial, *, seconds, batch):
        self.add_phase("prepare", seconds)
        self.trials[trial.index] = {"batch": batch, "prepare_seconds": seconds}

    def reject(self, index, config, reason):
        payload = self.job._payload(self.obligation, config)
        self.rejected.append(
            {"index": index, "config": payload.to_dict(), "reason": reason}
        )

    def observer(self, trials):
        """Return a ``measure_race_steps`` sample observer keyed by candidate index."""
        indices = [trial.index for trial in trials]
        rounds = self._rounds

        def observe(position, turn, repetition, samples):
            cell = rounds[indices[position]].setdefault(turn, [0, 0.0, 0.0, 0])
            cell[0] += len(samples)
            cell[1] += math.fsum(samples)
            cell[2] += math.fsum(value * value for value in samples)
            cell[3] += 1

        return observe

    def batch(self, number, trials, measurement, *, champion, winner, seconds):
        self.add_phase("measure", seconds)
        for trial in trials:
            self.trials.setdefault(
                trial.index, {"batch": number, "prepare_seconds": None}
            )
        self.batches.append(
            {
                "batch": number,
                "candidates": [trial.index for trial in trials],
                "carried_champion": None if champion is None else champion.index,
                "latencies_us": [float(value) for value in measurement.latencies_us],
                "winner": winner,
                "overlapped_samples": measurement.overlapped_samples,
                "seconds": seconds,
            }
        )

    def _round_payload(self, index):
        payload = []
        for turn in sorted(self._rounds.get(index, ())):
            count, total, squares, repetitions = self._rounds[index][turn]
            mean = total / count
            variance = max(squares / count - mean * mean, 0.0)
            payload.append(
                {
                    "round": turn,
                    "n": count,
                    "repetitions": repetitions,
                    "mean_us": mean,
                    "stdev_us": math.sqrt(variance),
                }
            )
        return payload

    def commit(self, cache_factory, *, measurements, champion_index, rejected_count):
        """Append this race's record. A ledger failure never fails the race."""
        try:
            record = self.record(
                measurements=measurements,
                champion_index=champion_index,
                rejected_count=rejected_count,
            )
            cache_factory().append_ledger(
                self.obligation.key,
                shard_index=self.shard_index,
                shard_count=self.shard_count,
                session=self.job.session.session_id,
                record=record,
            )
        except Exception:
            logger.warning(
                "Tuning ledger record for %s was not written",
                self.obligation.request.name,
                exc_info=True,
            )

    def record(self, *, measurements, champion_index, rejected_count):
        job, obligation = self.job, self.obligation
        session, request = job.session, obligation.request
        plan, configuration = request.plan, obligation.configuration
        contract, space = plan.contract, configuration.space
        coverage = dict(obligation.coverage)
        levels, counts = admission_levels(
            space,
            obligation.candidates,
            cartesian_count=coverage.get("cartesian_count"),
        )
        default_payload = contract.config_payload(configuration.default)
        default_index = None
        local = set(self.local_indices)
        payloads = [
            self.job._payload(obligation, config) for _, config in obligation.candidates
        ]
        candidates = []
        for index, (assignment, _) in enumerate(obligation.candidates):
            payload = payloads[index]
            if default_index is None and payload == default_payload:
                default_index = index
            trial = self.trials.get(index, {})
            latency = measurements.get(index)
            # Other shards' candidates carry their identity only; the shard that
            # measured them records programs and rounds.
            candidates.append(
                {
                    "index": index,
                    "assignment": assignment.to_dict(),
                    "config": payload.to_dict(),
                    "programs": sorted(
                        [program.dialect, program.key]
                        for program in obligation.programs.get(payload, ())
                    )
                    if index in local
                    else [],
                    "admission": None if levels is None else levels[index],
                    "local": index in local,
                    "restored": index in self.restored,
                    "precompiled": self.precompiled.get(index),
                    "batch": trial.get("batch"),
                    "prepare_seconds": trial.get("prepare_seconds"),
                    "latency_us": None if latency is None else float(latency),
                    "rounds": self._round_payload(index),
                }
            )
        champion_latency = measurements.get(champion_index)
        return {
            "ledger_version": LEDGER_VERSION,
            "kind": self.kind,
            "key": obligation.key,
            "request": request.name,
            "component_id": plan.component_id,
            "query": configuration.encoded_query.to_dict(),
            "invocation": plan.invocation.to_dict(),
            "pin": None
            if configuration.pinned is None
            else contract.config_payload(configuration.pinned).to_dict(),
            "dependencies": list(request.dependencies),
            "effort": "max" if space.exhaustive else space.autotune_effort.value,
            "protocol": {
                "rounds": session.rounds,
                "samples": session.samples,
                "race_batch": session.race_batch,
                "cache_policy": session.cache_policy,
                "elimination_margin": ELIMINATION_MARGIN,
                "survivor_rounds": SURVIVOR_ROUNDS,
                "round_budget_us": ROUND_BUDGET_US,
            },
            "shard": {
                "index": self.shard_index,
                "count": self.shard_count,
                "rank": session._tuning_rank,
                "ranks": list(session._tuning_ranks),
                "local_candidates": self.local_indices,
            },
            "coverage": {**coverage, "legal_by_effort": counts},
            "default_index": default_index,
            "champion": {
                "index": champion_index,
                "latency_us": None
                if champion_latency is None
                else float(champion_latency),
            },
            "candidates": candidates,
            "rejected": self.rejected,
            "rejected_count": rejected_count,
            "batches": self.batches,
            "phase_seconds": dict(self.phase_seconds),
            "started_at": self.started_wall,
            "seconds": time.perf_counter() - self.started,
        }
