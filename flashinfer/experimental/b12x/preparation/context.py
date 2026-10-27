"""Bounded contextual races over already prepared plan combinations.

The caller captures each complete computation before this race and supplies
its graph replay as a PreparedCall. The race never changes a member Plan's
selection or retains a trial graph after returning. Member plans stay owned
by their PreparationSession until the caller tears down trial graphs and
releases losing plans explicitly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .types import Plan, PreparedCall, Selection, call_scope


@dataclass(frozen=True, kw_only=True)
class ContextCandidate:
    """One legal, already prepared plan tuple and its captured computation."""

    name: str
    plans: tuple[Plan, ...]
    call: PreparedCall

    def __post_init__(self):
        plans = tuple(self.plans)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("context candidates require a nonempty name")
        if not plans or any(not isinstance(plan, Plan) for plan in plans):
            raise TypeError("context candidates require prepared scalar Plans")
        if len(set(plans)) != len(plans):
            raise ValueError("context candidates cannot repeat a member Plan")
        if not isinstance(self.call, PreparedCall) or self.call.produce is None:
            raise TypeError(
                "context candidates require an activation-producing PreparedCall"
            )
        object.__setattr__(self, "plans", plans)


@dataclass(frozen=True, kw_only=True)
class ContextRaceResult:
    """Contextual winner metadata, separate from standalone plan selections."""

    winner: str
    latencies_us: Mapping[str, float]
    selections: Mapping[str, tuple[Selection, ...]]
    cache_policy: str

    def __post_init__(self):
        object.__setattr__(
            self, "latencies_us", MappingProxyType(dict(self.latencies_us))
        )
        object.__setattr__(
            self,
            "selections",
            MappingProxyType(
                {name: tuple(values) for name, values in self.selections.items()}
            ),
        )


def _catalog(candidates):
    return tuple(
        {
            "name": candidate.name,
            "plans": tuple(
                {
                    "component": plan.component_id,
                    "query": plan.selection.query.to_dict(),
                    "config": plan.contract.config_payload(
                        plan.selection.config
                    ).to_dict(),
                    "semantic_version": plan.contract.semantic_version,
                }
                for plan in candidate.plans
            ),
        }
        for candidate in candidates
    )


def _peers(exchange, step, payload, expected_count):
    peers = tuple(exchange(step, payload))
    if len(peers) != expected_count or any(
        not isinstance(peer, Mapping) for peer in peers
    ):
        raise ValueError("context collective must return every rank's payload")
    return peers


def race_contexts(
    session,
    candidates: Sequence[ContextCandidate],
    *,
    collective_exchange: Callable[[str, Mapping], Sequence[Mapping]] | None = None,
    collective_ranks: tuple[int, ...] | None = None,
    max_candidates: int = 16,
    rounds: int | None = None,
    samples: int | None = None,
) -> ContextRaceResult:
    """Measure captured complete computations with the preparation sampler.

    ``collective_exchange(step, payload)`` must return all ranks' payloads in
    fixed rank order. Catalog and launch exchanges enforce identical choices;
    sample exchanges use the maximum latency of every matching rank sample.
    Calls restore valid inputs through ``reset`` and ``produce`` before each
    timed replay. Cold-cache eviction follows those producers and precedes the
    gated graph event interval. Trial graph ``close`` callbacks run on exit.
    """
    from ._measurement import (
        _consume,
        _l2_flush_fn,
        measure_race_steps,
        no_compilation,
        prepare_race_steps,
    )
    from .types import _close_all, _prime

    session._check_thread()
    if (
        session.state in ("FROZEN", "CLOSED")
        or session._job is not None
        or session._pool is not None
    ):
        raise RuntimeError(
            "context races require a drained, unfrozen preparation session"
        )
    candidates = tuple(candidates)
    if (
        type(max_candidates) is not int
        or max_candidates <= 0
        or not candidates
        or len(candidates) > max_candidates
    ):
        raise ValueError("context race exceeds its positive candidate bound")
    names = tuple(candidate.name for candidate in candidates)
    if len(set(names)) != len(names):
        raise ValueError("context candidate names must be unique")
    original_plans = candidates[0].plans
    if any(
        len(candidate.plans) != len(original_plans)
        or any(
            not original.same_declaration_except_override(alternative)
            for original, alternative in zip(
                original_plans, candidate.plans, strict=True
            )
        )
        for candidate in candidates[1:]
    ):
        raise ValueError("context candidates must differ only in member overrides")
    if (collective_exchange is None) != (collective_ranks is None):
        raise ValueError("context collective requires both exchange and rank catalog")
    if collective_ranks is not None:
        ranks = tuple(collective_ranks)
        if (
            not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
            or ranks != tuple(sorted(set(ranks)))
        ):
            raise ValueError("context collective ranks must be sorted and unique")
        expected_count = len(ranks)
    else:
        expected_count = 0
    rounds = session.rounds if rounds is None else rounds
    samples = session.samples if samples is None else samples
    if (
        type(rounds) is not int
        or rounds <= 0
        or type(samples) is not int
        or samples <= 0
    ):
        raise ValueError("context race rounds and samples must be positive")
    for candidate in candidates:
        for plan in candidate.plans:
            if (
                plan not in session._plans
                or plan.prepared is None
                or plan.prepared.closed
            ):
                raise ValueError(
                    "context member Plan is not owned by this live session"
                )
            if plan.prepared.device.ordinal != session.device.ordinal:
                raise ValueError("context member Plan belongs to another device")
    catalog = _catalog(candidates)
    if collective_exchange is not None:
        peers = _peers(
            collective_exchange, "catalog", {"catalog": catalog}, expected_count
        )
        if any(peer.get("catalog") != catalog for peer in peers):
            raise ValueError("context ranks declared different candidate catalogs")

    def launch_exchange(indices, turn, repetition):
        peers = _peers(
            collective_exchange,
            "launch",
            {
                "indices": indices,
                "turn": turn,
                "repetition": repetition,
            },
            expected_count,
        )
        if any(
            tuple(peer.get("indices", ())) != indices
            or peer.get("turn") != turn
            or peer.get("repetition") != repetition
            for peer in peers
        ):
            raise ValueError("context ranks launched candidates in different orders")
        if False:
            yield

    def sample_exchange(indices, turn, repetition, groups):
        peers = _peers(
            collective_exchange,
            "samples",
            {
                "indices": indices,
                "turn": turn,
                "repetition": repetition,
                "groups": groups,
            },
            expected_count,
        )
        result = []
        for position, values in enumerate(groups):
            rows = []
            for peer in peers:
                if (
                    tuple(peer.get("indices", ())) != indices
                    or peer.get("turn") != turn
                    or peer.get("repetition") != repetition
                ):
                    raise ValueError("context ranks sampled different candidates")
                peer_groups = peer.get("groups", ())
                if len(peer_groups) != len(groups) or len(peer_groups[position]) != len(
                    values
                ):
                    raise ValueError("context ranks sampled different replay counts")
                row = peer_groups[position]
                if any(not math.isfinite(value) or value <= 0 for value in row):
                    raise ValueError("context rank returned an invalid latency")
                rows.append(row)
            result.append(
                tuple(max(row[index] for row in rows) for index in range(len(values)))
            )
        if False:
            yield
        return tuple(result)

    race = None
    try:
        with session._gpu_scope(), no_compilation():
            for index, candidate in enumerate(candidates):
                _prime(candidate.call)
                if candidate.call.validate is not None:
                    candidate.call.validate()
                if collective_exchange is not None:
                    peers = _peers(
                        collective_exchange,
                        "validated",
                        {"index": index},
                        expected_count,
                    )
                    if any(peer.get("index") != index for peer in peers):
                        raise ValueError(
                            "context ranks validated candidates in different orders"
                        )
            session._synchronize()
            eviction = _l2_flush_fn(
                session.device.ordinal, enabled=session.cache_policy == "cold"
            )
            race = _consume(
                prepare_race_steps(
                    tuple(candidate.call for candidate in candidates),
                    device_ordinal=session.device.ordinal,
                    samples=samples,
                    primed=True,
                    eviction=eviction,
                    cache_policy=session.cache_policy,
                    eviction_after_produce=True,
                )
            )
            measured = _consume(
                measure_race_steps(
                    race,
                    device_ordinal=session.device.ordinal,
                    rounds=rounds,
                    eliminate=False,
                    adaptive_repeats=False,
                    launch_exchange=launch_exchange if collective_exchange else None,
                    sample_exchange=sample_exchange if collective_exchange else None,
                )
            )
        if measured.overlapped_samples:
            raise RuntimeError("context race overlapped compilation")
        latencies = dict(zip(names, measured.latencies_us, strict=True))
        winner = min(range(len(names)), key=lambda i: (measured.latencies_us[i], i))
        return ContextRaceResult(
            winner=names[winner],
            latencies_us=latencies,
            selections={
                candidate.name: tuple(plan.selection for plan in candidate.plans)
                for candidate in candidates
            },
            cache_policy=session.cache_policy,
        )
    finally:
        closers = []
        restorers = []
        resource_closers = []
        if race is not None:
            closers.append(race.close)
        for candidate in candidates:
            if candidate.call.restore is not None:
                restorers.append(lambda call=candidate.call: _restore(call))
            if candidate.call.close is not None:
                resource_closers.append(candidate.call.close)
        _close_all((*closers, *restorers, session._synchronize, *resource_closers))


def _restore(call):
    with call_scope():
        call.restore()
