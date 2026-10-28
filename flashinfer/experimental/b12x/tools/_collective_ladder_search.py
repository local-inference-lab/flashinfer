"""CPU scheduling and raw timing reports for collective payload ladders."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import statistics
import time

from ._collective_ladder_sampling import (
    MAX_ROUNDS,
    SAMPLES_PER_ROUND,
    measurement_rounds,
)


COMPILED_FIELDS = ("implementation", "threads", "blocks")


def ladder_sizes():
    """Return sixteen aligned logarithmic intervals per octave, including endpoints."""
    return tuple(round(1024 * 2 ** (index / 16) / 16) * 16 for index in range(209))


def _key(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _copy(value):
    return json.loads(_key(value))


def _compiled(config):
    return {field: config[field] for field in COMPILED_FIELDS}


def _record(record):
    value = _copy(record)
    size = value.get("bytes", value.get("size_bytes"))
    if type(size) is not int or size <= 0:
        raise ValueError("measurement requires positive integer bytes")
    if "size_bytes" in value and value["size_bytes"] != size:
        raise ValueError("measurement byte fields disagree")
    if not isinstance(value.get("config"), dict):
        raise ValueError("measurement requires a config dictionary")
    value["bytes"] = size
    samples = value.get("samples_us", [])
    if not isinstance(samples, list) or any(
        type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples
    ):
        raise ValueError("samples_us must contain positive finite timings")
    if not samples and not value.get("error"):
        raise ValueError("measurement requires samples_us or an explicit error")
    return value


def _passed(record):
    return (
        bool(record.get("samples_us"))
        and not record.get("error")
        and record.get("correctness") in (True, "passed")
    )


@dataclass(frozen=True)
class Trial:
    size_bytes: int
    configs: tuple
    rounds: int = 3
    phase: str = "search"
    batch_id: int = 0


class Search:
    """Measure compiled/submode groups at every payload before local refinement.

    Runtime profiles remain finite catalog entries. Neighboring winners change
    measurement order only; recommendations always use measured exact-size data.
    """

    def __init__(
        self,
        catalogs,
        *,
        deadline,
        clock=time.monotonic,
        exhaustive=False,
        batch_size=32,
        rounds=3,
        confirmation_seconds=80,
        finalization_seconds=10,
        records=(),
    ):
        if not catalogs or type(batch_size) is not int or batch_size < 1:
            raise ValueError(
                "search requires nonempty catalogs and a positive batch size"
            )
        if (
            not math.isfinite(deadline)
            or any(
                not math.isfinite(x) or x < 0
                for x in (confirmation_seconds, finalization_seconds)
            )
            or type(rounds) is not int
            or not 1 <= rounds <= MAX_ROUNDS
        ):
            raise ValueError("deadline, rounds and reserved durations are invalid")
        self.deadline, self.clock, self.exhaustive = deadline, clock, exhaustive
        self.batch_size, self.rounds = batch_size, rounds
        self.confirmation_seconds, self.finalization_seconds = (
            confirmation_seconds,
            finalization_seconds,
        )
        self.catalogs, self.configs, self.groups, self.group_of = {}, {}, {}, {}
        for size, catalog in sorted(catalogs.items()):
            if type(size) is not int or size <= 0 or not catalog:
                raise ValueError("every payload requires a nonempty legal catalog")
            choices, groups = {}, {}
            for config in catalog:
                config = _copy(config)
                key = _key(config)
                group = _key(dict(_compiled(config), submode=config.get("submode", "")))
                if key in choices:
                    raise ValueError(
                        "legal catalogs must not contain duplicate configurations"
                    )
                choices[key] = config
                groups.setdefault(group, []).append(key)
                self.configs[size, key] = config
                self.group_of[size, key] = group
            self.catalogs[size], self.groups[size] = choices, groups
        self.sizes = tuple(self.catalogs)
        self.anchors = tuple(size for size in self.sizes if size & (size - 1) == 0)
        self.remaining = {s: dict(catalog) for s, catalog in self.catalogs.items()}
        self.by_size = {s: {} for s in self.sizes}
        self.coverage_left = {
            s: {g: dict.fromkeys(keys) for g, keys in groups.items()}
            for s, groups in self.groups.items()
        }
        self.covered = {s: set() for s in self.sizes}
        self.group_samples = {s: {} for s in self.sizes}
        self.group_best = {s: {} for s in self.sizes}
        self.anchor_group_best = {s: {} for s in self.anchors}
        self._anchor_samples = {}
        self._coverage_started = False
        self.group_points, self._priority_references = {}, {}
        self.records, self.samples, self.latencies = [], {}, {}
        self.final, self.cohorts, self.confirmation_sizes = {}, {}, set()
        self.confirmations = {}
        self._completed_once = set()
        self.batch_payloads = {}
        self.required_profiles = {size: {} for size in self.sizes}
        self.coarse_remaining = {size: {} for size in self.sizes}
        self._seeded = set()
        for size in self.sizes:
            for group in self.groups[size]:
                keys = self._numeric_profiles(size, group)
                self.required_profiles[size][group] = set(keys)
                self.coarse_remaining[size].update(dict.fromkeys(keys))
        self._next_batch = 1
        self.attempted, self.invalid = set(), set()
        self.best, self.pending, self.reason = {}, None, None
        self._confirmation, self._anchor_until = None, None
        self._last_deadline, self._pending_started, self._pending_deadline = (
            deadline,
            None,
            None,
        )
        self._seconds_per_config_round = 0.01
        self.phase_seconds, self.phase_trials = {}, {}
        for record in records:
            self._add(_record(record))
        for cohort in self.cohorts.values():
            size, batch = cohort[0]["bytes"], cohort[0]["batch"]
            closed_batch = max(
                self.final.get(size, {}).get("batch", 0),
                self.confirmations.get(size, [{}])[0].get("batch", 0),
            )
            if (
                cohort[0]["phase"] == "confirmation"
                and len(cohort) < cohort[0]["batch_size"]
                and batch > closed_batch
            ):
                for record in cohort:
                    self._add(record, screening_only=True)
        if self.records:
            for size in self.sizes:
                self._seed_profiles(size)
        failed_sizes = {size for size, _ in self.invalid}
        self._resumed_complete = {
            size
            for size in self.sizes
            if not exhaustive
            and size in self.final
            and len(self.covered[size]) == len(self.groups[size])
            and size not in failed_sizes
            and not self.coarse_remaining[size]
        }
        self._resumed_coverage_only = {
            size
            for size in self._completed_once
            if not exhaustive
            and size not in failed_sizes
            and len(self.covered[size]) == len(self.groups[size])
        }
        self._resumed_ready = {
            size
            for size in self.sizes
            if not exhaustive
            and (size in self._completed_once or size in self.confirmations)
            and len(self.covered[size]) == len(self.groups[size])
            and size not in failed_sizes
            and not self.coarse_remaining[size]
        }

    def _add(self, record, *, screening_only=False):
        size, key = record["bytes"], _key(record["config"])
        pair = size, key
        if pair not in self.configs:
            raise ValueError("measurement is outside the legal search catalog")
        if not screening_only:
            self.records.append(record)
            self._next_batch = max(self._next_batch, int(record.get("batch", 0)) + 1)
        if (
            not screening_only
            and record.get("phase") in ("confirmation", "final")
            and _passed(record)
        ):
            self._confirm(record)
            return
        if record.get("phase") == "coverage":
            self._coverage_started = True
        if record.get("phase") == "anchor" and size in self.anchor_group_best:
            group = self.group_of[pair]
            if _passed(record):
                self._anchor_samples.setdefault(pair, []).extend(record["samples_us"])
            else:
                self._anchor_samples.pop(pair, None)
            measured = [
                k for k in self.groups[size][group] if (size, k) in self._anchor_samples
            ]
            if measured:
                self.anchor_group_best[size][group] = min(
                    measured,
                    key=lambda k: (statistics.median(self._anchor_samples[size, k]), k),
                )
            else:
                self.anchor_group_best[size].pop(group, None)
        self.final.pop(size, None)
        self.confirmation_sizes.discard(size)
        self.confirmations.pop(size, None)
        self.attempted.add(pair)
        self.remaining[size].pop(key, None)
        if _passed(record):
            self.coarse_remaining[size].pop(key, None)
        group = self.group_of[pair]
        if group in self.coverage_left[size]:
            self.coverage_left[size][group].pop(key, None)
            if not self.coverage_left[size][group]:
                del self.coverage_left[size][group]
        if not _passed(record):
            self.invalid.add(pair)
            self.samples.pop(pair, None)
            self.latencies.pop(pair, None)
            self.by_size[size].pop(key, None)
            members = self.group_samples[size].get(group, set())
            members.discard(key)
            if not members:
                self.covered[size].discard(group)
                available = {
                    k: None
                    for k in self.groups[size][group]
                    if k in self.remaining[size]
                }
                if available:
                    self.coverage_left[size][group] = available
            if members:
                best = min(members, key=lambda k: (self.latencies[size, k], k))
                self.group_best[size][group] = best
                self.group_points[group][size] = best
            else:
                self.group_best[size].pop(group, None)
                self.group_points.get(group, {}).pop(size, None)
            if self.best.get(size) == key:
                if self.by_size[size]:
                    self.best[size] = min(
                        self.by_size[size], key=lambda k: (self.latencies[size, k], k)
                    )
                else:
                    self.best.pop(size, None)
        elif pair not in self.invalid:
            previous = self.latencies.get(pair)
            samples = self.samples.setdefault(pair, [])
            samples.extend(record["samples_us"])
            latency = self.latencies[pair] = statistics.median(samples)
            self.by_size[size][key] = samples
            self.group_samples[size].setdefault(group, set()).add(key)
            self.covered[size].add(group)
            self.coverage_left[size].pop(group, None)
            group_key = self.group_best[size].get(group)
            if group_key is None or (latency, key) < (
                self.latencies[size, group_key],
                group_key,
            ):
                self.group_best[size][group] = key
            elif group_key == key and previous is not None and latency > previous:
                self.group_best[size][group] = min(
                    self.group_samples[size][group],
                    key=lambda k: (self.latencies[size, k], k),
                )
            self.group_points.setdefault(group, {})[size] = self.group_best[size][group]
            current = self.best.get(size)
            if current is None or (latency, key) < (
                self.latencies[size, current],
                current,
            ):
                self.best[size] = key
            elif current == key and previous is not None and latency > previous:
                self.best[size] = min(
                    self.by_size[size], key=lambda k: (self.latencies[size, k], k)
                )

    def _confirm(self, record):
        size = record["bytes"]
        batch, count, rounds = (
            record.get(field) for field in ("batch", "batch_size", "rounds")
        )
        if (
            any(type(value) is not int or value < 1 for value in (batch, count, rounds))
            or rounds > MAX_ROUNDS
        ):
            return
        if self.batch_payloads.setdefault(batch, size) != size:
            raise ValueError("confirmation batch spans multiple payloads")
        cohort = self.cohorts.setdefault((size, batch), [])
        if len(cohort) >= count:
            raise ValueError("confirmation batch exceeds its declared size")
        if any(_key(row["config"]) == _key(record["config"]) for row in cohort):
            raise ValueError("duplicate configuration in a confirmation batch")
        if any(
            row["batch_size"] != count
            or row["rounds"] != rounds
            or row["phase"] != record["phase"]
            for row in cohort
        ):
            raise ValueError("inconsistent confirmation batch geometry")
        cohort.append(record)
        if (
            len(cohort) == count
            and len({len(row["samples_us"]) for row in cohort}) != 1
        ):
            raise ValueError("confirmation batch has unbalanced sample counts")
        if len(cohort) == count and all(
            len(row["samples_us"]) == rounds * SAMPLES_PER_ROUND for row in cohort
        ):
            ranked = sorted(
                cohort,
                key=lambda row: (
                    statistics.median(row["samples_us"]),
                    _key(row["config"]),
                ),
            )
            if record["phase"] == "confirmation":
                self.confirmations[size] = ranked
                self.final.pop(size, None)
                self.confirmation_sizes.discard(size)
            else:
                winner = ranked[0]
                self.final[size] = dict(
                    size_bytes=size,
                    config=_copy(winner["config"]),
                    median_us=statistics.median(winner["samples_us"]),
                    batch=batch,
                    rounds=rounds,
                    confirmed=True,
                )
                self.confirmation_sizes.add(size)
                self._completed_once.add(size)

    def _shortlist(self, size):
        ranked = sorted(
            self.by_size[size], key=lambda key: (self.latencies[size, key], key)
        )
        if not ranked:
            return []
        threshold = self.latencies[size, ranked[0]] * 1.10
        ties = {}
        for key in ranked:
            latency = self.latencies[size, key]
            if latency > threshold:
                break
            ties.setdefault(latency, {}).setdefault(
                _key(_compiled(self.configs[size, key])), []
            ).append(key)
        selected = []
        for geometries in ties.values():
            for index in range(max(map(len, geometries.values()))):
                selected.extend(
                    keys[index] for keys in geometries.values() if index < len(keys)
                )
        return selected[:8]

    def _neighbors(self, size):
        below = [s for s in self.best if s < size]
        above = [s for s in self.best if s > size]
        return [
            s
            for s in (max(below, default=None), min(above, default=None))
            if s is not None
        ]

    def _priority(self, size, key):
        group = self.group_of[size, key]
        cache_key = size, group
        if cache_key not in self._priority_references:
            points = self.group_points.get(group, {})
            below, above = (
                [s for s in points if s < size],
                [s for s in points if s > size],
            )
            sources = [
                s
                for s in (
                    size if size in points else None,
                    max(below, default=None),
                    min(above, default=None),
                )
                if s is not None
            ]
            self._priority_references[cache_key] = [
                self.configs[s, points[s]] for s in sources
            ]
        references = self._priority_references[cache_key]
        config = self.configs[size, key]

        def distance(other):
            compiled = sum(config[field] != other[field] for field in COMPILED_FIELDS)
            categorical, numeric = 0, 0.0
            for field, value in config.items():
                previous = other.get(field)
                if type(value) in (int, float) and type(previous) in (int, float):
                    numeric += abs(math.log2((abs(value) + 1) / (abs(previous) + 1)))
                else:
                    categorical += value != previous
            return compiled, categorical, numeric

        return min((distance(other) for other in references), default=(0, 0, 0)), key

    def _boundary(self, size):
        neighbors = self._neighbors(size)
        return len(neighbors) == 2 and _compiled(
            self.configs[neighbors[0], self.best[neighbors[0]]]
        ) != _compiled(self.configs[neighbors[1], self.best[neighbors[1]]])

    def _numeric_profiles(self, size, group):
        keys = self.groups[size][group]
        configs = [self.configs[size, key] for key in keys]
        fields = [
            field
            for field, value in configs[0].items()
            if field not in COMPILED_FIELDS and type(value) in (int, float)
        ]
        below = [s for s in self.anchors if s < size and group in self.groups[s]]
        above = [s for s in self.anchors if s > size and group in self.groups[s]]
        neighbors = [
            s
            for s in (max(below, default=None), min(above, default=None))
            if s is not None
        ]
        selected = []
        for field in fields:
            values = {config[field] for config in configs}
            targets = {max(values)}
            if neighbors:
                adjacent = {
                    self.configs[s, key][field]
                    for s in neighbors
                    for key in self.groups[s][group]
                }
                targets.update(values - adjacent)
            for value in sorted(targets, reverse=True):
                selected.append(
                    min(key for key in keys if self.configs[size, key][field] == value)
                )
        return list(dict.fromkeys(selected or keys[:1]))

    def _seed_profiles(self, size):
        if size in self._seeded:
            return
        for group, keys in self.groups[size].items():
            anchors = [
                s
                for s in self.anchors
                if s != size and group in self.anchor_group_best[s]
            ]
            below = max((s for s in anchors if s < size), default=None)
            above = min((s for s in anchors if s > size), default=None)
            sources = [s for s in (below, above) if s is not None]
            seeds = []
            if size not in self.anchors and sources:
                for source in sources:
                    reference = self.configs[
                        source, self.anchor_group_best[source][group]
                    ]
                    numeric = [
                        field
                        for field, value in reference.items()
                        if field not in COMPILED_FIELDS and type(value) in (int, float)
                    ]
                    seeds.append(
                        min(
                            keys,
                            key=lambda key: (
                                sum(
                                    abs(
                                        math.log2(
                                            (abs(self.configs[size, key][field]) + 1)
                                            / (abs(reference[field]) + 1)
                                        )
                                    )
                                    for field in numeric
                                ),
                                key,
                            ),
                        )
                    )
            elif group in self.group_best[size]:
                seeds.append(self.group_best[size][group])
            else:
                measured = [s for s in self.anchors if group in self.group_best[s]]
                if measured:
                    nearest = min(measured, key=lambda s: (abs(math.log2(s / size)), s))
                    reference = self.configs[nearest, self.group_best[nearest][group]]
                    seeds.append(
                        min(
                            keys,
                            key=lambda key: (
                                sum(
                                    abs(
                                        math.log2(
                                            (abs(self.configs[size, key][field]) + 1)
                                            / (abs(value) + 1)
                                        )
                                    )
                                    for field, value in reference.items()
                                    if field not in COMPILED_FIELDS
                                    and type(value) in (int, float)
                                ),
                                key,
                            ),
                        )
                    )
                else:
                    seeds.append(min(keys, key=lambda key: self._priority(size, key)))
            for seed in seeds:
                self.required_profiles[size][group].add(seed)
                if (size, seed) not in self.samples:
                    self.coarse_remaining[size][seed] = None
        self._seeded.add(size)

    def _coverage_choices(self, size):
        self._seed_profiles(size)
        per_group = []
        for group in self.groups[size]:
            keys = [
                key
                for key in self.required_profiles[size][group]
                if key in self.remaining[size] and key in self.coarse_remaining[size]
            ]
            per_group.append(sorted(keys, key=lambda key: self._priority(size, key)))
        return [
            keys[index]
            for index in range(max(map(len, per_group), default=0))
            for keys in per_group
            if index < len(keys)
        ]

    def _coarse_progress(self, size):
        required = sum(map(len, self.required_profiles[size].values()))
        return 1 - len(self.coarse_remaining[size]) / max(1, required)

    def _anchor_profiles(self, size):
        probes = []
        for group, keys in self.groups[size].items():
            by_active = {
                self.configs[size, key].get("active_blocks", 0): key for key in keys
            }
            maximum = max(by_active)
            selected = []
            for target in (1, 8, 32, maximum):
                active = min(by_active, key=lambda value: (abs(value - target), value))
                key = by_active[active]
                if key in self.remaining[size] and key not in selected:
                    selected.append(key)
            probes.append(selected)
        return [
            keys[index]
            for index in range(max(map(len, probes), default=0))
            for keys in probes
            if index < len(keys)
        ]

    def _rounds(self, size, phase):
        estimates = []
        if size in self.best:
            estimates.append(self.latencies[size, self.best[size]])
        if size in self.confirmations:
            estimates.append(
                statistics.median(self.confirmations[size][0]["samples_us"])
            )
        return measurement_rounds(
            size,
            phase,
            estimate_us=min(estimates, default=None),
            minimum_rounds=self.rounds,
        )

    def _admit(self, size, keys, rounds, phase, cutoff, *, whole=False):
        available = max(0.0, cutoff - self.clock())
        count = min(
            len(keys),
            len(keys) if whole else self.batch_size,
            int(available / (self._seconds_per_config_round * rounds)),
        )
        if count < 1 or (whole and count < len(keys)):
            return None
        self.pending = Trial(
            size,
            tuple(_copy(self.configs[size, key]) for key in keys[:count]),
            rounds,
            phase,
            self._next_batch,
        )
        self._next_batch += 1
        self._pending_started, self._pending_deadline = self.clock(), self.deadline
        return self.pending

    def propose(self):
        if self.pending is not None:
            raise RuntimeError("observe the pending trial before proposing another")
        if len(self._resumed_complete) == len(self.sizes):
            self.reason = "cached_complete"
            return None
        self._priority_references.clear()
        now = self.clock()
        final_cutoff = self.deadline - self.finalization_seconds
        search_cutoff = final_cutoff - self.confirmation_seconds
        if self._anchor_until is None:
            self._anchor_until = now + (
                0.0
                if self._coverage_started
                else min(30.0, max(0.0, search_cutoff - now) * 0.2)
            )
        else:
            self._anchor_until += self.deadline - self._last_deadline
        self._last_deadline = self.deadline
        if now >= final_cutoff:
            self.reason = "deadline"
            return None
        remaining = [
            s for s in self.sizes if self.remaining[s] and s not in self._resumed_ready
        ]
        if remaining and now < search_cutoff and self._confirmation is None:
            choices = {}
            if not self.exhaustive:
                candidates = self.anchors if now < self._anchor_until else self.sizes
                choices = {
                    s: self.coarse_remaining[s]
                    for s in candidates
                    if self.coarse_remaining[s].keys() & self.remaining[s].keys()
                    and s not in self._resumed_ready
                }
                if (
                    not choices
                    and candidates != self.sizes
                    and not any(
                        self.remaining[s]
                        for s in self.anchors
                        if s not in self._resumed_ready
                    )
                ):
                    choices = {
                        s: self.coarse_remaining[s]
                        for s in self.sizes
                        if self.coarse_remaining[s].keys() & self.remaining[s].keys()
                        and s not in self._resumed_ready
                    }
            phase = "refinement"
            anchor_remaining = [
                s
                for s in self.anchors
                if self.remaining[s] and s not in self._resumed_ready
            ]
            if self.exhaustive:
                size = min(remaining)
                keys = list(self.remaining[size])
            elif choices:
                phase = (
                    "anchor"
                    if now < self._anchor_until and set(choices).issubset(self.anchors)
                    else "coverage"
                )
                size = min(
                    choices,
                    key=lambda s: (
                        self._coarse_progress(s),
                        not self._boundary(s),
                        s,
                    ),
                )
                keys = self._coverage_choices(size)
            elif anchor_remaining and now < self._anchor_until:
                phase = "anchor"
                coarse = {
                    s: keys
                    for s in anchor_remaining
                    if (keys := self._anchor_profiles(s))
                }
                size = min(
                    coarse or anchor_remaining,
                    key=lambda s: (
                        1 - len(self.remaining[s]) / len(self.catalogs[s]),
                        s,
                    ),
                )
                keys = (
                    coarse[size]
                    if coarse
                    else sorted(
                        self.remaining[size], key=lambda k: self._priority(size, k)
                    )
                )
            else:
                size = min(
                    remaining,
                    key=lambda s: (
                        1 - len(self.remaining[s]) / len(self.catalogs[s]),
                        not self._boundary(s),
                        s,
                    ),
                )
                keys = sorted(
                    self.remaining[size], key=lambda k: self._priority(size, k)
                )
            trial = self._admit(
                size, keys, self._rounds(size, phase), phase, search_cutoff
            )
            if trial is not None:
                return trial
        if self._confirmation is None:
            self._confirmation = {
                size: self._shortlist(size)
                for size in self.sizes
                if size not in self._resumed_complete
            }
        final_choices = {
            size: [_key(row["config"]) for row in cohort[:2]]
            for size, cohort in self.confirmations.items()
            if size not in self.confirmation_sizes
            and size not in self._resumed_complete
        }
        choices = final_choices or {
            size: keys
            for size, keys in self._confirmation.items()
            if keys and size not in self.confirmations
        }
        if not choices:
            self.reason = "confirmation_complete"
            return None
        size = min(choices)
        phase = "final" if final_choices else "confirmation"
        trial = self._admit(
            size,
            choices[size],
            self._rounds(size, phase),
            phase,
            final_cutoff,
            whole=True,
        )
        if trial is None:
            self.reason = phase + "_budget_exhausted"
        return trial

    def observe(self, trial, records):
        if self.pending != trial:
            raise ValueError("observation does not match the pending trial")
        records = [_record(record) for record in records]
        expected = {_key(config) for config in trial.configs}
        seen = [_key(record["config"]) for record in records]
        if (
            len(seen) != len(set(seen))
            or set(seen) != expected
            or any(record["bytes"] != trial.size_bytes for record in records)
        ):
            raise ValueError(
                "observe requires one measurement for each proposed configuration"
            )
        if any(
            _passed(record)
            and len(record["samples_us"]) != trial.rounds * SAMPLES_PER_ROUND
            for record in records
        ):
            raise ValueError("measurement sample count does not match requested rounds")
        elapsed = (
            self.clock()
            - self._pending_started
            - (self.deadline - self._pending_deadline)
        )
        cost = max(0.0, elapsed) / (len(records) * trial.rounds)
        self._seconds_per_config_round = (
            max(self._seconds_per_config_round * 0.8, cost) * 1.1
        )
        self.phase_seconds[trial.phase] = self.phase_seconds.get(
            trial.phase, 0.0
        ) + max(0.0, elapsed)
        self.phase_trials[trial.phase] = self.phase_trials.get(trial.phase, 0) + 1
        for record in records:
            record.update(
                phase=trial.phase,
                batch=trial.batch_id,
                batch_size=len(trial.configs),
                rounds=trial.rounds,
            )
            self._add(record)
        if (
            trial.size_bytes in self._resumed_coverage_only
            and not self.coarse_remaining[trial.size_bytes]
        ):
            self._resumed_ready.add(trial.size_bytes)
        self.pending = None

    def summary(self):
        winners = [
            self.final[size]
            if size in self.final
            else dict(
                size_bytes=size,
                config=_copy(self.configs[size, self.best[size]]),
                median_us=self.latencies[size, self.best[size]],
                confirmed=False,
            )
            for size in self.sizes
            if size in self.best or size in self.final
        ]
        missing = [
            dict(
                size_bytes=size,
                groups=[
                    json.loads(group)
                    for group in groups
                    if group not in self.covered[size]
                ],
            )
            for size, groups in self.groups.items()
        ]
        missing = [entry for entry in missing if entry["groups"]]
        failures = [record for record in self.records if not _passed(record)]
        confirmed = len(self.confirmation_sizes)
        missing_profiles = [
            dict(
                size_bytes=size,
                configs=[_copy(self.configs[size, key]) for key in keys],
            )
            for size, keys in self.coarse_remaining.items()
            if keys
        ]
        complete = (
            not missing
            and not missing_profiles
            and not failures
            and confirmed == len(self.sizes)
            and (not self.exhaustive or not any(self.remaining.values()))
        )
        coverage = [
            dict(
                size_bytes=s,
                measured_groups=len(self.covered[s]),
                total_groups=len(self.groups[s]),
                measured_configurations=len(self.by_size[s]),
                total_configurations=len(self.catalogs[s]),
                required_profiles=sum(map(len, self.required_profiles[s].values())),
                missing_profiles=len(self.coarse_remaining[s]),
            )
            for s in self.sizes
        ]
        return dict(
            status="complete" if complete else "partial",
            reason=self.reason or "in_progress",
            winners=winners,
            total_sizes=len(self.sizes),
            measured_sizes=len(self.best),
            total_configurations=len(self.configs),
            measured_configurations=len(self.samples),
            attempted_configurations=len(self.attempted),
            failed_configurations=len(self.invalid),
            confirmed_sizes=confirmed,
            initial_confirmed_sizes=len(self.confirmations),
            missing=missing,
            missing_profiles=missing_profiles,
            failures=_copy(failures),
            search_coverage=coverage,
            unexplored_configurations=sum(
                len(left) for left in self.remaining.values()
            ),
            phase_seconds=dict(self.phase_seconds),
            phase_trials=dict(self.phase_trials),
            estimated_seconds_per_config_round=self._seconds_per_config_round,
            screening_group_winners=[
                dict(
                    size_bytes=size,
                    group=json.loads(group),
                    config=_copy(self.configs[size, key]),
                    median_us=self.latencies[size, key],
                )
                for size in self.sizes
                for group, key in self.group_best[size].items()
            ],
        )


def _read_lines(lines, path):
    metadata, records, summaries = None, [], []
    for index, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("report record must be a JSON object")
            if index == 1:
                if row.get("kind") != "metadata" or row.get("schema_version") != 1:
                    raise ValueError("unsupported report header")
                metadata = row["metadata"]
            elif row.get("kind") == "measurement":
                records.append(_record(row))
            elif row.get("kind") == "summary":
                summaries.append(row["summary"])
            else:
                raise ValueError("unsupported report record")
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{index}: {error}") from error
    if metadata is None:
        raise ValueError("report has no metadata header")
    return metadata, tuple(records), tuple(summaries)


def read_report(path):
    with Path(path).open() as stream:
        return _read_lines(stream, path)


class RawReport:
    """Append measured samples and summaries without overwriting an existing run."""

    def __init__(self, path, metadata, *, resume=False):
        self.path, self.metadata = Path(path), _copy(metadata)
        self._records = []
        if resume:
            repair = None
            try:
                old, records, _ = read_report(path)
            except ValueError:
                data = self.path.read_bytes()
                lines = data.splitlines(keepends=True)
                if not lines or lines[-1].endswith(b"\n"):
                    raise
                try:
                    json.loads(lines[-1])
                except (ValueError, UnicodeError):
                    prefix = b"".join(lines[:-1])
                    old, records, _ = _read_lines(prefix.decode().splitlines(), path)
                    repair = len(prefix)
                else:
                    raise
            if _key(old) != _key(self.metadata):
                raise ValueError(
                    "report metadata is incompatible with this measurement"
                )
            if repair is not None:
                with self.path.open("r+b") as stream:
                    stream.truncate(repair)
            elif self.path.stat().st_size:
                with self.path.open("rb") as stream:
                    stream.seek(-1, 2)
                    terminated = stream.read(1) == b"\n"
                if not terminated:
                    with self.path.open("ab") as stream:
                        stream.write(b"\n")
            self._records.extend(records)
            self.stream = self.path.open("a")
        else:
            self.stream = self.path.open("x")
            self._write(dict(kind="metadata", schema_version=1, metadata=self.metadata))

    @property
    def records(self):
        return tuple(self._records)

    def _write(self, record):
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.stream.flush()

    def append(self, record):
        record = _record(record)
        record["kind"] = "measurement"
        self._write(record)
        self._records.append(record)

    def append_summary(self, summary):
        self._write(dict(kind="summary", summary=_copy(summary)))

    def close(self):
        self.stream.close()
