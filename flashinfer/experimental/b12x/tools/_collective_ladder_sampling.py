"""Payload-scaled measurement counts for collective ladder timing."""

import math

SAMPLES_PER_ROUND = 6
MAX_ROUNDS = 512
_SCREENING_PHASES = frozenset(("search", "anchor", "coverage", "refinement"))


def measurement_rounds(size_bytes, phase, *, estimate_us=None, minimum_rounds=1):
    """Budget roughly 1 ms for confirmation and 4 ms for final candidate timing."""
    if type(size_bytes) is not int or size_bytes <= 0:
        raise ValueError("payload size must be a positive integer")
    if type(minimum_rounds) is not int or not 1 <= minimum_rounds <= MAX_ROUNDS:
        raise ValueError(f"minimum rounds must be in [1, {MAX_ROUNDS}]")
    if phase not in _SCREENING_PHASES | {"confirmation", "final", "replay"}:
        raise ValueError("unknown collective measurement phase")
    if estimate_us is not None and (
        type(estimate_us) not in (int, float)
        or not math.isfinite(estimate_us)
        or estimate_us <= 0
    ):
        raise ValueError("latency estimate must be positive and finite")
    rounds = min(8, math.ceil(32768 / size_bytes))
    floor = 1
    if phase not in _SCREENING_PHASES:
        floor = 2 if phase == "confirmation" else 4
        if estimate_us is not None:
            target_us = 1000 if phase == "confirmation" else 4000
            rounds = (
                MAX_ROUNDS
                if estimate_us <= target_us / (SAMPLES_PER_ROUND * MAX_ROUNDS)
                else math.ceil(target_us / (SAMPLES_PER_ROUND * estimate_us))
            )
    return min(MAX_ROUNDS, max(minimum_rounds, floor, rounds))
