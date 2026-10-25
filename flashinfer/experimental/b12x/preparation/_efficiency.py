"""Declaration-time controls and finite ladders for search efficiency."""

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum


class AutotuneEffort(str, Enum):
    """Heuristic preparation or progressively wider candidate searches.

    Low applies broad performance exclusions, medium is the default policy,
    high retains conservative exclusions, and max lifts all efficiency filters.
    Correctness, resources, and explicit configuration constraints always apply.
    """

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

    @property
    def rank(self) -> int:
        return _EFFORT_RANKS[self]


_EFFORT_RANKS = {effort: rank for rank, effort in enumerate(AutotuneEffort)}
_AUTOTUNE_EFFORT: ContextVar[AutotuneEffort | None] = ContextVar(
    "b12x_autotune_effort", default=None
)


@contextmanager
def autotune_effort(effort: AutotuneEffort | str):
    """Capture explicit effort in declarations and sessions within this scope.

    This setting takes precedence over environment controls. Off prepares the
    heuristic without races or tuned-selection cache lookups. Higher efforts
    expand the family's declared candidate axes, preserving mandatory legality.
    """
    token = _AUTOTUNE_EFFORT.set(AutotuneEffort(effort))
    try:
        yield
    finally:
        _AUTOTUNE_EFFORT.reset(token)


def capture_autotune_effort() -> AutotuneEffort:
    explicit = _AUTOTUNE_EFFORT.get()
    if explicit is not None:
        return explicit
    value = os.environ.get("B12X_AUTOTUNE_EFFORT")
    if value is not None:
        try:
            return AutotuneEffort(value)
        except ValueError:
            raise ValueError(
                "B12X_AUTOTUNE_EFFORT must be off, low, medium, high, or max"
            ) from None
    value = os.environ.get("B12X_AUTOTUNE_EXHAUSTIVE", "0")
    if value not in ("0", "1"):
        raise ValueError("B12X_AUTOTUNE_EXHAUSTIVE must be 0 or 1")
    return AutotuneEffort.MAX if value == "1" else AutotuneEffort.MEDIUM


def capture_exhaustive_search() -> bool:
    return capture_autotune_effort() is AutotuneEffort.MAX


@dataclass(frozen=True)
class EfficiencyPredicate:
    """Apply a performance filter below its minimum autotune effort."""

    predicate: Callable[[Mapping[str, object]], bool]
    minimum_effort: AutotuneEffort | str = AutotuneEffort.MAX

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_effort", AutotuneEffort(self.minimum_effort))
        if not callable(self.predicate):
            raise TypeError("efficiency predicate must be callable")

    def __call__(self, assignment: Mapping[str, object]) -> bool:
        return self.predicate(assignment)

    def applies(self, effort: AutotuneEffort | str) -> bool:
        return AutotuneEffort(effort).rank < self.minimum_effort.rank


def powers_of_two(capacity: int) -> frozenset[int]:
    return frozenset(1 << bit for bit in range(capacity.bit_length()))
