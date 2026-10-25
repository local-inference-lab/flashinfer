"""Autotune effort tiers and finite ladders for search efficiency."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum


class AutotuneEffort(str, Enum):
    """Heuristic preparation, the default pruned search, or the audit search.

    Off prepares the heuristic configuration without races. Default races the
    legal candidates that pass every declared efficiency predicate. Audit lifts
    the efficiency predicates and races the whole legal space; it exists to
    check that the default search prunes no winners. Correctness, resource, and
    explicit configuration constraints apply at every effort.
    """

    OFF = "off"
    DEFAULT = "default"
    AUDIT = "audit"


_AUTOTUNE_EFFORT: ContextVar[AutotuneEffort | None] = ContextVar(
    "b12x_autotune_effort", default=None
)


@contextmanager
def autotune_effort(effort: AutotuneEffort | str):
    """Capture explicit effort in declarations and sessions within this scope.

    This setting takes precedence over ``B12X_AUTOTUNE_EFFORT``.
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
    if value is None:
        return AutotuneEffort.DEFAULT
    try:
        return AutotuneEffort(value)
    except ValueError:
        raise ValueError(
            "B12X_AUTOTUNE_EFFORT must be off, default, or audit"
        ) from None


def powers_of_two(capacity: int) -> frozenset[int]:
    return frozenset(1 << bit for bit in range(capacity.bit_length()))
