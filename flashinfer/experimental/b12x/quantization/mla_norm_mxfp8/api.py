"""Public prepared split RMS-normalization and query-encoding interface."""

from ..._lib.gating import default_is_supported
from . import META
from ._tuning import Query, Config
from ._preparation import plan, run


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


__all__ = ["Query", "Config", "plan", "run", "is_supported"]
