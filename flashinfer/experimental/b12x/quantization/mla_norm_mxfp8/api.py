"""Public prepared split RMS-normalization and query-encoding interface.

Allocate ``values_base``, ``scale_rows_base``, and ``scale_mma_base`` with
``mxfp8.dense_rows_storage_shapes(rows, 1280)`` and keep them live through
the last graph replay. ``mxfp8.dense_rows_from_bases`` supplies the canonical
views for a prepared MXFP8 GEMM. ``run`` writes both BF16 normalization
outputs and every logical encoded query value and scale.
"""

from ..._lib.gating import default_is_supported
from . import META
from ._tuning import Query, Config
from ._preparation import plan, run


def is_supported(device=None):
    return default_is_supported(device, requires=META.requires)


__all__ = ["Query", "Config", "plan", "run", "is_supported"]
