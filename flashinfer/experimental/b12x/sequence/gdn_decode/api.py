"""Public surface for :mod:`b12x.sequence.gdn_decode`."""

from __future__ import annotations

from ..._lib.gating import has_cutlass_dsl, has_triton
from b12x.preparation import Plan

from . import reference
from ._impl import (
    Binding,
    Caps,
    KdaBinding,
    bind,
    bind_kda,
    run,
    run_kda,
)
from ._preparation import plan, invocation_from_tensors
from ._tuning import GdnConfig, GdnQuery
from ._commit import KdaCommitBinding, bind_kda_commit, run_kda_commit


def is_supported(device=None) -> bool:
    """True when mandatory Qwen CuTe and its Triton auxiliaries are usable."""
    del device
    return has_cutlass_dsl() and has_triton()


def prepare_kda_fused_verify(binding, conv, *, retain_examples=True):
    """Explicitly opt into experimental fused KDA verification before capture.

    Requires recovery-mode KDA with equal heads, a four-tap dilation-one
    convolution without bias, BF16 rounding, recurrent tiles of 16 or 32,
    and at most eight verification positions. Sequence metadata and the
    convolution-output/recurrence-input allocation must be shared. The fused
    kernel does not materialize that intermediate output.

    With ``retain_examples=False``, later calls must supply ``binding``
    and ``conv`` operands with the prepared metadata and alias contract.
    """
    from ._fused_verify import prepare_kda_fused_verify as prepare_fused

    return prepare_fused(binding, conv, retain_examples=retain_examples)


def run_kda_fused_verify(
    fused, *, binding=None, conv=None, lower_bound=-5.0, eps=1e-6, scale=None
):
    """Run the explicitly selected experimental fused verification on its operand device.

    The preparation restrictions in ``prepare_kda_fused_verify`` apply. Omit
    both operands to use retained examples, or provide both for dynamic
    execution. Mutable storage must not overlap live reads or other outputs;
    shapes, dtypes and compiled strides must match preparation. Scale and
    epsilon must be finite and positive; the gate lower bound finite and
    negative. Returns the supplied or retained output tensor.
    """
    from ._fused_verify import run_kda_fused_verify as run_fused

    return run_fused(
        fused, binding=binding, conv=conv, lower_bound=lower_bound, eps=eps, scale=scale
    )


__all__ = [
    "Binding",
    "Caps",
    "GdnConfig",
    "GdnQuery",
    "KdaBinding",
    "KdaCommitBinding",
    "Plan",
    "bind",
    "bind_kda",
    "bind_kda_commit",
    "is_supported",
    "plan",
    "invocation_from_tensors",
    "reference",
    "run",
    "run_kda",
    "run_kda_commit",
    "prepare_kda_fused_verify",
    "run_kda_fused_verify",
]
