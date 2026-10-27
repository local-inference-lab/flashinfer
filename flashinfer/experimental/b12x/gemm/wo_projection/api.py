"""Prepared public surface for native MXFP8 WO projection."""

from __future__ import annotations
from dataclasses import replace

import torch
from ..._lib.gating import default_is_supported
from ..._lib.scratch import require_disjoint
from ...preparation import Plan
from ...preparation.types import plan_from_handle, require_prepared
from .._shared.wo_mxfp8 import MXFP8Rows
from .._shared.wo_mxfp8 import WOProjectionBinding as Binding
from .._shared.wo_mxfp8 import WOProjectionInvRopeBinding as InvRopeBinding
from .._shared.wo_mxfp8 import WOProjectionMXFP8Weights as Weights
from .._shared.wo_mxfp8 import WOProjectionScratchCaps as Caps
from .._shared.wo_mxfp8 import (
    pack_wo_projection_fp8_block_scaled_weights_mxfp8 as pack_weights,
)
from .._shared import wo_mxfp8 as _shared
from ._preparation import plan
from ._tuning import WoProjectionConfig, WoProjectionQuery
from . import META


@torch.library.custom_op("b12x::wo_projection_out", mutates_args=("out", "scratch"))
def _run_out(
    source: torch.Tensor,
    weights: list[torch.Tensor],
    tiled_a: torch.Tensor | None,
    tiled_b: torch.Tensor | None,
    out: torch.Tensor,
    scratch: torch.Tensor,
    handle: int,
) -> None:
    require_disjoint(
        out, (source, scratch, *weights, tiled_a, tiled_b), owner="WO output"
    )
    plan = plan_from_handle(handle)
    state = require_prepared(plan, "gemm.wo_projection", source.device)
    q = plan.query

    def matrix(parts, tiled):
        return MXFP8Rows(
            parts[0],
            parts[1].view(torch.float8_e8m0fnu),
            parts[2].view(torch.float8_e8m0fnu),
            tiled,
        )

    packed = Weights(
        matrix(weights[:3], tiled_a),
        matrix(weights[3:], tiled_b),
        q.groups,
        q.group_width,
        q.rank,
        q.hidden,
        q.sfb_k_replicated,
    )
    binding = state.bind(
        source_tgd=source,
        weights=packed,
        out=out,
        scratch=scratch,
        return_3d=q.return_3d,
    )
    state.run(binding)


@_run_out.register_fake
def _run_out_fake(source, weights, tiled_a, tiled_b, out, scratch, handle) -> None:
    pass


def run_out(source, weights: Weights, *, out, scratch, plan: Plan):
    """Write a prepared projection into output with caller-owned private scratch."""
    operands = [
        tensor
        for packed in (weights.wo_a, weights.wo_b)
        for tensor in (
            packed.values,
            packed.scale_rows.view(torch.uint8),
            packed.scale_mma.view(torch.uint8),
        )
    ]
    _run_out(
        source,
        operands,
        weights.wo_a.values_tiled,
        weights.wo_b.values_tiled,
        out,
        scratch,
        plan.handle,
    )
    return out


def bind(plan: Plan, **kwargs) -> Binding:
    """Bind caller-owned WO tensors and scratch to a ready plan."""
    source = kwargs.get("source_tgd")
    state = require_prepared(
        plan, "gemm.wo_projection", source.device if source is not None else None
    )
    return replace(state.bind(**kwargs), plan=plan)


def bind_inv_rope(plan: Plan, **kwargs) -> InvRopeBinding:
    """Bind inverse-RoPE tensors and caller-owned scratch to a ready plan."""
    source = kwargs.get("o")
    state = require_prepared(
        plan, "gemm.wo_projection", source.device if source is not None else None
    )
    return replace(state.bind_inv_rope(**kwargs), plan=plan)


def run(*, binding: Binding, plan: Plan, stream=None):
    """Run a prepared WO projection; declarations and raw bindings are rejected."""
    state = require_prepared(plan, "gemm.wo_projection", binding.source_tgd.device)
    if binding.plan is not plan:
        raise ValueError("WO projection binding belongs to a different prepared plan")
    return state.run(binding, stream=stream)


def run_inv_rope(*, binding: InvRopeBinding, plan: Plan, stream=None):
    """Run a prepared inverse-RoPE WO projection."""
    state = require_prepared(plan, "gemm.wo_projection", binding.o.device)
    if binding.plan is not plan:
        raise ValueError(
            "WO projection inverse-RoPE binding belongs to a different prepared plan"
        )
    return state.run_inv_rope(binding, stream=stream)


def quantize_input(source_tgd, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", source_tgd.device)
    return state.quantize_a(source_tgd, out=out)


def quantize_input_inv_rope(
    o,
    positions,
    cos_sin_cache,
    *,
    groups,
    heads_per_group,
    nope_dim=448,
    rope_dim=64,
    plan: Plan,
    out=None,
):
    state = require_prepared(plan, "gemm.wo_projection", o.device)
    return state.quantize_a_inv_rope(
        o,
        positions,
        cos_sin_cache,
        groups=groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        out=out,
    )


def quantize_input_b(tmp_trg, *, plan: Plan, out=None):
    state = require_prepared(plan, "gemm.wo_projection", tmp_trg.device)
    return state.quantize_b(tmp_trg, out=out)


def is_supported(device=None) -> bool:
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Caps",
    "Plan",
    "Binding",
    "InvRopeBinding",
    "Weights",
    "MXFP8Rows",
    "WoProjectionConfig",
    "WoProjectionQuery",
    "plan",
    "bind",
    "bind_inv_rope",
    "run",
    "run_out",
    "run_inv_rope",
    "pack_weights",
    "quantize_input",
    "quantize_input_inv_rope",
    "quantize_input_b",
    "is_supported",
]
