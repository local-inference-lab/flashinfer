# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Public surface for :mod:`b12x.moe.situ_moe`."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Binding, make_plan
from ._quant import partition_csf_slab_offsets
from ._tuning import SituMoeConfig, SituMoeQuery, validate_query

_COMPONENT = "moe.situ_moe"


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static contract for one SM120/SM121 fused route-weighted SiTU MoE family."""

    device: torch.device | str
    weight_format: str  # "nvfp4" or "mxfp8"
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    max_tokens: int
    has_csf: bool = False
    deterministic_output: bool = False


def plan(caps: Caps, *, invocation=None, override: SituMoeConfig | None = None) -> Plan:
    """Declare a prepared SiTU MoE execution plan."""
    return make_plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch=(),
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    output: torch.Tensor,
    g13: torch.Tensor | None = None,
    g2: torch.Tensor | None = None,
    s13_exc: torch.Tensor | None = None,
    s13_slab_offsets: torch.Tensor | None = None,
    s2_exc: torch.Tensor | None = None,
    s2_slab_offsets: torch.Tensor | None = None,
) -> Binding:
    """Bind live activation, routing, weight, and scratch tensors to a prepared plan."""
    state = require_prepared(plan, _COMPONENT, x.device)
    return state.bind(
        plan=plan,
        scratch=scratch,
        x=x,
        ids=ids,
        weights=weights,
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        output=output,
        g13=g13,
        g2=g2,
        s13_exc=s13_exc,
        s13_slab_offsets=s13_slab_offsets,
        s2_exc=s2_exc,
        s2_slab_offsets=s2_slab_offsets,
    )


def run(binding: Binding) -> torch.Tensor:
    """Execute a bound prepared SiTU MoE plan."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be situ_moe.Binding")
    return require_prepared(binding.plan, _COMPONENT, binding.x.device).run(binding)


def is_supported(device=None) -> bool:
    """Return True if the target CUDA device supports SM120/SM121 SiTU MoE."""
    return default_is_supported(device, requires=META.requires)


def supports(
    *,
    weight_format: str,
    hidden: int,
    intermediate: int,
    num_experts: int,
    top_k: int,
    max_tokens: int,
    has_csf: bool = False,
    deterministic_output: bool = False,
) -> bool:
    """Return True if the requested geometry and quantization format are supported."""
    try:
        validate_query(
            SituMoeQuery(
                weight_format=weight_format,
                hidden=int(hidden),
                intermediate=int(intermediate),
                num_experts=int(num_experts),
                top_k=int(top_k),
                max_tokens=int(max_tokens),
                has_csf=bool(has_csf),
                deterministic_output=bool(deterministic_output),
                sm_count=1,
            )
        )
    except (TypeError, ValueError):
        return False
    return True


def clear_caches() -> None:
    """Compiled programs live on prepared plans."""


__all__ = [
    "Binding",
    "Caps",
    "Plan",
    "SituMoeConfig",
    "SituMoeQuery",
    "bind",
    "clear_caches",
    "is_supported",
    "partition_csf_slab_offsets",
    "plan",
    "run",
    "supports",
]
