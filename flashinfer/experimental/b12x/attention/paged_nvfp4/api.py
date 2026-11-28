# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Public surface for :mod:`b12x.attention.paged_nvfp4`.

Numerical and edge-case contract:
  - ``kv_format="nvfp4_4o6k"`` stores block-16 NVFP4 K (4-or-6 MAE block scale
    selection) and block-16 NVFP4 V (max/6 block scale).
  - Decode/verify uses BF16-MMA (preserving full E2M1 * E4M3 dynamic range up to
    ``2688.0``). Prefill uses SM120 FP8-MMA with in-register ``e2m1 * e4m3 ->
    e4m3`` saturation, clamping dequantized ``|K|`` and ``|V|`` to ``<= 448.0``.
  - Requests with ``cache_seqlens[b] <= 0`` or ``qo_len <= 0`` exit early without
    writing ``output`` rows (matching ``b12x.attention.paged_decode``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import Plan
from b12x.preparation.types import require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Binding, make_plan
from ._tuning import PagedNVFP4Config, PagedNVFP4Query, validate_query

_COMPONENT = "attention.paged_nvfp4"


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Static contract for one paged NVFP4 (4-or-6 MAE K) attention family."""

    device: torch.device | str
    mode: str  # "decode", "verify", or "prefill"
    kv_format: str = "nvfp4_4o6k"
    num_q_heads: int
    num_kv_heads: int
    head_dim_qk: int
    head_dim_vo: int
    page_size: int
    max_batch: int
    max_q_per_req: int = 1
    window_left: int = -1
    causal: bool = True
    has_sinks: bool = False
    sink_layout: str = "head"
    has_relative_bias: bool = False
    has_factored_relative_bias: bool = False
    caller_owned_placeholders: bool = False


def plan(
    caps: Caps, *, invocation=None, override: PagedNVFP4Config | None = None
) -> Plan:
    """Declare a prepared paged NVFP4 attention and KV-write plan."""
    return make_plan(caps, invocation=invocation, override=override)


def bind(
    plan: Plan,
    *,
    scratch=(),
    q: torch.Tensor,
    k_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_cache: torch.Tensor,
    v_scales: torch.Tensor,
    output: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    attention_sink_bias: torch.Tensor | None = None,
    relative_attention_bias: torch.Tensor | None = None,
    relative_bias_factors: torch.Tensor | None = None,
    relative_bias_projection: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> Binding:
    """Validate and bind live tensors to a prepared ``paged_nvfp4`` plan."""
    state = require_prepared(plan, _COMPONENT, q.device)
    return state.bind(
        plan=plan,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=output,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=attention_sink_bias,
        relative_attention_bias=relative_attention_bias,
        relative_bias_factors=relative_bias_factors,
        relative_bias_projection=relative_bias_projection,
        softmax_scale=softmax_scale,
    )


def run(binding: Binding) -> torch.Tensor:
    """Execute a bound prepared ``paged_nvfp4`` plan."""
    if not isinstance(binding, Binding):
        raise TypeError("binding must be paged_nvfp4.Binding")
    return require_prepared(binding.plan, _COMPONENT, binding.q.device).run(binding)


def write_kv(
    plan: Plan,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_cache: torch.Tensor,
    v_scales: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Quantize BF16 ``key``/``value`` and append to paged NVFP4 caches (negative slots skipped)."""
    require_prepared(plan, _COMPONENT, key.device).write_kv(
        key=key,
        value=value,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        slot_mapping=slot_mapping,
    )


def is_supported(device=None) -> bool:
    """Return True if the target CUDA device supports SM120/SM121 ``paged_nvfp4``."""
    return default_is_supported(device, requires=META.requires)


def supports(
    *,
    mode: str,
    kv_format: str = "nvfp4_4o6k",
    num_q_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    head_dim_vo: int,
    page_size: int,
    max_q_per_req: int = 1,
    window_left: int = -1,
    causal: bool = True,
    has_sinks: bool = False,
    sink_layout: str = "head",
    has_relative_bias: bool = False,
    has_factored_relative_bias: bool = False,
) -> bool:
    """Return True if the requested geometry and feature flags are supported."""
    try:
        validate_query(
            PagedNVFP4Query(
                mode=mode,
                kv_format=kv_format,
                num_q_heads=int(num_q_heads),
                num_kv_heads=int(num_kv_heads),
                head_dim_qk=int(head_dim_qk),
                head_dim_vo=int(head_dim_vo),
                page_size=int(page_size),
                max_batch=1,
                max_q_per_req=int(max_q_per_req),
                window_left=int(window_left),
                causal=bool(causal),
                has_sinks=bool(has_sinks),
                sink_layout=sink_layout,
                has_relative_bias=bool(has_relative_bias),
                has_factored_relative_bias=bool(has_factored_relative_bias),
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
    "PagedNVFP4Config",
    "PagedNVFP4Query",
    "Plan",
    "bind",
    "clear_caches",
    "is_supported",
    "plan",
    "run",
    "supports",
    "write_kv",
]
