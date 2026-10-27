"""Prepared batched and packed-varlen contiguous attention."""

from __future__ import annotations

from dataclasses import replace

import torch
from b12x._lib.scratch import require_disjoint
from b12x.preparation import Plan
from b12x.preparation.types import plan_from_handle, require_prepared

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import (
    BatchedBinding,
    VarlenBinding,
    bind,
    bind_batched,
    plan,
    plan_batched,
    run,
    run_batched,
)
from ._tuning import VarlenAttentionConfig, VarlenAttentionQuery


@torch.library.custom_op(
    "b12x::varlen_attention_out", mutates_args=("out", "lse", "scratch", "sink_storage")
)
def _run_out(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    scratch: torch.Tensor,
    sink: torch.Tensor | None,
    sink_storage: torch.Tensor | None,
    block_indices: torch.Tensor | None,
    block_offsets: torch.Tensor | None,
    max_q: int,
    max_k: int,
    softmax_scale: float,
    handle: int,
) -> None:
    operands = (
        q,
        k,
        v,
        cu_q,
        cu_k,
        scratch,
        sink,
        sink_storage,
        block_indices,
        block_offsets,
    )
    require_disjoint(out, (*operands, lse), owner="attention output")
    if lse is not None:
        require_disjoint(lse, operands, owner="attention LSE")
    plan = plan_from_handle(handle)
    state = require_prepared(plan, "attention.varlen", q.device)
    if sink_storage is not None:
        state = replace(state, sink_storage=sink_storage)
    binding = state.bind(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        out=out,
        lse=lse,
        scratch=scratch,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softmax_scale=softmax_scale,
        attention_sink_bias=sink,
        block_indices=block_indices,
        block_offsets=block_offsets,
    )
    state.run(binding)


@_run_out.register_fake
def _run_out_fake(
    q,
    k,
    v,
    cu_q,
    cu_k,
    out,
    lse,
    scratch,
    sink,
    sink_storage,
    block_indices,
    block_offsets,
    max_q,
    max_k,
    softmax_scale,
    handle,
) -> None:
    pass


def run_out(
    q,
    k,
    v,
    cu_seqlens_q,
    *,
    out,
    scratch,
    plan: Plan,
    lse=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    attention_sink_bias=None,
    block_indices=None,
    block_offsets=None,
):
    """Write attention and LSE outputs with caller-owned private workspace.

    When lse is omitted, normalization statistics stay private to scratch and
    the returned LSE is None. The dispatcher operation itself returns None.
    """
    state = require_prepared(plan, "attention.varlen", q.device)
    query = plan.query
    sink_storage = (
        getattr(state, "sink_storage", None)
        if plan.invocation["sink_requires_copy"]
        else None
    )
    _run_out(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k,
        out,
        lse,
        scratch,
        attention_sink_bias,
        sink_storage,
        block_indices,
        block_offsets,
        query.max_seqlen_q if max_seqlen_q is None else max_seqlen_q,
        query.max_seqlen_k if max_seqlen_k is None else max_seqlen_k,
        q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale,
        plan.handle,
    )
    return out, lse


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "BatchedBinding",
    "Plan",
    "VarlenAttentionConfig",
    "VarlenAttentionQuery",
    "VarlenBinding",
    "bind",
    "bind_batched",
    "is_supported",
    "plan",
    "plan_batched",
    "run",
    "run_out",
    "run_batched",
]
