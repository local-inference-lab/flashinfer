"""Experimental prepared selection over FP32 scores and int32 candidate IDs.

Only the first ``lengths[row]`` candidates participate. Missing output entries
are ``(-inf, -1)``. Candidate IDs are gathered without interpretation, including
negative sentinels. The caller owns every input and output allocation.
"""

from dataclasses import dataclass

import torch

from b12x._lib.gating import has_cutlass_dsl
from b12x.preparation import Plan, plan_from_handle, require_prepared
from ._preparation import RowTopKQuery, plan


def is_supported(device=None):
    del device
    return has_cutlass_dsl()


@torch.library.custom_op("b12x::row_topk", mutates_args=("out_values", "out_indices"))
def _topk(
    handle: int,
    scores: torch.Tensor,
    lengths: torch.Tensor,
    gather_table: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "attention.row_topk", scores.device
    )
    state.run(scores, lengths, gather_table, out_values, out_indices)


@_topk.register_fake
def _fake(handle, scores, lengths, gather_table, out_values, out_indices):
    return None


def topk(plan: Plan, scores, lengths, gather_table, out_values, out_indices):
    _topk(plan.handle, scores, lengths, gather_table, out_values, out_indices)
    return out_values, out_indices


@dataclass(frozen=True)
class Binding:
    plan: Plan

    def run(self, scores, lengths, gather_table, out_values, out_indices):
        return topk(self.plan, scores, lengths, gather_table, out_values, out_indices)

    __call__ = run


def bind(plan: Plan):
    require_prepared(plan, "attention.row_topk")
    return Binding(plan)


__all__ = ["RowTopKQuery", "plan", "bind", "topk", "is_supported"]
