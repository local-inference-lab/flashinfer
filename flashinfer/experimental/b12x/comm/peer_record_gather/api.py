"""Experimental prepared gathering from addressable peer record allocations.

``peer_bases`` contains one device address per peer, each covering at least
``peer_records * record_bytes`` bytes. The caller retains the allocations and
ensures peer access and producer completion before launch. Peer storage must
not overlap the output. Each row reads its
first ``counts[row]`` selections. Negative or out-of-range owner/slot entries,
null peer bases, and inactive selections produce zero bytes without loading
peer storage. FP8 outputs preserve bytes without numeric conversion.
"""

from dataclasses import dataclass

import torch

from b12x._lib.gating import has_triton
from b12x.preparation import Plan, plan_from_handle, require_prepared
from ._preparation import PeerRecordGatherQuery, plan
from ._stage import InterleavedStageQuery, plan_interleaved_stage


def is_supported(device=None):
    del device
    return has_triton()


@torch.library.custom_op("b12x::peer_record_gather", mutates_args=("out",))
def _gather(
    handle: int,
    peer_bases: torch.Tensor,
    owners: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "comm.peer_record_gather", out.device
    )
    state.run(peer_bases, owners, slots, counts, out)


@_gather.register_fake
def _fake(handle, peer_bases, owners, slots, counts, out):
    return None


def gather(plan: Plan, peer_bases, owners, physical_slots, counts, out):
    _gather(plan.handle, peer_bases, owners, physical_slots, counts, out)
    return out


@torch.library.custom_op("b12x::stage_interleaved_records", mutates_args=("out",))
def _stage_interleaved_records(
    handle: int,
    peer_bases: torch.Tensor,
    table: torch.Tensor,
    end: torch.Tensor,
    tokens: torch.Tensor,
    slots: torch.Tensor,
    counts: torch.Tensor,
    out: torch.Tensor,
    first: bool,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "comm.interleaved_record_stage", out.device
    )
    state.run(peer_bases, table, end, tokens, slots, counts, out, first)


@_stage_interleaved_records.register_fake
def _fake_stage(handle, peer_bases, table, end, tokens, slots, counts, out, first):
    pass


def stage_interleaved_records(
    plan, peer_bases, table, end, tokens, slots, counts, out, *, first
):
    """Stage a visible prefix once if it fits in the caller's slab.

    Peer p owns logical tokens p, p+world, ... in pages addressed by table.
    For end <= capacity the first call populates rows [0,end); subsequent
    slices reuse those bytes. Otherwise every call populates its selection in
    row-major order, using the peer identified by each token and its physical
    slot. Invalid selections or missing pages read as zeros. The consumer uses
    counts to mask padding. Peer bases must be word-aligned. The caller orders
    peer writes before staging and attention reads before reusing the slab.
    """
    _stage_interleaved_records(
        plan.handle, peer_bases, table, end, tokens, slots, counts, out, first
    )
    return out


@dataclass(frozen=True)
class Binding:
    plan: Plan

    def run(self, peer_bases, owners, physical_slots, counts, out):
        return gather(self.plan, peer_bases, owners, physical_slots, counts, out)

    __call__ = run


def bind(plan: Plan):
    require_prepared(plan, "comm.peer_record_gather")
    return Binding(plan)


__all__ = [
    "PeerRecordGatherQuery",
    "plan",
    "bind",
    "gather",
    "is_supported",
    "InterleavedStageQuery",
    "plan_interleaved_stage",
    "stage_interleaved_records",
]
