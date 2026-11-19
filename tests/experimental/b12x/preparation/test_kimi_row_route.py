"""A complete four-row route contributes both gather and selection to an unpinned race."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch
from b12x.comm.pcie import _preparation
from b12x.comm.pcie._collective_tuning import TUNING
from b12x.comm.pcie._tuning import TUNING as NATIVE
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A, _check_four_row_route
from b12x.preparation import DeviceIdentity


def make_route():
    owner = SimpleNamespace(
        rank=0,
        world_size=16,
        device=torch.device("cpu"),
        max_batch_size=8,
        total_heads=16,
        head_dim=8,
        query_head_dim=672,
        _signal_ptrs=tuple(range(16)),
        _slot_bytes=172544,
        _exchange_chunks=0,
        _exchange_offset=0,
        _exchange_counter_offset=345088,
        _threads_override=0,
        _block_limit_override=0,
    )
    owner._resolve_launch_config = MethodType(PCIeDCPA2A._resolve_launch_config, owner)
    call = dict(
        local_down=torch.empty(4, 224, dtype=torch.bfloat16),
        local_router=torch.empty(4, 56),
        correction_bias=torch.empty(896),
        out_down=torch.empty(4, 3584, dtype=torch.bfloat16),
        out_logits=torch.empty(4, 896),
        topk_weights=torch.empty(4, 16),
        topk_ids=torch.empty(4, 16, dtype=torch.int32),
    )
    query = _preparation.query_from_runtime(
        owner, surface="DcpAllToAll.all_gather_pair_kimi_topk", call=call
    )
    return owner, call, query


def test_unpinned_four_row_route_keeps_complete_control_and_device_eligibility():
    owner, call, query = make_route()
    plan = _preparation.plan(query, runtime=owner)
    assert plan.query.default.implementation == "dcp_route_separate"
    for capability, expected in (
        ((12, 0), {"dcp_route_row": 4, "dcp_route_separate": 1}),
        ((9, 0), {"dcp_route_separate": 1}),
    ):
        configs = TUNING.configure(
            plan.query,
            device=DeviceIdentity("nvidia", capability, 188, "test"),
            override=None,
        )
        assert {
            cfg.implementation: cfg.blocks for _, cfg in TUNING.iterate(configs)
        } == expected
    owner._threads_override = 128
    limited = _preparation.query_from_runtime(owner, surface=query.surface, call=call)
    assert limited.call["threads"] == 128
    assert set(_preparation.plan(limited, runtime=owner).query.alternatives) == {
        "dcp_route_separate"
    }
    for world in (2, 4, 8):
        with pytest.raises(ValueError, match="sixteen-rank"):
            NATIVE.validate_query(replace(query, world_size=world), None)


def test_partial_output_alias_and_unaligned_views_are_rejected():
    owner, call, _ = make_route()
    vals = list(call.values())
    vals[5] = vals[4].flatten()[8:72].view(4, 16)
    with pytest.raises(ValueError, match="disjoint"):
        _check_four_row_route(owner, *vals)
    vals = list(call.values())
    vals[0] = torch.empty(4 * 224 + 1, dtype=torch.bfloat16)[1:].view(4, 224)
    with pytest.raises(ValueError, match="aligned"):
        _check_four_row_route(owner, *vals)
