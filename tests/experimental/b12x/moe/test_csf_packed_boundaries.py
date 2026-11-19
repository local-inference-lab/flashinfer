"""Packed CSF execution preserves live capacities and standalone route paths."""

from dataclasses import replace

import pytest
import torch

from b12x._lib.quant.csf_blocks_tuning import (
    CsfExpansionConfig,
    plan_csf_block_expansion,
)
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from .test_mxfp4_csf import _prepare_arms


@pytest.mark.parametrize(
    "kind", ["fused", "mapped", "prefetched", "prefetched_after_bind", "large"]
)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_csf_live_and_fallback_graphs(kind, dtype, monkeypatch):
    _check_csf_graphs(kind, dtype, monkeypatch)


@pytest.mark.parametrize(
    "route_mode,dtype",
    [("direct", torch.int32), ("packed", torch.int32), ("packed", torch.int64)],
)
@pytest.mark.parametrize("expansion_plan", [False, True])
def test_compiled_mapped_csf_graphs(route_mode, dtype, expansion_plan, monkeypatch):
    torch.compiler.reset()
    _check_csf_graphs(
        "mapped",
        dtype,
        monkeypatch,
        route_mode=route_mode,
        compiled=True,
        expansion_plan=expansion_plan,
    )


@pytest.mark.parametrize(
    "route_mode,dtype",
    [("direct", torch.int32), ("packed", torch.int32), ("packed", torch.int64)],
)
def test_compiled_csf_global_route_namespace(route_mode, dtype, monkeypatch):
    torch.compiler.reset()
    _check_csf_graphs(
        "mapped",
        dtype,
        monkeypatch,
        route_mode=route_mode,
        compiled=True,
        expansion_plan=True,
        route_experts=16,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("routing", ["direct", "scan", "bitmap"])
def test_frozen_csf_first_mapped_binding(dtype, routing, monkeypatch):
    torch.compiler.reset()
    _check_csf_graphs(
        "mapped",
        dtype,
        monkeypatch,
        compiled=True,
        expansion_plan=True,
        prepare_mapped=False,
        compile_workers=3,
        token_capacity=4 if routing == "direct" else 33,
        bitmap_routes=routing == "bitmap",
    )


@pytest.mark.parametrize("compiled", [False, True])
def test_mapped_staged_mxfp4_matches_whole_expert_and_clears_unrouted_outputs(
    compiled, monkeypatch
):
    _check_csf_graphs(
        "mapped",
        torch.int32,
        monkeypatch,
        compiled=compiled,
        route_experts=16,
        staged=True,
    )


def _check_csf_graphs(
    kind,
    dtype,
    monkeypatch,
    *,
    route_mode="packed",
    compiled=False,
    expansion_plan=False,
    route_experts=None,
    prepare_mapped=True,
    compile_workers=0,
    token_capacity=None,
    bitmap_routes=False,
    staged=False,
):
    device = require_b12x()
    capacity = (
        token_capacity if token_capacity is not None else 65 if kind == "large" else 4
    )
    experts, topk, hidden = 8, 2, 512
    route_experts = experts if route_experts is None else route_experts
    route_offset = experts if route_experts > experts else 0
    (native, compressed), scales = _prepare_arms(
        hidden,
        192,
        "w31",
        monkeypatch,
        activation_mode="a16",
        packing="mma_packed" if staged else "source_native",
    )
    decoder = compressed._impl.nvfp4_csf
    if kind in ("prefetched", "prefetched_after_bind"):
        assert moe.expand_scales(compressed)
    expansion = (
        plan_csf_block_expansion(
            decoder.first,
            decoder.second,
            routes=capacity * topk,
            ids_dtype=dtype,
            override=CsfExpansionConfig(groups_per_task=4, bitmap_routes=bitmap_routes),
        )
        if expansion_plan
        else None
    )
    config = moe.MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode=route_mode,
        w4a16_tile_config=(128, 64, 64, 128) if route_mode == "packed" else None,
        w4a16_block_size_m=16 if route_mode == "packed" else None,
        w4a16_pipeline_stages=3 if route_mode == "packed" else None,
        csf_inline_scales=False,
    )
    plans = tuple(
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(
                max_tokens=capacity, top_k=topk, route_num_experts=route_experts
            ),
            invocation={"fast_math": False},
            override=replace(config, w4a16_packed_schedule="split")
            if staged and owner is compressed
            else config,
            scale_expansion=expansion if owner is compressed else None,
        )
        for owner in (native, compressed)
    )
    source = torch.randn(capacity, hidden, device=device, dtype=torch.bfloat16) * 0.1
    ids = (
        torch.arange(capacity * topk, device=device, dtype=dtype)
        .reshape(capacity, topk)
        .add(route_offset)
        % route_experts
    )
    weights = torch.full((capacity, topk), 0.5, device=device)
    mapping = (
        torch.arange(route_experts - 1, -1, -1, dtype=torch.int32, device=device)
        % experts
        if kind == "mapped"
        else None
    )

    if mapping is not None:
        mapping[route_offset] = -1

    def bind(state, rows, plan=None, mapped=True):
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source[:rows])
        bind_call = (
            state.bind if plan is None else lambda **kwargs: moe.bind(plan, **kwargs)
        )
        binding = bind_call(
            a=source[:rows],
            topk_ids=ids[:rows],
            topk_weights=weights[:rows],
            output=output,
            scratch=scratch,
            input_scales_static=True,
            route_expert_map=mapping if mapped else None,
            scales_expanded=kind == "prefetched",
        )
        if kind == "prefetched_after_bind":
            binding = replace(binding, scales_expanded=True)
        return binding, output, scratch

    def prepare(state):
        binding, output, scratch = bind(state, capacity, mapped=prepare_mapped)
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    def prepare_scales(state):
        active = torch.empty(experts, dtype=torch.int32, device=device)
        binding = state.bind(
            ids=ids,
            out13=compressed._impl.w1_blockscale,
            out2=compressed._impl.w2_blockscale,
            scratch=active,
        )
        return PreparedCall(run=binding.run, owners=(binding, active))

    with PreparationSession(
        device=device, autotune=compile_workers > 0, compile_workers=compile_workers
    ) as session:
        requests = tuple(
            plan.request(
                name=f"packed-csf-boundary-{i}",
                prepare_call=prepare,
                dependencies=("mapped-scales",)
                if expansion is not None and i == 1
                else (),
            )
            for i, plan in enumerate(plans)
        )
        if expansion is not None:
            requests = (
                expansion.request(name="mapped-scales", prepare_call=prepare_scales),
                *requests,
            )
        session.prepare(requests)
        session.freeze()
        for rows in (capacity,) if route_mode == "direct" else (1, 3, capacity):
            bindings = tuple(
                bind(plan.prepared.state, rows, plan if compiled else None)
                for plan in plans
            )
            assert getattr(bindings[1][0], "csf_route_pack", False) == (
                kind in ("fused", "prefetched_after_bind")
            )
            graphs = []
            for binding, _, _ in bindings:

                def run(binding=binding):
                    return moe.run(binding=binding)

                forward = torch.compile(run, fullgraph=True) if compiled else run
                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    forward()
                graphs.append(graph)
            for pattern in (0, 1, 2):
                ids.copy_(
                    torch.arange(capacity * topk, device=device, dtype=dtype)
                    .reshape(capacity, topk)
                    .add(route_offset)
                    % route_experts
                )
                if pattern == 1:
                    ids[0, 0] = -1
                    ids[rows - 1, -1] = route_experts
                    if mapping is not None:
                        mapping[route_offset + 1 : route_offset + 3] = 5
                        mapping[-1] = experts
                elif pattern == 2:
                    ids.fill_(-1)
                    if mapping is not None:
                        mapping.fill_(-1)
                elif mapping is not None:
                    mapping.copy_(
                        torch.arange(
                            route_experts - 1, -1, -1, dtype=torch.int32, device=device
                        )
                        % experts
                    )
                    mapping[route_offset] = -1
                source.neg_()
                for scale in scales[0]:
                    scale.view(torch.uint8).fill_(0xFF)
                if kind in ("prefetched", "prefetched_after_bind"):
                    assert moe.expand_scales(compressed)
                for binding, output, _ in bindings:
                    output.fill_(float("nan"))
                    for buffer in (
                        binding.packed_route_indices,
                        binding.block_expert_ids,
                        binding.packed_route_count,
                    ):
                        buffer.fill_(-42)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                for graph in graphs:
                    graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                torch.testing.assert_close(
                    bindings[0][1], bindings[1][1], rtol=0, atol=0
                )
                if pattern == 0:
                    assert torch.isfinite(bindings[0][1]).all()
                    assert torch.count_nonzero(bindings[0][1]).item() > 0
                if pattern == 2:
                    assert torch.count_nonzero(bindings[1][1]).item() == 0
            for graph in graphs:
                graph.reset()
