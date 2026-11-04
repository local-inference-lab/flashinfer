"""Selected dense MLA splits determine scratch and launch geometry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import dense_mla
from b12x.attention.dense_mla._scratch import plan_dense_mla_scratch
from b12x.attention.dense_mla.planner import choose_num_splits


def _caps(**kwargs):
    return dense_mla.Caps(
        device="cpu",
        mode="decode",
        kv_dtype=torch.bfloat16,
        num_q_heads=64,
        page_size=64,
        max_total_q=1,
        max_batch=1,
        max_cache_tokens=4096,
        max_page_table_width=64,
        num_cache_pages=64,
        **kwargs,
    )


def test_selected_splits_bypass_occupancy_default():
    heuristic = choose_num_splits(
        max_cache_tokens=4096,
        max_total_q=1,
        num_q_heads=64,
        query_tile=1,
        sm_count=148,
        budget=None,
    )
    assert heuristic == 16
    caps = _caps()
    assert plan_dense_mla_scratch(caps).num_splits == 1
    selected = plan_dense_mla_scratch(caps, num_splits=64)
    assert selected.num_splits == 64
    assert selected.chunks_per_split == 1


@pytest.mark.parametrize("window_size", [None, 128])
def test_selected_splits_match_preparation_memory_and_materialization(window_size):
    caps = _caps(window_size=window_size)
    declaration = dense_mla.plan(caps)
    config = dense_mla.DenseMlaConfig(max_splits=64)
    expected = plan_dense_mla_scratch(caps, num_splits=64)
    memory = declaration._memory_requirements(config, None)
    state = declaration._materialize(SimpleNamespace(config=config), None)
    assert state.layout.num_splits == 64
    assert state.layout.scratch_specs() == expected.scratch_specs()
    assert memory.scratch == expected.scratch_specs()


@pytest.mark.parametrize("splits", [0, 65, True, 1.5])
def test_selected_splits_reject_invalid_capacity(splits):
    with pytest.raises(ValueError, match="cache chunk count"):
        plan_dense_mla_scratch(_caps(), num_splits=splits)


@pytest.mark.parametrize(
    "budget,match",
    [
        (dense_mla.Budget(max_splits=16), "split budget"),
        (dense_mla.Budget(max_partial_rows=16), "partial rows"),
    ],
)
def test_selected_splits_respect_caller_budgets(budget, match):
    caps = _caps(budget=budget)
    with pytest.raises(ValueError, match=match):
        plan_dense_mla_scratch(caps, num_splits=64)
    declaration = dense_mla.plan(caps)
    config = dense_mla.DenseMlaConfig(max_splits=64)
    with pytest.raises(ValueError, match=match):
        declaration._memory_requirements(config, None)
    assert (
        plan_dense_mla_scratch(replace(caps, budget=None), num_splits=1).num_splits == 1
    )


@pytest.mark.parametrize(
    "budget,limit",
    [
        (dense_mla.Budget(max_splits=7), 7),
        (dense_mla.Budget(max_partial_rows=56), 2),
        (dense_mla.Budget(max_partial_rows=0), 1),
        (dense_mla.Budget(max_partial_rows=27), 1),
        (dense_mla.Budget(max_splits=7, max_partial_rows=84), 3),
    ],
)
def test_autotuning_enumerates_only_candidates_within_caller_budgets(budget, limit):
    caps = replace(
        _caps(budget=budget),
        max_total_q=28,
        max_batch=28,
        max_cache_tokens=65536,
        max_page_table_width=1024,
    )
    declaration = dense_mla.plan(caps)
    contract = declaration.contract
    configuration = contract.configure(
        declaration.query, device=None, override=None, search=True
    )
    candidates = [
        configuration.default,
        *(config for _, config in contract.iterate(configuration)),
    ]
    assert len(candidates) > 1
    assert max(config.max_splits for config in candidates) == limit
    for config in candidates:
        memory = declaration._memory_requirements(config, None)
        state = declaration._materialize(SimpleNamespace(config=config), None)
        assert memory.scratch == state.layout.scratch_specs()
    payload = declaration.query.to_dict()
    assert dense_mla.DenseMlaQuery.from_dict(payload) == declaration.query
    assert declaration.query != dense_mla.plan(replace(caps, budget=None)).query
    with pytest.raises(ValueError, match="scratch budget"):
        contract.configure(
            declaration.query,
            device=None,
            override=dense_mla.DenseMlaConfig(max_splits=limit + 1),
            search=True,
        )


@torch.inference_mode()
def test_budgeted_long_context_graph_matches_attention_reference():
    from .test_dense_mla import _bind, _prepared_scope
    from ..conftest import require_b12x

    device = require_b12x()
    rows, heads, tokens = 28, 96, 65536
    budget = dense_mla.Budget(max_partial_rows=(64 << 20) // (heads * 513 * 4))
    caps = dense_mla.Caps(
        device=device,
        mode="decode",
        kv_dtype=torch.float8_e4m3fn,
        num_q_heads=heads,
        page_size=128,
        max_total_q=rows,
        max_batch=rows,
        max_cache_tokens=tokens,
        max_page_table_width=tokens // 128,
        num_cache_pages=tokens // 128,
        partial_dtype=torch.float32,
        use_cuda_graph=True,
        budget=budget,
    )
    # Exercise the largest legal partial allocation, including the final reduction.
    plan = dense_mla.plan(
        caps,
        override=dense_mla.DenseMlaConfig(max_splits=budget.max_partial_rows // rows),
    )
    torch.manual_seed(31)
    query = (torch.randn(rows, heads, 576, device=device) * 0.1).to(torch.float8_e4m3fn)
    cache = (torch.randn(tokens // 128, 128, 576, device=device) * 0.1).to(query.dtype)
    pages = torch.arange(tokens // 128, device=device, dtype=torch.int32).repeat(
        rows, 1
    )
    lengths = torch.full((rows,), tokens, device=device, dtype=torch.int32)
    starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
    output = torch.empty(rows, heads, 512, dtype=torch.bfloat16, device=device)
    unit = torch.ones(1, device=device)
    with _prepared_scope():
        binding = _bind(
            plan,
            q=query,
            kv_cache=cache,
            output=output,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=starts,
            sm_scale=192**-0.5,
            kv_scale=unit,
            q_scale=unit,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, lse = dense_mla.run(binding=binding)
        for context in (65536, 8192):
            lengths.fill_(context)
            expected, expected_lse = dense_mla.reference(
                query,
                cache,
                pages,
                lengths,
                starts,
                sm_scale=192**-0.5,
                kv_scale=unit,
                q_scale=unit,
            )
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(
                actual.float(), expected.float(), rtol=0.02, atol=5e-4
            )
            torch.testing.assert_close(lse, expected_lse, rtol=2e-5, atol=2e-5)
        graph.reset()
