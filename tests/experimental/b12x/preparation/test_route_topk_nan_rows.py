"""Route top-k rows without a selectable expert report ID -1 and read in bounds."""

from __future__ import annotations

import pytest
import torch

from b12x.moe._shared import routing

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# name, experts, top_k, score_func, renormalize, correction bias
CASES = [
    ("sigmoid_bias", 288, 8, "sigmoid", True, True),
    ("sigmoid_plain", 288, 8, "sigmoid", False, False),
    ("sqrtsoftplus_bias", 384, 6, "sqrtsoftplus", True, True),
    ("softmax_bias", 256, 8, "softmax", True, True),
    ("softmax_plain", 128, 8, "softmax", True, False),
]


def _route(logits, top_k, score_func, renormalize, bias, valid):
    rows = logits.shape[0]
    selected = torch.full((rows, top_k), 7.0, device=logits.device)
    ids = torch.full((rows, top_k), -5, dtype=torch.int32, device=logits.device)
    weights = torch.full_like(selected, 7.0)
    routing.route_topk(
        logits,
        selected,
        ids,
        weights,
        renormalize=renormalize,
        score_func=score_func,
        correction_bias=bias,
        num_valid_tokens=valid,
        routed_scaling_factor=2.5,
    )
    torch.cuda.synchronize()
    return selected, ids, weights


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("rows", [1, 5, 129])
def test_rows_of_nan_logits_select_nothing(case, rows):
    _, experts, top_k, score_func, renormalize, with_bias = case
    logits = torch.full((rows, experts), float("nan"), device="cuda")
    bias = torch.linspace(-0.5, 0.5, experts, device="cuda") if with_bias else None
    valid = torch.tensor([rows], dtype=torch.int32, device="cuda")
    for _ in range(3):
        selected, ids, weights = _route(
            logits, top_k, score_func, renormalize, bias, valid
        )
        assert bool((ids == -1).all())
        assert bool(torch.isnan(selected).all())
        assert bool((weights == 0).all())


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_rows_with_special_values_are_a_function_of_the_input(case):
    _, experts, top_k, score_func, renormalize, with_bias = case
    rows = 4096
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn(rows, experts, device="cuda", generator=generator) * 3
    flat = logits.view(-1)
    picks = torch.randperm(flat.numel(), device="cuda", generator=generator)[
        : flat.numel() // 16
    ]
    specials = torch.tensor([float("nan"), float("inf"), -float("inf")], device="cuda")
    flat[picks] = specials[torch.arange(picks.numel(), device="cuda") % 3]
    bias = torch.linspace(-0.5, 0.5, experts, device="cuda") if with_bias else None
    first = _route(logits, top_k, score_func, renormalize, bias, None)
    assert bool(((first[1] >= -1) & (first[1] < experts)).all())
    unfilled = first[1] < 0
    assert bool(torch.isnan(first[0][unfilled]).all())
    assert bool((first[2][unfilled] == 0).all())
    for _ in range(4):
        again = _route(logits, top_k, score_func, renormalize, bias, None)
        for got, want in zip(again, first):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("renormalize", [True, False])
def test_partially_filled_rows_keep_the_softmax_of_the_filled_slots(renormalize):
    """More than ``num_experts - top_k`` NaN logits leave some slots unfilled.

    The filled slots hold the finite logits, largest first, and their weights
    are the softmax over those logits alone. The unfilled slots report ID -1, a
    NaN logit and zero weight.
    """
    experts, top_k, rows = 128, 8, 64
    finite = experts - (experts - top_k + 1)
    assert finite == top_k - 1
    generator = torch.Generator(device="cuda").manual_seed(11)
    logits = torch.full((rows, experts), float("nan"), device="cuda")
    values = torch.randn(rows, finite, device="cuda", generator=generator) * 2
    for row in range(rows):
        positions = torch.randperm(experts, device="cuda", generator=generator)[:finite]
        logits[row, positions] = values[row]
    selected, ids, weights = _route(logits, top_k, "softmax", renormalize, None, None)
    assert bool((ids[:, :finite] >= 0).all())
    assert bool((ids[:, finite:] == -1).all())
    ordered = values.sort(dim=1, descending=True).values
    assert torch.equal(selected[:, :finite], ordered)
    assert bool(torch.isnan(selected[:, finite:]).all())
    assert bool((weights[:, finite:] == 0).all())
    if renormalize:
        expected = torch.softmax(ordered, dim=1) * 2.5
    else:
        expected = ordered * 2.5
    torch.testing.assert_close(weights[:, :finite], expected, rtol=1e-5, atol=1e-6)
