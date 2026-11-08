"""Every route top-k launch geometry reproduces the original program."""

from __future__ import annotations

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe import _tuning
from b12x.preparation import PreparationSession, PreparedCall

CASES = [
    # name, experts, top_k, score_func, renormalize, bias, scale, image bias
    ("glm", 288, 8, "sigmoid", True, True, 2.5, False),
    ("glm_image", 288, 8, "sigmoid", True, True, 2.5, True),
    ("sigmoid_plain", 288, 8, "sigmoid", False, False, 1.0, False),
    ("sqrt", 384, 6, "sqrtsoftplus", True, True, 1.5, False),
    ("softmax_plain", 128, 8, "softmax", True, False, 1.0, False),
    ("softmax_bias", 256, 8, "softmax", True, True, 1.0, False),
]


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def make_logits(rows, experts, device, kind):
    generator = torch.Generator(device=device).manual_seed(7)
    logits = torch.randn(rows, experts, device=device, generator=generator) * 3
    if kind == "ties":
        # Quantize so many experts tie exactly, including at the selection boundary.
        logits = (logits * 2).round() / 2
    elif kind == "constant":
        logits = torch.zeros_like(logits)
    elif kind == "special":
        flat = logits.view(-1)
        picks = torch.randperm(flat.numel(), device=device, generator=generator)[
            : flat.numel() // 16
        ]
        values = torch.tensor(
            [float("nan"), float("inf"), -float("inf")], device=device
        )
        flat[picks] = values[torch.arange(picks.numel(), device=device) % 3]
    return logits


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("rows", [1, 5, 129, 4096])
def test_all_geometries_match_the_original_bitwise(case, rows):
    device = require_cuda()
    _, experts, top_k, score_func, renormalize, with_bias, scale, image = case
    invocation = fused_moe.RouteTopKInvocation(
        num_tokens=rows,
        num_experts=experts,
        top_k=top_k,
        logits_dtype="float32",
        score_func=score_func,
        renormalize=renormalize,
        has_correction_bias=with_bias,
        has_image_correction_bias=image,
        has_image_mask=image,
        has_valid_tokens=True,
        routed_scaling_factor=scale,
    )
    query = invocation.query()
    configs = [
        _tuning.MoeRouteConfig(backend="triton", **dict(choice))
        for choice in map(
            lambda c: {k: v for k, v in c.items() if k != "backend"},
            _tuning.ROUTE_TUNING.parameter_space(query, None).configurations(),
        )
    ]
    default = _tuning.ROUTE_TUNING.default_config(query, None)
    assert default in configs
    bias = torch.linspace(-0.5, 0.5, experts, device=device) if with_bias else None
    options = {} if bias is None else {"correction_bias": bias}
    if image:
        options["image_correction_bias"] = torch.linspace(
            0.5, -0.5, experts, device=device
        ).flip(0)
        options["image_mask"] = torch.arange(rows, device=device) % 2 == 1
    live = max(rows - rows // 3, 0)
    count = torch.tensor([live], dtype=torch.int32, device=device)
    for kind in ("random", "ties", "constant", "special"):
        logits = make_logits(rows, experts, device, kind)
        results = {}
        for config in configs:
            plan = fused_moe.plan_route_topk(invocation, override=config)
            selected = torch.empty(rows, top_k, device=device)
            ids = torch.empty(rows, top_k, dtype=torch.int32, device=device)
            weights = torch.empty_like(selected)
            with PreparationSession(
                device=device, autotune=False, compile_workers=0
            ) as session:
                session.prepare(
                    (
                        plan.request(
                            name="route",
                            prepare_call=lambda state: PreparedCall(
                                run=lambda: state.run(
                                    logits,
                                    selected,
                                    ids,
                                    weights,
                                    num_valid_tokens=count,
                                    **options,
                                )
                            ),
                        ),
                    )
                )
                selected.fill_(7.0)
                ids.fill_(-5)
                weights.fill_(7.0)
                plan.prepared.state.run(
                    logits, selected, ids, weights, num_valid_tokens=count, **options
                )
                torch.cuda.synchronize()
            results[config] = (
                ids.clone(),
                selected.view(torch.int32).clone(),
                weights.view(torch.int32).clone(),
            )
        reference = results[default]
        assert bool((reference[0][live:] == -1).all())
        for config, value in results.items():
            for got, want, name in zip(value, reference, ("ids", "logits", "weights")):
                assert torch.equal(got, want), (kind, config, name)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("rows", [1, 5, 129])
def test_rows_of_nan_logits_select_nothing_in_every_geometry(case, rows):
    """A row whose scores are all NaN reports ID -1, NaN logits and zero weights.

    No expert is selectable, so the gather must not read outside the row; the
    result is the same on every launch and in every launch geometry.
    """
    device = require_cuda()
    _, experts, top_k, score_func, renormalize, with_bias, scale, image = case
    invocation = fused_moe.RouteTopKInvocation(
        num_tokens=rows,
        num_experts=experts,
        top_k=top_k,
        logits_dtype="float32",
        score_func=score_func,
        renormalize=renormalize,
        has_correction_bias=with_bias,
        has_image_correction_bias=image,
        has_image_mask=image,
        has_valid_tokens=True,
        routed_scaling_factor=scale,
    )
    query = invocation.query()
    configs = [
        _tuning.MoeRouteConfig(backend="triton", **choice)
        for choice in (
            {k: v for k, v in c.items() if k != "backend"}
            for c in _tuning.ROUTE_TUNING.parameter_space(query, None).configurations()
        )
    ]
    options = {}
    if with_bias:
        options["correction_bias"] = torch.linspace(-0.5, 0.5, experts, device=device)
    if image:
        options["image_correction_bias"] = torch.linspace(
            0.5, -0.5, experts, device=device
        )
        options["image_mask"] = torch.arange(rows, device=device) % 2 == 1
    count = torch.tensor([rows], dtype=torch.int32, device=device)
    logits = torch.full((rows, experts), float("nan"), device=device)
    for config in configs:
        plan = fused_moe.plan_route_topk(invocation, override=config)
        selected = torch.empty(rows, top_k, device=device)
        ids = torch.empty(rows, top_k, dtype=torch.int32, device=device)
        weights = torch.empty_like(selected)
        with PreparationSession(
            device=device, autotune=False, compile_workers=0
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="route",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(
                                logits,
                                selected,
                                ids,
                                weights,
                                num_valid_tokens=count,
                                **options,
                            )
                        ),
                    ),
                )
            )
            for _ in range(3):
                selected.fill_(7.0)
                ids.fill_(-5)
                weights.fill_(7.0)
                plan.prepared.state.run(
                    logits, selected, ids, weights, num_valid_tokens=count, **options
                )
                torch.cuda.synchronize()
                assert bool((ids == -1).all()), config
                assert bool(torch.isnan(selected).all()), config
                assert bool((weights == 0).all()), config


def test_partially_filled_softmax_rows_match_the_filled_reference_in_every_geometry():
    """More than ``num_experts - top_k`` NaN logits leave the last slots unfilled.

    The filled slots hold the finite logits, largest first, and their weights
    are the softmax over those logits alone; the unfilled slots report ID -1, a
    NaN logit and zero weight.
    """
    device = require_cuda()
    experts, top_k, rows = 128, 8, 129
    finite = top_k - 1
    invocation = fused_moe.RouteTopKInvocation(
        num_tokens=rows,
        num_experts=experts,
        top_k=top_k,
        logits_dtype="float32",
        score_func="softmax",
        renormalize=True,
        has_correction_bias=False,
        has_image_correction_bias=False,
        has_image_mask=False,
        has_valid_tokens=True,
        routed_scaling_factor=1.0,
    )
    query = invocation.query()
    configs = [
        _tuning.MoeRouteConfig(backend="triton", **choice)
        for choice in (
            {k: v for k, v in c.items() if k != "backend"}
            for c in _tuning.ROUTE_TUNING.parameter_space(query, None).configurations()
        )
    ]
    assert len(configs) > 1
    generator = torch.Generator(device=device).manual_seed(11)
    values = torch.randn(rows, finite, device=device, generator=generator) * 2
    logits = torch.full((rows, experts), float("nan"), device=device)
    for row in range(rows):
        positions = torch.randperm(experts, device=device, generator=generator)[:finite]
        logits[row, positions] = values[row]
    ordered = values.sort(dim=1, descending=True).values
    expected = torch.softmax(ordered, dim=1)
    count = torch.tensor([rows], dtype=torch.int32, device=device)
    for config in configs:
        plan = fused_moe.plan_route_topk(invocation, override=config)
        selected = torch.empty(rows, top_k, device=device)
        ids = torch.empty(rows, top_k, dtype=torch.int32, device=device)
        weights = torch.empty_like(selected)
        with PreparationSession(
            device=device, autotune=False, compile_workers=0
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="route",
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(
                                logits,
                                selected,
                                ids,
                                weights,
                                num_valid_tokens=count,
                            )
                        ),
                    ),
                )
            )
            plan.prepared.state.run(
                logits, selected, ids, weights, num_valid_tokens=count
            )
            torch.cuda.synchronize()
        assert bool((ids[:, :finite] >= 0).all()), config
        assert bool((ids[:, finite:] == -1).all()), config
        assert torch.equal(selected[:, :finite], ordered), config
        assert bool(torch.isnan(selected[:, finite:]).all()), config
        assert bool((weights[:, finite:] == 0).all()), config
        torch.testing.assert_close(
            weights[:, :finite], expected, rtol=1e-5, atol=1e-6, msg=str(config)
        )
