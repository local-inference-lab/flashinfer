"""Compact W4A8 backend and inline-scale choices are raced, not pruned."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping
from tests.experimental.b12x.preparation.test_precision_choices import _nvfp4_query

GB10 = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
RTX = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")


def _query(tokens, effort="default", **changes):
    arguments = dict(
        quant_mode="w4a8_mx",
        quant_modes=("w4a8_mx",),
        source_format="fp4_e8m0_k32",
        activation="silu",
        io_dtype="bfloat16",
        num_experts=384,
        route_num_experts=384,
        hidden_size=5120,
        intermediate_size=576,
        top_k=6,
        num_tokens=tokens,
        routed_rows=tokens * 6,
        csf_inline_scales=True,
        shared_input_scales=False,
        swiglu_limit=10.0,
        autotune_effort=effort,
        controls=FrozenMapping(),
    )
    arguments.update(changes)
    return replace(_nvfp4_query(), **arguments)


def _choices(device, tokens, effort="default"):
    """The production candidate set: every entry has been materialized."""
    plan = TUNING.eligible_plan(_query(tokens, effort), device)
    return {(config.backend, config.csf_inline_scales) for _, config in plan.candidates}


@pytest.mark.parametrize("device", [GB10, RTX])
@pytest.mark.parametrize("effort", ["default", "audit"])
@pytest.mark.parametrize("tokens", [4, 8, 16, 64, 1024, 2048, 4096])
def test_inline_and_expansion_are_both_candidates(device, effort, tokens):
    choices = _choices(device, tokens, effort)
    assert any(inline for _, inline in choices)
    assert any(not inline for _, inline in choices)


@pytest.mark.parametrize("tokens", [4, 8, 16, 64, 1024])
def test_micro_and_dynamic_race_through_1024_tokens(tokens):
    assert {"micro", "dynamic"} <= {backend for backend, _ in _choices(GB10, tokens)}


@pytest.mark.parametrize("tokens", [2048, 4096])
def test_uncapped_micro_is_rejected_past_the_grid_limit(tokens):
    # Direct compact FC2 uses grid.z = tokens * ceil(hidden / 128); without a
    # cluster cap it would exceed CUDA's 65535 limit here.
    assert "micro" not in {backend for backend, _ in _choices(GB10, tokens)}


def test_off_effort_keeps_expansion():
    # `effort=off` runs no race; the heuristic default stays expansion.
    config = TUNING.configure(_query(768, "off"), device=GB10, search=False).default
    assert not config.csf_inline_scales
