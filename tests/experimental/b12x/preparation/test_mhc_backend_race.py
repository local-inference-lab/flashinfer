"""The mHC native/TF32 crossover is measured, not pruned by a token count."""

import pytest

from b12x.norm.mhc import _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping

GB10 = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
RTX = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")


def query(tokens, effort="default"):
    return _tuning.MhcQuery(
        dtype="bfloat16",
        max_tokens=tokens,
        hidden_size=5120,
        split_k=5120 // 64,
        operation="post_pre",
        has_norm_weight=True,
        rms_eps=1e-20,
        norm_eps=1e-20,
        smem_limit=100 << 10,
        controls=FrozenMapping({"B12X_AUTOTUNE_EFFORT": effort}),
    )


def backends(device, tokens, effort):
    space = _tuning.TUNING.parameter_space(query(tokens, effort), device)
    seen = set()
    for assignment in space.configurations():
        seen.add(assignment["backend"])
        if {"native", "tf32_tma"} <= seen:
            break
    return seen


@pytest.mark.parametrize("device", [GB10, RTX])
@pytest.mark.parametrize("effort", ["default", "audit"])
@pytest.mark.parametrize("tokens", [4, 8, 16, 32, 51, 52, 64, 128, 256, 1024])
def test_both_backends_are_candidates_at_every_token_count(device, effort, tokens):
    assert {"native", "tf32_tma"} <= backends(device, tokens, effort)


def test_off_effort_keeps_the_static_fallback():
    # `effort=off` runs no race, so one constant decides; it is not a device rule.
    assert _tuning._projection_default_config(query(51, "off"), GB10).backend == "native"
    assert _tuning._projection_default_config(query(52, "off"), GB10).backend == "tf32_tma"
    assert _tuning._projection_default_config(query(51, "off"), RTX).backend == "native"
    assert _tuning._projection_default_config(query(52, "off"), RTX).backend == "tf32_tma"
