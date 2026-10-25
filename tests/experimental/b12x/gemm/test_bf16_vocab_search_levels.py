"""Host-only search policy and legality checks for BF16 vocabulary projection."""

from dataclasses import replace

import pytest

from b12x.gemm.bf16_vocab_projection._preparation import _compile_query_payload
from b12x.gemm.bf16_vocab_projection._tuning import (
    Bf16VocabProjectionConfig as Config,
    Bf16VocabProjectionQuery as Query,
    TUNING,
)
from b12x.preparation import DeviceIdentity


DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 1),
    sm_count=48,
    product_name="Synthetic SM121",
)


def query(**changes):
    return replace(
        Query(
            dtype="bfloat16",
            max_tokens=1,
            in_features=5120,
            out_features=129280,
            autotune_effort="default",
        ),
        **changes,
    )


def choices(q):
    return {config for _, config in TUNING.choices(q, DEVICE)}


@pytest.mark.parametrize("tokens", [1, 2, 8, 9, 65_535, 65_536])
@pytest.mark.parametrize("k", [256, 5120, 8192, 9216])
def test_nested_search_and_natural_default_membership(tokens, k):
    q = query(max_tokens=tokens, in_features=k)
    default = TUNING.default_config(q, DEVICE)
    previous = {default}
    for effort in ("off", "default", "audit"):
        active = replace(q, autotune_effort=effort)
        space = TUNING.parameter_space(active, DEVICE)
        assignment = TUNING.config_payload(default)
        space.validate(assignment)
        assert assignment in tuple(space.configurations())
        selected = choices(active)
        assert previous <= selected
        if effort == "off":
            assert selected == {default}
        previous = selected


def test_deepseek_single_token_default_keeps_full_declared_space():
    q = query()
    assert len(choices(q)) == 17
    assert choices(q) == choices(replace(q, autotune_effort="audit"))


@pytest.mark.parametrize(
    "changes,config,raced_at",
    [
        (
            {"max_tokens": 2},
            Config(backend="triton", algorithm="row", block_k=8192, num_warps=8),
            ("audit",),
        ),
        (
            {"in_features": 9216},
            Config(backend="triton", algorithm="loop", block_k=1024, num_warps=4),
            ("audit",),
        ),
        # Triton beyond eight tokens is dominated at every effort.
        (
            {"max_tokens": 9},
            Config(backend="triton", algorithm="row", block_k=8192, num_warps=8),
            (),
        ),
    ],
)
def test_policy_filters_do_not_reject_explicit_configs(changes, config, raced_at):
    q = query(**changes)
    TUNING.validate_config(q, config, DEVICE)
    for effort in ("default", "audit"):
        active = replace(q, autotune_effort=effort)
        assert (config in choices(active)) == (effort in raced_at)
        pinned = TUNING.configure(active, device=DEVICE, override=config)
        assert pinned.default == pinned.pinned == config


@pytest.mark.parametrize(
    "changes,config",
    [
        (
            {"in_features": 9216},
            Config(backend="triton", algorithm="row", block_k=16384, num_warps=8),
        ),
        (
            {},
            Config(backend="triton", algorithm="row", block_k=4096, num_warps=8),
        ),
        (
            {},
            Config(backend="triton", algorithm="loop", block_k=2048, num_warps=8),
        ),
        (
            {"max_tokens": 65_536},
            Config(backend="triton", algorithm="row", block_k=8192, num_warps=8),
        ),
    ],
)
def test_audit_preserves_kernel_resource_and_axis_constraints(changes, config):
    q = query(autotune_effort="audit", **changes)
    with pytest.raises(ValueError):
        TUNING.validate_config(q, config, DEVICE)
    assert config not in choices(q)


def test_effort_is_captured_in_selection_but_not_compilation(monkeypatch):
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
    low = Query(dtype="bfloat16", max_tokens=1, in_features=5120, out_features=129280)
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    audit = Query(dtype="bfloat16", max_tokens=1, in_features=5120, out_features=129280)
    assert low.autotune_effort == "default"
    assert audit.autotune_effort == "audit"
    assert TUNING.encode_query(low) != TUNING.encode_query(audit)
    assert _compile_query_payload(low) == _compile_query_payload(audit)
    assert "autotune_effort" not in _compile_query_payload(low)
