"""Host-only WO projection search policy and lowering contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.gemm import wo_projection as wo
from b12x.gemm.wo_projection._preparation import _compile_query_payload
from b12x.gemm.wo_projection._tuning import (
    TUNING,
    WoProjectionConfig,
    WoProjectionQuery,
)
from b12x.preparation import DeviceIdentity, FrozenMapping


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "Synthetic SM120")


def query(**changes):
    return replace(
        WoProjectionQuery(
            dtype="bfloat16",
            max_tokens=1,
            groups=2,
            group_width=4096,
            rank=1024,
            hidden=5120,
            autotune_effort="default",
        ),
        **changes,
    )


def choices(q):
    return {config for _, config in TUNING.choices(q, DEVICE)}


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 2, 8, 9])
@pytest.mark.parametrize("dynamic", [False, True])
def test_nested_efforts_and_natural_heuristic_membership(groups, tokens, dynamic):
    q = query(groups=groups, max_tokens=tokens, dynamic_tokens=dynamic)
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


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 8])
def test_default_search_races_tp_decode_tiles_and_lowering(groups, tokens):
    q = query(groups=groups, max_tokens=tokens)
    assert {config.decode_tile_n for config in choices(q)} == {0, 64, 128}
    assert choices(q) == choices(replace(q, autotune_effort="audit"))
    caps = wo.Caps(
        device="cuda:0",
        max_tokens=tokens,
        groups=groups,
        group_width=4096,
        rank=1024,
        hidden=5120,
    )
    declaration = wo.plan(caps)
    device = SimpleNamespace(identity=DEVICE, ordinal=0)
    for tile in (64, 128):
        config = WoProjectionConfig(decode_tile_n=tile)
        TUNING.validate_config(q, config, DEVICE)
        pinned = TUNING.configure(
            replace(q, autotune_effort="off"), device=DEVICE, override=config
        )
        assert pinned.pinned == pinned.default == config
        jobs = declaration._compile_jobs(config, device)
        fused = next(
            job.args[0]
            for job in jobs
            if job.factory.endswith(":_compile_dense_fused_quant_lowering")
        )
        assert fused["mma_tiler_mn"] == (16, tile)
        assert (fused["m"], fused["n"], fused["k"]) == (tokens, 5120, groups * 1024)


@pytest.mark.parametrize("tile", [64, 128])
def test_packed_weights_preserve_layout_and_selected_tile(tile):
    caps = wo.Caps(
        device="cuda:0",
        max_tokens=1,
        groups=4,
        group_width=4096,
        rank=1024,
        hidden=4096,
    )
    declaration = wo.plan(
        caps,
        invocation={
            "sfb_k_replicated": True,
            "wo_a_tiled": True,
            "wo_b_tiled": True,
        },
    )
    config = WoProjectionConfig(decode_tile_n=tile)
    TUNING.validate_config(declaration.query, config, DEVICE)
    jobs = declaration._compile_jobs(
        config, SimpleNamespace(identity=DEVICE, ordinal=0)
    )
    fused = next(
        job.args[0]
        for job in jobs
        if job.factory.endswith(":_compile_dense_fused_quant_lowering")
    )
    assert fused["b_tile_major"] and fused["sfb_k_replicated"]
    assert fused["mma_tiler_mn"] == (16, tile)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_tokens": 9},
        {"dynamic_tokens": True},
        {"dtype": "float16"},
        {"groups": 1, "rank": 96},
        {"groups": 8, "rank": 48},
    ],
)
def test_audit_preserves_fused_quantization_legality(changes):
    q = query(autotune_effort="audit", **changes)
    assert choices(q) == {WoProjectionConfig()}
    with pytest.raises(ValueError, match="static BF16"):
        TUNING.validate_config(q, WoProjectionConfig(decode_tile_n=64), DEVICE)


@pytest.mark.parametrize(
    "changes",
    [
        {"wo_a_tiled": True},
        {"wo_b_tiled": True},
    ],
)
def test_audit_preserves_packed_weight_geometry(changes):
    with pytest.raises(ValueError, match="tiled WO"):
        TUNING.validate_query(query(autotune_effort="audit", **changes), DEVICE)


@pytest.mark.parametrize("tile", [64, 128])
def test_captured_tile_override_preserves_heuristic_and_conflict_checks(tile):
    q = query(codegen=FrozenMapping({"wo_b_fused_tile": f"16x{tile}"}))
    for effort in ("off", "default", "audit"):
        active = replace(q, autotune_effort=effort)
        default = TUNING.default_config(active, DEVICE)
        assert default in choices(active)
        TUNING.parameter_space(active, DEVICE).validate(TUNING.config_payload(default))
    assert {config.decode_tile_n for config in choices(q)} == {0, tile}
    with pytest.raises(ValueError, match="conflicts"):
        TUNING.validate_config(q, WoProjectionConfig(decode_tile_n=192 - tile), DEVICE)


def test_effort_selection_identity_excludes_compilation_payload(monkeypatch):
    caps = wo.Caps(
        device="cuda:0",
        max_tokens=1,
        groups=2,
        group_width=4096,
        rank=1024,
        hidden=5120,
    )
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
    low_plan = wo.plan(caps)
    low = low_plan.query
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    maximum_plan = wo.plan(caps)
    maximum = maximum_plan.query
    assert low.autotune_effort == "default"
    assert maximum.autotune_effort == "audit"
    assert TUNING.encode_query(low) != TUNING.encode_query(maximum)
    assert _compile_query_payload(low) == _compile_query_payload(maximum)
    assert "autotune_effort" not in _compile_query_payload(low)
    device = SimpleNamespace(identity=DEVICE, ordinal=0)
    config = WoProjectionConfig(decode_tile_n=64)
    assert low_plan._compile_jobs(config, device) == maximum_plan._compile_jobs(
        config, device
    )


def test_unset_effort_defaults_and_invalid_effort_is_rejected(monkeypatch):
    monkeypatch.delenv("B12X_AUTOTUNE_EFFORT", raising=False)
    q = WoProjectionQuery(
        dtype="bfloat16",
        max_tokens=1,
        groups=4,
        group_width=4096,
        rank=1024,
        hidden=5120,
    )
    assert q.autotune_effort == "default"
    assert {config.decode_tile_n for config in choices(q)} == {0, 64, 128}
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "invalid")
    with pytest.raises(ValueError, match="B12X_AUTOTUNE_EFFORT"):
        wo.plan(
            wo.Caps(
                device="cpu",
                max_tokens=1,
                groups=4,
                group_width=4096,
                rank=1024,
                hidden=5120,
            )
        )
