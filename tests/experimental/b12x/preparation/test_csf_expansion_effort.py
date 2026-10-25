"""Host-only compact scale expansion search and compilation metadata checks."""

from contextlib import nullcontext
from dataclasses import asdict, replace

import pytest

from b12x._lib.quant import csf_blocks_tuning as expansion


def query(routes, effort="default", experts=384):
    first = (4608, 160, 1, 9, 160, 4608, experts, 1, 1)
    second = (5120, 72, 1, 9, 72, 5120, experts, 1, 1)
    return expansion.CsfExpansionQuery(
        first, second, routes, "int32", autotune_effort=effort
    )


@pytest.mark.parametrize("routes", (6, 63, 64, 384, 390, 1536, 6138, 6144))
def test_nested_candidates_and_heuristic_default(routes):
    candidates = []
    for effort in ("off", "default", "audit"):
        q = query(routes, effort)
        configuration = expansion.TUNING.configure(q, device=None)
        default = configuration.default
        payload = expansion.TUNING.config_payload(default)
        configuration.space.validate(payload)
        configs = {config for _, config in expansion.TUNING.iterate(configuration)}
        assert default in configs
        assert not any(c.direct_routes and c.bitmap_routes for c in configs)
        if effort == "off":
            assert configs == {default}
        candidates.append(configs)
    assert candidates[0] <= candidates[1] == candidates[2]
    if 384 < routes < 6144:
        assert len(candidates[1]) == 28
    elif 64 <= routes <= 384:
        assert len(candidates[1]) == 42
    else:
        assert len(candidates[1]) == 14


def test_above_expert_count_direct_routes_are_never_raced_but_can_be_pinned():
    config = expansion.CsfExpansionConfig(direct_routes=True)
    payload = asdict(config)
    for effort in ("off", "default", "audit"):
        q = query(390, effort)
        expansion.TUNING.validate_config(q, config, None)
        with pytest.raises(ValueError, match="correctness"):
            expansion.TUNING.parameter_space(q, None).validate(payload)
        assert expansion.TUNING.configure(q, device=None, override=config).pinned == (
            config
        )


@pytest.mark.parametrize("routes", (63, 6144))
@pytest.mark.parametrize("switch", ("direct_routes", "bitmap_routes"))
def test_audit_preserves_fixed_route_policy(routes, switch):
    q = query(routes, "audit")
    config = expansion.CsfExpansionConfig(**{switch: True})
    with pytest.raises(ValueError, match="presence-routed"):
        expansion.TUNING.validate_config(q, config, None)
    with pytest.raises(ValueError):
        expansion.TUNING.lower(q, None, asdict(config))


def test_audit_rejects_direct_bitmap_combination():
    q = query(390, "audit")
    config = expansion.CsfExpansionConfig(direct_routes=True, bitmap_routes=True)
    with pytest.raises(ValueError, match="do not use"):
        expansion.TUNING.validate_config(q, config, None)
    with pytest.raises(ValueError, match="eligible values"):
        expansion.TUNING.parameter_space(q, None).validate(asdict(config))


def test_effort_captured_by_query_and_excluded_from_program_metadata(monkeypatch):
    base = query(390)
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    captured = expansion.CsfExpansionQuery(
        base.first, base.second, base.routes, base.ids_dtype
    )
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "off")
    assert captured.autotune_effort == "audit"
    assert len(expansion.TUNING.eligible_plan(captured, None).candidates) == 28
    assert len(expansion.TUNING.eligible_plan(base, None).candidates) == 28
    assert expansion.TUNING.encode_query(captured) != expansion.TUNING.encode_query(
        base
    )
    program_payload = expansion._program_query_payload(base)
    assert "autotune_effort" not in program_payload
    assert program_payload == expansion._program_query_payload(captured)

    calls = []
    monkeypatch.setattr(expansion.torch.cuda, "device", lambda ordinal: nullcontext())
    monkeypatch.setattr(
        expansion, "compile_nvfp4_csf_pair", lambda *args: calls.append(args)
    )
    monkeypatch.setattr(
        expansion, "compile_csf_active_experts", lambda *args: calls.append(args)
    )
    config = asdict(expansion.CsfExpansionConfig())
    for effort in ("off", "default", "audit"):
        expansion.compile_expansion._function(
            expansion._program_query_payload(replace(base, autotune_effort=effort)),
            config,
            0,
        )
    assert len(calls) == 12
    assert calls[:4] == calls[4:8] == calls[8:]
