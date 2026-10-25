"""The audit effort lifts efficiency rules while retaining kernel legality."""

from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity, FrozenMapping, Knob, ParameterSpace


def test_audit_lifts_only_efficiency_predicates():
    space = ParameterSpace.create(
        (Knob(name="tile", values=(8, 16, 32)),),
        predicates=(lambda p: p["tile"] >= 16,),
        efficiency_predicates=(lambda p: p["tile"] <= 16,),
    )
    audit = replace(space, autotune_effort="audit")
    assert [p["tile"] for p in space.configurations()] == [16]
    assert [p["tile"] for p in audit.configurations()] == [16, 32]
    with pytest.raises(ValueError, match="efficiency predicates"):
        space.validate({"tile": 32})
    audit.validate({"tile": 32})
    for active in (space, audit):
        with pytest.raises(ValueError, match="correctness predicates"):
            active.validate({"tile": 8})
        with pytest.raises(ValueError, match="eligible values"):
            active.validate({"tile": 64})


def test_efficiency_errors_are_not_silent_rejections():
    def broken(p):
        raise RuntimeError("invalid planner state")

    space = ParameterSpace(
        knobs=(Knob(name="tile", values=(16,)),), efficiency_predicates=(broken,)
    )
    with pytest.raises(RuntimeError, match="invalid planner state"):
        list(space.configurations())
    assert len(list(replace(space, autotune_effort="audit").configurations())) == 1
    with pytest.raises(ValueError):
        replace(space, autotune_effort="exhaustive")


def _query():
    from b12x.norm.mhc._tuning import MhcQuery

    return MhcQuery(
        dtype="bfloat16",
        max_tokens=64,
        hidden_size=5120,
        split_k=80,
        operation="post_pre",
        has_norm_weight=True,
        lagged_mix=True,
        norm_eps=1e-20,
        rms_eps=1e-20,
        smem_limit=100 << 10,
    )


def _audit(query):
    return replace(query, controls=FrozenMapping({"B12X_AUTOTUNE_EFFORT": "audit"}))


def _choice(**changes):
    return (
        dict(
            backend="tf32_tma",
            lagged_prepare=False,
            partials_per_cta=4,
            projection_tile_n=24,
            projection_tile_k=64,
            projection_num_stages=2,
            projection_num_m_warps=2,
            projection_num_n_warps=1,
            projection_k_splits=20,
        )
        | changes
    )


@pytest.mark.parametrize(
    "changes",
    [
        # A wider M tile with the same grid only adds padding.
        {"projection_num_m_warps": 3},
        # N tiles beyond the covering power of two add masked columns.
        {"projection_tile_n": 64},
        # A fourth N warp over 24 mixes is wholly masked.
        {"projection_tile_n": 32, "projection_num_n_warps": 4},
        # Four stages cannot buffer a two-iteration K loop.
        {
            "projection_tile_k": 256,
            "projection_k_splits": 40,
            "projection_num_stages": 4,
            "projection_num_m_warps": 1,
            "projection_tile_n": 8,
        },
    ],
)
def test_mhc_dominance_rules_hold_at_every_effort_and_allow_pins(changes):
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    device = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    query = _query()
    choice = _choice(**changes)
    for active in (query, _audit(query)):
        with pytest.raises(ValueError, match="correctness predicates"):
            TUNING.parameter_space(active, device).validate(choice)
    config = MhcConfig(
        projection_tile_m=16 * choice["projection_num_m_warps"], **choice
    )
    assert TUNING.configure(query, device=device, override=config).pinned == config


@pytest.mark.parametrize(
    "changes",
    [
        {"projection_tile_k": 16},
        {"projection_tile_n": 24, "projection_num_n_warps": 2},
        {"projection_tile_k": 256, "projection_k_splits": 32},
        {"projection_num_m_warps": 16, "projection_num_n_warps": 3},
        {
            "projection_num_m_warps": 16,
            "projection_tile_k": 256,
            "projection_num_stages": 4,
        },
    ],
)
def test_mhc_audit_preserves_correctness_rules(changes):
    from b12x.norm.mhc._tuning import TUNING

    with pytest.raises(ValueError, match="correctness predicates"):
        TUNING.parameter_space(_audit(_query()), None).validate(_choice(**changes))


def test_mhc_underoccupied_projection_remains_eligible():
    from b12x.norm.mhc._tuning import TUNING

    query = replace(_query(), max_tokens=1)
    choice = _choice(projection_num_m_warps=1, projection_k_splits=8)
    for capability, sm in (((12, 1), 48), ((12, 0), 188)):
        TUNING.parameter_space(
            query, DeviceIdentity("nvidia", capability, sm, "Blackwell")
        ).validate(choice)


def test_mhc_declaration_captures_effort(monkeypatch):
    from b12x.norm import mhc
    from b12x.norm.mhc._tuning import TUNING

    caps = mhc.Caps(device="cpu", max_tokens=1, hidden_size=5120)
    invocation = FrozenMapping({"operation": "collapse"})
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
    pruned = mhc.plan(caps, invocation=invocation)
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    audit = mhc.plan(caps, invocation=invocation)
    assert pruned.query.controls["B12X_AUTOTUNE_EFFORT"] == "default"
    assert audit.query.controls["B12X_AUTOTUNE_EFFORT"] == "audit"
    assert TUNING.encode_query(pruned.query) != TUNING.encode_query(audit.query)
    assert TUNING.parameter_space(_query(), None).autotune_effort == "default"
    with pytest.raises(ValueError, match="not a valid AutotuneEffort"):
        TUNING.configure(
            replace(_query(), controls=FrozenMapping({"B12X_AUTOTUNE_EFFORT": "yes"})),
            device=None,
        )


def test_effort_does_not_change_compiled_program_identity(monkeypatch):
    from b12x._lib.compiler import _static_compile_cache_context
    import cutlass.cute as cute

    try:
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
        _static_compile_cache_context.cache_clear()
        pruned = _static_compile_cache_context(cute.compile)
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
        _static_compile_cache_context.cache_clear()
        assert _static_compile_cache_context(cute.compile) == pruned
    finally:
        _static_compile_cache_context.cache_clear()


@pytest.mark.parametrize("capability,sm", [((12, 0), 188), ((12, 1), 48)])
@pytest.mark.parametrize(
    "boundary,changes",
    [
        (384, {"projection_num_m_warps": 1, "projection_num_n_warps": 3}),
        (384, {"projection_num_m_warps": 3}),
        (384, {"projection_tile_k": 32}),
        (384, {"projection_tile_k": 128, "projection_num_stages": 3}),
        (2048, {"projection_k_splits": 16}),
    ],
)
def test_mhc_prefill_correlations_preserve_products_and_pins(
    capability, sm, boundary, changes
):
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    device = DeviceIdentity("nvidia", capability, sm, "Blackwell")
    query = replace(_query(), max_tokens=boundary)
    choice = _choice(
        **(dict(projection_num_m_warps=4, projection_k_splits=8) | changes)
    )
    before = TUNING.parameter_space(replace(query, max_tokens=boundary - 1), device)
    after = TUNING.parameter_space(query, device)
    if sm == 48 and changes == {"projection_k_splits": 16}:
        with pytest.raises(ValueError, match="efficiency predicates"):
            before.validate(choice)
    else:
        before.validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        after.validate(choice)
    replace(after, autotune_effort="audit").validate(choice)
    config = MhcConfig(
        projection_tile_m=16 * choice["projection_num_m_warps"], **choice
    )
    assert TUNING.configure(query, device=device, override=config).pinned == config
    assert before.knobs == after.knobs


@pytest.mark.parametrize("capability,sm", [((12, 0), 188), ((12, 1), 48)])
@pytest.mark.parametrize(
    "hidden,m,tm,tn,tk,nw,stages,splits",
    [
        (5120, 64, 16, 24, 64, 3, 3, 40),
        (5120, 4096, 128, 24, 128, 1, 2, 5),
        (5120, 4096, 128, 32, 64, 1, 2, 5),
        (4096, 384, 64, 24, 64, 1, 3, 8),
        (4096, 1024, 32, 8, 256, 1, 1, 1),
        (7168, 3584, 64, 24, 64, 1, 3, 8),
        (4096, 8192, 128, 24, 64, 1, 2, 4),
    ],
)
def test_mhc_prefill_correlations_retain_measured_winner_geometries(
    capability,
    sm,
    hidden,
    m,
    tm,
    tn,
    tk,
    nw,
    stages,
    splits,
):
    from b12x.norm.mhc._tuning import TUNING

    query = replace(_query(), max_tokens=m, hidden_size=hidden, split_k=hidden // 64)
    choice = _choice(
        projection_num_m_warps=tm // 16,
        projection_tile_n=tn,
        projection_tile_k=tk,
        projection_num_n_warps=nw,
        projection_num_stages=stages,
        projection_k_splits=splits,
    )
    TUNING.parameter_space(
        query, DeviceIdentity("nvidia", capability, sm, "Blackwell")
    ).validate(choice)


def test_mhc_excess_split_grid_scales_with_sm_count_and_preserves_pins():
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    query = replace(_query(), max_tokens=384)
    choice = _choice(projection_num_m_warps=4, projection_tile_n=8)
    small = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
    large = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    TUNING.parameter_space(query, large).validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, small).validate(choice)
    replace(TUNING.parameter_space(query, small), autotune_effort="audit").validate(
        choice
    )
    config = MhcConfig(projection_tile_m=64, **choice)
    assert TUNING.configure(query, device=small, override=config).pinned == config
