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


def test_mhc_grid_fill_scales_with_sm_count_and_preserves_pins():
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    # Eight K splits over four 16-row M tiles fill two thirds of a 48-SM device
    # and a sixth of a 188-SM device; the default tier wants at least 40%.
    query = _query()
    choice = _choice(
        projection_num_m_warps=1,
        projection_num_n_warps=3,
        projection_num_stages=3,
        projection_k_splits=8,
    )
    small = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
    large = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    TUNING.parameter_space(query, small).validate(choice)
    with pytest.raises(ValueError, match="efficiency predicates"):
        TUNING.parameter_space(query, large).validate(choice)
    TUNING.parameter_space(_audit(query), large).validate(choice)
    config = MhcConfig(projection_tile_m=16, **choice)
    assert TUNING.configure(query, device=large, override=config).pinned == config
    # A K split never extends the grid past one wave on either device.
    wide = {**choice, "projection_k_splits": 64}
    for device in (small, large):
        with pytest.raises(ValueError, match="efficiency predicates"):
            TUNING.parameter_space(query, device).validate(wide)
        TUNING.parameter_space(_audit(query), device).validate(wide)


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


@pytest.mark.parametrize(
    "short,long",
    [
        # Short M splits the N tile one MMA tile per warp; long M keeps one N warp.
        (
            dict(
                projection_num_m_warps=2,
                projection_num_n_warps=3,
                projection_k_splits=10,
            ),
            dict(
                projection_num_m_warps=4,
                projection_num_n_warps=1,
                projection_k_splits=20,
            ),
        ),
        # Short M admits any M tile up to 64 rows; long M wants whole warp quads.
        (
            dict(
                projection_num_m_warps=3,
                projection_num_n_warps=3,
                projection_k_splits=10,
            ),
            dict(
                projection_num_m_warps=8,
                projection_num_n_warps=1,
                projection_k_splits=40,
            ),
        ),
    ],
)
def test_mhc_long_m_boundary_swaps_warp_layout_and_preserves_pins(short, long):
    from b12x.norm.mhc._tuning import MhcConfig, TUNING

    device = DeviceIdentity("nvidia", (12, 0), 188, "Blackwell")
    below = replace(_query(), max_tokens=383)
    above = replace(_query(), max_tokens=384)
    for favored, other, choice in ((below, above, short), (above, below, long)):
        choice = _choice(projection_num_stages=3, **choice)
        TUNING.parameter_space(favored, device).validate(choice)
        with pytest.raises(ValueError, match="efficiency predicates"):
            TUNING.parameter_space(other, device).validate(choice)
        TUNING.parameter_space(_audit(other), device).validate(choice)
        config = MhcConfig(
            projection_tile_m=16 * choice["projection_num_m_warps"], **choice
        )
        assert TUNING.configure(other, device=device, override=config).pinned == config
    assert (
        TUNING.parameter_space(below, device).knobs
        == TUNING.parameter_space(above, device).knobs
    )


@pytest.mark.parametrize(
    "hidden,tokens,m_warps,tn,tk,nw,stages,splits",
    [
        # Race winners on a 188-SM SM120 device: hidden 5120 from the tuning
        # ledger, hidden 4096 from audit races over the whole legal space.
        (5120, 64, 1, 24, 64, 3, 3, 20),
        (5120, 128, 2, 16, 128, 2, 2, 20),
        (5120, 512, 4, 24, 64, 1, 3, 20),
        (5120, 2048, 4, 24, 64, 1, 3, 5),
        (5120, 4096, 8, 24, 64, 1, 3, 5),
        (4096, 3072, 12, 32, 64, 1, 3, 8),
        (4096, 4096, 12, 24, 64, 1, 3, 8),
        (4096, 8192, 8, 32, 64, 1, 3, 2),
        (4096, 16384, 8, 32, 128, 1, 2, 1),
    ],
)
def test_mhc_default_tier_retains_measured_winner_geometries(
    hidden, tokens, m_warps, tn, tk, nw, stages, splits
):
    from b12x.norm.mhc._tuning import TUNING

    query = replace(
        _query(), max_tokens=tokens, hidden_size=hidden, split_k=hidden // 64
    )
    choice = _choice(
        projection_num_m_warps=m_warps,
        projection_tile_n=tn,
        projection_tile_k=tk,
        projection_num_n_warps=nw,
        projection_num_stages=stages,
        projection_k_splits=splits,
    )
    device = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000")
    TUNING.parameter_space(query, device).validate(choice)
    assert (
        FrozenMapping(choice) in TUNING.parameter_space(query, device).configurations()
    )


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("sm", [48, 132, 188])
@pytest.mark.parametrize(
    "tokens", [1, 31, 32, 52, 127, 128, 383, 384, 2048, 4096, 16384, 32768]
)
def test_mhc_heuristic_default_is_raced_at_default_effort(hidden, sm, tokens):
    from b12x.norm.mhc._tuning import TUNING

    query = replace(
        _query(), max_tokens=tokens, hidden_size=hidden, split_k=hidden // 64
    )
    device = DeviceIdentity("nvidia", (12, 0), sm, "Blackwell")
    default = TUNING.default_config(query, device)
    assert default in [config for _, config in TUNING.choices(query, device)]
