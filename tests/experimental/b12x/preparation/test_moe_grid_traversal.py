"""MoE resident-grid ladders form the complete ordered search domain."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import TUNING
from b12x.preparation import DeviceIdentity
from .test_fused_moe_effort import query
from .test_precision_choices import _nvfp4_a16_query


CASES = (
    *((tokens, 576, "w4a8_mx") for tokens in (1, 4, 8, 16, 24, 64, 1025)),
    (8, 512, "w4a8_mx"),
    (24, 512, "w4a8_mx"),
    (8, 512, "nvfp4"),
    (1025, 512, "nvfp4"),
)


def _grid(space):
    return next(k for k in space.knobs if k.name == "max_active_clusters")


@pytest.mark.parametrize("sm_count", (7, 48, 96, 188))
@pytest.mark.parametrize("tokens,width,recipe", CASES)
@pytest.mark.parametrize("effort", ("off", "default", "audit"))
def test_ordered_ladder_and_default(sm_count, tokens, width, recipe, effort):
    device = DeviceIdentity("nvidia", (12, 0), sm_count, "Blackwell")
    q = replace(
        query(tokens=tokens, recipe=recipe, autotune_effort=effort),
        intermediate_size=width,
        csf_inline_scales=True,
    )
    configuration = TUNING.configure(q, device=device)
    candidates = list(TUNING.iterate(configuration))
    assert candidates
    assert configuration.default in [config for _, config in candidates]
    configuration.space.validate(TUNING.config_payload(configuration.default))
    ladder = _grid(configuration.space).values
    assert ladder[0] is None
    assert list(ladder[1:]) == sorted(set(ladder[1:]))
    assert all(1 <= value <= 2 * sm_count for value in ladder[1:])


@pytest.mark.parametrize("sm_count", (7, 48, 188))
@pytest.mark.parametrize("tokens,width", ((8, 576), (24, 576), (8, 512)))
def test_caps_off_the_ladder_are_never_eligible(sm_count, tokens, width):
    device = DeviceIdentity("nvidia", (12, 0), sm_count, "Blackwell")
    q = replace(query(tokens=tokens), intermediate_size=width, csf_inline_scales=True)
    configuration = TUNING.configure(q, device=device)
    grid = _grid(configuration.space)
    choice = dict(TUNING.config_payload(configuration.default))
    if not grid.active(choice):
        return
    choice["max_active_clusters"] = next(
        value for value in range(1, 2 * sm_count + 2) if value not in grid.values
    )
    for effort in ("default", "audit"):
        space = replace(configuration.space, autotune_effort=effort)
        with pytest.raises(ValueError, match="outside"):
            space.validate(choice)


@pytest.mark.parametrize("multi", (False, True))
@pytest.mark.parametrize("effort", ("default", "audit"))
def test_w4a16_ordered_domain_with_conditional_grid(multi, effort):
    device = DeviceIdentity("nvidia", (12, 1), 48, "GB10")
    q = replace(_nvfp4_a16_query(8), autotune_effort=effort)
    if multi:
        q = replace(
            q,
            quant_mode="multi",
            quant_modes=("nvfp4", "w4a16"),
            weight_layouts=("mma_view", "source_native"),
        )
    configuration = TUNING.configure(q, device=device)
    candidates = list(TUNING.iterate(configuration))
    assert configuration.default in [config for _, config in candidates]
    assert len({payload for payload, _ in candidates}) == len(candidates)


def test_ladder_traversal_counts():
    device = DeviceIdentity("nvidia", (12, 0), 188, "Blackwell")
    for tokens, expected_raw, expected_legal, expected_audit in (
        (8, 2444, 175, 325),
        (16, 2288, 133, 247),
        (24, 2288, 49, 91),
    ):
        q = replace(query(tokens=tokens), intermediate_size=576, csf_inline_scales=True)
        configuration = TUNING.configure(q, device=device)
        iterator = TUNING.iterate(configuration)
        list(iterator)
        assert iterator.cartesian_count == expected_raw
        assert iterator.legal_count == expected_legal
        audit = TUNING.iterate(
            TUNING.configure(replace(q, autotune_effort="audit"), device=device)
        )
        list(audit)
        assert audit.legal_count == expected_audit


def test_default_omits_small_caps_and_off_band_backends():
    device = DeviceIdentity("nvidia", (12, 0), 188, "Blackwell")
    for tokens, backends in (
        (4, {"micro"}),
        (12, {"micro", "dynamic"}),
        (24, {"dynamic"}),
    ):
        q = replace(query(tokens=tokens), intermediate_size=576, csf_inline_scales=True)
        raced = [config for _, config in TUNING.choices(q, device)]
        assert {config.backend for config in raced} == backends
        caps = {config.max_active_clusters for config in raced} - {None}
        assert caps and min(caps) >= device.sm_count // 4
        audited = [
            config
            for _, config in TUNING.choices(replace(q, autotune_effort="audit"), device)
        ]
        assert {config.backend for config in audited} >= backends
        assert min({c.max_active_clusters for c in audited} - {None}) == 1
    # Below eight tokens the dynamic kernel is legal but only the audit races it.
    q = replace(query(tokens=4), intermediate_size=576, csf_inline_scales=True)
    audited = TUNING.choices(replace(q, autotune_effort="audit"), device)
    assert {config.backend for _, config in audited} == {"micro", "dynamic"}
