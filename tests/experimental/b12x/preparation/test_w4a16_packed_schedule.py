"""Packed gated projections expose independent kernel schedules to the tuning contract."""

from dataclasses import replace

from b12x.moe.fused_moe._tuning import TUNING, MoeDecodeConfig
from b12x.preparation import FrozenMapping
from .test_precision_choices import _nvfp4_a16_query
from .test_w4a16_direct_schedule import DEVICE


def test_packed_mxfp4_stages_are_distinct_legal_tuning_choices():
    query = replace(
        _nvfp4_a16_query(4),
        activation="situ",
        source_format="fp4_e8m0_k32",
        w4a16_scale_format="e8m0_k32",
        w4a16_weight_layout="packed",
        weight_layouts=("mma_packed",),
        hidden_size=256,
        intermediate_size=192,
    )
    candidates = {
        config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    }
    fused = MoeDecodeConfig("w4a16", "internal", None, w4a16_route_mode="packed")
    split = replace(fused, w4a16_packed_schedule="split")
    assert {fused, split} <= candidates
    assert TUNING.config_payload(fused) != TUNING.config_payload(split)
    assert MoeDecodeConfig.from_config(FrozenMapping(split.to_dict())) == split
    legacy = fused.to_dict()
    del legacy["w4a16_packed_schedule"]
    assert MoeDecodeConfig.from_config(FrozenMapping(legacy)) == fused
    for unsupported in (
        replace(query, w4a16_weight_layout="modelopt"),
        replace(query, activation="relu2"),
        replace(query, collect_activation_amax=True),
    ):
        assert not any(
            c.w4a16_packed_schedule == "split"
            for _, c in TUNING.eligible_plan(unsupported, DEVICE).candidates
        )


def test_decode_search_lifts_only_efficiency_pruning_for_audit():
    query = replace(
        _nvfp4_a16_query(4),
        activation="situ",
        source_format="fp4_e8m0_k32",
        w4a16_scale_format="e8m0_k32",
        w4a16_weight_layout="packed",
        weight_layouts=("mma_packed",),
        hidden_size=256,
        intermediate_size=192,
    )
    default = {c for _, c in TUNING.eligible_plan(query, DEVICE).candidates}
    audit = {
        c
        for _, c in TUNING.eligible_plan(
            replace(query, autotune_effort="audit"), DEVICE
        ).candidates
    }
    assert default < audit
    assert any(c.w4a16_block_size_m == 32 for c in audit - default)
    assert any(c.w4a16_packed_schedule == "split" for c in default)
