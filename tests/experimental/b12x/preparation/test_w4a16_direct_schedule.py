"""Native W4A16 direct schedules share precision and routing requirements."""

from dataclasses import replace

import pytest

from b12x.moe.fused_moe._tuning import MoeDecodeConfig, TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping
from .test_precision_choices import _nvfp4_a16_query


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "RTX PRO 6000 Blackwell")


@pytest.mark.parametrize("activation", ("situ", "silu", "relu2"))
@pytest.mark.parametrize("rows", (2, 4, 6, 8))
def test_native_direct_schedules_are_tuning_candidates(activation, rows):
    query = replace(
        _nvfp4_a16_query(rows),
        source_format="fp4_e8m0_k32",
        w4a16_scale_format="e8m0_k32",
        activation=activation,
        hidden_size=3584,
        intermediate_size=192,
    )
    candidates = {
        config for _, config in TUNING.eligible_plan(query, DEVICE).candidates
    }
    fused = MoeDecodeConfig("w4a16", "internal", None, w4a16_route_mode="direct")
    split = replace(fused, w4a16_direct_schedule="split")
    assert fused in candidates
    assert split in candidates
    assert MoeDecodeConfig.from_config(FrozenMapping(split.to_dict())) == split
    with pytest.raises(ValueError, match="outside its eligible values"):
        TUNING.lower(
            query,
            DEVICE,
            TUNING.config_payload(replace(split, w4a16_route_mode="packed")),
        )
    with pytest.raises(ValueError, match="native direct"):
        TUNING.lower(
            replace(query, w4a16_weight_layout="packed"),
            DEVICE,
            TUNING.config_payload(split),
        )


@pytest.mark.parametrize(
    "layout,rows,expected",
    [("modelopt", 4, "direct"), ("packed", 4, "packed"), ("packed", 1, "direct")],
)
def test_mxfp4_k64_shard_route_heuristic_uses_the_prepared_layout(
    layout, rows, expected
):
    query = replace(
        _nvfp4_a16_query(rows),
        source_format="fp4_e8m0_k32",
        w4a16_scale_format="e8m0_k32",
        w4a16_weight_layout=layout,
        weight_layouts=("mma_packed" if layout == "packed" else "source_native",),
        activation="situ",
        hidden_size=256,
        intermediate_size=192,
    )
    assert TUNING.default_config(query, DEVICE).w4a16_route_mode == expected
