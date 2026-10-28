"""Route top-k launch geometry is raced on every device where it preserves the arithmetic."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import pytest

from b12x.moe.fused_moe import RouteTopKInvocation
from b12x.moe.fused_moe import _preparation, _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping

TUNING = _tuning.ROUTE_TUNING
SM103 = DeviceIdentity(
    vendor="nvidia", compute_capability=(10, 3), sm_count=152, product_name="GB300"
)
SM120 = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 0),
    sm_count=188,
    product_name="RTX PRO 6000",
)
SM121 = DeviceIdentity(
    vendor="nvidia", compute_capability=(12, 1), sm_count=48, product_name="GB10"
)


def query(effort="default", **overrides):
    values = dict(
        num_tokens=4096,
        num_experts=288,
        top_k=8,
        logits_dtype="float32",
        score_func="sigmoid",
        renormalize=True,
        has_correction_bias=True,
        has_image_correction_bias=False,
        has_image_mask=False,
        routed_scaling_factor=2.5,
        logits_row_stride=288,
        topk_row_stride=8,
        ids_row_stride=8,
        weights_row_stride=8,
        autotune_effort=effort,
    )
    values.update(overrides)
    return _tuning.MoeRouteQuery(**values)


def geometries(q):
    return {
        (choice["num_warps"], choice["rows_per_program"])
        for choice in TUNING.parameter_space(q, None).configurations()
    }


def test_glm_sigmoid_route_races_warps_and_rows():
    audit = geometries(query("audit"))
    assert audit == {(w, r) for w in (1, 2, 4, 8) for r in (1, 2, 4)}
    default = geometries(query("default"))
    assert default == {(w, r) for w in (1, 2, 4, 8) for r in (1, 2, 4)} - {
        (8, 2),
        (8, 4),
    }


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"score_func": "softmax", "has_correction_bias": False},
        {"score_func": "softmax", "has_correction_bias": True},
        {"score_func": "sqrtsoftplus"},
        {"num_experts": 8, "top_k": 2, "logits_row_stride": 8},
        {"num_experts": 1024, "top_k": 64, "logits_row_stride": 1024},
        {"num_tokens": 1},
    ],
)
def test_default_is_the_original_geometry_and_belongs_to_the_space(effort, overrides):
    q = query(effort, **overrides)
    default = TUNING.default_config(q, None)
    assert default.rows_per_program == 1
    assert default.num_warps == (4 if q.num_experts <= 256 else 8)
    TUNING.parameter_space(q, None).validate(
        {key: value for key, value in default.to_dict().items()}
    )
    assert (default.num_warps, 1) in geometries(q)
    configuration = TUNING.configure(q, device=None)
    candidates = [config for _, config in TUNING.iterate(configuration)]
    assert default in candidates
    if effort == "off":
        assert candidates == [default]


def test_softmax_with_selection_bias_keeps_the_expert_wide_sum_order():
    q = query("audit", score_func="softmax")
    assert {warps for warps, _ in geometries(q)} == {8}
    q = query("audit", score_func="softmax", has_correction_bias=False)
    assert {warps for warps, _ in geometries(q)} == {1, 2, 4, 8}
    q = query(
        "audit",
        score_func="softmax",
        has_correction_bias=False,
        has_image_correction_bias=True,
        has_image_mask=True,
    )
    assert {warps for warps, _ in geometries(q)} == {8}


def test_wide_top_k_slot_sum_keeps_the_original_warp_count():
    q = query("audit", num_experts=1024, top_k=33, logits_row_stride=1024)
    assert {warps for warps, _ in geometries(q)} == {8}
    q = query("audit", num_experts=1024, top_k=32, logits_row_stride=1024)
    assert {warps for warps, _ in geometries(q)} == {1, 2, 4, 8}


def test_warps_never_exceed_the_expert_lanes():
    q = query("audit", num_experts=64, top_k=4, logits_row_stride=64)
    assert {warps for warps, _ in geometries(q)} == {1, 2, 4}


def test_rows_per_program_never_exceed_the_token_capacity():
    assert {rows for _, rows in geometries(query("audit", num_tokens=1))} == {1}
    assert {rows for _, rows in geometries(query("audit", num_tokens=2))} == {1, 2}
    assert {rows for _, rows in geometries(query("audit", num_tokens=3))} == {1, 2}


def test_validate_config_rejects_geometry_that_changes_arithmetic():
    q = query(score_func="softmax")
    default = TUNING.default_config(q, None)
    TUNING.validate_config(q, replace(default, rows_per_program=4), None)
    with pytest.raises(ValueError, match="geometry"):
        TUNING.validate_config(q, replace(default, num_warps=4), None)
    with pytest.raises(ValueError, match="geometry"):
        TUNING.validate_config(q, replace(default, rows_per_program=3), None)
    with pytest.raises(ValueError, match="backend"):
        TUNING.validate_config(q, replace(default, backend="cutedsl"), None)
    with pytest.raises(TypeError):
        TUNING.validate_config(q, object(), None)


def test_config_payload_round_trips_and_rejects_extra_fields():
    q = query()
    config = _tuning.MoeRouteConfig(backend="triton", num_warps=2, rows_per_program=4)
    payload = TUNING.config_payload(config)
    assert TUNING.decode_config(payload) == config
    with pytest.raises(ValueError, match="config fields"):
        TUNING.decode_config(FrozenMapping({**payload, "extra": 1}))
    assert TUNING.encode_query(q)["autotune_effort"] == "default"


def test_plan_compile_jobs_exclude_search_effort_and_carry_the_geometry():
    plans = []
    for effort in ("default", "audit"):
        from b12x.preparation import autotune_effort

        with autotune_effort(effort):
            plans.append(
                _preparation.plan_route_topk(
                    RouteTopKInvocation(
                        num_tokens=4096,
                        num_experts=288,
                        top_k=8,
                        logits_dtype="float32",
                        score_func="sigmoid",
                        has_correction_bias=True,
                        routed_scaling_factor=2.5,
                    )
                )
            )
    assert [plan.query.autotune_effort for plan in plans] == ["default", "audit"]
    config = _tuning.MoeRouteConfig(backend="triton", num_warps=2, rows_per_program=2)
    jobs = [plan._compile_jobs(config, None) for plan in plans]
    assert jobs[0] == jobs[1]
    (job,) = jobs[0]
    assert job.factory == "b12x.moe.fused_moe._preparation:compile_route_topk"
    assert "autotune_effort" not in job.args[0]
    assert job.args[2] == config.to_dict()
    assert job.args[1] is _preparation.CompileJob.DEVICE_ORDINAL


def test_fused_route_programs_use_the_original_geometry(monkeypatch):
    captured = {}

    class Kernel:
        @staticmethod
        def warmup(*args, **kwargs):
            captured.update(kwargs)
            return "program"

    import b12x.moe._shared.routing as routing
    import torch

    monkeypatch.setattr(routing, "_route_topk_kernel", Kernel)
    monkeypatch.setattr(routing, "_route_topk_rows_kernel", Kernel)
    monkeypatch.setattr(torch.cuda, "device", lambda ordinal: nullcontext())
    payload = _preparation._route_program_payload(TUNING.encode_query(query()))
    _preparation.compile_route_topk(payload, 0)
    assert "ROWS_PER_PROGRAM" not in captured
    assert captured["num_warps"] == 8
    assert captured["grid"] == (4096,)
    _preparation.compile_route_topk(
        payload,
        0,
        {"backend": "triton", "num_warps": 2, "rows_per_program": 4},
    )
    assert captured["ROWS_PER_PROGRAM"] == 4
    assert captured["num_warps"] == 2
    assert captured["grid"] == (1024,)


@pytest.mark.parametrize("device", [None, SM103, SM120, SM121])
def test_geometry_candidates_do_not_depend_on_the_device(device):
    q = query("audit")
    offered = {
        (c["num_warps"], c["rows_per_program"])
        for c in TUNING.parameter_space(q, device).configurations()
    }
    assert offered == geometries(q)
    assert len(offered) == 12
    default = TUNING.default_config(q, device)
    assert (default.num_warps, default.rows_per_program) == (8, 1)
    TUNING.validate_config(q, default, device)
    TUNING.validate_config(q, replace(default, num_warps=2), device)
    with pytest.raises(ValueError, match="geometry"):
        TUNING.validate_config(
            replace(q, score_func="softmax"), replace(default, num_warps=2), device
        )
