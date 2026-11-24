from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
    is_gated_moe_activation,
    moe_activation_w1_rows,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from b12x.moe._shared.kernels.reference import _apply_gated_activation


def test_situ_activation_contract() -> None:
    assert normalize_moe_activation(SITU) == SITU
    assert is_gated_moe_activation(SITU)
    assert moe_activation_w1_rows(SITU, 128) == 256
    assert normalize_swiglu_limit_for_activation(SITU, None) is None
    assert normalize_swiglu_alpha_for_activation(SITU, None) == 1.0
    assert normalize_swiglu_beta_for_activation(SITU, None) == 0.0


def test_situ_rejects_swiglu_parameters() -> None:
    with pytest.raises(ValueError, match="not part of the situ"):
        normalize_swiglu_limit_for_activation(SITU, 7.0)
    with pytest.raises(ValueError, match="only configurable"):
        normalize_swiglu_alpha_for_activation(SITU, 1.702)
    with pytest.raises(ValueError, match="only configurable"):
        normalize_swiglu_beta_for_activation(SITU, 1.0)


def test_situ_reference_matches_kimi_k3_beta_scaled_contract() -> None:
    gate = torch.tensor(
        [-100.0, -2.0, 0.0, 2.0, 100.0],
        dtype=torch.float32,
    )
    up = torch.tensor(
        [100.0, -2.0, 0.0, 2.0, -100.0],
        dtype=torch.float32,
    )
    actual = _apply_gated_activation(
        gate,
        up,
        activation=SITU,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    situ_gate = (
        SITU_DEFAULT_BETA * torch.tanh(gate / SITU_DEFAULT_BETA) * torch.sigmoid(gate)
    )
    situ_up = SITU_DEFAULT_LINEAR_BETA * torch.tanh(up / SITU_DEFAULT_LINEAR_BETA)
    expected = situ_gate * situ_up
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("m", [1, 4])
def test_tiny_decode_situ_matches_w4a8_reference(monkeypatch, m: int) -> None:
    from b12x.moe.fused_moe import _impl
    from b12x.moe.fused_moe._compat import Caps
    from b12x.moe.fused_moe._impl import plan_tp_moe_scratch as plan
    from b12x.moe._shared.kernels.reference import (
        compare_to_reference,
        moe_reference_w4a8_mx,
    )
    from b12x.testing.reference.helpers import (
        prepare_tp_moe_fp4_experts,
        require_b12x,
    )

    from .test_w4a8_tiny_decode_swiglu_limit import (
        _make_scaled_inputs,
        _make_weights,
        _reset_dispatch_environment,
        _spy_tiny_launches,
    )

    device = require_b12x()
    experts, k, n, topk = 64, 2048, 1024, 8
    assert _impl._w4a8_mx_micro_supported(
        num_tokens=m,
        k=k,
        n=n,
        num_topk=topk,
        weight_E=experts,
        activation=SITU,
        source_format="fp4_e8m0_k32",
    )
    weights = _make_weights(device, seed=511 + m, experts=experts, k=k, n=n)
    inputs = _make_scaled_inputs(
        device,
        m=m,
        seed=521 + m,
        route_shift=0,
        experts=experts,
        k=k,
        topk=topk,
        projection_rms=3.0,
    )

    def _ref(act: str) -> torch.Tensor:
        return moe_reference_w4a8_mx(
            inputs.a.float(),
            weights.w1_fp4,
            weights.w1_scale,
            None,
            weights.w1_alpha,
            weights.w2_fp4,
            weights.w2_scale,
            None,
            weights.w2_alpha,
            inputs.topk_ids,
            inputs.topk_weights,
            experts,
            k,
            n,
            activation=act,
            swiglu_limit=None,
            w13_layout="w13",
        )

    ref_situ = _ref(SITU)
    ref_silu = _ref("silu")
    assert compare_to_reference(ref_situ, ref_silu).cos < 0.99

    _reset_dispatch_environment(monkeypatch)
    calls = _spy_tiny_launches(monkeypatch)
    prepared_experts = prepare_tp_moe_fp4_experts(
        a=inputs.a,
        a1_gscale=weights.a1_scale,
        w1_fp4=weights.w1_fp4,
        w1_blockscale=weights.w1_scale,
        w1_alphas=weights.w1_alpha,
        a2_gscale=weights.a2_scale,
        w2_fp4=weights.w2_fp4,
        w2_blockscale=weights.w2_scale,
        w2_alphas=weights.w2_alpha,
        activation=SITU,
        quant_mode="w4a8_mx",
        source_format="fp4_e8m0_k32",
        w13_layout="w13",
    )._impl
    scratch_plan = plan(
        Caps(
            max_tokens=m,
            num_topk=topk,
            device=inputs.a.device,
            weight_plan=prepared_experts.plan,
            quant_mode="w4a8_mx",
            core_token_counts=(m,),
            route_num_experts=0,
            frozen=True,
        )
    )
    scratch = tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in scratch_plan.scratch_specs()
    )
    output = torch.empty_like(inputs.a)
    binding = scratch_plan.bind(
        scratch=scratch,
        a=inputs.a,
        experts=prepared_experts,
        topk_weights=inputs.topk_weights,
        topk_ids=inputs.topk_ids,
        output=output,
        input_scales_static=True,
    )
    _impl.b12x_moe_fp4(binding=binding)
    torch.cuda.synchronize()
    assert binding.implementation == "micro"
    assert len(calls) == 1
    metrics = compare_to_reference(output.float(), ref_situ.float())
    nrmse = metrics.rmse / ref_situ.float().square().mean().sqrt().item()
    assert metrics.cos >= 0.995, f"tiny-decode SITU cos={metrics.cos}"
    assert nrmse <= 0.12, f"tiny-decode SITU nrmse={nrmse}"
