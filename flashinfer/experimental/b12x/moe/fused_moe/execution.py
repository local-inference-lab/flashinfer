"""Declarative canonical fused-MoE execution preparation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from b12x.preparation import FrozenMapping


@dataclass(frozen=True, kw_only=True)
class ExecutionCapacity:
    """Token capacity, decode specializations, and output storage geometry.

    ``token`` outputs hold weighted [tokens, hidden] values. ``routes`` exposes
    unweighted [tokens, top_k, hidden] GEMM outputs in activation precision for
    token or expert outer scaling. Invalid route entries are unspecified;
    consumers must mask them with the route IDs before applying their weights.
    Expert-outer NVFP4 routes reduce FC2 slices in a fixed FP32 order before
    the final BF16 conversion.
    """

    max_tokens: int
    top_k: int
    warmup_token_counts: tuple[int, ...] = ()
    route_num_experts: int | None = None
    output_layout: str = "token"
    input_format: str = "bf16"
    reserve_sms: int = 0

    def __post_init__(self):
        if self.output_layout not in ("token", "routes"):
            raise ValueError("MoE output layout must be token or routes")
        if self.input_format not in ("bf16", "mxfp8"):
            raise ValueError("MoE input_format must be bf16 or mxfp8")
        if (
            isinstance(self.reserve_sms, bool)
            or not isinstance(self.reserve_sms, int)
            or self.reserve_sms < 0
        ):
            raise ValueError("reserve_sms must be a non-negative integer")
        max_tokens, top_k = int(self.max_tokens), int(self.top_k)
        if max_tokens <= 0 or top_k <= 0:
            raise ValueError("max_tokens and top_k must be positive")
        counts = tuple(sorted({int(value) for value in self.warmup_token_counts}))
        if any(value <= 0 or value > max_tokens for value in counts):
            raise ValueError("warmup token counts must be positive and within capacity")
        if self.route_num_experts is not None and int(self.route_num_experts) < 0:
            raise ValueError("route_num_experts cannot be negative")
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "top_k", top_k)
        object.__setattr__(self, "warmup_token_counts", counts)
        if self.route_num_experts is not None:
            object.__setattr__(self, "route_num_experts", int(self.route_num_experts))


@dataclass(frozen=True, kw_only=True)
class RoutingSpec:
    """Immutable routing and numerical controls captured by preparation."""

    apply_router_weight_on_input: bool = False
    logits_dtype: torch.dtype | None = None
    deterministic_output: bool | None = None
    collect_activation_amax: bool = False
    score_func: str = "softmax"
    renormalize: bool = True
    has_correction_bias: bool = False
    has_image_correction_bias: bool = False
    has_image_mask: bool = False
    has_valid_tokens: bool = False
    routed_scaling_factor: float = 1.0


def plan_execution(
    *,
    experts,
    capacity: ExecutionCapacity,
    routing: RoutingSpec | None = None,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
    scale_expansion=None,
    activation_mode=None,
):
    """Declare execution with an optional A4/A16 override of the token cutoff.

    An explicit mode must be available on the prepared weight owner. Omission
    preserves the owner's activation mode and inclusive A16 token cutoff.
    """
    from .planning import ActivationMode, _packed_recipe
    from .source import PackedSource

    if activation_mode is not None:
        activation_mode = ActivationMode(activation_mode)
        if activation_mode not in (ActivationMode.A4, ActivationMode.A16):
            raise ValueError("execution activation_mode must be a4 or a16")
        if (
            not isinstance(experts.plan.source, PackedSource)
            or _packed_recipe(experts.plan.source, activation_mode)
            not in experts.plan._impl.quant_modes
        ):
            raise ValueError(
                "execution activation mode is unavailable on the prepared weight owner"
            )
    if experts.plan.activation.outer_scale == "token":
        from ._token_outer import plan
    else:
        if (
            capacity.output_layout == "routes"
            and experts.plan._impl.quant_modes != frozenset({"nvfp4"})
        ):
            raise ValueError("expert-scale route outputs require NVFP4 A4 preparation")
        if routing is not None and routing.has_valid_tokens:
            raise ValueError(
                "valid-token integrated routing requires token outer scaling"
            )
        from ._preparation import plan

    if (
        capacity.output_layout == "routes"
        and routing is not None
        and routing.apply_router_weight_on_input
    ):
        raise ValueError(
            "unweighted route outputs cannot apply router weights on input"
        )

    options = (
        {}
        if experts.plan.activation.outer_scale == "token"
        else {"activation_mode": activation_mode}
    )
    return plan(
        experts,
        capacity=capacity,
        routing=routing,
        invocation=invocation,
        override=override,
        scale_expansion=scale_expansion,
        **options,
    )


__all__ = ["ExecutionCapacity", "RoutingSpec", "plan_execution"]
