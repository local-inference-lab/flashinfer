"""Compiler-visible prepared MoE execution over caller-owned storage."""

from copy import copy
from dataclasses import replace

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


def scale_buffers(binding):
    """Expose CSF planes as reads or mutations according to the invocation."""
    if hasattr(binding, "experts"):
        experts = binding.experts
        decoder = experts.nvfp4_csf
        if decoder is None or binding.execution_plan.decode_config.csf_inline_scales:
            return [], []
        writes = not binding.scales_expanded or binding.implementation == "micro"
    else:
        experts = binding.state.experts._impl
        decoder = binding.state.scale_decoder
        if decoder is None:
            return [], []
        writes = not binding.scales_expanded
    buffers = [
        experts.w1_blockscale.view(torch.uint8),
        experts.w2_blockscale.view(torch.uint8),
        decoder.active,
    ]
    return ([], buffers) if writes else (buffers, [])


def _scale_view(experts, buffers, decoder=None):
    buffers = [
        buffers[0].view(experts.w1_blockscale.dtype),
        buffers[1].view(experts.w2_blockscale.dtype),
        buffers[2],
    ]
    for source, target in zip(
        (experts.w1_blockscale, experts.w2_blockscale), buffers[:2]
    ):
        if (
            target.shape != source.shape
            or target.stride() != source.stride()
            or target.dtype != source.dtype
            or target.device != source.device
        ):
            raise ValueError("CSF scale buffers must preserve prepared tensor metadata")
    # Weight values were validated at preparation; only mutable storage is rebound.
    result = copy(experts)
    object.__setattr__(result, "w1_blockscale", buffers[0])
    object.__setattr__(result, "w2_blockscale", buffers[1])
    representation = experts.representation
    if representation is not None:
        payload = representation.value
        updates = {}
        for names, before, after in zip(
            (("w13_scale", "w13_sfb"), ("w2_scale", "w2_sfb")),
            (experts.w1_blockscale, experts.w2_blockscale),
            buffers[:2],
        ):
            for name in names:
                value = getattr(payload, name, None)
                if (
                    isinstance(value, torch.Tensor)
                    and value.untyped_storage().data_ptr()
                    == before.untyped_storage().data_ptr()
                ):
                    offset = (
                        value.storage_offset() * value.element_size()
                        - before.storage_offset() * before.element_size()
                        + after.storage_offset() * after.element_size()
                    ) // value.element_size()
                    updates[name] = after.view(value.dtype).as_strided(
                        value.shape, value.stride(), offset
                    )
        if updates:
            object.__setattr__(
                result,
                "representation",
                replace(representation, value=replace(payload, **updates)),
            )
    if decoder is not None:
        object.__setattr__(result, "nvfp4_csf", decoder)
    return result


def _bind_scale_buffers(binding, buffers):
    if not buffers:
        return binding
    if len(buffers) != 3:
        raise ValueError("CSF execution requires two scale planes and routing storage")
    decoder = (
        binding.experts.nvfp4_csf
        if hasattr(binding, "experts")
        else binding.state.scale_decoder
    )
    active = decoder.active
    if (
        buffers[2].shape != active.shape
        or buffers[2].stride() != active.stride()
        or buffers[2].dtype != active.dtype
        or buffers[2].device != active.device
    ):
        raise ValueError("CSF routing storage must preserve prepared tensor metadata")
    if hasattr(binding, "experts"):
        experts = binding.experts
        decoder = replace(experts.nvfp4_csf, active=buffers[2])
        return replace(binding, experts=_scale_view(experts, buffers, decoder))
    state = binding.state
    experts = _scale_view(state.experts._impl, buffers)
    state = replace(
        state,
        experts=replace(state.experts, _impl=experts),
        scale_decoder=replace(state.scale_decoder, active=buffers[2]),
    )
    return replace(binding, state=state)


@torch.library.custom_op(
    "b12x::prepared_moe_out",
    mutates_args=("output", "scratch", "activation_amax", "scale_storage"),
)
def prepared_moe_out(
    a: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    scale_inputs: list[torch.Tensor],
    scale_storage: list[torch.Tensor],
    plan_handle: int,
    input_scales_static: bool,
    fast_math: bool | None,
    activation_amax: torch.Tensor | None,
    layer_idx: int | None,
    route_expert_map: torch.Tensor | None,
    output_expert_map: torch.Tensor | None,
    scales_expanded: bool,
    a4_prefill: bool | None,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "moe.decode", a.device)
    binding = state.bind(
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        scratch=scratch,
        input_scales_static=input_scales_static,
        fast_math=fast_math,
        activation_amax=activation_amax,
        layer_idx=layer_idx,
        route_expert_map=route_expert_map,
        output_expert_map=output_expert_map,
        scales_expanded=scales_expanded,
        a4_prefill=a4_prefill,
    )
    state.run(_bind_scale_buffers(binding, scale_storage or scale_inputs))


@prepared_moe_out.register_fake
def _fake(
    a,
    topk_weights,
    topk_ids,
    output,
    scratch,
    scale_inputs,
    scale_storage,
    plan_handle,
    input_scales_static,
    fast_math,
    activation_amax,
    layer_idx,
    route_expert_map,
    output_expert_map,
    scales_expanded,
    a4_prefill,
) -> None:
    pass


@torch.library.custom_op(
    "b12x::prepared_routed_moe_out",
    mutates_args=(
        "topk_weights",
        "topk_ids",
        "selected_logits",
        "output",
        "scratch",
        "scale_storage",
    ),
)
def prepared_routed_moe_out(
    a: torch.Tensor,
    router_logits: torch.Tensor,
    selected_logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    scale_inputs: list[torch.Tensor],
    scale_storage: list[torch.Tensor],
    plan_handle: int,
    correction_bias: torch.Tensor | None,
    image_correction_bias: torch.Tensor | None,
    image_mask: torch.Tensor | None,
    num_valid_tokens: torch.Tensor | None,
    scales_expanded: bool,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "moe.decode", a.device)
    binding = state.bind(
        a=a,
        router_logits=router_logits,
        selected_logits=selected_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        scratch=scratch,
        correction_bias=correction_bias,
        image_correction_bias=image_correction_bias,
        image_mask=image_mask,
        num_valid_tokens=num_valid_tokens,
        scales_expanded=scales_expanded,
    )
    state.run(_bind_scale_buffers(binding, scale_storage or scale_inputs))


@prepared_routed_moe_out.register_fake
def _fake_routed(
    a,
    router_logits,
    selected_logits,
    topk_weights,
    topk_ids,
    output,
    scratch,
    scale_inputs,
    scale_storage,
    plan_handle,
    correction_bias,
    image_correction_bias,
    image_mask,
    num_valid_tokens,
    scales_expanded,
) -> None:
    pass
