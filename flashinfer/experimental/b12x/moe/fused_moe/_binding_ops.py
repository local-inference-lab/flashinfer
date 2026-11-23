"""Compiler-visible prepared MoE execution over caller-owned storage."""

from copy import copy
from dataclasses import dataclass, replace

import torch

from b12x.preparation.types import plan_from_handle, require_prepared


@dataclass(frozen=True)
class _ScaleLayout:
    shape: tuple
    stride: tuple
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def of(cls, tensor):
        return cls(tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device)

    def placeholder(self):
        return torch.empty_strided(
            self.shape, self.stride, dtype=self.dtype, device="meta"
        )


@dataclass(frozen=True)
class _ScaleTemplates:
    planes: tuple
    active: _ScaleLayout
    aliases: tuple


def _scale_templates(experts):
    metadata = getattr(experts, "_csf_scale_templates", None)
    if metadata is not None:
        return metadata
    planes = (experts.w1_blockscale, experts.w2_blockscale)
    aliases = []
    if experts.representation is not None:
        payload = experts.representation.value
        for names, index, plane in (
            (("w13_scale", "w13_sfb", "micro_w13_scale"), 0, planes[0]),
            (("w2_scale", "w2_sfb", "micro_w2_scale"), 1, planes[1]),
        ):
            for name in names:
                tensor = getattr(payload, name, None)
                if (
                    isinstance(tensor, torch.Tensor)
                    and tensor.untyped_storage().data_ptr()
                    == plane.untyped_storage().data_ptr()
                ):
                    offset = (
                        tensor.storage_offset() * tensor.element_size()
                        - plane.storage_offset() * plane.element_size()
                    )
                    aliases.append((name, index, offset, _ScaleLayout.of(tensor)))
    return _ScaleTemplates(
        tuple(_ScaleLayout.of(t) for t in planes),
        _ScaleLayout.of(experts.nvfp4_csf.active),
        tuple(aliases),
    )


def release_weight_views(experts):
    """Evict declaration views borrowing the prepared expert weight destinations."""
    from . import _impl

    pointers = experts.w1_fp4.data_ptr(), experts.w2_fp4.data_ptr()
    for key in tuple(_impl._WEIGHT_CACHE):
        if len(key) >= 4 and (key[0], key[2]) == pointers:
            _impl._WEIGHT_CACHE.pop(key)
    key = _impl._LAST_WEIGHTS[0]
    if key is not None and len(key) >= 4 and (key[0], key[2]) == pointers:
        _impl._LAST_WEIGHTS = (None, None)


def materialize_scale_templates(experts):
    """Allocate CSF declaration planes when reusable weights enter preparation."""
    impl = experts._impl
    metadata = getattr(impl, "_csf_scale_templates", None)
    if metadata is None or not impl.w1_blockscale.is_meta:
        return
    planes = [
        torch.empty_strided(
            layout.shape, layout.stride, dtype=layout.dtype, device=layout.device
        )
        for layout in metadata.planes
    ]
    layout = metadata.active
    active = torch.empty_strided(
        layout.shape, layout.stride, dtype=layout.dtype, device=layout.device
    ).zero_()
    rebound = _scale_view(
        impl,
        [*(plane.view(torch.uint8) for plane in planes), active],
        replace(impl.nvfp4_csf, active=active),
    )
    for name in ("w1_blockscale", "w2_blockscale", "representation", "nvfp4_csf"):
        object.__setattr__(impl, name, getattr(rebound, name))


def release_scale_templates(experts, plan):
    """Release expanded CSF destinations after every prepared binding has finalized.

    Functional invocations must supply scale planes before binding. Compressed
    records, weights, programs, and immutable scale layouts remain owned.
    """

    def release_impl(impl):
        release_weight_views(impl)
        if impl.nvfp4_csf is None or impl.w1_blockscale.is_meta:
            return
        metadata = _scale_templates(impl)
        object.__setattr__(impl, "_csf_scale_templates", metadata)
        planes = tuple(layout.placeholder() for layout in metadata.planes)
        object.__setattr__(impl, "w1_blockscale", planes[0])
        object.__setattr__(impl, "w2_blockscale", planes[1])
        if impl.representation is not None:
            payload = copy(impl.representation.value)
            for name, index, offset, layout in metadata.aliases:
                object.__setattr__(
                    payload,
                    name,
                    planes[index]
                    .view(layout.dtype)
                    .as_strided(
                        layout.shape, layout.stride, offset // layout.dtype.itemsize
                    ),
                )
            object.__setattr__(
                impl, "representation", replace(impl.representation, value=payload)
            )
        object.__setattr__(
            impl,
            "nvfp4_csf",
            replace(impl.nvfp4_csf, active=metadata.active.placeholder()),
        )

    def release_state(state):
        from ._preparation import _FusedMoeCapacityState, _FusedMoeState
        from ._token_outer import State

        if isinstance(state, _FusedMoeCapacityState):
            for variant in state.variants.values():
                release_state(variant)
        elif isinstance(state, _FusedMoeState):
            release_impl(state.experts._impl)
            if state.execution_experts is not None:
                release_impl(state.execution_experts)
        elif isinstance(state, State):
            release_impl(state.experts._impl)
            if state.scale_decoder is not None:
                layout = _ScaleLayout.of(state.scale_decoder.active)
                object.__setattr__(
                    state,
                    "scale_decoder",
                    replace(state.scale_decoder, active=layout.placeholder()),
                )

    release_impl(experts._impl)
    release_state(require_prepared(plan, "moe.decode"))


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
    metadata = _scale_templates(experts)
    buffers = [
        buffers[0].view(metadata.planes[0].dtype),
        buffers[1].view(metadata.planes[1].dtype),
        buffers[2],
    ]
    for source, target in zip(metadata.planes, buffers[:2]):
        if (
            target.shape != source.shape
            or target.stride() != source.stride
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
        for name, index, offset, layout in metadata.aliases:
            after = buffers[index].view(layout.dtype)
            updates[name] = after.as_strided(
                layout.shape,
                layout.stride,
                after.storage_offset() + offset // layout.dtype.itemsize,
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
    experts = (
        binding.experts if hasattr(binding, "experts") else binding.state.experts._impl
    )
    active = _scale_templates(experts).active
    if (
        buffers[2].shape != active.shape
        or buffers[2].stride() != active.stride
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
        _scale_buffers=scale_storage or scale_inputs,
        _weight_views_cache={},
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


def supports_prequant_mxfp8(plan, tokens: int) -> bool:
    """Return True if the prepared MoE plan accepts pre-quantized MXFP8 activations."""
    prepared = getattr(plan, "_prepared", None)
    state = getattr(prepared, "state", None)
    if state is None:
        return False
    variant = state.for_tokens(int(tokens)) if hasattr(state, "for_tokens") else state
    return bool(
        getattr(variant, "compact_launches", None) is not None
        and not getattr(variant.config, "compact_fuse_input_quantization", False)
        and int(variant.scratch.caps.max_tokens) == int(tokens)
    )


@torch.library.custom_op(
    "b12x::prepared_moe_prequant_out",
    mutates_args=("output", "scratch", "activation_amax", "scale_storage"),
)
def prepared_moe_prequant_out(
    a: torch.Tensor,
    prequant_a_values: torch.Tensor,
    prequant_a_scales: torch.Tensor,
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
        prequant_a_values=prequant_a_values,
        prequant_a_scales=prequant_a_scales,
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
        _scale_buffers=scale_storage or scale_inputs,
        _weight_views_cache={},
    )
    state.run(_bind_scale_buffers(binding, scale_storage or scale_inputs))


@prepared_moe_prequant_out.register_fake
def _fake_prequant(
    a,
    prequant_a_values,
    prequant_a_scales,
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
        _scale_buffers=scale_storage or scale_inputs,
        _weight_views_cache={},
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
