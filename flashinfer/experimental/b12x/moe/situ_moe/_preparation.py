# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared lifecycle for SM120/SM121 fused route-weighted SiTU MoE."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_tensor

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec, scratch_buffer_spec, scratch_tensor
from b12x._lib.utils.device import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._tuning import (
    TUNING,
    SituMoeConfig,
    SituMoeQuery,
    validate_config,
    validate_query,
)

_COMPILE_VERSION = 2


def _fake_operand(dtype, rank: int, align: int):
    shape = tuple(cute.sym_int(32) for _ in range(rank))
    strides = tuple(1 if i == rank - 1 else cute.sym_int(64) for i in range(rank))
    return make_fake_tensor(dtype, shape, strides, assumed_align=align)


def _kernel_operand(tensor: torch.Tensor, dtype, align: int):
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"situ_moe operands require unit inner stride, got {tensor.stride()}"
        )
    converted = from_dlpack(tensor, assumed_align=align)
    converted.element_type = dtype
    return converted.mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _compile_key(query: SituMoeQuery, config: SituMoeConfig) -> tuple:
    fields = {**TUNING.encode_query(query), **config.to_dict()}
    excluded = {"sm_count", "autotune_effort"}
    if not config.use_decode_kernel:
        excluded.add("max_tokens")
    return tuple(sorted((k, v) for k, v in fields.items() if k not in excluded))


@program_cache(scope="preparation")
def compile_situ_moe(query_payload, config_payload, ordinal):
    from ._decode_mxfp8 import SituDecodeFC1MXFP8Kernel, SituDecodeFC2MXFP8Kernel
    from ._decode_nvfp4 import SituDecodeFC1NVFP4Kernel, SituDecodeFC2NVFP4Kernel
    from ._prefill import (
        SituPrefillBuildTasksKernel,
        SituPrefillMXFP8Kernel,
        SituPrefillNVFP4Kernel,
        SituPrefillRouteClearKernel,
        SituPrefillRoutePackKernel,
    )
    from ._quant import (
        SituInputQuantMXFP8Kernel,
        SituInputQuantNVFP4Kernel,
        SituRouteSumKernel,
        SituSmallRoutePackKernel,
    )

    query = SituMoeQuery(**dict(query_payload))
    config = SituMoeConfig.from_config(FrozenMapping(config_payload))
    validate_query(query)
    validate_config(query, config)
    key = _compile_key(query, config)
    programs = {}
    is_nvfp4 = query.weight_format == "nvfp4"

    with torch.cuda.device(ordinal):
        stream = current_cuda_stream()
        # 1. Input & intermediate quantizers
        if is_nvfp4:
            q1_args = (
                _fake_operand(cutlass.BFloat16, 2, 16),
                _fake_operand(cutlass.Uint8, 2, 16),
                _fake_operand(cutlass.Uint8, 2, 8),
                _fake_operand(cutlass.Float32, 1, 4),
            )
            programs["quant_fc1"] = b12x_compile(
                SituInputQuantNVFP4Kernel(query.hidden),
                *q1_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.quant_nvfp4_fc1", _COMPILE_VERSION, key
                ),
            )
            programs["quant_fc2"] = b12x_compile(
                SituInputQuantNVFP4Kernel(query.intermediate),
                *q1_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.quant_nvfp4_fc2", _COMPILE_VERSION, key
                ),
            )
        else:
            q_args = (
                _fake_operand(cutlass.BFloat16, 2, 16),
                _fake_operand(cutlass.Uint8, 2, 16),
                _fake_operand(cutlass.Uint8, 2, 8),
            )
            programs["quant_fc1"] = b12x_compile(
                SituInputQuantMXFP8Kernel(query.hidden),
                *q_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.quant_mxfp8_fc1", _COMPILE_VERSION, key
                ),
            )
            programs["quant_fc2"] = b12x_compile(
                SituInputQuantMXFP8Kernel(query.intermediate),
                *q_args,
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.quant_mxfp8_fc2", _COMPILE_VERSION, key
                ),
            )

        # 2. Route reduction (only needed when FC2 does not fuse scatter-add)
        if not config.fused_route_sum:
            programs["route_sum"] = b12x_compile(
                SituRouteSumKernel(query.hidden, query.top_k),
                _fake_operand(cutlass.BFloat16, 2, 16),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.BFloat16, 2, 16),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.route_sum", _COMPILE_VERSION, key
                ),
            )

        # 3. Route grouping + FC1/FC2 compute kernels
        if config.use_decode_kernel:
            programs["small_pack"] = b12x_compile(
                SituSmallRoutePackKernel(query.num_experts, query.max_routes),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 2, 4),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.small_pack", _COMPILE_VERSION, key
                ),
            )
            if is_nvfp4:
                fc1_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc1"] = b12x_compile(
                    SituDecodeFC1NVFP4Kernel(
                        hidden=query.hidden,
                        intermediate=query.intermediate,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        max_tokens=query.max_tokens,
                        has_csf=query.has_csf,
                    ),
                    *fc1_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.decode_fc1_nvfp4", _COMPILE_VERSION, key
                    ),
                )
                fc2_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc2"] = b12x_compile(
                    SituDecodeFC2NVFP4Kernel(
                        hidden=query.hidden,
                        intermediate=query.intermediate,
                        num_experts=query.num_experts,
                        max_tokens=query.max_tokens,
                        has_csf=query.has_csf,
                    ),
                    *fc2_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.decode_fc2_nvfp4", _COMPILE_VERSION, key
                    ),
                )
            else:
                fc1_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 3, 8),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc1"] = b12x_compile(
                    SituDecodeFC1MXFP8Kernel(
                        hidden=query.hidden,
                        intermediate=query.intermediate,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        max_tokens=query.max_tokens,
                    ),
                    *fc1_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.decode_fc1_mxfp8", _COMPILE_VERSION, key
                    ),
                )
                fc2_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 3, 8),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc2"] = b12x_compile(
                    SituDecodeFC2MXFP8Kernel(
                        hidden=query.hidden,
                        intermediate=query.intermediate,
                        num_experts=query.num_experts,
                        max_tokens=query.max_tokens,
                    ),
                    *fc2_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.decode_fc2_mxfp8", _COMPILE_VERSION, key
                    ),
                )
        else:
            bm = config.prefill_tile_m
            max_tasks = (query.max_routes + bm - 1) // bm + query.num_experts
            programs["route_clear"] = b12x_compile(
                SituPrefillRouteClearKernel(query.num_experts, max_tasks),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.route_clear",
                    _COMPILE_VERSION,
                    (query.num_experts, max_tasks),
                ),
            )
            programs["route_pack"] = b12x_compile(
                SituPrefillRoutePackKernel(query.num_experts, query.max_routes),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 2, 4),
                cutlass.Int32(query.max_routes),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.route_pack",
                    _COMPILE_VERSION,
                    (query.num_experts, query.max_routes),
                ),
            )
            programs["build_tasks"] = b12x_compile(
                SituPrefillBuildTasksKernel(query.num_experts, bm),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                _fake_operand(cutlass.Int32, 1, 4),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    "moe.situ_moe.build_tasks",
                    _COMPILE_VERSION,
                    (query.num_experts, bm),
                ),
            )
            if is_nvfp4:
                prefill_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc1"] = b12x_compile(
                    SituPrefillNVFP4Kernel(
                        n_dim=2 * query.intermediate,
                        k_dim=query.hidden,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        is_fc1=True,
                        has_csf=query.has_csf,
                        tile_m=bm,
                    ),
                    *prefill_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.prefill_fc1_nvfp4", _COMPILE_VERSION, key
                    ),
                )
                programs["fc2"] = b12x_compile(
                    SituPrefillNVFP4Kernel(
                        n_dim=query.hidden,
                        k_dim=query.intermediate,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        is_fc1=False,
                        has_csf=query.has_csf,
                        tile_m=bm,
                        fused_route_sum=config.fused_route_sum,
                    ),
                    *prefill_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.prefill_fc2_nvfp4", _COMPILE_VERSION, key
                    ),
                )
            else:
                prefill_args = (
                    _fake_operand(cutlass.Uint8, 2, 16),
                    _fake_operand(cutlass.Uint8, 2, 8),
                    _fake_operand(cutlass.Uint8, 1, 16),
                    _fake_operand(cutlass.Uint8, 3, 8),
                    _fake_operand(cutlass.Float32, 1, 4),
                    _fake_operand(cutlass.Int32, 2, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.Int32, 1, 4),
                    _fake_operand(cutlass.BFloat16, 2, 16),
                )
                programs["fc1"] = b12x_compile(
                    SituPrefillMXFP8Kernel(
                        n_dim=2 * query.intermediate,
                        k_dim=query.hidden,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        is_fc1=True,
                        tile_m=bm,
                    ),
                    *prefill_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.prefill_fc1_mxfp8", _COMPILE_VERSION, key
                    ),
                )
                programs["fc2"] = b12x_compile(
                    SituPrefillMXFP8Kernel(
                        n_dim=query.hidden,
                        k_dim=query.intermediate,
                        num_experts=query.num_experts,
                        top_k=query.top_k,
                        is_fc1=False,
                        tile_m=bm,
                        fused_route_sum=config.fused_route_sum,
                    ),
                    *prefill_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        "moe.situ_moe.prefill_fc2_mxfp8", _COMPILE_VERSION, key
                    ),
                )

    return programs


def _scratch_layout(
    query: SituMoeQuery, config: SituMoeConfig
) -> tuple[dict[str, tuple[int, tuple[int, ...], torch.dtype]], int]:
    """Compute 256B-aligned byte offsets inside one single flat uint8 scratch arena."""
    m = query.max_tokens
    r = query.max_routes
    h = query.hidden
    i = query.intermediate
    e = query.num_experts
    is_nvfp4 = query.weight_format == "nvfp4"

    entries: list[tuple[str, tuple[int, ...], torch.dtype]] = [
        ("aq1", (m, h // 2 if is_nvfp4 else h), torch.uint8),
        ("as1", (m, h // (16 if is_nvfp4 else 32)), torch.uint8),
        ("inter", (r, i), torch.bfloat16),
        ("aq2", (r, i // 2 if is_nvfp4 else i), torch.uint8),
        ("as2", (r, i // (16 if is_nvfp4 else 32)), torch.uint8),
    ]
    if not config.fused_route_sum:
        entries.append(("route_out", (r, h), torch.bfloat16))
    if is_nvfp4:
        entries.append(("aouter1", (m,), torch.float32))
        entries.append(("aouter2", (r,), torch.float32))
    if config.use_decode_kernel:
        max_slots = min(e, r)
        entries.extend(
            [
                ("active_experts", (max_slots,), torch.int32),
                ("active_counts", (max_slots,), torch.int32),
                ("active_routes", (max_slots, r), torch.int32),
            ]
        )
    else:
        bm = config.prefill_tile_m
        max_tasks = (r + bm - 1) // bm + e
        entries.extend(
            [
                ("counts", (e,), torch.int32),
                ("rows", (e, r), torch.int32),
                ("task_expert", (max_tasks,), torch.int32),
                ("task_begin", (max_tasks,), torch.int32),
            ]
        )

    layout: dict[str, tuple[int, tuple[int, ...], torch.dtype]] = {}
    offset = 0
    for name, shape, dtype in entries:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * dtype.itemsize
        layout[name] = (offset, shape, dtype)
        offset = ((offset + nbytes + 255) // 256) * 256
    return layout, max(offset, 256)


def scratch_specs(
    query: SituMoeQuery, config: SituMoeConfig, device: torch.device
) -> tuple[ScratchBufferSpec, ...]:
    """Caller-owned scratch specification (zero scale_scratch even with CSF!)."""
    _, total_bytes = _scratch_layout(query, config)
    return (
        scratch_buffer_spec("situ_moe_workspace", nbytes=total_bytes, device=device),
    )


def _slice_scratch(
    arena: torch.Tensor, layout: dict[str, tuple[int, tuple[int, ...], torch.dtype]]
) -> dict[str, torch.Tensor]:
    views: dict[str, torch.Tensor] = {}
    for name, (offset, shape, dtype) in layout.items():
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * dtype.itemsize
        views[name] = arena[offset : offset + nbytes].view(dtype).view(shape)
    return views


@dataclass(frozen=True)
class Binding:
    """Bound tensors for one prepared ``situ_moe`` invocation."""

    plan: Plan
    x: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    output: torch.Tensor
    g13: torch.Tensor | None
    g2: torch.Tensor | None
    s13_exc: torch.Tensor | None
    s13_slab_offsets: torch.Tensor | None
    s2_exc: torch.Tensor | None
    s2_slab_offsets: torch.Tensor | None
    scratch_views: dict[str, torch.Tensor]


class PreparedSituMoe:
    """Compiled runtime state for one prepared SiTU MoE family."""

    def __init__(
        self,
        *,
        query: SituMoeQuery,
        config: SituMoeConfig,
        device: torch.device,
        programs: dict,
        dummy_u32: torch.Tensor,
        dummy_i32: torch.Tensor,
    ):
        self.query = query
        self.config = config
        self.device = device
        self.programs = programs
        self.dummy_u32 = dummy_u32
        self.dummy_i32 = dummy_i32
        self._specs = scratch_specs(query, config, device)
        self._layout, _ = _scratch_layout(query, config)

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._specs

    def bind(
        self,
        *,
        plan: Plan,
        scratch,
        x: torch.Tensor,
        ids: torch.Tensor,
        weights: torch.Tensor,
        w13: torch.Tensor,
        s13: torch.Tensor,
        w2: torch.Tensor,
        s2: torch.Tensor,
        output: torch.Tensor,
        g13: torch.Tensor | None = None,
        g2: torch.Tensor | None = None,
        s13_exc: torch.Tensor | None = None,
        s13_slab_offsets: torch.Tensor | None = None,
        s2_exc: torch.Tensor | None = None,
        s2_slab_offsets: torch.Tensor | None = None,
    ) -> Binding:
        q = self.query
        if x.device != self.device or output.device != self.device:
            raise ValueError(f"x and output must be on {self.device}")
        if x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
            raise ValueError("x and output must be bfloat16")
        if x.ndim != 2 or x.shape[1] != q.hidden:
            raise ValueError(f"x must have shape [M, {q.hidden}], got {tuple(x.shape)}")
        if x.shape[0] > q.max_tokens:
            raise ValueError(
                f"x rows {x.shape[0]} exceed planned max_tokens {q.max_tokens}"
            )
        if output.shape != x.shape:
            raise ValueError(
                f"output shape {tuple(output.shape)} must match x {tuple(x.shape)}"
            )
        if ids.dtype != torch.int32 or weights.dtype != torch.float32:
            raise ValueError("ids must be int32 and weights must be float32")
        if (
            ids.numel() != x.shape[0] * q.top_k
            or weights.numel() != x.shape[0] * q.top_k
        ):
            raise ValueError("ids and weights must have M * top_k elements")
        if q.weight_format == "nvfp4" and (g13 is None or g2 is None):
            raise ValueError("nvfp4 situ_moe requires g13 and g2 global scale tensors")
        if q.has_csf:
            if (
                s13_exc is None
                or s13_slab_offsets is None
                or s2_exc is None
                or s2_slab_offsets is None
            ):
                raise ValueError(
                    "CSF situ_moe requires s13_exc, s13_slab_offsets, s2_exc, and s2_slab_offsets"
                )
        arena = scratch_tensor(scratch, self._specs, owner="situ_moe")
        views = _slice_scratch(arena, self._layout)
        return Binding(
            plan=plan,
            x=x,
            ids=ids.reshape(-1),
            weights=weights.reshape(-1),
            w13=w13,
            s13=s13,
            w2=w2,
            s2=s2,
            output=output,
            g13=g13,
            g2=g2,
            s13_exc=s13_exc,
            s13_slab_offsets=s13_slab_offsets,
            s2_exc=s2_exc,
            s2_slab_offsets=s2_slab_offsets,
            scratch_views=views,
        )

    def run(self, binding: Binding) -> torch.Tensor:
        q = self.query
        cfg = self.config
        m = int(binding.x.shape[0])
        if m == 0:
            return binding.output
        r = m * q.top_k
        sv = binding.scratch_views
        is_nvfp4 = q.weight_format == "nvfp4"

        aq1 = sv["aq1"][:m]
        as1 = sv["as1"][:m]
        inter = sv["inter"][:r]
        aq2 = sv["aq2"][:r]
        as2 = sv["as2"][:r]
        route_out = None if cfg.fused_route_sum else sv["route_out"][:r]
        fc2_out = binding.output if cfg.fused_route_sum else route_out

        w13_flat = binding.w13.view(torch.uint8).reshape(-1)
        w2_flat = binding.w2.view(torch.uint8).reshape(-1)

        with torch.cuda.device(self.device):
            stream = current_cuda_stream()
            # 1. Quantize input [M, H]
            if is_nvfp4:
                aouter1 = sv["aouter1"][:m]
                aouter2 = sv["aouter2"][:r]
                run_compiled(
                    self.programs["quant_fc1"],
                    (
                        _kernel_operand(binding.x, cutlass.BFloat16, 16),
                        _kernel_operand(aq1, cutlass.Uint8, 16),
                        _kernel_operand(as1, cutlass.Uint8, 8),
                        _kernel_operand(aouter1, cutlass.Float32, 4),
                        stream,
                    ),
                )
            else:
                run_compiled(
                    self.programs["quant_fc1"],
                    (
                        _kernel_operand(binding.x, cutlass.BFloat16, 16),
                        _kernel_operand(aq1, cutlass.Uint8, 16),
                        _kernel_operand(as1, cutlass.Uint8, 8),
                        stream,
                    ),
                )

            # 2. Group routes + FC1 (fused SiTU) + quantize inter + FC2
            if cfg.use_decode_kernel:
                max_slots = min(q.num_experts, r)
                act_exp = sv["active_experts"][:max_slots]
                act_cnt = sv["active_counts"][:max_slots]
                act_rts = sv["active_routes"][:max_slots, : q.max_routes]
                run_compiled(
                    self.programs["small_pack"],
                    (
                        _kernel_operand(binding.ids, cutlass.Int32, 4),
                        _kernel_operand(act_exp, cutlass.Int32, 4),
                        _kernel_operand(act_cnt, cutlass.Int32, 4),
                        _kernel_operand(act_rts, cutlass.Int32, 4),
                        stream,
                    ),
                )
                if is_nvfp4:
                    s13_flat = binding.s13.view(torch.uint8).reshape(-1)
                    s2_flat = binding.s2.view(torch.uint8).reshape(-1)
                    exc1 = (
                        binding.s13_exc
                        if q.has_csf and binding.s13_exc.numel() > 0
                        else self.dummy_u32
                    )
                    off1 = binding.s13_slab_offsets if q.has_csf else self.dummy_i32
                    exc2 = (
                        binding.s2_exc
                        if q.has_csf and binding.s2_exc.numel() > 0
                        else self.dummy_u32
                    )
                    off2 = binding.s2_slab_offsets if q.has_csf else self.dummy_i32
                    run_compiled(
                        self.programs["fc1"],
                        (
                            _kernel_operand(aq1, cutlass.Uint8, 16),
                            _kernel_operand(as1, cutlass.Uint8, 8),
                            _kernel_operand(aouter1, cutlass.Float32, 4),
                            _kernel_operand(w13_flat, cutlass.Uint8, 16),
                            _kernel_operand(s13_flat, cutlass.Uint8, 16),
                            _kernel_operand(exc1, cutlass.Uint32, 4),
                            _kernel_operand(off1, cutlass.Int32, 4),
                            _kernel_operand(
                                binding.g13.reshape(-1), cutlass.Float32, 4
                            ),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(act_exp, cutlass.Int32, 4),
                            _kernel_operand(act_cnt, cutlass.Int32, 4),
                            _kernel_operand(act_rts, cutlass.Int32, 4),
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["quant_fc2"],
                        (
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(aouter2, cutlass.Float32, 4),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["fc2"],
                        (
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(aouter2, cutlass.Float32, 4),
                            _kernel_operand(w2_flat, cutlass.Uint8, 16),
                            _kernel_operand(s2_flat, cutlass.Uint8, 16),
                            _kernel_operand(exc2, cutlass.Uint32, 4),
                            _kernel_operand(off2, cutlass.Int32, 4),
                            _kernel_operand(binding.g2.reshape(-1), cutlass.Float32, 4),
                            _kernel_operand(act_exp, cutlass.Int32, 4),
                            _kernel_operand(act_cnt, cutlass.Int32, 4),
                            _kernel_operand(act_rts, cutlass.Int32, 4),
                            _kernel_operand(route_out, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                else:
                    s13_u8 = binding.s13.view(torch.uint8)
                    s2_u8 = binding.s2.view(torch.uint8)
                    run_compiled(
                        self.programs["fc1"],
                        (
                            _kernel_operand(aq1, cutlass.Uint8, 16),
                            _kernel_operand(as1, cutlass.Uint8, 8),
                            _kernel_operand(w13_flat, cutlass.Uint8, 16),
                            _kernel_operand(s13_u8, cutlass.Uint8, 8),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(act_exp, cutlass.Int32, 4),
                            _kernel_operand(act_cnt, cutlass.Int32, 4),
                            _kernel_operand(act_rts, cutlass.Int32, 4),
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["quant_fc2"],
                        (
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["fc2"],
                        (
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(w2_flat, cutlass.Uint8, 16),
                            _kernel_operand(s2_u8, cutlass.Uint8, 8),
                            _kernel_operand(act_exp, cutlass.Int32, 4),
                            _kernel_operand(act_cnt, cutlass.Int32, 4),
                            _kernel_operand(act_rts, cutlass.Int32, 4),
                            _kernel_operand(route_out, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
            else:
                counts = sv["counts"]
                rows = sv["rows"]
                bm = cfg.prefill_tile_m
                cur_tasks = (r + bm - 1) // bm + q.num_experts
                task_exp = sv["task_expert"][:cur_tasks]
                task_beg = sv["task_begin"][:cur_tasks]
                run_compiled(
                    self.programs["route_clear"],
                    (
                        _kernel_operand(counts, cutlass.Int32, 4),
                        _kernel_operand(sv["task_expert"], cutlass.Int32, 4),
                        stream,
                    ),
                )
                run_compiled(
                    self.programs["route_pack"],
                    (
                        _kernel_operand(binding.ids, cutlass.Int32, 4),
                        _kernel_operand(counts, cutlass.Int32, 4),
                        _kernel_operand(rows, cutlass.Int32, 4),
                        cutlass.Int32(r),
                        stream,
                    ),
                )
                run_compiled(
                    self.programs["build_tasks"],
                    (
                        _kernel_operand(counts, cutlass.Int32, 4),
                        _kernel_operand(task_exp, cutlass.Int32, 4),
                        _kernel_operand(task_beg, cutlass.Int32, 4),
                        stream,
                    ),
                )
                if is_nvfp4:
                    s13_flat = binding.s13.view(torch.uint8).reshape(-1)
                    s2_flat = binding.s2.view(torch.uint8).reshape(-1)
                    exc1 = (
                        binding.s13_exc
                        if q.has_csf and binding.s13_exc.numel() > 0
                        else self.dummy_u32
                    )
                    off1 = binding.s13_slab_offsets if q.has_csf else self.dummy_i32
                    exc2 = (
                        binding.s2_exc
                        if q.has_csf and binding.s2_exc.numel() > 0
                        else self.dummy_u32
                    )
                    off2 = binding.s2_slab_offsets if q.has_csf else self.dummy_i32
                    run_compiled(
                        self.programs["fc1"],
                        (
                            _kernel_operand(aq1, cutlass.Uint8, 16),
                            _kernel_operand(as1, cutlass.Uint8, 8),
                            _kernel_operand(aouter1, cutlass.Float32, 4),
                            _kernel_operand(w13_flat, cutlass.Uint8, 16),
                            _kernel_operand(s13_flat, cutlass.Uint8, 16),
                            _kernel_operand(exc1, cutlass.Uint32, 4),
                            _kernel_operand(off1, cutlass.Int32, 4),
                            _kernel_operand(
                                binding.g13.reshape(-1), cutlass.Float32, 4
                            ),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(rows, cutlass.Int32, 4),
                            _kernel_operand(counts, cutlass.Int32, 4),
                            _kernel_operand(task_exp, cutlass.Int32, 4),
                            _kernel_operand(task_beg, cutlass.Int32, 4),
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["quant_fc2"],
                        (
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(aouter2, cutlass.Float32, 4),
                            stream,
                        ),
                    )
                    if cfg.fused_route_sum:
                        binding.output.zero_()
                    run_compiled(
                        self.programs["fc2"],
                        (
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(aouter2, cutlass.Float32, 4),
                            _kernel_operand(w2_flat, cutlass.Uint8, 16),
                            _kernel_operand(s2_flat, cutlass.Uint8, 16),
                            _kernel_operand(exc2, cutlass.Uint32, 4),
                            _kernel_operand(off2, cutlass.Int32, 4),
                            _kernel_operand(binding.g2.reshape(-1), cutlass.Float32, 4),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(rows, cutlass.Int32, 4),
                            _kernel_operand(counts, cutlass.Int32, 4),
                            _kernel_operand(task_exp, cutlass.Int32, 4),
                            _kernel_operand(task_beg, cutlass.Int32, 4),
                            _kernel_operand(fc2_out, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                else:
                    s13_u8 = binding.s13.view(torch.uint8)
                    s2_u8 = binding.s2.view(torch.uint8)
                    run_compiled(
                        self.programs["fc1"],
                        (
                            _kernel_operand(aq1, cutlass.Uint8, 16),
                            _kernel_operand(as1, cutlass.Uint8, 8),
                            _kernel_operand(w13_flat, cutlass.Uint8, 16),
                            _kernel_operand(s13_u8, cutlass.Uint8, 8),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(rows, cutlass.Int32, 4),
                            _kernel_operand(counts, cutlass.Int32, 4),
                            _kernel_operand(task_exp, cutlass.Int32, 4),
                            _kernel_operand(task_beg, cutlass.Int32, 4),
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )
                    run_compiled(
                        self.programs["quant_fc2"],
                        (
                            _kernel_operand(inter, cutlass.BFloat16, 16),
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            stream,
                        ),
                    )
                    if cfg.fused_route_sum:
                        binding.output.zero_()
                    run_compiled(
                        self.programs["fc2"],
                        (
                            _kernel_operand(aq2, cutlass.Uint8, 16),
                            _kernel_operand(as2, cutlass.Uint8, 8),
                            _kernel_operand(w2_flat, cutlass.Uint8, 16),
                            _kernel_operand(s2_u8, cutlass.Uint8, 8),
                            _kernel_operand(binding.weights, cutlass.Float32, 4),
                            _kernel_operand(rows, cutlass.Int32, 4),
                            _kernel_operand(counts, cutlass.Int32, 4),
                            _kernel_operand(task_exp, cutlass.Int32, 4),
                            _kernel_operand(task_beg, cutlass.Int32, 4),
                            _kernel_operand(fc2_out, cutlass.BFloat16, 16),
                            stream,
                        ),
                    )

            # 3. Deterministic FP32 sum across top-K routes -> BF16 [M, H]
            if not cfg.fused_route_sum:
                run_compiled(
                    self.programs["route_sum"],
                    (
                        _kernel_operand(route_out, cutlass.BFloat16, 16),
                        _kernel_operand(binding.ids, cutlass.Int32, 4),
                        _kernel_operand(binding.output, cutlass.BFloat16, 16),
                        stream,
                    ),
                )

        return binding.output


def make_plan(caps, *, invocation=None, override=None) -> Plan:
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be situ_moe.Caps")
    if invocation:
        raise ValueError("situ_moe declarations carry no invocation metadata")
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    query = SituMoeQuery(
        weight_format=caps.weight_format,
        hidden=int(caps.hidden),
        intermediate=int(caps.intermediate),
        num_experts=int(caps.num_experts),
        top_k=int(caps.top_k),
        max_tokens=int(caps.max_tokens),
        has_csf=bool(caps.has_csf),
        deterministic_output=bool(caps.deterministic_output),
        sm_count=int(sm_count),
    )
    validate_query(query)

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.moe.situ_moe._preparation:compile_situ_moe",
                TUNING.encode_query(query),
                config.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def mem_reqs(config, detected):
        specs = scratch_specs(query, config, torch.device("cuda", detected.ordinal))
        return MemoryRequirements(scratch=specs)

    def materialize(selection, detected):
        cfg = selection.config
        programs = compile_situ_moe(
            TUNING.encode_query(query), cfg.to_dict(), detected.ordinal
        )
        load_programs(programs)
        dev = torch.device("cuda", detected.ordinal)
        state = PreparedSituMoe(
            query=query,
            config=cfg,
            device=dev,
            programs=programs,
            dummy_u32=torch.zeros((1,), dtype=torch.uint32, device=dev),
            dummy_i32=torch.zeros((1,), dtype=torch.int32, device=dev),
        )
        return attach_programs(state, *programs.values())

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        _device=device,
        _compile_jobs=jobs,
        _memory_requirements=mem_reqs,
        _materialize=materialize,
    )
