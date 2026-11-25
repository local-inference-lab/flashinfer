# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared lifecycle for SM120 paged NVFP4 (4-or-6 MAE K) attention and KV writes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack, make_fake_tensor

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.utils.device import current_cuda_stream
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan

from ._prefill_kernel import REL_RANK
from ._tuning import (
    TUNING,
    PagedNVFP4Config,
    PagedNVFP4Query,
    _consumer_warps,
    validate_config,
    validate_query,
)

_DECODE_ID = "attention.paged_nvfp4.decode"
_MERGE_ID = "attention.paged_nvfp4.merge"
_PREFILL_ID = "attention.paged_nvfp4.prefill"
_KV_WRITE_ID = "attention.paged_nvfp4.kv_write"
_COMPILE_VERSION = 5

_DECODE_OPERANDS = (
    ("q", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, cutlass.Uint8, 16),
    ("k_scales", 4, cutlass.Uint8, 8),
    ("v_cache", 4, cutlass.Uint8, 16),
    ("v_scales", 4, cutlass.Uint8, 8),
    ("page_table", 2, cutlass.Int32, 4),
    ("cache_seqlens", 1, cutlass.Int32, 4),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("output", 3, cutlass.BFloat16, 16),
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
    ("sinks", "sinks", cutlass.Float32, 4),
    ("relative_bias", 3, cutlass.BFloat16, 2),
)

_PREFILL_OPERANDS = (
    ("q", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, cutlass.Uint8, 16),
    ("k_scales", 4, cutlass.Uint8, 8),
    ("v_cache", 4, cutlass.Uint8, 16),
    ("v_scales", 4, cutlass.Uint8, 8),
    ("page_table", 2, cutlass.Int32, 4),
    ("cache_seqlens", 1, cutlass.Int32, 4),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("output", 3, cutlass.BFloat16, 16),
    ("sinks", "sinks", cutlass.Float32, 4),
    ("relative_bias_factors", 3, cutlass.BFloat16, 4),
    ("relative_bias_projection", 2, cutlass.BFloat16, 4),
)

_MERGE_OPERANDS = (
    ("o_part", 4, cutlass.Float32, 16),
    ("lse_part", 3, cutlass.Float32, 4),
    ("cache_seqlens", 1, cutlass.Int32, 4),
    ("cu_seqlens_q", 1, cutlass.Int32, 4),
    ("output", 3, cutlass.BFloat16, 16),
    ("sinks", "sinks", cutlass.Float32, 4),
)

_KV_WRITE_OPERANDS = (
    ("key", 3, cutlass.BFloat16, 16),
    ("value", 3, cutlass.BFloat16, 16),
    ("k_cache", 4, cutlass.Uint8, 16),
    ("k_scales", 4, cutlass.Uint8, 8),
    ("v_cache", 4, cutlass.Uint8, 16),
    ("v_scales", 4, cutlass.Uint8, 8),
    ("slot_mapping", 1, cutlass.Int64, 8),
)


def _operand_rank(rank, query: PagedNVFP4Query) -> int:
    if rank == "sinks":
        return 2 if query.sink_layout == "row" else 1
    return rank


def _fake_operand(dtype, rank: int, align: int):
    shape = tuple(cute.sym_int(32) for _ in range(rank))
    strides = tuple(1 if i == rank - 1 else cute.sym_int(64) for i in range(rank))
    return make_fake_tensor(dtype, shape, strides, assumed_align=align)


def _kernel_operand(tensor: torch.Tensor, dtype, align: int):
    if tensor.stride(-1) != 1:
        raise ValueError(
            f"paged_nvfp4 operands require unit inner stride, got {tensor.stride()}"
        )
    converted = from_dlpack(tensor, assumed_align=align)
    converted.element_type = dtype
    return converted.mark_layout_dynamic(leading_dim=tensor.ndim - 1)


def _pack_operands(spec, tensors: dict[str, torch.Tensor]) -> list:
    return [_kernel_operand(tensors[name], dt, align) for name, _, dt, align in spec]


def _validate_operand(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    rank: int,
    align: int,
    dtype: torch.dtype | None = None,
):
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != rank or tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have rank {rank} and unit inner stride")
    if tensor.numel() and tensor.data_ptr() % align:
        raise ValueError(f"{name} must be {align}-byte aligned")
    elem = tensor.element_size()
    for d in range(rank - 1):
        if tensor.shape[d] > 1 and (tensor.stride(d) * elem) % align:
            raise ValueError(
                f"{name} dim {d} byte stride {tensor.stride(d) * elem} must be {align}-byte aligned"
            )


def _slice_scratch(buf: torch.Tensor, spec: ScratchBufferSpec) -> torch.Tensor:
    if buf.device != spec.device:
        raise ValueError(f"{spec.name} scratch device {buf.device} != {spec.device}")
    if not buf.is_contiguous():
        raise ValueError(f"{spec.name} scratch must be contiguous")
    nbytes = buf.numel() * buf.element_size()
    if nbytes < spec.nbytes:
        raise ValueError(
            f"{spec.name} scratch has {nbytes} bytes, requires {spec.nbytes}"
        )
    if buf.data_ptr() % 16:
        raise ValueError(f"{spec.name} scratch must be 16-byte aligned")
    return (
        buf.view(torch.uint8)
        .reshape(-1)[: spec.nbytes]
        .view(spec.dtype)
        .view(spec.shape)
    )


def _compile_key(query: PagedNVFP4Query, config: PagedNVFP4Config) -> tuple:
    fields = {**TUNING.encode_query(query), **config.to_dict()}
    fields["ncw"] = _consumer_warps(query)
    excluded = {
        "max_batch",
        "sm_count",
        "max_splits",
        "autotune_effort",
        "caller_owned_placeholders",
    }
    if query.mode == "prefill":
        excluded.add("max_q_per_req")
    return tuple(sorted((k, v) for k, v in fields.items() if k not in excluded))


def _merge_key(query: PagedNVFP4Query, config: PagedNVFP4Config) -> tuple:
    return (
        query.num_q_heads,
        query.max_q_per_req,
        config.min_tiles_per_split,
        config.stage_rows,
        query.head_dim_vo,
        query.window_left,
        query.has_sinks,
        query.sink_layout,
    )


def _kv_write_key(query: PagedNVFP4Query) -> tuple:
    return (
        query.num_kv_heads,
        query.head_dim_qk,
        query.head_dim_vo,
        query.page_size,
    )


@program_cache(scope="preparation")
def compile_paged_nvfp4(query_payload, config_payload, ordinal):
    from ._decode_kernel import PagedDecodeMergeKernel, PagedNVFP4DecodeKernel
    from ._kv_write import PagedNVFP4KVWriteKernel
    from ._prefill_kernel import PagedNVFP4PrefillKernel

    query = PagedNVFP4Query(**dict(query_payload))
    config = PagedNVFP4Config.from_config(FrozenMapping(config_payload))
    validate_query(query)
    validate_config(query, config)
    key = _compile_key(query, config)
    programs = {}
    with torch.cuda.device(ordinal):
        stream = current_cuda_stream()
        if query.mode == "prefill":
            args = [
                _fake_operand(dt, _operand_rank(rank, query), align)
                for _, rank, dt, align in _PREFILL_OPERANDS
            ]
            kernel = PagedNVFP4PrefillKernel(
                group_size=query.group_size,
                window_left=query.window_left,
                has_sinks=query.has_sinks,
                sink_layout=query.sink_layout,
                stage_rows=config.stage_rows,
                num_stages=config.num_stages,
                head_dim_qk=query.head_dim_qk,
                head_dim_vo=query.head_dim_vo,
                page_size=query.page_size,
                has_factored_relative_bias=query.has_factored_relative_bias,
            )
            programs["forward"] = b12x_compile(
                kernel,
                *args,
                Int32(1),
                Float32(1.0),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    _PREFILL_ID, _COMPILE_VERSION, key
                ),
            )
        else:
            args = [
                _fake_operand(dt, _operand_rank(rank, query), align)
                for _, rank, dt, align in _DECODE_OPERANDS
            ]
            kernel = PagedNVFP4DecodeKernel(
                group_size=query.group_size,
                max_q_per_req=query.max_q_per_req,
                window_left=query.window_left,
                has_sinks=query.has_sinks,
                split_kv=config.split_kv,
                stage_rows=config.stage_rows,
                num_stages=config.num_stages,
                num_consumer_warps=_consumer_warps(query),
                min_tiles_per_split=config.min_tiles_per_split,
                head_dim_qk=query.head_dim_qk,
                head_dim_vo=query.head_dim_vo,
                page_size=query.page_size,
                head_splits=config.head_splits,
                has_relative_bias=query.has_relative_bias,
                sink_layout=query.sink_layout,
            )
            programs["forward"] = b12x_compile(
                kernel,
                *args,
                Float32(1.0),
                stream,
                compile_spec=KernelCompileSpec.from_key(
                    _DECODE_ID, _COMPILE_VERSION, key
                ),
            )
            if config.split_kv:
                merge_args = [
                    _fake_operand(dt, _operand_rank(rank, query), align)
                    for _, rank, dt, align in _MERGE_OPERANDS
                ]
                heads_per_cta = next(h for h in (4, 2, 1) if query.num_q_heads % h == 0)
                merge_kernel = PagedDecodeMergeKernel(
                    num_q_heads=query.num_q_heads,
                    max_q_per_req=query.max_q_per_req,
                    min_tiles_per_split=config.min_tiles_per_split,
                    stage_rows=config.stage_rows,
                    head_dim_vo=query.head_dim_vo,
                    window_left=query.window_left,
                    heads_per_cta=heads_per_cta,
                    has_sinks=query.has_sinks,
                    sink_layout=query.sink_layout,
                )
                programs["merge"] = b12x_compile(
                    merge_kernel,
                    *merge_args,
                    stream,
                    compile_spec=KernelCompileSpec.from_key(
                        _MERGE_ID, _COMPILE_VERSION, _merge_key(query, config)
                    ),
                )
        write_args = [
            _fake_operand(dt, rank, align) for _, rank, dt, align in _KV_WRITE_OPERANDS
        ]
        writer = PagedNVFP4KVWriteKernel(
            num_kv_heads=query.num_kv_heads,
            head_dim_k=query.head_dim_qk,
            head_dim_v=query.head_dim_vo,
            page_size=query.page_size,
        )
        programs["kv_write"] = b12x_compile(
            writer,
            *write_args,
            Int32(1),
            stream,
            compile_spec=KernelCompileSpec.from_key(
                _KV_WRITE_ID, _COMPILE_VERSION, _kv_write_key(query)
            ),
        )
    return programs


def scratch_specs(
    query: PagedNVFP4Query, config: PagedNVFP4Config, device
) -> tuple[ScratchBufferSpec, ...]:
    specs: list[ScratchBufferSpec] = []
    heads = query.num_q_heads
    if query.mode != "prefill" and (config.split_kv or query.caller_owned_placeholders):
        rows = query.max_total_q if config.split_kv else 1
        splits = config.max_splits if config.split_kv else 1
        specs.append(
            ScratchBufferSpec(
                name="paged_nvfp4.o_part",
                shape=(rows, heads, splits, query.head_dim_vo),
                dtype=torch.float32,
                device=device,
            )
        )
        specs.append(
            ScratchBufferSpec(
                name="paged_nvfp4.lse_part",
                shape=(rows, heads, splits),
                dtype=torch.float32,
                device=device,
            )
        )
    if query.caller_owned_placeholders:
        if not query.has_sinks:
            sink_shape = (1, heads) if query.sink_layout == "row" else (heads,)
            specs.append(
                ScratchBufferSpec(
                    name="paged_nvfp4.dummy_sinks",
                    shape=sink_shape,
                    dtype=torch.float32,
                    device=device,
                )
            )
        if query.mode == "prefill" and not query.has_factored_relative_bias:
            specs.append(
                ScratchBufferSpec(
                    name="paged_nvfp4.dummy_rel_proj",
                    shape=(1, REL_RANK),
                    dtype=torch.bfloat16,
                    device=device,
                )
            )
    return tuple(specs)


@dataclass(frozen=True)
class Binding:
    """Bound tensors for one prepared ``paged_nvfp4`` attention invocation."""

    plan: Plan
    q: torch.Tensor
    k_cache: torch.Tensor
    k_scales: torch.Tensor
    v_cache: torch.Tensor
    v_scales: torch.Tensor
    output: torch.Tensor
    page_table: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    o_part: torch.Tensor | None
    lse_part: torch.Tensor | None
    sinks: torch.Tensor
    relative_bias: torch.Tensor | None
    relative_bias_factors: torch.Tensor | None
    relative_bias_projection: torch.Tensor | None
    softmax_scale: float


@dataclass(frozen=True)
class PreparedPagedNVFP4:
    query: PagedNVFP4Query
    config: PagedNVFP4Config
    device: torch.device
    programs: dict
    dummy_o_part: torch.Tensor | None
    dummy_lse_part: torch.Tensor | None
    dummy_sinks: torch.Tensor | None
    dummy_rel_proj: torch.Tensor | None

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return scratch_specs(self.query, self.config, self.device)

    def bind(
        self,
        *,
        plan: Plan,
        scratch,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_cache: torch.Tensor,
        v_scales: torch.Tensor,
        output: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        attention_sink_bias: torch.Tensor | None = None,
        relative_attention_bias: torch.Tensor | None = None,
        relative_bias_factors: torch.Tensor | None = None,
        relative_bias_projection: torch.Tensor | None = None,
        softmax_scale: float | None = None,
    ) -> Binding:
        query, cfg = self.query, self.config
        _validate_operand(
            "q", q, device=self.device, rank=3, align=16, dtype=torch.bfloat16
        )
        _validate_operand(
            "output", output, device=self.device, rank=3, align=16, dtype=torch.bfloat16
        )
        _validate_operand(
            "page_table",
            page_table,
            device=self.device,
            rank=2,
            align=4,
            dtype=torch.int32,
        )
        _validate_operand(
            "cache_seqlens",
            cache_seqlens,
            device=self.device,
            rank=1,
            align=4,
            dtype=torch.int32,
        )
        _validate_operand(
            "cu_seqlens_q",
            cu_seqlens_q,
            device=self.device,
            rank=1,
            align=4,
            dtype=torch.int32,
        )

        batch = int(cache_seqlens.shape[0])
        total_q = int(q.shape[0])
        if batch <= 0 or batch > query.max_batch:
            raise ValueError(f"batch {batch} out of range (1..{query.max_batch})")
        if total_q > query.max_total_q:
            raise ValueError(
                f"total_q {total_q} exceeds max_total_q {query.max_total_q}"
            )
        if tuple(q.shape[1:]) != (query.num_q_heads, query.head_dim_qk):
            raise ValueError(f"q shape mismatch: {tuple(q.shape)}")
        if tuple(output.shape) != (total_q, query.num_q_heads, query.head_dim_vo):
            raise ValueError(f"output shape mismatch: {tuple(output.shape)}")
        if int(page_table.shape[0]) != batch:
            raise ValueError(f"page_table batch {page_table.shape[0]} != {batch}")
        if int(cu_seqlens_q.shape[0]) != batch + 1:
            raise ValueError(f"cu_seqlens_q len {cu_seqlens_q.shape[0]} != {batch + 1}")

        k_cache_u8 = k_cache.view(torch.uint8)
        k_scales_u8 = k_scales.view(torch.uint8)
        v_cache_u8 = v_cache.view(torch.uint8)
        v_scales_u8 = v_scales.view(torch.uint8)
        _validate_operand("k_cache", k_cache_u8, device=self.device, rank=4, align=16)
        _validate_operand("k_scales", k_scales_u8, device=self.device, rank=4, align=8)
        _validate_operand("v_cache", v_cache_u8, device=self.device, rank=4, align=16)
        _validate_operand("v_scales", v_scales_u8, device=self.device, rank=4, align=8)

        pages = int(k_cache_u8.shape[0])
        if (
            int(k_scales_u8.shape[0]) != pages
            or int(v_cache_u8.shape[0]) != pages
            or int(v_scales_u8.shape[0]) != pages
        ):
            raise ValueError(
                "k_cache, k_scales, v_cache, and v_scales must have identical page counts"
            )
        if tuple(k_cache_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_qk // 2,
        ):
            raise ValueError(f"k_cache shape mismatch: {tuple(k_cache_u8.shape)}")
        if tuple(k_scales_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_qk // 16,
        ):
            raise ValueError(f"k_scales shape mismatch: {tuple(k_scales_u8.shape)}")
        if tuple(v_cache_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_vo // 2,
        ):
            raise ValueError(f"v_cache shape mismatch: {tuple(v_cache_u8.shape)}")
        if tuple(v_scales_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_vo // 16,
        ):
            raise ValueError(f"v_scales shape mismatch: {tuple(v_scales_u8.shape)}")

        specs = self.scratch_specs()
        buffers = tuple(scratch)
        if len(buffers) != len(specs):
            raise ValueError(
                f"scratch buffer count mismatch: expected {len(specs)}, got {len(buffers)}"
            )
        sliced = {
            spec.name: _slice_scratch(buf, spec)
            for spec, buf in zip(specs, buffers, strict=True)
        }
        o_part = sliced.get("paged_nvfp4.o_part", self.dummy_o_part)
        lse_part = sliced.get("paged_nvfp4.lse_part", self.dummy_lse_part)

        if query.has_sinks:
            if attention_sink_bias is None:
                raise ValueError("plan requires attention_sink_bias")
            _validate_operand(
                "attention_sink_bias",
                attention_sink_bias,
                device=self.device,
                rank=_operand_rank("sinks", query),
                align=4,
                dtype=torch.float32,
            )
            expected_sink_shape = (
                (total_q, query.num_q_heads)
                if query.sink_layout == "row"
                else (query.num_q_heads,)
            )
            if tuple(attention_sink_bias.shape) != expected_sink_shape:
                raise ValueError(
                    f"attention_sink_bias shape {tuple(attention_sink_bias.shape)} != {expected_sink_shape}"
                )
            sinks = attention_sink_bias
        else:
            if attention_sink_bias is not None:
                raise ValueError(
                    "attention_sink_bias passed to plan with has_sinks=False"
                )
            sinks = sliced.get("paged_nvfp4.dummy_sinks", self.dummy_sinks)

        rel_bias = rel_fac = rel_proj = None
        if query.mode == "prefill":
            if relative_attention_bias is not None:
                raise ValueError(
                    "prefill mode does not accept dense relative_attention_bias"
                )
            if query.has_factored_relative_bias:
                if relative_bias_factors is None or relative_bias_projection is None:
                    raise ValueError(
                        "factored relative bias requires both factors and projection"
                    )
                _validate_operand(
                    "relative_bias_factors",
                    relative_bias_factors,
                    device=self.device,
                    rank=3,
                    align=4,
                    dtype=torch.bfloat16,
                )
                _validate_operand(
                    "relative_bias_projection",
                    relative_bias_projection,
                    device=self.device,
                    rank=2,
                    align=4,
                    dtype=torch.bfloat16,
                )
                if tuple(relative_bias_factors.shape) != (
                    total_q,
                    query.num_q_heads,
                    REL_RANK,
                ):
                    raise ValueError(
                        f"relative_bias_factors shape mismatch: {tuple(relative_bias_factors.shape)}"
                    )
                if int(relative_bias_projection.shape[1]) != REL_RANK:
                    raise ValueError(
                        f"relative_bias_projection rank {relative_bias_projection.shape[1]} != {REL_RANK}"
                    )
                rel_fac, rel_proj = relative_bias_factors, relative_bias_projection
            else:
                if (
                    relative_bias_factors is not None
                    or relative_bias_projection is not None
                ):
                    raise ValueError(
                        "factored relative bias passed to plan with has_factored_relative_bias=False"
                    )
                rel_fac = q
                rel_proj = sliced.get("paged_nvfp4.dummy_rel_proj", self.dummy_rel_proj)
        else:
            if (
                relative_bias_factors is not None
                or relative_bias_projection is not None
            ):
                raise ValueError(
                    "decode/verify mode does not accept factored relative bias"
                )
            if query.has_relative_bias:
                if relative_attention_bias is None:
                    raise ValueError("plan requires relative_attention_bias")
                _validate_operand(
                    "relative_attention_bias",
                    relative_attention_bias,
                    device=self.device,
                    rank=3,
                    align=2,
                    dtype=torch.bfloat16,
                )
                if tuple(relative_attention_bias.shape[:2]) != (
                    total_q,
                    query.num_q_heads,
                ):
                    raise ValueError(
                        f"relative_attention_bias shape mismatch: {tuple(relative_attention_bias.shape)}"
                    )
                rel_bias = relative_attention_bias
            else:
                if relative_attention_bias is not None:
                    raise ValueError(
                        "relative_attention_bias passed to plan with has_relative_bias=False"
                    )
                rel_bias = q

        scale = (
            float(softmax_scale)
            if softmax_scale is not None
            else query.head_dim_qk**-0.5
        )
        return Binding(
            plan=plan,
            q=q,
            k_cache=k_cache_u8,
            k_scales=k_scales_u8,
            v_cache=v_cache_u8,
            v_scales=v_scales_u8,
            output=output,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            o_part=o_part,
            lse_part=lse_part,
            sinks=sinks,
            relative_bias=rel_bias,
            relative_bias_factors=rel_fac,
            relative_bias_projection=rel_proj,
            softmax_scale=scale,
        )

    def run(self, binding: Binding) -> torch.Tensor:
        query = self.query
        tensors = {
            "q": binding.q,
            "k_cache": binding.k_cache,
            "k_scales": binding.k_scales,
            "v_cache": binding.v_cache,
            "v_scales": binding.v_scales,
            "page_table": binding.page_table,
            "cache_seqlens": binding.cache_seqlens,
            "cu_seqlens_q": binding.cu_seqlens_q,
            "output": binding.output,
            "o_part": binding.o_part,
            "lse_part": binding.lse_part,
            "sinks": binding.sinks,
            "relative_bias": binding.relative_bias,
            "relative_bias_factors": binding.relative_bias_factors,
            "relative_bias_projection": binding.relative_bias_projection,
        }
        with torch.cuda.device(self.device):
            stream = current_cuda_stream()
            if query.mode == "prefill":
                args = _pack_operands(_PREFILL_OPERANDS, tensors)
                max_q = min(int(binding.q.shape[0]), query.max_q_per_req)
                run_compiled(
                    self.programs["forward"],
                    (*args, Int32(max_q), Float32(binding.softmax_scale), stream),
                )
            else:
                args = _pack_operands(_DECODE_OPERANDS, tensors)
                run_compiled(
                    self.programs["forward"],
                    (*args, Float32(binding.softmax_scale), stream),
                )
                if self.config.split_kv:
                    m_args = _pack_operands(_MERGE_OPERANDS, tensors)
                    run_compiled(self.programs["merge"], (*m_args, stream))
        return binding.output

    def write_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        k_scales: torch.Tensor,
        v_cache: torch.Tensor,
        v_scales: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        query = self.query
        num_tokens = int(slot_mapping.shape[0])
        if num_tokens == 0:
            return
        k_u8 = k_cache.view(torch.uint8)
        ks_u8 = k_scales.view(torch.uint8)
        v_u8 = v_cache.view(torch.uint8)
        vs_u8 = v_scales.view(torch.uint8)
        _validate_operand(
            "key", key, device=self.device, rank=3, align=16, dtype=torch.bfloat16
        )
        _validate_operand(
            "value", value, device=self.device, rank=3, align=16, dtype=torch.bfloat16
        )
        _validate_operand(
            "slot_mapping",
            slot_mapping,
            device=self.device,
            rank=1,
            align=8,
            dtype=torch.int64,
        )
        _validate_operand("k_cache", k_u8, device=self.device, rank=4, align=16)
        _validate_operand("k_scales", ks_u8, device=self.device, rank=4, align=8)
        _validate_operand("v_cache", v_u8, device=self.device, rank=4, align=16)
        _validate_operand("v_scales", vs_u8, device=self.device, rank=4, align=8)
        if int(key.shape[0]) < num_tokens or int(value.shape[0]) < num_tokens:
            raise ValueError("key and value must have at least len(slot_mapping) rows")
        if tuple(key.shape[1:]) != (query.num_kv_heads, query.head_dim_qk):
            raise ValueError(f"key shape mismatch: {tuple(key.shape)}")
        if tuple(value.shape[1:]) != (query.num_kv_heads, query.head_dim_vo):
            raise ValueError(f"value shape mismatch: {tuple(value.shape)}")
        pages = int(k_u8.shape[0])
        if (
            int(ks_u8.shape[0]) != pages
            or int(v_u8.shape[0]) != pages
            or int(vs_u8.shape[0]) != pages
        ):
            raise ValueError(
                "k_cache, k_scales, v_cache, and v_scales must have identical page counts"
            )
        if tuple(k_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_qk // 2,
        ):
            raise ValueError(f"k_cache shape mismatch: {tuple(k_u8.shape)}")
        if tuple(ks_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_qk // 16,
        ):
            raise ValueError(f"k_scales shape mismatch: {tuple(ks_u8.shape)}")
        if tuple(v_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_vo // 2,
        ):
            raise ValueError(f"v_cache shape mismatch: {tuple(v_u8.shape)}")
        if tuple(vs_u8.shape[1:]) != (
            query.page_size,
            query.num_kv_heads,
            query.head_dim_vo // 16,
        ):
            raise ValueError(f"v_scales shape mismatch: {tuple(vs_u8.shape)}")
        tensors = {
            "key": key,
            "value": value,
            "k_cache": k_u8,
            "k_scales": ks_u8,
            "v_cache": v_u8,
            "v_scales": vs_u8,
            "slot_mapping": slot_mapping,
        }
        args = _pack_operands(_KV_WRITE_OPERANDS, tensors)
        with torch.cuda.device(self.device):
            run_compiled(
                self.programs["kv_write"],
                (*args, Int32(num_tokens), current_cuda_stream()),
            )


def make_plan(caps, *, invocation=None, override=None) -> Plan:
    from .api import Caps

    if not isinstance(caps, Caps):
        raise TypeError("caps must be paged_nvfp4.Caps")
    if invocation:
        raise ValueError("paged_nvfp4 declarations carry no invocation metadata")
    device = torch.device(caps.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    query = PagedNVFP4Query(
        mode=caps.mode,
        kv_format=caps.kv_format,
        num_q_heads=int(caps.num_q_heads),
        num_kv_heads=int(caps.num_kv_heads),
        head_dim_qk=int(caps.head_dim_qk),
        head_dim_vo=int(caps.head_dim_vo),
        page_size=int(caps.page_size),
        max_batch=int(caps.max_batch),
        max_q_per_req=int(caps.max_q_per_req),
        window_left=int(caps.window_left),
        causal=bool(caps.causal),
        has_sinks=bool(caps.has_sinks),
        sink_layout=caps.sink_layout,
        has_relative_bias=bool(caps.has_relative_bias),
        has_factored_relative_bias=bool(caps.has_factored_relative_bias),
        sm_count=int(sm_count),
        caller_owned_placeholders=bool(caps.caller_owned_placeholders),
    )
    validate_query(query)
    compile_query = replace(query, autotune_effort="default")

    def jobs(config, detected):
        return (
            CompileJob.create(
                "b12x.attention.paged_nvfp4._preparation:compile_paged_nvfp4",
                TUNING.encode_query(compile_query),
                config.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def mem_reqs(config, detected):
        specs = scratch_specs(query, config, torch.device("cuda", detected.ordinal))
        return MemoryRequirements(scratch=specs)

    def materialize(selection, detected):
        cfg = selection.config
        programs = compile_paged_nvfp4(
            TUNING.encode_query(compile_query), cfg.to_dict(), detected.ordinal
        )
        load_programs(programs)
        dev = torch.device("cuda", detected.ordinal)
        if query.caller_owned_placeholders:
            d_o = d_lse = d_sinks = d_rel_proj = None
        else:
            d_o = (
                torch.empty(
                    (1, query.num_q_heads, 1, query.head_dim_vo),
                    dtype=torch.float32,
                    device=dev,
                )
                if query.mode != "prefill" and not cfg.split_kv
                else None
            )
            d_lse = (
                torch.empty((1, query.num_q_heads, 1), dtype=torch.float32, device=dev)
                if query.mode != "prefill" and not cfg.split_kv
                else None
            )
            s_shape = (
                (1, query.num_q_heads)
                if query.sink_layout == "row"
                else (query.num_q_heads,)
            )
            d_sinks = (
                torch.zeros(s_shape, dtype=torch.float32, device=dev)
                if not query.has_sinks
                else None
            )
            d_rel_proj = (
                torch.zeros((1, REL_RANK), dtype=torch.bfloat16, device=dev)
                if query.mode == "prefill" and not query.has_factored_relative_bias
                else None
            )
        state = PreparedPagedNVFP4(
            query=query,
            config=cfg,
            device=dev,
            programs=programs,
            dummy_o_part=d_o,
            dummy_lse_part=d_lse,
            dummy_sinks=d_sinks,
            dummy_rel_proj=d_rel_proj,
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
