"""Native GEMM declarations and prepared functional/output boundaries."""

from __future__ import annotations

import b12x._lib.dense_gemm.buffers as _module__lib_dense_gemm_buffers


from dataclasses import dataclass

import torch

from b12x._lib.compile_pool import CompileJob
from b12x._lib.scratch import ScratchBufferSpec
from b12x._lib.utils.device import cuda_stream_to_int
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    PersistentMemory,
    Plan,
    current_plan,
    current_prepared_state,
)
from b12x.preparation.types import plan_from_handle, require_prepared
from ._tuning import DenseGemmQuery, TUNING, launch_options, operand_options


def _metadata_operands(query):
    options = operand_options(query)
    fp4 = query.recipe in ("nvfp4", "mxfp4")
    fp6 = query.recipe.startswith(("mxfp6_", "w6a8_"))
    a_k = query.in_features // 2 if fp4 else query.in_features
    b_k = (
        3 * query.in_features // 4 if fp6 and query.weight_storage == "packed" else a_k
    )
    value_dtype = torch.uint8 if fp4 or fp6 else torch.float8_e4m3fn
    a_dtype = torch.float8_e4m3fn if query.recipe.startswith("w6a8_") else value_dtype

    def values(rows, columns, dtype):
        return torch.empty_strided(
            (rows, columns, query.batch),
            (columns, 1, rows * columns),
            dtype=dtype,
            device="meta",
        )

    def scales(rows, weight=False):
        dtype = getattr(torch, options["sf_dtype"])
        if query.recipe == "block_fp8":
            return torch.empty(
                (rows // 128 if weight else rows, query.in_features // 128),
                dtype=dtype,
                device="meta",
            )
        rb = (rows + 127) // 128
        kb = (query.in_features // options["sf_vec_size"] + 3) // 4
        return torch.empty_strided(
            (32, 4, rb, 4, kb, query.batch),
            (16, 4, kb * 512, 1, 512, rb * kb * 512),
            dtype=dtype,
            device="meta",
        )

    lhs = values(query.max_rows, a_k, a_dtype), scales(query.max_rows)
    rhs = values(query.out_features, b_k, value_dtype), scales(query.out_features, True)
    out = (
        None
        if query.output_mode == "functional"
        else values(
            query.max_rows, query.out_features, getattr(torch, query.output_dtype)
        )
    )
    alpha = (
        None
        if query.alpha_mode == "unit"
        else torch.empty((1,), dtype=torch.float32, device="meta")
    )
    return lhs, rhs, out, alpha, options


def _lower_query(query, device, constraints):
    if device is None:
        raise ValueError("dense preparation requires a device SM count")
    from b12x._lib.dense_gemm.preparation import _lower_dense_gemm

    lhs, rhs, out, alpha, options = _metadata_operands(query)
    return _lower_dense_gemm(
        lhs,
        rhs,
        out,
        **options,
        c_dtype=query.output_dtype,
        alpha=alpha,
        expected_m=query.expected_m,
        allow_bf16_atomic_reduction=query.allow_bf16_atomic_reduction,
        sm_count=query.sm_count if query.sm_count is not None else device.sm_count,
        **constraints,
    )


def _default_lowering(query, device):
    return _lower_query(query, device, dict(query.overrides))


def _configured_lowering(query, config, device):
    return _lower_query(query, device, launch_options(query, config))


def _split_k_elements(p):
    return (
        p.policy.split_k_slices * p.m * p.n
        if p.policy.split_k_slices > 1 and not p.policy.split_k_atomic_bf16
        else 0
    )


@dataclass(frozen=True)
class _PreparedDense:
    query: DenseGemmQuery
    core: object
    workspace: torch.Tensor | None = None

    def run(self, a, sfa, b, sfb, alpha, out, workspace, output_dtype, stream):
        q = self.query
        if self.core.lowering.sf_dtype == "float8_e8m0fnu":
            if sfa.dtype == torch.uint8:
                sfa = sfa.view(torch.float8_e8m0fnu)
            if sfb.dtype == torch.uint8:
                sfb = sfb.view(torch.float8_e8m0fnu)
        if output_dtype != getattr(torch, q.output_dtype):
            raise ValueError("dense output dtype differs from prepared invocation")
        if (out is None) != (q.output_mode == "functional"):
            raise ValueError("dense output form differs from prepared invocation")
        if q.workspace_form == "owned":
            if workspace is not None:
                raise ValueError(
                    "owned dense execution does not accept a caller workspace"
                )
            workspace = self.workspace
        elif workspace is None:
            raise ValueError("dense workspace form differs from prepared invocation")
        if (
            workspace is not None
            and q.workspace_nbytes is not None
            and workspace.numel() * workspace.element_size() < q.workspace_nbytes
        ):
            raise ValueError("dense workspace capacity is smaller than declared")
        return self.core.run(
            (a, sfa),
            (b, sfb),
            out=out,
            alpha=alpha,
            stream=stream,
            split_k_workspace=workspace,
        )


def plan(query: DenseGemmQuery, *, invocation=FrozenMapping(), override=None):
    if not isinstance(query, DenseGemmQuery):
        raise TypeError("query must be DenseGemmQuery")
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("native GEMM invocation constraints belong in DenseGemmQuery")
    lowered = {}

    def lower(config, device):
        key = (config, device.identity)
        if key not in lowered:
            lowered[key] = _configured_lowering(query, config, device.identity)
        return lowered[key]

    def memory(config, device):
        p = lower(config, device)
        needed = _split_k_elements(p)
        scratch = ()
        if query.workspace_form == "provided" and needed:
            scratch = (
                ScratchBufferSpec(
                    name="dense.split_k",
                    shape=(needed,),
                    dtype=torch.float32,
                    device=torch.device("cuda", device.ordinal),
                ),
            )
        persistent = []
        if query.workspace_form == "owned" and needed:
            owned_nbytes = needed * 4
            existing = current_prepared_state()
            resident = (
                0
                if existing is None or existing.workspace is None
                else min(
                    owned_nbytes,
                    existing.workspace.numel() * existing.workspace.element_size(),
                )
            )
            persistent.append(
                PersistentMemory(
                    ("dense.owned", current_plan()), owned_nbytes, resident
                )
            )
        if p.alpha_is_one:
            resident = _module__lib_dense_gemm_buffers._ALPHA_ONE_CACHE.get(
                ("cuda", device.ordinal)
            )
            persistent.append(
                PersistentMemory(
                    ("dense.alpha_one", device.ordinal),
                    4,
                    0
                    if resident is None
                    else resident.numel() * resident.element_size(),
                )
            )
        return MemoryRequirements(scratch, tuple(persistent))

    def materialize(selection, device):
        from b12x._lib.dense_gemm.preparation import _materialize_dense

        p = lower(selection.config, device)
        resolved_device = torch.device("cuda", device.ordinal)
        core = _materialize_dense(p, resolved_device)
        workspace = None
        if query.workspace_form == "owned":
            needed = _split_k_elements(p)
            if needed:
                existing = current_prepared_state()
                previous = None if existing is None else existing.workspace
                if (
                    previous is not None
                    and previous.numel() >= needed
                    and previous.device == resolved_device
                ):
                    workspace = previous
                else:
                    workspace = torch.empty(
                        needed, dtype=torch.float32, device=resolved_device
                    )
        return _PreparedDense(query, core, workspace)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x._lib.dense_gemm.preparation:_compile_dense_lowering",
                lower(config, device).to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=memory,
        _materialize=materialize,
    )


def _result_metadata(a, b, output_dtype):
    from b12x._lib.dense_gemm.buffers import _empty_dense_gemm_output

    return _empty_dense_gemm_output(
        a.shape[0], b.shape[0], a.shape[2], dtype=output_dtype, device=a.device
    )


@torch.library.custom_op("b12x::dense_prepared_functional", mutates_args=())
def _functional(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> torch.Tensor:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.mm", a.device)
    return state.run(a, sfa, b, sfb, alpha, None, None, output_dtype, stream)


@_functional.register_fake
def _functional_fake(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> torch.Tensor:
    return _result_metadata(a, b, output_dtype)


@torch.library.custom_op(
    "b12x::dense_prepared_functional_workspace", mutates_args=("workspace",)
)
def _functional_workspace(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    workspace: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> torch.Tensor:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.mm", a.device)
    return state.run(a, sfa, b, sfb, alpha, None, workspace, output_dtype, stream)


@_functional_workspace.register_fake
def _functional_workspace_fake(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    workspace: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> torch.Tensor:
    return _result_metadata(a, b, output_dtype)


@torch.library.custom_op("b12x::dense_prepared_out", mutates_args=("out",))
def _out(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    out: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.mm", a.device)
    state.run(a, sfa, b, sfb, alpha, out, None, output_dtype, stream)


@_out.register_fake
def _out_fake(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    out: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> None:
    del a, sfa, b, sfb, alpha, out, output_dtype, stream, plan_handle


@torch.library.custom_op(
    "b12x::dense_prepared_out_workspace", mutates_args=("out", "workspace")
)
def _out_workspace(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    out: torch.Tensor,
    workspace: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "gemm.mm", a.device)
    state.run(a, sfa, b, sfb, alpha, out, workspace, output_dtype, stream)


@_out_workspace.register_fake
def _out_workspace_fake(
    a: torch.Tensor,
    sfa: torch.Tensor,
    b: torch.Tensor,
    sfb: torch.Tensor,
    alpha: torch.Tensor | None,
    out: torch.Tensor,
    workspace: torch.Tensor,
    output_dtype: torch.dtype,
    stream: int | None,
    plan_handle: int,
) -> None:
    del a, sfa, b, sfb, alpha, out, workspace, output_dtype, stream, plan_handle


def mm(
    lhs,
    rhs,
    out=None,
    *,
    plan: Plan,
    alpha=None,
    out_dtype=None,
    workspace=None,
    stream=None,
):
    """Execute the declared recipe; planning hints belong to the declaration."""
    a, sfa = lhs
    b, sfb = rhs
    if sfa.dtype == torch.uint8 or sfb.dtype == torch.uint8:
        raise TypeError(
            "native dense scales require their declared floating-point dtype"
        )
    # Preserve the existing UE8M0 auto-functionalization workaround.
    if sfa.dtype == torch.float8_e8m0fnu:
        sfa = sfa.view(torch.uint8)
    if sfb.dtype == torch.float8_e8m0fnu:
        sfb = sfb.view(torch.uint8)
    if out_dtype is None:
        out_dtype = torch.bfloat16 if out is None else out.dtype
    stream_int = cuda_stream_to_int(stream)
    handle = plan.handle
    if out is None:
        if workspace is None:
            return torch.ops.b12x.dense_prepared_functional(
                a,
                sfa,
                b,
                sfb,
                alpha,
                out_dtype,
                stream_int,
                handle,
            )
        return torch.ops.b12x.dense_prepared_functional_workspace(
            a,
            sfa,
            b,
            sfb,
            alpha,
            workspace,
            out_dtype,
            stream_int,
            handle,
        )
    if workspace is None:
        torch.ops.b12x.dense_prepared_out(
            a,
            sfa,
            b,
            sfb,
            alpha,
            out,
            out_dtype,
            stream_int,
            handle,
        )
    else:
        torch.ops.b12x.dense_prepared_out_workspace(
            a,
            sfa,
            b,
            sfb,
            alpha,
            out,
            workspace,
            out_dtype,
            stream_int,
            handle,
        )
    return out


@torch.library.custom_op(
    "b12x::dense_mxfp8_prepared_bases",
    mutates_args=("out_base", "workspace_base"),
)
def _mxfp8_mm_bases_prepared(
    a_values_base: torch.Tensor,
    a_scale_mma_base: torch.Tensor,
    b_values_base: torch.Tensor,
    b_scale_mma_base: torch.Tensor,
    out_base: torch.Tensor,
    workspace_base: torch.Tensor,
    plan_handle: int,
) -> None:
    from b12x.quantization.mxfp8 import dense_scale_view_from_storage

    state = require_prepared(
        plan_from_handle(plan_handle), "gemm.mm", a_values_base.device
    )
    query = state.query
    if (
        query.recipe != "mxfp8"
        or query.output_mode != "provided"
        or query.alpha_mode != "unit"
        or query.workspace_form != "provided"
    ):
        raise ValueError(
            "MXFP8 physical-base launch requires a prepared unit-alpha output plan"
        )
    if (
        a_values_base.ndim != 2
        or b_values_base.ndim != 2
        or a_values_base.dtype != torch.float8_e4m3fn
        or b_values_base.dtype != torch.float8_e4m3fn
        or not a_values_base.is_contiguous()
        or not b_values_base.is_contiguous()
        or out_base.shape != (a_values_base.shape[0], b_values_base.shape[0])
        or out_base.dtype != torch.bfloat16
        or not out_base.is_contiguous()
        or workspace_base.dtype != torch.uint8
        or workspace_base.ndim != 1
        or not workspace_base.is_contiguous()
        or workspace_base.numel() % 4
    ):
        raise ValueError(
            "MXFP8 physical-base operands differ from the prepared dense layout"
        )
    m, k = (int(size) for size in a_values_base.shape)
    n = int(b_values_base.shape[0])
    if b_values_base.shape[1] != k:
        raise ValueError("MXFP8 physical-base operands require matching K")
    sfa = dense_scale_view_from_storage(a_scale_mma_base, m, k)
    sfb = dense_scale_view_from_storage(b_scale_mma_base, n, k)
    state.run(
        a_values_base.view(m, k, 1),
        sfa,
        b_values_base.view(n, k, 1),
        sfb,
        None,
        out_base.view(m, n, 1),
        workspace_base.view(torch.float32),
        torch.bfloat16,
        None,
    )


@_mxfp8_mm_bases_prepared.register_fake
def _mxfp8_mm_bases_prepared_fake(
    a_values_base,
    a_scale_mma_base,
    b_values_base,
    b_scale_mma_base,
    out_base,
    workspace_base,
    plan_handle,
) -> None:
    del (
        a_values_base,
        a_scale_mma_base,
        b_values_base,
        b_scale_mma_base,
        out_base,
        workspace_base,
        plan_handle,
    )


def mxfp8_mm_bases(
    a_values_base: torch.Tensor,
    a_scale_mma_base: torch.Tensor,
    b_values_base: torch.Tensor,
    b_scale_mma_base: torch.Tensor,
    out_base: torch.Tensor,
    workspace_base: torch.Tensor,
    *,
    plan: Plan,
) -> torch.Tensor:
    """Execute a prepared CuTe MXFP8 GEMM from caller-owned physical bases."""
    _mxfp8_mm_bases_prepared(
        a_values_base,
        a_scale_mma_base,
        b_values_base,
        b_scale_mma_base,
        out_base,
        workspace_base,
        plan.handle,
    )
    return out_base
