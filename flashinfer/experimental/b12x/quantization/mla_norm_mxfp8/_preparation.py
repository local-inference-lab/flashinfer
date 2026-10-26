"""Ordinary compile jobs and caller-owned output bindings for MLA normalization."""

from dataclasses import dataclass
import torch
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import plan_from_handle, require_prepared
from ._tuning import Query, Config, TUNING


@program_cache(scope="preparation")
def compile_norm(payload, pdl, ordinal):
    from ._kernels import mla_norm_mxfp8_kernel

    query = Query(**dict(payload))
    weight_dtype = getattr(torch, query.norm_weight_dtype)
    with torch.cuda.device(ordinal):
        return mla_norm_mxfp8_kernel.warmup(
            torch.bfloat16,
            weight_dtype,
            weight_dtype,
            torch.bfloat16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            query.rows,
            pdl,
            grid=(query.rows, 2, 1),
            num_warps=8,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )


@dataclass(frozen=True)
class State:
    query: Query
    device: torch.device
    programmatic_serialization: bool
    program: object

    def run(self, projected, qw, kw, query, kv, values, scales, scale_rows):
        q = self.query
        physical = ((q.rows + 127) // 128) * 10 * 512
        facts = [
            (projected, (q.rows, 1792), torch.bfloat16),
            (qw, (1280,), getattr(torch, q.norm_weight_dtype)),
            (kw, (512,), getattr(torch, q.norm_weight_dtype)),
            (query, (q.rows, 1280), torch.bfloat16),
            (kv, (q.rows, 512), torch.bfloat16),
            (values, (q.rows, 1280), torch.float8_e4m3fn),
            (scales, (physical,), torch.uint8),
            (scale_rows, (q.rows, 40), torch.uint8),
        ]
        for tensor, shape, dtype in facts:
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    "MLA normalization operand differs from its declared device, dtype, shape or alignment"
                )
        for position, (left, _, _) in enumerate(facts):
            for index, (right, _, _) in enumerate(facts[position + 1 :], position + 1):
                if position == 1 and index == 2:
                    continue
                a, b = left.data_ptr(), right.data_ptr()
                if (
                    a < b + right.numel() * right.element_size()
                    and b < a + left.numel() * left.element_size()
                ):
                    raise ValueError(
                        "MLA normalization requires disjoint projected input, affine weights and outputs"
                    )
        self.program[(q.rows, 2, 1)](
            projected,
            qw,
            kw,
            query,
            kv,
            values,
            scales,
            scale_rows,
            q.rows,
            self.programmatic_serialization,
        )


def plan(query, *, invocation=FrozenMapping(), override=None):
    if not isinstance(query, Query):
        raise TypeError("MLA normalization requires Query")
    if invocation:
        raise ValueError("MLA normalization semantics belong in Query")

    def materialize(selection, device):
        program = compile_norm(
            TUNING.encode_query(query),
            selection.config.programmatic_serialization,
            device.ordinal,
        )
        load_programs(program)
        return attach_programs(
            State(
                query,
                torch.device("cuda", device.ordinal),
                selection.config.programmatic_serialization,
                program,
            ),
            program,
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        override=override,
        shared=True,
        _compile_jobs=lambda config, device: (
            CompileJob.create(
                "b12x.quantization.mla_norm_mxfp8._preparation:compile_norm",
                TUNING.encode_query(query),
                config.programmatic_serialization,
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda c, d: MemoryRequirements(),
        _materialize=materialize,
    )


@torch.library.custom_op(
    "b12x::mla_norm_mxfp8_prepared",
    mutates_args=("query", "kv", "values", "scales", "scale_rows"),
)
def _execute(
    projected: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    query: torch.Tensor,
    kv: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor,
    scale_rows: torch.Tensor,
    handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(handle), "quantization.mla_norm_mxfp8", projected.device
    )
    state.run(projected, qw, kw, query, kv, values, scales, scale_rows)


@_execute.register_fake
def _fake(projected, qw, kw, query, kv, values, scales, scale_rows, handle):
    pass


def run(projected, qw, kw, query, kv, values, scales, scale_rows, *, plan):
    _execute(projected, qw, kw, query, kv, values, scales, scale_rows, plan.handle)
