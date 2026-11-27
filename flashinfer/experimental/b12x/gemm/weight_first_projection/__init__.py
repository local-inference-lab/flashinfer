"""Prepare two BF16 projections with FP32 and BF16 results for dependent CUDA launch."""

from dataclasses import dataclass

import torch
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)

from ._kernel import brick_for, compile_projection, run_projection


@dataclass(frozen=True)
class Query:
    rows: int
    width: int
    first_rows: int
    second_rows: int
    staging_depth: int
    pdl: bool

    def __post_init__(self):
        if (
            not 1 <= self.rows <= 16
            or min(self.width, self.first_rows, self.second_rows) <= 0
        ):
            raise ValueError(
                "weight-first projections require 1–16 input rows and positive matrix dimensions"
            )
        if self.staging_depth not in (0, 1, 2, 3, 4, 6, 8):
            raise ValueError("unsupported weight-first projection staging depth")
        brick_for(self.first_rows + self.second_rows, self.width)

    @property
    def partial_shape(self):
        total = self.first_rows + self.second_rows
        _, kt = brick_for(total, self.width)
        return self.width // kt, 16, total

    @property
    def counter_shape(self):
        total = self.first_rows + self.second_rows
        nt, _ = brick_for(total, self.width)
        return ((total + nt - 1) // nt,)


TUNING = make_fixed_contract(
    component_id="gemm.weight_first_projection", query_type=Query, backend="cutedsl"
)


@program_cache(scope="preparation")
def compile_for_device(first_rows, second_rows, width, ordinal):
    with torch.cuda.device(ordinal):
        return compile_projection(first_rows, second_rows, width)


@dataclass(frozen=True)
class _State:
    query: Query
    device: torch.device
    launch: object

    def run(self, source, first, second, first_out, second_out, partials, counters):
        q = self.query
        arguments = (
            (source, (q.rows, q.width), torch.bfloat16),
            (first, (q.first_rows, q.width), torch.bfloat16),
            (second, (q.second_rows, q.width), torch.bfloat16),
            (first_out, (q.rows, q.first_rows), torch.float32),
            (second_out, (q.rows, q.second_rows), torch.bfloat16),
            (partials, q.partial_shape, torch.float32),
            (counters, q.counter_shape, torch.int32),
        )
        for tensor, shape, dtype in arguments:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != self.device
                or not tensor.is_contiguous()
                or tensor.data_ptr() % 16
            ):
                raise ValueError(
                    f"weight-first projection expects aligned contiguous {dtype} {shape} on {self.device}"
                )
        run_projection(
            self.launch,
            source,
            first,
            second,
            first_out,
            second_out,
            partials,
            counters,
            q.staging_depth,
            q.pdl,
        )


def plan(query):
    """Bind the geometry before capture. Counters begin at zero and reset after every launch."""
    if not isinstance(query, Query):
        raise TypeError("weight-first projection requires its shape and launch query")
    args = (query.first_rows, query.second_rows, query.width)

    def compile_jobs(config, device):
        return (
            CompileJob.create(
                "b12x.gemm.weight_first_projection:compile_for_device",
                *args,
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def materialize(selection, device):
        ordinal = device.ordinal
        return _State(
            query, torch.device("cuda", ordinal), compile_for_device(*args, ordinal)
        )

    def memory(config, device):
        location = torch.device("cuda", device.ordinal)
        return MemoryRequirements(
            scratch=(
                ScratchBufferSpec(
                    name="partial",
                    shape=query.partial_shape,
                    dtype=torch.float32,
                    device=location,
                ),
                ScratchBufferSpec(
                    name="counters",
                    shape=query.counter_shape,
                    dtype=torch.int32,
                    device=location,
                ),
            )
        )

    return Plan(
        contract=TUNING,
        query=query,
        invocation=FrozenMapping(),
        _compile_jobs=compile_jobs,
        _materialize=materialize,
        _memory_requirements=memory,
    )


@torch.library.custom_op(
    "b12x::weight_first_projection_out",
    mutates_args=("first_out", "second_out", "partials", "counters"),
)
def project(
    source: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    first_out: torch.Tensor,
    second_out: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    handle: int,
) -> None:
    """Write FP32 router and BF16 gate/up projections using the prepared dependent launch."""
    from b12x.preparation.types import plan_from_handle, require_prepared

    state = require_prepared(
        plan_from_handle(handle), TUNING.component_id, source.device
    )
    state.run(source, first, second, first_out, second_out, partials, counters)


@project.register_fake
def _fake_project(
    source, first, second, first_out, second_out, partials, counters, handle
):
    pass
