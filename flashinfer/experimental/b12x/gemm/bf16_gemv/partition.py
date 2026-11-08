"""Output partitions inherit an unpartitioned BF16 projection's prepared arithmetic."""

from dataclasses import dataclass, replace

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import MemoryRequirements, Plan, make_fixed_contract
from b12x.preparation.types import plan_from_handle, require_prepared

from ._preparation import compile_gemv
from ._tuning import GemvConfig, GemvQuery, TUNING as GEMV_TUNING


@dataclass(frozen=True)
class PartitionQuery:
    rows: int
    inner: int
    columns: int
    logical_columns: int


TUNING = make_fixed_contract(
    component_id="gemm.bf16_output_partition",
    query_type=PartitionQuery,
    backend="reference",
)


def _key(config):
    return f"{config.backend}:{config.rows_per_tile}:{config.block_threads}"


@program_cache(scope="preparation")
def compile_partition(payload, ordinal):
    query = GemvQuery(**dict(payload))
    configs = [
        GemvConfig(backend="simt", rows_per_tile=rows, block_threads=threads)
        for rows in (1, 2, 4, 8)
        for threads in (128, 256)
    ]
    configs += [GemvConfig(backend=backend) for backend in ("mma", "tc", "prefill")]
    programs = {}
    for config in configs:
        try:
            GEMV_TUNING.validate_config(query, config, None)
        except ValueError:
            continue
        compiled = compile_gemv(payload, GEMV_TUNING.encode_config(config), ordinal)
        programs[_key(config)] = compiled["gemv"]
    return programs


@dataclass(frozen=True)
class _Execution:
    query: PartitionQuery
    device: torch.device
    launcher: object
    matmul: object | None

    def run(self, source, weight, *, out, workspace):
        q = self.query
        if (
            tuple(source.shape) != (q.rows, q.inner)
            or tuple(weight.shape) != (q.columns, q.inner)
            or tuple(out.shape) != (q.rows, q.columns)
            or any(
                t.dtype != torch.bfloat16
                or t.device != self.device
                or not t.is_contiguous()
                for t in (source, weight, out)
            )
        ):
            raise ValueError(
                "BF16 output-partition operands differ from their prepared geometry"
            )
        if (
            workspace.dtype != torch.uint8
            or workspace.device != self.device
            or workspace.ndim != 1
            or not workspace.is_contiguous()
        ):
            raise ValueError("BF16 output partitions require caller-owned byte scratch")
        from b12x.gemm.blockscaled._a16 import _overlap

        if any(_overlap(out, t) for t in (source, weight, workspace)) or any(
            _overlap(workspace, t) for t in (source, weight)
        ):
            raise ValueError(
                "BF16 output-partition outputs and scratch must not overlap inputs"
            )
        if self.matmul is not None:
            self.matmul(source, weight, out, workspace)
        else:
            self.launcher(source, weight, out, None)
        return out


def plan(reference, columns, *, dependency):
    """Declare a partition dependent on a full-width projection's tuning result.

    The reference request benchmarks the actual unpartitioned matrix. Its
    selected backend and reduction schedule control this plan; the partition
    does not independently retune arithmetic. Compile the eligible local
    programs before materialization, and retain only caller-owned execution data.
    """
    r = reference.query
    if (
        not isinstance(r, GemvQuery)
        or not dependency
        or type(columns) is not int
        or not 0 < columns <= r.out_features
        or r.out_features % columns
        or r.source_dtype != "bfloat16"
        or r.weight_dtype != "bfloat16"
        or r.output_dtype != "bfloat16"
        or r.bias_dtype is not None
        or not all(
            (
                r.source_contiguous,
                r.source_aligned,
                r.weight_contiguous,
                r.weight_aligned,
                r.output_contiguous,
                r.output_aligned,
            )
        )
    ):
        raise ValueError(
            "output partitions require aligned contiguous BF16 projections and a reference dependency"
        )
    query = PartitionQuery(r.max_rows, r.in_features, columns, r.out_features)
    payload = GEMV_TUNING.encode_query(
        replace(r, out_features=columns, autotune_effort="default")
    )
    workspace_bytes = {}

    def jobs(config, device):
        return (
            CompileJob.create(
                "b12x.gemm.bf16_gemv.partition:compile_partition",
                payload,
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def memory(config, device):
        from b12x.gemm._shared.reference_matmul import MAX_WORKSPACE_BYTES

        size = workspace_bytes.get(device.ordinal, MAX_WORKSPACE_BYTES)
        return MemoryRequirements(
            scratch=()
            if size == 0
            else (
                ScratchBufferSpec(
                    name="workspace",
                    shape=(size,),
                    dtype=torch.uint8,
                    device=torch.device("cuda", device.ordinal),
                ),
            )
        )

    def materialize(selection, device):
        if reference.selection is None:
            raise RuntimeError(
                "output-partition reference must be prepared before its dependent plan"
            )
        config = reference.selection.config
        resolved = torch.device("cuda", device.ordinal)
        programs = compile_partition(payload, device.ordinal)
        load_programs(programs)
        matmul = launcher = None
        if config.backend == "torch":
            from b12x.gemm._shared.reference_matmul import ReferenceMatmul

            matmul = ReferenceMatmul(
                rows=query.rows,
                columns=columns,
                inner=query.inner,
                logical_rows=query.rows,
                logical_columns=query.logical_columns,
                device=resolved,
            )
        else:
            launcher = programs[_key(config)]
        workspace_bytes[device.ordinal] = (
            0 if matmul is None else matmul.workspace_bytes
        )
        dependency = reference.prepared.state if reference.prepared.programs else None
        return attach_programs(
            _Execution(query, resolved, launcher, matmul), programs, dependency
        )

    return Plan(
        contract=TUNING,
        query=query,
        shared=False,
        dependencies=(dependency,),
        _compile_jobs=jobs,
        _memory_requirements=memory,
        _materialize=materialize,
    )


@torch.library.custom_op(
    "b12x::bf16_output_partition", mutates_args=("output", "workspace")
)
def _execute(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    handle: int,
) -> None:
    require_prepared(plan_from_handle(handle), TUNING.component_id, source.device).run(
        source, weight, out=output, workspace=workspace
    )


@_execute.register_fake
def _fake(source, weight, output, workspace, handle):
    pass


def mm(source, weight, *, out, workspace, plan):
    _execute(source, weight, out, workspace, plan.handle)
    return out
