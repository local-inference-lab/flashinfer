"""Compile the existing row selector from tensor metadata before capture."""

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)


@dataclass(frozen=True, kw_only=True)
class RowTopKQuery:
    rows: int
    width: int
    topk: int = 2048

    def __post_init__(self):
        values = (self.rows, self.width, self.topk)
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("row top-k dimensions must be positive integers")
        if self.topk not in (512, 1024, 2048):
            raise ValueError("row top-k supports K=512, 1024, or 2048")
        if max(self.rows * self.width, self.rows * self.topk) >= 2**31:
            raise ValueError("row top-k extents exceed int32 indexing")


TUNING = make_fixed_contract(
    component_id="attention.row_topk", query_type=RowTopKQuery, backend="cute"
)


def _need(name, tensor, shape, dtype, device):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _validate(query, device, scores, lengths, gather_table, out_values, out_indices):
    from b12x.sequence.ple_hash._contracts import _require_mutation_alias_contract

    _need("scores", scores, (query.rows, query.width), torch.float32, device)
    _need("lengths", lengths, (query.rows,), torch.int32, device)
    _need("gather_table", gather_table, (query.rows, query.width), torch.int32, device)
    _need("out_values", out_values, (query.rows, query.topk), torch.float32, device)
    _need("out_indices", out_indices, (query.rows, query.topk), torch.int32, device)
    _require_mutation_alias_contract(
        mutable=(("out_values", out_values), ("out_indices", out_indices)),
        read_only=(
            ("scores", scores),
            ("lengths", lengths),
            ("gather_table", gather_table),
        ),
    )


@dataclass(frozen=True)
class _State:
    query: RowTopKQuery
    device: torch.device
    launcher: object

    def run(self, scores, lengths, gather_table, out_values, out_indices):
        from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

        _validate(
            self.query,
            self.device,
            scores,
            lengths,
            gather_table,
            out_values,
            out_indices,
        )
        with torch.cuda.device(self.device):
            run_row_topk(
                row_logits=scores,
                lengths=lengths,
                topk=self.query.topk,
                output_values=out_values,
                output_indices=out_indices,
                output_gather_table=gather_table,
                launcher=self.launcher,
            )
        return out_values, out_indices


@program_cache(scope="preparation")
def compile_row_topk(payload, ordinal):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

    query = RowTopKQuery(**dict(payload))
    device = torch.device("cuda", ordinal)
    launchers = {}
    with torch.cuda.device(device), FakeTensorMode(), compile_only_launches():
        scores = torch.empty(
            (query.rows, query.width), dtype=torch.float32, device=device
        )
        lengths = torch.empty((query.rows,), dtype=torch.int32, device=device)
        gather = torch.empty(
            (query.rows, query.width), dtype=torch.int32, device=device
        )
        values = torch.empty(
            (query.rows, query.topk), dtype=torch.float32, device=device
        )
        indices = torch.empty(
            (query.rows, query.topk), dtype=torch.int32, device=device
        )
        run_row_topk(
            row_logits=scores,
            lengths=lengths,
            topk=query.topk,
            output_values=values,
            output_indices=indices,
            output_gather_table=gather,
            launcher_sink=launchers,
        )
    launcher = launchers[("row", True, True)]
    return attach_programs(_State(query, device, launcher), launcher)


def plan(query: RowTopKQuery, *, device, invocation=FrozenMapping(), override=None):
    if not isinstance(query, RowTopKQuery):
        raise TypeError("row top-k plan requires RowTopKQuery")
    if invocation:
        raise ValueError("row top-k invocation is fully described by RowTopKQuery")
    if torch.device(device).type != "cuda":
        raise ValueError("row top-k requires CUDA")

    def materialize(selection, detected):
        state = compile_row_topk(TUNING.encode_query(query), detected.ordinal)
        load_programs(state)
        return state

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.attention.row_topk._preparation:compile_row_topk",
                TUNING.encode_query(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
    )
