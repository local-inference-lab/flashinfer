"""Metadata-only peer record declarations and prepared Triton launchers."""

from dataclasses import dataclass

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)


@dataclass(frozen=True, kw_only=True)
class PeerRecordGatherQuery:
    rows: int
    topk: int
    world_size: int
    peer_records: int
    record_bytes: int = 576

    def __post_init__(self):
        values = (
            self.rows,
            self.topk,
            self.world_size,
            self.peer_records,
            self.record_bytes,
        )
        if any(type(v) is not int or v <= 0 for v in values):
            raise ValueError("peer record dimensions must be positive integers")
        if max(self.rows * self.topk, *values) >= 2**31:
            raise ValueError("peer record extents exceed int32 indexing")
        if max(self.rows * self.topk, self.peer_records) * self.record_bytes >= 2**63:
            raise ValueError("peer record byte extents exceed int64 indexing")


TUNING = make_fixed_contract(
    component_id="comm.peer_record_gather",
    query_type=PeerRecordGatherQuery,
    backend="triton",
)


def _need(name, tensor, shape, dtypes, device):
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype in {dtypes}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _validate(query, device, peer_bases, owners, slots, counts, out):
    from b12x.sequence.ple_hash._contracts import _require_mutation_alias_contract

    _need(
        "peer_bases",
        peer_bases,
        (query.world_size,),
        (torch.int64, torch.uint64),
        device,
    )
    _need("owners", owners, (query.rows, query.topk), (torch.int32,), device)
    _need("slots", slots, (query.rows, query.topk), (torch.int32,), device)
    _need("counts", counts, (query.rows,), (torch.int32,), device)
    _need(
        "out",
        out,
        (query.rows, query.topk, query.record_bytes),
        (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2),
        device,
    )
    _require_mutation_alias_contract(
        mutable=(("out", out),),
        read_only=(
            ("peer_bases", peer_bases),
            ("owners", owners),
            ("slots", slots),
            ("counts", counts),
        ),
    )


@dataclass(frozen=True)
class _State:
    query: PeerRecordGatherQuery
    device: torch.device
    launcher: object

    def run(self, peer_bases, owners, slots, counts, out):
        import triton

        _validate(self.query, self.device, peer_bases, owners, slots, counts, out)
        q = self.query
        with torch.cuda.device(self.device):
            self.launcher[(q.rows * q.topk, 1, 1)](
                peer_bases.view(torch.int64),
                owners,
                slots,
                counts,
                out.view(torch.uint8),
                q.topk,
                q.world_size,
                q.peer_records,
                q.record_bytes,
                triton.next_power_of_2(q.record_bytes),
            )
        return out


@program_cache(scope="preparation")
def compile_gather(payload, ordinal):
    import triton
    from ._kernel import _gather_records

    q = PeerRecordGatherQuery(**dict(payload))
    with torch.cuda.device(ordinal):
        launcher = _gather_records.warmup(
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.uint8,
            q.topk,
            q.world_size,
            q.peer_records,
            q.record_bytes,
            triton.next_power_of_2(q.record_bytes),
            grid=(q.rows * q.topk,),
            num_warps=4,
        )
    return attach_programs(_State(q, torch.device("cuda", ordinal), launcher), launcher)


def plan(
    query: PeerRecordGatherQuery, *, device, invocation=FrozenMapping(), override=None
):
    if not isinstance(query, PeerRecordGatherQuery):
        raise TypeError("peer record gather plan requires PeerRecordGatherQuery")
    if invocation:
        raise ValueError("peer record invocation is described by PeerRecordGatherQuery")
    if torch.device(device).type != "cuda":
        raise ValueError("peer record gather requires CUDA")

    def materialize(selection, detected):
        state = compile_gather(TUNING.encode_query(query), detected.ordinal)
        load_programs(state)
        return state

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.comm.peer_record_gather._preparation:compile_gather",
                TUNING.encode_query(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=materialize,
    )
