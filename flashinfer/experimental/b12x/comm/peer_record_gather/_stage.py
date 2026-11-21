"""Prepare reusable storage for attention over token-interleaved peer pages."""

from dataclasses import asdict, dataclass, fields

import torch

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import Knob, MemoryRequirements, Plan, TuningContract

from ._preparation import _need


@dataclass(frozen=True, kw_only=True)
class InterleavedStageQuery:
    """Row slots address one peer. Token IDs address the shared logical sequence."""

    rows: int
    topk: int
    world_size: int
    page_size: int
    table_columns: int
    peer_pages: int
    capacity: int
    record_bytes: int

    def __post_init__(self):
        if any(type(x) is not int or not 0 < x < 2**31 for x in asdict(self).values()):
            raise ValueError(
                "interleaved staging dimensions must be positive int32 values"
            )
        if self.record_bytes % 4 or self.record_bytes > 4096:
            raise ValueError(
                "staged records must be word-aligned and at most 4096 bytes wide"
            )
        if (
            self.capacity < self.rows * self.topk
            or self.page_size * self.peer_pages >= 2**31
        ):
            raise ValueError(
                "the slab must fit a selection and peer slots must fit int32"
            )


@dataclass(frozen=True)
class _Config:
    records_per_block: int


def _validate_query(query, device):
    if not isinstance(query, InterleavedStageQuery):
        raise TypeError("expected an InterleavedStageQuery")


def _validate_config(query, config, device):
    if not isinstance(config, _Config) or config.records_per_block not in (4, 8, 16):
        raise ValueError("invalid interleaved staging block geometry")


TUNING = TuningContract(
    component_id="comm.interleaved_record_stage",
    query_schema_version=1,
    config_schema_version=1,
    query_fields=frozenset(f.name for f in fields(InterleavedStageQuery)),
    config_fields=frozenset({"records_per_block"}),
    encode_query=asdict,
    encode_config=asdict,
    decode_config=lambda payload: _Config(**dict(payload)),
    validate_query=_validate_query,
    validate_config=_validate_config,
    default_config=lambda query, device: _Config(8),
    knobs=(Knob(name="records_per_block", values=(4, 8, 16)),),
)


@dataclass(frozen=True)
class _State:
    query: InterleavedStageQuery
    device: torch.device
    launch: object
    grid: int
    layout: tuple

    def run(self, peers, table, end, tokens, slots, counts, out, first):
        from b12x.sequence.ple_hash._contracts import _require_mutation_alias_contract

        q = self.query
        _need("peers", peers, (q.world_size,), (torch.int64,), self.device)
        _need("table", table, (q.table_columns,), (torch.int32,), self.device)
        _need("end", end, (1,), (torch.int64,), self.device)
        for name, tensor in (("tokens", tokens), ("slots", slots)):
            _need(name, tensor, (q.rows, q.topk), (torch.int32,), self.device)
        _need("counts", counts, (q.rows,), (torch.int32,), self.device)
        _need(
            "out",
            out,
            (q.capacity, q.record_bytes),
            (torch.uint8, torch.float8_e4m3fn),
            self.device,
        )
        _require_mutation_alias_contract(
            mutable=(("out", out),),
            read_only=(
                ("peers", peers),
                ("table", table),
                ("end", end),
                ("tokens", tokens),
                ("slots", slots),
                ("counts", counts),
            ),
        )
        if type(first) is not bool:
            raise TypeError("first must identify the first selection slice")
        with torch.cuda.device(self.device):
            self.launch[(self.grid, 1, 1)](
                peers,
                table,
                end,
                tokens,
                slots,
                counts,
                out.view(torch.int32),
                int(first),
                *self.layout,
            )
        return out


@program_cache(scope="preparation")
def compile_stage(payload, tile, ordinal):
    import triton

    from ._kernel import _stage_interleaved

    q = InterleavedStageQuery(**dict(payload))
    with torch.cuda.device(ordinal):
        device = torch.device("cuda", ordinal)
        waves = torch.cuda.get_device_properties(device).multi_processor_count
        grid = min(triton.cdiv(q.capacity, tile), 4 * waves)
        words = q.record_bytes // 4
        layout = (
            q.rows,
            q.topk,
            q.world_size,
            q.page_size,
            q.table_columns,
            q.peer_pages,
            q.capacity,
            words,
            tile,
            triton.next_power_of_2(words),
        )
        launch = _stage_interleaved.warmup(
            torch.int64,
            torch.int32,
            torch.int64,
            torch.int32,
            torch.int32,
            torch.int32,
            torch.int32,
            2,
            *layout,
            grid=(grid,),
            num_warps=4,
        )
    return attach_programs(_State(q, device, launch, grid, layout), launch)


def plan_interleaved_stage(query, *, device, override=None):
    if torch.device(device).type != "cuda":
        raise ValueError("interleaved record staging requires CUDA")

    def materialize(selection, detected):
        state = compile_stage(
            TUNING.encode_query(query),
            selection.config.records_per_block,
            detected.ordinal,
        )
        load_programs(state)
        return state

    return Plan(
        contract=TUNING,
        query=query,
        override=override,
        _device=device,
        _compile_jobs=lambda config, detected: (
            CompileJob.create(
                "b12x.comm.peer_record_gather._stage:compile_stage",
                TUNING.encode_query(query),
                config.records_per_block,
                CompileJob.DEVICE_ORDINAL,
            ),
        ),
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )
