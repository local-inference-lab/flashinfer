"""Prepared native DeepSeek V4 and V4.1 compressed-cache writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import (
    FrozenMapping,
    MemoryRequirements,
    Plan,
    make_fixed_contract,
)
from b12x.preparation.types import require_prepared


@dataclass(frozen=True, kw_only=True)
class CacheWriterQuery:
    max_rows: int
    page_size: int
    cache_kind: str
    cache_format: str = "deepseek_v41"
    slot_dtype: str = "int64"
    snapshot_tokens: int = 0
    max_requests: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_query(query: CacheWriterQuery, _device) -> None:
    if not isinstance(query, CacheWriterQuery):
        raise TypeError("query must be CacheWriterQuery")
    if type(query.max_rows) is not int or query.max_rows <= 0:
        raise ValueError("max_rows must be a positive integer")
    if type(query.page_size) is not int or query.page_size <= 0:
        raise ValueError("page_size must be a positive integer")
    if query.cache_kind not in ("swa", "indexed"):
        raise ValueError("cache_kind must be 'swa' or 'indexed'")
    if query.cache_format not in ("deepseek_v4", "deepseek_v41"):
        raise ValueError("cache_format must be 'deepseek_v4' or 'deepseek_v41'")
    if query.slot_dtype not in ("int32", "int64"):
        raise ValueError("slot_dtype must be 'int32' or 'int64'")
    if type(query.snapshot_tokens) is not int or type(query.max_requests) is not int:
        raise TypeError("snapshot capacities must be integers")
    if query.snapshot_tokens == 0:
        if query.max_requests != 0:
            raise ValueError("max_requests requires a snapshot prefix")
    elif (
        query.snapshot_tokens != 128
        or query.max_requests <= 0
        or query.cache_format != "deepseek_v41"
        or query.cache_kind != "swa"
        or query.page_size <= query.snapshot_tokens
    ):
        raise ValueError(
            "snapshots require DSV41 SWA pages with 128 prefix records and a suffix"
        )
    if query.snapshot_tokens and 66 * query.max_requests + query.max_rows >= 2**31:
        raise ValueError("snapshot launch geometry must fit in int32")


TUNING = replace(
    make_fixed_contract(
        component_id="attention.compressed_sparse_mla.cache_writer",
        query_type=CacheWriterQuery,
        backend="cute",
    ),
    query_schema_version=24,
    validate_query=_validate_query,
)


def page_nbytes(
    page_size: int,
    *,
    cache_kind: str,
    cache_format: str = "deepseek_v41",
) -> int:
    from b12x.attention._shared.mla.kv_cache import page_nbytes as _page_nbytes

    return _page_nbytes(page_size, cache_kind=cache_kind, cache_format=cache_format)


def _slot_dtype(name: str) -> torch.dtype:
    return torch.int64 if name == "int64" else torch.int32


def _representative(query: CacheWriterQuery, ordinal: int):
    from b12x.attention._shared.mla import kv_cache as writer

    device = torch.device("cuda", ordinal)
    size = page_nbytes(
        query.page_size, cache_kind=query.cache_kind, cache_format=query.cache_format
    )
    operands = (
        writer,
        torch.empty((query.max_rows, 512), dtype=torch.bfloat16, device=device),
        torch.empty((query.max_requests or 1, size), dtype=torch.uint8, device=device),
        torch.empty(
            (query.max_rows,), dtype=_slot_dtype(query.slot_dtype), device=device
        ),
    )
    if not query.snapshot_tokens:
        return (*operands, None, None)
    return (
        *operands,
        torch.empty(
            (query.max_requests + 1, query.snapshot_tokens * 528),
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty((query.max_requests,), dtype=torch.int64, device=device),
    )


@program_cache(scope="preparation")
def compile_cache_writer(query_payload, ordinal):
    """Compiler-pool factory; intentionally private to the public API."""
    query = CacheWriterQuery(**dict(query_payload))
    _validate_query(query, None)
    with torch.cuda.device(ordinal), FakeTensorMode():
        writer, kv, cache, slots, history, history_rows = _representative(
            query, ordinal
        )
        if query.snapshot_tokens:
            kernel, args, spec = writer._snapshot_cache_writer_launch(
                kv,
                cache,
                slots,
                history,
                history_rows,
                query.page_size,
                query.snapshot_tokens,
            )
        else:
            kernel, args, spec = writer._compressed_cache_writer_launch(
                kv, cache, slots, query.page_size, query.cache_kind, query.cache_format
            )
        return writer.compile_cute(kernel, *args, compile_spec=spec)


@dataclass(frozen=True)
class _CacheWriterState:
    query: CacheWriterQuery
    compiled: object

    def run(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        history: torch.Tensor | None = None,
        history_rows: torch.Tensor | None = None,
    ) -> None:
        """Prime or execute this exact prepared writer specialization."""
        self.write(
            kv,
            cache,
            slot_mapping,
            page_size=None,
            cache_kind=None,
            cache_format=None,
            history=history,
            history_rows=history_rows,
        )

    def write(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        page_size: int | None,
        cache_kind: str | None,
        cache_format: str | None,
        history: torch.Tensor | None = None,
        history_rows: torch.Tensor | None = None,
    ) -> None:
        from b12x.attention._shared.mla import kv_cache as writer

        if page_size is not None and page_size != self.query.page_size:
            raise ValueError("page_size contradicts prepared cache-writer query")
        if cache_kind is not None and cache_kind != self.query.cache_kind:
            raise ValueError("cache_kind contradicts prepared cache-writer query")
        if cache_format is not None and cache_format != self.query.cache_format:
            raise ValueError("cache_format contradicts prepared cache-writer query")
        writer._validate_compressed_cache_writer(
            kv,
            cache,
            slot_mapping,
            self.query.page_size,
            self.query.cache_kind,
            self.query.cache_format,
        )
        if str(slot_mapping.dtype).removeprefix("torch.") != self.query.slot_dtype:
            raise ValueError(
                "slot_mapping dtype contradicts prepared cache-writer query"
            )
        if int(slot_mapping.shape[0]) > self.query.max_rows:
            raise ValueError("cache-writer rows exceed prepared capacity")
        if self.query.snapshot_tokens:
            writer._validate_cache_snapshot(
                history,
                history_rows,
                kv,
                cache,
                slot_mapping,
                snapshot_tokens=self.query.snapshot_tokens,
                max_requests=self.query.max_requests,
            )
            args = writer._snapshot_cache_writer_args(
                kv, cache, slot_mapping, history, history_rows, self.query.page_size
            )
        else:
            if history is not None or history_rows is not None:
                raise ValueError("history operands require a declared snapshot writer")
            if int(slot_mapping.shape[0]) == 0:
                return
            args = writer._compressed_cache_writer_args(
                kv,
                cache,
                slot_mapping,
                self.query.page_size,
                self.query.cache_kind,
                self.query.cache_format,
            )
        writer.run_compiled(self.compiled, args)


def plan(
    query: CacheWriterQuery,
    *,
    device: object,
    invocation: FrozenMapping = FrozenMapping(),
    override=None,
) -> Plan:
    """Declare one cache-writer specialization without allocating or compiling."""
    _validate_query(query, None)
    invocation = FrozenMapping(invocation)
    if invocation:
        raise ValueError("cache-writer invocation metadata is not supported")
    device = torch.device(device)

    def jobs(_config, detected):
        return (
            CompileJob.create(
                "b12x.attention.compressed_sparse_mla.cache_writer:compile_cache_writer",
                query.to_dict(),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def materialize(_selection, detected):
        compiled = compile_cache_writer(query.to_dict(), detected.ordinal)
        load_programs(compiled)
        return attach_programs(_CacheWriterState(query, compiled), compiled)

    return Plan(
        contract=TUNING,
        query=query,
        invocation=invocation,
        override=override,
        _compile_jobs=jobs,
        _memory_requirements=lambda _config, _device: MemoryRequirements(),
        _materialize=materialize,
        _device=device,
    )


def write_cache(
    kv: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    plan: Plan,
    page_size: int | None = None,
    cache_kind: str | None = None,
    cache_format: str | None = None,
    history: torch.Tensor | None = None,
    history_rows: torch.Tensor | None = None,
) -> None:
    """Encode records, optionally copying an immutable 128-record history prefix.

    Snapshot mode maps every destination page through an int64 history row ID,
    including row zero. Appends targeting the reserved prefix, negative slots,
    and slots beyond destination capacity are ignored. History and row IDs are
    read-only and must not overlap the destination. The source history row IDs
    must refer to backed rows; invalid IDs leave their destination prefix alone.
    """
    state = require_prepared(
        plan, "attention.compressed_sparse_mla.cache_writer", kv.device
    )
    state.write(
        kv,
        cache,
        slot_mapping,
        page_size=page_size,
        cache_kind=cache_kind,
        cache_format=cache_format,
        history=history,
        history_rows=history_rows,
    )


__all__ = ["CacheWriterQuery", "page_nbytes", "plan", "write_cache"]
