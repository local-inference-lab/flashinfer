"""Host validation for immutable history snapshots and prepared writer identities."""

from dataclasses import replace

import pytest
import torch

from b12x.attention.compressed_sparse_mla.cache_writer import (
    CacheWriterQuery,
    TUNING,
    _validate_query,
)
from b12x.attention._shared.mla import kv_cache as writer


def snapshot_query(**changes):
    return replace(
        CacheWriterQuery(
            max_rows=8,
            page_size=136,
            cache_kind="swa",
            snapshot_tokens=128,
            max_requests=1,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"snapshot_tokens": 64},
        {"max_requests": 0},
        {"page_size": 128},
        {"cache_format": "deepseek_v4"},
        {"cache_kind": "indexed"},
        {"snapshot_tokens": 0},
        {"snapshot_tokens": True},
        {"max_requests": 2**31},
    ],
)
def test_snapshot_query_rejects_incompatible_contracts(changes):
    with pytest.raises((TypeError, ValueError)):
        _validate_query(snapshot_query(**changes), None)


def test_snapshot_query_records_geometry_and_preserves_plain_writer_defaults():
    plain = CacheWriterQuery(max_rows=8, page_size=136, cache_kind="swa")
    snapshot = snapshot_query()
    for query in (plain, snapshot):
        _validate_query(query, None)
        encoded = TUNING.encode_query(query)
        assert set(encoded) == TUNING.query_fields
    assert TUNING.encode_query(plain) != TUNING.encode_query(snapshot)
    assert (plain.snapshot_tokens, plain.max_requests) == (0, 0)
    assert TUNING.query_schema_version == 24


def operands():
    history = torch.zeros((5, 128 * 528 + 256), dtype=torch.uint8)[:, : 128 * 528]
    cache = torch.empty((2, 136 * 528 + 32), dtype=torch.uint8)[:, : 136 * 528]
    rows = torch.tensor([4, 0], dtype=torch.int64)
    kv = torch.empty((8, 512), dtype=torch.bfloat16)
    slots = torch.empty(8, dtype=torch.int64)
    return history, rows, kv, cache, slots


def test_snapshot_layout_accepts_padded_source_and_destination_rows():
    args = operands()
    writer._validate_cache_snapshot(*args, snapshot_tokens=128, max_requests=2)


@pytest.mark.parametrize(
    "defect", ["rows_dtype", "rows_stride", "page_count", "history_width", "alias"]
)
def test_snapshot_layout_rejects_unsupported_views_and_read_write_aliases(defect):
    history, rows, kv, cache, slots = operands()
    if defect == "rows_dtype":
        rows = rows.int()
    elif defect == "rows_stride":
        rows = torch.zeros(4, dtype=torch.int64)[::2]
    elif defect == "page_count":
        cache = cache[:1]
    elif defect == "history_width":
        history = history[:, :-1]
    else:
        history = cache[:, : 128 * 528]
    with pytest.raises(ValueError):
        writer._validate_cache_snapshot(
            history, rows, kv, cache, slots, snapshot_tokens=128, max_requests=2
        )


def test_snapshot_program_identity_ignores_layer_offsets_and_runtime_row_strides(
    monkeypatch,
):
    # Only construct program facts; CUDA tensor bindings and compilation are not exercised.
    monkeypatch.setattr(writer, "_snapshot_cache_writer_args", lambda *args: ())
    history, rows, kv, cache, slots = operands()
    first = writer._snapshot_cache_writer_launch(
        kv, cache, slots, history, rows, 136, 128
    )[2]
    backing = torch.empty((9, 128 * 528 + 512), dtype=torch.uint8)
    shifted = backing[:, 256 : 256 + 128 * 528]
    wider_cache = torch.empty((2, 136 * 528 + 64), dtype=torch.uint8)[:, : 136 * 528]
    strided_kv = torch.empty((8, 520), dtype=torch.bfloat16)[:, :512]
    second = writer._snapshot_cache_writer_launch(
        strided_kv, wider_cache, slots, shifted, rows, 136, 128
    )[2]
    assert first.hash_key == second.hash_key
    changed = writer._snapshot_cache_writer_launch(
        kv, cache, slots, history, rows, 135, 128
    )[2]
    assert first.hash_key != changed.hash_key
