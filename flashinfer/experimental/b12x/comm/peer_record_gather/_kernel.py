"""Masked record loads over caller-provided peer base addresses."""

import triton
import triton.language as tl


@triton.jit
def _gather_records(
    peer_bases,
    owners,
    slots,
    counts,
    out,
    TOPK: tl.constexpr,
    WORLD: tl.constexpr,
    PEER_RECORDS: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    record = tl.program_id(0)
    row = record // TOPK
    column = record % TOPK
    owner = tl.load(owners + record)
    slot = tl.load(slots + record)
    count = tl.load(counts + row)
    valid = (
        (column < count)
        & (owner >= 0)
        & (owner < WORLD)
        & (slot >= 0)
        & (slot < PEER_RECORDS)
    )
    base = tl.load(peer_bases + owner, mask=valid, other=0).to(tl.uint64)
    valid = valid & (base != 0)
    offset = tl.arange(0, BLOCK)
    address = (base + slot.to(tl.uint64) * RECORD_BYTES + offset).to(
        tl.pointer_type(tl.uint8)
    )
    data = tl.load(address, mask=valid & (offset < RECORD_BYTES), other=0)
    tl.store(
        out + record.to(tl.int64) * RECORD_BYTES + offset,
        data,
        mask=offset < RECORD_BYTES,
    )


@triton.jit(do_not_specialize=["first"])
def _stage_interleaved(
    Peers,
    Table,
    End,
    Tokens,
    Slots,
    Counts,
    Out,
    first,
    ROWS: tl.constexpr,
    TOPK: tl.constexpr,
    WORLD: tl.constexpr,
    PAGE: tl.constexpr,
    COLUMNS: tl.constexpr,
    PEER_PAGES: tl.constexpr,
    CAPACITY: tl.constexpr,
    WORDS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    end = tl.maximum(tl.load(End), 0)
    prefix = end <= CAPACITY
    if not prefix or first != 0:
        limit = tl.where(prefix, end, ROWS * TOPK)
        column = tl.arange(0, BLOCK)
        start = tl.program_id(0).to(tl.int64) * TILE
        while start < limit:
            row = start + tl.arange(0, TILE)
            live = row < limit
            if prefix:
                token = row
                local = token // WORLD
                page_column = local // PAGE
                page = tl.load(
                    Table + page_column, live & (page_column < COLUMNS), other=-1
                ).to(tl.int64)
                slot = page * PAGE + local % PAGE
            else:
                token = tl.load(Tokens + row, live, other=-1).to(tl.int64)
                slot = tl.load(Slots + row, live, other=-1).to(tl.int64)
                count = tl.load(Counts + row // TOPK, live, other=0)
                live &= (row % TOPK < count) & (token >= 0) & (token < end)
            live &= (slot >= 0) & (slot < PEER_PAGES * PAGE)
            base = tl.load(Peers + token % WORLD, live, other=0).to(tl.uint64)
            addr = base.to(tl.pointer_type(tl.int32)) + slot * WORDS
            value = tl.load(
                addr[:, None] + column[None, :],
                ((base != 0) & live)[:, None] & (column < WORDS)[None, :],
                other=0,
            )
            tl.store(
                Out + row[:, None] * WORDS + column[None, :],
                value,
                (row < limit)[:, None] & (column < WORDS)[None, :],
            )
            start += tl.num_programs(0) * TILE
