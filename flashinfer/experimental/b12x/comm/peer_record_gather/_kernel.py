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


@triton.jit
def _gather_records_words(
    peer_bases,
    owners,
    slots,
    counts,
    out,
    TOTAL: tl.constexpr,
    TOPK: tl.constexpr,
    WORLD: tl.constexpr,
    PEER_RECORDS: tl.constexpr,
    WORDS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    column = tl.arange(0, BLOCK)
    col_valid = column < WORDS
    start = tl.program_id(0).to(tl.int64) * TILE
    step = tl.num_programs(0).to(tl.int64) * TILE
    while start < TOTAL:
        record = start + tl.arange(0, TILE)
        in_bounds = record < TOTAL
        row = record // TOPK
        col = record % TOPK
        owner = tl.load(owners + record, mask=in_bounds, other=-1)
        slot = tl.load(slots + record, mask=in_bounds, other=-1)
        count = tl.load(counts + row, mask=in_bounds, other=0)
        valid = (
            in_bounds
            & (col < count)
            & (owner >= 0)
            & (owner < WORLD)
            & (slot >= 0)
            & (slot < PEER_RECORDS)
        )
        base = tl.load(peer_bases + owner, mask=valid, other=0).to(tl.uint64)
        valid = valid & (base != 0)
        addr = base.to(tl.pointer_type(tl.int32)) + slot.to(tl.int64) * WORDS
        data = tl.load(
            addr[:, None] + column[None, :],
            mask=valid[:, None] & col_valid[None, :],
            other=0,
        )
        tl.store(
            out + record[:, None] * WORDS + column[None, :],
            data,
            mask=in_bounds[:, None] & col_valid[None, :],
        )
        start += step


@triton.jit
def _gather_records_words_chunked(
    peer_bases,
    owners,
    slots,
    counts,
    out,
    TOPK: tl.constexpr,
    WORLD: tl.constexpr,
    PEER_RECORDS: tl.constexpr,
    WORDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    record = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1).to(tl.int64)
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
    src_base = base.to(tl.pointer_type(tl.int32)) + slot.to(tl.int64) * WORDS
    dst_base = out + record * WORDS
    col = split * CHUNK + tl.arange(0, CHUNK)
    col_mask = col < WORDS
    data = tl.load(src_base + col, mask=valid & col_mask, other=0)
    tl.store(dst_base + col, data, mask=col_mask)


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
        column = tl.arange(0, BLOCK)
        col_mask = column < WORDS
        if prefix:
            # Group contiguous local slots from one peer per CTA so each warp streams
            # contiguous peer DRAM instead of striding across all WORLD peers.
            per_peer = (end + WORLD - 1) // WORLD
            blocks_per_peer = (per_peer + TILE - 1) // TILE
            total_blocks = blocks_per_peer * WORLD
            bid = tl.program_id(0).to(tl.int64)
            step = tl.num_programs(0).to(tl.int64)
            while bid < total_blocks:
                peer = bid % WORLD
                local_blk = bid // WORLD
                local_idx = local_blk * TILE + tl.arange(0, TILE)
                token = local_idx * WORLD + peer
                live = token < end
                page_column = local_idx // PAGE
                page = tl.load(
                    Table + page_column, live & (page_column < COLUMNS), other=-1
                ).to(tl.int64)
                slot = page * PAGE + local_idx % PAGE
                live &= (slot >= 0) & (slot < PEER_PAGES * PAGE)
                base = tl.load(Peers + peer).to(tl.uint64)
                addr = base.to(tl.pointer_type(tl.int32)) + slot * WORDS
                value = tl.load(
                    addr[:, None] + column[None, :],
                    ((base != 0) & live)[:, None] & col_mask[None, :],
                    other=0,
                )
                tl.store(
                    Out + token[:, None] * WORDS + column[None, :],
                    value,
                    (token < end)[:, None] & col_mask[None, :],
                )
                bid += step
        else:
            limit = ROWS * TOPK
            start = tl.program_id(0).to(tl.int64) * TILE
            while start < limit:
                row = start + tl.arange(0, TILE)
                live = row < limit
                token = tl.load(Tokens + row, live, other=-1).to(tl.int64)
                slot = tl.load(Slots + row, live, other=-1).to(tl.int64)
                count = tl.load(Counts + row // TOPK, live, other=0)
                live &= (row % TOPK < count) & (token >= 0) & (token < end)
                live &= (slot >= 0) & (slot < PEER_PAGES * PAGE)
                base = tl.load(Peers + token % WORLD, live, other=0).to(tl.uint64)
                addr = base.to(tl.pointer_type(tl.int32)) + slot * WORDS
                value = tl.load(
                    addr[:, None] + column[None, :],
                    ((base != 0) & live)[:, None] & col_mask[None, :],
                    other=0,
                )
                tl.store(
                    Out + row[:, None] * WORDS + column[None, :],
                    value,
                    (row < limit)[:, None] & col_mask[None, :],
                )
                start += tl.num_programs(0) * TILE
