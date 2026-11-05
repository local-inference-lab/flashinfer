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
