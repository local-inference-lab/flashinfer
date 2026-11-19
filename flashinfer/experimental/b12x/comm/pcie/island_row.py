# SPDX-License-Identifier: Apache-2.0
"""Four-row BF16 reduction for Triton epilogues using an established TP16 PCIe channel.

Each CTA completes one row through the two four-input, BF16-rounded sums of the
native island protocol. Every physical CTA advances its bank counter. Only the
first four advance participant generations. Calls on the channel remain serialized.
"""

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from ._island_rs_cute import MAX_BLOCKS, _AG_ARRIVED, _RS_ARRIVED, _XS_ARRIVED

PHYSICAL_COUNTERS = tl.constexpr(MAX_BLOCKS)
LOCAL_READY = tl.constexpr(_RS_ARRIVED)
CROSS_READY = tl.constexpr(_XS_ARRIVED)
FINAL_READY = tl.constexpr(_AG_ARRIVED)


@triton.jit
def _signal_peers(p0, p1, p2, p3, own, row, gen, peer_id, region: tl.constexpr):
    # A scalar Triton instruction is issued by the whole CTA. Exactly three
    # hardware lanes must perform the system fence and peer handshake.
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,pending;
          .reg .u32 t,seen,o;
          .reg .s32 diff;
          .reg .u64 to,from,region;
          mov.u32 $0,0; cvt.u64.u32 region,$9;
          mov.u32 t,%tid.x;
          setp.ge.u32 sel,t,4;
          @sel bra DONE;
          setp.eq.u32 sel,t,$8;
          @sel bra DONE;
          mov.u64 to,$1;
          setp.eq.u32 sel,t,1; selp.u64 to,$2,to,sel;
          setp.eq.u32 sel,t,2; selp.u64 to,$3,to,sel;
          setp.eq.u32 sel,t,3; selp.u64 to,$4,to,sel;
          mad.lo.u32 o,$6,4,$8;
          mad.wide.u32 to,o,128,to;
          add.u64 to,to,region;
          mad.lo.u32 o,$6,4,t;
          mad.wide.u32 from,o,128,$5;
          add.u64 from,from,region;
          fence.sc.sys;
          st.relaxed.sys.global.u32 [to],$7;
          AGAIN: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7;
          setp.lt.s32 pending,diff,0;
          @pending nanosleep.u32 24;
          @pending bra AGAIN;
          DONE:
        }""",
        constraints="=r,l,l,l,l,l,r,r,r,r",
        args=[p0, p1, p2, p3, own, row, gen, peer_id, region],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_packets(source, destination, packets: tl.constexpr):
    tl.inline_asm_elementwise(
        asm="""{
        .reg .pred done;
        .reg .u32 t,n,v0,v1,v2,v3; .reg .u64 a,b;
        mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;
        NEXT: setp.ge.u32 done,t,$3; @done bra DONE;
        mad.wide.u32 a,t,16,$1; mad.wide.u32 b,t,16,$2;
        ld.global.cg.v4.b32 {v0,v1,v2,v3},[a]; st.global.v4.b32 [b],{v0,v1,v2,v3};
        add.u32 t,t,n; bra NEXT;
        DONE:
    }""",
        constraints="=r,l,l,r",
        args=[source.to(tl.uint64), destination.to(tl.uint64), packets],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


def _sum_packets(destinations):
    # Four adjacent BF16 pairs make one PCIe transaction. Start both accumulators
    # at positive zero, matching the signed native reduction even for all -0 inputs.
    lines = [
        "{ .reg .pred done; .reg .u32 t,n; .reg .u64 addr;",
        ".reg .b32 a<4>,b<4>,c<4>,d<4>,v<4>; .reg .b16 lo,hi; .reg .f32 x,y,sumlo,sumhi;",
        "mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;",
        f"NEXT: setp.ge.u32 done,t,${5 + destinations}; @done bra DONE;",
    ]
    for i, reg in enumerate("abcd"):
        lines.extend(
            [
                f"mad.wide.u32 addr,t,16,${i + 1};",
                "ld.global.cg.v4.b32 {"
                + ",".join(f"{reg}{j}" for j in range(4))
                + "},[addr];",
            ]
        )
    for j in range(4):
        lines.extend(["mov.f32 sumlo,0f00000000; mov.f32 sumhi,0f00000000;"])
        for reg in "abcd":
            lines.extend(
                [
                    f"mov.b32 {{lo,hi}},{reg}{j};",
                    "cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi;",
                    "add.rn.f32 sumlo,sumlo,x; add.rn.f32 sumhi,sumhi,y;",
                ]
            )
        lines.append(f"cvt.rn.bf16x2.f32 v{j},sumhi,sumlo;")
    for j in range(destinations):
        lines.append(
            f"mad.wide.u32 addr,t,16,${5 + j}; st.global.v4.b32 [addr],{{v0,v1,v2,v3}};"
        )
    lines.extend(["add.u32 t,t,n; bra NEXT; DONE: }"])
    return "\n".join(lines)


SUM_ONE = tl.constexpr(_sum_packets(1))
SUM_FOUR = tl.constexpr(_sum_packets(4))


@triton.jit
def _ptr(first, second, third, fourth, lane):
    return tl.where(
        lane < 2, tl.where(lane == 0, first, second), tl.where(lane == 2, third, fourth)
    ).to(tl.uint64)


@triton.jit
def reduce_rows(
    Input,
    Output,
    L0,
    L1,
    L2,
    L3,
    X0,
    X1,
    X2,
    X3,
    stage,
    part,
    final,
    capacity,
    LANE,
    ISLAND,
    WIDTH: tl.constexpr,
):
    row = tl.program_id(0)
    # IPC addresses are scalar arguments rather than embedded constants. A
    # valid mapping in the low address range must still use a 64-bit PTX operand.
    L0, L1, L2, L3 = (
        L0.to(tl.uint64),
        L1.to(tl.uint64),
        L2.to(tl.uint64),
        L3.to(tl.uint64),
    )
    X0, X1, X2, X3 = (
        X0.to(tl.uint64),
        X1.to(tl.uint64),
        X2.to(tl.uint64),
        X3.to(tl.uint64),
    )
    own = _ptr(L0, L1, L2, L3, LANE)
    counter = own.to(tl.pointer_type(tl.uint32))
    parity = (tl.atomic_add(counter + row, 1, sem="relaxed") + 1) & 1
    if row < 4:
        gen = tl.atomic_add(counter + PHYSICAL_COUNTERS + row, 1, sem="relaxed") + 1
        quarter: tl.constexpr = WIDTH // 4
        qbytes: tl.constexpr = 2 * quarter
        packets: tl.constexpr = quarter // 8
        so = stage + parity.to(tl.int64) * capacity * 16
        po = part + parity.to(tl.int64) * capacity * 4
        fo = final + parity.to(tl.int64) * capacity * 16
        local = (L0, L1, L2, L3)
        source = Input.to(tl.uint64)
        output = Output.to(tl.uint64)
        for dest in tl.static_range(4):
            if dest != LANE:
                _copy_packets(
                    source + (4 * row + dest) * qbytes,
                    local[dest] + so + (4 * LANE + row) * qbytes,
                    packets,
                )
        tl.debug_barrier()
        _signal_peers(L0, L1, L2, L3, own, row, gen, LANE, LOCAL_READY)
        tl.debug_barrier()
        local_values = source + (4 * row + LANE) * qbytes
        a = tl.where(LANE == 0, local_values, own + so + row * qbytes)
        b = tl.where(LANE == 1, local_values, own + so + (4 + row) * qbytes)
        c = tl.where(LANE == 2, local_values, own + so + (8 + row) * qbytes)
        d = tl.where(LANE == 3, local_values, own + so + (12 + row) * qbytes)
        tl.inline_asm_elementwise(
            asm=SUM_ONE,
            constraints="=r,l,l,l,l,l,r",
            args=[a, b, c, d, own + po + row * qbytes, packets],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        _signal_peers(X0, X1, X2, X3, own, row, gen * 2, ISLAND, CROSS_READY)
        tl.debug_barrier()
        rowoff = po + row * qbytes
        col = (4 * row + LANE) * qbytes
        d0 = tl.where(LANE == 0, output, L0 + fo) + col
        d1 = tl.where(LANE == 1, output, L1 + fo) + col
        d2 = tl.where(LANE == 2, output, L2 + fo) + col
        d3 = tl.where(LANE == 3, output, L3 + fo) + col
        tl.inline_asm_elementwise(
            asm=SUM_FOUR,
            constraints="=r,l,l,l,l,l,l,l,l,r",
            args=[
                X0 + rowoff,
                X1 + rowoff,
                X2 + rowoff,
                X3 + rowoff,
                d0,
                d1,
                d2,
                d3,
                packets,
            ],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
        tl.debug_barrier()
        _signal_peers(L0, L1, L2, L3, own, row, gen, LANE, FINAL_READY)
        tl.debug_barrier()
        for sender in tl.static_range(4):
            if sender != LANE:
                col = (4 * row + sender) * qbytes
                _copy_packets(own + fo + col, output + col, packets)
        tl.debug_barrier()


@dataclass(frozen=True)
class RowReduction:
    """Borrow the six remote mappings and the local slab from a prepared reducer."""

    owner: object
    width: int

    @property
    def grid(self):
        return (self.owner.blocks,)

    @property
    def rank_coordinates(self):
        return self.owner.rank % 4, self.owner.rank // 4

    def arguments(self, source, destination):
        from .pcie_island_rs import _validate_output_overlap

        owner = self.owner
        if not owner.should_allreduce(source) or source.shape != (4, self.width):
            raise ValueError(
                "PCIe reduction source must be four contiguous BF16 rows on its owner"
            )
        if (
            source.data_ptr() % 16
            or destination.data_ptr() % 16
            or not destination.is_contiguous()
            or destination.dtype != torch.bfloat16
            or destination.shape != source.shape
            or destination.device != owner.device
        ):
            raise ValueError(
                "PCIe row reduction requires matching 16-byte aligned BF16 rows"
            )
        _validate_output_overlap(source, destination)
        lane, island = self.rank_coordinates
        stages = tuple(owner._slab_ptrs[4 * island + i] for i in range(4))
        across = tuple(owner._slab_ptrs[4 * i + lane] for i in range(4))
        return (
            *stages,
            *across,
            owner.stage_offset,
            owner.part_offset,
            owner.final_offset,
            owner.quarter_capacity,
        )


def prepared_rows(plan, *, rows, width, threads):
    """Return the native four-row layout if the selected owner and explicit grid permit it."""
    from .pcie_island_rs import PCIeIslandRSAllReduce
    from ._tuning import pairwise_device

    if plan.prepared is None or rows != 4 or width not in (3584, 7168):
        return None
    if not pairwise_device(plan.prepared.device.identity):
        return None
    state = plan.prepared.state
    owner = getattr(state, "runtime", None)
    if not isinstance(owner, PCIeIslandRSAllReduce):
        return None
    for operand in (state.query.call["inp"], state.query.call["output"]):
        if (
            operand is None
            or operand["alignment"] < 16
            or operand["dtype"] != "torch.bfloat16"
            or tuple(operand["shape"]) != (rows, width)
            or tuple(operand["stride"]) != (width, 1)
        ):
            return None
    if (
        owner.world_size != 16
        or owner.blocks < rows
        or owner.max_elements < rows * width
        or owner.quarter_capacity * 4 < rows * width // 4 * 2
        or owner._group_timing_records is not None
        or (state.query.setup.get("threads_override") and owner.threads != threads)
        or (
            state.query.setup.get("active_blocks_override")
            and state.query.call["active_blocks"] != rows
        )
    ):
        return None
    return RowReduction(owner, width)
