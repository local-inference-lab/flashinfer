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
def _signal_one_peer(peer_ptr, own, row, gen, self_id, peer_id, region: tl.constexpr):
    """Strict 1-to-1 single-peer handshake on one existing flag cache line."""
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,pending;
          .reg .u32 t,seen,o;
          .reg .s32 diff;
          .reg .u64 to,from,region;
          mov.u32 $0,0; cvt.u64.u32 region,$7;
          mov.u32 t,%tid.x;
          setp.ne.u32 sel,t,0;
          @sel bra DONE;
          mad.lo.u32 o,$3,4,$5;
          mad.wide.u32 to,o,128,$1;
          add.u64 to,to,region;
          mad.lo.u32 o,$3,4,$6;
          mad.wide.u32 from,o,128,$2;
          add.u64 from,from,region;
          fence.sc.sys;
          st.relaxed.sys.global.u32 [to],$4;
          AGAIN: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$4;
          setp.lt.s32 pending,diff,0;
          @pending nanosleep.u32 24;
          @pending bra AGAIN;
          DONE:
        }""",
        constraints="=r,l,l,r,r,r,r,r",
        args=[peer_ptr, own, row, gen, self_id, peer_id, region],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _load_grid_epoch(own, lane: tl.constexpr):
    return tl.inline_asm_elementwise(
        asm="""{
          .reg .u64 addr;
          add.u64 addr, $1, $2;
          ld.relaxed.gpu.global.u32 $0, [addr];
        }""",
        constraints="=r,l,l",
        args=[own, tl.full((), LOCAL_READY + lane * 128 + 68, tl.uint64)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _grid_pair_sync(
    peer_ptr,
    own,
    rows: tl.constexpr,
    tag,
    self_id: tl.constexpr,
    peer_id: tl.constexpr,
    region: tl.constexpr,
    lane: tl.constexpr,
):
    """GPU-local grid barrier + 1 single-peer PCIe flag exchange (zero cross-CTA drift)."""
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,is_last,pending;
          .reg .u32 t,old,seen;
          .reg .s32 diff;
          .reg .u64 arr,rel,to,from;
          mov.u32 $0,0;
          mov.u32 t,%tid.x;
          setp.ne.u32 sel,t,0;
          @sel bra DONE;
          add.u64 arr,$2,$7;
          add.u64 rel,arr,4;
          fence.sc.sys;
          atom.acq_rel.gpu.global.add.u32 old,[arr],1;
          setp.eq.u32 is_last,old,$3;
          @!is_last bra WAIT_REL;
          st.relaxed.gpu.global.u32 [arr],0;
          add.u64 to,$1,$5;
          add.u64 from,$2,$6;
          st.release.sys.global.u32 [to],$4;
          AGAIN_PEER: ld.acquire.sys.global.u32 seen,[from];
          sub.s32 diff,seen,$4;
          setp.lt.s32 pending,diff,0;
          @pending bra AGAIN_PEER;
          st.release.gpu.global.u32 [rel],$4;
          bra DONE;
          WAIT_REL: ld.acquire.gpu.global.u32 seen,[rel];
          sub.s32 diff,seen,$4;
          setp.lt.s32 pending,diff,0;
          @pending bra WAIT_REL;
          DONE:
        }""",
        constraints="=r,l,l,r,r,l,l,l",
        args=[
            peer_ptr,
            own,
            rows - 1,
            tag,
            tl.full((), region + self_id * 128 + 96, tl.uint64),
            tl.full((), region + peer_id * 128 + 96, tl.uint64),
            tl.full((), LOCAL_READY + lane * 128 + 64, tl.uint64),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _grid_peers_sync(
    p0,
    p1,
    p2,
    p3,
    own,
    rows: tl.constexpr,
    tag,
    self_id: tl.constexpr,
    region: tl.constexpr,
    lane: tl.constexpr,
):
    """GPU-local grid barrier + 1 leader CTA exchanging 3 peer flags."""
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,is_last,pending,skip;
          .reg .u32 t,old,seen;
          .reg .s32 diff;
          .reg .u64 arr,rel,to,from;
          mov.u32 $0,0;
          mov.u32 t,%tid.x;
          setp.ne.u32 sel,t,0;
          @sel bra DONE;
          add.u64 arr,$5,$9;
          add.u64 rel,arr,4;
          fence.sc.sys;
          atom.acq_rel.gpu.global.add.u32 old,[arr],1;
          setp.eq.u32 is_last,old,$6;
          @!is_last bra WAIT_REL;
          st.relaxed.gpu.global.u32 [arr],0;
          setp.eq.u32 skip,$8,0;
          @skip bra P1;
          add.u64 to,$1,$10;
          st.release.sys.global.u32 [to],$7;
          P1: setp.eq.u32 skip,$8,1;
          @skip bra P2;
          add.u64 to,$2,$10;
          st.release.sys.global.u32 [to],$7;
          P2: setp.eq.u32 skip,$8,2;
          @skip bra P3;
          add.u64 to,$3,$10;
          st.release.sys.global.u32 [to],$7;
          P3: setp.eq.u32 skip,$8,3;
          @skip bra W0;
          add.u64 to,$4,$10;
          st.release.sys.global.u32 [to],$7;
          W0: setp.eq.u32 skip,$8,0;
          @skip bra W1;
          add.u64 from,$5,$11;
          A0: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A0;
          W1: setp.eq.u32 skip,$8,1;
          @skip bra W2;
          add.u64 from,$5,$12;
          A1: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A1;
          W2: setp.eq.u32 skip,$8,2;
          @skip bra W3;
          add.u64 from,$5,$13;
          A2: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A2;
          W3: setp.eq.u32 skip,$8,3;
          @skip bra RELEASE;
          add.u64 from,$5,$14;
          A3: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A3;
          RELEASE: st.release.gpu.global.u32 [rel],$7;
          bra DONE;
          WAIT_REL: ld.acquire.gpu.global.u32 seen,[rel];
          sub.s32 diff,seen,$7;
          setp.lt.s32 pending,diff,0;
          @pending bra WAIT_REL;
          DONE:
        }""",
        constraints="=r,l,l,l,l,l,r,r,r,l,l,l,l,l,l",
        args=[
            p0,
            p1,
            p2,
            p3,
            own,
            rows - 1,
            tag,
            self_id,
            tl.full((), LOCAL_READY + lane * 128 + 64, tl.uint64),
            tl.full((), region + self_id * 128 + 96, tl.uint64),
            tl.full((), region + 0 * 128 + 96, tl.uint64),
            tl.full((), region + 1 * 128 + 96, tl.uint64),
            tl.full((), region + 2 * 128 + 96, tl.uint64),
            tl.full((), region + 3 * 128 + 96, tl.uint64),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _signal_peers_nosleep(p0, p1, p2, p3, own, row, gen, peer_id, region: tl.constexpr):
    """Per-CTA 3-peer barrier with tight acquire spin (lowest latency for small rows)."""
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
def _grid_pair_sync_fast(
    peer_ptr,
    own,
    rows: tl.constexpr,
    tag,
    self_id: tl.constexpr,
    peer_id: tl.constexpr,
    region: tl.constexpr,
    lane: tl.constexpr,
):
    """Single-fence grid barrier + 1 single-peer PCIe flag exchange (relaxed leader write)."""
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,is_last,pending;
          .reg .u32 t,old,seen;
          .reg .s32 diff;
          .reg .u64 arr,rel,to,from;
          mov.u32 $0,0;
          mov.u32 t,%tid.x;
          setp.ne.u32 sel,t,0;
          @sel bra DONE;
          add.u64 arr,$2,$7;
          add.u64 rel,arr,4;
          fence.sc.sys;
          atom.acq_rel.gpu.global.add.u32 old,[arr],1;
          setp.eq.u32 is_last,old,$3;
          @!is_last bra WAIT_REL;
          st.relaxed.gpu.global.u32 [arr],0;
          add.u64 to,$1,$5;
          add.u64 from,$2,$6;
          st.relaxed.sys.global.u32 [to],$4;
          AGAIN_PEER: ld.acquire.sys.global.u32 seen,[from];
          sub.s32 diff,seen,$4;
          setp.lt.s32 pending,diff,0;
          @pending bra AGAIN_PEER;
          st.release.gpu.global.u32 [rel],$4;
          bra DONE;
          WAIT_REL: ld.acquire.gpu.global.u32 seen,[rel];
          sub.s32 diff,seen,$4;
          setp.lt.s32 pending,diff,0;
          @pending bra WAIT_REL;
          DONE:
        }""",
        constraints="=r,l,l,r,r,l,l,l",
        args=[
            peer_ptr,
            own,
            rows - 1,
            tag,
            tl.full((), region + self_id * 128 + 96, tl.uint64),
            tl.full((), region + peer_id * 128 + 96, tl.uint64),
            tl.full((), LOCAL_READY + lane * 128 + 64, tl.uint64),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _grid_peers_sync_fast(
    p0,
    p1,
    p2,
    p3,
    own,
    rows: tl.constexpr,
    tag,
    self_id: tl.constexpr,
    region: tl.constexpr,
    lane: tl.constexpr,
):
    """Single-fence grid barrier + 1 leader CTA exchanging 3 peer flags (relaxed leader writes)."""
    tl.inline_asm_elementwise(
        asm="""{
          .reg .pred sel,is_last,pending,skip;
          .reg .u32 t,old,seen;
          .reg .s32 diff;
          .reg .u64 arr,rel,to,from;
          mov.u32 $0,0;
          mov.u32 t,%tid.x;
          setp.ne.u32 sel,t,0;
          @sel bra DONE;
          add.u64 arr,$5,$9;
          add.u64 rel,arr,4;
          fence.sc.sys;
          atom.acq_rel.gpu.global.add.u32 old,[arr],1;
          setp.eq.u32 is_last,old,$6;
          @!is_last bra WAIT_REL;
          st.relaxed.gpu.global.u32 [arr],0;
          setp.eq.u32 skip,$8,0;
          @skip bra P1;
          add.u64 to,$1,$10;
          st.relaxed.sys.global.u32 [to],$7;
          P1: setp.eq.u32 skip,$8,1;
          @skip bra P2;
          add.u64 to,$2,$10;
          st.relaxed.sys.global.u32 [to],$7;
          P2: setp.eq.u32 skip,$8,2;
          @skip bra P3;
          add.u64 to,$3,$10;
          st.relaxed.sys.global.u32 [to],$7;
          P3: setp.eq.u32 skip,$8,3;
          @skip bra W0;
          add.u64 to,$4,$10;
          st.relaxed.sys.global.u32 [to],$7;
          W0: setp.eq.u32 skip,$8,0;
          @skip bra W1;
          add.u64 from,$5,$11;
          A0: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A0;
          W1: setp.eq.u32 skip,$8,1;
          @skip bra W2;
          add.u64 from,$5,$12;
          A1: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A1;
          W2: setp.eq.u32 skip,$8,2;
          @skip bra W3;
          add.u64 from,$5,$13;
          A2: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A2;
          W3: setp.eq.u32 skip,$8,3;
          @skip bra RELEASE;
          add.u64 from,$5,$14;
          A3: ld.acquire.sys.global.u32 seen,[from]; sub.s32 diff,seen,$7; setp.lt.s32 pending,diff,0; @pending bra A3;
          RELEASE: st.release.gpu.global.u32 [rel],$7;
          bra DONE;
          WAIT_REL: ld.acquire.gpu.global.u32 seen,[rel];
          sub.s32 diff,seen,$7;
          setp.lt.s32 pending,diff,0;
          @pending bra WAIT_REL;
          DONE:
        }""",
        constraints="=r,l,l,l,l,l,r,r,r,l,l,l,l,l,l",
        args=[
            p0,
            p1,
            p2,
            p3,
            own,
            rows - 1,
            tag,
            self_id,
            tl.full((), LOCAL_READY + lane * 128 + 64, tl.uint64),
            tl.full((), region + self_id * 128 + 96, tl.uint64),
            tl.full((), region + 0 * 128 + 96, tl.uint64),
            tl.full((), region + 1 * 128 + 96, tl.uint64),
            tl.full((), region + 2 * 128 + 96, tl.uint64),
            tl.full((), region + 3 * 128 + 96, tl.uint64),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_packets_three(source, d0, d1, d2, packets: tl.constexpr):
    tl.inline_asm_elementwise(
        asm="""{
        .reg .pred done;
        .reg .u32 t,n,v0,v1,v2,v3; .reg .u64 a,b;
        mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;
        NEXT: setp.ge.u32 done,t,$5; @done bra DONE;
        mad.wide.u32 a,t,16,$1;
        ld.global.cg.v4.b32 {v0,v1,v2,v3},[a];
        mad.wide.u32 b,t,16,$2; st.global.v4.b32 [b],{v0,v1,v2,v3};
        mad.wide.u32 b,t,16,$3; st.global.v4.b32 [b],{v0,v1,v2,v3};
        mad.wide.u32 b,t,16,$4; st.global.v4.b32 [b],{v0,v1,v2,v3};
        add.u32 t,t,n; bra NEXT;
        DONE:
    }""",
        constraints="=r,l,l,l,l,r",
        args=[
            source.to(tl.uint64),
            d0.to(tl.uint64),
            d1.to(tl.uint64),
            d2.to(tl.uint64),
            packets,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _copy_packets_four(source, d0, d1, d2, d3, packets: tl.constexpr):
    tl.inline_asm_elementwise(
        asm="""{
        .reg .pred done;
        .reg .u32 t,n,v0,v1,v2,v3; .reg .u64 a,b;
        mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;
        NEXT: setp.ge.u32 done,t,$6; @done bra DONE;
        mad.wide.u32 a,t,16,$1;
        ld.global.cg.v4.b32 {v0,v1,v2,v3},[a];
        mad.wide.u32 b,t,16,$2; st.global.v4.b32 [b],{v0,v1,v2,v3};
        mad.wide.u32 b,t,16,$3; st.global.v4.b32 [b],{v0,v1,v2,v3};
        mad.wide.u32 b,t,16,$4; st.global.v4.b32 [b],{v0,v1,v2,v3};
        mad.wide.u32 b,t,16,$5; st.global.v4.b32 [b],{v0,v1,v2,v3};
        add.u32 t,t,n; bra NEXT;
        DONE:
    }""",
        constraints="=r,l,l,l,l,l,r",
        args=[
            source.to(tl.uint64),
            d0.to(tl.uint64),
            d1.to(tl.uint64),
            d2.to(tl.uint64),
            d3.to(tl.uint64),
            packets,
        ],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


def _make_sum_four_rot(island, load_mod="nc"):
    regs = ["a", "b", "c", "d"]
    load_order = [island, island ^ 1, island ^ 2, island ^ 3]
    lines = [
        "{ .reg .pred done; .reg .u32 t,n; .reg .u64 addr;",
        ".reg .b32 a<4>,b<4>,c<4>,d<4>,v<4>; .reg .b16 lo,hi; .reg .f32 x,y,sumlo,sumhi;",
        "mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;",
        "NEXT: setp.ge.u32 done,t,$9; @done bra DONE;",
    ]
    for idx in load_order:
        reg = regs[idx]
        lines.extend(
            [
                f"mad.wide.u32 addr,t,16,${idx + 1};",
                f"ld.global.{load_mod}.v4.b32 {{"
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
    for j in range(4):
        lines.append(
            f"mad.wide.u32 addr,t,16,${5 + j}; st.global.v4.b32 [addr],{{v0,v1,v2,v3}};"
        )
    lines.extend(["add.u32 t,t,n; bra NEXT; DONE: }"])
    return "\n".join(lines)


SUM4_ROT_NC_0 = tl.constexpr(_make_sum_four_rot(0, "nc"))
SUM4_ROT_NC_1 = tl.constexpr(_make_sum_four_rot(1, "nc"))
SUM4_ROT_NC_2 = tl.constexpr(_make_sum_four_rot(2, "nc"))
SUM4_ROT_NC_3 = tl.constexpr(_make_sum_four_rot(3, "nc"))


@triton.jit
def _sum_four_rot_nc(
    x0, x1, x2, x3, d0, d1, d2, d3, packets: tl.constexpr, ISLAND: tl.constexpr
):
    if ISLAND == 0:
        tl.inline_asm_elementwise(
            asm=SUM4_ROT_NC_0,
            constraints="=r,l,l,l,l,l,l,l,l,r",
            args=[x0, x1, x2, x3, d0, d1, d2, d3, packets],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    elif ISLAND == 1:
        tl.inline_asm_elementwise(
            asm=SUM4_ROT_NC_1,
            constraints="=r,l,l,l,l,l,l,l,l,r",
            args=[x0, x1, x2, x3, d0, d1, d2, d3, packets],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    elif ISLAND == 2:
        tl.inline_asm_elementwise(
            asm=SUM4_ROT_NC_2,
            constraints="=r,l,l,l,l,l,l,l,l,r",
            args=[x0, x1, x2, x3, d0, d1, d2, d3, packets],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        )
    else:
        tl.inline_asm_elementwise(
            asm=SUM4_ROT_NC_3,
            constraints="=r,l,l,l,l,l,l,l,l,r",
            args=[x0, x1, x2, x3, d0, d1, d2, d3, packets],
            dtype=tl.uint32,
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


@triton.jit
def _copy_packets_add(source, addend, destination, packets: tl.constexpr):
    tl.inline_asm_elementwise(
        asm="""{
        .reg .pred done;
        .reg .u32 t,n; .reg .u64 a,b,c;
        .reg .b32 s<4>,e<4>,v<4>; .reg .b16 lo,hi,elo,ehi; .reg .f32 x,y,ex,ey;
        mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;
        NEXT: setp.ge.u32 done,t,$4; @done bra DONE;
        mad.wide.u32 a,t,16,$1; mad.wide.u32 b,t,16,$2; mad.wide.u32 c,t,16,$3;
        ld.global.cg.v4.b32 {s0,s1,s2,s3},[a];
        ld.global.cg.v4.b32 {e0,e1,e2,e3},[b];
        mov.b32 {lo,hi},s0; mov.b32 {elo,ehi},e0; cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi; cvt.f32.bf16 ex,elo; cvt.f32.bf16 ey,ehi; add.rn.f32 x,x,ex; add.rn.f32 y,y,ey; cvt.rn.bf16x2.f32 v0,y,x;
        mov.b32 {lo,hi},s1; mov.b32 {elo,ehi},e1; cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi; cvt.f32.bf16 ex,elo; cvt.f32.bf16 ey,ehi; add.rn.f32 x,x,ex; add.rn.f32 y,y,ey; cvt.rn.bf16x2.f32 v1,y,x;
        mov.b32 {lo,hi},s2; mov.b32 {elo,ehi},e2; cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi; cvt.f32.bf16 ex,elo; cvt.f32.bf16 ey,ehi; add.rn.f32 x,x,ex; add.rn.f32 y,y,ey; cvt.rn.bf16x2.f32 v2,y,x;
        mov.b32 {lo,hi},s3; mov.b32 {elo,ehi},e3; cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi; cvt.f32.bf16 ex,elo; cvt.f32.bf16 ey,ehi; add.rn.f32 x,x,ex; add.rn.f32 y,y,ey; cvt.rn.bf16x2.f32 v3,y,x;
        st.global.v4.b32 [c],{v0,v1,v2,v3};
        add.u32 t,t,n; bra NEXT;
        DONE:
    }""",
        constraints="=r,l,l,l,r",
        args=[source.to(tl.uint64), addend.to(tl.uint64), destination.to(tl.uint64), packets],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


def _sum_packets_add_lane(add_lane, destinations=1):
    # Exact BF16 rounding of (source + addend) on `add_lane`, then FP32 reduction across lanes 0..3.
    lines = [
        "{ .reg .pred done; .reg .u32 t,n; .reg .u64 addr;",
        ".reg .b32 a<4>,b<4>,c<4>,d<4>,e<4>,v<4>; .reg .b16 lo,hi,elo,ehi; .reg .f32 x,y,ex,ey,sumlo,sumhi;",
        "mov.u32 $0,0; mov.u32 t,%tid.x; mov.u32 n,%ntid.x;",
        f"NEXT: setp.ge.u32 done,t,${6 + destinations}; @done bra DONE;",
    ]
    for i, reg in enumerate("abcde"):
        lines.extend(
            [
                f"mad.wide.u32 addr,t,16,${i + 1};",
                "ld.global.cg.v4.b32 {"
                + ",".join(f"{reg}{j}" for j in range(4))
                + "},[addr];",
            ]
        )
    target_reg = "abcd"[add_lane]
    for j in range(4):
        lines.extend(
            [
                f"mov.b32 {{lo,hi}},{target_reg}{j}; mov.b32 {{elo,ehi}},e{j};",
                "cvt.f32.bf16 x,lo; cvt.f32.bf16 y,hi; cvt.f32.bf16 ex,elo; cvt.f32.bf16 ey,ehi;",
                "add.rn.f32 x,x,ex; add.rn.f32 y,y,ey;",
                f"cvt.rn.bf16x2.f32 {target_reg}{j},y,x;",
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
            f"mad.wide.u32 addr,t,16,${6 + j}; st.global.v4.b32 [addr],{{v0,v1,v2,v3}};"
        )
    lines.extend(["add.u32 t,t,n; bra NEXT; DONE: }"])
    return "\n".join(lines)


SUM_ONE_ADD_0 = tl.constexpr(_sum_packets_add_lane(0, 1))
SUM_ONE_ADD_1 = tl.constexpr(_sum_packets_add_lane(1, 1))
SUM_ONE_ADD_2 = tl.constexpr(_sum_packets_add_lane(2, 1))
SUM_ONE_ADD_3 = tl.constexpr(_sum_packets_add_lane(3, 1))
SUM_FOUR_ADD_0 = tl.constexpr(_sum_packets_add_lane(0, 4))
SUM_FOUR_ADD_1 = tl.constexpr(_sum_packets_add_lane(1, 4))
SUM_FOUR_ADD_2 = tl.constexpr(_sum_packets_add_lane(2, 4))
SUM_FOUR_ADD_3 = tl.constexpr(_sum_packets_add_lane(3, 4))


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
    LANE: tl.constexpr,
    ISLAND: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr = 4,
    WRITE_OUTPUT: tl.constexpr = True,
    Addend=None,
    HAS_ADDEND: tl.constexpr = False,
    SCATTER_CROSS: tl.constexpr = False,
    PAIRWISE: tl.constexpr = True,
    PROTOCOL: tl.constexpr = 0,
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
    fo = final + parity.to(tl.int64) * capacity * 16
    if row < ROWS:
        gen = tl.atomic_add(counter + PHYSICAL_COUNTERS + row, 1, sem="relaxed") + 1
        quarter: tl.constexpr = WIDTH // 4
        qbytes: tl.constexpr = 2 * quarter
        packets: tl.constexpr = quarter // 8
        so = stage + parity.to(tl.int64) * capacity * 16
        po = part + parity.to(tl.int64) * capacity * 4
        source = Input.to(tl.uint64)
        output = Output.to(tl.uint64)
        if HAS_ADDEND:
            add_ptr = Addend.to(tl.uint64)
        if PROTOCOL >= 1:
            # Wave 11 Switch-4 protocols (all bitwise-exact FP32->BF16 4-lane + 4-island):
            # PROTOCOL 4: Per-CTA 3-barrier XOR-rotated Cross-Pull (ld.global.nc) + direct SUM4 (<= 72 KiB)
            # PROTOCOL 1: Grid 4-barrier P1 (2-step strict 1-to-1) + single-fence fast sync + direct SUM4 (<= 230 KiB)
            # PROTOCOL 2: Grid 4-barrier Cross-island Scatter-PULL + Gather-PULL
            # PROTOCOL 3: Grid 6-barrier P3 (4-step scattered 1-to-1) + single-fence fast sync + SUM4 + COPY4 + COPY3 (> 230 KiB)
            lp1 = _ptr(L0, L1, L2, L3, LANE ^ 1)
            lp2 = _ptr(L0, L1, L2, L3, LANE ^ 2)
            lp3 = _ptr(L0, L1, L2, L3, LANE ^ 3)
            USE_GRID: tl.constexpr = PROTOCOL != 4
            if USE_GRID:
                epoch = _load_grid_epoch(own, LANE)
            # Phase 1: Intra-switch 1-barrier Switch-4 scatter (XOR peer order, 49.6 GB/s switch)
            for step in tl.static_range(1, 4):
                dest = LANE ^ step
                dest_ptr = _ptr(L0, L1, L2, L3, dest)
                if HAS_ADDEND:
                    _copy_packets_add(
                        source + (4 * row + dest) * qbytes,
                        add_ptr + (4 * row + dest) * qbytes,
                        dest_ptr + so + (ROWS * LANE + row) * qbytes,
                        packets,
                    )
                else:
                    _copy_packets(
                        source + (4 * row + dest) * qbytes,
                        dest_ptr + so + (ROWS * LANE + row) * qbytes,
                        packets,
                    )
            tl.debug_barrier()
            if USE_GRID:
                _grid_peers_sync_fast(L0, L1, L2, L3, own, ROWS, epoch + 1, LANE, LOCAL_READY, LANE)
            else:
                _signal_peers_nosleep(L0, L1, L2, L3, own, row, gen, LANE, LOCAL_READY)
            tl.debug_barrier()
            # Exact local 4-lane FP32->BF16 reduction into own + po
            local_values = source + (4 * row + LANE) * qbytes
            a = local_values if LANE == 0 else own + so + (0 * ROWS + row) * qbytes
            b = local_values if LANE == 1 else own + so + (1 * ROWS + row) * qbytes
            c = local_values if LANE == 2 else own + so + (2 * ROWS + row) * qbytes
            d = local_values if LANE == 3 else own + so + (3 * ROWS + row) * qbytes
            if HAS_ADDEND:
                local_add = add_ptr + (4 * row + LANE) * qbytes
                if LANE == 0:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_0,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 1:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_1,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 2:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_2,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                else:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_3,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
            else:
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[a, b, c, d, own + po + row * qbytes, packets],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
            tl.debug_barrier()
            col_me = (4 * row + LANE) * qbytes
            if PROTOCOL == 4:
                # Per-CTA 3-barrier XOR-rotated Cross-Pull (ld.global.nc) + direct SUM4 (own, lp1, lp2, lp3)
                _signal_peers_nosleep(X0, X1, X2, X3, own, row, gen * 2, ISLAND, CROSS_READY)
                tl.debug_barrier()
                _sum_four_rot_nc(
                    X0 + po + row * qbytes,
                    X1 + po + row * qbytes,
                    X2 + po + row * qbytes,
                    X3 + po + row * qbytes,
                    own + fo + col_me,
                    lp1 + fo + col_me,
                    lp2 + fo + col_me,
                    lp3 + fo + col_me,
                    packets,
                    ISLAND,
                )
                tl.debug_barrier()
                _signal_peers_nosleep(L0, L1, L2, L3, own, row, gen, LANE, FINAL_READY)
                tl.debug_barrier()
            elif PROTOCOL == 1:
                # 2-step grid-synced strict 1-to-1 pairwise cross-island push + direct SUM4 (4 barriers total)
                x1_id: tl.constexpr = ISLAND ^ 1
                x2_id: tl.constexpr = ISLAND ^ 2
                xp1 = _ptr(X0, X1, X2, X3, x1_id)
                xp2 = _ptr(X0, X1, X2, X3, x2_id)
                copy_base = ((final + capacity.to(tl.int64) * 32 + 255) // 256) * 256
                xo = copy_base + 256 + parity.to(tl.int64) * capacity.to(tl.int64) * 12
                s_me_in_x1: tl.constexpr = ISLAND if ISLAND < x1_id else ISLAND - 1
                s_x1_in_me: tl.constexpr = x1_id if x1_id < ISLAND else x1_id - 1
                s_me_in_x2: tl.constexpr = ISLAND if ISLAND < x2_id else ISLAND - 1
                s_x1_in_x2: tl.constexpr = x1_id if x1_id < x2_id else x1_id - 1
                _copy_packets(
                    own + po + row * qbytes,
                    xp1 + xo + (s_me_in_x1 * ROWS + row) * qbytes,
                    packets,
                )
                tl.debug_barrier()
                _grid_pair_sync_fast(xp1, own, ROWS, epoch + 2, ISLAND, x1_id, CROSS_READY, LANE)
                tl.debug_barrier()
                _copy_packets(
                    own + po + row * qbytes,
                    xp2 + xo + (s_me_in_x2 * ROWS + row) * qbytes,
                    packets,
                )
                _copy_packets(
                    own + xo + (s_x1_in_me * ROWS + row) * qbytes,
                    xp2 + xo + (s_x1_in_x2 * ROWS + row) * qbytes,
                    packets,
                )
                tl.debug_barrier()
                _grid_pair_sync_fast(xp2, own, ROWS, epoch + 3, ISLAND, x2_id, CROSS_READY, LANE)
                tl.debug_barrier()
                x0 = own + po + row * qbytes if ISLAND == 0 else own + xo + (0 * ROWS + row) * qbytes
                x1 = own + po + row * qbytes if ISLAND == 1 else own + xo + ((1 if 1 < ISLAND else 0) * ROWS + row) * qbytes
                x2 = own + po + row * qbytes if ISLAND == 2 else own + xo + ((2 if 2 < ISLAND else 1) * ROWS + row) * qbytes
                x3 = own + po + row * qbytes if ISLAND == 3 else own + xo + (2 * ROWS + row) * qbytes
                tl.inline_asm_elementwise(
                    asm=SUM_FOUR,
                    constraints="=r,l,l,l,l,l,l,l,l,r",
                    args=[
                        x0,
                        x1,
                        x2,
                        x3,
                        own + fo + col_me,
                        lp1 + fo + col_me,
                        lp2 + fo + col_me,
                        lp3 + fo + col_me,
                        packets,
                    ],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                _grid_peers_sync_fast(L0, L1, L2, L3, own, ROWS, epoch + 4, LANE, FINAL_READY, LANE)
                tl.debug_barrier()
            elif PROTOCOL == 2:
                # Hybrid: Cross-island Scatter-PULL + Gather-PULL (4 barriers total)
                sbytes: tl.constexpr = qbytes // 4
                spackets: tl.constexpr = packets // 4
                _grid_peers_sync_fast(X0, X1, X2, X3, own, ROWS, epoch + 2, ISLAND, CROSS_READY, LANE)
                tl.debug_barrier()
                slice_off = row * qbytes + ISLAND * sbytes
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[
                        X0 + po + slice_off,
                        X1 + po + slice_off,
                        X2 + po + slice_off,
                        X3 + po + slice_off,
                        own + fo + col_me + ISLAND * sbytes,
                        spackets,
                    ],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                _grid_peers_sync_fast(X0, X1, X2, X3, own, ROWS, epoch + 3, ISLAND, CROSS_READY, LANE)
                tl.debug_barrier()
                for j in tl.static_range(4):
                    if j != ISLAND:
                        xj = _ptr(X0, X1, X2, X3, j)
                        _copy_packets(
                            xj + fo + col_me + j * sbytes,
                            own + fo + col_me + j * sbytes,
                            spackets,
                        )
                tl.debug_barrier()
                _copy_packets_three(
                    own + fo + col_me,
                    lp1 + fo + col_me,
                    lp2 + fo + col_me,
                    lp3 + fo + col_me,
                    packets,
                )
                tl.debug_barrier()
                _grid_peers_sync_fast(L0, L1, L2, L3, own, ROWS, epoch + 4, LANE, FINAL_READY, LANE)
                tl.debug_barrier()
            else:
                # PROTOCOL == 3: 4-step scattered strict 1-to-1 + fast sync + SUM4 + COPY4 + COPY3 (6 barriers)
                sbytes: tl.constexpr = qbytes // 4
                spackets: tl.constexpr = packets // 4
                x1_id: tl.constexpr = ISLAND ^ 1
                x2_id: tl.constexpr = ISLAND ^ 2
                x3_id: tl.constexpr = ISLAND ^ 3
                xp1 = _ptr(X0, X1, X2, X3, x1_id)
                xp2 = _ptr(X0, X1, X2, X3, x2_id)
                _copy_packets(
                    own + po + row * qbytes + x1_id * sbytes,
                    xp1 + so + (ROWS * LANE + row) * qbytes + ISLAND * sbytes,
                    spackets,
                )
                _copy_packets(
                    own + po + row * qbytes + x3_id * sbytes,
                    xp1 + fo + col_me + x3_id * sbytes,
                    spackets,
                )
                tl.debug_barrier()
                _grid_pair_sync_fast(xp1, own, ROWS, epoch + 2, ISLAND, x1_id, CROSS_READY, LANE)
                tl.debug_barrier()
                _copy_packets(
                    own + po + row * qbytes + x2_id * sbytes,
                    xp2 + so + (ROWS * LANE + row) * qbytes + ISLAND * sbytes,
                    spackets,
                )
                _copy_packets(
                    own + fo + col_me + x2_id * sbytes,
                    xp2 + so + (ROWS * LANE + row) * qbytes + x1_id * sbytes,
                    spackets,
                )
                tl.debug_barrier()
                _grid_pair_sync_fast(xp2, own, ROWS, epoch + 3, ISLAND, x2_id, CROSS_READY, LANE)
                tl.debug_barrier()
                xbase = own + so + (ROWS * LANE + row) * qbytes
                own_slice = own + po + row * qbytes + ISLAND * sbytes
                xa = own_slice if ISLAND == 0 else xbase + 0 * sbytes
                xb = own_slice if ISLAND == 1 else xbase + 1 * sbytes
                xc = own_slice if ISLAND == 2 else xbase + 2 * sbytes
                xd = own_slice if ISLAND == 3 else xbase + 3 * sbytes
                dst_off = col_me + ISLAND * sbytes
                tl.inline_asm_elementwise(
                    asm=SUM_FOUR,
                    constraints="=r,l,l,l,l,l,l,l,l,r",
                    args=[
                        xa,
                        xb,
                        xc,
                        xd,
                        own + fo + dst_off,
                        lp1 + fo + dst_off,
                        lp2 + fo + dst_off,
                        lp3 + fo + dst_off,
                        spackets,
                    ],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                _copy_packets(own + fo + dst_off, xp2 + fo + dst_off, spackets)
                tl.debug_barrier()
                _grid_pair_sync_fast(xp2, own, ROWS, epoch + 4, ISLAND, x2_id, CROSS_READY, LANE)
                tl.debug_barrier()
                _copy_packets(own + fo + dst_off, xp1 + fo + dst_off, spackets)
                _copy_packets_four(
                    own + fo + col_me + x2_id * sbytes,
                    xp1 + fo + col_me + x2_id * sbytes,
                    lp1 + fo + col_me + x2_id * sbytes,
                    lp2 + fo + col_me + x2_id * sbytes,
                    lp3 + fo + col_me + x2_id * sbytes,
                    spackets,
                )
                tl.debug_barrier()
                _grid_pair_sync_fast(xp1, own, ROWS, epoch + 5, ISLAND, x1_id, CROSS_READY, LANE)
                tl.debug_barrier()
                _copy_packets_three(
                    own + fo + col_me + x1_id * sbytes,
                    lp1 + fo + col_me + x1_id * sbytes,
                    lp2 + fo + col_me + x1_id * sbytes,
                    lp3 + fo + col_me + x1_id * sbytes,
                    spackets,
                )
                _copy_packets_three(
                    own + fo + col_me + x3_id * sbytes,
                    lp1 + fo + col_me + x3_id * sbytes,
                    lp2 + fo + col_me + x3_id * sbytes,
                    lp3 + fo + col_me + x3_id * sbytes,
                    spackets,
                )
                tl.debug_barrier()
                _grid_peers_sync_fast(L0, L1, L2, L3, own, ROWS, epoch + 6, LANE, FINAL_READY, LANE)
                tl.debug_barrier()
            if WRITE_OUTPUT:
                for q in tl.static_range(4):
                    col_q = (4 * row + q) * qbytes
                    _copy_packets(own + fo + col_q, output + col_q, packets)
                tl.debug_barrier()
        elif PAIRWISE:
            # Strict 1-to-1 pairwise protocol (6 single-peer barriers total):
            # Step 1a: r^1 (PIX)  Step 1b: r^2 -> exact local 4-lane BF16 sum
            # Step 2a: r^4        Step 2b: r^8 -> exact cross 4-island BF16 sum
            # Step 3a: r^2        Step 3b: r^1 (PIX) -> all 4 final quarters in fo
            p1: tl.constexpr = LANE ^ 1
            p2: tl.constexpr = LANE ^ 2
            p3: tl.constexpr = LANE ^ 3
            lp1 = _ptr(L0, L1, L2, L3, p1)
            lp2 = _ptr(L0, L1, L2, L3, p2)
            # Step 1a: push quarters p1 and p3 to p1 (PIX switch!)
            if HAS_ADDEND:
                _copy_packets_add(
                    source + (4 * row + p1) * qbytes,
                    add_ptr + (4 * row + p1) * qbytes,
                    lp1 + so + (ROWS * LANE + row) * qbytes,
                    packets,
                )
                _copy_packets_add(
                    source + (4 * row + p3) * qbytes,
                    add_ptr + (4 * row + p3) * qbytes,
                    lp1 + fo + (4 * row + p3) * qbytes,
                    packets,
                )
            else:
                _copy_packets(
                    source + (4 * row + p1) * qbytes,
                    lp1 + so + (ROWS * LANE + row) * qbytes,
                    packets,
                )
                _copy_packets(
                    source + (4 * row + p3) * qbytes,
                    lp1 + fo + (4 * row + p3) * qbytes,
                    packets,
                )
            tl.debug_barrier()
            _signal_one_peer(lp1, own, row, gen, LANE, p1, LOCAL_READY)
            tl.debug_barrier()
            # Step 1b: push own and p1's contributions for quarter p2 to p2
            if HAS_ADDEND:
                _copy_packets_add(
                    source + (4 * row + p2) * qbytes,
                    add_ptr + (4 * row + p2) * qbytes,
                    lp2 + so + (ROWS * LANE + row) * qbytes,
                    packets,
                )
            else:
                _copy_packets(
                    source + (4 * row + p2) * qbytes,
                    lp2 + so + (ROWS * LANE + row) * qbytes,
                    packets,
                )
            _copy_packets(
                own + fo + (4 * row + p2) * qbytes,
                lp2 + so + (ROWS * p1 + row) * qbytes,
                packets,
            )
            tl.debug_barrier()
            _signal_one_peer(lp2, own, row, gen, LANE, p2, LOCAL_READY)
            tl.debug_barrier()
            # Exact local 4-lane FP32->BF16 reduction into own + po
            local_values = source + (4 * row + LANE) * qbytes
            a = local_values if LANE == 0 else own + so + (0 * ROWS + row) * qbytes
            b = local_values if LANE == 1 else own + so + (1 * ROWS + row) * qbytes
            c = local_values if LANE == 2 else own + so + (2 * ROWS + row) * qbytes
            d = local_values if LANE == 3 else own + so + (3 * ROWS + row) * qbytes
            if HAS_ADDEND:
                local_add = add_ptr + (4 * row + LANE) * qbytes
                if LANE == 0:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_0,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 1:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_1,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 2:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_2,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                else:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_3,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
            else:
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[a, b, c, d, own + po + row * qbytes, packets],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
            tl.debug_barrier()
            # Step 2: Cross-island strict 1-to-1 pairwise exchange (x1 = ISLAND^1, x2 = ISLAND^2)
            x1_id: tl.constexpr = ISLAND ^ 1
            x2_id: tl.constexpr = ISLAND ^ 2
            xp1 = _ptr(X0, X1, X2, X3, x1_id)
            xp2 = _ptr(X0, X1, X2, X3, x2_id)
            copy_base = ((final + capacity.to(tl.int64) * 32 + 255) // 256) * 256
            xo = copy_base + 256 + parity.to(tl.int64) * capacity.to(tl.int64) * 12
            s_me_in_x1: tl.constexpr = ISLAND if ISLAND < x1_id else ISLAND - 1
            s_x1_in_me: tl.constexpr = x1_id if x1_id < ISLAND else x1_id - 1
            s_me_in_x2: tl.constexpr = ISLAND if ISLAND < x2_id else ISLAND - 1
            s_x1_in_x2: tl.constexpr = x1_id if x1_id < x2_id else x1_id - 1
            # Step 2a: exchange po with x1
            _copy_packets(
                own + po + row * qbytes,
                xp1 + xo + (s_me_in_x1 * ROWS + row) * qbytes,
                packets,
            )
            tl.debug_barrier()
            _signal_one_peer(xp1, own, row, gen * 2, ISLAND, x1_id, CROSS_READY)
            tl.debug_barrier()
            # Step 2b: forward own po + x1's po to x2
            _copy_packets(
                own + po + row * qbytes,
                xp2 + xo + (s_me_in_x2 * ROWS + row) * qbytes,
                packets,
            )
            _copy_packets(
                own + xo + (s_x1_in_me * ROWS + row) * qbytes,
                xp2 + xo + (s_x1_in_x2 * ROWS + row) * qbytes,
                packets,
            )
            tl.debug_barrier()
            _signal_one_peer(xp2, own, row, gen * 2, ISLAND, x2_id, CROSS_READY)
            tl.debug_barrier()
            # Exact cross-island 4-input FP32->BF16 reduction into own + fo[LANE]
            x0 = (
                own + po + row * qbytes
                if ISLAND == 0
                else own + xo + (0 * ROWS + row) * qbytes
            )
            x1 = (
                own + po + row * qbytes
                if ISLAND == 1
                else own + xo + ((1 if 1 < ISLAND else 0) * ROWS + row) * qbytes
            )
            x2 = (
                own + po + row * qbytes
                if ISLAND == 2
                else own + xo + ((2 if 2 < ISLAND else 1) * ROWS + row) * qbytes
            )
            x3 = (
                own + po + row * qbytes
                if ISLAND == 3
                else own + xo + (2 * ROWS + row) * qbytes
            )
            col_me = (4 * row + LANE) * qbytes
            col_p2 = (4 * row + p2) * qbytes
            tl.inline_asm_elementwise(
                asm=SUM_ONE,
                constraints="=r,l,l,l,l,l,r",
                args=[x0, x1, x2, x3, own + fo + col_me, packets],
                dtype=tl.uint32,
                is_pure=False,
                pack=1,
            )
            tl.debug_barrier()
            # Step 3a: push own completed quarter to p2 (strict 1-to-1)
            _copy_packets(own + fo + col_me, lp2 + fo + col_me, packets)
            tl.debug_barrier()
            _signal_one_peer(lp2, own, row, gen, LANE, p2, FINAL_READY)
            tl.debug_barrier()
            # Step 3b: push own and p2's completed quarters to p1 (PIX switch!)
            _copy_packets(own + fo + col_me, lp1 + fo + col_me, packets)
            _copy_packets(own + fo + col_p2, lp1 + fo + col_p2, packets)
            tl.debug_barrier()
            _signal_one_peer(lp1, own, row, gen, LANE, p1, FINAL_READY)
            tl.debug_barrier()
            if WRITE_OUTPUT:
                for q in tl.static_range(4):
                    col_q = (4 * row + q) * qbytes
                    _copy_packets(own + fo + col_q, output + col_q, packets)
                tl.debug_barrier()
        else:
            for step in tl.static_range(1, 4):
                dest = (LANE + step) % 4
                dest_ptr = _ptr(L0, L1, L2, L3, dest)
                if HAS_ADDEND:
                    _copy_packets_add(
                        source + (4 * row + dest) * qbytes,
                        add_ptr + (4 * row + dest) * qbytes,
                        dest_ptr + so + (ROWS * LANE + row) * qbytes,
                        packets,
                    )
                else:
                    _copy_packets(
                        source + (4 * row + dest) * qbytes,
                        dest_ptr + so + (ROWS * LANE + row) * qbytes,
                        packets,
                    )
            tl.debug_barrier()
            _signal_peers(L0, L1, L2, L3, own, row, gen, LANE, LOCAL_READY)
            tl.debug_barrier()
            local_values = source + (4 * row + LANE) * qbytes
            a = local_values if LANE == 0 else own + so + (0 * ROWS + row) * qbytes
            b = local_values if LANE == 1 else own + so + (1 * ROWS + row) * qbytes
            c = local_values if LANE == 2 else own + so + (2 * ROWS + row) * qbytes
            d = local_values if LANE == 3 else own + so + (3 * ROWS + row) * qbytes
            if HAS_ADDEND:
                local_add = add_ptr + (4 * row + LANE) * qbytes
                if LANE == 0:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_0,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 1:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_1,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                elif LANE == 2:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_2,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
                else:
                    tl.inline_asm_elementwise(
                        asm=SUM_ONE_ADD_3,
                        constraints="=r,l,l,l,l,l,l,r",
                        args=[a, b, c, d, local_add, own + po + row * qbytes, packets],
                        dtype=tl.uint32,
                        is_pure=False,
                        pack=1,
                    )
            else:
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[a, b, c, d, own + po + row * qbytes, packets],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
            tl.debug_barrier()
            if SCATTER_CROSS:
                sbytes: tl.constexpr = qbytes // 4
                spackets: tl.constexpr = packets // 4
                for step in tl.static_range(1, 4):
                    dst_island = (ISLAND + step) % 4
                    dst_x = _ptr(X0, X1, X2, X3, dst_island)
                    _copy_packets(
                        own + po + row * qbytes + dst_island * sbytes,
                        dst_x + so + (ROWS * LANE + row) * qbytes + ISLAND * sbytes,
                        spackets,
                    )
                tl.debug_barrier()
                _signal_peers(X0, X1, X2, X3, own, row, gen * 2, ISLAND, CROSS_READY)
                tl.debug_barrier()
                xbase = own + so + (ROWS * LANE + row) * qbytes
                own_slice = own + po + row * qbytes + ISLAND * sbytes
                xa = own_slice if ISLAND == 0 else xbase + 0 * sbytes
                xb = own_slice if ISLAND == 1 else xbase + 1 * sbytes
                xc = own_slice if ISLAND == 2 else xbase + 2 * sbytes
                xd = own_slice if ISLAND == 3 else xbase + 3 * sbytes
                col_slice = (4 * row + LANE) * qbytes + ISLAND * sbytes
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[xa, xb, xc, xd, own + fo + col_slice, spackets],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                for step in tl.static_range(1, 4):
                    dst_island = (ISLAND + step) % 4
                    dst_x = _ptr(X0, X1, X2, X3, dst_island)
                    _copy_packets(
                        own + fo + col_slice,
                        dst_x + fo + col_slice,
                        spackets,
                    )
                tl.debug_barrier()
                _signal_peers(
                    X0, X1, X2, X3, own, row, gen * 2 + 1, ISLAND, CROSS_READY
                )
                tl.debug_barrier()
                col = (4 * row + LANE) * qbytes
                for step in tl.static_range(1, 4):
                    dest = (LANE + step) % 4
                    dest_ptr = _ptr(L0, L1, L2, L3, dest)
                    _copy_packets(own + fo + col, dest_ptr + fo + col, packets)
                if WRITE_OUTPUT:
                    _copy_packets(own + fo + col, output + col, packets)
                tl.debug_barrier()
                _signal_peers(L0, L1, L2, L3, own, row, gen, LANE, FINAL_READY)
                tl.debug_barrier()
            else:
                copy_base = ((final + capacity.to(tl.int64) * 32 + 255) // 256) * 256
                xo = copy_base + 256 + parity.to(tl.int64) * capacity.to(tl.int64) * 12
                for step in tl.static_range(1, 4):
                    dst_island = (ISLAND + step) % 4
                    dst_slot = ISLAND if ISLAND < dst_island else ISLAND - 1
                    dst_x = _ptr(X0, X1, X2, X3, dst_island)
                    _copy_packets(
                        own + po + row * qbytes,
                        dst_x + xo + (dst_slot * ROWS + row) * qbytes,
                        packets,
                    )
                tl.debug_barrier()
                _signal_peers(X0, X1, X2, X3, own, row, gen * 2, ISLAND, CROSS_READY)
                tl.debug_barrier()
                x0 = (
                    own + po + row * qbytes
                    if ISLAND == 0
                    else own + xo + (0 * ROWS + row) * qbytes
                )
                x1 = (
                    own + po + row * qbytes
                    if ISLAND == 1
                    else own + xo + ((1 if 1 < ISLAND else 0) * ROWS + row) * qbytes
                )
                x2 = (
                    own + po + row * qbytes
                    if ISLAND == 2
                    else own + xo + ((2 if 2 < ISLAND else 1) * ROWS + row) * qbytes
                )
                x3 = (
                    own + po + row * qbytes
                    if ISLAND == 3
                    else own + xo + (2 * ROWS + row) * qbytes
                )
                col = (4 * row + LANE) * qbytes
                local_dst = (output + col) if WRITE_OUTPUT else (own + fo + col)
                tl.inline_asm_elementwise(
                    asm=SUM_ONE,
                    constraints="=r,l,l,l,l,l,r",
                    args=[x0, x1, x2, x3, local_dst, packets],
                    dtype=tl.uint32,
                    is_pure=False,
                    pack=1,
                )
                tl.debug_barrier()
                for step in tl.static_range(1, 4):
                    dest = (LANE + step) % 4
                    dest_ptr = _ptr(L0, L1, L2, L3, dest)
                    _copy_packets(local_dst, dest_ptr + fo + col, packets)
                tl.debug_barrier()
                _signal_peers(L0, L1, L2, L3, own, row, gen, LANE, FINAL_READY)
                tl.debug_barrier()
            if WRITE_OUTPUT:
                for sender in tl.static_range(4):
                    if sender != LANE:
                        col = (4 * row + sender) * qbytes
                        _copy_packets(own + fo + col, output + col, packets)
                tl.debug_barrier()
    if WRITE_OUTPUT:
        return Output
    return (own + fo).to(tl.pointer_type(tl.bfloat16))


def select_row_protocol(rows: int, width: int) -> int:
    """Select the fastest bitwise-exact TP16 Switch-4 reduction protocol for (rows, width)."""
    elements = int(rows) * int(width)
    if elements <= 36_000:
        # <= 72 KiB (e.g. 1..5x7168, 1..10x3584): per-CTA 3-barrier XOR Cross-Pull + direct SUM4.
        return 4
    if elements <= 115_000:
        # 72..230 KiB (e.g. 6..16x7168, 11..32x3584): grid 4-barrier P1 + fast sync + direct SUM4.
        return 1
    # > 230 KiB (e.g. 17..32x7168): grid 6-barrier P3 + fast sync + direct SUM4 + COPY4 + COPY3.
    return 3


@dataclass(frozen=True)
class RowReduction:
    """Borrow the six remote mappings and the local slab from a prepared reducer."""

    owner: object
    width: int
    rows: int = 4

    @property
    def grid(self):
        return (self.owner.blocks,)

    @property
    def rank_coordinates(self):
        return self.owner.rank % 4, self.owner.rank // 4

    def arguments(self, source, destination):
        from .pcie_island_rs import _validate_output_overlap

        owner = self.owner
        if not owner.should_allreduce(source) or source.shape != (self.rows, self.width):
            raise ValueError(
                f"PCIe reduction source must be {self.rows} contiguous BF16 rows on its owner"
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


@dataclass(frozen=True)
class TP8RoRowReduction:
    """Borrow the existing PCIeOneshotAllReduce IPC slabs for fused TP8 RO epilogues (0 extra VRAM)."""

    owner: object
    width: int
    rows: int
    transport: str
    plain_region_packs: int
    plain_capacity_packs: int
    ar_blocks: int
    ar_threads: int
    is_tp8_ro: bool = True

    def resolve_slot(self):
        state = self.owner._ext._state(self.owner._ptr)
        capturing = torch.cuda.is_current_stream_capturing()
        if state.device_slot_selection or capturing:
            if not state.device_slot_selection:
                state.slot_bias = state.eager_slot & 1
                state.device_slot_selection = True
            return True, int(state.slot_bias) & 1, 0, int(state.eager_tables[0]), int(state.signal_table_address)
        slot = int(state.eager_slot) & 1
        state.eager_slot ^= 1
        return False, 0, slot, int(state.eager_tables[0]), int(state.signal_table_address)


def prepared_rows(plan, *, rows, width, threads):
    """Return the native multi-row layout if the selected owner and explicit grid permit it."""
    from .pcie_island_rs import PCIeIslandRSAllReduce
    from .pcie_oneshot import PCIeOneshotAllReduce
    from ._tuning import pairwise_device

    if plan.prepared is None or not 1 <= rows <= 64 or width not in (3584, 7168):
        return None
    if not pairwise_device(plan.prepared.device.identity):
        return None
    state = plan.prepared.state
    owner = getattr(state, "runtime", None)
    if isinstance(owner, PCIeOneshotAllReduce):
        if owner.world_size != 8 or not 1 <= rows <= 32:
            return None
        transport = state.query.call.get("transport")
        if transport not in ("remote_push_full", "remote_push_rsag"):
            return None
        setup = state.query.setup
        if (
            setup.get("dtype") != "bfloat16"
            or tuple(setup.get("shape", ())) != (rows, width)
            or tuple(setup.get("stride", ())) != (width, 1)
        ):
            return None
        native = owner._ext._state(owner._ptr)
        if native.eager_tables is None or not native.eager_buffer_bytes:
            return None
        plain_capacity_packs = int(native.eager_buffer_bytes) // 16
        plain_region_packs = int(
            state.query.call.get("payload_ready_region_packs", 0)
            or native.plain_payload_ready_region_packs
        )
        if plain_region_packs == 0:
            base_shards = 8 if native.sharded_eager_storage else 1
            tagged = 2 * 7 if native.transport_policy[2] else 0
            plain_region_packs = (base_shards + tagged) * plain_capacity_packs
        ar_blocks = int(state.query.call["blocks"])
        ar_threads = int(state.query.call["threads"])
        return TP8RoRowReduction(
            owner=owner,
            width=width,
            rows=rows,
            transport=transport,
            plain_region_packs=plain_region_packs,
            plain_capacity_packs=plain_capacity_packs,
            ar_blocks=ar_blocks,
            ar_threads=ar_threads,
        )
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
        or owner.quarter_capacity * 4 < rows * (width // 4) * 2
        or owner._group_timing_records is not None
        or (state.query.setup.get("threads_override") and owner.threads != threads)
        or (
            state.query.setup.get("active_blocks_override")
            and state.query.call["active_blocks"] != rows
        )
    ):
        return None
    return RowReduction(owner, width, rows)
