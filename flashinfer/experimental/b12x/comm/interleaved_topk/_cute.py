# SPDX-License-Identifier: Apache-2.0
"""Pure CuTeDSL kernel for distributed rank-major stable top-2048 slot selection.

Selects the local attention records consumed by one interleaved DCP shard across
World in {4, 8, 16} CUDA IPC peers for 1..64 query rows.  Each peer contributes
up to 2048 (FP32 score bits, shard-local token index) pairs; padding may appear
anywhere, and a valid token's rank-major candidate ordinal breaks ties.
"""

from __future__ import annotations

from collections.abc import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

K = 2048
BLOCK_ITEMS = 32
SAMPLES = K // BLOCK_ITEMS  # 64
STAGE_ITEMS = K + SAMPLES  # 2112
THREADS = 256
MAX_ROWS = 64
SUPPORTED_WORLDS = (4, 8, 16)
_PAD_KEY = 0x007FFFFF  # ordered(-infinity)
_WRAP_MASK = 31  # 32 generations x 64 rows = 2048 signal records
_WRAP_SHIFT = 5
_GEN_ROLLOVER = 0xFFFFFF80  # divisible by 32 and 64
_SPEC_VERSION = 1


# ---------------------------------------------------------------------------
# PTX intrinsics
# ---------------------------------------------------------------------------


@dsl_user_op
def _ordered_key(raw: Uint32, *, loc=None, ip=None) -> Uint32:
    """Map raw IEEE-754 float32 bits onto an unsigned order-preserving u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(raw).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 sbits;
                .reg .u32 mask;
                shr.s32 sbits, $1, 31;
                or.b32 mask, sbits, 0x80000000;
                xor.b32 $0, $1, mask;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _not_u32(val: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(val).ir_value(loc=loc, ip=ip)],
            "not.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _not_u64(val: Uint64, *, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Uint64(val).ir_value(loc=loc, ip=ip)],
            "not.b64 $0, $1;",
            "=l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_u64(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                Uint32(lo).ir_value(loc=loc, ip=ip),
                Uint32(hi).ir_value(loc=loc, ip=ip),
            ],
            "mov.b64 $0, {$1, $2};",
            "=l,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unpack_u64_lo(val: Uint64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint64(val).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 lo, hi; mov.b64 {lo, hi}, $1; mov.b32 $0, lo; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _unpack_u64_hi(val: Uint64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint64(val).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 lo, hi; mov.b64 {lo, hi}, $1; mov.b32 $0, hi; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nibble_u64(val: Uint64, bit_pos: Int32, *, loc=None, ip=None) -> Int32:
    """Extract the 4-bit unsigned digit at ``bit_pos`` from a 64-bit value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint64(val).ir_value(loc=loc, ip=ip),
                Int32(bit_pos).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .u64 shifted;
                .reg .u32 lo;
                shr.u64 shifted, $1, $2;
                cvt.u32.u64 lo, shifted;
                and.b32 $0, lo, 15;
            }
            """,
            "=r,l,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_i32(val: Int32, src_lane: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(val).ir_value(loc=loc, ip=ip),
                Int32(src_lane).ir_value(loc=loc, ip=ip),
            ],
            "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _gt_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .pred p; setp.gt.u32 p, $1, $2; selp.s32 $0, 1, 0, p; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lt_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .pred p; setp.lt.u32 p, $1, $2; selp.s32 $0, 1, 0, p; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _eq_u32(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .pred p; setp.eq.u32 p, $1, $2; selp.s32 $0, 1, 0, p; }",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ge_u64(a: Uint64, b: Uint64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint64(a).ir_value(loc=loc, ip=ip),
                Uint64(b).ir_value(loc=loc, ip=ip),
            ],
            "{ .reg .pred p; setp.ge.u64 p, $1, $2; selp.s32 $0, 1, 0, p; }",
            "=r,l,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _vote_ballot_u32(pred: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p;
                setp.ne.s32 p, $1, 0;
                vote.sync.ballot.b32 $0, p, 0xffffffff;
            }
            """,
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _match4_mask(
    digit: Int32,
    b0: Uint32,
    b1: Uint32,
    b2: Uint32,
    b3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Return the 32-lane mask of lanes whose 4-bit digit matches ``digit``."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(digit).ir_value(loc=loc, ip=ip),
                Uint32(b0).ir_value(loc=loc, ip=ip),
                Uint32(b1).ir_value(loc=loc, ip=ip),
                Uint32(b2).ir_value(loc=loc, ip=ip),
                Uint32(b3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p0, p1, p2, p3;
                .reg .b32 t0, t1, t2, t3, m0, m1, m2, m3;
                and.b32 t0, $1, 1; setp.ne.b32 p0, t0, 0;
                not.b32 m0, $2; selp.b32 m0, $2, m0, p0;
                and.b32 t1, $1, 2; setp.ne.b32 p1, t1, 0;
                not.b32 m1, $3; selp.b32 m1, $3, m1, p1;
                and.b32 t2, $1, 4; setp.ne.b32 p2, t2, 0;
                not.b32 m2, $4; selp.b32 m2, $4, m2, p2;
                and.b32 t3, $1, 8; setp.ne.b32 p3, t3, 0;
                not.b32 m3, $5; selp.b32 m3, $5, m3, p3;
                and.b32 m0, m0, m1;
                and.b32 m2, m2, m3;
                and.b32 $0, m0, m2;
            }
            """,
            "=r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _popc_u32(mask: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(mask).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lanemask_lt(*, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %lanemask_lt;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_exclusive_sum_i32(val: Int32, *, loc=None, ip=None) -> tuple[Int32, Int32]:
    """Return ``(exclusive_prefix, warp_total)`` across all 32 lanes."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(val).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p;
            .reg .s32 x, tmp;
            mov.s32 x, $2;
            shfl.sync.up.b32 tmp|p, x, 1, 0, 0xffffffff;
            @p add.s32 x, x, tmp;
            shfl.sync.up.b32 tmp|p, x, 2, 0, 0xffffffff;
            @p add.s32 x, x, tmp;
            shfl.sync.up.b32 tmp|p, x, 4, 0, 0xffffffff;
            @p add.s32 x, x, tmp;
            shfl.sync.up.b32 tmp|p, x, 8, 0, 0xffffffff;
            @p add.s32 x, x, tmp;
            shfl.sync.up.b32 tmp|p, x, 16, 0, 0xffffffff;
            @p add.s32 x, x, tmp;
            sub.s32 $0, x, $2;
            shfl.sync.idx.b32 $1, x, 31, 31, 0xffffffff;
        }
        """,
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _redux_add_s32(val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(val).ir_value(loc=loc, ip=ip)],
            "redux.sync.add.s32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_max_s32(val: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(val).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.s32 $0, $1, 0xffffffff;",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_global_i64(addr: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u64(addr: Int64, val: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint64(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_relaxed_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_cg_u64(addr: Int64, *, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_release_sys_u64(addr: Int64, val: Uint64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint64(val).ir_value(loc=loc, ip=ip),
        ],
        "st.release.sys.global.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_acquire_sys_u64(addr: Int64, *, loc=None, ip=None) -> Uint64:
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.sys.global.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _advance_graph_epoch(
    epoch_addr: Int64,
    arrived_addr: Int64,
    grid_x: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Advance the channel graph epoch once every CTA has finished staging."""
    llvm.inline_asm(
        None,
        [
            Int64(epoch_addr).ir_value(loc=loc, ip=ip),
            Int64(arrived_addr).ir_value(loc=loc, ip=ip),
            Uint32(grid_x).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .pred not_last;
            .reg .u32 prior, arrived, discarded;
            atom.global.add.u32 prior, [$1], 1;
            add.u32 arrived, prior, 1;
            setp.ne.u32 not_last, arrived, $2;
            @not_last bra itk_epoch_done;
            atom.global.exch.b32 discarded, [$1], 0;
            atom.global.add.u32 discarded, [$0], 1;
        itk_epoch_done:
        }
        """,
        "l,l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# CuTeDSL kernel
# ---------------------------------------------------------------------------


class InterleavedTopKSlotsLaunch:
    """One-CTA-per-row distributed top-2048 slot selector for World in {4, 8, 16}."""

    def __init__(self, world: int) -> None:
        if world not in SUPPORTED_WORLDS:
            raise ValueError(
                f"interleaved top-k requires world size in {SUPPORTED_WORLDS}, got {world}"
            )
        self._world = int(world)

    @cute.jit
    def _receive_sys(self, addr: Int64, gen: Uint32) -> Uint64:
        val = _ld_acquire_sys_u64(addr)
        while _unpack_u64_hi(val) != gen:
            val = _ld_acquire_sys_u64(addr)
        return val

    @cute.jit
    def _block_exclusive_sum(
        self,
        flag: Int32,
        warp: Int32,
        lane: Int32,
        warp_sums,
    ) -> tuple[Int32, Int32]:
        w_excl, w_tot = _warp_exclusive_sum_i32(flag)
        if lane == Int32(0):
            warp_sums[warp] = w_tot
        cute.arch.sync_threads()
        warp_base = Int32(0)
        total = Int32(0)
        for w in cutlass.range_constexpr(8):
            wt = Int32(warp_sums[Int32(w)])
            if Int32(w) < warp:
                warp_base = warp_base + wt
            total = total + wt
        cute.arch.sync_threads()
        return warp_base + w_excl, total

    @cute.jit
    def _radix_pass(
        self,
        src,
        dst,
        warp_hist,
        warp: Int32,
        lane: Int32,
        lanemask_lt: Uint32,
        ipt: cutlass.Constexpr[int],
        bit_pos: Int32,
    ) -> None:
        """One stable 4-bit (16-bin) warp-ballot LSD radix scatter pass."""
        vals = cute.make_rmem_tensor((ipt,), Uint64)
        digits = cute.make_rmem_tensor((ipt,), Int32)
        intra = cute.make_rmem_tensor((ipt,), Int32)
        running = Int32(0)

        for i in cutlass.range_constexpr(ipt):
            idx = (warp * Int32(ipt) + Int32(i)) * Int32(32) + lane
            val = Uint64(src[idx])
            vals[i] = val
            d = _nibble_u64(val, bit_pos)
            digits[i] = d

            b0 = _vote_ballot_u32(d & Int32(1))
            b1 = _vote_ballot_u32(d & Int32(2))
            b2 = _vote_ballot_u32(d & Int32(4))
            b3 = _vote_ballot_u32(d & Int32(8))

            same = _match4_mask(d, b0, b1, b2, b3)
            rank_in_step = _popc_u32(same & lanemask_lt)
            prior = _shfl_idx_i32(running, d)
            intra[i] = prior + rank_in_step

            if lane < Int32(16):
                bin_mask = _match4_mask(lane, b0, b1, b2, b3)
                running = running + _popc_u32(bin_mask)

        if lane < Int32(16):
            warp_hist[lane * Int32(8) + warp] = running
        cute.arch.sync_threads()

        # Single-warp exclusive scan over all 128 (bin, warp) counters.
        if warp == Int32(0):
            base_i = lane * Int32(4)
            v0 = Int32(warp_hist[base_i])
            v1 = Int32(warp_hist[base_i + Int32(1)])
            v2 = Int32(warp_hist[base_i + Int32(2)])
            v3 = Int32(warp_hist[base_i + Int32(3)])
            prefix, _ = _warp_exclusive_sum_i32(v0 + v1 + v2 + v3)
            warp_hist[base_i] = prefix
            warp_hist[base_i + Int32(1)] = prefix + v0
            warp_hist[base_i + Int32(2)] = prefix + v0 + v1
            warp_hist[base_i + Int32(3)] = prefix + v0 + v1 + v2
        cute.arch.sync_threads()

        for i in cutlass.range_constexpr(ipt):
            d = Int32(digits[i])
            out_idx = Int32(warp_hist[d * Int32(8) + warp]) + Int32(intra[i])
            dst[out_idx] = Uint64(vals[i])
        cute.arch.sync_threads()

    @cute.jit
    def _sort_local_2048(
        self,
        buf_a,
        buf_b,
        warp_hist,
        warp: Int32,
        lane: Int32,
        lanemask_lt: Uint32,
    ) -> None:
        """Stably sort 2048 inverted tie-keys ascending by their high 32 bits."""
        bit_pos = Int32(32)
        while bit_pos < Int32(64):
            self._radix_pass(
                buf_a, buf_b, warp_hist, warp, lane, lanemask_lt, 8, bit_pos
            )
            self._radix_pass(
                buf_b, buf_a, warp_hist, warp, lane, lanemask_lt, 8, bit_pos + Int32(4)
            )
            bit_pos = bit_pos + Int32(8)

    @cute.jit
    def _sort_boundary_1024(
        self,
        buf_a,
        buf_b,
        warp_hist,
        warp: Int32,
        lane: Int32,
        lanemask_lt: Uint32,
    ) -> None:
        """Sort 1024 inverted tie-keys (or ~0 padding) ascending.

        Every valid tie-key has inverted bits 31..15 equal to 0, while padding ~0
        has bits 31..15 equal to 0x1FFFF. Sorting the 12 nibbles covering bits
        [15..0] and [63..32] is therefore bit-exact for all 2^32 float32 score
        words (including -NaN) and all rank-major ordinals < 32768.
        """
        bit_pos = Int32(0)
        while bit_pos < Int32(64):
            self._radix_pass(
                buf_a, buf_b, warp_hist, warp, lane, lanemask_lt, 4, bit_pos
            )
            self._radix_pass(
                buf_b, buf_a, warp_hist, warp, lane, lanemask_lt, 4, bit_pos + Int32(4)
            )
            bit_pos = bit_pos + Int32(8)
            if bit_pos == Int32(16):
                bit_pos = Int32(32)

    @cute.jit
    def __call__(
        self,
        ids: cute.Pointer,
        scores: cute.Pointer,
        positions: cute.Pointer,
        tables: cute.Pointer,
        owners: cute.Pointer,
        slots: cute.Pointer,
        counts: cute.Pointer,
        stages: tuple,
        signals: tuple,
        bank_stride: Int64,
        rank: Int32,
        table_stride: Int64,
        table_columns: Int64,
        cache_pages: Int32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            ids,
            scores,
            positions,
            tables,
            owners,
            slots,
            counts,
            stages,
            signals,
            bank_stride,
            rank,
            table_stride,
            table_columns,
            cache_pages,
        ).launch(grid=(rows, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        ids,
        scores,
        positions,
        tables,
        owners,
        slots,
        counts,
        stages,
        signals,
        bank_stride,
        rank,
        table_stride,
        table_columns,
        cache_pages,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        tid = Int32(tid)
        row = Int32(row)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        lanemask_lt = _lanemask_lt()

        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            buf_a: cute.struct.Align[cute.struct.MemRange[Uint64, 2048], 16]
            buf_b: cute.struct.Align[cute.struct.MemRange[Uint64, 2048], 16]
            warp_hist: cute.struct.Align[cute.struct.MemRange[Int32, 128], 16]
            stage_addrs: cute.struct.Align[cute.struct.MemRange[Int64, 16], 16]
            flag_addrs: cute.struct.Align[cute.struct.MemRange[Int64, 16], 16]
            peer_counts: cute.struct.Align[cute.struct.MemRange[Uint32, 16], 16]
            first: cute.struct.Align[cute.struct.MemRange[Int32, 16], 16]
            stop: cute.struct.Align[cute.struct.MemRange[Int32, 16], 16]
            offset: cute.struct.Align[cute.struct.MemRange[Int32, 16], 16]
            peer_for_block: cute.struct.Align[cute.struct.MemRange[Int32, 32], 16]
            local_block: cute.struct.Align[cute.struct.MemRange[Int32, 32], 16]
            scalars_i32: cute.struct.Align[cute.struct.MemRange[Int32, 16], 16]
            scalars_u32: cute.struct.Align[cute.struct.MemRange[Uint32, 8], 16]
            scalars_u64: cute.struct.Align[cute.struct.MemRange[Uint64, 4], 16]

        storage = smem_alloc.allocate(SharedStorage)
        buf_a = storage.buf_a.get_tensor(cute.make_layout((2048,), stride=(1,)))
        buf_b = storage.buf_b.get_tensor(cute.make_layout((2048,), stride=(1,)))
        warp_hist = storage.warp_hist.get_tensor(cute.make_layout((128,), stride=(1,)))
        stage_addrs = storage.stage_addrs.get_tensor(
            cute.make_layout((16,), stride=(1,))
        )
        flag_addrs = storage.flag_addrs.get_tensor(
            cute.make_layout((16,), stride=(1,))
        )
        peer_counts = storage.peer_counts.get_tensor(
            cute.make_layout((16,), stride=(1,))
        )
        first_arr = storage.first.get_tensor(cute.make_layout((16,), stride=(1,)))
        stop_arr = storage.stop.get_tensor(cute.make_layout((16,), stride=(1,)))
        offset_arr = storage.offset.get_tensor(cute.make_layout((16,), stride=(1,)))
        peer_for_block = storage.peer_for_block.get_tensor(
            cute.make_layout((32,), stride=(1,))
        )
        local_block = storage.local_block.get_tensor(
            cute.make_layout((32,), stride=(1,))
        )
        scalars_i32 = storage.scalars_i32.get_tensor(
            cute.make_layout((16,), stride=(1,))
        )
        scalars_u32 = storage.scalars_u32.get_tensor(
            cute.make_layout((8,), stride=(1,))
        )
        scalars_u64 = storage.scalars_u64.get_tensor(
            cute.make_layout((4,), stride=(1,))
        )

        ids_addr = Int64(ids.toint())
        scores_addr = Int64(scores.toint())
        positions_addr = Int64(positions.toint())
        tables_addr = Int64(tables.toint())
        owners_addr = Int64(owners.toint())
        slots_addr = Int64(slots.toint())
        counts_addr = Int64(counts.toint())
        row_k_offset = Int64(row) * Int64(K)

        position = _ld_global_i64(positions_addr + Int64(row) * Int64(8))
        # Overflow-safe check equivalent to:
        # total = max(0, min(position, World * K) + 1); short_row = total <= K
        # Since World >= 4, World * K >= 8192 > K, so total <= K iff position < K.
        short_row = Int32(0)
        if position < Int64(K):
            short_row = Int32(1)

        # Initialize per-peer addresses and row generation on thread 0.
        if tid == Int32(0):
            for p in cutlass.range_constexpr(self._world):
                flag_addrs[Int32(p)] = Int64(signals[p].toint())
            header0 = Int64(flag_addrs[rank])
            graph_epoch = _ld_relaxed_gpu_u32(header0 + Int64(1025 * 4))
            bank_offset = Int64(graph_epoch & Uint32(1)) * Int64(bank_stride)
            row_stage_bytes = Int64(row) * Int64(STAGE_ITEMS * 8)
            for p in cutlass.range_constexpr(self._world):
                stage_addrs[Int32(p)] = (
                    Int64(stages[p].toint()) + bank_offset + row_stage_bytes
                )
            staging0 = Int32(0)
            r = Int32(0)
            gx = Int32(grid_x)
            while r < gx:
                if _ld_global_i64(positions_addr + Int64(r) * Int64(8)) >= Int64(K):
                    staging0 = Int32(1)
                r = r + Int32(1)
            seq_word = (
                Int64(1024) + Int64(row) * Int64(16 * 32) + Int64(4) + Int64(staging0)
            )
            seq_addr = header0 + seq_word * Int64(4)
            sequence = _ld_global_u32(seq_addr)
            gen0 = sequence + Uint32(1)
            if sequence == Uint32(_GEN_ROLLOVER):
                gen0 = Uint32(1)
            _st_global_u32(seq_addr, gen0)
            scalars_u32[Int32(0)] = gen0
            scalars_i32[Int32(0)] = staging0
            scalars_i32[Int32(1)] = Int32(0)  # above
            scalars_i32[Int32(2)] = Int32(0)  # below
            scalars_i32[Int32(3)] = Int32(0)  # local_ties
            scalars_i32[Int32(4)] = Int32(0)  # extent
            scalars_i32[Int32(5)] = Int32(K)  # remaining
            scalars_i32[Int32(6)] = Int32(0)  # preceding
        cute.arch.sync_threads()

        gen = Uint32(scalars_u32[Int32(0)])
        staging = Int32(scalars_i32[Int32(0)])
        header_addr = Int64(flag_addrs[rank])
        local_stage_addr = Int64(stage_addrs[rank])

        hi = Int32(0)
        lo = Int32(0)
        special = Int32(0)
        last = Int32(0)

        if short_row != Int32(0):
            if position >= Int64(0):
                for s in cutlass.range_constexpr(8):
                    j = Int32(s * THREADS) + tid
                    elem_off = (row_k_offset + Int64(j)) * Int64(4)
                    id_val = _ld_global_i32(ids_addr + elem_off)
                    if id_val >= Int32(0):
                        v = _ordered_key(_ld_global_u32(scores_addr + elem_off))
                        hi = hi + _gt_u32(v, Uint32(_PAD_KEY))
                        lo = lo + _lt_u32(v, Uint32(_PAD_KEY))
                        special = special + _eq_u32(v, Uint32(_PAD_KEY))
                        last = j + Int32(1)
            w_hi = _redux_add_s32(hi)
            w_lo = _redux_add_s32(lo)
            w_sp = _redux_add_s32(special)
            w_last = _redux_max_s32(last)
            if lane == Int32(0):
                warp_hist[warp] = w_hi
                warp_hist[Int32(8) + warp] = w_lo
                warp_hist[Int32(16) + warp] = w_sp
                warp_hist[Int32(24) + warp] = w_last
            cute.arch.sync_threads()
            if tid == Int32(0):
                t_hi = Int32(0)
                t_lo = Int32(0)
                t_sp = Int32(0)
                t_last = Int32(0)
                for w in cutlass.range_constexpr(8):
                    t_hi = t_hi + Int32(warp_hist[Int32(w)])
                    t_lo = t_lo + Int32(warp_hist[Int32(8 + w)])
                    t_sp = t_sp + Int32(warp_hist[Int32(16 + w)])
                    wl = Int32(warp_hist[Int32(24 + w)])
                    if wl > t_last:
                        t_last = wl
                scalars_i32[Int32(1)] = t_hi
                scalars_i32[Int32(2)] = t_lo
                scalars_i32[Int32(3)] = t_sp
                scalars_i32[Int32(4)] = t_last
            cute.arch.sync_threads()
        else:
            rank_base = Uint32(rank) * Uint32(K)
            for i in cutlass.range_constexpr(8):
                j = (warp * Int32(8) + Int32(i)) * Int32(32) + lane
                elem_off = (row_k_offset + Int64(j)) * Int64(4)
                id_val = _ld_global_i32(ids_addr + elem_off)
                v = Uint32(_PAD_KEY)
                if id_val >= Int32(0):
                    last = j + Int32(1)
                    v = _ordered_key(_ld_global_u32(scores_addr + elem_off))
                buf_a[j] = _pack_u64(rank_base + Uint32(j), _not_u32(v))
            w_last = _redux_max_s32(last)
            if lane == Int32(0):
                warp_hist[warp] = w_last
            cute.arch.sync_threads()
            if tid == Int32(0):
                t_last = Int32(0)
                for w in cutlass.range_constexpr(8):
                    wl = Int32(warp_hist[Int32(w)])
                    if wl > t_last:
                        t_last = wl
                scalars_i32[Int32(4)] = t_last
            cute.arch.sync_threads()

            self._sort_local_2048(buf_a, buf_b, warp_hist, warp, lane, lanemask_lt)

            for s in cutlass.range_constexpr(8):
                idx = Int32(s * THREADS) + tid
                tk = _not_u64(Uint64(buf_a[idx]))
                _st_global_u64(local_stage_addr + Int64(idx) * Int64(8), tk)
            if tid < Int32(SAMPLES):
                tk_min = _not_u64(Uint64(buf_a[tid * Int32(BLOCK_ITEMS) + Int32(31)]))
                _st_global_u64(
                    local_stage_addr + Int64(K + tid) * Int64(8),
                    tk_min,
                )
            cute.arch.sync_threads()

        above = Int32(scalars_i32[Int32(1)])
        below = Int32(scalars_i32[Int32(2)])
        local_ties = Int32(scalars_i32[Int32(3)])
        extent = Int32(scalars_i32[Int32(4)])
        summary = Uint32(above) | (Uint32(below) << Uint32(16))

        # Cross-rank synchronization: short-row summary ring or staging mailbox.
        if staging == Int32(0):
            seq = gen - Uint32(1)
            if (seq & Uint32(_WRAP_MASK)) == Uint32(0):
                if tid < Int32(self._world):
                    wbank = Int64((seq >> Uint32(_WRAP_SHIFT)) & Uint32(1))
                    peer_flag = Int64(flag_addrs[tid])
                    peer_wrap = (
                        Int64(1024)
                        + ((wbank * Int64(64) + Int64(row)) * Int64(16) + Int64(rank))
                        * Int64(32)
                        + Int64(16)
                    )
                    my_wrap = (
                        Int64(1024)
                        + ((wbank * Int64(64) + Int64(row)) * Int64(16) + Int64(tid))
                        * Int64(32)
                        + Int64(16)
                    )
                    _st_release_sys_u64(
                        peer_flag + peer_wrap * Int64(4),
                        _pack_u64(Uint32(0), gen),
                    )
                    _ = self._receive_sys(header_addr + my_wrap * Int64(4), gen)
            cute.arch.sync_threads()

            ring_word = (
                Int64(1024)
                + (Int64(seq & Uint32(_WRAP_MASK)) * Int64(MAX_ROWS) + Int64(row))
                * Int64(32)
                + Int64(20)
            )
            if tid == Int32(0):
                _st_release_sys_u64(
                    header_addr + ring_word * Int64(4),
                    _pack_u64(summary, gen),
                )
            cute.arch.sync_threads()

            if local_ties > Int32(0):
                if tid < Int32(self._world):
                    peer_flag = Int64(flag_addrs[tid])
                    received = self._receive_sys(peer_flag + ring_word * Int64(4), gen)
                    peer_counts[tid] = _unpack_u64_lo(received)
        else:
            if tid < Int32(self._world):
                mbank = Int64(gen & Uint32(1))
                peer_flag = Int64(flag_addrs[tid])
                peer_mbox = (
                    Int64(1024)
                    + ((mbank * Int64(64) + Int64(row)) * Int64(16) + Int64(rank))
                    * Int64(32)
                    + Int64(8)
                )
                my_mbox = (
                    Int64(1024)
                    + ((mbank * Int64(64) + Int64(row)) * Int64(16) + Int64(tid))
                    * Int64(32)
                    + Int64(8)
                )
                _st_release_sys_u64(
                    peer_flag + peer_mbox * Int64(4),
                    _pack_u64(summary, gen),
                )
                received = self._receive_sys(header_addr + my_mbox * Int64(4), gen)
                peer_counts[tid] = _unpack_u64_lo(received)
        cute.arch.sync_threads()

        # Compute either short-row tie quota or exact distributed boundary threshold.
        threshold = Uint64(0)
        if short_row != Int32(0):
            if tid == Int32(0):
                rem = Int32(K)
                prec = rank * Int32(K)
                skip_peer_sum = Int32(0)
                if staging == Int32(0):
                    if local_ties == Int32(0):
                        skip_peer_sum = Int32(1)
                if skip_peer_sum != Int32(0):
                    rem = Int32(0)
                else:
                    for p in cutlass.range_constexpr(self._world):
                        pc = Uint32(peer_counts[Int32(p)])
                        greater = Int32(pc & Uint32(0xFFFF))
                        lower = Int32(pc >> Uint32(16))
                        rem = rem - greater
                        if Int32(p) < rank:
                            prec = prec - (greater + lower)
                scalars_i32[Int32(5)] = rem
                scalars_i32[Int32(6)] = prec
            cute.arch.sync_threads()
        else:
            # Load World * 64 block-minimum samples; keep unsorted copy in buf_a[1024..2047].
            for s in cutlass.range_constexpr(4):
                j = Int32(s * THREADS) + tid
                val = Uint64(0)
                if j < Int32(self._world * SAMPLES):
                    peer = j >> Int32(6)
                    sample_idx = j & Int32(63)
                    val = _ld_cg_u64(
                        Int64(stage_addrs[peer])
                        + Int64(K + sample_idx) * Int64(8)
                    )
                buf_a[Int32(1024) + j] = val
                buf_a[j] = _not_u64(val)
            cute.arch.sync_threads()

            self._sort_boundary_1024(buf_a, buf_b, warp_hist, warp, lane, lanemask_lt)

            if tid == Int32(0):
                scalars_u64[Int32(0)] = _not_u64(
                    Uint64(buf_a[Int32(SAMPLES - self._world - 1)])
                )
                scalars_u64[Int32(1)] = _not_u64(Uint64(buf_a[Int32(SAMPLES - 1)]))
            cute.arch.sync_threads()

            upper = Uint64(scalars_u64[Int32(0)])
            lower = Uint64(scalars_u64[Int32(1)])
            if tid < Int32(self._world):
                fc = Int32(0)
                sc = Int32(0)
                base_s = Int32(1024) + tid * Int32(SAMPLES)
                i = Int32(0)
                while i < Int32(SAMPLES):
                    sv = Uint64(buf_a[base_s + i])
                    fc = fc + _ge_u64(sv, upper)
                    sc = sc + _ge_u64(sv, lower)
                    i = i + Int32(1)
                first_arr[tid] = fc
                sc_next = sc + Int32(1)
                if sc_next > Int32(SAMPLES):
                    sc_next = Int32(SAMPLES)
                stop_arr[tid] = sc_next
            cute.arch.sync_threads()

            if tid == Int32(0):
                off = Int32(0)
                for p in cutlass.range_constexpr(self._world):
                    offset_arr[Int32(p)] = off
                    off = off + (Int32(stop_arr[Int32(p)]) - Int32(first_arr[Int32(p)]))
                i = off
                while i < Int32(2 * self._world):
                    peer_for_block[i] = Int32(-1)
                    i = i + Int32(1)
            cute.arch.sync_threads()

            if tid < Int32(self._world):
                off = Int32(offset_arr[tid])
                cur = Int32(first_arr[tid])
                end = Int32(stop_arr[tid])
                while cur < end:
                    peer_for_block[off] = tid
                    local_block[off] = cur
                    cur = cur + Int32(1)
                    off = off + Int32(1)
            cute.arch.sync_threads()

            for s in cutlass.range_constexpr(4):
                j = Int32(s * THREADS) + tid
                val = Uint64(0)
                if j < Int32(2 * self._world * BLOCK_ITEMS):
                    blk = j >> Int32(5)
                    peer = Int32(peer_for_block[blk])
                    if peer >= Int32(0):
                        elem = Int32(local_block[blk]) * Int32(BLOCK_ITEMS) + (
                            j & Int32(31)
                        )
                        val = _ld_cg_u64(
                            Int64(stage_addrs[peer]) + Int64(elem) * Int64(8)
                        )
                buf_a[j] = _not_u64(val)
            cute.arch.sync_threads()

            self._sort_boundary_1024(buf_a, buf_b, warp_hist, warp, lane, lanemask_lt)

            threshold = _not_u64(Uint64(buf_a[Int32(self._world * BLOCK_ITEMS - 1)]))

        # Preserve original local indexer order, map valid winners to physical slots.
        local_count = extent
        need_ties = Int32(0)
        if short_row != Int32(0):
            if local_ties > Int32(0):
                need_ties = Int32(1)
        local_remaining = Int32(scalars_i32[Int32(5)])
        local_preceding = Int32(scalars_i32[Int32(6)])
        local_nout = Int32(0)
        rank_base_u32 = Uint32(rank) * Uint32(K)
        row_table_addr = tables_addr + Int64(row) * Int64(table_stride) * Int64(4)

        j0 = Int32(0)
        while j0 < local_count:
            j = j0 + tid
            local_id = Int32(-1)
            raw_score = Uint32(0xFF800000)
            if j < local_count:
                elem_off = (row_k_offset + Int64(j)) * Int64(4)
                local_id = _ld_global_i32(ids_addr + elem_off)
                if local_id >= Int32(0):
                    raw_score = _ld_global_u32(scores_addr + elem_off)
            v = Uint32(_PAD_KEY)
            if local_id >= Int32(0):
                v = _ordered_key(raw_score)

            chosen = Int32(0)
            if short_row != Int32(0):
                ties_before = Int32(0)
                ties_total = Int32(0)
                if need_ties != Int32(0):
                    is_tie = Int32(0)
                    if j < local_count:
                        is_tie = _eq_u32(v, Uint32(_PAD_KEY))
                    ties_before, ties_total = self._block_exclusive_sum(
                        is_tie, warp, lane, warp_hist
                    )
                if _gt_u32(v, Uint32(_PAD_KEY)) != Int32(0):
                    chosen = Int32(1)
                elif _eq_u32(v, Uint32(_PAD_KEY)) != Int32(0):
                    if local_preceding + ties_before < local_remaining:
                        chosen = Int32(1)
                local_preceding = local_preceding + ties_total
            else:
                tk = _pack_u64(_not_u32(rank_base_u32 + Uint32(j)), v)
                chosen = _ge_u64(tk, threshold)

            mapped = Int32(-1)
            if j < local_count:
                if chosen != Int32(0):
                    if local_id >= Int32(0):
                        token = Int64(local_id) * Int64(self._world) + Int64(rank)
                        if token <= position:
                            col = Int64(local_id >> Int32(7))
                            if col < Int64(table_columns):
                                page = _ld_global_i32(row_table_addr + col * Int64(4))
                                if page >= Int32(0):
                                    if page < cache_pages:
                                        mapped = page * Int32(128) + (
                                            local_id & Int32(127)
                                        )

            is_mapped = Int32(0)
            if mapped >= Int32(0):
                is_mapped = Int32(1)
            prior, count = self._block_exclusive_sum(is_mapped, warp, lane, warp_hist)
            if mapped >= Int32(0):
                out_off = (row_k_offset + Int64(local_nout + prior)) * Int64(4)
                _st_global_i32(slots_addr + out_off, mapped)
                _st_global_i32(owners_addr + out_off, rank)
            local_nout = local_nout + count
            j0 = j0 + Int32(THREADS)

        pad_j = local_nout + tid
        while pad_j < Int32(K):
            out_off = (row_k_offset + Int64(pad_j)) * Int64(4)
            _st_global_i32(owners_addr + out_off, Int32(-1))
            _st_global_i32(slots_addr + out_off, Int32(-1))
            pad_j = pad_j + Int32(THREADS)

        if tid == Int32(0):
            _st_global_i32(counts_addr + Int64(row) * Int64(4), local_nout)
            if staging != Int32(0):
                _advance_graph_epoch(
                    header_addr + Int64(1025 * 4),
                    header_addr + Int64(1026 * 4),
                    Uint32(grid_x),
                )


def _i32_ptr(addr: int, align: int = 4):
    return make_ptr(Int32, int(addr), cute.AddressSpace.gmem, assumed_align=align)


def _u32_ptr(addr: int, align: int = 4):
    return make_ptr(Uint32, int(addr), cute.AddressSpace.gmem, assumed_align=align)


def _i64_ptr(addr: int, align: int = 8):
    return make_ptr(Int64, int(addr), cute.AddressSpace.gmem, assumed_align=align)


def _u8_ptr(addr: int, align: int = 8):
    return make_ptr(Uint8, int(addr), cute.AddressSpace.gmem, assumed_align=align)


@program_cache(scope="preparation")
def compile_interleaved_topk_slots(world: int, device_ordinal: int = 0):
    """Compile the pure CuTeDSL interleaved top-2048 slot selector for ``world``."""
    import torch

    world = int(world)
    launch = InterleavedTopKSlotsLaunch(world)
    with torch.cuda.device(int(device_ordinal)):
        raw = b12x_compile(
            launch,
            _i32_ptr(16),
            _u32_ptr(16),
            _i64_ptr(16),
            _i32_ptr(16),
            _i32_ptr(16),
            _i32_ptr(16),
            _i32_ptr(16),
            (_u8_ptr(16),) * world,
            (_u32_ptr(16, align=8),) * world,
            Int64(1),
            Int32(0),
            Int64(1),
            Int64(1),
            Int32(1),
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "comm.pcie.interleaved_topk_slots",
                _SPEC_VERSION,
                (world,),
                labels=("world",),
            ),
        )

    def run(
        ids,
        scores,
        positions,
        tables,
        owners,
        slots,
        counts,
        stages: Sequence[int],
        signals: Sequence[int],
        bank_stride: int,
        rank: int,
        cache_pages: int,
    ) -> None:
        raw(
            _i32_ptr(ids.data_ptr()),
            _u32_ptr(scores.data_ptr()),
            _i64_ptr(positions.data_ptr()),
            _i32_ptr(tables.data_ptr()),
            _i32_ptr(owners.data_ptr()),
            _i32_ptr(slots.data_ptr()),
            _i32_ptr(counts.data_ptr()),
            tuple(_u8_ptr(ptr) for ptr in stages),
            tuple(_u32_ptr(ptr, align=8) for ptr in signals),
            Int64(bank_stride),
            Int32(rank),
            Int64(tables.stride(0)),
            Int64(tables.size(1)),
            Int32(cache_pages),
            Int32(ids.size(0)),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
