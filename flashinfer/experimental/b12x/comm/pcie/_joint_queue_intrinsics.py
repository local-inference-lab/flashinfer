"""GPU-scoped queue operations; payload publication uses a separate system fence."""

from cutlass import Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ._joint_profile import JOINT_QUEUE_ARRIVAL_LOCK


@dsl_user_op
def queue_arrive(address: Int64, active: Uint32, *, loc=None, ip=None) -> Uint32:
    """Elect initializer (1), admit a follower (0), or elect final arrival (2).

    The initializer holds a high-bit lock until it has published all queue
    records. Followers cannot consume a stale phase token across epoch wrap.
    Every successful arrival CAS acquires initialization and preceding arrivals.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(address).ir_value(loc=loc, ip=ip),
                Uint32(active).ir_value(loc=loc, ip=ip),
                Uint32(JOINT_QUEUE_ARRIVAL_LOCK).ir_value(loc=loc, ip=ip),
            ],
            """
        {
            .reg .pred busy, first, last, won;
            .reg .b32 seen, wanted, prior;
        retry_arrival:
            ld.acquire.gpu.global.u32 seen, [$1];
            setp.eq.u32 busy, seen, $3;
            @busy bra retry_arrival;
            setp.eq.u32 first, seen, 0;
            @first bra first_arrival;
            add.u32 wanted, seen, 1;
            setp.eq.u32 last, wanted, $2;
            @last mov.u32 wanted, 0;
            atom.acq_rel.gpu.global.cas.b32 prior, [$1], seen, wanted;
            setp.eq.u32 won, prior, seen;
            @!won bra retry_arrival;
            selp.u32 $0, 2, 0, last;
            bra arrival_done;
        first_arrival:
            atom.acq_rel.gpu.global.cas.b32 prior, [$1], 0, $3;
            setp.eq.u32 won, prior, 0;
            @!won bra retry_arrival;
            mov.u32 $0, 1;
        arrival_done:
        }
        """,
            "=r,l,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def queue_claim(address: Int64, *, loc=None, ip=None) -> Uint64:
    """Return one relaxed GPU ticket; exhausted callers must await phase change.

    Each CTA contributes at most one exhausted ticket per phase. At least 128
    packs per tile and positive Int64 payload sizes keep tile count plus the
    maximum 64 workers below the uint64 limit.
    """
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], 1;",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def queue_complete(address: Int64, *, loc=None, ip=None) -> Uint64:
    """Acquire preceding tile publications and release this tile's stores."""
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u64 $0, [$1], 1;",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def queue_load(address: Int64, *, loc=None, ip=None) -> Uint64:
    """Acquire a generation-tagged phase publication on this GPU."""
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u64 $0, [$1];",
            "=l,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def queue_store(address: Int64, value: Uint64, *, loc=None, ip=None):
    """Release initialization or a completed phase to this GPU's workers."""
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.gpu.global.u64 [$0], $1;",
        "l,l,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def queue_store_u32(address: Int64, value: Uint32, *, loc=None, ip=None):
    """Release initialized queue state before admitting another CTA."""
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
