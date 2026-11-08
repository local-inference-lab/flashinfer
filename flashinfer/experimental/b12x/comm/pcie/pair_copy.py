"""Exact two-field row packing and owner-row reads from synchronized peer storage."""

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.preparation import MemoryRequirements, Plan, make_fixed_contract
from b12x.preparation.types import plan_from_handle, require_prepared


@dsl_user_op
def _peer_word(address: Int64, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.relaxed.sys.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dataclass(frozen=True)
class Query:
    mode: str
    rows: int
    first_bytes: int
    second_bytes: int
    world: int

    def __post_init__(self):
        if (
            self.mode not in ("pack", "unpack", "pull")
            or self.world not in (2, 4, 8, 16)
            or type(self.rows) is not int
            or self.rows <= 0
            or any(
                type(n) is not int or n <= 0 or n % 4
                for n in (self.first_bytes, self.second_bytes)
            )
            or self.mode == "pull"
            and self.rows % self.world
        ):
            raise ValueError(
                "pair copies require positive word-aligned fields and complete owner rows"
            )


class _Copy:
    def __init__(self, mode, first, second, world):
        self.mode, self.first, self.second, self.world = (
            mode,
            first // 4,
            second // 4,
            world,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Pointer,
        b: cute.Pointer,
        out_a: cute.Pointer,
        out_b: cute.Pointer,
        peers: cute.Pointer,
        rank: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        words = self.first + self.second
        count = rows * words
        owners = 1
        if cutlass.const_expr(self.mode == "unpack"):
            count *= self.world
        if cutlass.const_expr(self.mode == "pull"):
            count = (rows // self.world) * words
            owners = self.world
        self.kernel(a, b, out_a, out_b, peers, rank, rows).launch(
            grid=((count + 255) // 256, owners, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        a: cute.Pointer,
        b: cute.Pointer,
        out_a: cute.Pointer,
        out_b: cute.Pointer,
        peers: cute.Pointer,
        rank: cute.Pointer,
        rows: Int32,
    ):
        block, peer, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        item = Int64(block) * 256 + Int64(thread)
        words = self.first + self.second
        if cutlass.const_expr(self.mode == "pack"):
            row, column = item // words, item % words
            if row < rows:
                if column < self.first:
                    out_a[item] = a[row * self.first + column]
                else:
                    out_a[item] = b[row * self.second + column - self.first]
        elif cutlass.const_expr(self.mode == "unpack"):
            row, within = item // (words * self.world), item % (words * self.world)
            owner, column = within // words, within % words
            if row < rows:
                value = a[item]
                if column < self.first:
                    out_a[
                        row * self.world * self.first + owner * self.first + column
                    ] = value
                else:
                    out_b[
                        row * self.world * self.second
                        + owner * self.second
                        + column
                        - self.first
                    ] = value
        else:
            row, column = item // words, item % words
            owner = rank[0]
            if row < rows // self.world and owner >= 0 and owner < self.world:
                address = (
                    peers[peer] + ((row * self.world + owner) * words + column) * 4
                )
                value = _peer_word(address)
                if column < self.first:
                    out_a[
                        row * self.world * self.first
                        + Int64(peer) * self.first
                        + column
                    ] = value
                else:
                    out_b[
                        row * self.world * self.second
                        + Int64(peer) * self.second
                        + column
                        - self.first
                    ] = value


@program_cache(scope="preparation")
def compile_copy(payload, ordinal):
    query = Query(**dict(payload))
    operation = _Copy(query.mode, query.first_bytes, query.second_bytes, query.world)

    def pointer(dtype, value=16):
        return make_ptr(
            dtype,
            value,
            cute.AddressSpace.gmem,
            assumed_align=4 if dtype == Uint32 else 8,
        )

    executable = compile_kernel(
        operation,
        *(pointer(Uint32) for _ in range(4)),
        pointer(Int64),
        pointer(Int64),
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.pair_copy",
            1,
            (query.mode, query.first_bytes, query.second_bytes, query.world),
        ),
    )

    def run(a, b, out_a, out_b, peers, rank, rows):
        executable(
            *(pointer(Uint32, value) for value in (a, b, out_a, out_b)),
            pointer(Int64, peers),
            pointer(Int64, rank),
            Int32(rows),
            current_cuda_stream(),
        )

    return attach_programs(run, executable)


TUNING = make_fixed_contract(
    component_id="comm.pcie.pair_copy", query_type=Query, backend="cute"
)


@dataclass(frozen=True)
class _State:
    query: Query
    device: torch.device
    program: object

    def run(
        self, first, second, out_first, out_second, *, peers=None, rank=None, ready=None
    ):
        q = self.query
        tensors = [t for t in (first, second, out_first, out_second) if t is not None]
        if any(
            t.device != self.device
            or t.ndim != 2
            or not t.is_contiguous()
            or t.data_ptr() % 4
            for t in tensors
        ):
            raise ValueError("pair copies require aligned contiguous device tensors")

        def overlaps(a, b):
            return (
                a.data_ptr() < b.data_ptr() + b.nbytes
                and b.data_ptr() < a.data_ptr() + a.nbytes
            )

        outputs = [out_first] if out_second is None else [out_first, out_second]
        reads = [t for t in (first, second, peers, rank, ready) if t is not None]
        if any(overlaps(output, value) for output in outputs for value in reads) or (
            out_second is not None and overlaps(out_first, out_second)
        ):
            raise ValueError(
                "pair-copy destinations must be disjoint from inputs and one another"
            )
        packed = q.rows * (q.first_bytes + q.second_bytes)
        if q.mode == "pack":
            sizes = (first.nbytes, second.nbytes, out_first.nbytes)
            expected = (q.rows * q.first_bytes, q.rows * q.second_bytes, packed)
        elif q.mode == "unpack":
            sizes = (first.nbytes, out_first.nbytes, out_second.nbytes)
            expected = (
                packed * q.world,
                q.rows * q.first_bytes * q.world,
                q.rows * q.second_bytes * q.world,
            )
        else:
            sizes = (first.nbytes, out_first.nbytes, out_second.nbytes)
            expected = (packed, q.rows * q.first_bytes, q.rows * q.second_bytes)
            if (
                peers is None
                or tuple(peers.shape) != (q.world,)
                or peers.dtype != torch.int64
                or peers.device != self.device
                or rank is None
                or rank.numel() != 1
                or rank.dtype != torch.int64
                or rank.device != self.device
                or not peers.is_contiguous()
                or not rank.is_contiguous()
                or ready is None
                or ready.device != self.device
            ):
                raise ValueError(
                    "peer row reads require mapped addresses, rank and a producer-readiness dependency"
                )
        if sizes != expected:
            raise ValueError("pair-copy byte extents differ from preparation")

        def pointer(t):
            return 0 if t is None else t.data_ptr()

        self.program(
            pointer(first),
            pointer(second),
            pointer(out_first),
            pointer(out_second),
            pointer(peers),
            pointer(rank),
            q.rows,
        )


def plan(query):
    def jobs(config, device):
        return (
            CompileJob.create(
                "b12x.comm.pcie.pair_copy:compile_copy",
                TUNING.encode_query(query),
                CompileJob.DEVICE_ORDINAL,
            ),
        )

    def materialize(selection, device):
        program = compile_copy(TUNING.encode_query(query), device.ordinal)
        load_programs(program)
        return attach_programs(
            _State(query, torch.device("cuda", device.ordinal), program), program
        )

    return Plan(
        contract=TUNING,
        query=query,
        shared=True,
        _compile_jobs=jobs,
        _memory_requirements=lambda config, device: MemoryRequirements(),
        _materialize=materialize,
    )


@torch.library.custom_op("b12x::pair_copy", mutates_args=("out_first", "out_second"))
def _execute(
    first: torch.Tensor,
    second: torch.Tensor | None,
    out_first: torch.Tensor,
    out_second: torch.Tensor | None,
    peers: torch.Tensor | None,
    rank: torch.Tensor | None,
    ready: torch.Tensor | None,
    handle: int,
) -> None:
    require_prepared(plan_from_handle(handle), TUNING.component_id, first.device).run(
        first, second, out_first, out_second, peers=peers, rank=rank, ready=ready
    )


@_execute.register_fake
def _fake(first, second, out_first, out_second, peers, rank, ready, handle):
    pass


def run(
    first,
    second,
    out_first,
    out_second=None,
    *,
    plan,
    peers=None,
    rank=None,
    ready=None,
):
    """Copy raw words. Peer pulls require a system-scope producer barrier and a later consumer join.

    ``first`` retains the local staging tensor as a compiler dependency. ``ready``
    is the barrier result, preserving its order before any peer reads. The caller
    must keep peer staging immutable until every consumer has completed and
    supply an immutable rank in ``[0, world)`` with valid mapped peer addresses.
    """
    _execute(first, second, out_first, out_second, peers, rank, ready, plan.handle)
