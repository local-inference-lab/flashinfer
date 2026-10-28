"""Device timing, rank agreement, and publication for prepared joint profiles."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._cute_intrinsics import pcie_arrive_and_wait
from ._oneshot_cute import _dummy

SIGNAL_BYTES = 64 + 2 * 16 * 32 * 4
SCRATCH_BYTES = SIGNAL_BYTES + 32
SAMPLES_PER_MODE = 21


@dsl_user_op
def _globaltimer(*, loc=None, ip=None):
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _Timing:
    def __init__(self, world, rank, action, modes):
        self.world, self.rank, self.action, self.modes = world, rank, action, modes

    @cute.jit
    def __call__(
        self,
        peers: cute.Pointer,
        local: cute.Pointer,
        samples: cute.Pointer,
        invalid: cute.Pointer,
        profiles: cute.Pointer,
        selector: cute.Pointer,
        status: cute.Pointer,
        sample: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            peers, local, samples, invalid, profiles, selector, status, sample
        ).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.jit
    def barrier(self, peers: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        cute.arch.sync_threads()
        if tid < Int32(self.world):
            own, peer = Int64(peers[self.rank]), Int64(peers[tid])
            pcie_arrive_and_wait(
                own + Int64(tid) * Int64(4),
                peer + Int64(64 + self.rank * 128),
                own + Int64(64) + Int64(tid) * Int64(128),
                Int64(2048),
            )
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        peers: cute.Pointer,
        local: cute.Pointer,
        samples: cute.Pointer,
        invalid: cute.Pointer,
        profiles: cute.Pointer,
        selector: cute.Pointer,
        status: cute.Pointer,
        sample: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.action == "reset"):
            if tid == 0:
                local[2] = Int64(0)
                local[3] = Int64(0)
        elif cutlass.const_expr(self.action == "validate"):
            if tid == 0:
                if invalid[0] != 0:
                    local[2] = Int64(1)
        elif cutlass.const_expr(self.action == "begin"):
            self.barrier(peers)
            if tid == 0:
                local[0] = _globaltimer()
        elif cutlass.const_expr(self.action == "end"):
            if tid == 0:
                local[1] = _globaltimer() - local[0]
            self.barrier(peers)
            if tid == 0:
                maximum, bad = Int64(0), Int64(0)
                for rank in cutlass.range_constexpr(self.world):
                    remote = cute.make_ptr(
                        Int64,
                        Int64(peers[rank]) + Int64(SIGNAL_BYTES),
                        cute.AddressSpace.gmem,
                        assumed_align=8,
                    )
                    maximum = cutlass.max(maximum, remote[1])
                    bad = bad | remote[2]
                samples[sample] = maximum
                local[3] = local[3] | bad
            self.barrier(peers)
        else:
            if tid == 0:
                best, winner = Int64(0x7FFFFFFFFFFFFFFF), Int32(0)
                bad = local[3]
                for mode in cutlass.range_constexpr(self.modes):
                    values = cute.make_rmem_tensor((SAMPLES_PER_MODE,), Int64)
                    for i in cutlass.range_constexpr(SAMPLES_PER_MODE):
                        values[i] = samples[mode * SAMPLES_PER_MODE + i]
                        if values[i] <= 0:
                            bad = Int64(1)
                    for i in range(SAMPLES_PER_MODE):
                        for j in range(SAMPLES_PER_MODE - 1):
                            left, right = values[j], values[j + 1]
                            values[j] = cutlass.min(left, right)
                            values[j + 1] = cutlass.max(left, right)
                    median = values[SAMPLES_PER_MODE // 2]
                    if median < best:
                        best, winner = median, Int32(mode)
                if bad == 0:
                    selector[0] = profiles[winner]
                    status[0] = Int32(0)


@program_cache(scope="preparation")
def timing_launcher(world, rank, action, modes, ordinal):
    kernel = _Timing(world, rank, action, modes)
    dtypes = (cutlass.Uint64, Int64, Int64, Int32, Uint32, Uint32, Int32)
    raw = b12x_compile(
        kernel,
        *(_dummy(t, 8 if t in (cutlass.Uint64, Int64) else 4) for t in dtypes),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.joint_startup", 2, (world, rank, action, modes)
        ),
    )

    def run(peers, local, samples, invalid, profiles, selector, status, sample):
        addresses = (peers, local, samples, invalid, profiles, selector, status)
        raw(
            *(
                make_ptr(
                    t,
                    p,
                    cute.AddressSpace.gmem,
                    assumed_align=8 if t in (cutlass.Uint64, Int64) else 4,
                )
                for t, p in zip(dtypes, addresses)
            ),
            Int32(sample),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
