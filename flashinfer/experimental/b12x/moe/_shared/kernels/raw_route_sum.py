"""Fixed-order FP32 FC2 slice reduction into caller-owned BF16 routes."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr


class _RawRouteSum:
    def __init__(self, hidden, slices):
        self.hidden, self.slices = hidden, slices

    @cute.jit
    def __call__(
        self,
        parts: cute.Pointer,
        ids: cute.Pointer,
        output: cute.Pointer,
        routes: Int32,
        experts: Int32,
        stream: cuda.CUstream,
    ):
        layout = cute.make_layout((Int64(1) << Int64(50),))
        parts = cute.make_tensor(parts, layout)
        ids = cute.make_tensor(ids, layout)
        output = cute.make_tensor(output, layout)
        self.kernel(parts, ids, output, routes, experts).launch(
            grid=(
                (Int64(routes) * Int64(self.hidden) + Int64(255)) // Int64(256),
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        parts: cute.Tensor,
        ids: cute.Tensor,
        output: cute.Tensor,
        routes: Int32,
        experts: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        index = Int64(bid) * Int64(256) + Int64(tid)
        if index < Int64(routes) * Int64(self.hidden):
            route = index // Int64(self.hidden)
            expert = ids[route].to(Int64)
            total = Float32(0.0)
            if expert >= Int64(0) and expert < Int64(experts):
                for plane in cutlass.range_constexpr(self.slices):
                    offset = Int64(plane) * Int64(routes) * Int64(self.hidden) + Int64(
                        index
                    )
                    total = total + parts[offset]
            output[index] = total.to(cutlass.BFloat16)


@program_cache
def compile_raw_route_sum(ordinal, hidden, slices, ids_dtype):
    """Retain one reduction launcher for dynamic live route counts."""
    id_type = cutlass.Int32 if ids_dtype == torch.int32 else cutlass.Int64
    id_align = 4 if ids_dtype == torch.int32 else 8
    kernel = _RawRouteSum(hidden, slices)
    key = hidden, slices, str(ids_dtype)
    with torch.cuda.device(ordinal):

        def pointer(dtype, alignment):
            return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)

        raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=key)
        raw = b12x_compile(
            kernel,
            pointer(Float32, 16),
            pointer(id_type, id_align),
            pointer(cutlass.BFloat16, 16),
            Int32(1),
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("moe.raw_route_sum", 1, key),
        )

    def run(parts, ids, output, routes, experts):
        def pointer(value, dtype, alignment):
            return make_ptr(
                dtype, value.data_ptr(), cute.AddressSpace.gmem, assumed_align=alignment
            )

        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    pointer(parts, Float32, 16),
                    pointer(ids, id_type, id_align),
                    pointer(output, cutlass.BFloat16, 16),
                    Int32(routes),
                    Int32(experts),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)
