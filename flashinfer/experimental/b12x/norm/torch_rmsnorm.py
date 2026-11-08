# SPDX-License-Identifier: Apache-2.0
"""Caller-owned BF16 RMSNorm with Torch-compatible FP32 reduction order.

The numerical schedule follows compute_stats and vectorized_layer_norm_kernel_impl
in PyTorch revision 5c4886908584029761b579af026dcfb627c84070:
https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/layer_norm_kernel.cu
This implements the contiguous, four-element-aligned BF16 case only.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from b12x._lib.intrinsics.math import div_rn_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr


class TorchOrderNorm:
    def __init__(self, width):
        self.width = width

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        rows: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(source, weight, output, eps).launch(
            grid=(rows, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        output: cute.Pointer,
        eps: Float32,
    ):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        lane, warp = Int32(thread) % 32, Int32(thread) // 32
        base = Int64(row) * Int64(self.width)
        total = Float32(0.0)
        for vector in cutlass.range_constexpr((self.width // 4 + 127) // 128):
            first = (Int32(thread) + vector * 128) * 4
            if first < self.width:
                for element in cutlass.range_constexpr(4):
                    value = Float32(source[base + Int64(first + element)])
                    total = total + value * value
        for stage in cutlass.range_constexpr(5):
            total = total + cute.arch.shuffle_sync_down(total, offset=16 >> stage)
        allocator = cutlass.utils.SmemAllocator()
        partials = allocator.allocate_tensor(
            Float32, cute.make_layout((4,)), byte_alignment=16
        )
        for stage in cutlass.range_constexpr(2):
            offset = 2 >> stage
            if lane == 0 and warp >= offset and warp < 2 * offset:
                partials[warp - offset] = total
            cute.arch.sync_threads()
            if lane == 0 and warp < offset:
                total = total + partials[warp]
            cute.arch.sync_threads()
        if thread == 0:
            partials[0] = div_rn_f32(total, Float32(self.width))
        cute.arch.sync_threads()
        inverse = cute.math.rsqrt(Float32(partials[0]) + eps, fastmath=True)
        for vector in cutlass.range_constexpr((self.width // 4 + 127) // 128):
            first = (Int32(thread) + vector * 128) * 4
            if first < self.width:
                for element in cutlass.range_constexpr(4):
                    column = first + element
                    value = Float32(source[base + Int64(column)])
                    gain = Float32(weight[column])
                    output[base + Int64(column)] = BFloat16(gain * (inverse * value))


@program_cache(scope="preparation")
def _compile(width, device):
    entry = TorchOrderNorm(width)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=entry, cache_key=(width, device)
    )
    pointers = [
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=8)
        for _ in range(3)
    ]
    with torch.cuda.device(device):
        return compile_cute(
            entry,
            *pointers,
            Int32(1),
            Float32(1e-5),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key("norm.torch_rmsnorm", 1, (width,)),
        )


def _check(source, weight, output, eps):
    if (
        source.ndim != 2
        or not 0 < source.shape[0] < 2**31
        or not 0 < source.shape[1] <= 2**24
        or source.dtype != torch.bfloat16
        or weight.dtype != source.dtype
        or output.dtype != source.dtype
        or source.shape != output.shape
        or weight.shape != source.shape[1:]
        or source.shape[1] % 4
        or any(
            not tensor.is_cuda
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 8
            or tensor.device != source.device
            for tensor in (source, weight, output)
        )
    ):
        raise ValueError(
            "Torch-order RMSNorm requires aligned contiguous CUDA BF16 rows and matching weights/output"
        )
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("RMSNorm epsilon must be finite and nonnegative")
    a, b, w = source.data_ptr(), output.data_ptr(), weight.data_ptr()
    if (
        (a != b and a < b + output.nbytes and b < a + source.nbytes)
        or w < b + output.nbytes
        and b < w + weight.nbytes
    ):
        raise ValueError(
            "RMSNorm output may alias its input exactly, but not partially or overlap weights"
        )


def query_from_call(source, weight, *, out, eps):
    _check(source, weight, out, eps)
    return RMSNormQuery(hidden_size=source.shape[1])


def _run_state(state, source, weight, output, eps):
    _check(source, weight, output, eps)
    if source.shape[1] != state.query.hidden_size:
        raise ValueError("RMSNorm width differs from the prepared reduction")
    pointers = [
        make_ptr(BFloat16, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=8)
        for tensor in (source, weight, output)
    ]
    run_compiled(
        state.compiled,
        (*pointers, Int32(source.shape[0]), Float32(eps), current_cuda_stream()),
    )
    return output


@torch.library.custom_op("b12x::torch_rmsnorm", mutates_args=("output",))
def _torch_rmsnorm(
    source: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    eps: float,
    handle: int,
) -> None:
    from b12x.preparation.types import plan_from_handle, require_prepared

    state = require_prepared(
        plan_from_handle(handle), "norm.torch_rmsnorm", source.device
    )
    _run_state(state, source, weight, output, eps)


@_torch_rmsnorm.register_fake
def _fake_torch_rmsnorm(source, weight, output, eps, handle):
    pass


def run(source, weight, *, out, eps, plan):
    """Normalize contiguous BF16 rows; exact in-place output is supported.

    Accumulation and affine multiplication remain FP32 until the final BF16
    store. Epsilon is explicit; Torch's default for this path is FP32 epsilon.
    """
    torch.ops.b12x.torch_rmsnorm(source, weight, out, float(eps), plan.handle)
    return out


from ._torch_rmsnorm_preparation import RMSNormQuery, RMSNormState, TUNING, plan

__all__ = ["RMSNormQuery", "RMSNormState", "TUNING", "query_from_call", "plan", "run"]
