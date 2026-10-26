"""Device-only CUDA compilation and retained streaming-GEMM execution."""

from dataclasses import dataclass
from pathlib import Path

import torch

from b12x._lib.cuda_program import CudaCompileSpec, CudaKernelSpec, compile_cuda
from b12x._lib.compile_plan import attach_programs, load_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream


def source_spec(config, *, cuda_home=None):
    if cuda_home is None:
        from flashinfer.jit.cpp_ext import get_cuda_path

        cuda_home = get_cuda_path()
    root = Path(cuda_home)
    pdl, cache = int(config.programmatic_serialization), int(config.evict_first)
    return CudaCompileSpec(
        source=str(Path(__file__).parent / "csrc/projection.cu"),
        nvcc=str(root / "bin/nvcc"),
        architecture="sm_120",
        kernel=CudaKernelSpec(
            f"b12x_mxfp8_stream_p{pdl}_c{cache}",
            ("ptr", "ptr", "ptr", "ptr", "ptr", "i32", "i32", "i32"),
            (256, 1, 1),
            17408,
            bool(pdl),
        ),
        options=("-O3", "--std=c++20", "--use_fast_math"),
        include_dirs=(str(root / "include"),),
    )


@program_cache(scope="preparation")
def compile_streaming(spec):
    return compile_cuda(spec)


@dataclass(frozen=True)
class PreparedStreamingDense:
    query: object
    device: torch.device
    program: object

    @property
    def required_workspace(self):
        return 0

    @property
    def workspace(self):
        return None

    def run(self, a, sfa, b, sfb, alpha, out, workspace, output_dtype, stream):
        q = self.query
        if alpha is not None:
            raise ValueError("streaming MXFP8 requires the declared unit alpha")
        if output_dtype != torch.bfloat16 or out is None:
            raise ValueError("streaming MXFP8 requires caller-owned BF16 output")
        m = int(a.shape[0])
        k = q.in_features
        n = q.out_features
        for value, shape, dtype in [
            (a, (m, k, 1), torch.float8_e4m3fn),
            (b, (n, k, 1), torch.float8_e4m3fn),
            (out, (m, n, 1), torch.bfloat16),
        ]:
            if (
                value.device != self.device
                or tuple(value.shape) != shape
                or value.dtype != dtype
                or not value.is_contiguous()
                or value.data_ptr() % (2 if dtype == torch.bfloat16 else 16)
            ):
                raise ValueError(
                    "streaming MXFP8 operand differs from prepared values/output layout"
                )
        if not 1 <= m <= q.max_rows:
            raise ValueError("streaming MXFP8 row count exceeds its declaration")
        for scales, rows in [(sfa, m), (sfb, n)]:
            rb = (rows + 127) // 128
            kb = (k // 32 + 3) // 4
            if (
                scales.device != self.device
                or scales.dtype not in (torch.uint8, torch.float8_e8m0fnu)
                or scales.data_ptr() % 4
                or tuple(scales.shape) != (32, 4, rb, 4, kb, 1)
                or tuple(scales.stride()) != (16, 4, kb * 512, 1, 512, rb * kb * 512)
            ):
                raise ValueError(
                    "streaming MXFP8 scales require the declared F8_128x4 layout"
                )
        stream = current_cuda_stream() if stream is None else stream
        self.program(
            a, sfa, b, sfb, out, m, n, k, grid=((n + 7) // 8, 1, 1), stream=int(stream)
        )
        return out


def materialize(query, config, device):
    program = compile_streaming(source_spec(config))
    with torch.cuda.device(device.ordinal):
        load_programs(program)
    return attach_programs(
        PreparedStreamingDense(query, torch.device("cuda", device.ordinal), program),
        program,
    )
