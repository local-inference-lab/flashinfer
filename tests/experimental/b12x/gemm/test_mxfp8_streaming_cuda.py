"""Prepared CUDA projection preserves encoded GEMM through capture and replay."""

from dataclasses import asdict

import pytest
import torch

from b12x import gemm
from b12x._lib.cuda_program import cuda_compile_count
from b12x._lib.intrinsics.reference import as_grouped_scale_view_mx
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x


def streaming_config(pdl, evict):
    return gemm.DenseGemmConfig(
        backend="cuda_streaming",
        tile_m=16,
        tile_n=8,
        tile_k=128,
        load_path="streaming",
        swap_ab=False,
        split_k_slices=1,
        large_m_unroll=False,
        target_occupancy=None,
        programmatic_serialization=pdl,
        evict_first=evict,
    )


def make_case(rows, columns, inner, device):
    generator = torch.Generator(device=device).manual_seed(1709)

    def values(count):
        return torch.randn((count, inner, 1), device=device, generator=generator).to(
            torch.float8_e4m3fn
        )

    a, b = values(rows), values(columns)

    def scales(count):
        storage = torch.full(
            (1, ((count + 127) // 128) * (inner // 128) * 512),
            127,
            dtype=torch.uint8,
            device=device,
        )
        return as_grouped_scale_view_mx(storage, count, inner)

    sa, sb = scales(rows), scales(columns)
    output = torch.empty((rows, columns, 1), device=device, dtype=torch.bfloat16)
    query = gemm.DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=rows,
        in_features=inner,
        out_features=columns,
        output_mode="provided",
        alpha_mode="unit",
        scale_domain="bounded_fp32_products",
        rhs_prefetch_ready=True,
    )
    return query, a, sa, b, sb, output


@pytest.mark.parametrize("pdl,evict", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize(
    "rows,columns,inner", [(8, 1792, 5120), (7, 9, 128), (1, 1, 256)]
)
def test_prepared_encoded_projection_replays_mutable_inputs(
    pdl, evict, rows, columns, inner
):
    device = require_b12x()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("streaming CUDA projection is qualified on SM120")
    query, a, sa, b, sb, output = make_case(rows, columns, inner, device)
    declaration = gemm.plan(query, override=streaming_config(pdl, evict))
    request = declaration.request(
        name="encoded-projection",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                a, sa, b, sb, None, output, None, torch.bfloat16, None
            )
        ),
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        result = session.prepare((request,))
        assert result.program_counts["cuda"] == 1
        print(
            {
                "program_counts": dict(result.program_counts),
                "compilation": None
                if result.compilation is None
                else asdict(result.compilation),
            },
            flush=True,
        )

        def forward(a, sa, b, sb, out):
            return gemm.mm((a, sa), (b, sb), out=out, plan=declaration)

        forward(a, sa, b, sb, output)
        compiled = torch.compile(forward, fullgraph=True)
        compiled(a, sa, b, sb, output)
        torch.cuda.synchronize()
        expected = (a[:, :, 0].float() @ b[:, :, 0].float().T).bfloat16()
        torch.testing.assert_close(output[:, :, 0], expected, rtol=0.01, atol=0.0625)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                compiled(a, sa, b, sb, output)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compiled(a, sa, b, sb, output)
        torch.cuda.synchronize()
        compiled_count = cuda_compile_count()
        allocated = torch.cuda.memory_allocated()
        for iteration in range(4):
            a.view(torch.uint8).zero_() if iteration % 2 == 0 else a.fill_(1)
            output.fill_(float("nan"))
            with kernel_resolution_guard("native graph replay"):
                graph.replay()
            torch.cuda.synchronize()
            expected = (a[:, :, 0].float() @ b[:, :, 0].float().T).bfloat16()
            torch.testing.assert_close(
                output[:, :, 0], expected, rtol=0.01, atol=0.0625
            )
        del expected
        assert cuda_compile_count() == compiled_count
        assert torch.cuda.memory_allocated() <= allocated


def test_scale_domain_boundaries_and_zero_blocks():
    """Bounded scales agree with native MMA; excluded scales retain their guard."""
    from dataclasses import replace
    from b12x.preparation import require_prepared
    from b12x.preparation.device import detect_device
    from b12x.gemm._tuning import TUNING
    from b12x._lib.utils.device import current_cuda_stream

    device = require_b12x()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("streaming CUDA projection is qualified on SM120")
    query, a, sa, b, sb, cuda_output = make_case(8, 1792, 5120, device)
    cute_output = torch.empty_like(cuda_output)
    cuda_plan = gemm.plan(query, override=streaming_config(True, True))
    cute_query = replace(query, scale_domain="ue8m0")
    # Select each arithmetic path explicitly for correctness, without a race.
    cute_config = TUNING.default_config(cute_query, detect_device(device).identity)
    cute_plan = gemm.plan(cute_query, override=cute_config)

    def request(name, declaration, output):
        return declaration.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    a, sa, b, sb, None, output, None, torch.bfloat16, None
                )
            ),
        )

    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        result = session.prepare(
            (
                request("cuda", cuda_plan, cuda_output),
                request("cutedsl", cute_plan, cute_output),
            )
        )
        print(
            {
                "program_counts": dict(result.program_counts),
                "compilation": None
                if result.compilation is None
                else asdict(result.compilation),
            },
            flush=True,
        )

        def forward(a, sa, b, sb, cuda_out, cute_out):
            gemm.mm((a, sa), (b, sb), out=cuda_out, plan=cuda_plan)
            gemm.mm((a, sa), (b, sb), out=cute_out, plan=cute_plan)

        compiled = torch.compile(forward, fullgraph=True)
        compiled(a, sa, b, sb, cuda_output, cute_output)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(a, sa, b, sb, cuda_output, cute_output)
        for ae, be, av, bv in [
            (64, 64, 1.0, 1.0),
            (190, 190, 2**-6, 2**-6),
            (64, 190, 1.0, 1.0),
            (190, 64, 1.0, 1.0),
            (0, 190, 0.0, 1.0),
            (190, 0, 1.0, 0.0),
            (0, 0, 0.0, 0.0),
        ]:
            a.fill_(av)
            b.fill_(bv)
            sa.view(torch.uint8).fill_(ae)
            sb.view(torch.uint8).fill_(be)
            cuda_output.fill_(float("nan"))
            cute_output.fill_(float("nan"))
            with kernel_resolution_guard("scale-boundary graph replay"):
                graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.isfinite(cuda_output).all() and torch.isfinite(cute_output).all()
            )
            assert torch.equal(cuda_output, cute_output), (ae, be)
            print(
                {
                    "scale_bytes": [ae, be],
                    "bitwise": True,
                    "value": cuda_output.flatten()[0].item(),
                },
                flush=True,
            )
        # The bounded declaration is deliberately bypassed only for this raw
        # counterexample. The ordinary full-domain plan remains the comparator.
        a.zero_()
        b.zero_()
        sa.view(torch.uint8).fill_(254)
        sb.view(torch.uint8).fill_(254)
        state = require_prepared(cuda_plan, "gemm.mm", a.device)
        state.program(
            a,
            sa,
            b,
            sb,
            cuda_output,
            8,
            1792,
            5120,
            grid=(224, 1, 1),
            stream=int(current_cuda_stream()),
        )
        gemm.mm((a, sa), (b, sb), out=cute_output, plan=cute_plan)
        torch.cuda.synchronize()
        print(
            {
                "excluded_scale_bytes": [254, 254],
                "raw_cuda_all_nan": bool(torch.isnan(cuda_output).all()),
                "cute_all_zero": bool(torch.count_nonzero(cute_output) == 0),
            },
            flush=True,
        )
        assert torch.isnan(cuda_output).all()
        assert torch.count_nonzero(cute_output) == 0


def test_mutable_rhs_written_after_programmatic_release_keeps_dependency():
    """The ordinary candidate waits for a producer of values and RHS scales."""
    import triton
    import triton.language as tl
    from dataclasses import replace

    device = require_b12x()
    if torch.cuda.get_device_capability(device) != (12, 0):
        pytest.skip("streaming CUDA projection is qualified on SM120")

    @triton.jit
    def producer(A, B, SF, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        x = tl.program_id(0) * 256 + tl.arange(0, 256)
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
        tl.store(A + x, 1.0, x < M * K)
        tl.store(B + x, 1.0, x < N * K)
        tl.store(SF + x, 128, x < 512)

    query, a, sa, b, sb, output = make_case(8, 128, 128, device)
    query = replace(query, rhs_prefetch_ready=False)
    declaration = gemm.plan(query, override=streaming_config(False, False))
    request = declaration.request(
        name="mutable-rhs",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                a, sa, b, sb, None, output, None, torch.bfloat16, None
            )
        ),
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))
        program = producer.warmup(
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.uint8,
            8,
            128,
            128,
            grid=(64, 1, 1),
            num_warps=4,
            launch_pdl=True,
        )

        def forward():
            program[(64, 1, 1)](a, b, sb.view(torch.uint8), 8, 128, 128)
            gemm.mm((a, sa), (b, sb), out=output, plan=declaration)

        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        for _ in range(4):
            a.view(torch.uint8).zero_()
            b.view(torch.uint8).zero_()
            sb.view(torch.uint8).fill_(127)
            output.zero_()
            with kernel_resolution_guard("mutable-RHS producer replay"):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, torch.full_like(output, 256))
