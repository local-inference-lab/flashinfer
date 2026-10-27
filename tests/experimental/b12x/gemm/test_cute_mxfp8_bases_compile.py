"""Contiguous MXFP8 bases remain valid through compiled dependent operations."""

import pytest
import torch

from b12x import gemm
from b12x.preparation import PreparationSession, PreparedCall
from b12x.quantization import mxfp8
from ..conftest import require_b12x


@pytest.mark.parametrize("rows", (1, 8))
@pytest.mark.parametrize("release_dependents", (False, True))
def test_floored_quantizer_then_dense_gemm_compiles_and_replays(
    rows, release_dependents
):
    device = require_b12x()
    k, n = 1280, 8192
    source = (torch.randn((rows, k), device=device) * 0.125).bfloat16()
    shapes = mxfp8.dense_rows_storage_shapes(rows, k)
    values = torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device)
    scale_rows = torch.empty(shapes[1], dtype=torch.uint8, device=device)
    scale_mma = torch.full(shapes[2], 127, dtype=torch.uint8, device=device)
    encoded = mxfp8.dense_rows_from_bases(values, scale_rows, scale_mma)
    weight = (torch.randn((n, k), device=device) * 0.125).to(torch.float8_e4m3fn)
    weight_shapes = mxfp8.dense_rows_storage_shapes(n, k)
    weight_rows = torch.full(weight_shapes[1], 127, dtype=torch.uint8, device=device)
    weight_mma = torch.full(weight_shapes[2], 127, dtype=torch.uint8, device=device)
    packed = mxfp8.dense_rows_from_bases(weight, weight_rows, weight_mma)
    weight_base = mxfp8.dense_scale_storage(packed.scale_mma)
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    workspace = torch.empty((0,), dtype=torch.uint8, device=device)

    quant = mxfp8.plan(
        mxfp8.query_from_call(
            source,
            encoded.values,
            encoded.scale_rows,
            encoded.scale_mma,
            min_amax=1e-4,
            release_dependents=release_dependents,
        )
    )
    dense = gemm.plan(
        gemm.DenseGemmQuery(
            recipe="mxfp8",
            entry_point="gemm.mm",
            weight_storage="native",
            output_dtype="bfloat16",
            batch=1,
            max_rows=rows,
            in_features=k,
            out_features=n,
            output_mode="provided",
            alpha_mode="unit",
            workspace_form="provided",
            workspace_nbytes=0,
        ),
        override=gemm.DenseGemmConfig(
            backend="cutedsl",
            tile_m=16,
            tile_n=64,
            tile_k=128,
            load_path="tma",
            swap_ab=False,
            split_k_slices=1,
            large_m_unroll=True,
            target_occupancy=None,
        ),
    )

    def quant_call(state):
        return PreparedCall(
            run=lambda: state.run(
                source, encoded.values, encoded.scale_rows, encoded.scale_mma
            ),
            owners=(values, scale_rows, scale_mma),
        )

    def dense_call(state):
        values.zero_()
        scale_mma.fill_(127)
        return PreparedCall(
            run=lambda: state.run(
                values.view(rows, k, 1),
                encoded.scale_mma,
                weight.view(n, k, 1),
                packed.scale_mma,
                None,
                out.view(rows, n, 1),
                workspace.view(torch.float32),
                torch.bfloat16,
                None,
            ),
            owners=(values, scale_mma, weight, weight_mma, out, workspace),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (
                quant.request(name="compiled-rows", prepare_call=quant_call),
                dense.request(name="compiled-dense", prepare_call=dense_call),
            )
        )

        def forward():
            mxfp8.quantize_dense_rows(source, values, scale_rows, scale_mma, plan=quant)
            return gemm.mxfp8_mm_bases(
                values, scale_mma, weight, weight_base, out, workspace, plan=dense
            )

        eager = forward().clone()
        compiled = torch.compile(forward, fullgraph=True)
        actual = compiled()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture():
            with torch.cuda.graph(graph):
                compiled()
        session.freeze()
        for factor in (0.5, 1.5):
            source.mul_(factor)
            expected = forward().clone()
            values.view(torch.uint8).fill_(255)
            scale_mma.fill_(255)
            out.fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[key] == after[key]
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        graph.reset()
