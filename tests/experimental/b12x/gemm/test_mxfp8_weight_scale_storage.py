"""Immutable MXFP8 matrix weights need one scale layout, without requantization."""

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm._shared.wo_mxfp8 import (
    quantize_mxfp8_rows_torch,
    materialize_mxfp8_scale_rows,
    _check_mxfp8_rows_storage,
)
from b12x.gemm.blockscaled._linear import MXFP8LinearWeight
from b12x.preparation import PreparationSession, PreparedCall
from b12x.gemm.blockscaled._a16 import _weight_parts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires assigned CUDA GPU"
)


@pytest.mark.parametrize("shape", [(33, 128), (360, 7168), (145, 256, 3)])
@torch.no_grad()
def test_scale_layout_omission_and_exact_reconstruction(shape):
    source = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    full = quantize_mxfp8_rows_torch(source)
    compact = quantize_mxfp8_rows_torch(source, retain_scale_rows=False)
    assert compact.scale_rows is None
    assert torch.equal(full.values.view(torch.uint8), compact.values.view(torch.uint8))
    assert torch.equal(
        full.scale_mma.view(torch.uint8), compact.scale_mma.view(torch.uint8)
    )
    assert materialize_mxfp8_scale_rows(full) is full.scale_rows
    restored = materialize_mxfp8_scale_rows(compact)
    assert torch.equal(restored.view(torch.uint8), full.scale_rows.view(torch.uint8))
    with pytest.raises(ValueError, match="destinations require compact"):
        _check_mxfp8_rows_storage(
            compact,
            m=shape[0],
            k=shape[1],
            num_groups=1 if len(shape) == 2 else shape[2],
        )


@pytest.mark.parametrize("rows", [1, 8, 32, 128, 3072])
@torch.no_grad()
def test_a16_graph_does_not_require_redundant_compact_weight_scales(rows):
    source = torch.randn((rows, 1536), device="cuda", dtype=torch.bfloat16)
    raw = torch.randn((256, 1536), device="cuda", dtype=torch.bfloat16)
    weights = [
        MXFP8LinearWeight(
            quantize_mxfp8_rows_torch(raw, retain_scale_rows=keep), 1536, 1536, 256
        )
        for keep in (True, False)
    ]
    outputs = [
        torch.empty((rows, 256), device="cuda", dtype=torch.bfloat16) for _ in weights
    ]
    workspace = torch.empty(32 << 20, device="cuda", dtype=torch.uint8)
    with PreparationSession(
        device="cuda:0", autotune=False, compile_workers=1
    ) as session:
        plans = []
        requests = []
        for i, (weight, output) in enumerate(zip(weights, outputs)):
            plan = blockscaled.plan(
                blockscaled.query_from_call(
                    source,
                    weight,
                    out=output,
                    workspace=workspace,
                    activation_mode="a16",
                )
            )
            values, scales, gain, _ = _weight_parts(weight)

            def prepare(state, values=values, scales=scales, gain=gain, output=output):
                return PreparedCall(
                    run=lambda: state.run(
                        source, values, scales, gain, out=output, workspace=workspace
                    )
                )

            requests.append(plan.request(name=f"weight{i}", prepare_call=prepare))
            plans.append(plan)
        session.prepare(requests)

        def run():
            for weight, output, plan in zip(weights, outputs, plans):
                blockscaled.mm(
                    source, weight, out=output, workspace=workspace, plan=plan
                )

        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for multiplier in (1.0, -0.5, 2.0, 0.0):
            source.mul_(multiplier)
            for output in outputs:
                output.fill_(float("nan"))
            workspace.fill_(165)
            before = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            assert torch.equal(
                outputs[0].view(torch.int16), outputs[1].view(torch.int16)
            )
        graph.reset()
