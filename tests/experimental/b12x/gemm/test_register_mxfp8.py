"""Register-staged MXFP8 keeps quantized operands and bounded row capacity."""

import pytest
import torch
import cutlass.cute as cute
from cutlass import BFloat16, Int32, Uint8
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x._lib.intrinsics.reference import (
    quantize_grouped_mxfp8_torch,
    quant_dequant_mxfp8_torch,
)
from b12x.gemm.blockscaled._register_mxfp8 import compile_register
from tests.experimental.b12x.conftest import require_b12x


@pytest.mark.parametrize("k,n", [(1280, 128), (1280, 8192), (5120, 128)])
@pytest.mark.parametrize("m", [1, 3, 8])
def test_register_mxfp8_quantized_products(k, n, m):
    device = require_b12x()
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    w = torch.randn((n, k), device=device, dtype=torch.bfloat16)
    rows = torch.tensor([m], device=device, dtype=torch.int32)
    xp, xs = quantize_grouped_mxfp8_torch(x[None], rows)
    wp, ws = quantize_grouped_mxfp8_torch(
        w[None], torch.tensor([n], device=device, dtype=torch.int32)
    )
    y = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    program = compile_register(k, n)
    tensors = (xp, xs, wp, ws, y)
    types = (Uint8, Uint8, Uint8, Uint8, BFloat16)

    def run():
        program(
            *(
                make_ptr(t, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                for a, t in zip(tensors, types)
            ),
            Int32(m),
            current_cuda_stream(),
        )

    run()
    ref = quant_dequant_mxfp8_torch(x).float() @ quant_dequant_mxfp8_torch(w).float().T
    torch.testing.assert_close(y, ref.bfloat16(), atol=0.015625, rtol=0.015625)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        y.fill_(float("nan"))
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, ref.bfloat16(), atol=0.015625, rtol=0.015625)


@pytest.mark.parametrize("m", [1, 3, 8])
@pytest.mark.parametrize("provided", [False, True])
def test_prepared_register_mxfp8_replays(m, provided):
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._tuning import BlockscaledConfig
    from b12x._lib.compile_plan import program_keys
    from tests.experimental.b12x.gemm.test_blockscaled_a16 import (
        make_weight,
        make_workspace,
        assert_close,
    )

    device = require_b12x()
    k, n = 1280, 128
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    weight, unpacked, _ = make_weight("mxfp8", n, k, device)
    config = BlockscaledConfig(mode="quantized", quantized_backend="register_rows")
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16) if provided else None
    workspace = (
        make_workspace(x, weight, activation_mode="auto", out=out, config=config)
        if provided
        else None
    )
    from tests.experimental.b12x.gemm._blockscaled import prepared

    with prepared(
        x, weight, activation_mode="auto", out=out, workspace=workspace, override=config
    ) as plan:

        def run():
            return blockscaled.mm(x, weight, plan=plan, out=out, workspace=workspace)

        actual = run()
        expected = quant_dequant_mxfp8_torch(x).float() @ unpacked.T
        assert_close(actual, expected.bfloat16())
        from b12x.preparation.types import require_prepared

        state = require_prepared(plan, "gemm.blockscaled_precision", x.device)
        assert any(key.dialect == "cute" for key in program_keys(state.programs))
        if provided:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            for _ in range(3):
                workspace.fill_(37)
                out.fill_(float("nan"))
                graph.replay()
            torch.cuda.synchronize()
            assert_close(out, expected.bfloat16())
