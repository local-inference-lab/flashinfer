"""Lagged fused mHC without RMSNorm preserves live coefficient propagation."""

import pytest
import torch

from benchmarks.experimental.b12x.common import require_sm120
from b12x.norm import mhc
from b12x.testing.mhc import make_inputs, post_reference, pre_reference
from ._mhc import bind, mhc_session, prepare
from .test_mhc_lagged import _lagged_reference


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("rows", [1, 8, 17, 128])
@pytest.mark.parametrize("tile_m", [16, 32])
@pytest.mark.parametrize("operation", ["pre", "post_pre"])
def test_fused_without_norm_graph(hidden, rows, operation, tile_m, mhc_session):
    device = require_sm120()
    residual, x, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=92150, device=device
    )
    incoming = torch.tensor([0.75, -0.5, 0.25, 0.125], device=device)
    incoming = incoming.expand(rows, -1).contiguous()
    options = dict(pre_mix=incoming, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
    _, post, comb = pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    args = (
        (residual, fn, scale, bias)
        if operation == "pre"
        else (x, residual, post, comb, fn, scale, bias)
    )
    plan = prepare(
        mhc_session, operation, args, options, backend="tf32_fused", fused_tile_m=tile_m
    )
    predicted = torch.empty_like(incoming)
    binding = bind(plan, pre_out=predicted)
    run = getattr(mhc, "run_" + operation)
    run(*args, binding=binding, **options)
    mhc_session.freeze()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(*args, binding=binding, **options)
    for factor in (1.0, 0.5, 2.0):
        residual.mul_(factor)
        incoming.copy_(incoming.roll(1, 1))
        fn.mul_(1.125)
        binding.partials.fill_(float("nan"))
        binding.y.fill_(float("nan"))
        predicted.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected_residual = (
            residual if operation == "pre" else post_reference(x, residual, post, comb)
        )
        torch.testing.assert_close(output[0], expected_residual, rtol=0, atol=0.008)
        expected_post, expected_comb, expected_y, expected_pre = _lagged_reference(
            output[0], fn, scale, bias, incoming, None
        )
        for actual, expected in (
            (output[1], expected_post),
            (output[2], expected_comb),
            (predicted, expected_pre),
        ):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=4e-5)
        torch.testing.assert_close(output[3], expected_y, rtol=0, atol=0.008)
