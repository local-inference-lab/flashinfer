from __future__ import annotations

import pytest
import torch

from benchmarks.experimental.b12x.common import require_sm120
from b12x.norm import mhc
from b12x.testing.mhc import make_inputs, post_reference, pre_reference
from ._mhc import bind, mhc_session, prepare
from .test_mhc_lagged import _lagged_reference


@pytest.mark.parametrize(
    "backend,unaligned_weights",
    [("native", False), ("tf32_fused", False), ("tf32_fused", True)],
)
@pytest.mark.parametrize("phase", ["pre", "post_pre"])
@pytest.mark.parametrize("rows", [1, 8, 17, 32, 128, 256])
def test_fused_tf32_graph(phase, rows, backend, unaligned_weights, mhc_session):
    device = require_sm120()
    hidden = 5120
    residual, x, fn, scale, bias = make_inputs(
        tokens=rows, hidden_size=hidden, seed=9137, device=device
    )
    if unaligned_weights:
        storage = torch.empty(fn.numel() + 1, dtype=fn.dtype, device=device)
        fn = storage[1:].view_as(fn).copy_(fn)
        assert fn.data_ptr() % 16 == 4
    incoming = (
        torch.tensor([0.75, -0.5, 0.25, 0.125], device=device)
        .expand(rows, -1)
        .contiguous()
    )
    norm = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    options = dict(
        pre_mix=incoming,
        norm_weight=norm,
        norm_eps=1e-20,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
    )
    _, prev_post, prev_comb = pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20
    )
    args = (
        (residual, fn, scale, bias)
        if phase == "pre"
        else (
            x,
            residual,
            prev_post.contiguous(),
            prev_comb.contiguous(),
            fn,
            scale,
            bias,
        )
    )
    plan = prepare(
        mhc_session,
        phase,
        args,
        options,
        backend=backend,
        lagged_prepare=backend == "native",
    )
    predicted = torch.empty_like(incoming)
    binding = bind(plan, pre_out=predicted)
    run = getattr(mhc, "run_" + phase)
    actual = run(*args, binding=binding, **options)
    torch.cuda.synchronize()
    mhc_session.freeze()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run(*args, binding=binding, **options)
    for factor in (1.0, 0.5):
        residual.mul_(factor)
        binding.partials.fill_(float("nan"))
        binding.y.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        expected_residual = (
            residual
            if phase == "pre"
            else post_reference(x, residual, prev_post, prev_comb)
        )
        torch.testing.assert_close(actual[0], expected_residual, rtol=0, atol=8e-3)
        post, comb, y, pre = _lagged_reference(
            actual[0], fn, scale, bias, incoming, norm
        )
        for got, expected in ((actual[1], post), (actual[2], comb), (predicted, pre)):
            torch.testing.assert_close(got, expected, rtol=2e-5, atol=4e-5)
        torch.testing.assert_close(actual[3], y, rtol=0, atol=8e-3)
