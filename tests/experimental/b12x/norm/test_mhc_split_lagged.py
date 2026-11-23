"""CUDA tests for SM120 split-lagged mHC, prequantized compact W4A8 MoE, and BF16 TC-GEMV split_k=5,10."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.compile_plan import load_programs
from b12x._lib.quant.mxfp8_rows import quantize_mxfp8_rows_cute
from b12x.gemm import bf16_gemv
from b12x.gemm.bf16_gemv._tuning import (
    GemvConfig,
    GemvQuery,
    _tc_split_k_candidates,
)
from b12x.moe import fused_moe
from b12x.moe.fused_moe._binding_ops import (
    prepared_moe_out,
    prepared_moe_prequant_out,
    supports_prequant_mxfp8,
)
from b12x.norm import mhc
from b12x.norm.mhc import _impl as mhc_impl
from b12x.norm.mhc._split_lagged import (
    compile_cluster_post_y_norm,
    compile_readonly_fn_sinkhorn,
    split_lagged_partials_shape,
    supports_split_lagged,
)
from b12x.preparation import PreparationSession, PreparedCall


def _sm120_required():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        pytest.skip("SM120+ required for cluster split-lagged mHC")


@pytest.mark.parametrize("hidden", [4096, 5120, 7168])
@pytest.mark.parametrize("tokens", [1, 4, 7, 8])
@pytest.mark.parametrize("pre_only", [False, True])
@pytest.mark.parametrize("norm_dtype", [torch.bfloat16, torch.float32])
def test_split_lagged_mhc_matches_reference_and_emits_bit_exact_mxfp8(
    hidden: int, tokens: int, pre_only: bool, norm_dtype: torch.dtype
):
    _sm120_required()
    if hidden != 5120 and (tokens not in (1, 8) or norm_dtype == torch.float32):
        pytest.skip("reduced matrix for non-5120 hidden sizes")

    device = torch.device("cuda", torch.cuda.current_device())
    assert supports_split_lagged(device, tokens, hidden, norm_dtype)

    torch.manual_seed(1234 + hidden + tokens)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device) * 0.5
    prev_post = torch.rand(tokens, 4, dtype=torch.float32, device=device)
    prev_comb = torch.softmax(
        torch.randn(tokens, 4, 4, dtype=torch.float32, device=device), dim=-1
    )
    fn = torch.randn(24, 4 * hidden, dtype=torch.float32, device=device) * 0.02
    pre_mix = torch.softmax(
        torch.randn(tokens, 4, dtype=torch.float32, device=device), dim=-1
    )
    scale = torch.tensor([0.1, 0.1, 0.1], dtype=torch.float32, device=device)
    bias = torch.randn(24, dtype=torch.float32, device=device) * 0.1
    norm_weight = (
        torch.randn(hidden, dtype=norm_dtype, device=device) * 0.2 + 1.0
    ).contiguous()

    op = "pre" if pre_only else "post_pre"
    inv = {
        "operation": op,
        "output_mode": "provided",
        "has_norm_weight": True,
        "norm_weight_dtype": "bfloat16" if norm_dtype == torch.bfloat16 else "float32",
        "expanded_residual": pre_only,
        "rms_eps": 1e-20,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "norm_eps": 1e-20,
        "lagged_mix": True,
    }
    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden), invocation=inv
    )
    session = PreparationSession(compile_workers=0)
    out_ref = torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device)
    post_ref = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    comb_ref = torch.empty((tokens, 4, 4), dtype=torch.float32, device=device)
    y_ref = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    pre_ref = torch.empty((tokens, 4), dtype=torch.float32, device=device)
    entry = getattr(mhc_impl, f"_b12x_mhc_{op}_impl")
    inputs = (residual,) if pre_only else (x, residual, prev_post, prev_comb)

    def prepare(state):
        scratch = [
            torch.empty(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch_specs()
        ]
        binding = state.bind(
            scratch=scratch,
            tokens=tokens,
            pre_out=pre_ref,
            out=out_ref,
            post=post_ref,
            comb=comb_ref,
            y=y_ref,
        )
        return PreparedCall(
            run=lambda: entry(
                *inputs,
                fn,
                scale,
                bias,
                rms_eps=1e-20,
                hc_eps=1e-6,
                sinkhorn_iters=20,
                norm_weight=norm_weight,
                norm_eps=1e-20,
                pre_mix=pre_mix,
                binding=binding,
                _state=state,
            )
        )

    session.prepare((plan.request(name="ref", prepare_call=prepare),))
    state = plan._prepared.state
    scratch = [
        torch.empty(spec.nbytes, dtype=spec.dtype, device=device)
        for spec in state.scratch_specs()
    ]
    binding = state.bind(
        scratch=scratch,
        tokens=tokens,
        pre_out=pre_ref,
        out=out_ref,
        post=post_ref,
        comb=comb_ref,
        y=y_ref,
    )
    entry(
        *inputs,
        fn,
        scale,
        bias,
        rms_eps=1e-20,
        hc_eps=1e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1e-20,
        pre_mix=pre_mix,
        binding=binding,
        _state=state,
    )

    ordinal = int(device.index or 0)
    post_y_runner = compile_cluster_post_y_norm(
        ordinal, hidden, pre_only, 1e-20, norm_dtype, emit_mxfp8=True
    )
    fn_sink_runner = compile_readonly_fn_sinkhorn(ordinal, hidden, 1e-20, 1e-6, 20)
    load_programs(post_y_runner)
    load_programs(fn_sink_runner)

    out_new = torch.empty_like(out_ref)
    y_new = torch.empty_like(y_ref)
    a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    a_scales = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)
    partials = torch.empty(
        split_lagged_partials_shape(tokens, hidden), dtype=torch.float32, device=device
    )
    pre_new = torch.empty_like(pre_ref)
    post_new = torch.empty_like(post_ref)
    comb_new = torch.empty_like(comb_ref)

    post_y_runner(
        None if pre_only else x,
        residual,
        None if pre_only else prev_post,
        None if pre_only else prev_comb,
        pre_mix,
        norm_weight,
        out_new,
        y_new,
        a_values=a_values,
        a_scales=a_scales,
    )
    fn_sink_runner(out_new, fn, scale, bias, partials, pre_new, post_new, comb_new)

    av_ref = torch.empty_like(a_values)
    as_ref = torch.empty_like(a_scales)
    am_ref = torch.empty(
        ((tokens + 127) // 128 * ((hidden + 127) // 128) * 512,),
        dtype=torch.uint8,
        device=device,
    )
    quantize_mxfp8_rows_cute(y_new, av_ref, as_ref, am_ref, expected_m=tokens)
    torch.cuda.synchronize()

    torch.testing.assert_close(out_new.float(), out_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(y_new.float(), y_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(pre_new, pre_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(post_new, post_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(comb_new, comb_ref, atol=1e-5, rtol=1e-5)
    assert torch.equal(a_values, av_ref)
    assert torch.equal(a_scales, as_ref)


@pytest.mark.parametrize("tokens", [1, 4, 8])
def test_compact_micro_moe_prequant_mxfp8_is_bit_exact(tokens: int):
    _sm120_required()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(404 + tokens)
    num_experts, hidden, inter, top_k = 64, 5120, 192, 6
    w1_fp4 = torch.randint(
        0, 256, (num_experts, 2 * inter, hidden // 2), dtype=torch.uint8, device=device
    )
    w2_fp4 = torch.randint(
        0, 256, (num_experts, hidden, inter // 2), dtype=torch.uint8, device=device
    )
    w1_sc = torch.randint(
        124,
        130,
        (num_experts, 2 * inter, hidden // 32),
        dtype=torch.uint8,
        device=device,
    )
    w2_sc = torch.randint(
        124, 130, (num_experts, hidden, inter // 32), dtype=torch.uint8, device=device
    )
    ones = torch.ones((num_experts,), dtype=torch.float32, device=device)

    wplan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=num_experts, hidden_size=hidden, intermediate_size=inter
        ),
    )
    experts = fused_moe.prepare_weights(
        plan=wplan,
        weights=fused_moe.PackedWeights(
            w13=w1_fp4,
            w2=w2_fp4,
            w13_block_scales=w1_sc,
            w2_block_scales=w2_sc,
            w13_global_scales=ones,
            w2_global_scales=ones,
        ),
    )

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(tokens, 4, hidden, dtype=torch.bfloat16, device=device) * 0.5
    prev_post = torch.rand(tokens, 4, dtype=torch.float32, device=device)
    prev_comb = torch.softmax(
        torch.randn(tokens, 4, 4, dtype=torch.float32, device=device), dim=-1
    )
    pre_mix = torch.softmax(
        torch.randn(tokens, 4, dtype=torch.float32, device=device), dim=-1
    )
    norm_w = (
        torch.randn(hidden, dtype=torch.bfloat16, device=device) * 0.2 + 1.0
    ).contiguous()
    res_out = torch.empty_like(residual)
    y_norm = torch.empty_like(x)
    a_values = torch.empty((tokens, hidden), dtype=torch.uint8, device=device)
    a_scales = torch.empty((tokens, hidden // 32), dtype=torch.uint8, device=device)

    runner = compile_cluster_post_y_norm(
        int(device.index or 0), hidden, False, 1e-20, torch.bfloat16, emit_mxfp8=True
    )
    load_programs(runner)
    runner(
        x,
        residual,
        prev_post,
        prev_comb,
        pre_mix,
        norm_w,
        res_out,
        y_norm,
        a_values,
        a_scales,
    )

    ids = torch.randint(
        0, num_experts, (tokens, top_k), dtype=torch.int32, device=device
    )
    weights = torch.softmax(
        torch.randn(tokens, top_k, dtype=torch.float32, device=device), dim=-1
    )
    out_ref = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)
    out_prequant = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device)

    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=tokens, top_k=top_k),
    )
    session = PreparationSession(compile_workers=0, autotune=False)

    def prepare(state):
        sc = [
            torch.empty(spec.nbytes, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        ]
        binding = state.bind(
            a=y_norm,
            topk_ids=ids,
            topk_weights=weights,
            output=out_ref,
            scratch=sc,
            input_scales_static=True,
        )
        return PreparedCall(run=lambda: state.run(binding))

    session.prepare((plan.request(name=f"moe_{tokens}", prepare_call=prepare),))
    assert supports_prequant_mxfp8(plan, tokens)
    scratch = torch.empty(
        plan.scratch_specs()[0].nbytes, dtype=torch.uint8, device=device
    )
    prepared_moe_out(
        y_norm,
        weights,
        ids,
        out_ref,
        scratch,
        [],
        [],
        plan.handle,
        True,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
    )
    prepared_moe_prequant_out(
        y_norm,
        a_values,
        a_scales,
        weights,
        ids,
        out_prequant,
        scratch,
        [],
        [],
        plan.handle,
        True,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_prequant, out_ref)


@pytest.mark.parametrize("split_k", [5, 10])
def test_bf16_tc_gemv_router_split_k_5_and_10(split_k: int):
    _sm120_required()
    device = torch.device("cuda", torch.cuda.current_device())
    query = GemvQuery(
        source_dtype="bfloat16",
        weight_dtype="bfloat16",
        output_dtype="float32",
        max_rows=8,
        in_features=5120,
        out_features=384,
        source_contiguous=True,
        source_aligned=True,
        weight_contiguous=True,
        weight_aligned=True,
    )
    assert split_k in _tc_split_k_candidates(query)
    torch.manual_seed(77 + split_k)
    x = torch.randn(8, 5120, dtype=torch.bfloat16, device=device) * 0.5
    w = torch.randn(384, 5120, dtype=torch.bfloat16, device=device) * 0.05
    out = torch.empty(8, 384, dtype=torch.float32, device=device)
    plan = bf16_gemv.plan(query, override=GemvConfig(backend="tc", split_k=split_k))
    session = PreparationSession(compile_workers=0)
    session.prepare(
        (
            plan.request(
                name=f"router_sk{split_k}",
                prepare_call=lambda st: PreparedCall(run=lambda: st.run(x, w, out=out)),
            ),
        )
    )
    plan._prepared.state.run(x, w, out=out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x.float() @ w.float().T, atol=5e-2, rtol=1e-2)
