# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Correctness, CSF equivalence, determinism, and CUDA-graph tests for b12x.moe.situ_moe."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.quant.csf_encode import encode_csf_source
from b12x.moe import situ_moe
from b12x.preparation import PreparationSession, PreparedCall


def _prepare(plan, device: torch.device) -> None:
    with PreparationSession(device=device) as session:
        session.prepare(
            (
                plan.request(
                    name="test",
                    prepare_call=lambda state: PreparedCall(run=lambda: None),
                ),
            )
        )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not situ_moe.is_supported(),
    reason="b12x.moe.situ_moe requires SM120/SM121 CUDA device",
)

_FP4_LUT = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def _dequant_nvfp4(
    w_u8: torch.Tensor, s_u8: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """Reference FP32 dequantization of NVFP4 weights ``[E, N, K//2]``."""
    e, n, half_k = w_u8.shape
    lut = _FP4_LUT.to(w_u8.device)
    lo = lut[(w_u8 & 0xF).to(torch.int64)]
    hi = lut[((w_u8 >> 4) & 0xF).to(torch.int64)]
    vals = torch.stack((lo, hi), dim=-1).reshape(e, n, half_k * 2)
    scales = s_u8.view(torch.float8_e4m3fn).to(torch.float32)
    scales = scales.repeat_interleave(16, dim=-1)
    # Effective weight scale = (g[e] / g[E + e])? Wait: exact situ_experts formula is
    # y = dot(q_a * s_a, q_w * s_w) * (amax_a * g[e]) / (1536 * g[E + e])
    # and q_a * s_a * (amax_a / 1536) == x_quant, so w_eff = (q_w * s_w) * (g[e] / g[E + e]).
    w_gain = (g[:e] / g[e:]).view(e, 1, 1)
    return vals * scales * w_gain


def _dequant_mxfp8(w_u8: torch.Tensor, s_u8: torch.Tensor) -> torch.Tensor:
    """Reference FP32 dequantization of MXFP8 weights ``[E, N, K]``."""
    vals = w_u8.view(torch.float8_e4m3fn).to(torch.float32)
    exp = s_u8.to(torch.int32)
    bits = torch.where(exp == 0, torch.full_like(exp, 0x00400000), exp << 23)
    scales = bits.view(torch.float32).repeat_interleave(32, dim=-1)
    return vals * scales


def _quant_nvfp4_4o6_ref(x: torch.Tensor) -> torch.Tensor:
    """Reference dequantized 4-or-6 MAE NVFP4 activation (for exact reference)."""
    m, k = x.shape
    xf = x.to(torch.float32)
    amax = xf.abs().amax(dim=-1, keepdim=True)
    encode = torch.clamp(1536.0 / amax, max=3.4028234663852886e38)
    decoder = 1.0 / encode
    outer = amax * (1.0 / 1536.0)
    xb = xf.reshape(m, k // 16, 16)
    bmax = xb.abs().amax(dim=-1, keepdim=True)

    lut = _FP4_LUT.to(x.device)

    def _try_scale(div: float):
        raw = torch.clamp(bmax * (encode.unsqueeze(-1) / div), max=448.0)
        sf = raw.to(torch.float8_e4m3fn).to(torch.float32)
        inv = torch.clamp(1.0 / (sf * decoder.unsqueeze(-1)), max=3.4028234663852886e38)
        scaled = xb * inv
        # Nearest E2M1 value (round-to-nearest-even ties break identically on typical floats)
        idx = (scaled.unsqueeze(-1) - lut).abs().argmin(dim=-1)
        rec = lut[idx] * (sf * outer.unsqueeze(-1))
        err = (rec - xb).abs().sum(dim=-1, keepdim=True)
        return rec, err

    r6, e6 = _try_scale(6.0)
    r4, e4 = _try_scale(4.0)
    return torch.where(e4 < e6, r4, r6).reshape(m, k)


def _quant_mxfp8_ref(x: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    xb = x.to(torch.float32).reshape(m, k // 32, 32)
    raw = xb.abs().amax(dim=-1, keepdim=True) * (1.0 / 448.0)
    minimum = 2.0**-127
    bits = raw.view(torch.int32)
    exp = ((bits + 0x007FFFFF) & 0x7F800000) >> 23
    exp = torch.where(raw <= minimum, torch.zeros_like(exp), torch.clamp(exp, max=254))
    scale = torch.where(
        exp == 0, torch.full_like(raw, minimum), (exp << 23).view(torch.float32)
    )
    inv = torch.where(exp == 0, torch.full_like(raw, 2.0**127), 1.0 / scale)
    q = torch.clamp(xb * inv, -448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)
    return (q * scale).reshape(m, k)


def _situ_ref(gate: torch.Tensor, up: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
    g = gate.to(torch.bfloat16).to(torch.float32)
    u = up.to(torch.bfloat16).to(torch.float32)
    sig = torch.sigmoid(g)
    val = (4.0 * torch.tanh(g / 4.0) * sig) * (25.0 * torch.tanh(u / 25.0))
    return (val * rw).to(torch.bfloat16)


def _reference_situ_moe(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    w13_f32: torch.Tensor,
    w2_f32: torch.Tensor,
    *,
    is_nvfp4: bool,
) -> torch.Tensor:
    m, h = x.shape
    top_k = ids.shape[1]
    xq = _quant_nvfp4_4o6_ref(x) if is_nvfp4 else _quant_mxfp8_ref(x)
    out = torch.zeros((m, h), dtype=torch.float32, device=x.device)
    for tok in range(m):
        for k in range(top_k):
            e = int(ids[tok, k].item())
            if e < 0:
                continue
            fc1 = xq[tok] @ w13_f32[e].T
            gate = fc1[0::2]
            up = fc1[1::2]
            inter = _situ_ref(gate, up, weights[tok, k])
            iq = (
                _quant_nvfp4_4o6_ref(inter.unsqueeze(0))
                if is_nvfp4
                else _quant_mxfp8_ref(inter.unsqueeze(0))
            )[0]
            fc2 = (iq @ w2_f32[e].T).to(torch.bfloat16).to(torch.float32)
            out[tok] += fc2
    return out.to(torch.bfloat16)


def _make_nvfp4_weights(e: int, h: int, inter: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(1234)
    w13 = torch.randint(
        0, 256, (e, 2 * inter, h // 2), dtype=torch.uint8, generator=gen
    ).to(device)
    w2 = torch.randint(0, 256, (e, h, inter // 2), dtype=torch.uint8, generator=gen).to(
        device
    )
    # Realistic clustered positive E4M3 scales with occasional outliers (tests CSF exceptions!)
    s13_f = (0.25 + 0.15 * torch.rand((e, 2 * inter, h // 16), generator=gen)).to(
        torch.float8_e4m3fn
    )
    s2_f = (0.25 + 0.15 * torch.rand((e, h, inter // 16), generator=gen)).to(
        torch.float8_e4m3fn
    )
    # Inject sparse extreme scales so CSF exceptions are exercised heavily.
    s13_f[:, ::17, ::19] = torch.tensor(12.0, dtype=torch.float8_e4m3fn)
    s2_f[:, ::13, ::11] = torch.tensor(16.0, dtype=torch.float8_e4m3fn)
    s13 = s13_f.view(torch.uint8).to(device).contiguous()
    s2 = s2_f.view(torch.uint8).to(device).contiguous()
    g13 = torch.cat([torch.full((e,), 0.08), torch.full((e,), 1.0)]).to(device)
    g2 = torch.cat([torch.full((e,), 0.08), torch.full((e,), 1.0)]).to(device)
    return w13, s13, g13, w2, s2, g2


def _make_mxfp8_weights(
    e: int, h: int, inter: int, device: torch.device, *, full_range: bool = False
):
    gen = torch.Generator(device="cpu").manual_seed(5678)
    # Full-range E4M3 weights (up to +/-448) with small UE8M0 scales match real
    # quantized checkpoints and exercise FP32 accumulation (448*448 > FP16_MAX).
    scale = 200.0 if full_range else 0.5
    s_lo, s_hi = (114, 120) if full_range else (122, 128)
    w13 = (
        (torch.randn((e, 2 * inter, h), generator=gen) * scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(device)
        .contiguous()
    )
    w2 = (
        (torch.randn((e, h, inter), generator=gen) * scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(device)
        .contiguous()
    )
    s13 = (
        torch.randint(
            s_lo, s_hi, (e, 2 * inter, h // 32), dtype=torch.uint8, generator=gen
        )
        .to(device)
        .contiguous()
    )
    s2 = (
        torch.randint(s_lo, s_hi, (e, h, inter // 32), dtype=torch.uint8, generator=gen)
        .to(device)
        .contiguous()
    )
    return w13, s13, w2, s2


@pytest.mark.parametrize("tokens", [1, 4, 16, 64])
@pytest.mark.parametrize("has_csf", [False, True])
def test_nvfp4_accuracy_and_csf_equivalence(tokens: int, has_csf: bool):
    device = torch.device("cuda")
    e, h, inter, top_k = 16, 512, 256, 4
    w13, s13, g13, w2, s2, g2 = _make_nvfp4_weights(e, h, inter, device)

    gen = torch.Generator(device="cpu").manual_seed(100 + tokens)
    x = (torch.randn((tokens, h), generator=gen) * 0.5).to(torch.bfloat16).to(device)
    ids = (
        torch.stack([torch.randperm(e, generator=gen)[:top_k] for _ in range(tokens)])
        .to(torch.int32)
        .to(device)
    )
    if tokens >= 4:
        ids[-1, -1] = -1  # exercise inactive route masking
    weights = (
        torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=-1)
        .to(torch.float32)
        .to(device)
    )

    caps = situ_moe.Caps(
        device=device,
        weight_format="nvfp4",
        hidden=h,
        intermediate=inter,
        num_experts=e,
        top_k=top_k,
        max_tokens=tokens,
        has_csf=has_csf,
    )
    plan = situ_moe.plan(caps)
    _prepare(plan, device)
    scratch = [
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    ]
    out = torch.empty_like(x)

    if has_csf:
        csf13 = encode_csf_source(s13, quantization="nvfp4")
        csf2 = encode_csf_source(s2, quantization="nvfp4")
        exc13, off13 = situ_moe.partition_csf_slab_offsets(
            csf13.components[1], csf13.starts, e, 2 * inter, h // 16
        )
        exc2, off2 = situ_moe.partition_csf_slab_offsets(
            csf2.components[1], csf2.starts, e, h, inter // 16
        )
        binding = situ_moe.bind(
            plan,
            scratch=scratch,
            x=x,
            ids=ids,
            weights=weights,
            w13=w13,
            s13=csf13.components[0],
            w2=w2,
            s2=csf2.components[0],
            output=out,
            g13=g13,
            g2=g2,
            s13_exc=exc13,
            s13_slab_offsets=off13,
            s2_exc=exc2,
            s2_slab_offsets=off2,
        )
    else:
        binding = situ_moe.bind(
            plan,
            scratch=scratch,
            x=x,
            ids=ids,
            weights=weights,
            w13=w13,
            s13=s13,
            w2=w2,
            s2=s2,
            output=out,
            g13=g13,
            g2=g2,
        )

    situ_moe.run(binding)
    torch.cuda.synchronize()

    w13_ref = _dequant_nvfp4(w13, s13, g13)
    w2_ref = _dequant_nvfp4(w2, s2, g2)
    ref = _reference_situ_moe(x, ids, weights, w13_ref, w2_ref, is_nvfp4=True)

    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    rel_l2 = (out.float() - ref.float()).norm().item() / max(
        ref.float().norm().item(), 1e-6
    )
    assert cos > 0.999, (
        f"NVFP4 (M={tokens}, csf={has_csf}) cosine={cos:.6f}, rel_l2={rel_l2:.6f}"
    )
    assert rel_l2 < 0.03, f"NVFP4 (M={tokens}, csf={has_csf}) rel_l2={rel_l2:.6f}"

    # Verify exact equivalence between dense and inline CSF (deterministic mode is bit-exact;
    # default fused_route_sum uses relaxed atomic BF16x2 scatter-add -> cosine > 0.9999).
    if has_csf:
        for det in (True, False):
            caps_c = situ_moe.Caps(
                device=device,
                weight_format="nvfp4",
                hidden=h,
                intermediate=inter,
                num_experts=e,
                top_k=top_k,
                max_tokens=tokens,
                has_csf=True,
                deterministic_output=det,
            )
            caps_d = situ_moe.Caps(
                device=device,
                weight_format="nvfp4",
                hidden=h,
                intermediate=inter,
                num_experts=e,
                top_k=top_k,
                max_tokens=tokens,
                has_csf=False,
                deterministic_output=det,
            )
            plan_c, plan_d = situ_moe.plan(caps_c), situ_moe.plan(caps_d)
            _prepare(plan_c, device)
            _prepare(plan_d, device)
            sc_c = [torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in plan_c.scratch_specs()]
            sc_d = [torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in plan_d.scratch_specs()]
            out_c, out_d = torch.empty_like(x), torch.empty_like(x)
            situ_moe.run(situ_moe.bind(
                plan_c, scratch=sc_c, x=x, ids=ids, weights=weights,
                w13=w13, s13=csf13.components[0], w2=w2, s2=csf2.components[0], output=out_c,
                g13=g13, g2=g2, s13_exc=exc13, s13_slab_offsets=off13, s2_exc=exc2, s2_slab_offsets=off2,
            ))
            situ_moe.run(situ_moe.bind(
                plan_d, scratch=sc_d, x=x, ids=ids, weights=weights,
                w13=w13, s13=s13, w2=w2, s2=s2, output=out_d, g13=g13, g2=g2,
            ))
            torch.cuda.synchronize()
            if det:
                assert torch.equal(out_c, out_d), (
                    "Deterministic inline CSF must be bit-exact with dense NVFP4 scales"
                )
            else:
                csf_cos = torch.nn.functional.cosine_similarity(
                    out_c.float().flatten(), out_d.float().flatten(), dim=0
                ).item()
                assert csf_cos > 0.9999, f"Fused CSF vs dense cosine={csf_cos:.6f}"


@pytest.mark.parametrize("tokens", [1, 4, 16, 64])
@pytest.mark.parametrize("full_range", [False, True])
def test_mxfp8_accuracy(tokens: int, full_range: bool):
    device = torch.device("cuda")
    e, h, inter, top_k = 16, 512, 256, 4
    w13, s13, w2, s2 = _make_mxfp8_weights(e, h, inter, device, full_range=full_range)

    gen = torch.Generator(device="cpu").manual_seed(200 + tokens)
    x = (torch.randn((tokens, h), generator=gen) * 0.5).to(torch.bfloat16).to(device)
    ids = (
        torch.stack([torch.randperm(e, generator=gen)[:top_k] for _ in range(tokens)])
        .to(torch.int32)
        .to(device)
    )
    if tokens >= 4:
        ids[0, 1] = -1
    weights = (
        torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=-1)
        .to(torch.float32)
        .to(device)
    )

    caps = situ_moe.Caps(
        device=device,
        weight_format="mxfp8",
        hidden=h,
        intermediate=inter,
        num_experts=e,
        top_k=top_k,
        max_tokens=tokens,
    )
    plan = situ_moe.plan(caps)
    _prepare(plan, device)
    scratch = [
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    ]
    out = torch.empty_like(x)
    binding = situ_moe.bind(
        plan,
        scratch=scratch,
        x=x,
        ids=ids,
        weights=weights,
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        output=out,
    )
    situ_moe.run(binding)
    torch.cuda.synchronize()

    w13_ref = _dequant_mxfp8(w13, s13)
    w2_ref = _dequant_mxfp8(w2, s2)
    ref = _reference_situ_moe(x, ids, weights, w13_ref, w2_ref, is_nvfp4=False)

    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    rel_l2 = (out.float() - ref.float()).norm().item() / max(
        ref.float().norm().item(), 1e-6
    )
    assert cos > 0.9995, f"MXFP8 (M={tokens}) cosine={cos:.6f}, rel_l2={rel_l2:.6f}"
    assert rel_l2 < 0.015, f"MXFP8 (M={tokens}) rel_l2={rel_l2:.6f}"


@pytest.mark.parametrize(
    "weight_format,has_csf,tokens",
    [
        ("nvfp4", True, 8),
        ("nvfp4", True, 64),
        ("mxfp8", False, 8),
        ("mxfp8", False, 64),
    ],
)
def test_cuda_graph_capture_and_bit_exact_replay(
    weight_format: str, has_csf: bool, tokens: int
):
    device = torch.device("cuda")
    e, h, inter, top_k = 16, 512, 256, 4
    gen = torch.Generator(device="cpu").manual_seed(999)
    x = (torch.randn((tokens, h), generator=gen) * 0.5).to(torch.bfloat16).to(device)
    ids = (
        torch.stack([torch.randperm(e, generator=gen)[:top_k] for _ in range(tokens)])
        .to(torch.int32)
        .to(device)
    )
    weights = (
        torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=-1)
        .to(torch.float32)
        .to(device)
    )

    caps = situ_moe.Caps(
        device=device,
        weight_format=weight_format,
        hidden=h,
        intermediate=inter,
        num_experts=e,
        top_k=top_k,
        max_tokens=tokens,
        has_csf=has_csf,
        deterministic_output=True,
    )
    plan = situ_moe.plan(caps)
    _prepare(plan, device)
    scratch = [
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in plan.scratch_specs()
    ]
    out_eager = torch.empty_like(x)
    out_graph = torch.empty_like(x)

    if weight_format == "nvfp4":
        w13, s13, g13, w2, s2, g2 = _make_nvfp4_weights(e, h, inter, device)
        csf13 = encode_csf_source(s13, quantization="nvfp4")
        csf2 = encode_csf_source(s2, quantization="nvfp4")
        exc13, off13 = situ_moe.partition_csf_slab_offsets(
            csf13.components[1], csf13.starts, e, 2 * inter, h // 16
        )
        exc2, off2 = situ_moe.partition_csf_slab_offsets(
            csf2.components[1], csf2.starts, e, h, inter // 16
        )
        kw = dict(
            w13=w13,
            s13=csf13.components[0],
            w2=w2,
            s2=csf2.components[0],
            g13=g13,
            g2=g2,
            s13_exc=exc13,
            s13_slab_offsets=off13,
            s2_exc=exc2,
            s2_slab_offsets=off2,
        )
    else:
        w13, s13, w2, s2 = _make_mxfp8_weights(e, h, inter, device)
        kw = dict(w13=w13, s13=s13, w2=w2, s2=s2)

    b_eager = situ_moe.bind(
        plan, scratch=scratch, x=x, ids=ids, weights=weights, output=out_eager, **kw
    )
    situ_moe.run(b_eager)
    torch.cuda.synchronize()

    b_graph = situ_moe.bind(
        plan, scratch=scratch, x=x, ids=ids, weights=weights, output=out_graph, **kw
    )
    # Warmup on capture stream
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        situ_moe.run(b_graph)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        situ_moe.run(b_graph)
    out_graph.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_graph, out_eager), (
        "CUDA graph replay must be bit-exact with eager"
    )
