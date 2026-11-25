# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 reference, accuracy, edge-case, and CUDA-graph tests for b12x.attention.paged_nvfp4."""

from __future__ import annotations

import pytest
import torch

from b12x.attention import paged_nvfp4
from b12x.preparation import PreparationSession

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="requires SM120+ CUDA GPU",
)


def _dequant_nvfp4_block16(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch oracle matching SM120 E2M1 * E4M3 -> BF16 dequant."""
    lut = torch.tensor(
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
        device=packed.device,
        dtype=torch.float32,
    )
    lo = lut[(packed & 0xF).long()]
    hi = lut[((packed >> 4) & 0xF).long()]
    vals = torch.stack((lo, hi), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    sf = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return (vals * sf).to(torch.bfloat16)


def _reference_attention(
    q: torch.Tensor,
    k_bf16: torch.Tensor,
    v_bf16: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window_left: int = -1,
    sinks: torch.Tensor | None = None,
    rel_bias: torch.Tensor | None = None,
    rel_factors: torch.Tensor | None = None,
    rel_proj: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact FP32 causal attention reference over dequantized paged K/V."""
    _, hq, d = q.shape
    page_size, hkv = k_bf16.shape[1], k_bf16.shape[2]
    gqa = hq // hkv
    scale = d**-0.5
    out = torch.zeros_like(q)
    batch = int(cache_seqlens.numel())
    for b in range(batch):
        qs = int(cu_seqlens_q[b].item())
        qe = int(cu_seqlens_q[b + 1].item())
        qlen = qe - qs
        seqlen = int(cache_seqlens[b].item())
        if qlen <= 0 or seqlen <= 0:
            continue
        ctx = seqlen - qlen
        num_pages = (seqlen + page_size - 1) // page_size
        pids = page_table[b, :num_pages].long()
        k_seq = k_bf16[pids].reshape(num_pages * page_size, hkv, d)[:seqlen]
        v_seq = v_bf16[pids].reshape(num_pages * page_size, hkv, d)[:seqlen]
        k_seq = k_seq.repeat_interleave(gqa, dim=1).float()  # [S, Hq, D]
        v_seq = v_seq.repeat_interleave(gqa, dim=1).float()
        qb = q[qs:qe].float()  # [Q, Hq, D]
        logits = torch.einsum("qhd,khd->hqk", qb, k_seq) * scale  # [Hq, Q, S]

        q_pos = torch.arange(ctx, ctx + qlen, device=q.device)[:, None]
        k_pos = torch.arange(0, seqlen, device=q.device)[None, :]
        dist = q_pos - k_pos

        if rel_bias is not None:
            ext = rel_bias.shape[2]
            valid = (dist >= 0) & (dist < ext)
            clamped = dist.clamp(0, ext - 1)
            rb = rel_bias[qs:qe].float().permute(1, 0, 2)  # [Hq, Q, ext]
            gathered = rb.gather(2, clamped[None, :, :].expand(hq, -1, -1))
            logits = logits + torch.where(valid[None, :, :], gathered, 0.0)
        elif rel_factors is not None and rel_proj is not None:
            ext = rel_proj.shape[0]
            valid = (dist >= 0) & (dist < ext)
            clamped = dist.clamp(0, ext - 1)
            rf = rel_factors[qs:qe].float()  # [Q, Hq, 16]
            dense_b = (
                torch.einsum("qhr,er->hqe", rf, rel_proj.float())
                .to(torch.bfloat16)
                .float()
            )
            gathered = dense_b.gather(2, clamped[None, :, :].expand(hq, -1, -1))
            logits = logits + torch.where(valid[None, :, :], gathered, 0.0)

        mask = k_pos <= q_pos
        if window_left >= 0:
            mask &= k_pos >= (q_pos - window_left)
        logits = logits.masked_fill(~mask[None, :, :], float("-inf"))

        if sinks is not None:
            if sinks.ndim == 1:
                s_col = sinks[:, None, None].expand(hq, qlen, 1)
            else:
                s_col = sinks[qs:qe].t().unsqueeze(-1)  # [Hq, Q, 1]
            ext_logits = torch.cat((logits, s_col.float()), dim=-1)
            probs = torch.softmax(ext_logits, dim=-1)[..., :-1]
            probs = torch.where(torch.isposinf(s_col), torch.zeros_like(probs), probs)
        else:
            probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        ob = torch.einsum("hqk,khd->qhd", probs, v_seq)
        out[qs:qe] = ob.to(torch.bfloat16)
    return out


def _prepare_plan(caps: paged_nvfp4.Caps):
    from b12x.preparation import PreparedCall

    decl = paged_nvfp4.plan(caps)
    with PreparationSession(device=caps.device) as session:
        session.prepare(
            (
                decl.request(
                    name="test",
                    prepare_call=lambda state: PreparedCall(run=lambda: None),
                ),
            )
        )
    return decl


def _alloc_caches(pages: int, page_size: int, hkv: int, d: int, device: torch.device):
    k_cache = torch.zeros(
        (pages, page_size, hkv, d // 2), dtype=torch.uint8, device=device
    )
    k_scales = torch.zeros(
        (pages, page_size, hkv, d // 16), dtype=torch.uint8, device=device
    )
    v_cache = torch.zeros(
        (pages, page_size, hkv, d // 2), dtype=torch.uint8, device=device
    )
    v_scales = torch.zeros(
        (pages, page_size, hkv, d // 16), dtype=torch.uint8, device=device
    )
    return k_cache, k_scales, v_cache, v_scales


def _alloc_scratch(decl, device: torch.device, *, oversize_bytes: int = 0):
    return [
        torch.empty(s.nbytes + oversize_bytes, dtype=torch.uint8, device=device)
        for s in decl.scratch_specs()
    ]


@torch.no_grad()
def test_paged_nvfp4_kv_write_accuracy_and_skip_negative_slots():
    device = torch.device("cuda")
    torch.manual_seed(101)
    hkv, d, page_size, pages, tokens = 8, 128, 128, 4, 37
    caps = paged_nvfp4.Caps(
        device=device,
        mode="decode",
        num_q_heads=16,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=page_size,
        max_batch=2,
        max_q_per_req=1,
    )
    decl = _prepare_plan(caps)
    key = (torch.randn(tokens, hkv, d, device=device) * 1.5).to(torch.bfloat16)
    val = (torch.randn(tokens, hkv, d, device=device) * 1.5).to(torch.bfloat16)
    k_cache, k_scales, v_cache, v_scales = _alloc_caches(
        pages, page_size, hkv, d, device
    )

    slots = torch.arange(tokens, dtype=torch.int64, device=device) * 3 + 5
    slots[-3:] = -1  # padded rows must be skipped
    paged_nvfp4.write_kv(
        decl,
        key=key,
        value=val,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        slot_mapping=slots,
    )
    k_deq = _dequant_nvfp4_block16(k_cache, k_scales)
    v_deq = _dequant_nvfp4_block16(v_cache, v_scales)
    for t in range(tokens - 3):
        p, o = int(slots[t].item()) // page_size, int(slots[t].item()) % page_size
        k_cos = torch.nn.functional.cosine_similarity(
            k_deq[p, o].float().flatten(), key[t].float().flatten(), dim=0
        )
        v_cos = torch.nn.functional.cosine_similarity(
            v_deq[p, o].float().flatten(), val[t].float().flatten(), dim=0
        )
        assert k_cos.item() > 0.990
        assert v_cos.item() > 0.990
    # Unwritten slot 0 must stay zero.
    assert k_cache[0, 0].count_nonzero().item() == 0
    assert v_cache[0, 0].count_nonzero().item() == 0


@torch.no_grad()
@pytest.mark.parametrize(
    "batch,q_per_req,seq_lens",
    [
        (1, 1, [37]),
        (4, 1, [129, 513, 1500, 2048]),
        (2, 4, [260, 1040]),
    ],
)
def test_paged_nvfp4_decode_matches_dequant_reference_with_sinks_and_relbias(
    batch, q_per_req, seq_lens
):
    device = torch.device("cuda")
    torch.manual_seed(202 + batch * 7 + q_per_req)
    hq, hkv, d, page_size = 16, 8, 128, 128
    max_seq = max(seq_lens)
    pages_per_req = (max_seq + page_size - 1) // page_size
    total_pages = batch * pages_per_req + 2
    caps = paged_nvfp4.Caps(
        device=device,
        mode="decode" if q_per_req == 1 else "verify",
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=page_size,
        max_batch=batch,
        max_q_per_req=q_per_req,
        has_sinks=True,
        sink_layout="row",
        has_relative_bias=True,
    )
    decl = _prepare_plan(caps)
    # Use oversized pooled 1D uint8 scratch to exercise [:spec.nbytes] slicing.
    scratch = _alloc_scratch(decl, device, oversize_bytes=1024)
    k_cache, k_scales, v_cache, v_scales = _alloc_caches(
        total_pages, page_size, hkv, d, device
    )

    page_table = torch.zeros((batch, pages_per_req), dtype=torch.int32, device=device)
    next_p = 0
    for b, sl in enumerate(seq_lens):
        npages = (sl + page_size - 1) // page_size
        pids = torch.arange(next_p, next_p + npages, dtype=torch.int32, device=device)
        page_table[b, :npages] = pids
        next_p += npages
        k_raw = (torch.randn(sl, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        v_raw = (torch.randn(sl, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        pos = torch.arange(sl, dtype=torch.int64, device=device)
        slots = pids[pos // page_size].long() * page_size + (pos % page_size)
        paged_nvfp4.write_kv(
            decl,
            key=k_raw,
            value=v_raw,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            slot_mapping=slots,
        )

    total_q = batch * q_per_req
    q = (torch.randn(total_q, hq, d, device=device) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(
        0, total_q + 1, q_per_req, dtype=torch.int32, device=device
    )
    sinks = torch.randn(total_q, hq, dtype=torch.float32, device=device) * 0.5
    rel_bias = (torch.randn(total_q, hq, 1024, device=device) * 0.1).to(torch.bfloat16)

    binding = paged_nvfp4.bind(
        decl,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=sinks,
        relative_attention_bias=rel_bias,
    )
    paged_nvfp4.run(binding)

    k_deq = _dequant_nvfp4_block16(k_cache, k_scales)
    v_deq = _dequant_nvfp4_block16(v_cache, v_scales)
    ref = _reference_attention(
        q,
        k_deq,
        v_deq,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        sinks=sinks,
        rel_bias=rel_bias,
    )
    rel_l2 = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    assert torch.isfinite(out).all()
    assert cos.item() > 0.9999, f"cos={cos.item()}"
    assert rel_l2.item() < 5e-3, f"rel_l2={rel_l2.item()}"

    # CUDA graph capture & bit-exact replay check.
    out_graph = torch.empty_like(q)
    g_binding = paged_nvfp4.bind(
        decl,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out_graph,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=sinks,
        relative_attention_bias=rel_bias,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_nvfp4.run(g_binding)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_graph, out, rtol=0, atol=0)
    graph.reset()


@torch.no_grad()
@pytest.mark.parametrize(
    "ctx_lens,q_lens",
    [
        ([0], [192]),
        ([1024], [512]),
        ([2048], [256]),
        ([0, 317, 1024], [128, 192, 64]),
    ],
)
def test_paged_nvfp4_prefill_matches_dequant_reference_with_factored_relbias(
    ctx_lens, q_lens
):
    device = torch.device("cuda")
    torch.manual_seed(303 + sum(ctx_lens) + sum(q_lens))
    hq, hkv, d, page_size = 16, 8, 128, 128
    batch = len(ctx_lens)
    total_seqs = [c + q for c, q in zip(ctx_lens, q_lens, strict=True)]
    pages_per_req = max((s + page_size - 1) // page_size for s in total_seqs)
    total_pages = batch * pages_per_req
    caps = paged_nvfp4.Caps(
        device=device,
        mode="prefill",
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=page_size,
        max_batch=batch,
        max_q_per_req=max(q_lens) + 64,  # deliberately larger to test runtime max_q
        has_sinks=True,
        sink_layout="row",
        has_factored_relative_bias=True,
    )
    decl = _prepare_plan(caps)
    k_cache, k_scales, v_cache, v_scales = _alloc_caches(
        total_pages, page_size, hkv, d, device
    )

    page_table = torch.zeros((batch, pages_per_req), dtype=torch.int32, device=device)
    next_p = 0
    for b, sl in enumerate(total_seqs):
        npages = (sl + page_size - 1) // page_size
        pids = torch.arange(next_p, next_p + npages, dtype=torch.int32, device=device)
        page_table[b, :npages] = pids
        next_p += npages
        k_raw = (torch.randn(sl, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        v_raw = (torch.randn(sl, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        pos = torch.arange(sl, dtype=torch.int64, device=device)
        slots = pids[pos // page_size].long() * page_size + (pos % page_size)
        paged_nvfp4.write_kv(
            decl,
            key=k_raw,
            value=v_raw,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            slot_mapping=slots,
        )

    total_q = sum(q_lens)
    q = (torch.randn(total_q, hq, d, device=device) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    cache_seqlens = torch.tensor(total_seqs, dtype=torch.int32, device=device)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    sinks = torch.randn(total_q, hq, dtype=torch.float32, device=device) * 0.5
    rel_factors = (torch.randn(total_q, hq, 16, device=device) * 0.25).to(
        torch.bfloat16
    )
    rel_proj = (torch.randn(1024, 16, device=device) * 0.25).to(torch.bfloat16)

    binding = paged_nvfp4.bind(
        decl,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=sinks,
        relative_bias_factors=rel_factors,
        relative_bias_projection=rel_proj,
    )
    paged_nvfp4.run(binding)

    k_deq = _dequant_nvfp4_block16(k_cache, k_scales)
    v_deq = _dequant_nvfp4_block16(v_cache, v_scales)
    ref = _reference_attention(
        q,
        k_deq,
        v_deq,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        sinks=sinks,
        rel_factors=rel_factors,
        rel_proj=rel_proj,
    )
    rel_l2 = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    )
    assert torch.isfinite(out).all()
    # Prefill uses SM120 FP8 E4M3 MMA (matching b12x.attention.paged FP8 extend precision).
    assert cos.item() > 0.9990, f"cos={cos.item()}"
    assert rel_l2.item() < 4e-2, f"rel_l2={rel_l2.item()}"

    # Prefill CUDA graph capture & bit-exact replay check.
    out_graph = torch.empty_like(q)
    g_binding = paged_nvfp4.bind(
        decl,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out_graph,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=sinks,
        relative_bias_factors=rel_factors,
        relative_bias_projection=rel_proj,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_nvfp4.run(g_binding)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_graph, out, rtol=0, atol=0)
    graph.reset()


@torch.no_grad()
def test_paged_nvfp4_decode_window_and_head_sinks():
    """Verify sliding-window masking and per-head FP32 sinks (including +inf sink)."""
    device = torch.device("cuda")
    torch.manual_seed(404)
    hq, hkv, d, page_size, batch, seq_len, window_left = 16, 8, 128, 128, 2, 700, 255
    pages_per_req = (seq_len + page_size - 1) // page_size
    total_pages = batch * pages_per_req
    caps = paged_nvfp4.Caps(
        device=device,
        mode="decode",
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=page_size,
        max_batch=batch,
        max_q_per_req=1,
        window_left=window_left,
        has_sinks=True,
        sink_layout="head",
    )
    decl = _prepare_plan(caps)
    scratch = _alloc_scratch(decl, device)
    k_cache, k_scales, v_cache, v_scales = _alloc_caches(
        total_pages, page_size, hkv, d, device
    )

    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch, pages_per_req
    )
    for b in range(batch):
        k_raw = (torch.randn(seq_len, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        v_raw = (torch.randn(seq_len, hkv, d, device=device) * 0.5).to(torch.bfloat16)
        slots = torch.arange(
            b * pages_per_req * page_size,
            b * pages_per_req * page_size + seq_len,
            dtype=torch.int64,
            device=device,
        )
        paged_nvfp4.write_kv(
            decl,
            key=k_raw,
            value=v_raw,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            slot_mapping=slots,
        )

    q = (torch.randn(batch, hq, d, device=device) * 0.5).to(torch.bfloat16)
    out = torch.empty_like(q)
    cache_seqlens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.arange(0, batch + 1, dtype=torch.int32, device=device)
    sinks = torch.randn(hq, dtype=torch.float32, device=device) * 0.5
    sinks[0] = float("inf")  # +inf sink must produce exact zero output on head 0

    binding = paged_nvfp4.bind(
        decl,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attention_sink_bias=sinks,
    )
    paged_nvfp4.run(binding)

    k_deq = _dequant_nvfp4_block16(k_cache, k_scales)
    v_deq = _dequant_nvfp4_block16(v_cache, v_scales)
    ref = _reference_attention(
        q,
        k_deq,
        v_deq,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        window_left=window_left,
        sinks=sinks,
    )
    assert out[:, 0].abs().max().item() == 0.0
    cos = torch.nn.functional.cosine_similarity(
        out[:, 1:].float().flatten(), ref[:, 1:].float().flatten(), dim=0
    )
    assert cos.item() > 0.9999, f"cos={cos.item()}"


@torch.no_grad()
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_paged_nvfp4_caller_owned_placeholders_zero_seqlen_and_nan_tail(mode):
    """Verify caller_owned_placeholders=True, NaN-poisoned page tails, and seq_len=0."""
    device = torch.device("cuda")
    torch.manual_seed(505)
    hq, hkv, d, page_size = 16, 8, 128, 128
    batch = 2
    q_per_req = 1 if mode == "decode" else 64
    active_seq = 135  # 1 full page + 7 tokens on partial page 1
    caps = paged_nvfp4.Caps(
        device=device,
        mode=mode,
        num_q_heads=hq,
        num_kv_heads=hkv,
        head_dim_qk=d,
        head_dim_vo=d,
        page_size=page_size,
        max_batch=batch,
        max_q_per_req=q_per_req,
        caller_owned_placeholders=True,
    )
    decl = _prepare_plan(caps)
    scratch = _alloc_scratch(decl, device)
    # In prefill mode with caller_owned_placeholders=True and no sinks/relbias,
    # only the placeholder buffers should be requested (no decode split-KV buffers).
    if mode == "prefill":
        spec_names = {s.name for s in decl.scratch_specs()}
        assert "paged_nvfp4.o_part" not in spec_names
        assert "paged_nvfp4.lse_part" not in spec_names
        assert "paged_nvfp4.dummy_rel_proj" in spec_names

    k_cache, k_scales, v_cache, v_scales = _alloc_caches(4, page_size, hkv, d, device)
    # Poison every scale byte with E4M3 NaN (0x7F) and data with 0xFF before writing active tokens.
    k_cache.fill_(0xFF)
    v_cache.fill_(0xFF)
    k_scales.fill_(0x7F)
    v_scales.fill_(0x7F)

    k_raw = (torch.randn(active_seq, hkv, d, device=device) * 0.5).to(torch.bfloat16)
    v_raw = (torch.randn(active_seq, hkv, d, device=device) * 0.5).to(torch.bfloat16)
    slots = torch.arange(active_seq, dtype=torch.int64, device=device)
    paged_nvfp4.write_kv(
        decl,
        key=k_raw,
        value=v_raw,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        slot_mapping=slots,
    )

    total_q = batch * q_per_req
    q = (torch.randn(total_q, hq, d, device=device) * 0.5).to(torch.bfloat16)
    sentinel = 3.25
    out = torch.full_like(q, sentinel)
    page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device=device)
    # Batch 0 active with partial NaN-tailed page; batch 1 has seq_len == 0.
    cache_seqlens = torch.tensor([active_seq, 0], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor(
        [0, q_per_req, total_q], dtype=torch.int32, device=device
    )

    # Capture write_kv + run inside a single CUDA graph under caller_owned_placeholders=True.
    binding = paged_nvfp4.bind(
        decl,
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        k_scales=k_scales,
        v_cache=v_cache,
        v_scales=v_scales,
        output=out,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_nvfp4.write_kv(
            decl,
            key=k_raw,
            value=v_raw,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            slot_mapping=slots,
        )
        paged_nvfp4.run(binding)
    graph.replay()
    torch.cuda.synchronize()
    graph.reset()

    # Active request 0 must be finite and match reference despite NaN-poisoned page tail.
    k_deq = _dequant_nvfp4_block16(k_cache, k_scales)
    v_deq = _dequant_nvfp4_block16(v_cache, v_scales)
    ref = _reference_attention(q, k_deq, v_deq, page_table, cache_seqlens, cu_seqlens_q)
    assert torch.isfinite(out[:q_per_req]).all()
    cos = torch.nn.functional.cosine_similarity(
        out[:q_per_req].float().flatten(), ref[:q_per_req].float().flatten(), dim=0
    )
    assert cos.item() > (0.9999 if mode == "decode" else 0.9990), f"cos={cos.item()}"
    # Inactive request 1 (cache_seqlens == 0) must exit without overwriting output.
    assert torch.equal(out[q_per_req:], torch.full_like(out[q_per_req:], sentinel))


def test_paged_nvfp4_validation_rejects_unsupported_and_mismatched_operands():
    device = torch.device("cuda")
    # Unsupported query shapes / formats.
    assert not paged_nvfp4.supports(
        mode="decode",
        kv_format="nvfp4",
        num_q_heads=16,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
    )
    assert not paged_nvfp4.supports(
        mode="decode",
        num_q_heads=16,
        num_kv_heads=8,
        head_dim_qk=64,
        head_dim_vo=64,
        page_size=128,
    )
    assert not paged_nvfp4.supports(
        mode="decode",
        num_q_heads=16,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        causal=False,
    )
    assert paged_nvfp4.supports(
        mode="decode",
        kv_format="nvfp4_4o6k",
        num_q_heads=16,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
    )

    caps = paged_nvfp4.Caps(
        device=device,
        mode="decode",
        num_q_heads=16,
        num_kv_heads=8,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        max_batch=2,
        max_q_per_req=1,
    )
    decl = _prepare_plan(caps)
    scratch = _alloc_scratch(decl, device)
    k_cache, k_scales, v_cache, v_scales = _alloc_caches(2, 128, 8, 128, device)
    q = torch.zeros((2, 16, 128), dtype=torch.bfloat16, device=device)
    out = torch.zeros_like(q)
    page_table = torch.zeros((2, 1), dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([1, 1], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)

    # Mismatched page count between k_cache and v_cache.
    with pytest.raises(ValueError, match="page count"):
        paged_nvfp4.bind(
            decl,
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache[:1],
            v_scales=v_scales,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
        )

    # Unexpected optional sinks when has_sinks=False.
    with pytest.raises(ValueError, match="attention_sink_bias"):
        paged_nvfp4.bind(
            decl,
            scratch=scratch,
            q=q,
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            output=out,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            attention_sink_bias=torch.zeros(16, dtype=torch.float32, device=device),
        )

    # Wrong write_kv dtype.
    with pytest.raises(ValueError, match="bfloat16"):
        paged_nvfp4.write_kv(
            decl,
            key=torch.zeros((4, 8, 128), dtype=torch.float16, device=device),
            value=torch.zeros((4, 8, 128), dtype=torch.bfloat16, device=device),
            k_cache=k_cache,
            k_scales=k_scales,
            v_cache=v_cache,
            v_scales=v_scales,
            slot_mapping=torch.arange(4, dtype=torch.int64, device=device),
        )
