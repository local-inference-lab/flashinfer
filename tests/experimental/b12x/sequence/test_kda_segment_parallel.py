"""GPU verification for distributed 4-segment KDA prefill vs sequential kda_prefill."""

import pytest
import torch

from b12x.preparation import PreparationSession
from b12x.preparation.types import PreparedCall
from b12x.sequence import kda_prefill


def _affine_scan_4(init_state, A_list, B_list, *, bf16_transfer=False):
    """Exact 4-rank FP32 affine prefix scan: S_{r+1} = S_r @ A_r + B_r (transposed state [H, V, K])."""
    S = init_state.clone()
    incoming = []
    for A_r, B_r in zip(A_list, B_list, strict=True):
        incoming.append(S.clone())
        A = A_r.to(torch.bfloat16).to(torch.float32) if bf16_transfer else A_r
        S = torch.bmm(S, A) + B_r
    return incoming, S


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("live_tokens,fresh,bf16_transfer", [
    (4096, True, False),
    (4096, False, False),
    (4096, False, True),
    (1500, False, False),
    (1500, False, True),
    (1024, True, False),
    (3073, False, True),
    (17, False, False),
    (0, False, False),
])
def test_kda_segment_parallel_matches_sequential(live_tokens, fresh, bf16_transfer):
    device = torch.device("cuda:0")
    torch.manual_seed(42 + live_tokens)
    total_tokens = 4096
    world = 4
    seg_tokens = total_tokens // world
    heads = 64
    head_dim = 128

    q = torch.randn((total_tokens, heads, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    k = torch.randn((total_tokens, heads, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    v = torch.randn((total_tokens, heads, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    raw_g = torch.randn((total_tokens, heads, head_dim), dtype=torch.bfloat16, device=device) * 0.5
    raw_beta = torch.randn((total_tokens, heads), dtype=torch.bfloat16, device=device)
    A_log = torch.randn((heads,), dtype=torch.float32, device=device) * 0.5 - 1.0
    dt_bias = torch.randn((heads, head_dim), dtype=torch.float32, device=device) * 0.2

    init_state = (
        torch.zeros((heads, head_dim, head_dim), dtype=torch.float32, device=device)
        if fresh
        else torch.randn((heads, head_dim, head_dim), dtype=torch.float32, device=device) * 0.1
    )

    # 1. Sequential reference using standard kda_prefill
    seq_state = torch.zeros((2, heads, head_dim, head_dim), dtype=torch.float32, device=device)
    seq_state[1].copy_(init_state)
    seq_out = torch.zeros((total_tokens, heads, head_dim), dtype=torch.bfloat16, device=device)
    cu_full = torch.tensor([0, live_tokens], dtype=torch.int32, device=device)
    idx_init = torch.tensor([0 if fresh else 1], dtype=torch.int64, device=device)
    idx_final = torch.tensor([1], dtype=torch.int64, device=device)
    idx_zero = torch.tensor([0], dtype=torch.int64, device=device)
    off_zero = torch.tensor([0], dtype=torch.int32, device=device)
    num_seqs = torch.tensor([1 if live_tokens > 0 else 0], dtype=torch.int32, device=device)
    num_toks = torch.tensor([live_tokens], dtype=torch.int32, device=device)

    seq_caps = kda_prefill.Caps(
        device=device, max_tokens=total_tokens, max_seqs=1, max_state_slots=2,
        heads=heads, null_state_index=0, checkpoint_export=False,
    )
    seq_plan = kda_prefill.plan(seq_caps, invocation=kda_prefill.invocation_from_tensors(
        A_log=A_log, dt_bias=dt_bias, initial_state_indices=idx_init))

    seg_caps = kda_prefill.Caps(
        device=device, max_tokens=seg_tokens, max_seqs=1, max_state_slots=6,
        heads=heads, null_state_index=0, checkpoint_export=False,
    )
    seg_plan = kda_prefill.segment_plan(seg_caps, invocation=kda_prefill.invocation_from_tensors(
        A_log=A_log, dt_bias=dt_bias, initial_state_indices=idx_init))

    session = PreparationSession(device=device, autotune=False)
    noop = lambda state: PreparedCall(run=lambda: None)
    session.prepare((seq_plan.request(name="seq", prepare_call=noop), seg_plan.request(name="seg", prepare_call=noop)))

    seq_scratch = torch.empty(seq_plan.memory_requirements().scratch[0].nbytes, dtype=torch.uint8, device=device)
    seq_binding = kda_prefill.bind(
        seq_plan, scratch=seq_scratch, q=q, k=k, v=v, raw_g=raw_g, raw_beta=raw_beta,
        A_log=A_log, dt_bias=dt_bias, recurrent_state=seq_state, cu_seqlens=cu_full,
        initial_state_indices=idx_init, final_state_indices=idx_final,
        checkpoint_state_indices=idx_zero, checkpoint_offsets=off_zero,
        num_seqs=num_seqs, num_tokens=num_toks, output=seq_out,
    )
    kda_prefill.run(seq_binding, lower_bound=-5.0, eps=1e-6)

    # 2. 4-segment distributed parallel execution
    seg_scratch_bytes = seg_plan.memory_requirements().scratch[0].nbytes
    eye = torch.eye(head_dim, dtype=torch.float32, device=device).unsqueeze(0).expand(heads, -1, -1).contiguous()

    scratches, pools, outs, seg_metas = [], [], [], []
    A_list, B_list = [], []
    for r in range(world):
        start = r * seg_tokens
        end = start + seg_tokens
        live_r = max(0, min(seg_tokens, live_tokens - start))
        sc = torch.empty(seg_scratch_bytes, dtype=torch.uint8, device=device)
        pool = torch.zeros((6, heads, head_dim, head_dim), dtype=torch.float32, device=device)
        pool[1].copy_(eye)
        out_r = torch.zeros((seg_tokens, heads, head_dim), dtype=torch.bfloat16, device=device)
        cu_r = torch.tensor([0, live_r], dtype=torch.int32, device=device)
        ns_r = torch.tensor([1], dtype=torch.int32, device=device)
        nt_r = torch.tensor([live_r], dtype=torch.int32, device=device)
        kda_prefill.segment_summarize(
            seg_plan,
            q=q[start:end], k=k[start:end], v=v[start:end],
            raw_g=raw_g[start:end], raw_beta=raw_beta[start:end],
            A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu_r,
            checkpoint_state_indices=idx_zero, checkpoint_offsets=off_zero,
            num_seqs=ns_r, num_tokens=nt_r,
            scratch=sc, pool=pool, output=out_r,
            lower_bound=-5.0, eps=1e-6,
        )
        scratches.append(sc)
        pools.append(pool)
        outs.append(out_r)
        seg_metas.append((start, end, cu_r, ns_r, nt_r))
        A_list.append(pool[2].clone())
        B_list.append(pool[3].clone())

    incoming, final_state = _affine_scan_4(init_state, A_list, B_list, bf16_transfer=bf16_transfer)

    for r in range(world):
        start, end, cu_r, ns_r, nt_r = seg_metas[r]
        pools[r][4].copy_(incoming[r])
        kda_prefill.segment_finalize(
            seg_plan,
            q=q[start:end], k=k[start:end], v=v[start:end],
            raw_g=raw_g[start:end], raw_beta=raw_beta[start:end],
            A_log=A_log, dt_bias=dt_bias, cu_seqlens=cu_r,
            checkpoint_state_indices=idx_zero, checkpoint_offsets=off_zero,
            num_seqs=ns_r, num_tokens=nt_r,
            scratch=scratches[r], pool=pools[r], output=outs[r],
        )

    par_out = torch.cat(outs, dim=0)
    torch.cuda.synchronize()

    state_diff = (final_state - seq_state[1]).abs().max().item()
    out_diff = (par_out[:live_tokens].float() - seq_out[:live_tokens].float()).abs().max().item() if live_tokens else 0.0
    print(f"  [L={live_tokens} fresh={fresh} bf16A={bf16_transfer}] state_max_diff={state_diff:.6f} out_max_diff={out_diff:.6f}")
    assert state_diff < (2e-2 if bf16_transfer else 5e-3), f"state max diff {state_diff}"
    session.close()
    assert out_diff < (3e-2 if bf16_transfer else 2e-2), f"output max diff {out_diff}"
