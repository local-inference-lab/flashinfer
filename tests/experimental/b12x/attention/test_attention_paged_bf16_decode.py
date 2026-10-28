"""BF16 exact-plane decode covers legal MMA layouts and physical page tails."""

import pytest
import torch

from b12x.attention.paged.reference import paged_attention_reference
from b12x.testing.reference.helpers import require_b12x
from b12x.testing.reference.paged_attention_helpers import make_paged_inputs

from b12x.attention import paged
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.parametrize("head_dim", [64, 128, 192])
@torch.inference_mode()
def test_bf16_decode_replay_across_physical_page_boundaries(head_dim):
    require_b12x()
    q, key, value, pages, lengths, cu = make_paged_inputs(
        q_seqlens=[1] * 4,
        cache_seqlens=[127, 128, 129, 2049],
        page_size=128,
        q_heads=16,
        kv_heads=2,
        head_dim=head_dim,
        page_table_width=20,
        num_pages=80,
        seed=9251,
    )
    output = torch.empty_like(q)
    caps = paged.Caps(
        device=q.device,
        mode="decode",
        dtype=q.dtype,
        kv_dtype=key.dtype,
        num_q_heads=16,
        num_kv_heads=2,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        page_size=128,
        max_total_q=4,
        max_batch=4,
        max_page_table_width=20,
        max_work_items=2048,
        max_partial_rows=2048,
        num_cache_pages=80,
        use_cuda_graph=True,
    )
    arguments = dict(
        q=q,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=pages,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
    )
    plan = paged.plan(caps, invocation=paged.invocation_from_tensors(caps, **arguments))
    retained = []

    def prepare(state):
        scratch = [
            torch.empty(spec.shape, dtype=spec.dtype, device=q.device)
            for spec in state.scratch_plan.scratch_specs()
        ]
        binding = state.bind(scratch=scratch, **arguments)
        retained[:] = [scratch, binding]
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(
        device=q.device, autotune=False, compile_workers=1
    ) as session:
        result = session.prepare((plan.request(name="decode", prepare_call=prepare),))
    plan = result.plans["decode"]

    def run():
        paged.run(binding=paged.bind(plan, scratch=retained[0], **arguments), plan=plan)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for counts in ([127, 128, 129, 2049], [128, 129, 130, 2050]):
        lengths.copy_(torch.tensor(counts, device=q.device, dtype=torch.int32))
        reference, _ = paged_attention_reference(
            q, key, value, pages, lengths, cu, causal=True
        )
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(output, reference, atol=0.002, rtol=0.02)
    graph.reset()
    result.close()
