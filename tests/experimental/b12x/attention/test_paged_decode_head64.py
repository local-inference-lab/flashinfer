"""Head64 paged decode and noncausal blocks preserve BF16 and FP8 cache layouts."""

import pytest
import torch

from b12x.attention import paged_decode
from b12x.attention.paged.reference import paged_attention_reference
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@torch.no_grad()
def test_head64_captured_ragged_blocks(batch, width, causal, kv_dtype):
    device = require_b12x()
    torch.manual_seed(1713)
    pages = 17
    key = torch.randn(batch * pages, 128, 2, 64, device=device).bfloat16() * 0.1
    value = torch.randn_like(key) * 0.1
    key, value = key.to(kv_dtype), value.to(kv_dtype)
    query = torch.randn(batch * width, 16, 64, device=device).bfloat16() * 0.1
    output = torch.empty_like(query)
    table = torch.arange(batch * pages, device=device, dtype=torch.int32).view(
        batch, pages
    )
    lengths = torch.full((batch,), 2056, device=device, dtype=torch.int32)
    cu = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    args = dict(
        q=query,
        k_cache=key,
        v_cache=value,
        output=output,
        page_table=table,
        cache_seqlens=lengths,
        cu_seqlens_q=cu,
    )
    if kv_dtype == torch.float8_e4m3fn:
        args.update(
            k_descale=torch.ones(1, device=device),
            v_descale=torch.ones(1, device=device),
        )
    plan = paged_decode.plan(
        paged_decode.Caps(
            device=device,
            num_q_heads=16,
            num_kv_heads=2,
            head_dim_qk=64,
            head_dim_vo=64,
            page_size=128,
            max_batch=batch,
            max_q_per_req=width,
            window_left=2047,
            causal=causal,
            kv_dtype=kv_dtype,
        )
    )
    scratch = None
    with PreparationSession(device=device, autotune=False) as session:

        def prepare(state):
            nonlocal scratch
            scratch = [
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in plan.scratch_specs()
            ]
            binding = state.bind(plan=plan, scratch=scratch, **args)
            return PreparedCall(
                run=lambda: state.run(binding), output=output, owners=(scratch, binding)
            )

        result = session.prepare((plan.request(name="head64", prepare_call=prepare),))
        binding = paged_decode.bind(plan, scratch=scratch, **args)
        paged_decode.run(binding)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            paged_decode.run(binding)
        for endpoint in (127, 2048, 2056):
            counts = [width if i % 2 == 0 else max(1, width - 3) for i in range(batch)]
            if endpoint == 2048 and batch > 1:
                counts[1] = 0
            cu.copy_(
                torch.tensor(
                    [0, *torch.tensor(counts).cumsum(0).tolist()],
                    device=device,
                    dtype=torch.int32,
                )
            )
            lengths.fill_(endpoint)
            query.normal_(std=0.1)
            output.fill_(73)
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for name in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[name] == after[name]
            reference, _ = paged_attention_reference(
                query,
                key,
                value,
                table,
                lengths,
                cu,
                causal=causal,
                window_left=2047,
                k_descale=torch.ones(batch, device=device)
                if kv_dtype == torch.float8_e4m3fn
                else None,
                v_descale=torch.ones(batch, device=device)
                if kv_dtype == torch.float8_e4m3fn
                else None,
            )
            live = sum(counts)
            torch.testing.assert_close(
                output[:live], reference[:live], atol=0.02, rtol=0.02
            )
            cosine = torch.nn.functional.cosine_similarity(
                output[:live].float().reshape(-1),
                reference[:live].float().reshape(-1),
                dim=0,
            ).item()
            assert cosine >= 0.9999, cosine
            assert torch.all(output[live:] == 73)
            saved = output.clone()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(saved, output)
        graph.reset()
        result.close()
