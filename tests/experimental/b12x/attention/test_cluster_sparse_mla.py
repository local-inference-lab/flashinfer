"""Cluster-fused BF16 decode preserves packed caches, masks and replay state."""

from dataclasses import replace
import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla._tuning import TUNING
from b12x.attention._shared.mla.compressed_reference import (
    pack_deepseek_v41_cache_reference,
    compressed_sparse_mla_reference,
)
from b12x.preparation import PreparationSession
from tests.experimental.b12x.attention.test_compressed_sparse_mla_v41 import (
    _prepare_binding,
)
from tests.experimental.b12x.conftest import require_b12x


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize("accumulators", [1, 2])
@pytest.mark.parametrize("rows", [1, 3, 19])
@pytest.mark.parametrize("has_sink", [False, True])
def test_cluster_bf16_mapped_decode_replay(rows, has_sink, heads, accumulators):
    device = require_b12x()
    width, page = 512, 64
    q = (torch.randn((rows, heads, 512), device=device) * 0.2).bfloat16()
    swa_values = torch.randn((128, 512), device=device).bfloat16()
    main_values = torch.randn((width, 512), device=device).bfloat16()
    swa_packed = pack_deepseek_v41_cache_reference(
        swa_values, page_size=page, cache_kind="swa"
    )
    main_packed = pack_deepseek_v41_cache_reference(
        main_values, page_size=page, cache_kind="indexed"
    )
    swa = torch.full((3, page * 528), 127, dtype=torch.uint8, device=device)
    main = torch.full((9, page * 288), 127, dtype=torch.uint8, device=device)
    swa[1:].copy_(swa_packed)
    main[1:].copy_(main_packed)
    ids = torch.arange(width, dtype=torch.int32, device=device)[None].repeat(rows, 1)
    ids[:, 300:320] = -1
    swa_ids = (
        torch.arange(128, dtype=torch.int32, device=device)[None].repeat(rows, 1) + page
    )
    lengths = torch.full((rows,), 128, dtype=torch.int32, device=device)
    counts = torch.full((rows,), width, dtype=torch.int32, device=device)
    lengths[0] = counts[0] = 0
    table = torch.arange(1, 9, dtype=torch.int32, device=device)[None].repeat(rows, 1)
    out = torch.empty_like(q)
    sink = torch.randn(heads, dtype=torch.float32, device=device) if has_sink else None
    caps = mla.Caps(
        device=q.device,
        num_q_heads=heads,
        max_q_rows=rows,
        max_width=640,
        swa_width=128,
        indexed_width=width,
        max_page_table_width=8,
        swa_page_size=page,
        indexed_page_size=page,
        cache_format="deepseek_v41",
        mode="decode",
        use_cuda_graph=True,
        compute_mode="bf16",
    )
    invocation = mla.invocation_from_tensors(
        q=q, swa_k_cache=swa, indexed_k_cache=main, attn_sink=sink, out=out
    )
    with PreparationSession(
        device=q.device, autotune=False, compile_workers=2
    ) as session:
        declaration = mla.plan(caps, invocation=invocation)
        config = replace(
            TUNING.configure(declaration.query, device=session.device.identity).default,
            v41_compute_mode="bf16",
            v41_qk_accumulators=accumulators,
            v41_merge="cluster8",
        )
        plan = mla.plan(caps, invocation=invocation, override=config)
        bind = dict(
            q=q,
            swa_indices=swa_ids,
            swa_lengths=lengths,
            indexed_indices=ids,
            indexed_lengths=counts,
            indexed_page_table=table,
        )
        run_args = dict(
            swa_k_cache=swa, indexed_k_cache=main, sm_scale=512**-0.5, attn_sink=sink
        )
        binding, storage = _prepare_binding(session, plan, bind, run_args, out)

        def run():
            return mla.run(
                binding=binding,
                **run_args,
                swa_page_size=page,
                indexed_page_size=page,
                out=out,
            )

        def check():
            physical = (
                table.gather(1, (ids.clamp_min(0) // page).long()) * page + ids % page
            )
            physical.masked_fill_(ids < 0, -1)
            # The reference accepts a valid prefix followed by padding. The
            # decode contract masks each negative ID independently, including
            # holes before later visible IDs; compact only the oracle's inputs.
            compact = torch.full_like(physical, -1)
            valid_counts = torch.zeros_like(counts)
            columns = torch.arange(width, device=device)
            for row in range(rows):
                valid = (columns < counts[row]) & (physical[row] >= 0)
                selected = physical[row, valid]
                compact[row, : selected.numel()] = selected
                valid_counts[row] = selected.numel()
            expected = compressed_sparse_mla_reference(
                q,
                swa,
                swa_ids,
                lengths,
                extra_k_cache=main,
                extra_indices=compact,
                extra_topk_lengths=valid_counts,
                swa_page_size=page,
                extra_page_size=page,
                sm_scale=512**-0.5,
                attn_sink=sink,
                cache_format="deepseek_v41",
            )
            torch.testing.assert_close(out, expected, atol=0.035, rtol=0.035)
            assert torch.equal(out[0], torch.zeros_like(out[0]))

        run()
        check()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        original = (swa.clone(), main.clone())
        for _ in range(3):
            table.copy_(table.flip(1))
            lengths[1:] = 3
            counts[1:] = 7
            storage.fill_(193)
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check()
        assert torch.equal(swa, original[0]) and torch.equal(main, original[1])
        graph.reset()
