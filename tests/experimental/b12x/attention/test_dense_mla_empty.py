"""Empty context rows overwrite prior final output under captured replay."""

import pytest
import torch

from b12x.attention import dense_mla
from b12x.attention.dense_mla._tuning import DenseMlaConfig

from .test_dense_mla import _bind, _prepared_scope
from ..conftest import require_b12x


@pytest.mark.parametrize("splits", [1, 4])
@torch.no_grad()
def test_empty_rows_after_populated_replay(splits):
    device = require_b12x()
    rows, heads, records = 8, 96, 1024
    query = torch.full((rows, heads, 576), 0.125, device=device, dtype=torch.bfloat16)
    cache = torch.full(
        (records, 1, 576), 0.25, device=device, dtype=torch.float8_e4m3fn
    )
    output = torch.empty((rows, heads, 512), device=device, dtype=torch.bfloat16)
    pages = torch.arange(records, device=device, dtype=torch.int32)[None].repeat(
        rows, 1
    )
    lengths = torch.full((rows,), records, device=device, dtype=torch.int32)
    starts = torch.arange(rows + 1, device=device, dtype=torch.int32)
    plan = dense_mla.plan(
        dense_mla.Caps(
            device=device,
            mode="decode",
            q_dtype=query.dtype,
            kv_dtype=cache.dtype,
            num_q_heads=heads,
            page_size=1,
            max_total_q=rows,
            max_batch=rows,
            max_cache_tokens=records,
            max_page_table_width=records,
            num_cache_pages=records,
            use_cuda_graph=True,
            partial_dtype=torch.float32,
        ),
        override=DenseMlaConfig(max_splits=splits),
    )
    scale = torch.ones(1, device=device)
    with _prepared_scope():
        binding = _bind(
            plan,
            q=query,
            kv_cache=cache,
            output=output,
            page_table=pages,
            cache_seqlens=lengths,
            cu_seqlens_q=starts,
            kv_scale=scale,
            q_scale=scale,
            sm_scale=192**-0.5,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _, lse = dense_mla.run(binding=binding)
        for count in (64, 0, 1, 0):
            lengths[::2] = count
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                output[1::2], torch.full_like(output[1::2], 0.25), atol=0.002, rtol=0.01
            )
            if count == 0:
                assert not output[::2].count_nonzero()
                assert torch.isneginf(lse[::2]).all()
            else:
                torch.testing.assert_close(
                    output[::2],
                    torch.full_like(output[::2], 0.25),
                    atol=0.002,
                    rtol=0.01,
                )
        graph.reset()
