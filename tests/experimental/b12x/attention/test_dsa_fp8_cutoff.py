"""FP8 index selection resolves equal scores by logical token position."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((12, 0), (12, 1)),
    reason="FP8 sparse selection requires SM120 or SM121",
)


def check_graph(call, scores, logical, lengths, topk, *, physical=None):
    expected = []
    for row, length in enumerate(lengths):
        candidates = sorted(
            range(length),
            key=lambda slot: (-float(scores[row, slot]), int(logical[row, slot])),
        )[:topk]
        ids = logical[row, candidates]
        if physical is not None:
            ids = physical[row, ids]
        expected.append((ids, scores[row, candidates]))
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, indices = call()
    for _ in range(3):
        graph.replay()
        for row, (ids, reference) in enumerate(expected):
            actual = indices[row].cpu()
            order = actual[actual >= 0].argsort()
            assert int((actual == -1).sum()) == topk - len(ids)
            assert torch.all(actual >= -1)
            torch.testing.assert_close(
                actual[actual >= 0][order], ids[ids.argsort()].int(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                values[row].cpu()[actual >= 0][order],
                reference[ids.argsort()],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("overflow", [False, True])
def test_row_cutoff(topk, fold, overflow):
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

    width = 8704 if overflow else topk + 91
    seed = torch.Generator().manual_seed(568)
    ids = torch.arange(width).expand(4, -1).clone()
    logical = (
        torch.stack([torch.randperm(width, generator=seed) for _ in ids])
        if fold
        else ids
    )
    # Each row has a different live length. The last two rows truncate ties.
    lengths = [0, topk - 1, topk + 1, width]
    scores = torch.full((4, width), -2.0 if overflow else 2.0)
    if not overflow:
        scores[:, : topk - 17] = 3.0
        scores[:, -4:] = -3.0
    values = torch.empty((4, topk), device="cuda")
    output = torch.empty_like(values, dtype=torch.int32)
    args = dict(
        row_logits=scores.cuda(),
        lengths=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        topk=topk,
        output_indices=output,
        output_values=values,
        output_gather_table=logical.int().cuda() if fold else None,
    )
    check_graph(lambda: run_row_topk(**args), scores, logical, lengths, topk)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("ctas,threshold", [(1, 0), (4, 1 << 30), (4, 0)])
@pytest.mark.parametrize("physical", [False, True])
def test_fused_cutoff(topk, ctas, threshold, physical):
    from b12x.attention.dsa_indexer.fused_indexer import run_fused_paged_indexer

    heads = 32
    lengths = [topk - 1, topk + 1, 4 * topk + 37]
    rows = len(lengths)
    pages = (max(lengths) + 63) // 64
    width = pages * 64
    table = torch.arange(pages).roll(pages // 2)[None].expand(rows, -1).contiguous()
    logical = torch.arange(width)[None].expand(rows, -1)
    mapped = table[:, torch.arange(width) // 64] * 64 + logical % 64
    scales = torch.ones(pages, 64)
    scales.flatten()[::13] = 2
    weights = torch.zeros(rows, heads)
    weights[:, 0] = torch.tensor([1.0, -1.0, 1.0])
    scores = weights[:, :1] * scales.flatten()[mapped]
    q = torch.zeros(rows, heads, 128)
    q[:, :, 0] = 1
    k = torch.zeros(pages, 64, 128)
    k[:, :, 0] = 1
    args = dict(
        q_bytes=q.to(torch.float8_e4m3fn).view(torch.uint8).cuda(),
        weights=weights.cuda(),
        k_quant_bytes=k.to(torch.float8_e4m3fn).view(torch.uint8).cuda(),
        k_scales=scales.cuda(),
        real_page_table=table.int().cuda(),
        seqlens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        num_heads=heads,
        topk=topk,
        ctas_per_group=ctas,
        merge_threshold=threshold,
        output_physical_slots=physical,
    )

    def call():
        indices, values = run_fused_paged_indexer(**args)
        return values, indices

    check_graph(
        call, scores, logical, lengths, topk, physical=mapped if physical else None
    )


@pytest.mark.parametrize("topk", [512, 2048])
def test_tiled_cutoff_across_windows(topk):
    from b12x.attention.dsa_indexer.tiled_topk import run_tiled_supertile_topk

    width = topk * 3
    rows, block_q, block_k = 16, 16, 256
    ids = torch.arange(width)[None].expand(rows, -1)
    scores = torch.full((rows, width), -2.0)
    scores[:, ::29] = 7.0
    # Columns of a K tile are adjacent, then the block's Q rows.
    tiles = scores.view(rows, -1, block_k).transpose(0, 1).contiguous().view(-1).cuda()
    lengths = [width - i for i in range(rows)]
    args = dict(
        tile_logits=tiles,
        k_start=torch.zeros(rows, device="cuda", dtype=torch.int32),
        k_end=torch.tensor(lengths, device="cuda", dtype=torch.int32),
        topk=topk,
        block_q=block_q,
        block_k=block_k,
        supertile_k=topk,
    )
    check_graph(lambda: run_tiled_supertile_topk(**args), scores, ids, lengths, topk)


def test_signed_zero_cutoff():
    from b12x.attention.dsa_indexer.tiled_topk import run_row_topk

    topk = 512
    width = topk + 57
    ids = torch.arange(width)[None]
    scores = torch.zeros(1, width)
    scores[:, ::2] = -0.0
    args = dict(
        row_logits=scores.cuda(),
        lengths=torch.tensor([width], device="cuda", dtype=torch.int32),
        topk=topk,
        output_indices=torch.empty((1, topk), device="cuda", dtype=torch.int32),
        output_values=torch.empty((1, topk), device="cuda"),
    )
    check_graph(lambda: run_row_topk(**args), scores, ids, [width], topk)
