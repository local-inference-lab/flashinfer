"""Fused router rounding, tie ordering, graph replay, and storage contracts."""

import pytest
import torch

from b12x.moe._router_projection import compile_router


@pytest.mark.parametrize("rows", [1, 3, 6, 17, 128])
@pytest.mark.parametrize(
    "tile,warps,warps_per_expert",
    [(1, 4, 1), (2, 8, 2), (4, 8, 4), (8, 4, 4), (4, 8, 8)],
)
def test_router_projection_replay(rows, tile, warps, warps_per_expert):
    check_router_replay(rows, tile, warps, warps_per_expert, 2560, 512, 10)


@pytest.mark.parametrize(
    "hidden,experts,top_k", [(8, 32, 1), (264, 64, 33), (2560, 1024, 10), (256, 32, 32)]
)
def test_router_projection_geometry(hidden, experts, top_k):
    check_router_replay(3, 4, 8, 4, hidden, experts, top_k)


def check_router_replay(rows, tile, warps, warps_per_expert, hidden, experts, top_k):
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(73)
    weight = (
        torch.randn((experts, hidden), device=device, generator=generator).bfloat16()
        * 0.02
    )
    weight[1::2].copy_(weight[::2])
    source = torch.zeros((rows, hidden), device=device, dtype=torch.bfloat16)
    logits = torch.empty((rows, experts), device=device)
    counters = torch.zeros(
        ((rows + tile - 1) // tile,), device=device, dtype=torch.int32
    )
    selected = torch.empty((rows, top_k), device=device)
    ids = torch.empty((rows, top_k), device=device, dtype=torch.int32)
    scores = torch.empty_like(selected)
    run = compile_router(
        device=device,
        hidden=hidden,
        experts=experts,
        top_k=top_k,
        rows_per_tile=tile,
        warps=warps,
        warps_per_expert=warps_per_expert,
    )
    run(source, weight, logits, counters, selected, ids, scores)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(source, weight, logits, counters, selected, ids, scores)
    try:
        for mode in ("random", "zero", "basis", "random"):
            source.zero_()
            if mode == "random":
                source.copy_(
                    torch.randn(
                        source.shape, device=device, generator=generator
                    ).bfloat16()
                )
            elif mode == "basis":
                source[:, -1] = 1
            # Match FP32 accumulation followed by a single BF16 output rounding.
            reduced_precision = (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            try:
                expected = torch.nn.functional.linear(source, weight).float()
            finally:
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
                    reduced_precision
                )
            order = expected.flip(1).argsort(dim=1, descending=True, stable=True)[
                :, :top_k
            ]
            expected_ids = experts - 1 - order
            expected_selected = expected.gather(1, expected_ids)
            for _ in range(4):
                logits.fill_(float("nan"))
                selected.fill_(float("nan"))
                ids.fill_(-1)
                scores.fill_(float("nan"))
                torch.cuda.synchronize()
                allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats()["allocation.all.allocated"] == allocations
                )
                assert not counters.count_nonzero()
                torch.testing.assert_close(ids, expected_ids.int(), rtol=0, atol=0)
                torch.testing.assert_close(
                    selected, expected_selected, rtol=0.008, atol=0.002
                )
                torch.testing.assert_close(
                    scores, expected_selected.softmax(1), rtol=0.008, atol=0.002
                )
        with pytest.raises(ValueError, match="overlap"):
            run(source, weight, logits, counters, selected, ids, selected)
        unaligned = torch.empty(source.numel() + 1, device=device, dtype=source.dtype)[
            1:
        ].view_as(source)
        with pytest.raises(ValueError, match="aligned"):
            run(unaligned, weight, logits, counters, selected, ids, scores)
    finally:
        graph.reset()


@pytest.mark.parametrize(
    "options",
    [
        {"hidden": 2559},
        {"experts": 513},
        {"top_k": 0},
        {"rows_per_tile": 3},
        {"warps": 3},
        {"warps_per_expert": 3},
        {"warps": 4, "warps_per_expert": 8},
    ],
)
def test_router_projection_rejects_unsupported_geometry(options):
    with pytest.raises(ValueError):
        compile_router(device="cuda", **options)
