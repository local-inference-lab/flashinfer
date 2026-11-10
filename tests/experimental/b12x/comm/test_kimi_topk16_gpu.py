from __future__ import annotations

import pytest
import torch

from b12x.comm.pcie._dcp_a2a_cute import kimi_topk16


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires SM80 or newer for warp integer reductions",
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(918273)
    logits = torch.randn(8, 896, generator=generator, dtype=torch.float32)
    bias = torch.randn(896, generator=generator, dtype=torch.float32)
    logits[0, 0] = float("nan")
    logits[0, 1] = float("inf")
    logits[0, 2] = -float("inf")
    logits[0, 3] = 0.0
    logits[0, 4] = -0.0
    logits[1, :64] = 0.25
    logits[2, :32] = float("inf")
    logits[3, :] = -float("inf")
    logits[4, :] = float("nan")
    bias[:64] = 0.0
    bias[4] = -0.0
    bias[5] = float("nan")
    bias[6] = float("inf")
    bias[7] = -float("inf")
    return logits, bias


def _reference(
    logits: torch.Tensor, bias: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the ordered selection and FP32 sequential normalization."""
    unbiased = torch.sigmoid(logits[:rows])
    unbiased = torch.where(torch.isfinite(unbiased), unbiased, 0.0)
    selection = unbiased + bias
    selection = torch.where(
        torch.isfinite(selection),
        selection,
        torch.where(selection > 0, float("inf"), -float("inf")),
    )
    selection = torch.where(selection == 0, 0.0, selection)

    all_ids = []
    all_weights = []
    for scores, weights in zip(selection.cpu(), unbiased.cpu(), strict=True):
        # Python's sort is stable, and the secondary key makes lower expert IDs
        # win exact packed-score ties rather than relying on topk tie ordering.
        ids = sorted(
            range(896),
            key=lambda expert: (scores[expert].item(), -expert),
            reverse=True,
        )[:16]
        selected_weights = weights[ids]
        weight_sum = torch.zeros((), dtype=torch.float32)
        for weight in selected_weights:
            weight_sum = weight_sum + weight
        denominator = weight_sum if weight_sum > 0 else torch.ones_like(weight_sum)
        all_ids.append(torch.tensor(ids, dtype=torch.int32))
        all_weights.append(selected_weights * (1.0 / denominator))
    return torch.stack(all_weights), torch.stack(all_ids)


def _launch(logits, bias, weights, ids, rows: int, threads: int) -> None:
    kimi_topk16(
        router_logits_ptr=logits.data_ptr(),
        correction_bias_ptr=bias.data_ptr(),
        output_weights_ptr=weights.data_ptr(),
        output_ids_ptr=ids.data_ptr(),
        rows=rows,
        threads=threads,
    )


@pytest.mark.parametrize("rows", (1, 4, 8))
@pytest.mark.parametrize("threads", (128, 256, 512))
def test_two_level_topk16_matches_reference_and_graph_replay(rows, threads):
    logits, bias = (tensor.cuda() for tensor in _inputs())
    weights = torch.empty((rows, 16), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, 16), dtype=torch.int32, device="cuda")
    expected_weights, expected_ids = _reference(logits, bias, rows)

    def check_outputs():
        torch.cuda.synchronize()
        assert torch.equal(ids.cpu(), expected_ids)
        torch.testing.assert_close(
            weights.cpu(), expected_weights, rtol=2e-5, atol=2e-6
        )

    _launch(logits, bias, weights, ids, rows, threads)
    check_outputs()
    eager_weights = weights.clone()
    eager_ids = ids.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(logits, bias, weights, ids, rows, threads)
    weights.fill_(float("nan"))
    ids.fill_(-1)
    graph.replay()
    check_outputs()
    # Replay must be bit-identical to the eager launch, including sigmoid and
    # the sequential normalization, rather than merely close to the reference.
    assert torch.equal(weights, eager_weights)
    assert torch.equal(ids, eager_ids)
    graph.reset()


@pytest.mark.parametrize("threads", (128, 256, 512))
@pytest.mark.parametrize(
    "kind", ("finite_ties", "negative_infinity", "merge_exhaustion", "lane_exhaustion")
)
def test_topk16_never_selects_an_exhausted_candidate(kind, threads):
    logits = torch.zeros((1, 896), dtype=torch.float32, device="cuda")
    bias = torch.full((896,), -float("inf"), dtype=torch.float32, device="cuda")
    if kind == "finite_ties":
        bias.zero_()
        expected = list(range(16))
    elif kind == "negative_infinity":
        expected = list(range(16))
    elif kind == "merge_exhaustion":
        bias[0], bias[512] = 0.0, 1.0
        expected = [512, 0, *range(1, 15)]
    else:
        selected = list(range(0, 256, 32))
        bias[selected] = torch.arange(8, dtype=torch.float32, device="cuda")
        expected = [*reversed(selected), *range(1, 9)]
    weights = torch.empty((1, 16), dtype=torch.float32, device="cuda")
    ids = torch.empty((1, 16), dtype=torch.int32, device="cuda")
    _launch(logits, bias, weights, ids, 1, threads)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(logits, bias, weights, ids, 1, threads)
    ids.fill_(-1)
    weights.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert ids.cpu().tolist() == [expected]
    assert torch.equal(weights, torch.full_like(weights, 1.0 / 16))
    graph.reset()
