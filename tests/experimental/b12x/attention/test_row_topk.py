"""Exact prepared candidate-pair selection and graph replay contracts."""

import pytest
import torch

from b12x.attention import row_topk
from b12x.attention.row_topk._preparation import _validate
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0, "width": 32768},
        {"rows": 1, "width": 0},
        {"rows": 1, "width": 32768, "topk": 64},
        {"rows": 65536, "width": 32768},
        {"rows": 1.5, "width": 32768},
    ],
)
def test_query_rejects_invalid_extents(kwargs):
    with pytest.raises(ValueError):
        row_topk.RowTopKQuery(**kwargs)


def _cpu_inputs():
    return [
        torch.empty(2, 512, dtype=torch.float32),
        torch.empty(2, dtype=torch.int32),
        torch.empty(2, 512, dtype=torch.int32),
        torch.empty(2, 512, dtype=torch.float32),
        torch.empty(2, 512, dtype=torch.int32),
    ]


@pytest.mark.parametrize("index", range(5))
def test_tensor_contract_rejects_dtype_and_shape(index):
    q = row_topk.RowTopKQuery(rows=2, width=512, topk=512)
    values = _cpu_inputs()
    _validate(q, torch.device("cpu"), *values)
    values[index] = values[index].to(torch.float64)
    with pytest.raises(TypeError, match="dtype"):
        _validate(q, torch.device("cpu"), *values)
    values = _cpu_inputs()
    values[index] = values[index][:1]
    with pytest.raises(ValueError, match="shape"):
        _validate(q, torch.device("cpu"), *values)


def test_tensor_contract_rejects_output_alias():
    q = row_topk.RowTopKQuery(rows=2, width=512, topk=512)
    values = _cpu_inputs()
    values[3] = values[0]
    with pytest.raises(ValueError, match="overlap"):
        _validate(q, torch.device("cpu"), *values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_workers", [0, 2])
@pytest.mark.parametrize("rows", [5, 128])
def test_32768_candidate_pair_merge_graph_replay(compile_workers, rows):
    device = torch.device("cuda", 0)
    q = row_topk.RowTopKQuery(rows=rows, width=32768, topk=2048)
    scores = torch.randperm(q.width, device=device).float().repeat(q.rows, 1)
    ids = (torch.arange(q.width, device=device, dtype=torch.int32) * 3 + 7).repeat(
        q.rows, 1
    )
    lengths = torch.tensor(
        ([q.width, 2049, 17, 0, 1] * 26)[:rows], device=device, dtype=torch.int32
    )
    values = torch.empty((q.rows, q.topk), device=device)
    output = torch.empty((q.rows, q.topk), device=device, dtype=torch.int32)
    p = row_topk.plan(q, device=device)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(
        device=device, autotune=False, compile_workers=compile_workers
    ) as session:
        session.prepare(
            (
                p.request(
                    name="candidate_merge",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(scores, lengths, ids, values, output),
                        owners=(scores, lengths, ids, values, output),
                    ),
                ),
            )
        )
        bound = row_topk.bind(p)
        forward = torch.compile(bound.run, fullgraph=True, dynamic=False)
        forward(scores, lengths, ids, values, output)
        with session.capture(), torch.cuda.graph(graph):
            forward(scores, lengths, ids, values, output)
        session.freeze()
        try:
            for pattern in ([q.width, 2049, 17, 0, 1], [0, q.width, 2048, 3, 0]):
                live = (pattern * 26)[:rows]
                lengths.copy_(torch.tensor(live, device=device, dtype=torch.int32))
                scores.mul_(-1)
                ids.add_(11)
                scores[0, 1500:] = -float("inf")
                ids[0, 1500:] = -1
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                for row, count in enumerate(live):
                    n = min(count, q.topk)
                    expected_values, positions = scores[row, :count].topk(n)
                    expected_ids = ids[row].gather(0, positions)
                    expected_valid = expected_ids >= 0
                    valid = output[row] >= 0
                    assert int(valid.sum()) == int(expected_valid.sum())
                    torch.testing.assert_close(
                        values[row][valid].sort().values,
                        expected_values[expected_valid].sort().values,
                        rtol=0,
                        atol=0,
                    )
                    assert torch.equal(
                        output[row][valid].sort().values,
                        expected_ids[expected_valid].sort().values,
                    )
                    assert torch.isneginf(values[row][~valid]).all()
                    assert (output[row][~valid] == -1).all()
        finally:
            graph.reset()
