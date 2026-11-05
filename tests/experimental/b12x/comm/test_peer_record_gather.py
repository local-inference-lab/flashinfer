"""Byte-exact peer storage reads with masked metadata and graph replay."""

import pytest
import torch

from b12x.comm import peer_record_gather
from b12x.comm.peer_record_gather._preparation import _validate
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 0},
        {"topk": -1},
        {"world_size": 0},
        {"peer_records": 0},
        {"record_bytes": 0},
        {"rows": 2**30},
        {"peer_records": 2**31},
        {"record_bytes": 1.5},
    ],
)
def test_query_rejects_invalid_extents(kwargs):
    options = dict(rows=2, topk=8, world_size=4, peer_records=17, record_bytes=576)
    options.update(kwargs)
    with pytest.raises(ValueError):
        peer_record_gather.PeerRecordGatherQuery(**options)


def _cpu_inputs():
    return [
        torch.empty(4, dtype=torch.int64),
        torch.empty(2, 8, dtype=torch.int32),
        torch.empty(2, 8, dtype=torch.int32),
        torch.empty(2, dtype=torch.int32),
        torch.empty(2, 8, 576, dtype=torch.uint8),
    ]


@pytest.mark.parametrize("index", range(5))
def test_tensor_contract_rejects_dtype_and_shape(index):
    q = peer_record_gather.PeerRecordGatherQuery(
        rows=2, topk=8, world_size=4, peer_records=17
    )
    values = _cpu_inputs()
    _validate(q, torch.device("cpu"), *values)
    values[index] = values[index].to(torch.float32)
    with pytest.raises(TypeError, match="dtype"):
        _validate(q, torch.device("cpu"), *values)
    values = _cpu_inputs()
    values[index] = values[index][:1]
    with pytest.raises(ValueError, match="shape"):
        _validate(q, torch.device("cpu"), *values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "output_dtype", [torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("compile_workers", [0, 2])
def test_exact_records_invalid_tails_and_graph_replay(output_dtype, compile_workers):
    device = torch.device("cuda", 0)
    q = peer_record_gather.PeerRecordGatherQuery(
        rows=3, topk=8, world_size=4, peer_records=17
    )
    storage = torch.randint(
        0, 256, (2, q.peer_records, q.record_bytes), device=device, dtype=torch.uint8
    )
    bases = torch.tensor(
        [storage[0].data_ptr(), storage[1].data_ptr(), 0, 1],
        dtype=torch.int64,
        device=device,
    )
    owners = torch.tensor(
        [[0, 1, -1, 4, 0, 1, 2, 3], [3] * 8, [1, 0, 1, 0, 1, 0, 1, 0]],
        device=device,
        dtype=torch.int32,
    )
    slots = torch.tensor(
        [[0, 16, 0, 0, -1, 17, 1, -1], [0] * 8, list(range(8))],
        device=device,
        dtype=torch.int32,
    )
    counts = torch.tensor([8, 0, 3], device=device, dtype=torch.int32)
    out = torch.empty(
        (q.rows, q.topk, q.record_bytes), dtype=output_dtype, device=device
    )
    p = peer_record_gather.plan(q, device=device)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(
        device=device, autotune=False, compile_workers=compile_workers
    ) as session:
        session.prepare(
            (
                p.request(
                    name="peer_records",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(bases, owners, slots, counts, out),
                        owners=(storage, bases, owners, slots, counts, out),
                    ),
                ),
            )
        )
        forward = torch.compile(
            peer_record_gather.bind(p).run, fullgraph=True, dynamic=False
        )
        forward(bases, owners, slots, counts, out)
        with session.capture(), torch.cuda.graph(graph):
            forward(bases, owners, slots, counts, out)
        session.freeze()
        try:
            for live in (3, 8, -1):
                counts[2] = live
                storage.bitwise_xor_(37)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == before
                )
                expected = torch.zeros(
                    (q.rows, q.topk, q.record_bytes), dtype=torch.uint8
                )
                expected[0, 0] = storage[0, 0].cpu()
                expected[0, 1] = storage[1, 16].cpu()
                for column in range(max(0, live)):
                    expected[2, column] = storage[1 - column % 2, column].cpu()
                assert torch.equal(out.view(torch.uint8).cpu(), expected)
        finally:
            graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_128_rows_2048_records_16_peers_graph_replay():
    device = torch.device("cuda", 0)
    q = peer_record_gather.PeerRecordGatherQuery(
        rows=128,
        topk=2048,
        world_size=16,
        peer_records=32,
    )
    storage = torch.randint(
        0,
        256,
        (q.world_size, q.peer_records, q.record_bytes),
        device=device,
        dtype=torch.uint8,
    )
    bases = torch.tensor(
        [storage[rank].data_ptr() for rank in range(q.world_size)],
        dtype=torch.uint64,
        device=device,
    )
    owners = (
        torch.arange(q.rows * q.topk, dtype=torch.int32, device=device).reshape(
            q.rows, q.topk
        )
        % q.world_size
    )
    slots = (
        torch.arange(q.rows * q.topk, dtype=torch.int32, device=device).reshape(
            q.rows, q.topk
        )
        % q.peer_records
    )
    counts = torch.full((q.rows,), q.topk, dtype=torch.int32, device=device)
    counts[::3] = 0
    counts[1::3] = 17
    out = torch.empty(
        (q.rows, q.topk, q.record_bytes),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    p = peer_record_gather.plan(q, device=device)
    graph = torch.cuda.CUDAGraph()
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                p.request(
                    name="peer_records",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(bases, owners, slots, counts, out),
                        owners=(storage, bases, owners, slots, counts, out),
                    ),
                ),
            )
        )
        with session.capture(), torch.cuda.graph(graph):
            peer_record_gather.gather(p, bases, owners, slots, counts, out)
        session.freeze()
        try:
            storage.bitwise_xor_(123)
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            expected = storage[owners.long(), slots.long()]
            expected[
                torch.arange(q.topk, device=device)[None, :] >= counts[:, None]
            ] = 0
            assert torch.equal(out.view(torch.uint8), expected)
        finally:
            graph.reset()
