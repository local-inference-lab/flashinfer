"""Check logical-prefix and selection addressing against independent byte lookups."""

from dataclasses import asdict

import pytest
import torch

from b12x.comm import peer_record_gather as gather
from b12x.preparation import PreparationSession, PreparedCall


def geometry(**kw):
    values = dict(
        rows=2,
        topk=8,
        world_size=4,
        page_size=4,
        table_columns=3,
        peer_pages=3,
        capacity=16,
        record_bytes=36,
    )
    values.update(kw)
    return gather.InterleavedStageQuery(**values)


@pytest.mark.parametrize(
    "overrides",
    [
        {"capacity": 15},
        {"peer_pages": 2**30},
        {"rows": 0},
        {"record_bytes": 33},
        {"world_size": False},
    ],
)
def test_staging_rejects_invalid_geometry(overrides):
    with pytest.raises(ValueError):
        geometry(**overrides)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float8_e4m3fn])
def test_replayed_prefix_and_direct_selection(dtype):
    from b12x.comm.peer_record_gather._stage import compile_stage

    device = torch.device("cuda", 0)
    q = geometry()
    source = torch.arange(4 * 12 * 36, device=device).to(torch.uint8).view(4, 12, 36)
    # One owner is absent. The third page is invalid and the first two pages are swapped.
    peers = torch.tensor(
        [source[0].data_ptr(), source[1].data_ptr(), 0, source[3].data_ptr()],
        dtype=torch.int64,
        device=device,
    )
    table = torch.tensor([1, 0, -1], device=device, dtype=torch.int32)
    ids = torch.tensor(
        [[0, 3, 8, 5, 6, -1, 1, 2], [4, 7, 18, 15, -1, -1, -1, -1]],
        device=device,
        dtype=torch.int32,
    )
    slots = torch.tensor(
        [[4, 4, 6, 5, 5, -1, 4, 4], [5, 5, 0, 7, -1, -1, -1, -1]],
        device=device,
        dtype=torch.int32,
    )
    counts = torch.tensor([6, 4], device=device, dtype=torch.int32)
    end = torch.tensor([0], device=device, dtype=torch.int64)
    out = torch.empty((q.capacity, q.record_bytes), dtype=dtype, device=device)
    args = (peers, table, end, ids, slots, counts, out)

    def expected(length, fresh, before):
        result = before.clone()

        def get(owner, slot):
            return (
                source[owner, slot].cpu()
                if owner != 2 and 0 <= slot < 12
                else torch.zeros(36, dtype=torch.uint8)
            )

        if length <= q.capacity:
            if fresh:
                for pos in range(max(length, 0)):
                    local = pos // 4
                    result[pos] = get(pos % 4, [1, 0, -1][local // 4] * 4 + local % 4)
        else:
            result.zero_()
            for row in range(2):
                for col in range(int(counts[row])):
                    pos = int(ids[row, col])
                    if 0 <= pos < length:
                        result[row * 8 + col] = get(pos % 4, int(slots[row, col]))
        return result

    for tile in (4, 8, 16):
        state = compile_stage(asdict(q), tile, 0)
        graphs = []
        for first in (True, False):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state.run(*args, first)
            graphs.append(graph)
        for length, which in (
            (0, 0),
            (15, 0),
            (16, 0),
            (16, 1),
            (17, 0),
            (29, 1),
            (-1, 0),
            (3, 0),
        ):
            out.view(torch.uint8).fill_(177)
            before = out.view(torch.uint8).cpu()
            end.fill_(length)
            source.bitwise_xor_(31)
            graphs[which].replay()
            assert torch.equal(
                out.view(torch.uint8).cpu(), expected(length, which == 0, before)
            )
        for graph in graphs:
            graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_normal_preparation_caches_and_captures_stage(tmp_path):
    device = torch.device("cuda", 0)
    q = geometry()
    storage = torch.randint(0, 256, (4, 12, 36), dtype=torch.uint8, device=device)
    peers = torch.tensor(
        [part.data_ptr() for part in storage], dtype=torch.int64, device=device
    )
    table = torch.arange(3, dtype=torch.int32, device=device)
    end = torch.tensor([16], device=device)
    ids = torch.arange(16, dtype=torch.int32, device=device).view(2, 8)
    slots = ids // 4
    counts = torch.full((2,), 8, device=device, dtype=torch.int32)
    out = torch.empty((q.capacity, q.record_bytes), device=device, dtype=torch.uint8)
    args = peers, table, end, ids, slots, counts, out

    def prepare(state):
        return PreparedCall(
            run=lambda: state.run(*args, True),
            produce=lambda: end.fill_(16),
            output=out,
        )

    for expected in ("tuned", "cached"):
        plan = gather.plan_interleaved_stage(q, device=device)
        with PreparationSession(
            device=device, autotune=True, compile_workers=1, cache_dir=tmp_path
        ) as session:
            session.prepare(
                (
                    plan.request(
                        name="stage", prepare_call=prepare, benchmark_call=prepare
                    ),
                )
            )
            assert plan.selection.source == expected
            assert not plan.scratch_specs()

            def forward():
                gather.stage_interleaved_records(plan, *args, first=True)

            compiled = torch.compile(forward, fullgraph=True)
            compiled()
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            end.fill_(17)
            out.zero_()
            graph.replay()
            assert torch.equal(out, storage.permute(1, 0, 2)[:4].reshape_as(out))
            graph.reset()
