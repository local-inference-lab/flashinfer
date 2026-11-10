from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import runtime as cudart

from b12x.comm.pcie import _owner_preparation as preparation
from b12x.preparation import PreparationSession

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    owner_stage_reference,
)


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_TOPK_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_TOPK_TEST=1 to run GPU tests",
)

MAX_ROWS = 64
TOPK = 2048


def _cuda_graph_kernel_count(graph: torch.cuda.CUDAGraph) -> int:
    graph_handle = graph.raw_cuda_graph()
    result, _, num_nodes = cudart.cudaGraphGetNodes(graph_handle)
    assert result == cudart.cudaError_t.cudaSuccess
    result, nodes, returned_nodes = cudart.cudaGraphGetNodes(
        graph_handle,
        num_nodes,
    )
    assert result == cudart.cudaError_t.cudaSuccess
    assert returned_nodes == num_nodes
    kernel_type = cudart.cudaGraphNodeType.cudaGraphNodeTypeKernel
    kernel_count = 0
    for node in nodes[:num_nodes]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        assert result == cudart.cudaError_t.cudaSuccess
        kernel_count += node_type == kernel_type
    return kernel_count


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _inputs(
    step: int, global_rank: int, rows: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
    indices = (base + global_rank * 1_000_000 + step * 10_000).to(device)
    score_bits = (
        torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK)
        + 0x3E800000
        + global_rank * 4096
        + step * 64
    )
    score_bits.view(-1)[:4] = torch.tensor(
        [0x7FC01234, -2147483648, -8388608, 0x7F800000], dtype=torch.int32
    )
    return indices, score_bits.to(device).view(torch.float32)


def _counts(
    step: int, global_rank: int, rows: int, device: torch.device
) -> torch.Tensor:
    values = torch.tensor([0, 1, 3, 7, 31, 2047, 2048, -4, 4096], dtype=torch.int32)
    indices = (torch.arange(rows) + step + global_rank).remainder(values.numel())
    return values[indices].to(device)


def _graph_inputs(step, rank, rows, device, pattern):
    indices, scores = _inputs(step, rank, rows, device)
    if pattern is None:
        return indices, scores
    if pattern == "short":
        counts = (torch.arange(rows, dtype=torch.int32, device=device) + rank) % 7 + 1
    else:
        length = -(rank % 2) if pattern == "zero" else TOPK + rank % 2
        counts = torch.full((rows,), length, dtype=torch.int32, device=device)
    if pattern == "exceptional":
        indices[:, 0] = -1
        scores[:, 0] = -torch.inf
        scores.view(torch.int32)[:, -1] = (
            0x7FC01234,
            0x7F800000,
            -8388608,
            -2147483648,
        )[rank % 4]
        if rank % 4 == 2:
            indices[:, -1] = -2
    return indices, scores, counts


def _assert_staged(staged, peers, owner_rank):
    expected_indices, expected_scores = owner_stage_reference(
        torch.stack([item[0] for item in peers]),
        torch.stack([item[1] for item in peers]),
        owner_rank,
    )
    indices, scores, *counts = staged
    if counts:
        expected_counts = torch.zeros_like(counts[0])
        owner_rows = expected_counts.shape[0]
        start = owner_rank * owner_rows
        local_counts = torch.stack(
            [item[2][start : start + owner_rows] for item in peers]
        )
        expected_counts[: local_counts.shape[1]].copy_(local_counts.T.clamp(0, TOPK))
        assert torch.equal(counts[0], expected_counts)
        # A vectorized copy can also write extra words in the final pack.
        # Only the words addressed by the received counts have defined values.
        columns = torch.arange(TOPK, device=indices.device)
        live = (columns < expected_counts[..., None]).flatten(1)
        assert torch.equal(indices[live], expected_indices[live])
        assert torch.equal(
            scores.view(torch.int32)[live], expected_scores.view(torch.int32)[live]
        )
    else:
        assert torch.equal(indices, expected_indices)
        assert torch.equal(scores.view(torch.int32), expected_scores.view(torch.int32))


def _check_graph_replay(runtime, plans, session, schedule, rank, group_ranks):
    device = runtime.device
    cases = {}
    try:
        for rows, pattern in dict(schedule).items():
            plan = plans[rows]
            inputs = _graph_inputs(10, rank, rows, device, pattern)
            reserved = runtime.graph_candidate_views(rows)
            if pattern is not None:
                reserved += (runtime.graph_count_views(rows),)
            consumed = tuple(torch.empty_like(view) for view in reserved)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            cases[rows] = graph, inputs, consumed
            dist.barrier()
            with session.capture(), runtime.capture(plan=plan), torch.cuda.graph(graph):
                staged = runtime.stage_candidates(*inputs, plan=plan)
                if runtime.rank == 0:
                    torch.cuda._sleep(20_000_000)
                for target, source in zip(consumed, staged, strict=True):
                    target.copy_(source)
            assert _cuda_graph_kernel_count(graph) == 1 + int(runtime.rank == 0)
            assert tuple(t.data_ptr() for t in staged) == tuple(
                t.data_ptr() for t in reserved
            )

        # Queue graphs with changing capacities and counts. The shared slot
        # cannot be reused until the slow owner's preceding consumer retires.
        replay_inputs = [
            _graph_inputs(20 + step, rank, rows, device, pattern)
            for step, (rows, pattern) in enumerate(schedule)
        ]
        snapshots = [
            tuple(torch.empty_like(view) for view in cases[rows][2])
            for rows, _ in schedule
        ]
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        for (rows, _), sources, saved in zip(
            schedule, replay_inputs, snapshots, strict=True
        ):
            graph, inputs, consumed = cases[rows]
            for target, source in zip(inputs, sources, strict=True):
                target.copy_(source)
            for target in consumed:
                target.view(torch.int32).fill_(0x12345678)
            graph.replay()
            for target, source in zip(saved, consumed, strict=True):
                target.copy_(source)
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
        for step, ((rows, pattern), saved) in enumerate(
            zip(schedule, snapshots, strict=True)
        ):
            peers = [
                _graph_inputs(20 + step, source, rows, device, pattern)
                for source in group_ranks
            ]
            _assert_staged(saved, peers, runtime.rank)
    finally:
        for graph, *_ in cases.values():
            graph.reset()


def _dcp_groups(tp_world_size: int, dcp_world_size: int):
    return [
        dist.new_group(
            list(range(start, start + dcp_world_size)),
            backend="nccl",
        )
        for start in range(0, tp_world_size, dcp_world_size)
    ]


def _worker(
    rank: int,
    tp_world_size: int,
    dcp_world_size: int,
    port: int,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=tp_world_size,
    )
    groups = _dcp_groups(tp_world_size, dcp_world_size)
    dcp_partition = rank // dcp_world_size
    dcp_rank = rank % dcp_world_size
    dcp_group = groups[dcp_partition]
    dcp_global_ranks = list(
        range(
            dcp_partition * dcp_world_size,
            (dcp_partition + 1) * dcp_world_size,
        )
    )

    max_rows = MAX_ROWS
    test_rows = sorted(
        {
            1,
            3,
            4,
            6,
            8,
            17,
            max_rows - 1,
            max_rows,
        }
    )
    owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )
    graph_owner = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=dcp_group,
        device=device,
        max_rows=max_rows,
        topk=TOPK,
    )

    def barrier(key, ranks):
        assert tuple(ranks) == tuple(dcp_global_ranks)
        with torch.cuda.device(device):
            dist.barrier(group=dcp_group, device_ids=[rank])

    session = PreparationSession(
        device=device,
        autotune=False,
        compile_workers=1,
        collective_barrier=barrier,
    )

    def prepare(runtime, rows, name, counts=None):
        indices, scores = _inputs(0, rank, rows, device)
        call = dict(local_indices=indices, local_scores=scores, local_counts=counts)
        query = preparation.query_from_runtime(runtime, call=call)
        plan = preparation.plan(query, runtime=runtime)
        session.prepare(
            (
                plan.request(
                    name=name,
                    collective=preparation.collective(runtime, key=name),
                    prepare_call=lambda state: preparation.prepared_call(state, **call),
                ),
            ),
            coordinator=lambda state: (
                state.ready_collectives[0].key if state.ready_collectives else None
            ),
        )
        return plan

    try:
        eager_plans = {
            rows: prepare(owner, rows, f"topk-eager-{rows}") for rows in test_rows
        }
        graph_plans = {
            rows: prepare(graph_owner, rows, f"topk-graph-{rows}") for rows in test_rows
        }
        counted_plans = {
            rows: prepare(
                owner, rows, f"topk-counted-{rows}", _counts(0, rank, rows, device)
            )
            for rows in test_rows
        }
        counted_graph_plans = {
            rows: prepare(
                graph_owner,
                rows,
                f"topk-counted-graph-{rows}",
                _counts(0, rank, rows, device),
            )
            for rows in test_rows
        }
        session.freeze()
        eager_output_ptrs = []
        for step, rows in enumerate(test_rows):
            local_indices, local_scores = _inputs(step, rank, rows, device)
            wrong_device = (rank + 1) % tp_world_size
            if step == 0:
                torch.cuda.set_device(wrong_device)
            candidate_indices, candidate_scores = owner.stage_candidates(
                local_indices, local_scores, plan=eager_plans[rows]
            )
            if step == 0:
                assert torch.cuda.current_device() == wrong_device
                torch.cuda.set_device(rank)
            eager_output_ptrs.append(candidate_indices.data_ptr())
            torch.cuda.synchronize(device)

            rank_inputs = [
                _inputs(step, source, rows, device) for source in dcp_global_ranks
            ]
            expected_indices, expected_scores = owner_stage_reference(
                torch.stack([item[0] for item in rank_inputs]),
                torch.stack([item[1] for item in rank_inputs]),
                dcp_rank,
            )
            if not torch.equal(candidate_indices, expected_indices):
                mismatch = (candidate_indices != expected_indices).nonzero()
                details = [
                    (
                        tuple(int(value) for value in index.tolist()),
                        int(candidate_indices[tuple(index)].item()),
                        int(expected_indices[tuple(index)].item()),
                    )
                    for index in mismatch[:16]
                ]
                raise AssertionError(
                    f"rank={rank} step={step} rows={rows} index mismatches={details}"
                )
            torch.testing.assert_close(
                candidate_indices, expected_indices, rtol=0, atol=0
            )
            assert torch.equal(
                candidate_scores.view(torch.int32),
                expected_scores.view(torch.int32),
            )
        assert eager_output_ptrs[0] != eager_output_ptrs[1]
        assert eager_output_ptrs[0] == eager_output_ptrs[2]
        assert eager_output_ptrs[1] == eager_output_ptrs[3]
        for step, rows in enumerate(test_rows, start=40):
            local_indices, local_scores = _inputs(step, rank, rows, device)
            local_counts = _counts(step, rank, rows, device)
            staged = owner.stage_candidates(
                local_indices, local_scores, local_counts, plan=counted_plans[rows]
            )
            peers = [
                (
                    *_inputs(step, source, rows, device),
                    _counts(step, source, rows, device),
                )
                for source in dcp_global_ranks
            ]
            _assert_staged(staged, peers, dcp_rank)

        row_schedule = (max_rows, 1, 8, 3, 17, 4, 6, max_rows - 1, max_rows, 3, 8, 1)
        _check_graph_replay(
            graph_owner,
            graph_plans,
            session,
            tuple((rows, None) for rows in row_schedule),
            rank,
            dcp_global_ranks,
        )
        counted_schedule = (
            (max_rows, "long"),
            (1, "short"),
            (max_rows, "zero"),
            (3, "exceptional"),
            (8, "long"),
            (8, "short"),
            (8, "zero"),
            (8, "exceptional"),
            (17, "short"),
            (max_rows - 1, "long"),
            (4, "zero"),
            (6, "exceptional"),
        )
        _check_graph_replay(
            graph_owner,
            counted_graph_plans,
            session,
            counted_schedule,
            rank,
            dcp_global_ranks,
        )
        dist.barrier()
        torch.cuda.synchronize(device)
    finally:
        session.close()
        graph_owner.close_coordinated()
        owner.close_coordinated()
        dist.destroy_process_group()


def test_pcie_dcp_topk_exact_owner_exchange():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp_world_size = int(os.getenv("B12X_PCIE_DCP_TOPK_TP", "8"))
    dcp_world_size = int(os.getenv("B12X_PCIE_DCP_TOPK_DCP", "4"))
    if (
        tp_world_size not in (2, 3, 4, 6, 8, 16)
        or dcp_world_size not in (2, 3, 4, 6, 8, 16)
        or tp_world_size % dcp_world_size != 0
    ):
        pytest.skip("unsupported TP/DCP top-k geometry")
    if torch.cuda.device_count() < tp_world_size:
        pytest.skip(f"need {tp_world_size} CUDA devices")
    mp.spawn(
        _worker,
        args=(tp_world_size, dcp_world_size, _free_port()),
        nprocs=tp_world_size,
        join=True,
    )
