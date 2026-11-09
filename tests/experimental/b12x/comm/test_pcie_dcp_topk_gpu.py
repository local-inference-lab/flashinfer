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

    def prepare(runtime, rows, name):
        indices, scores = _inputs(0, rank, rows, device)
        call = dict(local_indices=indices, local_scores=scores)
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
        reserved = {rows: graph_owner.graph_candidate_views(rows) for rows in test_rows}
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
        graph_cases = {}
        for graph_rows in test_rows:
            graph_plan = graph_plans[graph_rows]
            graph_indices, graph_scores = _inputs(10, rank, graph_rows, device)
            shape = (-(-graph_rows // dcp_world_size), dcp_world_size * TOPK)
            consumed_indices = torch.empty(shape, dtype=torch.int32, device=device)
            consumed_scores = torch.empty(shape, dtype=torch.float32, device=device)
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            dist.barrier()
            with (
                session.capture(),
                graph_owner.capture(plan=graph_plan),
                torch.cuda.graph(graph),
            ):
                candidates = graph_owner.stage_candidates(
                    graph_indices, graph_scores, plan=graph_plan
                )
                if dcp_rank == 0:
                    torch.cuda._sleep(20_000_000)
                consumed_indices.copy_(candidates[0])
                consumed_scores.copy_(candidates[1])
            assert _cuda_graph_kernel_count(graph) == 1 + int(dcp_rank == 0)
            assert tuple(t.data_ptr() for t in candidates) == tuple(
                t.data_ptr() for t in reserved[graph_rows]
            )
            graph_cases[graph_rows] = (
                graph,
                graph_indices,
                graph_scores,
                consumed_indices,
                consumed_scores,
            )

        # Alternate captured sizes without host/device synchronization. A
        # fixed-slot write must wait for every preceding consumer, including
        # peers whose owner partition was empty in the preceding invocation.
        row_schedule = (max_rows, 1, 8, 3, 17, 4, 6, max_rows - 1, max_rows, 3, 8, 1)
        replay_inputs = [
            _inputs(20 + index, rank, rows, device)
            for index, rows in enumerate(row_schedule)
        ]
        snapshots = [
            (
                torch.empty_like(graph_cases[rows][3]),
                torch.empty_like(graph_cases[rows][4]),
            )
            for rows in row_schedule
        ]
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        for rows, inputs, saved in zip(row_schedule, replay_inputs, snapshots):
            graph, indices, scores, consumed_indices, consumed_scores = graph_cases[
                rows
            ]
            indices.copy_(inputs[0])
            scores.copy_(inputs[1])
            consumed_indices.fill_(-123)
            consumed_scores.view(torch.int32).fill_(0x12345678)
            graph.replay()
            saved[0].copy_(consumed_indices)
            saved[1].copy_(consumed_scores)
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
        for index, (rows, saved) in enumerate(zip(row_schedule, snapshots)):
            peers = [
                _inputs(20 + index, source, rows, device) for source in dcp_global_ranks
            ]
            expected = owner_stage_reference(
                torch.stack([item[0] for item in peers]),
                torch.stack([item[1] for item in peers]),
                dcp_rank,
            )
            assert torch.equal(saved[0], expected[0])
            assert torch.equal(
                saved[1].view(torch.int32), expected[1].view(torch.int32)
            )
        for graph, *_ in graph_cases.values():
            graph.reset()
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
