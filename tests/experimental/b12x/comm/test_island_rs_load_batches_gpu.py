"""Exercise TP16 reduction tails and changing inputs through prepared graphs."""

from datetime import timedelta
import os
from pathlib import Path
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ISLAND_RS_TEST") != "1",
    reason="requires sixteen peer-accessible GPUs",
)


def _input(elements, rank, variation, device):
    generator = torch.Generator(device=device).manual_seed(
        19731 + elements + variation * 103 + rank * 701
    )
    values = torch.randn(elements, device=device, generator=generator)
    # Alternate large and small cancelling contributions without overflow.
    values.mul_(16.0 if rank % 4 < 2 else 0.125)
    values.add_((1 if rank % 2 == 0 else -1) * (32 + variation))
    return values.bfloat16()


def _ordered_reference(elements, variation, device):
    result = torch.zeros(elements, dtype=torch.float32, device=device)
    for island in range(4):
        partial = torch.zeros_like(result)
        for rank in range(island * 4, island * 4 + 4):
            partial.add_(_input(elements, rank, variation, device).float())
        result.add_(partial.bfloat16().float())
    return result.bfloat16()


def _worker(rank, rendezvous):
    from dataclasses import replace

    from b12x.comm.pcie import _owner_preparation as preparation
    from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce
    from b12x.preparation import FrozenMapping, PreparationSession

    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=16,
        init_method=rendezvous,
        timeout=timedelta(minutes=10),
    )
    capacity = 1024 * 7168
    try:
        # Use a separate fixed-grid owner for each geometry. Changing the
        # physical grid would stop advancing some per-CTA generation counters.
        for blocks, threads in ((16, 512), (32, 1024)):
            stride = blocks * threads
            shapes = (
                2,
                6,
                7168,
                24 * stride - 2,
                24 * stride + 2,
                32 * stride - 2,
                32 * stride + 2,
                capacity - 2,
            )
            owner = PCIeIslandRSAllReduce(
                exchange_group=dist.group.WORLD,
                device=device,
                max_elements=capacity,
                blocks=blocks,
            )
            session = PreparationSession(
                device=device,
                autotune=False,
                compile_workers=4,
            )
            cases = []
            graphs = []
            try:
                for elements in shapes:
                    # A four-byte offset exercises the documented alignment
                    # contract rather than relying on allocator alignment.
                    storage = torch.empty(
                        elements + 2, dtype=torch.bfloat16, device=device
                    )
                    source = storage[2:]
                    inputs = [_input(elements, rank, n, device) for n in range(3)]
                    expected = [
                        _ordered_reference(elements, n, device) for n in range(3)
                    ]
                    outputs = [torch.empty_like(source) for _ in range(3)]
                    plans = []
                    for index, output in enumerate(outputs):
                        query = preparation.query_from_runtime(
                            owner,
                            call=dict(inp=source, out=output, blocks=blocks),
                        )
                        query = replace(
                            query,
                            call=FrozenMapping({**query.call, "threads": threads}),
                        )
                        plan = preparation.plan(query, runtime=owner)
                        session.prepare(
                            (
                                plan.request(
                                    name=f"reduce-{elements}-{index}",
                                    prepare_call=lambda state,
                                    source=source,
                                    output=output: preparation.prepared_call(
                                        state,
                                        inp=source,
                                        out=output,
                                    ),
                                ),
                            )
                        )
                        plans.append(plan)
                    with session.capture():
                        for values, output, plan in zip(
                            inputs, outputs, plans, strict=True
                        ):
                            source.copy_(values)
                            owner.all_reduce(source, out=output, plan=plan)
                    torch.cuda.synchronize()
                    for output, reference in zip(outputs, expected, strict=True):
                        assert torch.equal(output, reference), (rank, elements, "eager")

                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), torch.cuda.graph(graph):
                        for values, output, plan in zip(
                            inputs, outputs, plans, strict=True
                        ):
                            source.copy_(values)
                            output.fill_(float("nan"))
                            owner.all_reduce(source, out=output, plan=plan)
                    graphs.append(graph)
                    cases.append((elements, source, inputs, outputs, expected))

                torch.cuda.synchronize()
                before = torch.cuda.memory_stats(device)
                for repetition in range(5):
                    if rank == repetition:
                        time.sleep(0.025)
                    sequence = graphs if repetition % 2 == 0 else reversed(graphs)
                    for graph in sequence:
                        graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats(device)
                for key in ("allocation.all.allocated", "segment.all.allocated"):
                    assert before[key] == after[key], key
                for elements, source, inputs, outputs, expected in cases:
                    assert torch.equal(source, inputs[-1]), (
                        rank,
                        elements,
                        "input mutation",
                    )
                    for output, reference in zip(outputs, expected, strict=True):
                        assert torch.equal(output, reference), (
                            rank,
                            elements,
                            "queued replay",
                        )
                dist.barrier()
                if rank == 0:
                    print(
                        f"TP16 blocks={blocks} threads={threads}: eight shapes, exact eager and queued replay passed",
                        flush=True,
                    )
            finally:
                for graph in graphs:
                    graph.reset()
                torch.cuda.synchronize()
                session.close()
                owner.close()
    finally:
        dist.destroy_process_group()


def test_island_rs_prepared_graphs_with_load_batch_tails():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-rs-load-test-") as directory:
        mp.spawn(
            _worker, args=(Path(directory, "gloo").as_uri(),), nprocs=16, join=True
        )
