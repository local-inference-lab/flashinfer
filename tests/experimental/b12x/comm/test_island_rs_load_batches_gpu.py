"""Exercise TP16 reduction tails and changing inputs through prepared graphs."""

from datetime import timedelta
import ctypes
import os
from pathlib import Path
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


pytestmark = [
    pytest.mark.node_exclusive,
    pytest.mark.skipif(
        os.getenv("B12X_RUN_PCIE_ISLAND_RS_TEST") != "1",
        reason="requires sixteen peer-accessible GPUs",
    ),
]


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


def _seed_generation(owner, generation):
    from b12x.comm.pcie._island_rs_cute import (
        HEADER_BYTES,
        _ACTIVE_EPOCH_OFFSET,
        MAX_BLOCKS,
        _RS_ARRIVED,
        _XS_ARRIVED,
        _AG_ARRIVED,
    )

    words = (ctypes.c_uint32 * (HEADER_BYTES // 4))()
    for block in range(MAX_BLOCKS):
        words[block] = generation
        words[_ACTIVE_EPOCH_OFFSET // 4 + block] = generation
        for peer in range(4):
            offset = (block * 4 + peer) * 128
            words[(_RS_ARRIVED + offset) // 4] = generation
            words[(_AG_ARRIVED + offset) // 4] = generation
            words[(_XS_ARRIVED + offset) // 4] = (2 * generation + 1) & 0xFFFFFFFF
    stream = torch.cuda.current_stream(owner.device)
    owner._ipc.cudaMemset(
        owner._local_ptr + owner.stage_offset,
        0xFF,
        owner.slab_bytes - owner.stage_offset,
    )
    owner._ipc.cudaMemcpyAsync(
        owner._local_ptr, ctypes.addressof(words), HEADER_BYTES, stream.cuda_stream
    )
    stream.synchronize()


def _worker(rank, rendezvous, cross_scatter=False, local_push=False):
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
    protocols = (
        ((False, False), (False, True), (True, False), (True, True))
        if local_push
        else (
            ((False, False), (False, True), (False, True))
            if cross_scatter
            else ((False, False),) * 3
        )
    )
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
                    inputs = [
                        _input(elements, rank, n, device) for n in range(len(protocols))
                    ]
                    expected = [
                        _ordered_reference(elements, n, device)
                        for n in range(len(protocols))
                    ]
                    if cross_scatter:
                        for sample in sorted({2, len(protocols) - 1}):
                            if rank == 0:
                                inputs[sample][0] = float("nan")
                                inputs[sample][1] = float("inf")
                            expected[sample][0] = float("nan")
                            expected[sample][1] = float("inf")
                            if elements > 2:
                                if rank in (0, 4):
                                    inputs[sample][2] = (
                                        float("inf") if rank == 0 else -float("inf")
                                    )
                                expected[sample][2] = float("nan")
                    outputs = [torch.empty_like(source) for _ in range(len(protocols))]
                    plans = []
                    for index, output in enumerate(outputs):
                        query = preparation.query_from_runtime(
                            owner,
                            call=dict(inp=source, out=output, blocks=blocks),
                        )
                        query = replace(
                            query,
                            call=FrozenMapping(
                                {
                                    **query.call,
                                    "threads": threads,
                                    "cross_island_rsag": protocols[index][1],
                                    "local_push": protocols[index][0],
                                }
                            ),
                        )
                        plan = preparation.plan(query, runtime=owner)
                        session.prepare(
                            (
                                plan.request(
                                    name=f"reduce-{elements}-{index}",
                                    prepare_call=lambda state,
                                    source=source,
                                    output=output: (
                                        preparation.prepared_call(
                                            state,
                                            inp=source,
                                            out=output,
                                        )
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
                    for index, (output, reference) in enumerate(
                        zip(outputs, expected, strict=True)
                    ):
                        context = f"rank={rank} elements={elements} blocks={blocks} threads={threads} protocol={protocols[index]} eager"
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                            msg=lambda detail: f"{context}\n{detail}",
                        )

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
                if cross_scatter:
                    dist.barrier()
                    _seed_generation(owner, 0xFFFFFFFE)
                    dist.barrier()
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
                    assert torch.equal(
                        source.view(torch.int16), inputs[-1].view(torch.int16)
                    ), (
                        rank,
                        elements,
                        "input mutation",
                    )
                    for index, (output, reference) in enumerate(
                        zip(outputs, expected, strict=True)
                    ):
                        context = f"rank={rank} elements={elements} blocks={blocks} threads={threads} protocol={protocols[index]} replay"
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=0,
                            atol=0,
                            equal_nan=True,
                            msg=lambda detail: f"{context}\n{detail}",
                        )
                if cross_scatter:
                    counters = (ctypes.c_uint32 * blocks)()
                    stream = torch.cuda.current_stream(device)
                    owner._ipc.cudaMemcpyAsync(
                        ctypes.addressof(counters),
                        owner._local_ptr,
                        blocks * 4,
                        stream.cuda_stream,
                    )
                    stream.synchronize()
                    expected_generation = (
                        0xFFFFFFFE + len(shapes) * len(protocols) * 5
                    ) & 0xFFFFFFFF
                    assert tuple(counters) == (expected_generation,) * blocks
                dist.barrier()
                if rank == 0:
                    print(
                        f"TP16 blocks={blocks} threads={threads}: eight shapes, exact eager and queued replay passed",
                        flush=True,
                    )
            except BaseException:
                import sys
                import traceback

                traceback.print_exc()
                sys.stderr.flush()
                raise
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


def test_island_cross_scatter_mixed_protocols_and_rollover():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-scatter-test-") as directory:
        mp.spawn(
            _worker, args=(Path(directory, "gloo").as_uri(), True), nprocs=16, join=True
        )


def test_island_local_push_mixed_protocols_and_rollover():
    if torch.cuda.device_count() != 16:
        pytest.skip("requires exactly sixteen visible GPUs")
    with tempfile.TemporaryDirectory(prefix="island-push-test-") as directory:
        mp.spawn(
            _worker,
            args=(Path(directory, "gloo").as_uri(), True, True),
            nprocs=16,
            join=True,
        )
