"""Packed payload-ready collectives across shared-channel graph transitions."""

import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest


def _worker(rank, world, directory, dtype_name, profile):
    import gc

    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=120),
    )
    try:
        _exercise_collectives(rank, world, device, dtype_name, profile)
    finally:
        # Closed plans and channels retain Gloo; release them before Python finalization.
        gc.collect()
        dist.destroy_process_group()
        gc.collect()


def _exercise_collectives(rank, world, device, dtype_name, profile):
    import torch
    import torch.distributed as dist

    from b12x.comm.pcie import _oneshot_preparation as one
    from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool
    from b12x.preparation import (
        CollectiveRequirement,
        FrozenMapping,
        PreparationSession,
    )

    dtype = getattr(torch, dtype_name)
    pack_counts = sorted({1, world - 1, world, world + 1, 2 * world + 3})
    capacity = max(pack_counts) * 16
    manager = PCIeOneshotAllReducePool.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=capacity,
        eager_buffer_bytes=capacity,
    )
    manager.prepare_channels(("payload-transition",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    target = manager._prepared_channel_for_stream(None, None)
    cases = []
    try:
        for packs in pack_counts:
            for alias in (False, True):
                source = torch.full((packs * 8,), rank + 1, device=device, dtype=dtype)
                out = source if alias else torch.empty_like(source)
                for transport in ("pull", "remote_push_full", "remote_push_rsag"):
                    query = one.query_from_runtime(
                        target,
                        surface="OneshotAllReduce.all_reduce",
                        call={"inp": source, "out": out},
                    )
                    if transport != "pull":
                        assert query.call["payload_ready_region_packs"] > 0
                    call = {
                        **query.call,
                        "transport": transport,
                        "threads": 128,
                        "blocks": 4,
                    }
                    query = replace(query, call=FrozenMapping(call))
                    plan = one.plan(query, runtime=target)
                    name = f"{transport}-{packs}-{alias}"
                    session.prepare(
                        (
                            plan.request(
                                name=name,
                                collective=CollectiveRequirement(
                                    name, tuple(range(world))
                                ),
                                prepare_call=lambda state,
                                source=source,
                                out=out: one._prepare_plain_call(
                                    state, inp=source, out=out
                                ),
                            ),
                        ),
                        coordinator=lambda progress: progress.ready_collectives[0].key
                        if progress.ready_collectives
                        else None,
                    )
                    graph = torch.cuda.CUDAGraph()
                    dist.barrier()
                    with session.capture(), manager.capture(), torch.cuda.graph(graph):
                        target.all_reduce(source, out=out, plan=plan)
                    cases.append((graph, source, out, alias))
        session.freeze()
        # Reverse shape and protocol order so consumed records cannot remain live.
        for step in range(4):
            sequence = cases if step % 2 == 0 else reversed(cases)
            for graph, source, out, alias in sequence:
                operands = (
                    torch.arange(world * source.numel()).reshape(world, -1) % 7 - 3
                )
                operands = operands.to(dtype)
                if profile == "positive":
                    operands = operands.abs() + 1
                elif step == 1:
                    operands.zero_()
                elif step == 2:
                    operands[:, 0] = float("inf")
                    operands[0, 1] = float("nan")
                expected = operands.float().sum(dim=0).to(dtype)
                for replay in range(3):
                    source.copy_(operands[rank])
                    if not alias:
                        out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    before = torch.cuda.memory_stats()
                    graph.replay()
                    torch.cuda.synchronize()
                    after = torch.cuda.memory_stats()
                    for counter in (
                        "allocation.all.allocated",
                        "allocation.all.freed",
                        "segment.all.allocated",
                        "segment.all.freed",
                    ):
                        assert before[counter] == after[counter]
                    torch.testing.assert_close(
                        out.cpu(), expected, rtol=0, atol=0, equal_nan=True
                    )
                    if not alias:
                        torch.testing.assert_close(
                            source.cpu(), operands[rank], rtol=0, atol=0, equal_nan=True
                        )
    finally:
        for graph, *_ in cases:
            graph.reset()
        session.close()
        manager.close()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_PAYLOAD_READY_TEST") != "1",
    reason="requires an assigned peer-accessible GPU group",
)
@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("dtype_name", ("bfloat16",))
@pytest.mark.parametrize("profile", ("positive", "special"))
def test_payload_ready_graph_transitions(world, dtype_name, profile, tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        pytest.skip("insufficient assigned GPUs")
    mp.spawn(
        _worker,
        args=(world, str(tmp_path), dtype_name, profile),
        nprocs=world,
        join=True,
    )
