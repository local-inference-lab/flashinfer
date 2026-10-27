"""Joint PCIe profiles change protocols inside one captured all-reduce graph."""

from datetime import timedelta
import ctypes
import gc
import os
from pathlib import Path

import pytest


def _reference(operands, rank, mode):
    import torch

    world, elements = operands.shape

    def reduce(owner, begin, end, zero_init=False):
        value = operands[owner, begin:end].float()
        if zero_init:
            value = torch.zeros_like(value) + value
        for peer in range(1, world):
            value += operands[(owner + peer) % world, begin:end].float()
        return value.to(operands.dtype)

    if mode in ("full", "pull"):
        return reduce(rank, 0, elements)
    packs = elements // 8
    bulk = mode in ("bulk_push", "bulk_pull")
    shard = -(-packs // world) if bulk else packs // world
    result = torch.empty(elements, dtype=operands.dtype)
    for owner in range(world):
        begin = min(elements, owner * shard * 8)
        end = elements if owner == world - 1 else min(elements, (owner + 1) * shard * 8)
        result[begin:end] = reduce(owner, begin, end, bulk)
    return result


def _operands(world, elements, step):
    import torch

    values = torch.randn(
        (world, elements), generator=torch.Generator().manual_seed(87131 + step)
    ).to(torch.bfloat16)
    if step % 6 == 1:
        values.zero_()
    elif step % 6 == 2:
        values[1] = -values[0]
    elif step % 6 == 3:
        values.mul_(2**-100)
    elif step % 6 == 4:
        values.mul_(128)
    elif step % 6 == 5:
        values[:, 0] = float("inf")
        values[0, 1] = float("nan")
    return values


def _poison_scratch(state, rollover):
    import torch
    import torch.distributed as dist

    from b12x.comm.pcie._joint_cute import _EPOCH_OFFSET

    work = state.workspace
    ipc = state.runtime._ipc
    local = work.shared.local_ptr
    bulk_offset = (state.query.world_size + 1) * work.capacity_packs * 16
    for slot in range(2):
        base = local + work.signal_bytes + slot * work.slot_bytes
        ipc.cudaMemset(base, 0xFF, work.capacity_packs * 16)
        ipc.cudaMemset(base + bulk_offset, 0xFF, work.slot_bytes - bulk_offset)
    if rollover:
        epoch = (ctypes.c_uint32 * 2)(0xFFFFFFFD, 0)
        stream = torch.cuda.current_stream()
        ipc.cudaMemcpyAsync(
            local + _EPOCH_OFFSET, ctypes.addressof(epoch), 8, stream.cuda_stream
        )
        stream.synchronize()
    dist.barrier()


def _assert_epoch_wrapped(state):
    import torch

    from b12x.comm.pcie._joint_cute import _EPOCH_OFFSET

    words = (ctypes.c_uint32 * 2)()
    stream = torch.cuda.current_stream()
    state.runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(words),
        state.workspace.shared.local_ptr + _EPOCH_OFFSET,
        8,
        stream.cuda_stream,
    )
    stream.synchronize()
    assert tuple(words) == (0, 0), tuple(words)


def _exercise(rank, world, family, alias):
    import torch
    import torch.distributed as dist

    from b12x.comm.pcie import CollectiveConfig
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import CollectiveRequirement, PreparationSession

    device = torch.device("cuda", rank)
    pack_counts = tuple(sorted({1, world - 1, world + 1, 257, 4099}))
    modes = (
        ("full", "payload_rsag", "pull")
        if family == "joint_latency"
        else ("payload_rsag", "bulk_push", "bulk_pull", "pull")
    )
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=max(pack_counts) * 16,
        eager_buffer_bytes=max(pack_counts) * 16,
    )
    manager.prepare_channels(("joint-profile-transitions",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    graphs = []
    try:
        for packs in pack_counts:
            source = torch.full(
                (packs * 8,), rank + 1, dtype=torch.bfloat16, device=device
            )
            out = source if alias else torch.empty_like(source)
            plan = manager.plan(
                source,
                out=out,
                override=CollectiveConfig(
                    implementation=family,
                    threads=256,
                    blocks=64,
                    submode=modes[0],
                    active_blocks=16,
                ),
            )
            name = f"{family}-{packs}-{alias}"
            session.prepare(
                (
                    plan.request(
                        name=name,
                        collective=CollectiveRequirement(name, tuple(range(world))),
                        prepare_call=lambda state,
                        source=source,
                        out=out: _prepare_plain_call(state, inp=source, out=out),
                    ),
                ),
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                manager.all_reduce(source, out=out, plan=plan)
            graphs.append((graph, source, out, plan.prepared.state))
        session.freeze()
        for graph, source, out, state in graphs:
            profiles = state.profiles
            assert isinstance(profiles, tuple)
            selector_address = state.selector.data_ptr()
            for step, active in enumerate((1, 4, 2, 8, 1, 64)):
                sequence = modes if step % 2 == 0 else tuple(reversed(modes))
                for mode in sequence:
                    state.select_profile(mode, active)
                    _poison_scratch(state, rollover=step == 2)
                    values = _operands(world, source.numel(), step)
                    expected = _reference(values, rank, mode)
                    for replay in range(3):
                        source.copy_(values[rank])
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
                            assert before[counter] == after[counter], (
                                mode,
                                active,
                                replay,
                                counter,
                            )
                        torch.testing.assert_close(
                            out.cpu(), expected, rtol=0, atol=0, equal_nan=True
                        )
                        if not alias:
                            torch.testing.assert_close(
                                source.cpu(),
                                values[rank],
                                rtol=0,
                                atol=0,
                                equal_nan=True,
                            )
                    if step == 2:
                        _assert_epoch_wrapped(state)
                    assert selector_address == state.selector.data_ptr()
                    assert state.profiles == profiles
    finally:
        for graph, *_ in graphs:
            graph.reset()
        session.close()
        manager.close()


def _worker(rank, world, directory, family, alias):
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=180),
    )
    try:
        _exercise(rank, world, family, alias)
    finally:
        gc.collect()
        dist.destroy_process_group()
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_PCIE_TEST") != "1",
    reason="requires an assigned peer-accessible GPU group",
)
@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("family", ("joint_latency", "joint_bandwidth"))
@pytest.mark.parametrize("alias", (False, True))
def test_joint_profile_graph_transitions(world, family, alias, tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world:
        pytest.skip("insufficient assigned GPUs")
    mp.spawn(
        _worker, args=(world, str(tmp_path), family, alias), nprocs=world, join=True
    )
