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
    bulk = mode in ("bulk_push", "bulk_pull", "serial_push")
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
        values[::2].neg_()
    elif step % 6 == 2:
        values[1] = -values[0]
    elif step % 6 == 3:
        values.mul_(2**-100)
    elif step % 6 == 4:
        values.mul_(128)
    elif step % 6 == 5:
        values[:, 0] = float("inf")
        values[0, 1] = float("nan")
    # These exactly representable BF16 inputs distinguish sequential FP32
    # addition from reassociation, including each shard owner's cyclic order.
    for rank in range(world):
        values[rank, 2::8] = (2**24, 1, -(2**24), 1)[rank] if rank < 4 else 0
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


def _barrier_counters(state):
    import torch

    from b12x.comm.pcie._joint_profile import JOINT_MAX_BLOCKS, JOINT_MAX_RANKS

    counters = (ctypes.c_uint32 * (JOINT_MAX_BLOCKS * JOINT_MAX_RANKS))()
    stream = torch.cuda.current_stream()
    state.runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(counters),
        state.workspace.shared.local_ptr,
        ctypes.sizeof(counters),
        stream.cuda_stream,
    )
    stream.synchronize()
    return tuple(counters)


def _assert_serial_push_progress(state, before, active, replays):
    from b12x.comm.pcie._joint_profile import JOINT_MAX_RANKS

    world, rank = state.query.world_size, state.query.rank
    for index, (start, end) in enumerate(
        zip(before, _barrier_counters(state), strict=True)
    ):
        block, peer = divmod(index, JOINT_MAX_RANKS)
        expected = (
            2 * replays if block < active and peer < world and peer != rank else 0
        )
        assert (end - start) & 0xFFFFFFFF == expected, (
            block,
            peer,
            start,
            end,
            expected,
        )


def _exercise(rank, world, family, alias, restricted=False):
    import torch
    import torch.distributed as dist

    from b12x.comm.pcie import CollectiveConfig
    from b12x.comm.pcie._joint_profile import joint_modes
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.preparation import CollectiveRequirement, PreparationSession

    device = torch.device("cuda", rank)
    pack_counts = tuple(
        sorted({1, world - 1, world + 1, 257, 4099, 8 * 640, 128 * 640, 4096 * 640})
    )
    modes = ("serial_push",) if restricted else joint_modes(family, world)
    manager = PCIeAllReduce.from_exchange_group(
        joint=True,
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
                joint_only=True,
                out=out,
                override=CollectiveConfig(
                    implementation=family,
                    threads=256,
                    blocks=64,
                    submode=modes[0],
                    active_blocks=1 if restricted else 16,
                ),
            )
            name = f"{family}-{packs}-{alias}"
            session.prepare(
                (
                    plan.request(
                        name=name,
                        collective=CollectiveRequirement(name, tuple(range(world))),
                        prepare_call=lambda state, source=source, out=out: (
                            _prepare_plain_call(state, inp=source, out=out)
                        ),
                    ),
                ),
                coordinator=lambda progress: (
                    progress.ready_collectives[0].key
                    if progress.ready_collectives
                    else None
                ),
            )
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            with (
                session.capture(),
                manager.capture(),
                torch.cuda.graph(
                    graph, stream=torch.cuda.current_stream() if restricted else None
                ),
            ):
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
                    selected_active = 1 if mode == "serial_push" else active
                    state.select_profile(mode, selected_active)
                    _poison_scratch(state, rollover=step == 2)
                    values = _operands(world, source.numel(), step)
                    expected = _reference(values, rank, mode)
                    counters = (
                        _barrier_counters(state) if mode == "serial_push" else None
                    )
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
                        if mode == "serial_push":
                            non_nan = ~torch.isnan(expected)
                            assert torch.equal(
                                out.cpu().view(torch.int16)[non_nan],
                                expected.view(torch.int16)[non_nan],
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
                    if mode == "serial_push":
                        _assert_serial_push_progress(
                            state, counters, selected_active, 3
                        )
                    assert selector_address == state.selector.data_ptr()
                    assert state.profiles == profiles
    finally:
        for graph, *_ in graphs:
            graph.reset()
        session.close()
        manager.close()


def _worker(rank, world, directory, family, alias, restricted=False):
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
        if family == "provisioning":
            _exercise_provisioning(rank)
        elif restricted:
            from cuda.bindings import driver
            from torch.cuda.green_contexts import GreenContext

            context = GreenContext.create(num_sms=8, device_id=rank)
            stream = context.Stream()
            result, resource = driver.cuStreamGetDevResource(
                stream.cuda_stream, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
            )
            assert result == driver.CUresult.CUDA_SUCCESS, result
            properties = torch.cuda.get_device_properties(rank)
            assert (
                resource.sm.smCount
                * (properties.max_threads_per_multi_processor // 256)
                < 64
            )
            with torch.cuda.stream(stream):
                _exercise(rank, world, family, alias, restricted=True)
            stream.synchronize()
        else:
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


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_PCIE_TEST") != "1",
    reason="requires an assigned peer-accessible GPU group",
)
@pytest.mark.parametrize("family", ("joint_latency", "joint_bandwidth"))
def test_serial_push_graph_replay_with_restricted_sms(family, tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 4:
        pytest.skip("insufficient assigned GPUs")
    mp.spawn(_worker, args=(4, str(tmp_path), family, True, True), nprocs=4, join=True)


def _exercise_provisioning(rank):
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
    from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
    from b12x.preparation import CollectiveRequirement, PreparationSession

    device = torch.device("cuda", rank)
    with pytest.raises(RuntimeError, match="contract differs across ranks"):
        PCIeAllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            single_channel=True,
            max_size=5120,
            eager_buffer_bytes=5120,
            joint=bool(rank),
        )
    for joint in (False, True):
        manager = PCIeAllReduce.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            single_channel=True,
            max_size=5120,
            eager_buffer_bytes=5120,
            joint=joint,
        )
        session = PreparationSession(device=device, autotune=False, compile_workers=1)
        graph = torch.cuda.CUDAGraph()
        try:
            channel = manager._runtime._prepared_channel_for_stream(None, None)
            workspace = getattr(channel, "_joint_workspace", None)
            assert (workspace is not None) == joint
            assert len(channel._owned_buffers) == (2 if joint else 1)
            source = torch.full((2560,), rank + 1, device=device, dtype=torch.bfloat16)
            out = torch.empty_like(source)
            if not joint:
                with pytest.raises(ValueError, match="provisioned BF16 joint channel"):
                    manager.plan(source, out=out, joint_only=True)
            plan = manager.plan(source, out=out, joint_only=joint)
            session.prepare(
                (
                    plan.request(
                        name="provisioning",
                        collective=CollectiveRequirement("provisioning", (0, 1)),
                        prepare_call=lambda state: _prepare_plain_call(
                            state, inp=source, out=out
                        ),
                    ),
                ),
                coordinator=lambda progress: (
                    progress.ready_collectives[0].key
                    if progress.ready_collectives
                    else None
                ),
            )
            state = plan.prepared.state
            state.run_plain(source, out)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                state.run_plain(source, out)
            for value in (1, 3, 2):
                source.fill_(value * (rank + 1))
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, torch.full_like(out, value * 3), rtol=0, atol=0
                )
        finally:
            graph.reset()
            session.close()
            manager.close()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_JOINT_PCIE_TEST") != "1",
    reason="requires an assigned peer-accessible GPU group",
)
def test_static_and_joint_workspace_provisioning(tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("insufficient assigned GPUs")
    mp.spawn(
        _worker, args=(2, str(tmp_path), "provisioning", False), nprocs=2, join=True
    )
