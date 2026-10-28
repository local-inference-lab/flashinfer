"""Prepared BF16 collectives and all-reduce dispatch on four CUDA devices.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
        --nproc-per-node=4 tests/experimental/b12x/comm/test_pcie_twoshot_bf16.py

Add --benchmark to print all-rank graph times, or --benchmark-only to time the
same dispatch path in another checkout without running its correctness checks.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.comm.pcie import PCIeTwoShotBF16
from b12x.comm.pcie import _twoshot_preparation as preparation
from b12x.comm.pcie._twoshot_bf16_cute import get_twoshot_bf16_launcher
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.comm.pcie.pcie_twoshot_bf16 import _make_layout, _pad_scalar_peer_ptrs
from b12x.preparation import CollectiveRequirement, PreparationSession


@pytest.mark.parametrize("shape", ((128, 1024), (32, 4096), (16, 8192), (131072,)))
def test_prepared_all_reduce_counts_packs(monkeypatch, shape):
    payload = torch.empty(shape, dtype=torch.bfloat16)
    out = torch.empty_like(payload)
    launches = []
    layout = _make_layout(32, 4096, 4)
    runtime = SimpleNamespace(
        world_size=4,
        rank=0,
        row_elems=4096,
        device=torch.device("cpu"),
        _pack_stride=layout.pack_stride,
        _slot_bytes=layout.slot_bytes,
        _push_shard=torch.empty(8 * 4096, dtype=torch.bfloat16),
        _capture_context_depth=0,
        _device_slot_selection=False,
        _device_slot_bias=0,
        _slot=0,
        _staging_ptrs=((1, 2, 3, 4), (5, 6, 7, 8)),
        _signal_ptrs=(9, 10, 11, 12),
    )
    state = SimpleNamespace(
        query=SimpleNamespace(
            call={
                "operation": "all_reduce",
                "transport": "push",
                "threads": 512,
                "block_limit": 64,
            }
        ),
        launcher=lambda *key: lambda *args: launches.append((key, args)),
    )
    import b12x.comm.pcie.pcie_twoshot_bf16 as runtime_module

    monkeypatch.setattr(runtime_module, "_is_current_stream_capturing", lambda _: False)
    PCIeTwoShotBF16._launch_prepared(
        runtime, payload, None, out, state=state, threads=512, block_limit=64
    )
    assert len(launches) == 2
    assert [entry[0][-1] for entry in launches] == ["reduce_scatter", "all_gather"]
    assert launches[0][1][0] == payload.data_ptr()
    assert launches[0][1][3] == runtime._push_shard.data_ptr()
    assert launches[1][1][0] == runtime._push_shard.data_ptr()
    assert launches[1][1][3] == out.data_ptr()
    assert launches[0][1][1] == runtime._staging_ptrs[0]
    assert launches[1][1][1] == runtime._staging_ptrs[1]
    assert all(entry[1][-3:-1] == (4096, 16384) for entry in launches)


def test_layout_has_payload_and_reduced_shard_slots():
    layout = _make_layout(131072, 8, 4)
    assert layout.reduced_offset == 2 << 20
    assert layout.slot_bytes >= layout.reduced_offset + layout.pack_stride * 16
    assert layout.slab_bytes == layout.signal_bytes + 2 * layout.slot_bytes
    assert _make_layout(128, 4096, 4).slab_bytes > _make_layout(64, 4096, 4).slab_bytes


@pytest.mark.parametrize("world", (1, 5, 7, 9, 12, 16))
def test_private_launchers_reject_unsupported_world_sizes(world):
    with pytest.raises(ValueError, match="support world sizes"):
        get_twoshot_bf16_launcher("reduce_scatter", world, 0, False, 0, 512, 8, 0)


@pytest.mark.parametrize("transport", ("push", "pull"))
def test_preparation_accounts_for_persistent_shard(transport):
    layout = _make_layout(131072, 8, 4)
    runtime = SimpleNamespace(
        rank=0,
        world_size=4,
        row_elems=8,
        max_rows=131072,
        device=torch.device("cpu"),
        _pack_stride=layout.pack_stride,
        _slot_bytes=layout.slot_bytes,
        _signal_ptrs=(100,) * 8,
        _staging_ptrs=(
            (100 + layout.signal_bytes,) * 8,
            (100 + layout.signal_bytes + layout.slot_bytes,) * 8,
        ),
    )
    inp = torch.empty(32768, dtype=torch.bfloat16)
    query = preparation.query_from_runtime(
        runtime,
        surface="PCIeTwoShotBF16.all_reduce",
        call={"inp": inp, "transport": transport},
    )
    metadata = preparation.query_from_metadata(
        runtime,
        surface="PCIeTwoShotBF16.all_reduce",
        shape=tuple(inp.shape),
        dtype=inp.dtype,
        transport=transport,
    )
    assert query.setup == metadata.setup
    assert query.setup["push_shard_nbytes"] == (2 << 20) // 4
    plan = preparation.plan(query, runtime=runtime)
    from b12x.preparation.types import _plan_scope

    with _plan_scope(plan):
        memory = plan._memory_requirements(None, None)
    expected = [1 << 19, layout.slab_bytes]
    assert sorted(entry.required_nbytes for entry in memory.persistent) == expected
    assert all(
        entry.required_nbytes == entry.resident_nbytes for entry in memory.persistent
    )


@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("packs", (1, 3, 17, 320, 16385))
def test_packed_allreduce_tail_storage_and_launch(monkeypatch, world, packs):
    import b12x.comm.pcie.pcie_twoshot_bf16 as runtime_module

    layout = _make_layout(packs, 8, world)
    shard_packs = (packs + world - 1) // world
    assert layout.pack_stride >= shard_packs
    assert layout.pack_stride % 16 == 0
    assert layout.slot_bytes >= world * layout.pack_stride * 16
    pointers = tuple(range(1, world + 1))
    padded = _pad_scalar_peer_ptrs(pointers, rank=world - 1, world_size=world)
    assert len(padded) == 16
    assert padded[:world] == pointers
    assert padded[world:] == (world,) * (16 - world)
    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.__dict__.update(
        rank=world - 1,
        world_size=world,
        device=torch.device("cpu"),
        max_rows=packs,
        row_elems=8,
        _closed=False,
        _pack_stride=layout.pack_stride,
        _slot_bytes=layout.slot_bytes,
        _push_shard=torch.empty(layout.pack_stride * 8, dtype=torch.bfloat16),
        _signal_ptrs=padded,
        _staging_ptrs=(padded, padded),
        _capture_context_depth=0,
        _device_slot_selection=False,
        _device_slot_bias=0,
        _slot=0,
    )
    payload = torch.empty(packs * 8, dtype=torch.bfloat16)
    out = torch.empty_like(payload)
    assert runtime.accepts(payload)
    assert not runtime.accepts(torch.empty(packs * 8 + 8, dtype=torch.bfloat16))
    launches = []
    state = SimpleNamespace(
        query=SimpleNamespace(
            call={
                "operation": "all_reduce",
                "transport": "push",
                "threads": 512,
                "block_limit": 64,
            }
        ),
        launcher=lambda *key: lambda *args: launches.append((key, args)),
    )
    monkeypatch.setattr(runtime_module, "_is_current_stream_capturing", lambda _: False)
    runtime._launch_prepared(
        payload, None, out, state=state, threads=512, block_limit=64
    )
    assert len(launches) == 2
    assert all(args[-3:-1] == (shard_packs, packs) for _, args in launches)
    assert launches[0][1][3] == launches[1][1][0] == runtime._push_shard.data_ptr()
    query = preparation.query_from_runtime(
        runtime, surface="PCIeTwoShotBF16.all_reduce", call={"inp": payload, "out": out}
    )
    metadata = preparation.query_from_metadata(
        runtime,
        surface="PCIeTwoShotBF16.all_reduce",
        shape=tuple(payload.shape),
        dtype=payload.dtype,
    )
    assert query.call == metadata.call
    assert query.setup == metadata.setup
    assert query.setup["push_shard_nbytes"] == runtime._push_shard.numel() * 2


def test_allreduce_geometry_is_independent_of_native_row_width():
    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.__dict__.update(
        rank=0,
        world_size=3,
        device=torch.device("cpu"),
        max_rows=2,
        row_elems=4096,
        _closed=False,
        _pack_stride=352,
        _slot_bytes=2 << 20,
        _signal_ptrs=(100,) * 16,
        _staging_ptrs=((100 + (2 << 20),) * 16,) * 2,
    )
    payload = torch.empty(24, dtype=torch.bfloat16)
    assert runtime.accepts(payload)
    query = preparation.query_from_runtime(
        runtime, surface="PCIeTwoShotBF16.all_reduce", call={"inp": payload}
    )
    metadata = preparation.query_from_metadata(
        runtime, surface="PCIeTwoShotBF16.all_reduce", shape=(24,), dtype=payload.dtype
    )
    assert query.call["rows"] == metadata.call["rows"] == 1
    with pytest.raises(ValueError, match="complete transport"):
        preparation.query_from_runtime(
            runtime, surface="PCIeTwoShotBF16.reduce_scatter", call={"payload": payload}
        )


@pytest.mark.parametrize(
    "output_kind", ("same_object", "same_span", "disjoint", "partial")
)
def test_allreduce_output_alias_contract(monkeypatch, output_kind):
    import b12x.comm.pcie.pcie_twoshot_bf16 as runtime_module

    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.__dict__.update(
        device=torch.device("cpu"), max_rows=8, row_elems=8, _closed=False
    )
    storage = torch.empty(72, dtype=torch.bfloat16)
    inp = storage[:64]
    outputs = {
        "same_object": inp,
        "same_span": inp.view_as(inp),
        "disjoint": torch.empty_like(inp),
        "partial": storage[8:],
    }
    out = outputs[output_kind]
    state = SimpleNamespace(
        require_runtime=lambda _: None,
        query=SimpleNamespace(surface="PCIeTwoShotBF16.all_reduce"),
    )
    launches = []
    monkeypatch.setattr(runtime_module, "require_prepared", lambda *args: state)
    monkeypatch.setattr(runtime_module, "_device_guard", lambda _: nullcontext())
    runtime._launch_prepared = lambda *args, **kwargs: launches.append(args)
    if output_kind == "partial":
        with pytest.raises(ValueError, match="must not overlap input"):
            runtime.all_reduce(inp, out=out, plan=object())
        assert not launches
    else:
        assert runtime.all_reduce(inp, out=out, plan=object()) is out
        assert launches[0][0] is inp and launches[0][2] is out


@pytest.mark.parametrize("operation", ("all_reduce", "reduce_scatter", "all_gather"))
@pytest.mark.parametrize("overlap", ("disjoint", "same_start", "partial"))
def test_prepared_run_enforces_output_overlap(operation, overlap):
    storage = torch.empty(64, dtype=torch.bfloat16)
    input_rows = 2 if operation == "all_gather" else 4
    output_rows = 2 if operation == "reduce_scatter" else 4
    inp = storage[: input_rows * 8].view(input_rows, 8)
    if overlap == "disjoint":
        out = torch.empty((output_rows, 8), dtype=torch.bfloat16)
    else:
        start = 8 if overlap == "partial" else 0
        out = storage[start : start + output_rows * 8].view(output_rows, 8)
    launches = []
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
        row_elems=8,
        max_rows=8,
        _pack_stride=16,
        _slot_bytes=2 << 20,
        _signal_ptrs=(100,) * 16,
        _staging_ptrs=((100 + (2 << 20),) * 16,) * 2,
        _launch_prepared=lambda *args, **kwargs: launches.append(args),
    )
    query = preparation.query_from_runtime(
        runtime,
        surface=f"PCIeTwoShotBF16.{operation}",
        call={"inp" if operation == "all_reduce" else "payload": inp, "out": out},
    )
    state = preparation._TwoShotExecutionState(query, runtime, {})
    prepared = preparation.prepared_call(state, payload=inp, out=out)
    allowed = overlap == "disjoint" or (
        operation == "all_reduce" and overlap == "same_start"
    )
    if allowed:
        prepared.run()
        assert len(launches) == 1
    else:
        with pytest.raises(ValueError, match="must not overlap"):
            prepared.run()
        assert not launches


def _prepare(session, runtime, operation, payload, out, *, transport="push"):
    call = {"inp" if operation == "all_reduce" else "payload": payload, "out": out}
    if isinstance(runtime, PCIeAllReduce):
        plan = runtime.plan(payload, out=out)
    else:
        call["transport"] = transport
        query = preparation.query_from_runtime(
            runtime, surface=f"PCIeTwoShotBF16.{operation}", call=call
        )
        plan = preparation.plan(query, runtime=runtime)
    name = f"{operation}-{payload.numel()}"
    prime_out = torch.empty_like(payload) if out is None else out
    if plan.query.surface.startswith("PCIeTwoShotBF16."):
        prime = lambda state: preparation.prepared_call(
            state, payload=payload, out=prime_out
        )
    else:
        from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call

        prime = lambda state: _prepare_plain_call(state, inp=payload, out=prime_out)
    session.prepare(
        (
            plan.request(
                name=name,
                collective=CollectiveRequirement(
                    key=name, ranks=tuple(range(dist.get_world_size()))
                ),
                prepare_call=prime,
            ),
        ),
        coordinator=lambda progress: (
            progress.ready_collectives[0].key if progress.ready_collectives else None
        ),
    )
    return plan


def _fill(tensor, rank, step):
    generator = torch.Generator(device=tensor.device).manual_seed(10007 * step + rank)
    tensor.copy_(
        torch.randn(tensor.shape, generator=generator, device=tensor.device) * 3
    )


def _reference(inp, *, owner_reduction=True):
    world = dist.get_world_size()
    values = [torch.empty_like(inp) for _ in range(world)]
    dist.all_gather(values, inp)
    if not owner_reduction:
        rank = dist.get_rank()
        acc = values[rank].float()
        for offset in range(1, world):
            acc.add_(values[(rank + offset) % world].float())
        return acc.to(inp.dtype)
    shard = ((inp.numel() // 8 + world - 1) // world) * 8
    result = torch.empty_like(inp).view(-1)
    for owner in range(world):
        interval = slice(owner * shard, min((owner + 1) * shard, inp.numel()))
        acc = torch.zeros_like(values[owner].view(-1)[interval], dtype=torch.float32)
        for offset in range(world):
            acc.add_(values[(owner + offset) % world].view(-1)[interval].float())
        result[interval] = acc.to(torch.bfloat16)
    return result.view_as(inp)


def _fill_exact(tensor, rank, step):
    values = torch.arange(tensor.numel(), device=tensor.device)
    tensor.copy_(values.add_(step).remainder_(7).add_(rank + 1).view_as(tensor))


def _check_packed_tails(
    session, pool, rank, *, exact_inputs=False, equal_shards=False, transport="push"
):
    cases = []
    pack_counts = (
        (4, 12, 20, 320, 512, 516, 16384, 32768, 65536)
        if equal_shards
        else (1, 3, 17, 320, 512, 513, 16385, 32768, 65537)
    )
    for packs in pack_counts:
        inp = torch.empty(packs * 8, dtype=torch.bfloat16, device=pool.device)
        storage = torch.full(
            (inp.numel() + 16,), 123, dtype=inp.dtype, device=inp.device
        )
        out = storage[8:-8]
        (_fill_exact if exact_inputs else _fill)(inp, rank, packs)
        plan = _prepare(session, pool, "all_reduce", inp, out, transport=transport)
        with session.capture():
            pool.all_reduce(inp, plan=plan, out=out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, _reference(inp), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), pool.capture(), torch.cuda.graph(graph):
            pool.all_reduce(inp, plan=plan, out=out)
        cases.append((inp, out, storage, plan, graph))
    epoch = _epoch(pool)
    shard_address = pool._push_shard.data_ptr()
    for step in range(8):
        for inp, out, storage, plan, graph in cases[:: 1 if step % 2 else -1]:
            (_fill_exact if exact_inputs else _fill)(inp, rank, 100 + step)
            if not exact_inputs and step == 1:
                inp.zero_()
            elif not exact_inputs and step == 2:
                inp.mul_(2**-100)
            elif not exact_inputs and step == 3:
                inp.fill_(-1 if rank % 2 else 1)
            expected = _reference(inp)
            out.fill_(float("nan"))
            torch.cuda.synchronize()
            before = _allocation_state(pool.device)
            with session.capture(), kernel_resolution_guard("packed BF16 replay"):
                if step % 2:
                    pool.all_reduce(inp, plan=plan, out=out)
                else:
                    graph.replay()
            torch.cuda.synchronize()
            assert _allocation_state(pool.device) == before
            epoch += 1 if transport == "pull" else 2
            assert _epoch(pool) == epoch
            assert pool._push_shard.data_ptr() == shard_address
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
            assert torch.all(storage[:8] == 123) and torch.all(storage[-8:] == 123)
    for *_, graph in cases:
        graph.reset()


def _check_exact_alias(session, pool, rank, *, transport="push"):
    cases = []
    for number, packs in enumerate((1, 3, 17, 320, 512, 513, 16385, 32768, 65537)):
        storage = torch.full(
            (packs * 8 + 24,), 123, device=pool.device, dtype=torch.bfloat16
        )
        inp = storage[8:-16]
        alias = inp if number % 2 else inp.view_as(inp)
        other_inp, other_out = torch.empty_like(inp), torch.empty_like(inp)
        _fill_exact(inp, rank, packs)
        other_inp.copy_(inp)
        alias_plan = _prepare(
            session, pool, "all_reduce", inp, alias, transport=transport
        )
        other_plan = _prepare(
            session, pool, "all_reduce", other_inp, other_out, transport=transport
        )
        # Priming the aliased call consumes its input.
        _fill_exact(inp, rank, packs)
        other_inp.copy_(inp)
        expected = _reference(inp)
        with session.capture():
            pool.all_reduce(inp, plan=alias_plan, out=alias)
            pool.all_reduce(other_inp, plan=other_plan, out=other_out)
        torch.cuda.synchronize()
        torch.testing.assert_close(alias, expected, rtol=0, atol=0)
        torch.testing.assert_close(alias, other_out, rtol=0, atol=0)
        if packs > 1:
            with pytest.raises(ValueError, match="must not overlap input"):
                pool.all_reduce(inp, plan=alias_plan, out=storage[16:-8])
        graph = torch.cuda.CUDAGraph()
        with session.capture(), pool.capture(), torch.cuda.graph(graph):
            pool.all_reduce(inp, plan=alias_plan, out=alias)
            pool.all_reduce(other_inp, plan=other_plan, out=other_out)
        cases.append(
            (storage, inp, alias, other_inp, other_out, alias_plan, other_plan, graph)
        )
    epoch = _epoch(pool)
    shard_address = pool._push_shard.data_ptr()
    for step in range(8):
        for (
            storage,
            inp,
            alias,
            other_inp,
            other_out,
            alias_plan,
            other_plan,
            graph,
        ) in cases[:: 1 if step % 2 else -1]:
            _fill_exact(inp, rank, 100 + step)
            other_inp.copy_(inp)
            other_out.fill_(float("nan"))
            expected = _reference(inp)
            torch.cuda.synchronize()
            before = _allocation_state(pool.device)
            if rank == step % pool.world_size:
                time.sleep(0.002)
            with session.capture(), kernel_resolution_guard("exact-alias BF16 replay"):
                if step % 2:
                    pool.all_reduce(inp, plan=alias_plan, out=alias)
                    pool.all_reduce(other_inp, plan=other_plan, out=other_out)
                else:
                    graph.replay()
            torch.cuda.synchronize()
            assert _allocation_state(pool.device) == before
            epoch += 2 if transport == "pull" else 4
            assert _epoch(pool) == epoch
            assert pool._push_shard.data_ptr() == shard_address
            torch.testing.assert_close(alias, expected, rtol=0, atol=0)
            torch.testing.assert_close(alias, other_out, rtol=0, atol=0)
            assert torch.all(storage[:8] == 123) and torch.all(storage[-16:] == 123)
    for *_, graph in cases:
        graph.reset()


def _check_payload_ready(session, manager, rank):
    for size in (
        16,
        32,
        48,
        64,
        24560,
        24576,
        24592,
        65520,
        65552,
        655344,
        655360,
        655376,
        2097168,
    ):
        for inplace in (False, True):
            inp = torch.empty(size // 2, dtype=torch.bfloat16, device=manager.device)
            out = inp if inplace else torch.empty_like(inp)
            plan = _prepare(session, manager, "all_reduce", inp, out)
            transport = plan.query.call.get("transport", "twoshot")
            owner_reduction = transport in ("twoshot", "remote_push_rsag")
            graph = torch.cuda.CUDAGraph()
            _fill(inp, rank, 123)
            with session.capture(), manager.capture(), torch.cuda.graph(graph):
                manager.all_reduce(inp, plan=plan, out=out)
            for step in range(6):
                _fill(inp, rank, 91 + step)
                if step == 1:
                    inp.zero_()
                elif step == 2:
                    inp.mul_(2**-100)
                elif step == 3:
                    inp.fill_(1 if rank % 2 else -1)
                expected = _reference(inp, owner_reduction=owner_reduction)
                if not inplace:
                    out.fill_(float("nan"))
                torch.cuda.synchronize()
                before = _allocation_state(manager.device)
                with session.capture(), kernel_resolution_guard("payload-ready replay"):
                    graph.replay()
                torch.cuda.synchronize()
                assert _allocation_state(manager.device) == before
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
            graph.reset()


def _check_payload_protocol_transitions(session, manager, rank):
    from b12x.comm.pcie import _oneshot_preparation as oneshot
    from b12x.comm.pcie._oneshot_cute import _PLAIN_GRAPH_EPOCH_OFFSET

    channel = manager._runtime._prepared_channel_for_stream(None, None)
    cases = []
    for rows, fused in ((1, False), (8, False), (3, True)):
        shape = (rows, 5120)
        inp = torch.ones(
            (*shape[:1], 3, shape[1]) if fused else shape,
            device=manager.device,
            dtype=torch.bfloat16,
        )
        out = torch.empty(shape, device=manager.device, dtype=torch.bfloat16)
        if fused:
            residual = torch.empty(shape, device=manager.device, dtype=torch.float32)
            addend = torch.zeros(shape, device=manager.device, dtype=torch.bfloat16)
            weights = torch.ones((rows, 3), device=manager.device)
            ids = torch.zeros((rows, 3), device=manager.device, dtype=torch.int32)
            options = dict(
                input_addend=addend,
                group_weights=weights,
                group_ids=ids,
                group_id_limit=8,
            )
            plan = manager.plan(
                inp, operation="all_reduce_fused_add", residual=residual, **options
            )
            assert plan.query.call["mode"] == "stage_bf16_sentinel"
            session.prepare(
                (
                    plan.request(
                        name="grouped-residual",
                        collective=CollectiveRequirement(
                            key="grouped-residual", ranks=tuple(range(4))
                        ),
                        prepare_call=lambda state: oneshot._prepare_fused_call(
                            state,
                            inp=inp,
                            residual=residual,
                            weight=None,
                            out=None,
                            residual_out=residual,
                            epsilon=0.0,
                            **options,
                        ),
                    ),
                ),
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
            run = lambda: manager.all_reduce_fused_add(
                inp, residual, plan=plan, **options
            )
        else:
            plan = _prepare(session, manager, "all_reduce", inp, out)
            run = lambda: manager.all_reduce(inp, plan=plan, out=out)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), manager.capture(), torch.cuda.graph(graph):
            run()
        cases.append((graph, inp, out, plan, fused))

    # The payload protocol drains consumed records and can wrap without retaining tags.
    epoch = ctypes.c_uint32(0xFFFFFFFC)
    stream = torch.cuda.current_stream(manager.device)
    channel._ipc.cudaMemcpyAsync(
        channel.signal_ptrs[rank] + _PLAIN_GRAPH_EPOCH_OFFSET,
        ctypes.addressof(epoch),
        ctypes.sizeof(epoch),
        int(stream.cuda_stream),
    )
    stream.synchronize()
    dist.barrier()
    for step in range(8):
        for graph, inp, out, plan, fused in cases[:: 1 if step % 2 else -1]:
            _fill(inp, rank, 500 + step)
            if fused:
                ids.fill_(-1 if step == 1 else 0)
                weights.fill_(0.25)
                addend.fill_(-0.5)
                residual.fill_(0.125)
                local = torch.zeros_like(residual)
                for group in range(3):
                    local += (
                        torch.where(
                            ids[:, group, None] >= 0, inp[:, group].float(), 0.0
                        )
                        * weights[:, group, None]
                    )
                local = (local.bfloat16().float() + addend.float()).bfloat16()
                values = [torch.empty_like(local) for _ in range(4)]
                dist.all_gather(values, local)
                expected = values[rank].float()
                for offset in range(1, 4):
                    expected += values[(rank + offset) % 4].float()
                expected += residual
                actual = residual
            else:
                expected = _reference(
                    inp,
                    owner_reduction=plan.query.call["transport"] == "remote_push_rsag",
                )
                actual = out
            torch.cuda.synchronize()
            before = _allocation_state(manager.device)
            with session.capture(), kernel_resolution_guard("mixed payload replay"):
                graph.replay()
            torch.cuda.synchronize()
            assert _allocation_state(manager.device) == before
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for graph, *_ in cases:
        graph.reset()


def _allocation_state(device):
    return (
        torch.cuda.memory_allocated(device),
        torch.cuda.memory_stats(device).get("allocation.all.allocated"),
    )


def _epoch(runtime):
    from b12x.comm.pcie._twoshot_cute import _GRAPH_EPOCH_OFFSET

    value = ctypes.c_uint32()
    runtime._ipc.cudaMemcpyAsync(
        ctypes.addressof(value),
        runtime._signal_ptrs[runtime.rank] + _GRAPH_EPOCH_OFFSET,
        4,
        int(torch.cuda.current_stream().cuda_stream),
    )
    torch.cuda.synchronize()
    return value.value


def _check_dispatch(session, manager, rank, *, benchmark=False, benchmark_only=False):
    sizes = (
        (24 << 10) - 64,
        24 << 10,
        (24 << 10) + 64,
        65536 - 64,
        65536,
        65536 + 64,
        (640 << 10) - 64,
        640 << 10,
        (640 << 10) + 64,
        (2 << 20) - 64,
        2 << 20,
        (2 << 20) + 64,
    )
    cases = []
    for size in sizes:
        inp = torch.empty((size // 64, 32), dtype=torch.bfloat16, device=manager.device)
        _fill(inp, rank, size)
        out = torch.empty_like(inp)
        plan = _prepare(session, manager, "all_reduce", inp, out)
        push = plan.query.surface == "PCIeTwoShotBF16.all_reduce"
        if not benchmark_only:
            assert push == (640 << 10 < size <= 2 << 20), (size, plan.query.surface)
            if not push:
                expected_transport = (
                    "remote_push_full" if size <= 24 << 10 else "remote_push_rsag"
                )
                assert plan.query.call["transport"] == expected_transport
        with session.capture():
            manager.all_reduce(inp, plan=plan, out=out)
        torch.cuda.synchronize()
        if not benchmark_only:
            assert torch.equal(out, _reference(inp)), ("eager", size)
        graph = torch.cuda.CUDAGraph()
        dist.barrier()
        with session.capture(), manager.capture(), torch.cuda.graph(graph):
            manager.all_reduce(inp, plan=plan, out=out)
        cases.append((size, inp, out, plan, graph))
    if not benchmark_only:
        fused_input = torch.empty(
            (1, 32768), device=manager.device, dtype=torch.bfloat16
        )
        fused_plan = manager.plan(
            fused_input, operation="all_reduce_fused_add_rms_norm"
        )
        assert (
            fused_plan.query.surface == "OneshotAllReduce.all_reduce_fused_add_rms_norm"
        )
        over_capacity = torch.empty(
            ((2 << 20) + 128) // 2, device=manager.device, dtype=torch.bfloat16
        )
        with pytest.raises(ValueError, match="requirements"):
            manager.plan(over_capacity)
        # Alternating sizes and live inputs share one persistent shard and epoch.
        for step in range(5):
            for size, inp, out, plan, graph in cases[:: 1 if step % 2 else -1]:
                _fill(inp, rank, 100 + step + size)
                expected = _reference(inp)
                out.fill_(float("nan"))
                torch.cuda.synchronize()
                dist.barrier()
                before = _allocation_state(manager.device)
                with (
                    session.capture(),
                    kernel_resolution_guard("prepared all-reduce replay"),
                ):
                    graph.replay()
                torch.cuda.synchronize()
                assert _allocation_state(manager.device) == before
                assert torch.equal(out, expected), ("graph", size, step)
    if benchmark:
        for size, inp, out, plan, graph in cases:
            durations = []
            for sample in range(5):
                for _ in range(8):
                    graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                local = torch.tensor(
                    [start.elapsed_time(end) * 1000 / 100], device=manager.device
                )
                dist.all_reduce(local, op=dist.ReduceOp.MAX)
                durations.append(round(local.item(), 3))
            if rank == 0:
                print(
                    f"allreduce bytes={size} surface={plan.query.surface} graph_us_max_rank={durations}",
                    flush=True,
                )
    for _, _, _, _, graph in cases:
        graph.reset()


def _check_dispatch_contracts(session, manager, rank):
    inp = torch.empty(524288, device=manager.device, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    _fill(inp, rank, 71)
    prepared = _prepare(session, manager, "all_reduce", inp, out)
    unused = manager.plan(inp, out=out)
    assert prepared is not unused and unused._prepared is None
    graph = torch.cuda.CUDAGraph()
    with session.capture(), manager.capture(), torch.cuda.graph(graph):
        manager.all_reduce(inp, plan=prepared, out=out)
    with session.capture():
        with pytest.raises(RuntimeError, match="not prepared|capture|frozen"):
            manager.all_reduce(inp, plan=unused, out=out)
    foreign = object.__new__(PCIeTwoShotBF16)
    foreign.device = manager.device
    with session.capture(), pytest.raises(ValueError, match="runtime|another"):
        foreign.all_reduce(inp, plan=prepared, out=out)
    expected = _reference(inp)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    graph.reset()

    omitted = _prepare(session, manager, "all_reduce", inp, None)
    assert omitted.query.surface == "PCIeTwoShotBF16.all_reduce"
    with session.capture():
        actual = manager.all_reduce(inp, plan=omitted)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)

    stream = torch.cuda.Stream(device=manager.device)
    with torch.cuda.stream(stream):
        streamed = manager.plan(inp, out=out, stream=stream)
    assert streamed.query.surface == "PCIeTwoShotBF16.all_reduce"
    # Use the prepared compatible ABI with the explicit stream launch surface.
    stream.wait_stream(torch.cuda.current_stream())
    with session.capture():
        manager.all_reduce(inp, plan=prepared, out=out, stream=stream)
    stream.synchronize()
    assert torch.equal(out, expected)

    alias = _prepare(session, manager, "all_reduce", inp, inp)
    assert alias.query.surface == "OneshotAllReduce.all_reduce"
    _fill(inp, rank, 72)
    expected = _reference(inp)
    with session.capture():
        manager.all_reduce(inp, plan=alias, out=inp)
    torch.cuda.synchronize()
    assert torch.equal(inp, expected)
    graph = torch.cuda.CUDAGraph()
    with session.capture(), manager.capture(), torch.cuda.graph(graph):
        manager.all_reduce(inp, plan=alias, out=inp)
    _fill(inp, rank, 73)
    expected = _reference(inp)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(inp, expected)
    graph.reset()


def _check_capability_failure(device, rank):
    from b12x.comm.pcie.pcie_allreduce import _bf16_twoshot_supported

    original = torch.cuda.get_device_properties

    def properties(queried_device):
        if rank == 1:
            torch.cuda.get_device_properties = original
            raise RuntimeError("injected rank-1 capability query failure")
        return original(queried_device)

    torch.cuda.get_device_properties = properties
    try:
        with pytest.raises(
            RuntimeError, match="injected rank-1 capability query failure"
        ):
            _bf16_twoshot_supported(
                exchange_group=dist.group.WORLD, device=device, world_size=4
            )
    finally:
        torch.cuda.get_device_properties = original
    dist.barrier()


def _check_mixed(session, pool, rank):
    # Native 8-element rows match the dispatcher's geometry, including odd slots.
    cases = []
    for rows in (4, 4096, 4092, pool.max_rows):
        inp = torch.empty(
            (rows, pool.row_elems), dtype=torch.bfloat16, device=pool.device
        )
        _fill(inp, rank, rows)
        out, shard, gathered = (
            torch.empty_like(inp),
            torch.empty_like(inp[: rows // 4]),
            torch.empty_like(inp),
        )
        ar = _prepare(session, pool, "all_reduce", inp, out)
        rs = _prepare(session, pool, "reduce_scatter", inp, shard)
        ag = _prepare(session, pool, "all_gather", shard, gathered)
        with session.capture():
            pool.reduce_scatter(inp, plan=rs, out=shard)  # Offset the initial slot.
            pool.all_reduce(inp, plan=ar, out=out)
            pool.all_gather(shard, plan=ag, out=gathered)
        torch.cuda.synchronize()
        expected = _reference(inp)
        assert torch.equal(out, expected) and torch.equal(gathered, expected)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), pool.capture(plan=ar), torch.cuda.graph(graph):
            pool.all_reduce(inp, plan=ar, out=out)
            pool.reduce_scatter(inp, plan=rs, out=shard)
            pool.all_gather(shard, plan=ag, out=gathered)
            pool.reduce_scatter(
                inp, plan=rs, out=shard
            )  # Five protocol epochs per replay.
        cases.append((inp, out, shard, gathered, ar, graph))
    epoch = _epoch(pool)
    shard_address = pool._push_shard.data_ptr()
    owners = tuple((b.local_ptr, tuple(b.remote_ptrs)) for b in pool._owned_buffers)
    for step in range(8):
        for inp, out, shard, gathered, ar, graph in cases[:: 1 if step % 2 else -1]:
            _fill(inp, rank, step + 1000 + inp.numel())
            expected = _reference(inp)
            for destination in (out, shard, gathered):
                destination.fill_(float("nan"))
            torch.cuda.synchronize()
            dist.barrier()
            before = _allocation_state(pool.device)
            with (
                session.capture(),
                kernel_resolution_guard("mixed BF16 collective replay"),
            ):
                graph.replay()
            torch.cuda.synchronize()
            assert _allocation_state(pool.device) == before
            epoch += 5
            assert _epoch(pool) == epoch
            assert pool._push_shard.data_ptr() == shard_address
            assert (
                tuple((b.local_ptr, tuple(b.remote_ptrs)) for b in pool._owned_buffers)
                == owners
            )
            assert torch.equal(out, expected) and torch.equal(gathered, expected)
            local = expected.view(4, -1)[rank].view_as(shard)
            assert torch.equal(shard, local)
    for *_, graph in cases:
        graph.reset()
    inp, out, shard, gathered, ar, _ = cases[0]
    rs = _prepare(session, pool, "reduce_scatter", inp, shard)
    ag = _prepare(session, pool, "all_gather", shard, gathered)
    overlap = torch.empty(inp.numel() + 8, dtype=inp.dtype, device=inp.device)
    overlap_inp, overlap_out = overlap[:-8].view_as(inp), overlap[8:].view_as(inp)
    for operation, payload, plan, invalid, error, message in (
        (pool.all_reduce, inp, ar, out.view(-1), ValueError, "output shape"),
        (
            pool.all_reduce,
            overlap_inp,
            ar,
            overlap_out,
            ValueError,
            "must not overlap input",
        ),
        (pool.reduce_scatter, inp, rs, inp[:1], ValueError, "must not overlap payload"),
        (pool.all_gather, inp[:1], ag, inp, ValueError, "must not overlap payload"),
        (
            pool.reduce_scatter,
            inp,
            rs,
            shard.to(torch.float16),
            TypeError,
            "must be bfloat16",
        ),
    ):
        with pytest.raises(error, match=message):
            operation(payload, plan=plan, out=invalid)
    assert not pool.accepts(
        torch.empty(
            pool.max_rows * pool.row_elems + 32,
            device=pool.device,
            dtype=torch.bfloat16,
        )
    )
    assert not pool.accepts(torch.empty(32, device=pool.device, dtype=torch.float16))
    storage = torch.empty(33, device=pool.device, dtype=torch.bfloat16)
    assert not pool.accepts(storage[1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("push", "pull"), default="push")
    parser.add_argument("--packed-only", action="store_true")
    parser.add_argument("--alias-only", action="store_true")
    parser.add_argument("--exact-inputs", action="store_true")
    parser.add_argument("--equal-shards", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args()
    if args.transport == "pull" and not (args.packed_only or args.alias_only):
        parser.error("--transport pull requires --packed-only or --alias-only")
    if (args.exact_inputs or args.equal_shards) and not args.packed_only:
        parser.error("--exact-inputs and --equal-shards require --packed-only")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if args.equal_shards and dist.get_world_size() != 4:
        parser.error("--equal-shards requires four ranks")
    if args.alias_only and args.packed_only:
        parser.error("--alias-only and --packed-only are mutually exclusive")
    if args.packed_only or args.alias_only:
        pool = PCIeTwoShotBF16.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_rows=65536 if args.equal_shards else 65537,
            row_elems=8,
        )
        session = PreparationSession(device=device, autotune=False, compile_workers=1)
        try:
            if args.alias_only:
                _check_exact_alias(session, pool, rank, transport=args.transport)
            else:
                _check_packed_tails(
                    session,
                    pool,
                    rank,
                    exact_inputs=args.exact_inputs,
                    equal_shards=args.equal_shards,
                    transport=args.transport,
                )
            if rank == 0:
                print(
                    f"TP{dist.get_world_size()} BF16 "
                    f"{'exact-alias' if args.alias_only else 'packed-tail'} eager/graph correctness OK",
                    flush=True,
                )
        finally:
            torch.cuda.synchronize()
            session.close()
            pool.close()
            dist.destroy_process_group()
        return
    assert dist.get_world_size() == 4
    _check_capability_failure(device, rank)
    manager = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=(2 << 20) + 64,
        single_channel=True,
    )
    manager.prepare_channels(("test:bf16",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    try:
        _check_dispatch(
            session,
            manager,
            rank,
            benchmark=args.benchmark or args.benchmark_only,
            benchmark_only=args.benchmark_only,
        )
        if not args.benchmark_only:
            _check_dispatch_contracts(session, manager, rank)
            _check_payload_ready(session, manager, rank)
            _check_payload_protocol_transitions(session, manager, rank)
            _check_mixed(session, manager._bf16_twoshot, rank)
            if rank == 0:
                print(
                    "TP4 BF16 prepared dispatch and mixed eager/graph correctness OK",
                    flush=True,
                )
    finally:
        torch.cuda.synchronize()
        session.close()
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
