"""Queued DCP gathers share bank origins while word packets keep canonical offsets."""

import ctypes
from dataclasses import replace
from datetime import timedelta
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="requires explicitly assigned peer-access GPUs",
)


def _rank(rank, world, threads, directory):
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie import _dcp_preparation
    from b12x.comm.pcie._dcp_a2a_cute import _GRAPH_ARRIVED_INDEX, _GRAPH_EPOCH_INDEX
    from b12x.comm.pcie._gather_preparation import GatherInputs, selected_gather_profile
    from b12x.preparation import (
        CollectiveRequirement,
        FrozenMapping,
        PreparationSession,
        PreparedCall,
    )

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=5),
    )
    kinds = (
        "ordinary",
        "selected_barrier",
        "selected_phased_barrier",
        "selected_phased_push",
        "joint",
        "selected_word_exchange",
    )
    for origin in (0, 1):
        runtime = pcie.DcpAllToAll.from_exchange_group(
            exchange_group=dist.group.WORLD,
            device=device,
            max_batch_size=32,
            total_heads=world,
            head_dim=64,
            stream_affine=False,
        )
        session = PreparationSession(
            device=device, autotune=False, compile_workers=2, prime=False
        )
        plans, fixtures, graphs, requests = {}, {}, {}, []
        try:
            for rows in (1, 3, 32):
                source = torch.empty((rows, 1, 64), device=device, dtype=torch.bfloat16)
                output = torch.empty(
                    (rows, world, 64), device=device, dtype=torch.bfloat16
                )
                query = _dcp_preparation.query_from_runtime(
                    runtime,
                    surface="DcpAllToAll.all_gather_heads",
                    call=dict(
                        local_input=source, out=output, threads=threads, block_limit=8
                    ),
                )
                assert (query.call["threads"], query.call["block_limit"]) == (
                    threads,
                    8,
                )
                fixture = GatherInputs(SimpleNamespace(query=query, runtime=runtime))
                fixture.guard = torch.full(
                    (fixture.out.nbytes + 32,), 0xA5, device=device, dtype=torch.uint8
                )
                fixture.out = (
                    fixture.guard[16:-16]
                    .view(fixture.out.dtype)
                    .view(fixture.out.shape)
                )
                fixtures[rows] = fixture
                for kind in kinds:
                    q = query
                    if kind != "ordinary":
                        q = replace(
                            q,
                            call=FrozenMapping(
                                dict(
                                    q.call,
                                    gather_transport=kind,
                                    submode="word_exchange"
                                    if kind == "selected_word_exchange"
                                    else "barrier",
                                    active_blocks=8
                                    if kind.startswith("selected_phased")
                                    else 4,
                                )
                            ),
                        )
                    plan = _dcp_preparation.plan(q, runtime=runtime)
                    name = f"{rows}-{kind}"

                    def prepare(state, fixture=fixture):
                        return PreparedCall(
                            run=lambda: state.run(
                                local_input=fixture.source, out=fixture.out
                            ),
                            output=fixture.out,
                            owners=(fixture,),
                        )

                    requests.append(
                        plan.request(
                            name=name,
                            prepare_call=prepare,
                            collective=CollectiveRequirement(
                                key=name, ranks=tuple(range(world))
                            ),
                        )
                    )
                    plans[rows, kind] = plan
            session.prepare(
                tuple(requests),
                coordinator=lambda progress: (
                    progress.ready_collectives[0].key
                    if progress.ready_collectives
                    else None
                ),
            )
            dist.barrier()
            fixtures[1].produce(0)
            for _ in range(1 if origin else 2):
                plans[1, "ordinary"].prepared.state.run(
                    local_input=fixtures[1].source, out=fixtures[1].out
                )
            torch.cuda.synchronize()
            dist.barrier()
            for key, plan in plans.items():
                fixture = fixtures[key[0]]
                graph = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graph):
                    plan.prepared.state.run(local_input=fixture.source, out=fixture.out)
                graphs[key] = graph
            assert runtime._graph_base_slot == origin and runtime._device_slot_selection
            stream = torch.cuda.current_stream()

            def words(address, count, values=None):
                host = (ctypes.c_uint32 * count)(
                    *([0] * count if values is None else values)
                )
                runtime._ipc.cudaMemcpyAsync(
                    ctypes.addressof(host) if values is None else address,
                    address if values is None else ctypes.addressof(host),
                    ctypes.sizeof(host),
                    stream.cuda_stream,
                )
                stream.synchronize()
                return list(host)

            for mode in ("ordinary", "selected", "mixed"):
                keys = [
                    key
                    for key in plans
                    if mode == "mixed" or (key[1] == "ordinary") == (mode == "ordinary")
                ]
                schedule = []
                for turn in range(3):
                    for rows, kind in keys[:: (-1 if turn % 2 else 1)]:
                        submode = (
                            "word_exchange"
                            if kind == "selected_word_exchange"
                            or (kind == "joint" and turn % 2)
                            else "barrier"
                        )
                        schedule.append((rows, kind, submode, turn))
                saved = [torch.empty_like(fixtures[rows].out) for rows, *_ in schedule]
                for stages in (runtime._staging0_ptrs, runtime._staging1_ptrs):
                    runtime._ipc.cudaMemset(stages[rank], 0, runtime._slot_bytes)
                epoch_address = runtime._signal_ptrs[rank] + 4 * _GRAPH_EPOCH_INDEX
                arrived_address = runtime._signal_ptrs[rank] + 4 * _GRAPH_ARRIVED_INDEX
                counter_address = (
                    runtime._staging0_ptrs[rank] + runtime._exchange_counter_offset
                )
                assert words(arrived_address, 1) == [0]
                words(epoch_address, 1, [0xFFFFFFFD])
                expected_counters = [0xFFFFFFFD] * runtime._exchange_chunks
                words(counter_address, len(expected_counters), expected_counters)
                expected_epoch = 0xFFFFFFFD
                torch.cuda.synchronize()
                dist.barrier()
                before = torch.cuda.memory_stats()
                for (rows, kind, submode, seed), result in zip(
                    schedule, saved, strict=True
                ):
                    fixture = fixtures[rows]
                    fixture.produce(seed)
                    fixture.out.view(torch.uint8).fill_(0xA5)
                    if kind != "ordinary":
                        active = 8 if kind.startswith("selected_phased") else 4
                        plans[rows, kind].prepared.state.selector.fill_(
                            selected_gather_profile(kind, 8, submode, active)
                        )
                    graphs[rows, kind].replay()
                    result.copy_(fixture.out)
                    if submode == "word_exchange":
                        for chunk in range((rows * 128 + 255) // 256):
                            old = expected_counters[chunk]
                            expected_counters[chunk] = (
                                2 if old == 0xFFFFFFFF else old + 1
                            )
                    else:
                        expected_epoch = (expected_epoch + 1) & 0xFFFFFFFF
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                errors = []
                for counter in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    if before[counter] != after[counter]:
                        errors.append(counter)
                for (rows, kind, submode, seed), result in zip(
                    schedule, saved, strict=True
                ):
                    if not torch.equal(
                        result.view(torch.uint8).reshape(-1),
                        fixtures[rows].reference.expected[seed],
                    ):
                        errors.append((rows, kind, submode, seed, "payload"))
                for fixture in fixtures.values():
                    if not torch.all(fixture.guard[:16] == 0xA5) or not torch.all(
                        fixture.guard[-16:] == 0xA5
                    ):
                        errors.append("guard")
                if words(epoch_address, 1) != [expected_epoch] or words(
                    arrived_address, 1
                ) != [0]:
                    errors.append("barrier epochs")
                if words(counter_address, len(expected_counters)) != expected_counters:
                    errors.append("word packet counters")
                gathered = [None] * world
                dist.all_gather_object(gathered, errors)
                assert not any(gathered), (world, threads, origin, mode, gathered)
                # Release retained snapshots before the next allocation measurement.
                del result, saved
        finally:
            torch.cuda.synchronize()
            for graph in graphs.values():
                graph.reset()
            session.close()
            runtime.close()
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("threads", [32, 128])
def test_queued_gather_banks_and_word_packets(world, threads, tmp_path):
    import torch
    import torch.multiprocessing as mp

    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "assign GPUs explicitly"
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    mp.spawn(_rank, args=(world, threads, str(tmp_path)), nprocs=world, join=True)
