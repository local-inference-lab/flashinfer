"""Exact distributed copies across launch changes, counter wrap and graph replay."""

import ctypes
from datetime import timedelta
import os
from pathlib import Path
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_WORD_EXCHANGE_TEST") != "1",
    reason="requires explicitly assigned peer-access GPUs",
)


def _rank(rank, world, folder, dtype_name):
    from b12x.comm import pcie
    from b12x.comm.pcie import _dcp_preparation as preparation
    from b12x.preparation import PreparationSession, CollectiveRequirement

    device = torch.device("cuda", rank)
    dtype = getattr(torch, dtype_name)
    words = 16 if dtype == torch.float8_e4m3fn else 32
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(folder) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=3),
    )
    runtime = pcie.DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=128,
        total_heads=world,
        head_dim=64,
        stream_affine=False,
    )
    graphs = {}
    data = {}
    plans = {}
    session = PreparationSession(device=device, autotune=False, compile_workers=2)

    def agree(errors):
        gathered = [None] * world
        dist.all_gather_object(gathered, errors)
        assert not any(gathered), gathered

    try:
        for label, rows, threads, blocks in (
            ("small", 1, 32, 1),
            ("wide", 128, 64, 32),
            ("grouped", 8, 128, 1),
            ("strided", 128, 32, 1),
            ("barrier", 8, 128, 16),
        ):
            source = torch.zeros((rows, 1, 64), dtype=dtype, device=device)
            output = torch.empty((rows, world, 64), dtype=source.dtype, device=device)
            query = pcie.query_from_runtime(
                runtime,
                surface="DcpAllToAll.all_gather_heads",
                call=dict(local_input=source, out=output),
            )
            if label != "barrier":
                chunks = (rows * words + 63) // 64
                threads = min(threads, 32 * (1 << (chunks - 1).bit_length()))
                covered = (rows * words + 64 * (threads // 32) - 1) // (
                    64 * (threads // 32)
                )
                blocks = min(blocks, covered)
            plan = pcie.plan(
                query,
                runtime=runtime,
                override=pcie.CollectiveConfig(
                    implementation="dcp" if label == "barrier" else "dcp_word",
                    threads=threads,
                    blocks=blocks,
                ),
            )
            session.prepare(
                (
                    plan.request(
                        name=label,
                        collective=CollectiveRequirement(label, tuple(range(world))),
                        prepare_call=lambda state,
                        source=source,
                        output=output: preparation.prepare_call(
                            state, local_input=source, out=output
                        ),
                    ),
                ),
                coordinator=lambda progress: progress.ready_collectives[0].key
                if progress.ready_collectives
                else None,
            )
            graph = torch.cuda.CUDAGraph()
            memory = plan.memory_requirements()
            allocation = next(
                item for item in memory.persistent if item.key[0] == "pcie-dcp-runtime"
            )
            assert (
                allocation.required_nbytes
                == allocation.resident_nbytes
                == sum(buffer.nbytes for buffer in runtime._owned_buffers)
            )
            dist.barrier()
            with session.capture(), torch.cuda.graph(graph):
                runtime.all_gather_heads(source, output, plan=plan)
            graphs[label] = graph
            data[label] = (source, output)
            plans[label] = plan
        session.freeze()
        ipc = runtime._ipc
        stream = int(torch.cuda.current_stream().cuda_stream)
        counter_address = (
            runtime._staging0_ptrs[rank] + runtime._exchange_counter_offset
        )

        def upload(address, value):
            ipc.cudaMemcpyAsync(
                address, ctypes.addressof(value), ctypes.sizeof(value), stream
            )
            torch.cuda.synchronize()

        def counters():
            result = (ctypes.c_uint32 * runtime._exchange_chunks)()
            ipc.cudaMemcpyAsync(
                ctypes.addressof(result), counter_address, ctypes.sizeof(result), stream
            )
            torch.cuda.synchronize()
            return list(result)

        # A stale sequence of two is reachable only for chunks whose counter
        # has already passed two. All chunks approach wrap, while the schedule
        # below advances them at different rates.
        initial = [0xFFFFFFFD] * runtime._exchange_chunks
        upload(counter_address, (ctypes.c_uint32 * len(initial))(*initial))
        agree([] if counters() == initial else ["counter initialization differs"])
        stale = (ctypes.c_uint64 * (world * runtime._exchange_chunks * 64))(
            *([2 << 32 | 0xDEADCAFE] * (world * runtime._exchange_chunks * 64))
        )
        for bank in (0, 1):
            upload(runtime._staging_ptrs[bank][rank] + runtime._exchange_offset, stale)
        expected_counters = initial.copy()
        for step, label in enumerate(
            ("small", "small", "wide", "barrier", "grouped", "strided", "small", "wide")
        ):
            source, output = data[label]
            rows = source.shape[0]
            bits = torch.randint(
                -(2**31),
                2**31,
                (world, rows, words),
                dtype=torch.int32,
                generator=torch.Generator().manual_seed(4193 + step),
            )
            bits[:, 0, :6] = torch.tensor(
                [0, -2147483648, -1, 2139095040, 2143289345, -8388608],
                dtype=torch.int32,
            )
            source.view(torch.int32).reshape(rows, words).copy_(bits[rank].to(device))
            output.view(torch.uint8).fill_(181)
            original = source.view(torch.int32).clone()
            dist.barrier()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graphs[label].replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            errors = []
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                if before[key] != after[key]:
                    errors.append((step, label, key, before[key], after[key]))
            expected = bits.permute(1, 0, 2).contiguous().reshape(rows, world * words)
            actual = output.view(torch.int32).reshape_as(expected).cpu()
            if not torch.equal(actual, expected):
                bad = torch.nonzero(actual != expected)[:12].tolist()
                errors.append(
                    (
                        step,
                        label,
                        "payload",
                        [
                            (pos, int(actual[tuple(pos)]), int(expected[tuple(pos)]))
                            for pos in bad
                        ],
                    )
                )
            if not torch.equal(source.view(torch.int32), original):
                errors.append((step, label, "source modified"))
            if label != "barrier":
                for chunk in range((rows * words + 63) // 64):
                    value = expected_counters[chunk]
                    expected_counters[chunk] = 2 if value == 0xFFFFFFFF else value + 1
            actual_counters = counters()
            if actual_counters != expected_counters:
                errors.append(
                    (step, label, "counters", actual_counters, expected_counters)
                )
            agree(errors)
        # Queue varying shapes without host synchronization between exchanges.
        # Result copies keep every payload observable after the shared buffers
        # have been reused by later graphs.
        history = []
        for step, label in enumerate(
            ("small", "wide", "grouped", "barrier", "strided") * 8
        ):
            source, output = data[label]
            rows = source.shape[0]
            bits = torch.randint(
                -(2**31),
                2**31,
                (world, rows, words),
                dtype=torch.int32,
                generator=torch.Generator().manual_seed(5821 + step),
            )
            history.append(
                (
                    label,
                    bits[rank].to(device),
                    bits.permute(1, 0, 2).contiguous(),
                    torch.empty_like(output),
                )
            )
        dist.barrier()
        for label, payload, _, saved in history:
            source, output = data[label]
            source.view(torch.int32).reshape_as(payload).copy_(payload)
            graphs[label].replay()
            saved.copy_(output)
        torch.cuda.synchronize()
        agree(
            [
                (step, label, "queued payload")
                for step, (label, _, expected, saved) in enumerate(history)
                if not torch.equal(saved.view(torch.int32).cpu(), expected)
            ]
        )
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16", "float8_e4m3fn"])
def test_word_exchange_replay_and_counter_lifetimes(world, dtype_name):
    if torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned GPUs")
    with tempfile.TemporaryDirectory() as folder:
        mp.spawn(_rank, args=(world, folder, dtype_name), nprocs=world, join=True)
