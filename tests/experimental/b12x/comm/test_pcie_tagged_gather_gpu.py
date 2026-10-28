"""Exact tagged gather across epoch wrap, mixed graph schedules and native reset.

Run with B12X_RUN_TAGGED_GATHER_TEST=1 and four explicit SM12x CUDA devices.
This test performs correctness replay only; it does not time GPU work.
"""

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
    os.getenv("B12X_RUN_TAGGED_GATHER_TEST") != "1",
    reason="requires four explicitly selected GPUs",
)


def _worker(rank, directory):
    from b12x.comm import pcie
    from b12x.comm.pcie import _dcp_preparation as prep
    from b12x.comm.pcie._tagged_gather import EPOCH_INDEX
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "gloo").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(minutes=5),
    )
    runtime = pcie.DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=128,
        total_heads=4,
        head_dim=64,
        stream_affine=False,
    )
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    sources, outputs, plans, graphs = {}, {}, {}, {}
    try:
        for mode in ("dcp", "dcp_tagged"):
            for rows in (1, 8, 128):
                name = (mode, rows)
                source = torch.zeros((rows, 1, 64), dtype=torch.bfloat16, device=device)
                output = torch.empty((rows, 4, 64), dtype=source.dtype, device=device)
                query = pcie.query_from_runtime(
                    runtime,
                    surface="DcpAllToAll.all_gather_heads",
                    call=dict(local_input=source, out=output),
                )
                plan = pcie.plan(
                    query,
                    runtime=runtime,
                    override=pcie.CollectiveConfig(
                        implementation=mode,
                        threads=128,
                        blocks=1 if mode == "dcp_tagged" else 16,
                    ),
                )
                request = plan.request(
                    name=f"{mode}.{rows}",
                    collective=CollectiveRequirement(f"{mode}.{rows}", tuple(range(4))),
                    prepare_call=lambda state, source=source, output=output: (
                        prep.prepare_call(state, local_input=source, out=output)
                    ),
                )
                session.prepare(
                    (request,),
                    coordinator=lambda progress: (
                        progress.ready_collectives[0].key
                        if progress.ready_collectives
                        else None
                    ),
                )
                sources[name], outputs[name], plans[name] = source, output, plan
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with session.capture(), torch.cuda.graph(graph):
                    runtime.all_gather_heads(source, output, plan=plan)
                graphs[name] = graph
        session.freeze()
        ipc = runtime._ipc
        stream = int(torch.cuda.current_stream().cuda_stream)

        def upload(address, value):
            ipc.cudaMemcpyAsync(
                address, ctypes.addressof(value), ctypes.sizeof(value), stream
            )
            torch.cuda.synchronize()

        def download(address, ctype):
            out = ctype()
            ipc.cudaMemcpyAsync(
                ctypes.addressof(out), address, ctypes.sizeof(out), stream
            )
            torch.cuda.synchronize()
            return out.value

        epoch_address = runtime._signal_ptrs[rank] + 4 * EPOCH_INDEX
        # Populate historical tag2 at every capacity slot, including words that
        # small graphs never overwrite. Wrap must clear these before growth.
        poison = (ctypes.c_uint64 * (4 * runtime._tagged_words))(
            *([2 << 32 | 0xDEADCAFE] * (4 * runtime._tagged_words))
        )
        for slot in (0, 1):
            upload(runtime._staging_ptrs[slot][rank] + runtime._tagged_offset, poison)
        upload(epoch_address, ctypes.c_uint32(0xFFFFFFFD))
        dist.barrier()
        schedule = [
            ("dcp_tagged", 1, 0xFFFFFFFE),
            ("dcp_tagged", 1, 0xFFFFFFFF),
            ("dcp_tagged", 8, 2),
            ("dcp_tagged", 128, 3),
            ("dcp", 8, 3),
            ("dcp_tagged", 1, 4),
            ("dcp", 1, 4),
            ("dcp_tagged", 8, 5),
        ]
        for step, (mode, rows, epoch) in enumerate(schedule):
            generator = torch.Generator().manual_seed(11237 + step)
            bits = torch.randint(
                -(2**31), 2**31, (4, rows, 32), dtype=torch.int32, generator=generator
            )
            bits[:, 0, :6] = torch.tensor(
                [0, -2147483648, 2143289345, -1, 2139095040, -8388608],
                dtype=torch.int32,
            )
            source, output = sources[mode, rows], outputs[mode, rows]
            source.view(torch.int32).copy_(bits[rank].view(rows, 1, 32).to(device))
            original = source.clone()
            output.view(torch.uint8).fill_(193)
            torch.cuda.synchronize()
            dist.barrier()
            before = torch.cuda.memory_stats()
            graphs[mode, rows].replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for counter in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[counter] == after[counter], counter
            expected = bits.permute(1, 0, 2).contiguous()
            torch.testing.assert_close(
                output.view(torch.int32).cpu(), expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                source.view(torch.int32), original.view(torch.int32), rtol=0, atol=0
            )
            assert download(epoch_address, ctypes.c_uint32) == epoch
            if epoch == 2:
                tail = (4 * runtime._tagged_words - 1) * 8
                for slot in (0, 1):
                    assert (
                        download(
                            runtime._staging_ptrs[slot][rank]
                            + runtime._tagged_offset
                            + tail,
                            ctypes.c_uint64,
                        )
                        == 0
                    )
        # Reject an overlapping view before any communication kernel launches.
        out = outputs["dcp_tagged", 8]
        alias = out.view(-1)[: 8 * 64].view(8, 1, 64)
        with pytest.raises(ValueError, match="disjoint"):
            runtime.all_gather_heads(alias, out, plan=plans["dcp_tagged", 8])
        # Exercise the exact native reset primitive used by package export,
        # retaining all IPC owners while it drains and zeros their allocations.
        import flare_heliopause

        context, _ = flare_heliopause.current_context()
        dist.barrier()
        flare_heliopause.export_reset_peer_storage(
            context, [(b.local_ptr, b.nbytes) for b in runtime._owned_buffers], runtime
        )
        dist.barrier()
        assert download(epoch_address, ctypes.c_uint32) == 0
        for rows in (8, 1, 128):
            source = sources["dcp_tagged", rows]
            source.view(torch.int32).fill_(rank + 17)
            output = outputs["dcp_tagged", rows]
            output.view(torch.uint8).fill_(193)
            graphs["dcp_tagged", rows].replay()
            torch.cuda.synchronize()
            expected = (
                torch.arange(17, 21, dtype=torch.int32)
                .view(1, 4, 1)
                .expand(rows, 4, 32)
            )
            torch.testing.assert_close(
                output.view(torch.int32).cpu(), expected, rtol=0, atol=0
            )
        print(
            f"rank{rank}: tagged wrap/mixed replay/exact bits/native reset passed",
            flush=True,
        )
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        session.close()
        runtime.close()
        dist.destroy_process_group()


def test_tagged_gather_wrap_mixed_replay_and_native_reset():
    assert torch.cuda.device_count() == 4
    with tempfile.TemporaryDirectory(prefix="b12x-tagged-gather-") as directory:
        mp.spawn(_worker, args=(directory,), nprocs=4, join=True)
