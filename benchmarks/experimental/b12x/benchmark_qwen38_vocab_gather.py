"""Two-rank BF16 vocabulary column gather with exact Qwen3.8 geometry."""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import b12x
from b12x.comm import pcie
from b12x.preparation import CollectiveRequirement, PreparationSession, PreparedCall

PRIMARY = (1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24)
GUARDS = (32, 64, 128)
COLUMNS = 124160


def prepare(session, requests, *, group, verbose):
    job = session.begin(requests)
    allowed = {}
    state = None
    last_readiness = None
    try:
        while True:
            error = None
            try:
                if state is None or not state.done:
                    state = job.advance(**allowed)
            except Exception as failure:
                error = str(failure)
            values = [None, None]
            dist.all_gather_object(values, (error, state), group=group)
            if any(value[0] is not None for value in values):
                raise RuntimeError(str([value[0] for value in values]))
            states = [value[1] for value in values]
            if all(item.done for item in states):
                return job.result()
            readiness = tuple(
                (
                    item.done,
                    item.pending_compilation,
                    bool(item.ready_cache),
                    bool(item.ready_tuning),
                    None
                    if item.ready_collective_tuning is None
                    else item.ready_collective_tuning.step,
                    bool(item.ready_collectives),
                )
                for item in states
            )
            if verbose and readiness != last_readiness and dist.get_rank() == 0:
                print("preparation", readiness, flush=True)
            last_readiness = readiness
            allowed = {}
            if all(item.ready_cache is not None for item in states):
                allowed = {"cache": tuple(item.ready_cache for item in states)}
            elif all(item.ready_tuning for item in states):
                allowed = {
                    "tuning": tuple(
                        value for item in states for value in item.ready_tuning
                    )
                }
            elif all(item.ready_collective_tuning is not None for item in states):
                allowed = {
                    "collective_tuning": tuple(
                        item.ready_collective_tuning for item in states
                    )
                }
            elif all(item.ready_collectives for item in states):
                pending = tuple(item.ready_collectives[0] for item in states)
                assert all(item == pending[0] for item in pending)
                allowed = {"collective_key": pending[0].key}
    except BaseException:
        job.close()
        raise


def worker(rank, port, args):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    host_group = dist.new_group(backend="gloo")
    root = Path(__file__).resolve().parents[3]
    assert Path(b12x.__file__).resolve().is_relative_to(root)
    assert (
        Path(importlib.util.find_spec("flashinfer").origin)
        .resolve()
        .is_relative_to(root)
    )
    runtime = pcie.DcpAllToAll.from_exchange_group(
        exchange_group=host_group,
        device=device,
        max_batch_size=max(args.rows),
        total_heads=3880,
        head_dim=64,
        stream_affine=False,
    )
    session = PreparationSession(
        cache_dir=args.cache / f"rank{rank}",
        device=device,
        autotune=args.tune,
        compile_workers=1,
        collective_barrier=lambda key, ranks: dist.barrier(group=host_group),
    )
    if rank == 0:
        print(
            json.dumps(
                dict(
                    b12x=b12x.__file__,
                    flashinfer=importlib.util.find_spec("flashinfer").origin,
                    rows=args.rows,
                    local_columns=COLUMNS,
                    dtype="bf16",
                    warmup=20,
                    operations=100,
                    samples=5,
                    statistic="median of rank-max event samples",
                )
            ),
            flush=True,
        )
    try:
        for rows in args.rows:
            inp = torch.empty((rows, 1940, 64), dtype=torch.bfloat16, device=device)
            out = torch.empty((rows, 3880, 64), dtype=inp.dtype, device=device)
            gathered = torch.empty((2, rows, COLUMNS), dtype=inp.dtype, device=device)
            call = dict(local_input=inp, out=out)
            query = pcie.query_from_runtime(
                runtime, surface="DcpAllToAll.all_gather_heads", call=call
            )
            plan = pcie.plan(query, runtime=runtime)
            prepare(
                session,
                (
                    plan.request(
                        name=f"gather{rows}",
                        collective=CollectiveRequirement(
                            key=f"gather{rows}", ranks=(0, 1)
                        ),
                        prepare_call=lambda state: PreparedCall(
                            run=lambda: state.run(**call), output=out
                        ),
                    ),
                ),
                group=host_group,
                verbose=args.verbose,
            )

            if rank == 0:
                print("selection", rows, plan.selection.config, flush=True)

            def native():
                runtime.all_gather_heads(inp, out=out, plan=plan)

            def nccl():
                dist.all_gather_into_tensor(
                    gathered.view(2 * rows, COLUMNS), inp.view(rows, COLUMNS)
                )
                out.view(rows, 2, COLUMNS).copy_(gathered.permute(1, 0, 2))

            for name, operation in (("native", native), ("nccl_pack", nccl)):
                if args.method != "both" and args.method != name:
                    continue
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        operation()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(10):
                        operation()
                for step in range(3):
                    values = []
                    for source in range(2):
                        generator = torch.Generator(device=device).manual_seed(
                            10007 * step + 97 * rows + source
                        )
                        values.append(
                            torch.randn(
                                inp.shape,
                                dtype=inp.dtype,
                                device=device,
                                generator=generator,
                            )
                        )
                    inp.copy_(values[rank])
                    out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        out, torch.cat(values, dim=1), rtol=0, atol=0
                    )
                samples = []
                for _ in range(5):
                    dist.barrier()
                    for _ in range(20):
                        graph.replay()
                    torch.cuda.synchronize()
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    elapsed = torch.tensor(
                        start.elapsed_time(end) * 10, dtype=torch.float64, device=device
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    samples.append(elapsed.item())
                if rank == 0:
                    print(
                        json.dumps(
                            dict(
                                method=name,
                                rows=rows,
                                bytes_per_rank=rows * COLUMNS * 2,
                                us=statistics.median(samples),
                                samples_us=samples,
                                correctness="exact",
                            )
                        ),
                        flush=True,
                    )
                del graph
                torch.cuda.synchronize()
    finally:
        runtime.close()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=lambda s: tuple(map(int, s.split(","))), default=PRIMARY + GUARDS
    )
    parser.add_argument(
        "--method", choices=("native", "nccl_pack", "both"), default="both"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
