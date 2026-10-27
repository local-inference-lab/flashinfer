"""Compare Fraser's phased CUDA kernels with prepared b12x PCIe collectives.

Use four explicit peer-access GPUs with torch.distributed.run. ``--measure``
permits preparation tuning and event timing; without it only correctness and
replay checks run. Summary controls use producer output containing 32 FP32 words per row;
``--summary-widths 32`` also gives the tagged source the same full payload.
The source extension is a read-only diagnostic dependency,
never a b12x execution backend. Its Python route eligibility is recorded for
messages outside the source wrapper's 16 KiB--1 MiB reduction interval.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from benchmark_pcie_allreduce_transports import prepare as prepare_reduce
from benchmark_pcie_allreduce_transports import prepare_tuned, reference
from b12x.comm import pcie
from b12x.comm.pcie import _dcp_preparation as dcp
from b12x.preparation import CollectiveRequirement, PreparationSession, autotune_effort


def source_module(source: Path, build: Path):
    from torch.utils.cpp_extension import load

    kernels = source / "src/brrr/kernels"
    files = [kernels / "phased_all_reduce.cu", *sorted(kernels.glob("*.cuh"))]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    build.mkdir(parents=True, exist_ok=True)
    module = load(
        name=f"fraser_phased_{digest[:12]}",
        sources=[str(files[0])],
        build_directory=str(build),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"],
        verbose=False,
    )
    return module, digest


class SourceExchange:
    """Process-lifetime IPC state owned by the unmodified diagnostic extension."""

    def __init__(self, ext, capacity):
        self.ext = ext
        self.rank = dist.get_rank()
        self.capacity = (capacity + 15) // 16 * 16
        self.handle = ext.create(
            self.exchange(ext.flag_bytes() + 8 * self.capacity),
            self.rank,
            self.capacity,
        )
        self.ll = ext.create_ll(self.exchange(ext.ll_bytes()), self.rank)
        dist.barrier()

    def exchange(self, size):
        mine = self.ext.alloc_ipc(size)
        handles = [None] * dist.get_world_size()
        dist.all_gather_object(handles, mine[1:])
        return [
            mine[0] if rank == self.rank else self.ext.open_ipc(handle)
            for rank, handle in enumerate(handles)
        ]

    @staticmethod
    def ctas(nbytes):
        return 16 if nbytes <= 128 << 10 else 32 if nbytes <= 512 << 10 else 64


def prepare_gather(session, runtime, source, output, measure):
    call = dict(local_input=source, out=output)
    query = pcie.query_from_runtime(
        runtime, surface="DcpAllToAll.all_gather_heads", call=call
    )
    plan = pcie.plan(query, runtime=runtime)
    name = f"gather-{source.numel()}"
    request = plan.request(
        name=name,
        prepare_call=lambda state: dcp.prepare_call(state, **call),
        collective=CollectiveRequirement(name, tuple(range(dist.get_world_size()))),
    )
    if measure:
        prepare_tuned(session, (request,))
    else:
        session.prepare(
            (request,),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
    return plan, lambda: runtime.all_gather_heads(source, output, plan=plan)


def capture(call, chain, owner=None, manager=None):
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    if owner is None:
        call()
        torch.cuda.synchronize()
        dist.barrier()
    with (
        owner.capture() if owner else nullcontext(),
        manager.capture() if manager else nullcontext(),
        torch.cuda.graph(graph),
    ):
        for _ in range(chain):
            call()
    return graph


def measure_graphs(graphs, args):
    if not args.measure:
        return None
    for graph in graphs.values():
        for _ in range(args.warmup):
            graph.replay()
    torch.cuda.synchronize()
    samples = {arm: [] for arm in graphs}
    for sample in range(args.samples):
        order = list(graphs) if sample % 2 == 0 else list(reversed(graphs))
        for arm in order:
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(args.replays):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(
                start.elapsed_time(end) * 1000 / (args.replays * args.chain)
            )
    ranks = [None] * dist.get_world_size()
    dist.all_gather_object(ranks, samples)
    return dict(
        rank_samples_us=ranks,
        max_rank_median_us={
            arm: max(median(rank[arm]) for rank in ranks) for arm in graphs
        },
    )


def replay_check(graph, source, output, expected, *, exact_bytes=True):
    original = source.clone()
    output.view(torch.uint8).fill_(193)
    torch.cuda.synchronize()
    dist.barrier()
    before = torch.cuda.memory_stats()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    for key in (
        "allocation.all.allocated",
        "allocation.all.freed",
        "segment.all.allocated",
        "segment.all.freed",
    ):
        assert before[key] == after[key], key
    if exact_bytes:
        torch.testing.assert_close(
            output.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
    else:
        # BF16 reduction has exact numeric parity; signed zero may differ with
        # the rank-ordered FP32 sum. Gathered metadata remains byte-exact.
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        source.view(torch.uint8), original.view(torch.uint8), rtol=0, atol=0
    )


def record_plan(plan):
    state = plan.prepared.state
    return dict(
        selection=plan.selection.source,
        config=str(plan.selection.config),
        surface=state.query.surface,
        call=dict(state.query.call),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--operations", default="reduce,gather,words")
    parser.add_argument("--rows", default="1,8,17,32,64,128")
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--gather-columns", type=int, default=1280)
    parser.add_argument("--summary-widths", default="3,5")
    parser.add_argument(
        "--summary-layout", choices=("rows32", "flat512"), default="rows32"
    )
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=30)
    parser.add_argument("--chain", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    ext, source_digest = source_module(args.source, args.build_directory)
    if args.compile_only:
        print(
            json.dumps(dict(compiled_module=ext.__file__, source_sha256=source_digest))
        )
        return
    if args.output is None:
        parser.error("--output is required for distributed execution")
    if min(args.samples, args.replays, args.chain, args.warmup) <= 0:
        parser.error("measurement counts must be positive")
    rows = [int(value) for value in args.rows.split(",")]
    summary_widths = [int(value) for value in args.summary_widths.split(",")]
    if min(rows) < 1 or args.hidden_size % 8 or args.gather_columns % 64:
        parser.error(
            "positive rows, hidden alignment 8, and gather alignment 64 required"
        )
    if max(rows) * max(summary_widths) > 4096:
        parser.error("source tagged gather supports at most 4096 words")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 or ext.RANKS != 4:
        raise ValueError("Fraser source transport requires exactly four ranks")
    capacity = max(rows) * max(args.hidden_size, args.gather_columns) * 2
    original = SourceExchange(ext, capacity)
    manager = pcie.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=capacity,
        eager_buffer_bytes=capacity,
    )
    manager.prepare_channels(("benchmark:phased",))
    gathers = []
    session = PreparationSession(
        device=device,
        autotune=args.measure,
        cache_dir=args.tuning_cache_dir,
        compile_workers=1,
        collective_barrier=lambda key, ranks: dist.barrier(),
    )
    import flashinfer
    import subprocess

    props = torch.cuda.get_device_properties(device)
    hardware = dict(
        name=props.name,
        uuid="GPU-" + str(props.uuid).removeprefix("GPU-"),
        sm=tuple(torch.cuda.get_device_capability(device)),
    )
    hardware["clocks"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--id=" + hardware["uuid"],
            "--query-gpu=uuid,clocks.gr,clocks.mem,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    rank_hardware = [None] * world
    dist.all_gather_object(rank_hardware, hardware)
    software = dict(
        torch=torch.__version__, cuda=torch.version.cuda, flashinfer=flashinfer.__file__
    )
    results = []

    def publish(record):
        results.append(record)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    dict(
                        arguments={
                            k: str(v) if isinstance(v, Path) else v
                            for k, v in vars(args).items()
                        },
                        source_sha256=source_digest,
                        software=software,
                        hardware=rank_hardware,
                        results=results,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(record), flush=True)

    try:
        operations = args.operations.split(",")
        if any(
            operation not in ("reduce", "gather", "words") for operation in operations
        ):
            raise ValueError("operations must be reduce, gather, or words")
        for operation in operations:
            widths = (
                summary_widths
                if operation == "words"
                else [
                    args.hidden_size if operation == "reduce" else args.gather_columns
                ]
            )
            for width in widths:
                if operation != "reduce":
                    columns = (
                        width
                        if operation == "gather"
                        else (64 if args.summary_layout == "rows32" else 256)
                    )
                    runtime = pcie.DcpAllToAll.from_exchange_group(
                        exchange_group=dist.group.WORLD,
                        device=device,
                        max_batch_size=max(rows),
                        total_heads=columns * world // 64,
                        head_dim=64,
                        stream_affine=False,
                    )
                    gathers.append(runtime)
                for count in rows:
                    nbytes = count * width * (4 if operation == "words" else 2)
                    if operation == "reduce":
                        inp = torch.empty(
                            (count, width), device=device, dtype=torch.bfloat16
                        )
                        inp.fill_(rank + 1)
                        baseline, source_out = (
                            torch.empty_like(inp),
                            torch.empty_like(inp),
                        )
                        plan, ordinary = prepare_reduce(
                            session,
                            manager,
                            inp,
                            baseline,
                            "tuned" if args.measure else "auto",
                            256,
                            16,
                        )
                        source_call = lambda: ext.all_reduce(
                            original.handle, inp, source_out, original.ctas(nbytes)
                        )
                        graphs = {
                            "b12x": capture(ordinary, args.chain, session, manager),
                            "fraser": capture(source_call, args.chain),
                        }
                    elif operation == "gather":
                        inp = torch.empty(
                            (count, width), device=device, dtype=torch.bfloat16
                        )
                        inp.fill_(rank + 1)
                        baseline = torch.empty(
                            (count, width * world), device=device, dtype=inp.dtype
                        )
                        source_out = torch.empty_like(baseline)
                        plan, ordinary = prepare_gather(
                            session,
                            runtime,
                            inp.view(count, width // 64, 64),
                            baseline.view(count, width * world // 64, 64),
                            args.measure,
                        )
                        source_call = lambda: ext.all_gather(
                            original.handle,
                            inp,
                            source_out,
                            width * 2,
                            1 if nbytes <= 16384 else original.ctas(nbytes),
                        )
                        graphs = {
                            "b12x": capture(ordinary, args.chain, session),
                            "fraser": capture(source_call, args.chain),
                        }
                    else:
                        words = count * width
                        tiles = (
                            count
                            if args.summary_layout == "rows32"
                            else (words + 127) // 128
                        )
                        carrier_columns = 64 if args.summary_layout == "rows32" else 256
                        inp = torch.empty((words,), device=device, dtype=torch.int32)
                        inp.fill_(rank + 1)
                        packed = torch.zeros(
                            (tiles, carrier_columns),
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        baseline = torch.empty(
                            (tiles, world * carrier_columns),
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        source_out = torch.empty(
                            (world, words), device=device, dtype=torch.int32
                        )
                        plan, gather = prepare_gather(
                            session,
                            runtime,
                            packed.view(tiles, carrier_columns // 64, 64),
                            baseline.view(tiles, world * carrier_columns // 64, 64),
                            args.measure,
                        )

                        def ordinary():
                            if args.summary_layout == "rows32":
                                packed.view(torch.int32)[:, :width].copy_(
                                    inp.view(count, width)
                                )
                            else:
                                packed.view(torch.int32).flatten()[:words].copy_(inp)
                            gather()

                        graphs = {
                            "b12x_owned_padding": capture(
                                ordinary, args.chain, session
                            ),
                            "b12x_carrier_native": capture(gather, args.chain, session),
                            "fraser_tagged": capture(
                                lambda: ext.gather_words(original.ll, inp, source_out),
                                args.chain,
                            ),
                        }
                    checks = []
                    for step in range(4):
                        generator = torch.Generator().manual_seed(6191 + step)
                        if operation == "words":
                            data = torch.randint(
                                -(2**31),
                                2**31,
                                (world, words),
                                dtype=torch.int32,
                                generator=generator,
                            )
                            if step == 1:
                                data.zero_()
                            inp.copy_(data[rank].to(device))
                            if args.summary_layout == "rows32":
                                rank_padding = torch.zeros(
                                    (world, count, 32), dtype=torch.int32
                                )
                                rank_padding[:, :, :width].copy_(
                                    data.view(world, count, width)
                                )
                                expected_packed = rank_padding.permute(
                                    1, 0, 2
                                ).contiguous()
                            else:
                                rank_padding = torch.zeros(
                                    (world, tiles * 128), dtype=torch.int32
                                )
                                rank_padding[:, :words].copy_(data)
                                expected_packed = (
                                    rank_padding.view(world, tiles, 128)
                                    .permute(1, 0, 2)
                                    .contiguous()
                                )
                            replay_check(
                                graphs["b12x_owned_padding"],
                                inp,
                                baseline,
                                expected_packed.reshape(
                                    tiles, world * carrier_columns // 2
                                )
                                .view(torch.bfloat16)
                                .to(device),
                            )
                            replay_check(
                                graphs["b12x_carrier_native"],
                                inp,
                                baseline,
                                expected_packed.reshape(
                                    tiles, world * carrier_columns // 2
                                )
                                .view(torch.bfloat16)
                                .to(device),
                            )
                            replay_check(
                                graphs["fraser_tagged"],
                                inp,
                                source_out,
                                data.to(device),
                            )
                        else:
                            data = torch.randn(
                                (world, inp.numel()), generator=generator
                            ).bfloat16()
                            if step == 1:
                                data.zero_()
                            elif step == 2:
                                data[1] = -data[0]
                            inp.copy_(data[rank].view_as(inp).to(device))
                            if operation == "reduce":
                                transport = plan.prepared.state.query.call.get(
                                    "transport", "twoshot"
                                )
                                expected = reference(data, rank, transport).view_as(inp)
                                source_expected = torch.zeros_like(
                                    data[0], dtype=torch.float32
                                )
                                for peer in range(world):
                                    source_expected += data[peer].float()
                                source_expected = source_expected.bfloat16().view_as(
                                    inp
                                )
                            else:
                                expected = torch.cat(
                                    [data[peer].view_as(inp) for peer in range(world)],
                                    dim=1,
                                )
                                source_expected = expected
                            replay_check(
                                graphs["b12x"],
                                inp,
                                baseline,
                                expected.to(device),
                                exact_bytes=operation != "reduce",
                            )
                            replay_check(
                                graphs["fraser"],
                                inp,
                                source_out,
                                source_expected.to(device),
                                exact_bytes=operation != "reduce",
                            )
                        checks.append(
                            "exact_numeric" if operation == "reduce" else "exact_bytes"
                        )
                    measurement = measure_graphs(graphs, args)
                    publish(
                        dict(
                            operation=operation,
                            rows=count,
                            width=width,
                            logical_bytes=nbytes,
                            b12x_physical_input_bytes=baseline.numel()
                            * baseline.element_size()
                            // (world if operation != "reduce" else 1),
                            source_python_route_eligible=(
                                16384 < nbytes <= 1048576
                                if operation == "reduce"
                                else nbytes <= 1048576
                            ),
                            b12x=record_plan(plan),
                            checks=checks,
                            measurement=measurement,
                        )
                    )
                    for graph in graphs.values():
                        graph.reset()
    finally:
        torch.cuda.synchronize()
        session.close()
        for runtime in gathers:
            runtime.close()
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    with autotune_effort("default"):
        main()
