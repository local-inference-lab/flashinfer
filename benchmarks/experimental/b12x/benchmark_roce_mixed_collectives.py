"""RoCEnante shard gathers and two-shot all-reduces replayed as one CUDA graph, with exact output checks.

Launch one rank per host with torchrun, for example on four DGX Sparks cabled
as a line in rank order, each naming its four RDMA functions::

    B12X_ROCE_HCA=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1 \\
    B12X_ROCE_GID_INDEX=3 \\
    torchrun --nnodes=4 --nproc-per-node=1 --node-rank=$RANK \\
        --master-addr=$MASTER --master-port=29650 \\
        benchmarks/experimental/b12x/benchmark_roce_mixed_collectives.py \\
        --iters 1000 --output mixed.json

``b12x`` must be importable: install this checkout, or set
``PYTHONPATH=$REPO:$REPO/flashinfer/experimental/b12x/_compat``.

One iteration replays a CUDA graph of pipelined collectives over BF16 operands
(``--mix``):

``mixed`` (default)
    ``gather_shards`` of 80 and 12 MiB, a 1 MiB two-shot ``all_reduce``,
    ``gather_shards`` of 16 and 16 MiB, a 1 MiB two-shot, ``gather_shards`` of
    8 MiB, a 1 MiB two-shot, and ``gather_shards`` of 128 KiB.  Gather sizes are
    those of the gathered output.
``gathers``
    The six ``gather_shards`` calls alone, in the same order.
``twoshots``
    Nine 1 MiB two-shot all-reduces.

The graph holds no barrier between ranks, so consecutive collectives follow one
another as closely as each rank's kernels allow.  Before every replay each rank
fills every source with a small integer derived from rank, operation and
iteration modulo 16, and fills every output with NaN.  The patterns can repeat
across operations and iterations and cannot distinguish stale data with the
same value or permutations within a peer's uniformly filled rows.  After the
replay it compares every output with its expected value exactly: a gather must
hold each rank's value in that rank's rows, and an all-reduce the sum over
ranks.  A wait timeout or a
proxy failure is reported as a stall, a wrong output as corruption; either ends
the run with exit status 3.

Each rank reports the median (p50), p90 and p99 of its iteration times, and
rank 0 reports the slowest rank's p50.  The timed interval runs from the graph
launch to the completion of everything queued on the device, so it includes
the fills that reset the operands for that iteration.

``--transport`` selects the open line (``ring_open=True``, the default), the
closed ring (``ring=True``) or the direct all-peer transport (``mesh``).  The row
counts require a world size that divides them: 2, 4, 8 or 16 ranks.

The benchmark cannot hold one rank's gather between staging its shard and
reading its peers' shards.  That delay needs a spin inside the gather kernel,
which this script does not add.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

MIB = 1 << 20

# (kind, rows, hidden) for a gather of rows x hidden BF16 values, (kind, bytes) for a two-shot.
GATHERS = (
    ("gather", 8192, 5120),  # 80 MiB
    ("gather", 6144, 1024),  # 12 MiB
    ("gather", 8192, 1024),  # 16 MiB
    ("gather", 8192, 1024),  # 16 MiB
    ("gather", 4096, 1024),  # 8 MiB
    ("gather", 512, 128),  # 128 KiB
)
TWOSHOT = ("twoshot", MIB)
MIXES = {
    "mixed": (
        GATHERS[0],
        GATHERS[1],
        TWOSHOT,
        GATHERS[2],
        GATHERS[3],
        TWOSHOT,
        GATHERS[4],
        TWOSHOT,
        GATHERS[5],
    ),
    "gathers": GATHERS,
    "twoshots": (TWOSHOT,) * 9,
}

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=1000)
ap.add_argument("--mix", choices=tuple(MIXES), default="mixed")
ap.add_argument("--transport", choices=("line", "ring", "mesh"), default="line")
ap.add_argument("--pipe-mib", type=int, default=128)
ap.add_argument("--chunk-kib", type=int, default=128)
ap.add_argument("--max-size-kib", type=int, default=64)
ap.add_argument("--output", default="")
args = ap.parse_args()
if args.iters < 1:
    ap.error("--iters must be at least 1")

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
device = torch.device("cuda", 0)
torch.cuda.set_device(device)

from b12x.comm import roce  # noqa: E402
from b12x.comm.roce import _preparation  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] rank{rank} {msg}", flush=True)


# Operand patterns vary by operation, rank and iteration modulo 16. Exact comparisons
# detect mismatched pattern values, but cannot distinguish stale data with the same value
# or permutations within a peer's uniformly filled rows.
def gather_value(index: int, iteration: int) -> int:
    return (iteration % 16) * 4 + index


def twoshot_value(index: int, iteration: int) -> int:
    return (iteration % 16) + index


def twoshot_sum(index: int, iteration: int) -> int:
    return world * (world + 1) // 2 + world * twoshot_value(index, iteration)


ops: list[dict] = []
for index, spec in enumerate(MIXES[args.mix]):
    if spec[0] == "gather":
        _, rows, hidden = spec
        if rows % world:
            raise SystemExit(f"{world} ranks do not divide the {rows} gathered rows")
        ops.append(
            {
                "kind": "gather",
                "index": index,
                "nbytes": rows * hidden * 2,
                "rows": rows,
                "hidden": hidden,
            }
        )
    else:
        ops.append(
            {
                "kind": "twoshot",
                "index": index,
                "nbytes": spec[1],
                "numel": spec[1] // 2,
            }
        )
if max(op["nbytes"] for op in ops) > (args.pipe_mib << 20):
    raise SystemExit(f"the largest operation exceeds the {args.pipe_mib} MiB pipe")
# The all-reduce check compares BF16 sums exactly, so every expected sum must be
# a BF16 value.
for op in ops:
    for iteration in range(16):
        if op["kind"] == "twoshot":
            total = twoshot_sum(op["index"], iteration)
            if torch.tensor(float(total), dtype=torch.bfloat16).item() != total:
                raise SystemExit(
                    f"the sum {total} over {world} ranks is not exact in BF16"
                )

runtime = roce.AllReduce(
    exchange_group=dist.group.WORLD,
    device=device,
    max_size=args.max_size_kib << 10,
    max_gather_bytes=1 << 20,
    pipe_bytes=args.pipe_mib << 20,
    # Every all-reduce of this benchmark takes the two-shot path.
    pipe_threshold=1,
    pipe_chunk_bytes=args.chunk_kib << 10,
    ring=args.transport == "ring",
    ring_open=args.transport == "line",
)
query = roce.query_from_runtime(
    runtime,
    surface="AllReduce.all_reduce",
    call={"dtypes": ("bfloat16",)},
    topology="roce_rdma",
    peer_hosts=tuple(f"rank-{r}" for r in range(world)),
)
plan = roce.plan(query, runtime=runtime)
seed_twoshot = torch.zeros((8, 8), dtype=torch.bfloat16, device=device)
seed_gather = torch.zeros((4, 8), dtype=torch.bfloat16, device=device)


def prepare(state) -> PreparedCall:
    calls = [
        _preparation.prepared_call(state, inp=seed_twoshot),
        _preparation.prepared_gather_shards_call(state, inp=seed_gather),
    ]
    return PreparedCall(run=lambda: [call.run() for call in calls])


with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
    session.prepare((plan.request(name="roce", prepare_call=prepare),))

for op in ops:
    if op["kind"] == "gather":
        op["src"] = torch.empty(
            (op["rows"] // world, op["hidden"]), dtype=torch.bfloat16, device=device
        )
        op["out"] = torch.full(
            (op["rows"], op["hidden"]),
            float("nan"),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        op["src"] = torch.empty(op["numel"], dtype=torch.bfloat16, device=device)
        op["out"] = torch.full(
            (op["numel"],), float("nan"), dtype=torch.bfloat16, device=device
        )


def launch(op: dict) -> None:
    if op["kind"] == "gather":
        runtime.gather_shards(op["src"], plan=plan, out=op["out"])
    else:
        runtime.all_reduce(op["src"], plan=plan, out=op["out"])


stream = torch.cuda.Stream(device=device)
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    for op in ops:
        launch(op)
torch.cuda.synchronize()
runtime.check_health()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    for op in ops:
        launch(op)
torch.cuda.synchronize()
runtime.check_health()

description = runtime.native_description()
log(
    "ready: "
    + json.dumps(
        {
            "mix": args.mix,
            "transport": args.transport,
            "bytes": [op["nbytes"] for op in ops],
            "pipe_mib": args.pipe_mib,
            "chunk_kib": args.chunk_kib,
            "proxy_abi": description["proxy_abi"],
            "ring_bytes": description.get("ring_bytes"),
            "rails": description["rails"],
            "lanes": description["lanes"],
            "spin_limit": runtime.spin_limit,
        }
    )
)

peers = torch.arange(world, dtype=torch.bfloat16, device=device)


def first_wrong_output(iteration: int) -> dict | None:
    """The first operation whose output differs from its expected value, or None."""
    for op in ops:
        if op["kind"] == "gather":
            view = op["out"].view(world, op["src"].shape[0], op["hidden"])
            want = peers + (1 + gather_value(op["index"], iteration))
            expected = want.view(world, 1, 1).expand_as(view)
            if not torch.equal(view, expected):
                wrong = view != expected
                return {
                    "op": op["index"],
                    "kind": op["kind"],
                    "nbytes": op["nbytes"],
                    "wrong_peers": [
                        int(p)
                        for p in wrong.any(dim=2).any(dim=1).nonzero().flatten()[:6]
                    ],
                    "wrong_rows": int(wrong.any(dim=2).sum().item()),
                }
        else:
            want_value = twoshot_sum(op["index"], iteration)
            if not bool((op["out"] == want_value).all()):
                return {
                    "op": op["index"],
                    "kind": op["kind"],
                    "nbytes": op["nbytes"],
                    "want": want_value,
                    "got_min": float(op["out"].min().item()),
                    "got_max": float(op["out"].max().item()),
                    "wrong": int((op["out"] != want_value).sum().item()),
                }
    return None


failures: list[dict] = []
completed = 0
timings: list[float] = []
started = time.perf_counter()

try:
    for iteration in range(args.iters):
        for op in ops:
            if op["kind"] == "gather":
                op["src"].fill_(float(rank + 1 + gather_value(op["index"], iteration)))
            else:
                op["src"].fill_(float(rank + 1 + twoshot_value(op["index"], iteration)))
            op["out"].fill_(float("nan"))
        begin = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - begin) * 1e3)
        try:
            runtime.check_health()
        except Exception as exc:  # noqa: BLE001 - a failed transport is a result
            failures.append(
                {
                    "class": "stall",
                    "iteration": iteration,
                    "message": str(exc).splitlines()[0][:400],
                }
            )
            log(f"STALL at iteration {iteration}: {exc}")
            break
        bad = first_wrong_output(iteration)
        if bad is not None:
            bad.update({"class": "corrupt", "iteration": iteration})
            failures.append(bad)
            log(f"CORRUPT at iteration {iteration}: {json.dumps(bad)}")
            break
        completed = iteration + 1
except Exception as exc:  # noqa: BLE001 - a launch on a poisoned transport raises
    failures.append(
        {
            "class": "stall",
            "iteration": completed,
            "message": f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}",
        }
    )
    log(f"STALL at iteration {completed}: {exc}")

elapsed = time.perf_counter() - started
ordered = sorted(timings)


def percentile(fraction: float) -> float | None:
    if not ordered:
        return None
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))], 3)


summary = {
    "rank": rank,
    "world": world,
    "mix": args.mix,
    "transport": args.transport,
    "proxy_abi": description["proxy_abi"],
    "iterations": completed,
    "ops_per_iteration": len(ops),
    "collectives": completed * len(ops),
    "failures": failures,
    "seconds": round(elapsed, 2),
    "p50_ms": percentile(0.50),
    "p90_ms": percentile(0.90),
    "p99_ms": percentile(0.99),
}
print("SUMMARY " + json.dumps(summary), flush=True)
sys.stdout.flush()
if failures:
    # Peers of a failed rank cannot finish their own collectives, so this rank
    # leaves without the closing exchange.
    os._exit(3)
summaries = [None] * world
dist.all_gather_object(summaries, summary)
if rank == 0:
    slowest = max(entry["p50_ms"] for entry in summaries)
    log(
        f"p50 per iteration by rank: {[entry['p50_ms'] for entry in summaries]} ms; "
        f"slowest rank {slowest} ms"
    )
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(
                {"args": vars(args), "slowest_p50_ms": slowest, "ranks": summaries},
                handle,
                indent=1,
            )
runtime.close()
dist.barrier()
dist.destroy_process_group()
