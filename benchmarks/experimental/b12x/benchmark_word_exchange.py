"""TP4 all-gather comparisons with exact payload checks and paired GPU timing.

Run with torchrun on four idle peer-access GPUs. The cold protocol times one
collective after L2 eviction; the warm protocol times a chain of32 collectives.
Compare candidates within each protocol, not the two protocols against each
other. Each sample uses the maximum CUDA event interval across the four ranks.
"""

import json
import random
import statistics
from pathlib import Path

import torch
from b12x.preparation import PreparedCall
from b12x.preparation._measurement import prepare_race_steps, measure_race_steps


def consume(generator):
    while True:
        try:
            next(generator)
        except StopIteration as result:
            return result.value


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config(plan):
    return plan.contract.config_payload(plan.selection.config).to_dict()


def measure(
    calls,
    producers,
    *,
    device,
    cache_policy,
    rounds=12,
    samples=7,
    repetitions=1,
    collective=None,
):
    """All arms use the same held-out producers and complete every balanced round."""
    names = tuple(calls)
    prepared = [
        PreparedCall(
            run=calls[name],
            produce=producers[0],
            benchmark_producers=tuple(producers),
            capture_safe=True,
        )
        for name in names
    ]
    raw = []
    rank_raw = []

    def samples_exchange(indices, turn, repetition, groups):
        peers = collective((indices, groups))
        assert all(peer[0] == indices for peer in peers)
        rank_raw.append(
            dict(
                turn=turn,
                repetition=repetition,
                indices=indices,
                ranks=[peer[1] for peer in peers],
            )
        )
        result = tuple(
            tuple(max(peer[1][i][j] for peer in peers) for j in range(len(row)))
            for i, row in enumerate(groups)
        )
        if False:
            yield
        return result

    def launch_exchange(indices, turn, repetition):
        state = (indices, turn, repetition)
        assert all(peer == state for peer in collective(state))
        if False:
            yield

    timing = consume(
        prepare_race_steps(
            prepared,
            device_ordinal=device.index or 0,
            samples=samples,
            primed=True,
            cache_policy=cache_policy,
            repetitions=repetitions,
        )
    )
    try:
        consume(
            measure_race_steps(
                timing,
                device_ordinal=device.index or 0,
                rounds=rounds,
                eliminate=False,
                adaptive_repeats=False,
                sample_observer=lambda arm, turn, repetition, values: raw.append(
                    dict(
                        arm=names[arm],
                        turn=turn,
                        repetition=repetition,
                        us=list(values),
                    )
                ),
                sample_exchange=samples_exchange if collective else None,
                launch_exchange=launch_exchange if collective else None,
            )
        )
    finally:
        timing.close()
    summaries = {}
    for name in names:
        by_round = [statistics.fmean(row["us"]) for row in raw if row["arm"] == name]
        summaries[name] = dict(
            mean_us=statistics.fmean(by_round),
            stdev_round_us=statistics.stdev(by_round),
            rounds=by_round,
        )
    return dict(
        cache_policy=cache_policy,
        repetitions=repetitions,
        summaries=summaries,
        samples=raw,
        rank_samples=rank_raw,
    )


def paired(result, control, candidate):
    a = result["summaries"][control]["rounds"]
    b = result["summaries"][candidate]["rounds"]
    changes = [y - x for x, y in zip(a, b, strict=True)]
    rng = random.Random(8831)
    means = sorted(
        statistics.fmean(rng.choices(changes, k=len(changes))) for _ in range(10000)
    )
    return dict(
        control=control,
        candidate=candidate,
        mean_delta_us=statistics.fmean(changes),
        mean_change_percent=(statistics.fmean(b) / statistics.fmean(a) - 1) * 100,
        bootstrap95_delta_us=[means[249], means[9749]],
    )


def graph_calls(arms, *, session, device, repetitions):
    graphs = {}
    for name, run in arms.items():
        run()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            for _ in range(repetitions):
                run()
        graphs[name] = graph
    # Queue enough actual work to leave idle clock states before measurement.
    for _ in range(100):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize(device)
    return graphs


import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from b12x.comm import pcie
from b12x.comm.pcie import _dcp_preparation as dcp
from b12x.preparation import (
    PreparationSession,
    CollectiveRequirement,
    FinalistPolicy,
    autotune_effort,
)


def exchange(value):
    peers = [None] * 4
    dist.all_gather_object(peers, value)
    return peers


def prepare(session, requests, *, tune=True):
    job = session.begin(requests, autotune=tune)
    state = None
    allowed = {}
    try:
        while True:
            failure = None
            try:
                if state is None or not state.done:
                    state = job.advance(**allowed)
            except Exception as error:
                failure = repr(error)
            peers = exchange((failure, state))
            if any(error for error, _ in peers):
                raise RuntimeError([error for error, _ in peers])
            states = [state for _, state in peers]
            if all(state.done for state in states):
                return job.result()
            allowed = {}
            if all(state.ready_collective_tuning is not None for state in states):
                allowed = {
                    "collective_tuning": tuple(
                        state.ready_collective_tuning for state in states
                    )
                }
            elif all(state.ready_collectives for state in states):
                items = [state.ready_collectives[0] for state in states]
                assert all(value == items[0] for value in items)
                allowed = {"collective_key": items[0].key}
    except BaseException:
        job.close()
        raise


def payload(rows, words, seed):
    bits = torch.randint(
        -(2**31),
        2**31,
        (4, rows, words),
        dtype=torch.int32,
        generator=torch.Generator().manual_seed(seed),
    )
    # Summary packets also contain zero padding; exercise both entropy patterns.
    if seed % 2:
        bits[..., 5:] = 0
    return bits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--selections",
        type=Path,
        help="independent preparation results; fixes all compared configs before evaluation",
    )
    parser.add_argument(
        "--cases", default="stats1,stats4,stats7,stats8,confidence,logprobs,capacity"
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    if args.output.exists():
        parser.error("output must not exist")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    cases = {
        "stats1": (1, 32, 8),
        "stats4": (4, 32, 8),
        "stats7": (7, 32, 8),
        "stats8": (8, 32, 8),
        "confidence": (1, 128, 11),
        "logprobs": (8, 256, 8),
        "capacity": (128, 32, 128),
    }
    owners = {}
    records = []
    selected = (
        {}
        if args.selections is None
        else {r["name"]: r for r in json.loads(args.selections.read_text())["results"]}
    )
    try:
        for name in args.cases.split(","):
            rows, words, capacity = cases[name]
            key = (words, capacity)
            if key not in owners:
                owners[key] = pcie.DcpAllToAll.from_exchange_group(
                    exchange_group=dist.group.WORLD,
                    device=device,
                    max_batch_size=capacity,
                    total_heads=words // 32 * 4,
                    head_dim=64,
                    stream_affine=False,
                )
            runtime = owners[key]
            source = torch.empty(
                (rows, words // 32, 64), dtype=torch.bfloat16, device=device
            )
            out = torch.empty(
                (rows, words // 32 * 4, 64), dtype=torch.bfloat16, device=device
            )
            train = payload(rows, words, 182).to(device)
            source.view(torch.int32).reshape(rows, words).copy_(train[rank])
            evaluation = [
                payload(rows, words, seed).to(device) for seed in (8122, 8123)
            ]
            query = pcie.query_from_runtime(
                runtime,
                surface="DcpAllToAll.all_gather_heads",
                call=dict(local_input=source, out=out),
            )
            plan = pcie.plan(query, runtime=runtime)

            def request(p, label, policy=None):
                return p.request(
                    name=label,
                    collective=CollectiveRequirement(label, tuple(range(4))),
                    prepare_call=lambda state: dcp.prepare_call(
                        state, local_input=source, out=out
                    ),
                    finalist_policy=policy,
                )

            with PreparationSession(
                device=device,
                autotune=args.selections is None,
                compile_workers=8,
                cache_dir=args.output.parent / f"gather-selections-rank{rank}",
                namespace={"case": name, "purpose": args.output.stem},
            ) as session:
                result = None
                if args.selections is None:
                    result = prepare(
                        session,
                        (
                            request(
                                plan,
                                name,
                                FinalistPolicy(
                                    family_fields=("implementation",), max_total=2
                                ),
                            ),
                        ),
                    )
                    choices = {
                        item.selection.config.implementation: item.selection.config
                        for item in result.finalists[name]
                    }
                    automatic = config(plan)
                else:
                    saved = selected[name]
                    assert (
                        saved["rows"],
                        saved["words_per_row"],
                        saved["capacity_rows"],
                    ) == (rows, words, capacity)
                    choices = {
                        label: pcie.CollectiveConfig(**c)
                        for label, c in saved["configs"].items()
                    }
                    automatic = saved["autotuner"]
                assert set(choices) == {"dcp", "dcp_word"}, choices
                plans = {
                    label: pcie.plan(query, runtime=runtime, override=value)
                    for label, value in choices.items()
                }
                variants = prepare(
                    session,
                    tuple(request(p, name + "." + label) for label, p in plans.items()),
                    tune=False,
                )
                session.freeze()
                arms = {
                    label: (lambda p=p: runtime.all_gather_heads(source, out, plan=p))
                    for label, p in plans.items()
                }
                for data in evaluation:
                    source.view(torch.int32).reshape(rows, words).copy_(data[rank])
                    expected = (
                        data.permute(1, 0, 2).contiguous().reshape(rows, words * 4)
                    )
                    for run in arms.values():
                        out.view(torch.uint8).fill_(193)
                        dist.barrier()
                        run()
                        torch.cuda.synchronize()
                        assert torch.equal(
                            out.view(torch.int32).reshape(rows, words * 4), expected
                        )
                        assert torch.equal(
                            source.view(torch.int32).reshape(rows, words), data[rank]
                        )
                row = dict(
                    name=name,
                    rows=rows,
                    words_per_row=words,
                    bytes_per_rank=rows * words * 4,
                    capacity_rows=capacity,
                    autotuner=automatic,
                    configs={k: config(v) for k, v in plans.items()},
                    correctness="passed",
                    measurements=[],
                )
                if not args.prepare_only:
                    producers = [
                        lambda data=data: source.view(torch.int32)
                        .reshape(rows, words)
                        .copy_(data[rank])
                        for data in evaluation
                    ]
                    for policy, repetitions in [("cold", 1), ("warm", 32)]:
                        dist.barrier()
                        graphs = graph_calls(
                            arms,
                            session=session,
                            device=device,
                            repetitions=repetitions,
                        )
                        try:
                            measured = measure(
                                {label: g.replay for label, g in graphs.items()},
                                producers,
                                device=device,
                                cache_policy=policy,
                                repetitions=repetitions,
                                collective=exchange,
                            )
                            measured["comparison"] = paired(measured, "dcp", "dcp_word")
                            row["measurements"].append(measured)
                        finally:
                            for graph in graphs.values():
                                graph.reset()
                if result is not None:
                    result.close()
                variants.close()
            records.append(row)
            if rank == 0:
                write(
                    args.output,
                    dict(
                        prepare_only=args.prepare_only,
                        selection_source=None
                        if args.selections is None
                        else str(args.selections),
                        results=records,
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            case=name,
                            bytes=rows * words * 4,
                            selected=row["autotuner"],
                            timings=[
                                {k: v["mean_us"] for k, v in r["summaries"].items()}
                                for r in row["measurements"]
                            ],
                        )
                    ),
                    flush=True,
                )
    finally:
        for runtime in owners.values():
            runtime.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    with autotune_effort("default"):
        main()
