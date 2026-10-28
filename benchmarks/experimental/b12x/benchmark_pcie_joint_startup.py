"""Validate and measure captured joint mode selection on an established TP group."""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "flashinfer/experimental/b12x/_compat"))

import torch
import torch.distributed as dist

from b12x.comm.pcie import (
    joint_finalist_policy,
    joint_startup_spec,
    prepare_joint_startup_tuner,
)
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.comm.pcie._oneshot_preparation import _prepare_plain_call
from b12x.preparation import CollectiveRequirement, PreparationSession, FrozenMapping
from benchmark_pcie_allreduce_transports import prepare_tuned


def run_case(args, manager, size, family, alias):
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    if size <= 0 or size % (args.hidden_size * 2):
        raise ValueError("payload must contain complete positive rows")
    source = torch.full(
        (size // (args.hidden_size * 2), args.hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device=device,
    )
    out = source if alias else torch.empty_like(source)
    session = PreparationSession(
        device=device, autotune=True, compile_workers=1, cache_dir=args.cache
    )
    plan = manager.plan(source, out=out, joint_only=True)
    native = plan.query.alternatives[family]
    default = replace(
        plan.query.default,
        implementation=family,
        submode="full" if family == "joint_latency" else "payload_rsag",
        active_blocks=16,
    )
    plan = replace(
        plan,
        query=replace(
            plan.query, alternatives=FrozenMapping({family: native}), default=default
        ),
    )
    plans, sources, outputs = [plan], [source], [out]
    for _ in range(1, args.consumers):
        plans.append(replace(plan))
        sources.append(torch.full_like(source, rank + 1))
        outputs.append(sources[-1] if alias else torch.empty_like(out))
    requests = tuple(
        candidate.request(
            name=f"startup-{index}",
            finalist_policy=joint_finalist_policy(),
            collective=CollectiveRequirement("startup", tuple(range(world))),
            prepare_call=lambda state, inp=inp, output=output: _prepare_plain_call(
                state, inp=inp, out=output
            ),
        )
        for index, (candidate, inp, output) in enumerate(
            zip(plans, sources, outputs, strict=True)
        )
    )
    result = prepare_tuned(session, requests)
    if args.require_cache:
        assert result.cache_hits > 0 and result.benchmarked_candidates == 0
    specs = tuple(joint_startup_spec(candidate) for candidate in plans)
    spec = specs[0]
    assert len({item.selector.data_ptr() for item in specs}) == args.consumers
    previous_eviction = getattr(spec.state.runtime, "_joint_startup_eviction", None)
    tuner = prepare_joint_startup_tuner(specs, cache_policy=args.cache_policy)
    if previous_eviction is not None and args.cache_policy == "cold":
        assert tuner.eviction is previous_eviction[0]
    main_graph, tuner_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with session.capture(), manager.capture(), torch.cuda.graph(main_graph):
        for candidate, inp, output in zip(plans, sources, outputs, strict=True):
            manager.all_reduce(inp, out=output, plan=candidate)
    tuner.run()
    torch.cuda.synchronize()
    assert tuner.status.item() == 0
    startup_start = torch.cuda.Event(enable_timing=True, external=True)
    startup_end = torch.cuda.Event(enable_timing=True, external=True)
    with session.capture(), manager.capture(), torch.cuda.graph(tuner_graph):
        startup_start.record()
        tuner.run()
        startup_end.record()
    session.freeze()
    selected, startup_ms, startup_wall_ms = [], [], []
    for replay in range(3):
        tuner.status.fill_(-1)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        wall_start = time.perf_counter()
        tuner_graph.replay()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        elapsed = (startup_start.elapsed_time(startup_end), wall_ms)
        rank_elapsed = [None] * world
        dist.all_gather_object(rank_elapsed, elapsed)
        startup_ms.append(max(value[0] for value in rank_elapsed))
        startup_wall_ms.append(max(value[1] for value in rank_elapsed))
        assert tuner.status.item() == 0
        packed = int(spec.selector.item())
        assert all(int(item.selector.item()) == packed for item in specs)
        values = [None] * world
        dist.all_gather_object(values, packed)
        assert len(set(values)) == 1
        samples = tuner.samples.cpu().tolist()
        expected_index = min(
            range(len(samples)), key=lambda index: (median(samples[index]), index)
        )
        assert packed == int(tuner.profiles[expected_index].item())
        for index, inp in enumerate(sources):
            inp.fill_((rank + 1) * (replay + 1) * (index + 1))
        main_graph.replay()
        torch.cuda.synchronize()
        for index, output in enumerate(outputs):
            torch.testing.assert_close(
                output,
                torch.full_like(
                    output, world * (world + 1) // 2 * (replay + 1) * (index + 1)
                ),
                rtol=0,
                atol=0,
            )
        selected.append(packed)
    expected = tuner.inputs.reference.expected[0].view(-1)
    saved = expected[:1].clone()
    if rank == 0:
        expected[0] = float("nan")
    tuner.status.fill_(-1)
    tuner_graph.replay()
    torch.cuda.synchronize()
    assert tuner.status.item() == -1
    expected[:1].copy_(saved)
    tuner.status.fill_(-1)
    tuner_graph.replay()
    torch.cuda.synchronize()
    assert tuner.status.item() == 0
    mode_count = len(spec.profiles)
    events = [
        [
            (
                torch.cuda.Event(enable_timing=True, external=True),
                torch.cuda.Event(enable_timing=True, external=True),
            )
            for _ in range(21)
        ]
        for _ in range(mode_count)
    ]
    chain = torch.cuda.CUDAGraph()
    with manager.capture(), torch.cuda.graph(chain):
        tuner._timing("reset", 0, spec.selector)
        for round_index in range(7):
            for producer in range(3):
                for offset in range(mode_count):
                    mode_index = (offset + round_index) % mode_count
                    mode, active = spec.profiles[mode_index]
                    spec.selector.fill_((mode << 16) | active)
                    if tuner.eviction is not None:
                        torch.sum(tuner.eviction, dim=0, out=tuner.eviction_sum)
                    tuner.inputs.produce(producer)
                    sample_index = round_index * 3 + producer
                    sample = mode_index * 21 + sample_index
                    start, end = events[mode_index][sample_index]
                    tuner._timing("begin", sample, spec.selector)
                    start.record()
                    for _ in range(tuner.inputs.repetitions):
                        tuner._collective()
                    end.record()
                    tuner._timing("end", sample, spec.selector)
    event_rounds, paired_device_rounds = [], []
    for _ in range(3):
        chain.replay()
        torch.cuda.synchronize()
        local_times = [
            [
                start.elapsed_time(end) * 1000 / tuner.inputs.repetitions
                for start, end in row
            ]
            for row in events
        ]
        rank_times = [None] * world
        dist.all_gather_object(rank_times, local_times)
        event_rounds.append(
            [
                [max(peer[mode][sample] for peer in rank_times) for sample in range(21)]
                for mode in range(mode_count)
            ]
        )
        paired_device_rounds.append(tuner.samples.cpu().tolist())
    event_samples, paired_device_samples = event_rounds[-1], paired_device_rounds[-1]
    chain.reset()
    record = dict(
        bytes=size,
        shape=list(source.shape),
        world_size=world,
        capacity_bytes=spec.state.workspace.capacity_packs * 16,
        cache_policy=args.cache_policy,
        family=family,
        alias=alias,
        consumers=args.consumers,
        cache_hits=result.cache_hits,
        startup_resident_bytes=sum(
            entry.resident_nbytes for entry in tuner.memory_requirements().persistent
        ),
        profiles=spec.profiles,
        selected=selected,
        startup_ms=startup_ms,
        startup_wall_ms=startup_wall_ms,
        repetitions=tuner.inputs.repetitions,
        device_samples_ns=samples,
        event_samples_us=event_samples,
        paired_device_samples_ns=paired_device_samples,
        event_replays_us=event_rounds,
        paired_device_replays_ns=paired_device_rounds,
        config=dict(
            threads=spec.state.query.call["threads"],
            blocks=spec.state.query.call["blocks"],
        ),
    )
    if rank == 0:
        print(json.dumps(record), flush=True)
    tuner_graph.reset()
    main_graph.reset()
    session.close()
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", default="5120,81920,1310720,5242880,20971520")
    parser.add_argument("--families", default="joint_latency,joint_bandwidth")
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--consumers", type=int, default=2)
    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--alias", action="store_true")
    parser.add_argument("--cache-policy", choices=("cold", "warm"), default="cold")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(minutes=15))
    sizes = [int(value) for value in args.bytes.split(",")]
    if args.consumers <= 0:
        raise ValueError("consumers must be positive")
    capacity = max(sizes) if args.capacity_bytes is None else args.capacity_bytes
    if capacity < max(sizes):
        raise ValueError("capacity must cover every payload")
    manager = PCIeAllReduce.from_exchange_group(
        joint=True,
        exchange_group=dist.group.WORLD,
        device=torch.device("cuda", rank),
        single_channel=True,
        max_size=capacity,
        eager_buffer_bytes=capacity,
    )
    manager.prepare_channels(("startup",))
    try:
        records = [
            run_case(args, manager, size, family, args.alias)
            for size in sizes
            for family in args.families.split(",")
        ]
        if rank == 0:
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
    finally:
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
