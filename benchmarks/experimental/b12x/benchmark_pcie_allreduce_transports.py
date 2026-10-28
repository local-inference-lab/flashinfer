"""Compare complete prepared PCIe all-reduces with changing CUDA graph inputs.

Run with torch.distributed.run and an explicit CUDA_VISIBLE_DEVICES assignment.
Transport overrides affect benchmark declarations only; auto uses normal routing.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import gc
import math
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

from b12x.comm.pcie import _oneshot_preparation as oneshot
from b12x.comm.pcie import _twoshot_preparation as twoshot
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.comm.pcie import CollectiveConfig
from b12x.comm.pcie._collective_tuning import plan as collective_plan
from b12x.comm.pcie._tuning import PcieQuery
from b12x.preparation import CollectiveRequirement, FrozenMapping, PreparationSession


def prepare_tuned(session, requests):
    job = session.begin(requests, autotune=True)
    allowed, state = {}, None
    try:
        while True:
            error = None
            try:
                if state is None or not state.done:
                    state = job.advance(**allowed)
            except Exception as failure:
                error = str(failure)
            values = [None] * dist.get_world_size()
            dist.all_gather_object(values, (error, state))
            if any(value[0] is not None for value in values):
                raise RuntimeError(str([value[0] for value in values]))
            states = [value[1] for value in values]
            if all(item.done for item in states):
                return job.result()
            allowed = {}
            if all(item.ready_collective_tuning is not None for item in states):
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


def prepare(
    session,
    manager,
    inp,
    out,
    arm,
    threads,
    blocks,
    config=None,
    joint_submode=None,
    joint_active_blocks=16,
    joint_max_blocks=64,
):
    if config is not None and arm.endswith("-tuned"):
        raise ValueError("family tuning arms do not accept pinned configurations")
    if config is not None and arm.startswith("joint-"):
        family = (
            "joint_latency" if arm.startswith("joint-latency") else "joint_bandwidth"
        )
        pinned = CollectiveConfig(**config)
        if pinned.implementation != family:
            raise ValueError(f"{arm} requires implementation={family!r}")
        threads = pinned.threads
        joint_max_blocks = pinned.blocks
        joint_submode = config.get("submode", joint_submode)
        joint_active_blocks = config.get("active_blocks", joint_active_blocks)
    elif config is not None and arm == "twoshot-pull":
        pinned = CollectiveConfig(**config)
        if pinned.implementation != "twoshot_pull":
            raise ValueError("twoshot-pull requires implementation='twoshot_pull'")
        if config.get("submode", "") or config.get("active_blocks", 0):
            raise ValueError("standalone two-shot pull has no joint profile fields")
    if arm.endswith("-tuned"):
        family = {
            "joint-latency-tuned": "joint_latency",
            "joint-bandwidth-tuned": "joint_bandwidth",
            "oneshot-tuned": "remote_push_full",
            "rsag-tuned": "remote_push_rsag",
            "pull-tuned": "pull",
            "twoshot-tuned": "twoshot",
            "twoshot-pull-tuned": "twoshot",
        }[arm]
        target = manager
        if arm == "twoshot-pull-tuned":
            runtime = manager._bf16_twoshot
            if runtime is None or not runtime.accepts(inp):
                raise ValueError("two-shot does not support this prepared shape")
            native = twoshot.query_from_runtime(
                runtime,
                surface="PCIeTwoShotBF16.all_reduce",
                call={"inp": inp, "out": out, "transport": "pull"},
            )
        else:
            declared = manager.plan(inp, out=out, joint_only=arm.startswith("joint-"))
            if family not in declared.query.alternatives:
                raise ValueError(f"{arm} is unavailable for this prepared shape")
            native = PcieQuery(**declared.query.alternatives[family])
            runtime = (
                manager._bf16_twoshot
                if family == "twoshot"
                else manager._runtime._prepared_channel_for_stream(None, None)
            )
        plan = collective_plan({family: (native, runtime)}, default=family)
    elif arm.startswith("joint-"):
        family = (
            "joint_latency" if arm.startswith("joint-latency") else "joint_bandwidth"
        )
        target = manager._runtime._prepared_channel_for_stream(None, None)
        query = oneshot.query_from_runtime(
            target,
            surface="OneshotAllReduce.all_reduce",
            call={"inp": inp, "out": out},
        )
        call = {
            **query.call,
            "transport": family,
            "threads": threads,
            "blocks": joint_max_blocks,
            "submode": joint_submode
            if joint_submode is not None
            else ("full" if family == "joint_latency" else "payload_rsag"),
            "active_blocks": joint_active_blocks,
        }
        query = replace(query, call=FrozenMapping(call))
        plan = oneshot.plan(query, runtime=target)
    elif config is not None and arm != "twoshot-pull":
        plan = manager.plan(inp, out=out, override=CollectiveConfig(**config))
        target = manager
    elif arm in ("auto", "tuned"):
        plan = manager.plan(inp, out=out)
        target = manager
    elif arm in ("twoshot", "twoshot-pull"):
        target = manager._bf16_twoshot
        if target is None or not target.accepts(inp):
            raise ValueError("two-shot does not support this prepared shape")
        query = twoshot.query_from_runtime(
            target,
            surface="PCIeTwoShotBF16.all_reduce",
            call={
                "inp": inp,
                "out": out,
                **({"transport": "pull"} if arm == "twoshot-pull" else {}),
                **(
                    {"threads": pinned.threads, "block_limit": pinned.blocks}
                    if config is not None
                    else {}
                ),
            },
        )
        plan = twoshot.plan(query, runtime=target)
    else:
        target = manager._runtime._prepared_channel_for_stream(None, None)
        query = oneshot.query_from_runtime(
            target, surface="OneshotAllReduce.all_reduce", call={"inp": inp, "out": out}
        )
        transport = {
            "pull": "pull",
            "ll": "tp2_remote_push",
            "simple": "tp2_remote_push_simple",
            "stream": "tp2_remote_push_stream",
            "simple-policy": "tp2_remote_push_simple",
            "generic-ll": "remote_push_ll",
            "generic-simple": "remote_push_simple",
            "oneshot": "remote_push_full",
            "rsag": "remote_push_rsag",
        }[arm]
        region = (
            "payload_ready_region_packs"
            if arm in ("oneshot", "rsag")
            else "remote_push_region_packs"
        )
        if arm != "pull" and not query.call[region]:
            raise ValueError("payload readiness requires prepared incoming storage")
        call = {**query.call, "transport": transport}
        if arm == "pull":
            call.update(
                threads=query.call["eager_threads"],
                blocks=query.call["eager_blocks"],
            )
        else:
            call.update(threads=threads, blocks=blocks)
        if arm == "simple-policy":
            packs = inp.numel() // 8
            policy_threads = (
                256 if inp.shape[-1] == 2560 and inp.shape[0] >= 16 else 128
            )
            call.update(
                threads=policy_threads,
                blocks=min(36, (packs + policy_threads - 1) // policy_threads),
            )
            if inp.shape[0] == 1:
                call.update(transport="tp2_remote_push", threads=64, blocks=36)
        query = replace(query, call=FrozenMapping(call))
        plan = oneshot.plan(query, runtime=target)

    def prime(state):
        if state.query.surface.startswith("PCIeTwoShotBF16."):
            return twoshot.prepared_call(state, payload=inp, out=out)
        return oneshot._prepare_plain_call(state, inp=inp, out=out)

    name = f"{arm}-{inp.numel()}"
    requests = (
        plan.request(
            name=name,
            collective=CollectiveRequirement(
                key=name, ranks=tuple(range(manager.world_size))
            ),
            prepare_call=prime,
        ),
    )
    if arm == "tuned" or arm.endswith("-tuned"):
        prepare_tuned(session, requests)
    else:
        session.prepare(
            requests,
            coordinator=lambda progress: progress.ready_collectives[0].key
            if progress.ready_collectives
            else None,
        )
    if target is manager._bf16_twoshot:
        prepared_call = plan.prepared.state.query.call
        return plan, lambda: target.all_reduce(
            inp,
            plan=plan,
            out=out,
            threads=prepared_call["threads"],
            block_limit=prepared_call["block_limit"],
        )
    return plan, lambda: target.all_reduce(inp, plan=plan, out=out)


def reference(operands, rank, transport, submode=None):
    world, elements = operands.shape
    if transport in ("joint_latency", "joint_bandwidth"):
        transport = {
            "full": "remote_push_full",
            "payload_rsag": "remote_push_rsag",
            "bulk_push": "bulk_push",
            "bulk_pull": "bulk_pull",
            "pull": "pull",
        }[submode]
    bulk = transport in ("bulk_push", "bulk_pull")
    if transport not in ("remote_push_rsag", "twoshot", "bulk_push", "bulk_pull"):
        total = operands[rank].float()
        for peer in range(1, world):
            total += operands[(rank + peer) % world].float()
        return total.to(operands.dtype)
    packs = elements // 8
    part = -(-packs // world) if transport == "twoshot" or bulk else packs // world
    result = torch.empty(elements, dtype=operands.dtype)
    for owner in range(world):
        begin = min(elements, owner * part * 8)
        end = elements if owner == world - 1 else min(elements, (owner + 1) * part * 8)
        total = operands[owner, begin:end].float()
        if bulk:
            total = torch.zeros_like(total) + total
        for peer in range(1, world):
            total += operands[(owner + peer) % world, begin:end].float()
        result[begin:end] = total.to(operands.dtype)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bytes",
        default="10240,20480,40960,65536,81920,163840,327680,655360,1048576,2097152",
    )
    parser.add_argument("--arms", default="pull,oneshot,rsag,twoshot")
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--capacity-bytes", type=int, default=0)
    parser.add_argument("--tuning-cache-dir", type=Path)
    parser.add_argument("--family-configs", type=Path)
    parser.add_argument("--sample-budget-ms", type=float, default=0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument(
        "--joint-submode",
        choices=("full", "payload_rsag", "bulk_push", "bulk_pull", "pull"),
    )
    parser.add_argument("--joint-active-blocks", type=int, default=16)
    parser.add_argument("--joint-max-blocks", type=int, default=64)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--chain", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=77231)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    family_configs = (
        None
        if args.family_configs is None
        else json.loads(args.family_configs.read_text())
    )
    if not math.isfinite(args.sample_budget_ms) or args.sample_budget_ms < 0:
        parser.error("sample budget must be finite and nonnegative")
    dtype = getattr(torch, args.dtype)
    sizes = [int(value) for value in args.bytes.split(",")]
    capacity = args.capacity_bytes or max(sizes)
    if capacity < max(sizes) or capacity % 16:
        parser.error("capacity must cover every message and be a multiple of 16 bytes")
    arms = args.arms.split(",")
    if args.hidden_size < 0 or (
        args.hidden_size and any(size % (2 * args.hidden_size) for size in sizes)
    ):
        parser.error("message sizes must contain complete hidden rows")
    if any(size <= 0 or size % 16 for size in sizes):
        parser.error("message sizes must be positive multiples of 16 bytes")
    if any(
        arm
        not in (
            "auto",
            "tuned",
            "pull",
            "ll",
            "simple",
            "stream",
            "simple-policy",
            "generic-ll",
            "generic-simple",
            "oneshot",
            "rsag",
            "twoshot",
            "twoshot-pull",
            "portable-rsag",
            "portable-best",
            "joint-latency",
            "joint-bandwidth",
            "joint-latency-control",
            "joint-bandwidth-control",
            "joint-latency-tuned",
            "joint-bandwidth-tuned",
            "oneshot-tuned",
            "rsag-tuned",
            "pull-tuned",
            "twoshot-tuned",
            "twoshot-pull-tuned",
        )
        for arm in arms
    ):
        parser.error("unknown comparison arm")
    if (
        any(arm.startswith("portable-") or arm.endswith("-control") for arm in arms)
        and family_configs is None
    ):
        parser.error("pinned comparison arms require explicit family configurations")
    if min(args.samples, args.replays, args.chain, args.warmup, args.cpu_threads) <= 0:
        parser.error("measurement counts and CPU threads must be positive")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    import b12x
    import flashinfer

    print(
        f"rank={rank} flashinfer={flashinfer.__file__} b12x={b12x.__file__}", flush=True
    )
    dist.init_process_group("gloo", timeout=timedelta(minutes=3))
    world = dist.get_world_size()
    if world not in (2, 3, 4, 6, 8, 10):
        raise ValueError("comparison requires a supported all-peer rank group")
    if any(arm in ("ll", "simple", "stream", "simple-policy") for arm in arms) and (
        world != 2
    ):
        raise ValueError("explicit TP2 transports require two ranks")
    manager = PCIeAllReduce.from_exchange_group(
        joint=any(arm.startswith("joint-") for arm in arms),
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=capacity,
        eager_buffer_bytes=capacity,
    )
    manager.prepare_channels(("benchmark:allreduce",))
    session = PreparationSession(device=device, autotune=False, compile_workers=1)
    tuned_session = (
        PreparationSession(
            device=device,
            autotune=True,
            cache_dir=args.tuning_cache_dir,
            compile_workers=1,
            collective_barrier=lambda key, ranks: dist.barrier(),
        )
        if any(arm == "tuned" or arm.endswith("-tuned") for arm in arms)
        else None
    )
    records = []
    graphs = {}
    try:
        for size in sizes:
            inp = torch.empty(
                (size // (2 * args.hidden_size), args.hidden_size)
                if args.hidden_size
                else (size // 2,),
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(inp)
            inp.fill_(rank + 1)
            calls, plans, graphs = {}, {}, {}
            preparation_seconds = {}
            for arm in arms:
                if rank == 0:
                    print(f"preparing bytes={size} arm={arm}", flush=True)
                owner = (
                    tuned_session
                    if arm == "tuned" or arm.endswith("-tuned")
                    else session
                )
                preparation_started = time.perf_counter()
                plans[arm], calls[arm] = prepare(
                    owner,
                    manager,
                    inp,
                    out,
                    arm,
                    args.threads,
                    args.blocks,
                    None if family_configs is None else family_configs[str(size)][arm],
                    args.joint_submode,
                    args.joint_active_blocks,
                    args.joint_max_blocks,
                )
                preparation_seconds[arm] = time.perf_counter() - preparation_started
                graph = torch.cuda.CUDAGraph()
                dist.barrier()
                with owner.capture(), manager.capture(), torch.cuda.graph(graph):
                    for _ in range(args.chain):
                        calls[arm]()
                graphs[arm] = graph
            for step in range(5):
                generator = torch.Generator().manual_seed(args.seed + step)
                operands = torch.randn((world, inp.numel()), generator=generator).to(
                    dtype
                )
                if step == 1:
                    operands.zero_()
                elif step == 2:
                    operands[1] = -operands[0]
                elif step == 3:
                    operands.mul_(2**-100)
                elif step == 4:
                    operands.mul_(128)
                inp.copy_(operands[rank].view_as(inp))
                for arm, graph in graphs.items():
                    gc.collect()
                    out.fill_(float("nan"))
                    torch.cuda.synchronize()
                    dist.barrier()
                    before = torch.cuda.memory_stats()
                    graph.replay()
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
                            arm,
                            counter,
                            before[counter],
                            after[counter],
                        )
                    query = plans[arm].prepared.state.query
                    call = query.call
                    transport = (
                        "twoshot"
                        if query.surface.startswith("PCIeTwoShotBF16.")
                        else call.get("transport", "twoshot")
                    )
                    expected = reference(operands, rank, transport, call.get("submode"))
                    torch.testing.assert_close(
                        out.cpu().flatten(), expected, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        inp.cpu().flatten(), operands[rank], rtol=0, atol=0
                    )
            if args.correctness_only:
                records.append(
                    {
                        "bytes": size,
                        "world_size": world,
                        "arms": arms,
                        "correctness": "passed",
                    }
                )
                if rank == 0:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps({"results": records}, indent=2) + "\n"
                    )
                    print(json.dumps(records[-1]), flush=True)
                for graph in graphs.values():
                    graph.reset()
                graphs = {}
                continue
            for graph in graphs.values():
                for _ in range(args.warmup):
                    graph.replay()
            torch.cuda.synchronize()
            replay_counts = dict.fromkeys(arms, args.replays)
            if args.sample_budget_ms:
                for arm in arms:
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    durations = [None] * world
                    dist.all_gather_object(durations, start.elapsed_time(end))
                    replay_counts[arm] = min(
                        args.replays,
                        max(1, math.ceil(args.sample_budget_ms / max(durations))),
                    )
            samples = {arm: [] for arm in arms}
            for sample in range(args.samples):
                for arm in arms if sample % 2 == 0 else reversed(arms):
                    dist.barrier()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(replay_counts[arm]):
                        graphs[arm].replay()
                    end.record()
                    end.synchronize()
                    samples[arm].append(
                        start.elapsed_time(end)
                        * 1000
                        / (replay_counts[arm] * args.chain)
                    )
            rank_samples = [None] * world
            dist.all_gather_object(rank_samples, samples)
            rank_preparation_seconds = [None] * world
            dist.all_gather_object(rank_preparation_seconds, preparation_seconds)
            record = {
                "bytes": size,
                "world_size": world,
                "replay_counts": replay_counts,
                "surfaces": {
                    arm: dict(
                        surface=plans[arm].prepared.state.query.surface,
                        call=dict(plans[arm].prepared.state.query.call),
                        selection_source=plans[arm].selection.source,
                    )
                    for arm in arms
                },
                "rank_samples_us": rank_samples,
                "rank_preparation_seconds": rank_preparation_seconds,
                "max_rank_preparation_seconds": {
                    arm: max(values[arm] for values in rank_preparation_seconds)
                    for arm in arms
                },
                "max_rank_median_us": {
                    arm: max(median(values[arm]) for values in rank_samples)
                    for arm in arms
                },
            }
            records.append(record)
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps(
                        {
                            "arguments": {
                                **vars(args),
                                "output": str(args.output),
                                "family_configs": str(args.family_configs)
                                if args.family_configs is not None
                                else None,
                                "tuning_cache_dir": str(args.tuning_cache_dir)
                                if args.tuning_cache_dir is not None
                                else None,
                            },
                            "results": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(
                    json.dumps({"bytes": size, **record["max_rank_median_us"]}),
                    flush=True,
                )
            for graph in graphs.values():
                graph.reset()
            graphs = {}
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
    finally:
        for graph in graphs.values():
            graph.reset()
        torch.cuda.synchronize()
        if tuned_session is not None:
            tuned_session.close()
        session.close()
        manager.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc(file=sys.__stderr__)
        raise
