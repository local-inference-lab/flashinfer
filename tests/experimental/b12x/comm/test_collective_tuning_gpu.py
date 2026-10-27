"""Whole-group tuning, cancellation, cache reuse, and graph replay on PCIe.

Run with B12X_RUN_COLLECTIVE_TUNING_TEST=1 and 2, 4, or 10 visible SM120 GPUs.
"""

import os
from pathlib import Path

import pytest


def _drive(session, requests, *, cancel_step=None):
    import torch.distributed as dist

    job = session.begin(requests)
    allowed = {}
    state = None
    try:
        while True:
            error = None
            try:
                if state is None or not state.done:
                    state = job.advance(**allowed)
                tuning = state.ready_collective_tuning
                if (
                    tuning is not None
                    and tuning.step.startswith(cancel_step or "\0")
                    and dist.get_rank() == 0
                ):
                    session.cancel_tuning()
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


def _worker(rank, world, directory):
    import gc
    import torch
    import torch.distributed as dist
    from b12x.comm.pcie import AllReduce
    from b12x.comm.pcie import _oneshot_preparation as one, _twoshot_preparation as two
    from b12x.preparation import (
        CollectiveRequirement,
        PreparationSession,
        autotune_effort,
    )

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
    )
    manager = AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        single_channel=True,
        max_size=2 << 20,
        eager_buffer_bytes=2 << 20,
    )
    assert manager.algorithm == "oneshot"
    assert manager._bf16_twoshot is not None
    manager.prepare_channels(("tuning",))
    members = tuple(range(world))

    def declarations(
        *, dtype=torch.bfloat16, mismatch=False, width=None, shape=None, override=None
    ):
        plans, requests, operands = [], [], []
        for index in range(2):
            rows = 16 if mismatch and rank else 8
            source = torch.ones(
                shape or (rows, width or (4096 if world == 2 else 5120)),
                dtype=dtype,
                device=device,
            )
            out = torch.empty_like(source)
            plan = manager.plan(source, out=out, override=override)

            def bind(state, source=source, out=out):
                if state.query.surface.startswith("PCIeTwoShotBF16."):
                    return two.prepared_call(state, payload=source, out=out)
                return one._prepare_plain_call(state, inp=source, out=out)

            requests.append(
                plan.request(
                    name=f"layer{index}",
                    prepare_call=bind,
                    collective=CollectiveRequirement(f"layer{index}", members),
                )
            )
            plans.append(plan)
            operands.append((source, out))
        return plans, requests, operands

    def session(name, **kwargs):
        return PreparationSession(
            device=device,
            cache_dir=Path(directory) / f"rank{rank}",
            namespace={"test": name},
            compile_workers=2,
            rounds=2,
            collective_barrier=lambda key, ranks: dist.barrier(),
            **kwargs,
        )

    with autotune_effort("default"):
        for iteration in range(2):
            owner = session("reuse", cache_only=iteration == 1)
            plans, requests, operands = declarations()
            result = _drive(owner, requests)
            assert plans[0].selection.config == plans[1].selection.config
            selections = [None] * world
            dist.all_gather_object(selections, plans[0].selection.config)
            assert all(config == selections[0] for config in selections)
            assert all(
                plan.selection.source == ("tuned" if iteration == 0 else "cached")
                for plan in plans
            )
            assert (
                result.benchmarked_candidates > 0
                if iteration == 0
                else result.benchmarked_candidates == 0
            )
            graphs = []
            for plan, (source, out) in zip(plans, operands, strict=True):
                graph = torch.cuda.CUDAGraph()
                with owner.capture(), manager.capture(), torch.cuda.graph(graph):
                    manager.all_reduce(source, out=out, plan=plan)
                graphs.append(graph)
            owner.freeze()
            for step, index in enumerate((0, 1, 1, 0)):
                source, out = operands[index]
                source.fill_(rank + step + 1)
                out.fill_(float("nan"))
                original = source.clone()
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graphs[index].replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for counter in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[counter] == after[counter]
                torch.testing.assert_close(source, original, rtol=0, atol=0)
                torch.testing.assert_close(
                    out,
                    torch.full_like(out, world * (step + 1) + world * (world - 1) // 2),
                    rtol=0,
                    atol=0,
                )
            for graph in graphs:
                graph.reset()
            result.close()
            owner.close()
            # A cache miss on one rank is resolved from another rank's group record.
            if iteration == 0 and rank == 1:
                for path in (Path(directory) / f"rank{rank}").glob("**/*.json"):
                    path.unlink()
        for point in ("catalog", "samples/"):
            owner = session(f"cancel-{point}")
            plans, requests, _ = declarations()
            result = _drive(owner, requests, cancel_step=point)
            assert all(plan.selection.source == "default" for plan in plans)
            result.close()
            owner.close()
        if world == 4:
            owner = session("mismatched-shapes")
            _, requests, _ = declarations(mismatch=True)
            with pytest.raises(RuntimeError, match="disagree on signature"):
                _drive(owner, requests)
            owner.close()
        if world == 4:
            from dataclasses import replace

            owner = session("inconsistent-workload-count")
            _, requests, _ = declarations()
            factory = requests[0].benchmark_call

            def benchmark(state):
                call = factory(state)
                return (
                    replace(call, benchmark_producers=call.benchmark_producers[:2])
                    if rank == 1
                    else call
                )

            requests = [
                replace(request, benchmark_call=benchmark) for request in requests
            ]
            with pytest.raises(RuntimeError, match="disagree on producers"):
                _drive(owner, requests)
            owner.close()
        if world == 2:
            for shape in ((8, 5120), (513, 1536)):
                owner = session(f"noncanonical-{shape}")
                plans, requests, operands = declarations(shape=shape)
                result = _drive(owner, requests)
                assert all(plan.selection.source == "tuned" for plan in plans)
                assert result.benchmarked_candidates > 0
                graph = torch.cuda.CUDAGraph()
                with owner.capture(), manager.capture(), torch.cuda.graph(graph):
                    manager.all_reduce(
                        operands[0][0], out=operands[0][1], plan=plans[0]
                    )
                owner.freeze()
                for step in range(3):
                    operands[0][0].fill_(rank + step + 1)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        operands[0][1],
                        torch.full_like(operands[0][1], 2 * step + 3),
                        rtol=0,
                        atol=0,
                    )
                graph.reset()
                result.close()
                owner.close()

            from b12x.comm.pcie import CollectiveConfig

            owner = session("pinned-transitions")
            retained, calls = [], []
            for shape, config in (
                (
                    (3, 1536),
                    CollectiveConfig(
                        implementation="tp2_remote_push", threads=64, blocks=8
                    ),
                ),
                (
                    (513, 1536),
                    CollectiveConfig(
                        implementation="tp2_remote_push_simple", threads=256, blocks=16
                    ),
                ),
                (
                    (3, 1536),
                    CollectiveConfig(
                        implementation="tp2_remote_push", threads=128, blocks=36
                    ),
                ),
                (
                    (513, 1536),
                    CollectiveConfig(implementation="pull", threads=256, blocks=8),
                ),
            ):
                plans, requests, operands = declarations(shape=shape, override=config)
                retained.append(_drive(owner, requests))
                assert all(plan.selection.config == config for plan in plans)
                calls.append((plans[0], *operands[0]))
            graph = torch.cuda.CUDAGraph()
            with owner.capture(), manager.capture(), torch.cuda.graph(graph):
                for plan, source, out in calls:
                    manager.all_reduce(source, out=out, plan=plan)
            owner.freeze()
            for step in range(4):
                for _, source, out in calls:
                    source.fill_(rank + step + 1)
                    out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for _, source, out in calls:
                    torch.testing.assert_close(
                        out, torch.full_like(out, 2 * step + 3), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        source, torch.full_like(source, rank + step + 1), rtol=0, atol=0
                    )
            graph.reset()
            for result in retained:
                result.close()
            owner.close()
        owner = session("missing-cache", cache_only=True)
        _, requests, _ = declarations()
        with pytest.raises(RuntimeError, match="no completed collective selection"):
            _drive(owner, requests)
        owner.close()
    manager.close()
    dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("B12X_RUN_COLLECTIVE_TUNING_TEST") != "1",
    reason="requires peer-accessible SM120 GPUs",
)
@pytest.mark.parametrize("world", (2, 4, 10))
def test_collective_tuning(world, tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < world or torch.cuda.get_device_capability() != (
        12,
        0,
    ):
        pytest.skip("requires peer-accessible SM120 GPUs")
    mp.spawn(_worker, args=(world, str(tmp_path)), nprocs=world, join=True)
