"""Two-rank contextual races score each replay by the slower GPU rank."""

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.preparation import (
    ContextCandidate,
    MemoryRequirements,
    Plan,
    PreparationSession,
    PreparedCall,
)
from .test_defaults import Config, Query, contract


def _rank(rank, rendezvous, result_dir):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        tuning = contract(values=(1, 2))
        original = Plan(
            contract=tuning,
            query=Query(3),
            _device=device,
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(
                width=selection.config.width
            ),
        )
        plans = tuple(replace(original, override=Config(width)) for width in (1, 2))
        candidates = []
        with PreparationSession(
            device=device, autotune=False, cache_policy="warm"
        ) as session:
            session.prepare(
                tuple(
                    plan.request(
                        name=f"plan-{index}",
                        prepare_call=lambda state: PreparedCall(run=lambda: None),
                    )
                    for index, plan in enumerate(plans)
                )
            )
            for index, plan in enumerate(plans):
                source = torch.ones((1,), device=device)
                output = torch.empty_like(source)
                graph = torch.cuda.CUDAGraph()
                # Rank zero prefers A locally; rank one makes A the slowest
                # member of the TP2 step. B has equal work on both ranks.
                delay = (100_000 if rank == 0 else 900_000) if index == 0 else 300_000
                with session.capture(), torch.cuda.graph(graph):
                    torch.cuda._sleep(delay)
                    output.copy_(source)
                candidates.append(
                    ContextCandidate(
                        name=("a", "b")[index],
                        plans=(plan,),
                        call=PreparedCall(
                            run=graph.replay,
                            reset=lambda output=output: output.fill_(float("nan")),
                            produce=lambda source=source: source.fill_(1),
                            validate=lambda source=source,
                            output=output: torch.testing.assert_close(
                                output, source, rtol=0, atol=0
                            ),
                            owners=(graph, source, output),
                            close=graph.reset,
                        ),
                    )
                )

            def exchange(step, payload):
                peers = [None, None]
                dist.all_gather_object(peers, dict(payload))
                return peers

            for field in ("rounds", "samples", "producer_counts"):
                mismatched = candidates
                options = {"rounds": 3, "samples": 2}
                if field == "producer_counts":
                    mismatched = [
                        replace(
                            candidate,
                            call=replace(
                                candidate.call,
                                benchmark_producers=(candidate.call.produce,)
                                * (rank + 1),
                            ),
                        )
                        for candidate in candidates
                    ]
                else:
                    options[field] += rank
                with pytest.raises(ValueError, match=field):
                    session.race_contexts(
                        mismatched,
                        collective_exchange=exchange,
                        collective_ranks=(0, 1),
                        **options,
                    )

            result = session.race_contexts(
                candidates,
                collective_exchange=exchange,
                collective_ranks=(0, 1),
                max_candidates=2,
                rounds=3,
                samples=2,
            )
            Path(result_dir, f"rank{rank}.json").write_text(
                json.dumps(
                    {
                        "winner": result.winner,
                        "latencies_us": dict(result.latencies_us),
                    }
                )
            )
    finally:
        dist.destroy_process_group()


def test_tp2_context_uses_max_rank_samples(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required for TP2 contextual replay")
    rendezvous = tmp_path / "rendezvous"
    processes = mp.spawn(
        _rank, args=(str(rendezvous), str(tmp_path)), nprocs=2, join=False
    )
    try:
        deadline = time.monotonic() + 60
        while not processes.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("context ranks did not finish their agreed replay schedule")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
        for process in processes.processes:
            process.join(timeout=5)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    assert ranks[0] == ranks[1]
    assert ranks[0]["winner"] == "b"
    assert ranks[0]["latencies_us"]["a"] > ranks[0]["latencies_us"]["b"]
