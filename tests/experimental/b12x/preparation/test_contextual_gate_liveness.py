"""Large collective races must drain restored inputs between gated candidates."""

import subprocess
import sys
import textwrap

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


_RACE = textwrap.dedent(
    """
    import torch
    from b12x.preparation import PreparedCall
    from b12x.preparation._measurement import (
        _consume, measure_race_steps, prepare_race_steps,
    )

    device = torch.device("cuda", 0)
    targets = tuple(torch.empty((8,), device=device) for _ in range(27))
    saved = tuple(
        tuple(torch.full_like(target, value) for target in targets)
        for value in (1.0, 2.0)
    )
    scratch = tuple(torch.empty((8,), device=device) for _ in range(4))
    output = torch.empty_like(targets[0])
    output.copy_(targets[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output.copy_(targets[0])

    def reset():
        for tensor in scratch:
            tensor.fill_(float("nan"))

    def producer(values):
        for target, source in zip(targets, values, strict=True):
            target.copy_(source)

    producers = tuple(lambda values=values: producer(values) for values in saved)
    call = PreparedCall(
        run=graph.replay, output=output, reset=reset, produce=producers[0],
        benchmark_producers=producers, owners=(graph, targets, saved, scratch),
    )
    calls = (call,) * 128
    race = _consume(prepare_race_steps(
        calls, device_ordinal=0, samples=2, primed=True,
        cache_policy="cold", eviction_after_produce=True,
    ))
    launches, samples = [], []

    def launch_exchange(indices, turn, repetition):
        assert (turn, repetition) == (0, 0)
        assert len(indices) == 1
        launches.append(indices[0])
        if False:
            yield

    def sample_exchange(indices, turn, repetition, groups):
        assert (turn, repetition) == (0, 0)
        assert len(indices) == len(groups) == 1
        samples.append(indices[0])
        if False:
            yield
        return groups

    try:
        measured = _consume(measure_race_steps(
            race, device_ordinal=0, rounds=1, adaptive_repeats=False,
            launch_exchange=launch_exchange, sample_exchange=sample_exchange,
        ))
        assert len(measured.latencies_us) == 128
        assert all(value > 0 for value in measured.latencies_us)
        assert launches == samples == list(range(128))
        torch.cuda.synchronize(device)
        assert torch.equal(output, saved[1][0])
        print("collective race drained 128 restore-heavy candidates", flush=True)
    finally:
        race.close()
        graph.reset()
    """
)


def test_collective_gate_drains_restore_heavy_candidates():
    try:
        result = subprocess.run(
            [sys.executable, "-c", _RACE], capture_output=True, text=True, timeout=45
        )
    except subprocess.TimeoutExpired:
        pytest.fail("collective race blocked while submitting restored inputs")
    assert result.returncode == 0, result.stderr[-2000:]
    assert "drained 128 restore-heavy candidates" in result.stdout
