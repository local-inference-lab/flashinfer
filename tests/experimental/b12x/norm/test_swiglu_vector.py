"""Vectorized SwiGLU is bitwise identical to the scalar kernel."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from b12x.norm import hyperconnection as hc
from b12x.norm.hyperconnection import _tuning
from b12x.preparation import PreparationSession, PreparedCall


def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda")


def prepare(device, rows, width, vector, **options):
    invocation = {"operation": "swiglu", **options}
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=rows, hidden_size=width),
        invocation=invocation,
    )
    default = _tuning.TUNING.default_config(plan.query, None)
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=rows, hidden_size=width),
        invocation=invocation,
        override=replace(default, pointwise_vector=vector),
    )
    return plan


def launchers(device, rows, width, **options):
    """Prepared launchers for the scalar and vector configurations."""
    plans = {v: prepare(device, rows, width, v, **options) for v in (1, 8)}
    states = {}
    gate_up = torch.empty(rows, 2 * width, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
    requests = tuple(
        plan.request(
            name=f"swiglu-v{vector}",
            prepare_call=lambda state, vector=vector: (
                states.setdefault(vector, state)
                and PreparedCall(run=lambda: state.launch(gate_up, gate_up, out))
            ),
        )
        for vector, plan in plans.items()
    )
    return plans, states, requests


def random_bits(shape, device, generator):
    bits = torch.randint(
        -(2**15), 2**15, shape, dtype=torch.int16, device=device, generator=generator
    )
    return bits.view(torch.bfloat16)


def bits(tensor):
    return tensor.view(torch.int16)


@pytest.mark.parametrize(
    "rows,width",
    [(1, 8), (7, 64), (33, 2048), (256, 768), (4096, 2048), (4, 12288), (33, 12288)],
)
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"round_silu": True},
        {"limit": 7.0},
        {"limit": 7.0, "clamp_after_silu": True},
        {"limit": 7.0, "swiglu_alpha": 1.702, "swiglu_beta": 1.0},
    ],
)
def test_vector_matches_scalar_bitwise(rows, width, options):
    device = require_cuda()
    generator = torch.Generator(device=device).manual_seed(rows * 31 + width)
    plans, states, requests = launchers(device, rows, width, **options)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(requests)
        for kind in ("normal", "wide", "bits"):
            if kind == "bits":
                source = random_bits((rows, 2 * width), device, generator)
            else:
                scale = 1.0 if kind == "normal" else 40.0
                source = (
                    torch.randn(rows, 2 * width, device=device, generator=generator)
                    * scale
                ).bfloat16()
            results = {}
            for vector, state in states.items():
                out = torch.full(
                    (rows, width), -1.0, dtype=torch.bfloat16, device=device
                )
                state.launch(source, source, out)
                torch.cuda.synchronize()
                results[vector] = out
            assert torch.equal(bits(results[1]), bits(results[8])), kind


def test_inactive_rows_and_capacity_are_untouched():
    device = require_cuda()
    capacity, live, width = 4096, 1000, 2048
    plans, states, requests = launchers(device, capacity, width)
    generator = torch.Generator(device=device).manual_seed(11)
    source = random_bits((capacity, 2 * width), device, generator)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(requests)
        outputs = {}
        for vector, state in states.items():
            out = torch.full(
                (capacity, width), 0x5A5A, dtype=torch.int16, device=device
            ).view(torch.bfloat16)
            state.launch(source[:live], source[:live], out[:live])
            torch.cuda.synchronize()
            outputs[vector] = out
        assert torch.equal(bits(outputs[1]), bits(outputs[8]))
        assert bool((bits(outputs[8][live:]) == 0x5A5A).all())


def test_misaligned_storage_is_rejected_by_the_vector_launcher():
    device = require_cuda()
    rows, width = 4, 64
    plans, states, requests = launchers(device, rows, width)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(requests)
        base = torch.zeros(rows * 2 * width + 8, dtype=torch.bfloat16, device=device)
        source = base[1 : 1 + rows * 2 * width].view(rows, 2 * width)
        out = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        with pytest.raises(ValueError, match="aligned"):
            states[8].launch(source, source, out)
        states[1].launch(source, source, out)


def test_vector_replays_in_a_graph_over_poisoned_outputs():
    device = require_cuda()
    rows, width = 4096, 2048
    plan = prepare(device, rows, width, 8)
    source = torch.randn(rows, 2 * width, device=device).bfloat16()
    out = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="swiglu-graph",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.launch(source, source, out)
                    ),
                ),
            )
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            hc.run_swiglu(source, limit=float("inf"), out=out, plan=plan)
        expected = None
        for value in (0.0, float("nan")):
            out.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            if expected is None:
                expected = out.clone()
            assert torch.equal(bits(out), bits(expected))
        gate, up = source.float().chunk(2, -1)
        reference = (torch.nn.functional.silu(gate) * up).bfloat16()
        torch.testing.assert_close(out, reference, rtol=0.01, atol=1e-6)


def test_width_not_divisible_by_eight_runs_scalar():
    device = require_cuda()
    rows, width = 9, 12
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=rows, hidden_size=width),
        invocation={"operation": "swiglu"},
    )
    assert _tuning.TUNING.default_config(plan.query, None).pointwise_vector == 1


def test_race_offers_scalar_and_vector_and_keeps_either_result_exact():
    device = require_cuda()
    rows, width = 4096, 12288
    options = {"round_silu": True}
    caps = hc.Caps(device=device, max_tokens=rows, hidden_size=width)
    plan = hc.plan(caps, invocation={"operation": "swiglu", **options})
    scalar_plan = prepare(device, rows, width, 1, **options)
    generator = torch.Generator(device=device).manual_seed(5)
    source = random_bits((rows, 2 * width), device, generator)
    out = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
    scalar_out = torch.empty_like(out)

    def request(name, target, destination):
        def call(state):
            return PreparedCall(
                run=lambda: state.launch(source, source, destination),
                output=destination,
                produce=lambda: None,
                owners=(source, destination),
            )

        return target.request(name=name, prepare_call=call, benchmark_call=call)

    with PreparationSession(device=device, autotune=True, compile_workers=2) as session:
        session.prepare(
            (
                request("swiglu-race", plan, out),
                request("swiglu-scalar", scalar_plan, scalar_out),
            )
        )
        selected = plan.selection.config.pointwise_vector
        assert selected in (1, 8)
        print("selected pointwise_vector", selected)
        assert scalar_plan.selection.config.pointwise_vector == 1
        out.fill_(0.0)
        scalar_out.fill_(1.0)
        plan.prepared.state.launch(source, source, out)
        scalar_plan.prepared.state.launch(source, source, scalar_out)
        torch.cuda.synchronize()
        assert torch.equal(bits(out), bits(scalar_out))


def test_declared_misalignment_is_planned_without_the_vector_candidate():
    device = require_cuda()
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=4, hidden_size=64),
        invocation={"operation": "swiglu", "left_alignment": 2},
    )
    space = _tuning.TUNING.parameter_space(plan.query, None)
    assert {choice["pointwise_vector"] for choice in space.configurations()} == {1}
