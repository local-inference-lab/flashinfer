"""Contextual graph racing can select a legal three-plan interaction."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import torch

from b12x.preparation import (
    ContextCandidate,
    FinalistPolicy,
    MemoryRequirements,
    Plan,
    PreparationSession,
    PreparedCall,
)
from ..conftest import require_b12x
from .test_defaults import Config, Query, contract


def test_three_plan_context_race_retains_independent_winners(tmp_path):
    device = require_b12x()
    names = ("projection", "normalization", "query")
    original, alternatives, requests = {}, {}, []
    for axis, name in enumerate(names):
        tuning = contract(default=1, values=(1, 2))
        plan = Plan(
            contract=tuning,
            query=Query(axis + 1),
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: SimpleNamespace(
                width=selection.config.width
            ),
        )
        original[name] = plan

        def benchmark(state):
            return PreparedCall(
                run=lambda: torch.cuda._sleep(100_000 if state.width == 1 else 180_000),
                produce=lambda: None,
            )

        requests.append(
            plan.request(
                name=name,
                prepare_call=lambda state: PreparedCall(run=lambda: None),
                benchmark_call=benchmark,
                finalist_policy=FinalistPolicy(family_fields=("width",), max_total=2),
            )
        )

    with PreparationSession(
        device=device,
        cache_dir=tmp_path,
        cache_policy="warm",
        compile_workers=0,
        race_batch=2,
        rounds=3,
        samples=2,
    ) as session:
        standalone = session.prepare(tuple(requests))
        assert all(standalone.selections[name].config.width == 1 for name in names)
        assert all(
            {item.selection.config.width for item in standalone.finalists[name]}
            == {1, 2}
            for name in names
        )
        cached_keys = frozenset(session._selection_cache().records)
        for name in names:
            alternatives[name] = {}
            for width in (1, 2):
                plan = replace(original[name], override=Config(width))
                assert original[name].same_declaration_except_override(plan)
                alternatives[name][width] = plan
                session.prepare(
                    (
                        plan.request(
                            name=f"{name}.{width}",
                            prepare_call=lambda state: PreparedCall(run=lambda: None),
                        ),
                    )
                )

        candidates = []
        for widths in product((1, 2), repeat=3):
            name = ".".join(map(str, widths))
            plans = tuple(
                alternatives[axis][width]
                for axis, width in zip(names, widths, strict=True)
            )
            # Every isolated width-2 kernel is slower. Only the complete
            # three-way schedule removes the shared producer delay.
            delay = (
                350_000
                if widths == (2, 2, 2)
                else 900_000 + 100_000 * sum(width == 2 for width in widths)
            )
            source = torch.ones((1,), device=device)
            encoded = torch.ones_like(source)
            scratch = torch.empty_like(source)
            output = torch.empty_like(source)
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                torch.cuda._sleep(delay)
                scratch.copy_(encoded)
                output.copy_(scratch)

            def make_call(
                graph=graph,
                source=source,
                encoded=encoded,
                scratch=scratch,
                output=output,
            ):
                def reset():
                    encoded.fill_(float("nan"))
                    scratch.fill_(float("nan"))
                    output.fill_(float("nan"))

                def produce():
                    source.fill_(1)
                    encoded.copy_(source)

                return PreparedCall(
                    run=graph.replay,
                    reset=reset,
                    produce=produce,
                    validate=lambda: torch.testing.assert_close(
                        output, source, rtol=0, atol=0
                    ),
                    owners=(graph, source, encoded, scratch, output),
                    close=graph.reset,
                )

            candidates.append(
                ContextCandidate(name=name, plans=plans, call=make_call())
            )
        sample = candidates[0].call
        sample.reset()
        sample.produce()
        torch.cuda.synchronize(device)
        before = torch.cuda.memory_stats(device)
        sample.run()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        for key in (
            "allocation.all.allocated",
            "allocation.all.freed",
            "segment.all.allocated",
            "segment.all.freed",
        ):
            assert before[key] == after[key]
        sample.validate()
        result = session.race_contexts(
            candidates, max_candidates=8, rounds=3, samples=2
        )
        assert result.winner not in {"1.1.1", "2.1.1", "1.2.1", "1.1.2"}
        assert all(original[name].selection.config.width == 1 for name in names)
        assert frozenset(session._selection_cache().records) == cached_keys
        chosen = next(
            candidate for candidate in candidates if candidate.name == result.winner
        )
        keep = set(chosen.plans)
        session.release_many(
            plan
            for axis in names
            for plan in alternatives[axis].values()
            if plan not in keep
        )
        assert all(plan.prepared is not None for plan in keep)
        session.freeze()
