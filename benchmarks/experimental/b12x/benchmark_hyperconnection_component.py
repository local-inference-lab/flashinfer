"""Measure prepared HC mixing with both projections and FP32 normalization gains.

Synthetic inputs reproduce the Qwen3.8 geometry and BF16 boundaries. The block
output is independent: decoder attention/MoE work is outside this component.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch
import flashinfer
import b12x
from b12x.gemm import bf16_gemv, hyperconnection_gate as gate
from b12x.norm import hyperconnection as hc
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.benchmark import transaction_samples
from benchmarks.experimental.b12x.common import capture_cuda_graph


def hc_benchmark(prepared, tensors, outputs):
    generator = torch.Generator(device="cuda").manual_seed(20260907)
    storage = tuple(
        torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        for t in tensors
    )
    inputs = storage[:-outputs]
    seeds = tuple(
        torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=generator)
        * 0.125
        for t in inputs
    )

    def produce():
        for target, source in zip(inputs, seeds, strict=True):
            target.copy_(source)

    return PreparedCall(
        run=lambda: prepared.launch(*storage),
        produce=produce,
        output=storage[-outputs:],
        owners=(*storage, *seeds),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 8, 12, 16, 20, 24, 32, 64, 128, 512, 2048],
    )
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    print(
        json.dumps(
            {
                "flashinfer": flashinfer.__file__,
                "b12x": b12x.__file__,
                "seed": args.seed,
                "samples": args.samples,
                "cache": "warm L2",
                "timing": "captured graph replay",
                "invocations_per_replay": 32,
                "rounds": 9,
                "autotune": args.autotune,
            }
        ),
        flush=True,
    )
    torch.manual_seed(args.seed)
    records = []
    for rows in args.rows:

        def rand(*shape, scale=1.0):
            return (torch.randn(*shape, device="cuda") * scale).bfloat16()

        state, block = rand(rows, 10240, scale=0.3), rand(rows, 2560, scale=0.2)
        gain = 1 + torch.randn(10240, device="cuda") * 0.03
        down_weight, up_weight = (
            rand(336, 10240, scale=0.01),
            rand(10240, 320, scale=0.03),
        )
        norm, next_norm, combined = (torch.empty_like(state) for _ in range(3))
        merged = torch.empty(rows, 336, device="cuda", dtype=torch.bfloat16)
        down, inject = merged[:, :320], merged[:, 320:324]
        bottle = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(block)
        workspace = torch.empty_like(state)
        caps = hc.Caps(
            device="cuda", max_tokens=rows, hidden_size=2560, streams=4, lowrank=320
        )
        plans = {}
        requests = []
        for name, tensors, options in (
            (
                "grouped_rmsnorm",
                (state, gain, norm),
                {"zero_centered": False, "weight_dtype": "float32"},
            ),
            ("scaled_silu", (down, bottle), {}),
            (
                "combine_norm",
                (state, block, inject, gain, combined, next_norm),
                {
                    "zero_centered": False,
                    "weight_dtype": "float32",
                    "output_mode": "provided",
                },
            ),
        ):
            plans[name] = p = hc.plan(caps, invocation={"operation": name, **options})
            requests.append(
                p.request(
                    name=name,
                    benchmark_call=lambda s,
                    t=tensors,
                    n=2 if name == "combine_norm" else 1: hc_benchmark(s, t, n),
                    prepare_call=lambda s, t=tensors: PreparedCall(
                        run=lambda: s.launch(*t)
                    ),
                )
            )
        dp = bf16_gemv.plan(bf16_gemv.query_from_call(norm, down_weight, out=merged))
        gp = gate.plan(gate.Query(max_rows=rows, hidden_size=2560, streams=4, rank=320))
        tuning_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
        tune_norm = torch.empty_like(norm)
        tune_bottle = torch.empty_like(bottle)
        tune_merged = torch.empty_like(merged)
        tune_output = torch.empty_like(output)
        tune_workspace = torch.empty_like(workspace)
        norm_seed = (
            torch.randn(
                norm.shape, device="cuda", dtype=norm.dtype, generator=tuning_generator
            )
            * 0.125
        )
        bottle_seed = (
            torch.randn(
                bottle.shape,
                device="cuda",
                dtype=bottle.dtype,
                generator=tuning_generator,
            )
            * 0.125
        )

        def produce_down():
            tune_norm.copy_(norm_seed)

        def produce_gate():
            tune_norm.copy_(norm_seed)
            tune_bottle.copy_(bottle_seed)

        requests.extend(
            (
                dp.request(
                    name="down",
                    benchmark_call=lambda s: PreparedCall(
                        run=lambda: s.run(tune_norm, down_weight, out=tune_merged),
                        produce=produce_down,
                        output=tune_merged,
                        owners=(tune_norm, down_weight, tune_merged, norm_seed),
                    ),
                    prepare_call=lambda s: PreparedCall(
                        run=lambda: s.run(norm, down_weight, out=merged)
                    ),
                ),
                gp.request(
                    name="gate",
                    benchmark_call=lambda s: PreparedCall(
                        run=lambda: s.run(
                            tune_bottle,
                            up_weight,
                            tune_norm,
                            tune_output,
                            tune_workspace,
                        ),
                        produce=produce_gate,
                        output=tune_output,
                        owners=(
                            tune_bottle,
                            up_weight,
                            tune_norm,
                            tune_output,
                            tune_workspace,
                            norm_seed,
                            bottle_seed,
                        ),
                    ),
                    prepare_call=lambda s: PreparedCall(
                        run=lambda: s.run(bottle, up_weight, norm, output, workspace)
                    ),
                ),
            )
        )
        with PreparationSession(
            device="cuda", autotune=args.autotune, compile_workers=0
        ) as session:
            session.prepare(tuple(requests))
            binding = {
                name: hc.bind(
                    plans[name],
                    tokens=rows,
                    normalized=norm,
                    bottleneck=bottle,
                    block_input=output,
                )
                for name in ("grouped_rmsnorm", "scaled_silu")
            }

            def normalize():
                hc.run_grouped_rmsnorm(
                    state,
                    gain,
                    eps=1e-6,
                    zero_centered=False,
                    binding=binding["grouped_rmsnorm"],
                )

            def project():
                bf16_gemv.mm(norm, down_weight, out=merged, plan=dp)

            def activate():
                hc.run_scaled_silu(down, binding=binding["scaled_silu"])

            def project_gate():
                gate.run(
                    bottle, up_weight, norm, out=output, workspace=workspace, plan=gp
                )

            def combine():
                hc.run_combine_norm(
                    state,
                    block,
                    inject,
                    gain,
                    eps=1e-6,
                    zero_centered=False,
                    plan=plans["combine_norm"],
                    out=(combined, next_norm),
                )

            def full():
                normalize()
                project()
                activate()
                project_gate()
                combine()

            full()

            def check():
                torch.testing.assert_close(
                    norm,
                    hc.reference.grouped_rmsnorm(
                        state, gain, streams=4, eps=1e-6, zero_centered=False
                    ),
                    rtol=0,
                    atol=0.02,
                )
                torch.testing.assert_close(
                    merged,
                    torch.nn.functional.linear(norm, down_weight),
                    rtol=0.008,
                    atol=0.002,
                )
                torch.testing.assert_close(
                    bottle,
                    hc.reference.scaled_silu(down, streams=4),
                    rtol=0,
                    atol=0.008,
                )
                torch.testing.assert_close(
                    output,
                    gate.reference(bottle, up_weight, norm, streams=4),
                    rtol=0.008,
                    atol=0.002,
                )
                ref = hc.reference.combine(state, block, inject, streams=4)
                torch.testing.assert_close(combined, ref, rtol=0.01, atol=0.016)
                torch.testing.assert_close(
                    next_norm,
                    hc.reference.grouped_rmsnorm(
                        ref, gain, streams=4, eps=1e-6, zero_centered=False
                    ),
                    rtol=0.01,
                    atol=0.032,
                )

            check()
            timings = {}
            for name, launch in [
                ("norm", normalize),
                ("down", project),
                ("silu", activate),
                ("gate", project_gate),
                ("combine_norm", combine),
                ("full", full),
            ]:

                def repeated():
                    for _ in range(32):
                        launch()

                graph = capture_cuda_graph(repeated, warmup=2)
                values = [
                    value / 32
                    for value in transaction_samples(
                        lambda: graph.replay(), samples=args.samples
                    )["replay_us"]
                ]
                timings[name] = {
                    "median_us": statistics.median(values),
                    "samples_us": values,
                }
                if name == "full":
                    state.normal_().mul_(0.3)
                    block.normal_().mul_(0.2)
                    for tensor in (norm, merged, bottle, output, combined, next_norm):
                        tensor.fill_(float("nan"))
                    before = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert (
                        torch.cuda.memory_stats()["allocation.all.allocated"] == before
                    )
                    check()
            record = {
                "rows": rows,
                "timings": timings,
                "configs": {
                    name: asdict(p.selection.config)
                    for name, p in {**plans, "down": dp, "gate": gp}.items()
                },
                "correctness": "passed; changed-input replay and poisoned outputs; no allocations",
            }
            records.append(record)
            print(
                json.dumps(
                    {"rows": rows, **{k: v["median_us"] for k, v in timings.items()}}
                ),
                flush=True,
            )
            args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
