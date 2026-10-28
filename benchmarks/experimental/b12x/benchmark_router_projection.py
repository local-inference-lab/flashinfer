"""Graph benchmark for BF16 router projection, descending-ID top-k, and softmax.

The split control uses prepared b12x projection and routing plans. The fused
CuTe DSL prototype has the same BF16 logit rounding boundary and FP32 outputs.
Checkpoint weights are optional; activations are synthetic in either mode.
Timings include projection, selection, normalization, and counter recycling.
Preparation, input generation, correctness checks, and L2 eviction are untimed.

Example (device ordinals respect CUDA_VISIBLE_DEVICES):
  python -m benchmarks.experimental.b12x.benchmark_router_projection \
    --device 10 --checkpoint CHECKPOINT --rows 1 2 3 4 5 6 --output router.json
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path

import torch
from b12x.gemm import bf16_gemv
from b12x.moe import fused_moe
from b12x.moe._router_projection import compile_router
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.benchmark import measure_calls
from benchmarks.experimental.b12x.checkpoint_loader import IndexedSafetensorLoader


def reference(source, weight, top_k):
    # Match FP32 accumulation followed by a single BF16 output rounding.
    reduced_precision = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    )
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        logits = torch.nn.functional.linear(source, weight).float()
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            reduced_precision
        )
    # Reverse columns before stable sorting so equal logits prefer larger IDs.
    order = logits.flip(1).argsort(dim=1, descending=True, stable=True)[:, :top_k]
    ids = logits.shape[1] - 1 - order
    selected = logits.gather(1, ids)
    return selected, ids.int(), selected.softmax(1)


def outputs(rows, top_k, device):
    return (
        torch.empty((rows, top_k), device=device, dtype=torch.float32),
        torch.empty((rows, top_k), device=device, dtype=torch.int32),
        torch.empty((rows, top_k), device=device, dtype=torch.float32),
    )


@contextmanager
def prepare_case(
    rows,
    weight,
    *,
    top_k=10,
    implementations=("split", "fused"),
    rows_per_tile=4,
    warps=8,
    warps_per_expert=4,
    projection="simt",
    split_k=1,
    autotune=True,
):
    experts, hidden = weight.shape
    source = torch.zeros((rows, hidden), device=weight.device, dtype=torch.bfloat16)
    split_logits = torch.empty(
        (rows, experts), device=weight.device, dtype=torch.bfloat16
    )
    split_out, fused_out = (
        outputs(rows, top_k, weight.device),
        outputs(rows, top_k, weight.device),
    )
    workspace = torch.empty(
        (rows * (split_k + 1 if projection == "tma" else 1), experts),
        device=weight.device,
        dtype=torch.float32,
    )
    counters = torch.zeros(
        (
            ((rows + rows_per_tile - 1) // rows_per_tile)
            * ((experts + 63) // 64 + 1 if projection == "tma" else 1),
        ),
        device=weight.device,
        dtype=torch.int32,
    )
    projection_plan = bf16_gemv.plan(
        bf16_gemv.query_from_call(source, weight, out=split_logits)
    )
    routing = fused_moe.plan_route_topk(
        fused_moe.RouteTopKInvocation(
            num_tokens=rows, num_experts=experts, top_k=top_k, logits_dtype="bfloat16"
        )
    )
    tuning_source = torch.randn_like(source)
    tuning_logits = torch.empty_like(split_logits)
    with PreparationSession(
        device=weight.device, autotune=autotune, compile_workers=4
    ) as session:
        session.prepare(
            (
                projection_plan.request(
                    name="projection",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(source, weight, out=split_logits)
                    ),
                    benchmark_call=lambda state: PreparedCall(
                        run=lambda: state.run(tuning_source, weight, out=tuning_logits),
                        produce=lambda: None,
                        output=tuning_logits,
                        owners=(tuning_source, weight, tuning_logits),
                    ),
                ),
                routing.request(
                    name="topk",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(split_logits, *split_out)
                    ),
                ),
            )
        )

        def split():
            bf16_gemv.mm(source, weight, out=split_logits, plan=projection_plan)
            fused_moe.route_topk(routing, split_logits, *split_out)

        calls = {}
        if "split" in implementations:
            calls["split"] = PreparedCall(
                run=split,
                produce=lambda: None,
                output=split_out,
                owners=(
                    source,
                    weight,
                    split_logits,
                    split_out,
                    projection_plan,
                    routing,
                ),
            )
        if "fused" in implementations:
            kernel = compile_router(
                device=weight.device,
                hidden=hidden,
                experts=experts,
                top_k=top_k,
                rows_per_tile=rows_per_tile,
                warps=warps,
                warps_per_expert=warps_per_expert,
                projection=projection,
                split_k=split_k,
                max_rows=rows,
            )
            calls["fused"] = PreparedCall(
                run=lambda: kernel(source, weight, workspace, counters, *fused_out),
                produce=lambda: None,
                output=fused_out,
                owners=(source, weight, workspace, counters, fused_out, kernel),
            )
        session.freeze()
        yield (
            source,
            calls,
            workspace,
            counters,
            asdict(projection_plan.selection.config),
        )


def check_case(source, weight, calls, workspace, counters, *, top_k, seeds):
    graphs = {}
    for name, call in calls.items():
        call.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call.run()
        graphs[name] = graph
    try:
        for seed in (*seeds, None):
            if seed is None:
                source.zero_()
            else:
                generator = torch.Generator(device=source.device).manual_seed(seed)
                source.copy_(
                    torch.randn(
                        source.shape,
                        dtype=source.dtype,
                        device=source.device,
                        generator=generator,
                    )
                )
            expected = reference(source, weight, top_k)
            for name, graph in graphs.items():
                for _ in range(3):
                    workspace.fill_(float("nan"))
                    for tensor in calls[name].output:
                        tensor.fill_(-1)
                    torch.cuda.synchronize()
                    allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
                    graph.replay()
                    torch.cuda.synchronize()
                    assert (
                        torch.cuda.memory_stats()["allocation.all.allocated"]
                        == allocations
                    )
                    selected, ids, scores = calls[name].output
                    torch.testing.assert_close(
                        ids,
                        expected[1],
                        rtol=0,
                        atol=0,
                        msg=lambda msg: f"{name}, seed={seed}: {msg}",
                    )
                    torch.testing.assert_close(
                        selected, expected[0], rtol=0.008, atol=0.002
                    )
                    torch.testing.assert_close(
                        scores, expected[2], rtol=0.008, atol=0.002
                    )
                    assert not counters.count_nonzero(), (
                        "counter did not recycle after replay"
                    )
    finally:
        for graph in graphs.values():
            graph.reset()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 16, 128]
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--layers", nargs="+", default=["0", "23", "47", "mtp"])
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["split", "fused"],
        default=["split", "fused"],
    )
    parser.add_argument("--rows-per-tile", type=int, default=4)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--warps-per-expert", type=int, default=4)
    parser.add_argument(
        "--projection", choices=("simt", "tensor_core", "tma"), default="simt"
    )
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[8719, 1927, 421])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--eviction-mib", type=int, default=128)
    parser.add_argument("--no-autotune", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    if any(row <= 0 for row in args.rows) or args.eviction_mib < 0:
        parser.error("row counts must be positive and eviction size nonnegative")
    torch.cuda.set_device(args.device)
    torch.manual_seed(2153)
    device = torch.device("cuda", args.device)
    loader = IndexedSafetensorLoader(args.checkpoint) if args.checkpoint else None
    flush = torch.zeros(args.eviction_mib << 20, device=device, dtype=torch.uint8)
    result = dict(
        gpu=torch.cuda.get_device_name(),
        gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        activations="synthetic BF16 normal",
        reference_accumulation="float32",
        eviction_mib=args.eviction_mib,
        rows_per_tile=args.rows_per_tile,
        warps=args.warps,
        warps_per_expert=args.warps_per_expert,
        projection=args.projection,
        split_k=args.split_k,
        autotune=not args.no_autotune,
        seeds=args.seeds,
        comparisons=[],
    )
    for layer in args.layers if loader else ["synthetic"]:
        if loader:
            prefix = (
                "mtp.layers.0"
                if layer == "mtp"
                else f"model.language_model.layers.{int(layer)}"
            )
            weight = loader.get_tensor(prefix + ".mlp.gate.weight").to(device)
            if weight.dtype != torch.bfloat16:
                raise ValueError(
                    "benchmark requires checkpoint BF16 router weights without conversion"
                )
        else:
            weight = (
                torch.randn((args.experts, args.hidden), device=device) * 0.02
            ).bfloat16()
        for rows in args.rows:
            with prepare_case(
                rows,
                weight,
                top_k=args.top_k,
                implementations=args.implementations,
                rows_per_tile=args.rows_per_tile,
                warps=args.warps,
                warps_per_expert=args.warps_per_expert,
                projection=args.projection,
                split_k=args.split_k,
                autotune=not args.no_autotune,
            ) as case:
                source, calls, workspace, counters, config = case
                check_case(
                    source,
                    weight,
                    calls,
                    workspace,
                    counters,
                    top_k=args.top_k,
                    seeds=args.seeds,
                )
                generator = torch.Generator(device=device).manual_seed(643)
                source.copy_(
                    torch.randn(
                        source.shape,
                        device=device,
                        dtype=source.dtype,
                        generator=generator,
                    )
                )
                record = dict(
                    layer=layer,
                    rows=rows,
                    hidden=weight.shape[1],
                    experts=weight.shape[0],
                    top_k=args.top_k,
                    split_config=config,
                    correctness="passed",
                )
                if not args.check_only:
                    measured = measure_calls(
                        calls,
                        samples=args.samples,
                        rounds=args.rounds,
                        eviction=(lambda: flush.add_(1)) if args.eviction_mib else None,
                    )
                    record.update(
                        latencies_us=measured.latencies_us,
                        method=measured.method,
                        samples=[asdict(s) for s in measured.samples],
                    )
                result["comparisons"].append(record)
                print(
                    json.dumps({k: v for k, v in record.items() if k != "samples"}),
                    flush=True,
                )
                if args.output:
                    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
