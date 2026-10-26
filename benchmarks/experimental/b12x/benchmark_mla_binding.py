"""Measure scratch binding plus prepared DeepSeek V4.1 compressed attention.

The captured call includes logical page mapping, query/cache conversion,
attention and merge. Inputs are synthetic normalized BF16 queries and packed
window/indexed caches; no model checkpoint or inference server is required.
Compare source revisions in separate processes, in A/B/B/A order, on an idle
GPU. Preparation uses normal default-tier autotuning without plan overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--verify-width", type=int, default=8)
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--ratios", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    parser.add_argument("--compile-workers", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=96741)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.rows, args.heads, args.verify_width, args.rounds, args.samples) <= 0:
        parser.error("rows, heads, verify width, rounds and samples must be positive")
    if any(
        context < 512 or context % 128 or context // 128 > 8192
        for context in args.contexts
    ):
        parser.error("contexts must be multiples of 128 between 512 and 1048576")
    if any(
        context // ratio < 512 for context in args.contexts for ratio in args.ratios
    ):
        parser.error("each context/ratio must provide at least 512 indexed entries")
    return args


def make_inputs(args, context, ratio, torch, pack):
    device = torch.device("cuda", args.device)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + args.rows * 100 + context
    )
    page, window_page = 128 // ratio, 136
    batch = (args.rows + args.verify_width - 1) // args.verify_width
    records = context // ratio
    blocks = (records + page - 1) // page
    scales = torch.linspace(0.03, 0.7, 32, device=device).repeat_interleave(16)

    def values(count):
        value = torch.randn((count, 512), device=device, generator=generator) * scales
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True))).bfloat16()

    window = pack(values(batch * window_page), page_size=window_page, cache_kind="swa")
    indexed = pack(values(batch * blocks * page), page_size=page, cache_kind="indexed")
    query = values(args.rows * args.heads).reshape(args.rows, args.heads, 512)
    request = torch.arange(args.rows, device=device) // args.verify_width
    window_ids = (
        request[:, None] * window_page
        + (
            torch.arange(128, device=device)[None]
            + torch.arange(args.rows, device=device)[:, None] % args.verify_width
        )
        % window_page
    ).int()
    logical = torch.stack(
        [
            torch.randperm(records, device=device, generator=generator)[:512]
            .sort()
            .values
            for _ in range(args.rows)
        ]
    ).int()
    table = torch.full((args.rows, 8192), -1, dtype=torch.int32, device=device)
    table[:, :blocks] = (
        request[:, None] * blocks + torch.arange(blocks, device=device)[None]
    ).int()
    physical = table.gather(1, (logical // page).long()) * page + logical % page
    lengths = torch.full((args.rows,), 128, dtype=torch.int32, device=device)
    indexed_lengths = torch.full_like(lengths, 512)
    return dict(
        q=query,
        window=window,
        indexed=indexed,
        window_ids=window_ids,
        logical=logical,
        table=table,
        physical=physical,
        lengths=lengths,
        indexed_lengths=indexed_lengths,
        sink=torch.linspace(-2, 2, args.heads, device=device),
        page=page,
        window_page=window_page,
    )


def main(argv=None):
    args = parse_args(argv)
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention._shared.mla.compressed_reference import (
        compressed_sparse_mla_reference,
        pack_deepseek_v41_cache_reference,
    )
    from b12x.preparation import PreparationSession, PreparedCall, autotune_effort
    from b12x.preparation._measurement import (
        _consume,
        _prepare_race,
        measure_race_steps,
    )

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    result = dict(
        configuration=vars(args) | {"output": str(args.output)},
        gpu=torch.cuda.get_device_name(device),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        effort="default",
        cases=[],
    )
    with torch.no_grad(), autotune_effort("default"):
        for context in args.contexts:
            for ratio in args.ratios:
                inputs = make_inputs(
                    args, context, ratio, torch, pack_deepseek_v41_cache_reference
                )
                output = torch.empty_like(inputs["q"])
                bind_args = dict(
                    q=inputs["q"],
                    swa_indices=inputs["window_ids"],
                    swa_lengths=inputs["lengths"],
                    indexed_indices=inputs["logical"],
                    indexed_lengths=inputs["indexed_lengths"],
                    indexed_page_table=inputs["table"],
                )
                run_args = dict(
                    swa_k_cache=inputs["window"],
                    indexed_k_cache=inputs["indexed"],
                    attn_sink=inputs["sink"],
                    sm_scale=512**-0.5,
                    out=output,
                )
                caps = mla.Caps(
                    device=device,
                    num_q_heads=args.heads,
                    max_q_rows=args.rows,
                    max_width=640,
                    swa_width=128,
                    indexed_width=512,
                    max_page_table_width=8192,
                    swa_page_size=136,
                    indexed_page_size=inputs["page"],
                    cache_format="deepseek_v41",
                    mode="decode",
                    decode_row_capacity=args.rows,
                    compute_mode="bf16",
                )
                with PreparationSession(
                    device=device, autotune=True, compile_workers=args.compile_workers
                ) as session:
                    plan = mla.plan(
                        caps,
                        invocation=mla.invocation_from_tensors(
                            q=inputs["q"],
                            swa_k_cache=inputs["window"],
                            indexed_k_cache=inputs["indexed"],
                            attn_sink=inputs["sink"],
                            out=output,
                        ),
                    )

                    def prepare(state):
                        (spec,) = state.scratch_specs()
                        scratch = torch.empty(
                            spec.shape, dtype=spec.dtype, device=device
                        )
                        binding = state.bind_for_preparation(
                            scratch=scratch, **bind_args
                        )
                        trial = torch.empty_like(output)
                        seed = inputs["q"].clone()
                        return PreparedCall(
                            run=lambda: state.run(
                                binding, **(run_args | {"out": trial})
                            ),
                            produce=lambda: inputs["q"].copy_(seed),
                            output=trial,
                            owners=(binding, scratch, trial, seed, inputs),
                        )

                    session.prepare(
                        (
                            plan.request(
                                name="attention",
                                prepare_call=prepare,
                                benchmark_call=prepare,
                            ),
                        )
                    )
                    (spec,) = plan.scratch_specs()
                    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)

                    def run():
                        binding = mla.bind(plan, scratch=scratch, **bind_args)
                        mla.run(plan=plan, binding=binding, **run_args)

                    run()
                    expected = compressed_sparse_mla_reference(
                        inputs["q"],
                        inputs["window"],
                        inputs["window_ids"],
                        inputs["lengths"],
                        extra_k_cache=inputs["indexed"],
                        extra_indices=inputs["physical"],
                        extra_topk_lengths=inputs["indexed_lengths"],
                        swa_page_size=136,
                        extra_page_size=inputs["page"],
                        sm_scale=512**-0.5,
                        attn_sink=inputs["sink"],
                        cache_format="deepseek_v41",
                    )
                    torch.testing.assert_close(output, expected, rtol=0.035, atol=0.035)
                    reference = output.clone()
                    session.freeze()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    try:
                        for _ in range(3):
                            scratch.fill_(255)
                            output.fill_(float("nan"))
                            graph.replay()
                            torch.cuda.synchronize(device)
                            torch.testing.assert_close(
                                output, reference, rtol=0, atol=0
                            )
                        seed = inputs["q"].clone()
                        call = PreparedCall(
                            run=graph.replay,
                            produce=lambda: inputs["q"].copy_(seed),
                            output=output,
                            owners=(graph, plan, scratch, inputs),
                            capture_safe=True,
                        )
                        case = dict(
                            context=context,
                            ratio=ratio,
                            selected_config=asdict(plan.selection.config),
                            measurements={},
                        )
                        for policy in ("cold", "warm"):
                            race = _prepare_race(
                                (call,),
                                device_ordinal=args.device,
                                samples=args.samples,
                                primed=True,
                                cache_policy=policy,
                            )
                            samples = []
                            try:
                                measured = _consume(
                                    measure_race_steps(
                                        race,
                                        device_ordinal=args.device,
                                        rounds=args.rounds,
                                        adaptive_repeats=False,
                                        sample_observer=lambda candidate,
                                        turn,
                                        repeat,
                                        values: (
                                            samples.append(
                                                dict(round=turn, values_us=list(values))
                                            )
                                        ),
                                    )
                                )
                                case["measurements"][policy] = dict(
                                    median_round_mean_us=measured.latencies_us[0],
                                    samples=samples,
                                )
                            finally:
                                race.close()
                        result["cases"].append(case)
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                        print(json.dumps(case), flush=True)
                    finally:
                        graph.reset()


if __name__ == "__main__":
    main()
