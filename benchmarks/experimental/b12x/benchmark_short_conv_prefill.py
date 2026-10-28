"""Race the prefill short-convolution tile geometry and time every candidate.

Run with CUDA_VISIBLE_DEVICES selecting one GPU. Each case is
TOKENS:LENGTH[,LENGTH...] with GLM-5.3 KDA geometry by default (24576 channels,
--channels selects another width), four taps,
BF16 input rows strided as a packed projection, and a conv-state pool. The
ordinary preparation race keeps every candidate as a finalist; the finalists
are then replayed as CUDA graphs, checked bitwise against the 32 x 128 tile of
four warps with Int64 addressing, and timed warm and cold with balanced rounds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import triton

from b12x.preparation import FinalistPolicy, PreparationSession, PreparedCall
from b12x.preparation._measurement import _l2_flush_fn
from b12x.sequence import short_conv
from b12x.testing.benchmark import measure_calls

CHANNELS = 24576


def candidate_name(config):
    width = "i64" if config.wide_index else "i32"
    return f"t{config.block_t}c{config.block_c}w{config.num_warps}{width}"


def _tiled_kernel_arm(binding, config, wide, device):
    """Graph of the tiled convolution kernel alone at one address width.

    The pipeline's history gather and request-id kernels run once beforehand;
    ``wide=True`` with the 32 x 128 tile is the Int64-addressed fixed geometry.
    """
    from b12x.sequence import _short_conv as kernels

    out = torch.empty_like(binding.out)
    grid = (
        triton.cdiv(binding.x.shape[0], config.block_t),
        triton.cdiv(CHANNELS, config.block_c),
    )

    def launch():
        kernels._short_conv_tiled_kernel[grid](
            binding.x,
            binding.gathered_state,
            binding.conv_weight,
            binding.query_start_loc,
            binding.request_ids,
            binding.state_slot_ids,
            binding.num_tokens,
            out,
            CHANNELS,
            3,
            4,
            1,
            config.block_c,
            binding.x.stride(0),
            config.block_t,
            binding.x.shape[0],
            None,
            True,
            wide,
            num_warps=config.num_warps,
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    return PreparedCall(
        run=graph.replay, output=out, produce=lambda: None, owners=(graph, out)
    )


def benchmark_case(
    tokens, lengths, *, stride, samples, rounds, workers, cache_dir, carried
):
    device = torch.device("cuda", 0)
    sequences = max(len(lengths), 1)
    slots = sequences + 2
    torch.manual_seed(2024)
    storage = torch.randn((tokens, stride), device=device).to(torch.bfloat16)
    x = storage[:, :CHANNELS]
    args = dict(
        x=x,
        conv_weight=(torch.randn((CHANNELS, 4), device=device) * 0.2).to(
            torch.bfloat16
        ),
        conv_state=torch.randn((slots, CHANNELS, 3), device=device).to(torch.bfloat16),
        out=torch.full_like(x, float("nan")),
        query_start_loc=torch.zeros(sequences + 1, dtype=torch.int32, device=device),
        state_slot_ids=torch.arange(1, sequences + 1, dtype=torch.int64, device=device),
        state_is_fresh=torch.full(
            (sequences,), not carried, dtype=torch.bool, device=device
        ),
        num_accepted_tokens=torch.ones(sequences, dtype=torch.int32, device=device),
        num_seqs=torch.full((1,), len(lengths), dtype=torch.int32, device=device),
        num_tokens=torch.full((1,), sum(lengths), dtype=torch.int32, device=device),
    )
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    args["query_start_loc"].copy_(torch.tensor(starts, dtype=torch.int32))
    initial = args["conv_state"].clone()
    source = x.clone()

    def produce():
        x.copy_(source)

    caps = short_conv.Caps(
        device=device,
        mode="prefill",
        channels=CHANNELS,
        kernel_size=4,
        max_tokens=tokens,
        max_seqs=sequences,
        max_state_slots=slots,
    )
    plan = short_conv.plan(caps, invocation=short_conv.invocation_from_tensors(**args))
    scratch = {
        spec.name: torch.empty(spec.shape, dtype=spec.dtype, device=device)
        for spec in plan.scratch_specs()
    }

    def reset():
        args["conv_state"].copy_(initial)

    def make_call(state):
        binding = state.bind(plan=plan, scratch=scratch, **args)

        def run():
            state.run(binding)

        reset()
        run()
        reset()
        return PreparedCall(
            run=run,
            output=binding.out,
            reset=reset,
            produce=produce,
            restore=reset,
            owners=(binding,),
        )

    request = plan.request(
        name="short_conv",
        prepare_call=make_call,
        benchmark_call=make_call,
        finalist_policy=FinalistPolicy(
            family_fields=("block_t", "block_c", "num_warps", "wide_index"),
            max_total=40,
        ),
    )
    with PreparationSession(
        device=device,
        compile_workers=workers,
        rounds=rounds,
        samples=samples,
        cache_policy="cold",
        cache_dir=cache_dir,
    ) as session:
        with session.prepare((request,)) as result:
            selected = asdict(result.selections["short_conv"].config)
            coverage = dict(result.coverage["short_conv"])
            finalists = result.finalists["short_conv"]
            names, arms, configurations, outputs = {}, {}, {}, {}
            for finalist in finalists:
                config = finalist.selection.config
                name = candidate_name(config)
                pinned = short_conv.plan(
                    caps,
                    invocation=short_conv.invocation_from_tensors(**args),
                    override=config,
                )
                with session.prepare(
                    (pinned.request(name=f"short_conv.{name}", prepare_call=make_call),)
                ):
                    binding = short_conv.bind(pinned, scratch=scratch, **args)
                    graph = torch.cuda.CUDAGraph()
                    with session.capture(), torch.cuda.graph(graph):
                        short_conv.run(binding)
                    reset()
                    args["out"].fill_(float("nan"))
                    before = torch.cuda.memory_allocated(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    if torch.cuda.memory_allocated(device) != before:
                        raise AssertionError("graph replay allocated GPU storage")
                    outputs[name] = (args["out"].clone(), args["conv_state"].clone())
                    reset()
                    arms[name] = PreparedCall(
                        run=graph.replay,
                        reset=reset,
                        produce=produce,
                        output=binding.out,
                        owners=(graph, binding, pinned),
                    )
                    configurations[name] = asdict(config)
                    names[name] = config
            default_name = "t32c128w4i64"
            reference = outputs[default_name]
            exact = {
                name: bool(
                    torch.equal(out, reference[0]) and torch.equal(state, reference[1])
                )
                for name, (out, state) in outputs.items()
            }
            if not all(exact.values()):
                raise AssertionError(
                    f"candidates differ from the default tile: {exact}"
                )
            # The last replay left the history gather and request ids in the
            # shared scratch; the kernel-only arms read them.
            torch.cuda.synchronize(device)
            kernel_arms = {
                name: _tiled_kernel_arm(binding, config, config.wide_index, device)
                for name, config in names.items()
            }
            kernel_measurements = {}
            for temperature in ("warm", "cold"):
                measured = measure_calls(
                    kernel_arms,
                    device=device,
                    samples=samples,
                    rounds=rounds,
                    eviction=_l2_flush_fn(device, enabled=temperature == "cold"),
                )
                kernel_measurements[temperature] = {
                    name: measured.latencies_us[name] for name in kernel_arms
                }
            measurements = {}
            for temperature in ("warm", "cold"):
                measured = measure_calls(
                    arms,
                    device=device,
                    samples=samples,
                    rounds=rounds,
                    eviction=_l2_flush_fn(device, enabled=temperature == "cold"),
                )
                measurements[temperature] = {
                    name: {
                        "median_us": measured.latencies_us[name],
                        "samples_us": measured.raw_samples(name),
                    }
                    for name in arms
                }
            return {
                "lengths": lengths,
                "tokens": tokens,
                "input_row_stride": stride,
                "carried_history": carried,
                "selected": selected,
                "coverage": coverage,
                "finalists": configurations,
                "race_latency_us": {
                    candidate_name(f.selection.config): f.latency_us for f in finalists
                },
                "bitwise_equal_to_default": exact,
                "measurements": measurements,
                "tiled_kernel_only_us": kernel_measurements,
                "timing": "complete captured prefill pipeline; restoration outside timing",
            }


def main():
    global CHANNELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--carried", action="store_true")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--compile-workers", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    CHANNELS = options.channels
    if options.stride is None:
        options.stride = CHANNELS + 192
    results = []
    for case in options.cases or ["4096:4096", "4096:1024,1024,1024,1024"]:
        tokens, _, lengths = case.partition(":")
        lengths = [int(value) for value in lengths.split(",")]
        print(f"Racing tokens={tokens} lengths={lengths}", flush=True)
        result = benchmark_case(
            int(tokens),
            lengths,
            stride=options.stride,
            samples=options.samples,
            rounds=options.rounds,
            workers=options.compile_workers,
            cache_dir=options.cache_dir,
            carried=options.carried,
        )
        results.append(result)
        options.output.write_text(json.dumps(results, indent=2))
        warm = result["measurements"]["warm"]
        best = min(warm, key=lambda name: warm[name]["median_us"])
        print(
            f"selected={result['selected']} best_warm={best} "
            f"{warm[best]['median_us']:.1f} us default "
            f"{warm['t32c128w4i64']['median_us']:.1f} us",
            flush=True,
        )


if __name__ == "__main__":
    main()
