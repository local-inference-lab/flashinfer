"""Compare complete EXL3 MoE graphs using unchanged and shared-input kernels.

The real checkpoint supplies one TP extent of one layer. Activations and routes
are deterministic fixtures, not captured language-model traffic. Timings include
routing, transforms, both projections, route reduction, and the public output cast.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import importlib.util
import json
from pathlib import Path
import statistics
import sys

import torch

from b12x.moe import fused_moe
from b12x.moe.checkpoints.exl3 import (
    read_exl3_layer,
    read_exl3_manifest,
    trellis_from_exl3,
)
from b12x.moe.fused_moe.config import TrellisConfig
from b12x.moe.fused_moe.weights import ScaleFactors
from b12x.moe._shared.kernels.w4a16.kernel import fused_launch
from b12x.preparation import PreparationSession, PreparedCall
from b12x.preparation.types import require_prepared


def expanded_input_scales(source, weights, experts):
    """Express the same shared factors as explicit expert rows for a control."""
    declaration = source.config.to_dict()
    declaration["scale"]["input_scales"]["vectors"] = "per_expert"
    vectors = weights.input_scales.vectors
    shape = (experts, *vectors.shape)
    return replace(source, config=TrellisConfig.from_dict(declaration)), replace(
        weights, input_scales=ScaleFactors(vectors.expand(shape).contiguous())
    )


def prepare_experts(source, weights, geometry, dtype):
    plan = fused_moe.plan_weights(
        source=source,
        activation=fused_moe.ActivationSpec(
            mode="a16",
            nonlinearity="situ",
            io_dtype=dtype,
            rotation_dtype=torch.float16,
        ),
        geometry=geometry,
    )
    return fused_moe.prepare_weights(plan=plan, weights=weights, device="cuda")


@dataclass
class Runner:
    binding: object
    output: torch.Tensor
    scratch: tuple
    graph: torch.cuda.CUDAGraph | None = None

    def run(self):
        return fused_moe.run(binding=self.binding)

    def poison(self):
        for tensor in self.scratch:
            tensor.view(torch.uint8).fill_(0xA5)
        # Barrier state is protocol storage, not an uninitialized numeric operand.
        self.binding.kernel_workspace.zero_()
        self.output.fill_(float("nan"))

    def capture(self):
        self.run()
        self.run()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.run()


@contextmanager
def prepared_runner(
    experts, x, ids, weights, kernel_class=None, *, output_dtype=None, output_offset=0
):
    original = fused_launch.W4A16FusedMoeKernel
    if kernel_class is not None:
        fused_launch.W4A16FusedMoeKernel = kernel_class
    session = PreparationSession(device=x.device, autotune=False, compile_workers=0)
    output = torch.empty(
        x.numel() + output_offset, dtype=output_dtype or x.dtype, device=x.device
    )
    output = output[output_offset:].view_as(x)
    declaration = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=x.shape[0], top_k=ids.shape[1]),
    )

    def allocate(state):
        return tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )

    def bind(state, scratch):
        return state.bind(
            a=x,
            experts=experts,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )

    def primer(state):
        scratch = allocate(state)
        binding = bind(state, scratch)
        return PreparedCall(run=lambda: state.run(binding), owners=scratch)

    result = None
    try:
        request = declaration.request(
            name="trellis-complete-layer", prepare_call=primer
        )
        result = session.prepare((request,))
        state = require_prepared(request.plan, "moe.decode")
        scratch = allocate(state)
        binding = fused_moe.bind(
            request.plan,
            a=x,
            experts=experts,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        assert binding.fused_launch.full_rotation
        fused_launch.W4A16FusedMoeKernel = original
        yield Runner(binding, output, scratch)
    finally:
        fused_launch.W4A16FusedMoeKernel = original
        torch.cuda.synchronize()
        if result is not None:
            result.close()
        session.close()


def inputs(rows, hidden, experts, topk, dtype, *, seed, clustered=False):
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, hidden, generator=generator) * 0.0625).to(dtype).cuda()
    pool = min(experts, 4 * topk) if clustered else experts
    ids = torch.stack(
        [torch.randperm(pool, generator=generator)[:topk] for _ in range(rows)]
    ).to(device="cuda", dtype=torch.int32)
    weights = torch.softmax(torch.randn(rows, topk, generator=generator), -1).cuda()
    return x, ids, weights


def qualify_pair(control, candidate, x, ids, weights, experts, *, seed):
    assert [tensor.nbytes for tensor in control.scratch] == [
        tensor.nbytes for tensor in candidate.scratch
    ]
    control.capture()
    candidate.capture()
    initial = (x.clone(), ids.clone(), weights.clone())
    for case in (
        "ordinary",
        "permuted",
        "invalid",
        "all-invalid",
        "cast-boundaries",
        "changed",
    ):
        x.copy_(initial[0])
        ids.copy_(initial[1])
        weights.copy_(initial[2])
        if case == "permuted":
            ids.copy_(initial[1].flip(1))
            weights.copy_(initial[2].flip(1))
        elif case == "invalid":
            ids[:, ::3] = -1
            ids[-1] = -1
            x[-1].fill_(float("nan"))
        elif case == "all-invalid":
            ids.fill_(-1)
            x.fill_(float("nan"))
        elif case == "cast-boundaries":
            pattern = torch.tensor(
                [
                    0.0,
                    1.0,
                    -1.0,
                    2.0**-24,
                    -(2.0**-24),
                    2.0**-14,
                    0.5 + 2.0**-12,
                    1.0 + 2.0**-11,
                ],
                dtype=x.dtype,
                device=x.device,
            )
            x.copy_(pattern.repeat((x.numel() + 7) // 8)[: x.numel()].view_as(x))
        elif case == "changed":
            a, b, c = inputs(*x.shape, experts, ids.shape[1], x.dtype, seed=seed + 1)
            x.copy_(a)
            ids.copy_(b)
            weights.copy_(c)
        for runner in (control, candidate):
            runner.poison()
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        for _ in range(3):
            control.graph.replay()
            candidate.graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        if not torch.equal(control.output, candidate.output):
            difference = (control.output.float() - candidate.output.float()).abs()
            raise AssertionError(
                f"{case}: max difference {difference.nan_to_num().max().item()}, "
                f"mismatches {(control.output != candidate.output).sum().item()}"
            )
        assert torch.isfinite(candidate.output).all(), case
    x.copy_(initial[0])
    ids.copy_(initial[1])
    weights.copy_(initial[2])


def time_pair(control, candidate, *, repeats, cold):
    # Evict outside the event interval; both graphs receive the same cache policy.
    eviction = (
        torch.empty(128 << 20, dtype=torch.uint8, device="cuda") if cold else None
    )
    values = {"control": [], "candidate": []}
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    for repetition in range(repeats):
        order = [("control", control), ("candidate", candidate)]
        if repetition % 2:
            order.reverse()
        for name, runner in order:
            if eviction is not None:
                eviction.fill_(repetition)
            start.record()
            runner.graph.replay()
            end.record()
            end.synchronize()
            values[name].append(start.elapsed_time(end) * 1000)
    return values


def load_control(path):
    spec = importlib.util.spec_from_file_location("trellis_unchanged_fused", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.W4A16FusedMoeKernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-fused", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--first-slot", type=int, default=0)
    parser.add_argument("--slot-count", type=int, default=12)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 128, 1024])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["bfloat16", "float16"],
        default=["bfloat16", "float16"],
    )
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly the assigned GPU through CUDA_VISIBLE_DEVICES")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = load_control(args.baseline_fused)
    manifest = read_exl3_manifest(args.checkpoint)
    layer = read_exl3_layer(
        args.checkpoint,
        manifest,
        args.layer,
        first_slot=args.first_slot,
        slot_count=args.slot_count,
    )
    source, weights = trellis_from_exl3(layer)
    geometry = fused_moe.MoEGeometry(
        num_experts=manifest.geometry.num_experts,
        hidden_size=manifest.geometry.hidden_size,
        intermediate_size=layer.local_intermediate_size,
    )
    summaries = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        experts = prepare_experts(source, weights, geometry, dtype)
        for rows in args.rows:
            x, ids, route_weights = inputs(
                rows,
                geometry.hidden_size,
                geometry.num_experts,
                16,
                dtype,
                seed=19103 + rows,
            )
            print(
                json.dumps({"stage": "prepare", "rows": rows, "dtype": dtype_name}),
                flush=True,
            )
            with (
                prepared_runner(experts, x, ids, route_weights, baseline) as control,
                prepared_runner(experts, x, ids, route_weights) as candidate,
            ):
                qualify_pair(
                    control,
                    candidate,
                    x,
                    ids,
                    route_weights,
                    geometry.num_experts,
                    seed=31007 + rows,
                )
                record = dict(
                    rows=rows,
                    dtype=dtype_name,
                    parity="bitwise",
                    scratch_bytes=[
                        sum(t.nbytes for t in runner.scratch)
                        for runner in (control, candidate)
                    ],
                    direct_lut=candidate.binding.fused_launch.trellis_direct_lut,
                    timings={},
                )
                for clustered in (False, True):
                    a, b, c = inputs(
                        rows,
                        geometry.hidden_size,
                        geometry.num_experts,
                        16,
                        dtype,
                        seed=78193 + rows,
                        clustered=clustered,
                    )
                    x.copy_(a)
                    ids.copy_(b)
                    route_weights.copy_(c)
                    for cold in (False, True):
                        label = f"{'clustered' if clustered else 'distributed'}-{'cold' if cold else 'warm'}"
                        samples = time_pair(
                            control, candidate, repeats=args.repeats, cold=cold
                        )
                        record["timings"][label] = samples
                (args.output / f"{dtype_name}-m{rows}.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                summaries.append(record)
                print(
                    json.dumps(
                        dict(
                            rows=rows,
                            dtype=dtype_name,
                            parity=record["parity"],
                            median_us={
                                name: {
                                    variant: statistics.median(values)
                                    for variant, values in samples.items()
                                }
                                for name, samples in record["timings"].items()
                            },
                        )
                    ),
                    flush=True,
                )
        del experts
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
