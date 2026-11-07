"""Compare direct typed MoE stores with the FP32-output-plus-copy contract."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_trellis_shared_input as shared


class CastRunner(shared.Runner):
    """Retain the FP32 result and perform the explicit caller-dtype copy."""

    def run(self):
        self.output.copy_(shared.fused_moe.run(binding=self.binding))
        return self.output


def reference_runner(fp32_runner, x, *, output_offset=0):
    output = torch.empty(x.numel() + output_offset, dtype=x.dtype, device=x.device)
    return CastRunner(
        fp32_runner.binding, output[output_offset:].view_as(x), fp32_runner.scratch
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 128, 1024])
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        parser.error("Expose exactly the assigned GPU")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = shared.read_exl3_manifest(args.checkpoint)
    layer = shared.read_exl3_layer(
        args.checkpoint, manifest, 1, first_slot=0, slot_count=12
    )
    source, weights = shared.trellis_from_exl3(layer)
    geometry = shared.fused_moe.MoEGeometry(
        num_experts=manifest.geometry.num_experts,
        hidden_size=manifest.geometry.hidden_size,
        intermediate_size=layer.local_intermediate_size,
    )
    records = []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        experts = shared.prepare_experts(source, weights, geometry, dtype)
        for rows in args.rows:
            x, ids, routes = shared.inputs(
                rows,
                geometry.hidden_size,
                geometry.num_experts,
                16,
                dtype,
                seed=6197 + rows,
            )
            print(
                json.dumps(dict(stage="prepare", rows=rows, dtype=dtype_name)),
                flush=True,
            )
            with (
                shared.prepared_runner(
                    experts, x, ids, routes, output_dtype=torch.float32
                ) as fp32,
                shared.prepared_runner(experts, x, ids, routes) as direct,
            ):
                reference = reference_runner(fp32, x)
                assert fp32.binding.output.dtype == torch.float32
                assert direct.binding.output.dtype == dtype
                assert direct.binding.output_cast_target is None
                shared.qualify_pair(
                    reference,
                    direct,
                    x,
                    ids,
                    routes,
                    geometry.num_experts,
                    seed=6239 + rows,
                )
                samples = {}
                for clustered in (False, True):
                    a, b, c = shared.inputs(
                        rows,
                        geometry.hidden_size,
                        geometry.num_experts,
                        16,
                        dtype,
                        seed=8923 + rows,
                        clustered=clustered,
                    )
                    x.copy_(a)
                    ids.copy_(b)
                    routes.copy_(c)
                    for cold in (False, True):
                        label = f"{'clustered' if clustered else 'distributed'}-{'cold' if cold else 'warm'}"
                        samples[label] = shared.time_pair(
                            reference, direct, repeats=args.repeats, cold=cold
                        )
                record = dict(
                    rows=rows,
                    dtype=dtype_name,
                    parity="bitwise",
                    samples=samples,
                    scratch_bytes=[
                        sum(t.nbytes for t in runner.scratch)
                        for runner in (reference, direct)
                    ],
                )
                records.append(record)
                (args.output / f"{dtype_name}-m{rows}.json").write_text(
                    json.dumps(record, indent=2)
                )
                print(
                    json.dumps(
                        dict(
                            rows=rows,
                            dtype=dtype_name,
                            medians_us={
                                policy: {
                                    name: statistics.median(values)
                                    for name, values in measured.items()
                                }
                                for policy, measured in samples.items()
                            },
                        )
                    ),
                    flush=True,
                )
    (args.output / "summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
