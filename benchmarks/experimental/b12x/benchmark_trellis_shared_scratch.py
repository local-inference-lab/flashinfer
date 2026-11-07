"""Measure concrete shared-input scratch savings and complete MoE graph latency."""

import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
from b12x._lib.compile_plan import program_keys

sys.path.insert(0, str(Path(__file__).parent))
import benchmark_trellis_shared_input as shared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 128, 1024])
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float16"])
    parser.add_argument("--repeats", type=int, default=60)
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
                seed=81371 + rows,
            )
            print(
                json.dumps(dict(stage="prepare", rows=rows, dtype=dtype_name)),
                flush=True,
            )
            with (
                shared.prepared_runner(
                    experts, x, ids, routes, freeze_transform=False
                ) as generic,
                shared.prepared_runner(experts, x, ids, routes) as compact,
            ):
                assert program_keys(generic.binding.fused_launch) == program_keys(
                    compact.binding.fused_launch
                )
                assert program_keys(generic.binding.topk_sum_launch) == program_keys(
                    compact.binding.topk_sum_launch
                )
                expected_saving = rows * 15 * geometry.hidden_size * 2
                shared.qualify_pair(
                    generic,
                    compact,
                    x,
                    ids,
                    routes,
                    geometry.num_experts,
                    seed=81401 + rows,
                    scratch_saving_bytes=expected_saving,
                )
                record = dict(
                    rows=rows,
                    dtype=dtype_name,
                    parity="bitwise",
                    generic_scratch_bytes=sum(t.nbytes for t in generic.scratch),
                    compact_scratch_bytes=sum(t.nbytes for t in compact.scratch),
                    expected_saving_bytes=expected_saving,
                    allocation_growth_bytes={},
                    samples={},
                )
                for label, runner in (("generic", generic), ("compact", compact)):
                    before = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    runner.graph.replay()
                    torch.cuda.synchronize()
                    record["allocation_growth_bytes"][label] = (
                        torch.cuda.max_memory_allocated() - before
                    )
                    assert record["allocation_growth_bytes"][label] == 0
                for clustered in (False, True):
                    a, b, c = shared.inputs(
                        rows,
                        geometry.hidden_size,
                        geometry.num_experts,
                        16,
                        dtype,
                        seed=81671 + rows,
                        clustered=clustered,
                    )
                    x.copy_(a)
                    ids.copy_(b)
                    routes.copy_(c)
                    for cold in (False, True):
                        label = f"{'clustered' if clustered else 'distributed'}-{'cold' if cold else 'warm'}"
                        record["samples"][label] = shared.time_pair(
                            generic, compact, repeats=args.repeats, cold=cold
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
                            generic_scratch_bytes=record["generic_scratch_bytes"],
                            compact_scratch_bytes=record["compact_scratch_bytes"],
                            replay_growth=record["allocation_growth_bytes"],
                            medians_us={
                                label: {
                                    kind: statistics.median(values)
                                    for kind, values in samples.items()
                                }
                                for label, samples in record["samples"].items()
                            },
                        )
                    ),
                    flush=True,
                )
    (args.output / "summary.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
