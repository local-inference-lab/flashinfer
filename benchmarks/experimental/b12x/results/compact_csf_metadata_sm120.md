# Compact CSF metadata prefetch

Run the public checkpoint-backed MoE benchmark from the repository root after
installing FlashInfer's b12x dependencies. Set `CHECKPOINT` to a local DeepSeek
V4.1 Flash 1017 source checkpoint. The benchmark loads the selected expert layer
and TP shard; it does not require Flare, a resident model or a server endpoint.
Select an idle GPU with `--device` and repeat the same command at the control
revision and the metadata-prefetch revision in A/B/B/A order.

```sh
CHECKPOINT=/path/to/DeepSeek-V4.1-Flash-1017
PYTHONPATH=flashinfer/experimental:. python benchmarks/experimental/b12x/benchmark_moe.py \
  --model-profile deepseek-v4.1-flash --model-path "$CHECKPOINT" \
  --layer-idx 3 --tp-size 4 --tp-rank 0 --batch-sizes 8 --device 0 \
  --scale-compression csf_32x2_v1 --routing-workload mixture \
  --tuning-samples 2 --routing-samples 2 --validate oracle \
  --graph-only --time-graph-replay --iters 9 --repeats 4 \
  --flush-l2 --output-json csf-cold.json
```

Repeat with `--no-flush-l2 --output-json csf-warm.json` for warm-cache replay.
Keep autotuning enabled and do not supply `--moe-config`. Record the selected
configuration; the optimization applies when normal selection chooses the
compact micro path with inline CSF scales. The public CLI losslessly encodes
source scale bytes before preparation and keeps input quantization, compressed
scale handling, FC1, activation, FC2 and top-k reduction in the measured call.
Router, shared-expert and TP-collective work are outside the interval.

## Public CLI measurements

[Raw public-CLI measurements](compact_csf_metadata_public_cli_sm120.json) use the
command above at control revision `344e4837` and patch revision `986cd5ecb`, in A/B/B/A order for each
cache policy. The native1017 checkpoint is losslessly CSF32x2-encoded during
setup. Default tuning and oracle validation remain enabled. The mixture has
three route distributions (`diffuse`, `skewed`, `correlated`), with two held-out
seeds each. Four timing passes per case each record 81 gated event intervals
with `--iters 9`; every interval is in the data file.

Both arms select compact micro N128, inline CSF, two copy groups and two pipeline
stages. The normally selected cluster budget varies between 164, 188 and automatic;
no config is pinned. Graph/eager consistency, the numerical oracle, mutated
inputs/weights and restored graph guards pass for every case. Means across the
six cases are averaged across the two matching ABBA arms:

| Cache policy | Control | Metadata prefetch | Reduction |
| --- | ---: | ---: | ---: |
| L2 eviction | 102.322667us | 101.269333us | 1.0294% |
| Warm replay | 94.834667us | 93.738666us | 1.1557% |

The GPU is RTX PRO 6000 Blackwell SM120, Torch 2.14.1+cu132/CUDA 13.2. Core policy
is automatic, the existing memory offset is +6000 MHz, and the benchmark applies
no clock lock. Resident services stay idle with unchanged counters and GPU
owners before/after. These are complete pre-routed component measurements under
normal tuning, including plan selection, not full-model/HTTP attribution.
Native scale execution and SM121/full-model performance remain unqualified.

## Historical component-wrapper measurements

[Raw per-round samples and selected configurations](compact_csf_metadata_sm120.json)
come from a checkpoint-shard component wrapper with the same operation boundary.
It used 384 experts, H5120, per-rank N576, top-6 routing and eight BF16 input rows
from layer3, TP4 rank0. Four route distributions (`diffuse`, `skewed`,
`correlated`, `shared_40`) each had evaluation seeds34121/71921; activations and
route probabilities used CPU generator seed81221. The data file describes that
producer and includes every arm's samples. Each of nine gated rounds has four
samples **per routing case**, with cold and warm policies measured separately.
The public CLI's route/input generator differs: the command above reproduces
the operation and geometry, not those exact operands or recorded timings. The public-CLI measurements above independently qualify that protocol.

On RTX PRO 6000 Blackwell SM120, Torch 2.14.1+cu132/CUDA 13.2, default-tier tuning
selected micro N128 with inline CSF, two copy groups and two pipeline stages.
Core clocks used automatic policy and the host retained a +6000 MHz memory
offset; the measurement applied no clock lock. Resident services were idle and
request counters/GPU owners stayed unchanged. Means of the two arm medians:

| Cache policy | Control | Metadata prefetch |
| --- | ---: | ---: |
| L2 eviction before replay | 99.05350us | 97.99400us |
| Warm replay | 97.76550us | 96.79800us |

All eight held-out routing outputs match bitwise across four arms. The focused
GPU checks also cover changed compressed payload bytes, invalid full-width
expert IDs, zero/tiny inputs and poisoned replay. This is an optimization of an
existing experimental b12x backend, selected explicitly through its benchmark;
it adds no stable API, auto-backend registration or AOT registration. Native
scale execution and SM121/full-model performance are outside this qualification.
