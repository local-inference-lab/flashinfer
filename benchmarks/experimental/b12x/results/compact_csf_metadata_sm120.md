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

## Recorded conditions and results

[Raw per-round samples and selected configurations](compact_csf_metadata_sm120.json)
come from a checkpoint-shard component wrapper with the same operation boundary.
It used 384 experts, H5120, per-rank N576, top-6 routing and eight BF16 input rows
from layer3, TP4 rank0. Four route distributions (`diffuse`, `skewed`,
`correlated`, `shared_40`) each had evaluation seeds34121/71921; activations and
route probabilities used CPU generator seed81221. The data file describes that
producer and includes every arm's samples. Each of nine gated rounds has four
samples **per routing case**, with cold and warm policies measured separately.
The public CLI's route/input generator differs: the command above reproduces
the operation and geometry, not those exact operands or recorded timings. A
paired public-CLI measurement remains to be qualified.

On RTX PRO 6000 Blackwell SM120, Torch2.14.1+cu132/CUDA13.2, default-tier tuning
selected micro N128 with inline CSF, two copy groups and two pipeline stages.
Core clocks used automatic policy and the host retained a +6000MHz memory
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
