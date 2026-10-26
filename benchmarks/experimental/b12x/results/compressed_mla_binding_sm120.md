# Prepared compressed MLA scratch binding

Run from the repository root with FlashInfer's b12x dependencies installed.
This standalone benchmark needs an idle SM120/SM121 GPU; it creates synthetic
BF16 query tensors and packed DeepSeek V4.1 caches, without a checkpoint,
Flare installation, resident model or server endpoint.

```sh
PYTHONPATH=flashinfer/experimental:. python benchmarks/experimental/b12x/benchmark_mla_binding.py \
  --device 0 --rows 8 --heads 16 --verify-width 8 \
  --contexts 4096 8192 --ratios 1 2 --seed 96741 \
  --compile-workers 16 --rounds 9 --samples 4 --output mla.json
```

Run the same command in separate control/patch processes in A/B/B/A order,
using a distinct output filename for each arm. The baseline does not contain
this runner: keep the script from the fill-removal checkout fixed and select
library source through `PYTHONPATH`, for example:

```sh
CONTROL=/path/to/flashinfer-control
BENCHMARK=/path/to/flashinfer-fill-removal/benchmarks/experimental/b12x/benchmark_mla_binding.py
PYTHONPATH="$CONTROL/flashinfer/experimental:$CONTROL" python "$BENCHMARK" \
  --device 0 --rows 8 --heads 16 --verify-width 8 \
  --contexts 4096 8192 --ratios 1 2 --seed 96741 \
  --compile-workers 16 --rounds 9 --samples 4 --output control-a.json
```

 Plans use normal default-tier
autotuning within the declared BF16 arithmetic contract; no split count, tile
or implementation is pinned. Both cold and warm policies use one stream-gated
interval around the captured call. Binding occurs **inside capture**, followed
by logical page mapping, query/cache conversion, attention and merge, so any
bind-time fill kernels are included. Compilation, fixture creation and
reference checks happen outside timing. The benchmark checks the packed-cache
numerical reference and three poisoned-scratch replays before measurement.

The fixture and gated measurement loop are the direct-FI form of the measured
component wrapper. The portable entrypoint has CPU help/syntax/source review;
its GPU execution has not been rerun. Source review preserves the emitted
bind+attention operation and input generator, but its direct-FI preparation
replaces the wrapper's preparation owner. Do not treat it as an additional GPU
qualification until run on the intended hardware.

## Recorded conditions and results

[Raw per-round samples and selected configurations](compressed_mla_binding_sm120.json)
cover rows8, heads16, head dimension512, 128 window and 512 indexed entries,
window-page size136 and page-table width8192. Cache compression ratio1/2 selects
indexed page size128/64. Queries/cache values are normalized BF16 random inputs,
with seed96741 + rows×100 + context. Four samples in each of nine gated rounds
produce a median round mean; the table averages the two matching ABBA-arm
medians. Every arm selects BF16, heads16, ten splits and chunk size64 through
normal tuning.

RTX PRO 6000 Blackwell SM120, Torch2.14.1+cu132/CUDA13.2; automatic core policy
and existing +6000MHz memory offset, without a benchmark clock lock. Resident
services were idle, with request counters/GPU owners unchanged before/after.

| Context / ratio | Control warm | Fill removal warm | Control cold | Fill removal cold |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 1 | 10.4800us | 9.1120us | 11.8560us | 10.8920us |
| 4096 / 2 | 10.5520us | 9.1760us | 11.8440us | 10.9200us |
| 8192 / 1 | 10.5800us | 9.0680us | 12.4040us | 10.8920us |
| 8192 / 2 | 10.5040us | 9.0000us | 12.6560us | 10.8800us |

All four fixtures preserve bitwise output across the four arms and pass the
packed-cache numerical oracle and poisoned repeated replay. Cold results have
more control-arm variation; retain the raw samples when assessing them. The
110-case attention/scratch qualification separately covers BF16/FP8 references,
partial/inactive rows, capture/replay and mapped offsets above2GiB. No full-model
or HTTP gain is claimed; SM121 execution remains unqualified. This changes an
existing explicitly selected experimental b12x backend, with no core dispatch,
feature gate or AOT registration changes.
