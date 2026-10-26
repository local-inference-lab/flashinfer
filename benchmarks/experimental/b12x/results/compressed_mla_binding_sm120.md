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
component wrapper. The entrypoint is GPU-qualified by the four-case paired
measurement below, with the reference and three poisoned-scratch replay checks
preserved.

## Portable entrypoint measurements

[Raw portable measurements](compressed_mla_binding_portable_sm120.json) use this
exact entrypoint with public main `344e4837` and patch revision `6351ddac0` in A/B/B/A order.
Each policy has nine gated rounds of four samples per case. The reported values
average the two matching arm medians. All numerical and poisoned-replay checks
pass, with normal default-tier tuning and no plan overrides.

| Context / ratio | Control warm | Removal warm | Control cold | Removal cold |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 1 | 10.8440us | 8.9440us | 13.0080us | 11.4160us |
| 4096 / 2 | 10.8080us | 9.0320us | 13.0440us | 11.0000us |
| 8192 / 1 | 10.8880us | 9.0200us | 13.0480us | 11.3640us |
| 8192 / 2 | 10.8360us | 9.1800us | 13.0240us | 11.2440us |

At 4096 both arms select 54 splits/chunk 12, so those rows compare matched plans.
At 8192 the control selects 10 splits/chunk 64 and fill removal selects 54/chunk 12;
those before/after numbers include normal plan selection and are not an isolated
fill-cost attribution. All plans retain BF16 arithmetic and 16 heads. Raw records
include every selected config and event interval.

RTX PRO 6000 Blackwell SM120, Torch 2.14.1+cu132/CUDA 13.2; automatic core policy and
existing +6000 MHz memory offset without a benchmark clock lock. Resident services
stay idle with counters/GPU owners unchanged. No full-model/HTTP gain is claimed;
SM121 remains unqualified.

## Historical component-wrapper measurements

[Raw per-round samples and selected configurations](compressed_mla_binding_sm120.json)
cover rows8, heads16, head dimension512, 128 window and 512 indexed entries,
window-page size136 and page-table width8192. Cache compression ratio1/2 selects
indexed page size128/64. Queries/cache values are normalized BF16 random inputs,
with seed96741 + rows×100 + context. Four samples in each of nine gated rounds
produce a median round mean; the table averages the two matching ABBA-arm
medians. Every arm selects BF16, heads16, ten splits and chunk size64 through
normal tuning.

RTX PRO 6000 Blackwell SM120, Torch 2.14.1+cu132/CUDA 13.2; automatic core policy
and existing +6000 MHz memory offset, without a benchmark clock lock. Resident
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
