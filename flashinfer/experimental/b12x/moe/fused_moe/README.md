# W4A16 with NVFP4 activation prefill

This experimental SM120 path reuses ModelOpt NVFP4 W4A16 weights for
NVFP4 activation GEMMs during prefill. Activation quantization changes model
outputs and requires model quality validation before deployment. The default
remains W4A16.

Set `B12X_W4A16_A4_PREFILL_MIN_TOKENS` to a positive integer before preparing
weights and execution plans to enable this path. Preparation requires finite,
positive input and intermediate calibration scales; absent scales and a pair
of all-one placeholder scales retain W4A16. The shared input global scale is
the minimum across experts. Intermediate scales remain per expert. Ordinary
A4 weight preparation retains its per-expert scales.

Supported weights use packed ModelOpt NVFP4 with ordinary or compressed
NVFP4-CSF block scales, BF16 input/output, SiLU gating, hidden size divisible
by 256, and intermediate size divisible by 128. A positive `swiglu_limit` uses
the W4A16 activation contract: clamp the BF16 GEMM gate above the limit and the
up input to the symmetric interval, round SiLU and up to BF16, multiply, then
round the intermediate to BF16 before quantization. Expert maps, input router
weighting, activation-amax collection, and unsupported layouts retain W4A16.

The execution binding accepts `a4_prefill`:

- `None` applies the prepared token threshold.
- `True` selects prepared A4 launches even below that threshold.
- `False` selects W4A16.

`ActivationSpec.a16_max_tokens` remains an inclusive A16 cutoff for every
choice. Callers that know request boundaries must select prefill semantically;
a token count alone cannot distinguish prefill from batched decode. Each call
must fit its prepared capacity and caller-owned scratch buffers.

`B12X_W4A16_A4_PREFILL_TERMS=1` uses one NVFP4 activation plane. A value of `2`
adds an independently quantized residual plane. `B12X_W4A16_A4_PREFILL_WARPS`
accepts `8` (default) or `16`. These controls and the threshold are immutable
execution-plan inputs and are part of tuning query schema 23. Changing the
environment requires preparing another execution plan. Kernel compilation
retains device-owned callables; replay performs no compilation or tensor
allocation. A4 always applies router weights in the FP32 final sum. Set
`B12X_W4A16_FP32_TOPK_WEIGHTS=1` before importing B12X when comparing with an
A16 baseline that also uses FP32 router weighting.

Run the synthetic quantization-contract and changed-input graph checks on an
SM120 GPU from the FlashInfer repository:

```sh
PYTHONPATH="$PWD:$PWD/flashinfer/experimental/b12x/_compat" \
  .venv/bin/python -m pytest \
  tests/experimental/b12x/moe/test_w4a16_a4_prefill.py -v
```

The two-device ownership check requires two visible SM120 GPUs. These checks
verify the specified lossy arithmetic and replay behavior; they do not measure
model accuracy or serving throughput.
