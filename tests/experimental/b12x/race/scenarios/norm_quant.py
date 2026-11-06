"""Race scenarios for b12x norms (mHC, HyperConnection) and quantizers.

- ``norm_mhc``: mHC ``pre`` (RMSNorm + projection partials + Sinkhorn mixing;
  the prefill path above ``B12X_MHC_PREFILL_MIN_TOKENS``) and fused
  ``post_pre``, both with split-K projection partials and PDL launches.
- ``norm_hc_engram``: HyperConnection Engram gated mixing.
- ``quant_nvfp4_tma``: persistent BF16 -> NVFP4 TMA quantizer.
- ``quant_mxfp8``: prepared MXFP8 row quantizer (row scales plus swizzled MMA
  scales).

FP8 outputs are compared as uint8 views (bit-exact either way).
"""

from __future__ import annotations

import torch

from ._base import Scenario, _generator


def _frozen_session(device, *requests):
    from b12x.preparation import PreparationSession

    session = PreparationSession(device=device, autotune=False, compile_workers=2)
    session.__enter__()
    try:
        session.prepare(tuple(requests))
        session.freeze()
    except BaseException:
        session.__exit__(None, None, None)
        raise
    return session


def _poison(tensors):
    def reset():
        for t in tensors:
            if t.is_floating_point():
                t.fill_(float("nan"))
            else:
                t.fill_(0xA5)

    return reset


# --------------------------------------------------------------------------
# mHC


def norm_mhc(device, case) -> Scenario:
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl
    from b12x.preparation import FrozenMapping, PreparedCall
    from b12x.testing.mhc import make_inputs, pre_reference

    operation = case.get("operation", "pre")
    tokens, hidden = case.get("tokens", 256), case.get("hidden", 4096)
    options = dict(rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)
    residual, x, fn, scale, bias = make_inputs(
        tokens=tokens,
        hidden_size=hidden,
        seed=int(case.get("seed", 91_460)),
        device=device,
    )
    if operation == "pre":
        args = (residual, fn, scale, bias)
    else:
        _, previous_post, previous_comb = pre_reference(
            residual, fn, scale, bias, **options
        )
        args = (x, residual, previous_post, previous_comb, fn, scale, bias)
    invocation = dict(
        operation=operation,
        output_mode="provided",
        expanded_residual=operation == "pre",
        lagged_mix=False,
        has_norm_weight=False,
        norm_weight_dtype="bfloat16",
        has_fn_bf16=False,
        norm_eps=0.0,
        **options,
    )
    plan = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
        invocation=FrozenMapping(invocation),
    )
    entry = getattr(_impl, f"_b12x_mhc_{operation}_impl")

    def outputs():
        return dict(
            out=torch.empty((tokens, 4, hidden), dtype=torch.bfloat16, device=device),
            post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
            comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
            y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
        )

    def call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        bound = state.bind(scratch=scratch, tokens=tokens, **outputs())
        return PreparedCall(
            run=lambda: entry(*args, **options, binding=bound, _state=state),
            owners=(scratch, bound),
        )

    session = _frozen_session(
        device, plan.request(name=f"race.mhc_{operation}", prepare_call=call)
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    result = outputs()
    binding = mhc.bind(plan, scratch=scratch, tokens=tokens, **result)
    run = mhc.run_pre if operation == "pre" else mhc.run_post_pre

    def forward():
        run(*args, **options, binding=binding)

    return Scenario(
        forward=forward,
        outputs=result,
        reset=_poison(result.values()),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, binding, scratch, args],
        description=f"mHC {operation} tokens={tokens} hidden={hidden}",
    )


# --------------------------------------------------------------------------
# HyperConnection Engram mix


def norm_hc_engram(device, case) -> Scenario:
    from b12x.norm import hyperconnection as hc
    from b12x.norm.hyperconnection import _impl
    from b12x.preparation import FrozenMapping, PreparedCall

    streams, hidden = 4, case.get("hidden", 5120)
    tokens = case.get("tokens", 256)
    gen = _generator(case)
    state = (
        torch.randn((tokens, streams * hidden), generator=gen)
        .to(torch.bfloat16)
        .to(device)
    )
    projected = (
        torch.randn((tokens, (streams + 1) * hidden), generator=gen)
        .to(torch.bfloat16)
        .to(device)
    )
    weights = torch.randn(streams * hidden, generator=gen).to(device)
    mask = None
    if case.get("mask"):
        mask = (torch.rand(tokens, generator=gen) < 0.75).to(device)
    output = torch.empty_like(state)
    plan = hc.plan(
        hc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
        invocation=FrozenMapping(
            {"operation": "engram_mix", "eps": 1e-20, "token_mask": mask is not None}
        ),
    )
    request = plan.request(
        name="race.engram",
        prepare_call=lambda prepared: PreparedCall(
            run=lambda: _impl.run_engram_mix_impl(
                state,
                projected,
                weights,
                eps=1e-20,
                plan=prepared,
                out=output,
                token_mask=mask,
            )
        ),
    )
    session = _frozen_session(device, request)

    def forward():
        hc.run_engram_mix(
            state, projected, weights, eps=1e-20, plan=plan, out=output, token_mask=mask
        )

    return Scenario(
        forward=forward,
        outputs={"out": output},
        reset=lambda: output.fill_(float("nan")),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, state, projected, weights, mask],
        description=f"HyperConnection engram_mix tokens={tokens} hidden={hidden} mask={mask is not None}",
    )


# --------------------------------------------------------------------------
# Quantizers


def quant_nvfp4_tma(device, case) -> Scenario:
    from b12x.quantization.nvfp4._impl import (
        allocate_bf16_to_fp4_tma_outputs,
        compile_bf16_to_fp4_tma,
    )
    from b12x.quantization.nvfp4._tuning import TUNING, Nvfp4QuantizationQuery

    m, k = case.get("m", 2048), case.get("k", 4096)
    gen = _generator(case)
    source = (torch.randn((m, k), generator=gen) / 4).to(torch.bfloat16).to(device)
    global_scale = (
        (448.0 * 6.0 / source.float().abs().max()).reshape(1).to(torch.float32)
    )
    strategy = (
        case.get("liveness")
        or TUNING.default_config(
            Nvfp4QuantizationQuery(dtype="bfloat16", rows=m, columns=k), None
        ).liveness_strategy
    )
    outputs = allocate_bf16_to_fp4_tma_outputs(m, k, device=device)
    launch = compile_bf16_to_fp4_tma(m, k, liveness_strategy=strategy)
    packed, scales = outputs.packed_a_flat, outputs.scale_flat

    def forward():
        launch(source, global_scale, packed, scales)

    return Scenario(
        forward=forward,
        outputs={"packed": outputs.packed_a_storage, "scales": outputs.scale_storage},
        reset=_poison((outputs.packed_a_storage, outputs.scale_storage)),
        keepalive=[outputs, launch, source, global_scale],
        description=f"BF16->NVFP4 TMA quantize {m}x{k} liveness={strategy}",
    )


def quant_mxfp8(device, case) -> Scenario:
    from b12x._lib.intrinsics import as_grouped_scale_view_mx
    from b12x.preparation import PreparedCall
    from b12x.quantization import mxfp8

    rows, columns = case.get("rows", 4096), case.get("columns", 4096)
    dtype = torch.float16 if case.get("fp16") else torch.bfloat16
    gen = _generator(case)
    source = (torch.randn((rows, columns), generator=gen) / 4).to(dtype).to(device)
    values = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=device)
    scale_rows = torch.empty((rows, columns // 32), dtype=torch.uint8, device=device)
    padded_rows, padded_groups = -(-rows // 128) * 128, -(-(columns // 32) // 4) * 4
    storage = torch.full(
        (1, padded_rows, padded_groups), 127, dtype=torch.uint8, device=device
    )
    scale_mma = as_grouped_scale_view_mx(storage, rows, columns).view(torch.uint8)
    query = mxfp8.query_from_call(source, values, scale_rows, scale_mma)
    plan = mxfp8.plan(query)
    request = plan.request(
        name="race.mxfp8",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, values, scale_rows, scale_mma)
        ),
    )
    session = _frozen_session(device, request)
    raw_values = values.view(torch.uint8)

    def forward():
        mxfp8.quantize_rows(source, values, scale_rows, scale_mma, plan=plan)

    return Scenario(
        forward=forward,
        outputs={"values": raw_values, "scale_rows": scale_rows, "scale_mma": storage},
        reset=_poison((raw_values, scale_rows, storage)),
        close=lambda: session.__exit__(None, None, None),
        keepalive=[session, plan, source, values, scale_mma],
        description=f"MXFP8 row quantize {rows}x{columns} {dtype}",
    )


FAMILIES = {
    "norm_mhc": norm_mhc,
    "norm_hc_engram": norm_hc_engram,
    "quant_nvfp4_tma": quant_nvfp4_tma,
    "quant_mxfp8": quant_mxfp8,
}

CASES = {
    "norm_mhc": [
        {"operation": "pre", "tokens": 256},
        {"operation": "post_pre", "tokens": 64},
    ],
    "norm_hc_engram": [
        {"tokens": 256},
        {"tokens": 129, "mask": True},
    ],
    "quant_nvfp4_tma": [
        {"m": 2048, "k": 4096},
        {"m": 128, "k": 7168},
    ],
    "quant_mxfp8": [
        {"rows": 4096, "columns": 4096},
        {"rows": 300, "columns": 7168, "fp16": True},
    ],
}

FUZZ = {
    "norm_mhc": lambda rng: {
        "operation": rng.choice(("pre", "post_pre")),
        "tokens": rng.choice((1, 3, 8, 17, 64, 96, 128, 300)),
        "hidden": rng.choice((4096, 5120, 7168)),
        "seed": rng.getrandbits(31),
    },
    "norm_hc_engram": lambda rng: {
        "tokens": rng.randint(1, 512),
        "hidden": 256 * rng.randint(4, 28),
        "mask": rng.random() < 0.5,
        "seed": rng.getrandbits(31),
    },
    "quant_nvfp4_tma": lambda rng: {
        "m": 128 * rng.randint(1, 24),
        "k": 128 * rng.randint(1, 64),
        "liveness": rng.choice(("retain", "packed", None)),
        "seed": rng.getrandbits(31),
    },
    "quant_mxfp8": lambda rng: {
        "rows": rng.randint(1, 4096),
        "columns": 128 * rng.randint(1, 64),
        "fp16": rng.random() < 0.3,
        "seed": rng.getrandbits(31),
    },
}
