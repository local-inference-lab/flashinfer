"""Race scenarios for the other b12x GEMM engines and launch paths.

``gemm.py`` covers packed MXFP8. This module covers the remaining paths of the
shared dense engine (``DenseGemmKernel``): packed NVFP4 (quantized and A16,
including A16 split-K with its separate fixed-order reduce), packed MXFP4,
raw NVFP4 operands on both load paths (TMA and the non-TMA ``cpasync`` path,
which only NVFP4 supports), block-FP8 linear, the IQ2/Q8_0 block codecs and
the two-GEMM WO projection. It also covers the GEMMs that use other engines:
the MXFP8 BMM, the fused MLA query projection, and the BF16 projection
backends.

Cases use production configurations, including FP8 decode split-K with BF16
atomics (``B12X_DENSE_SPLITK_TURBO=1``, the default policy for FP8, MXFP8 and
block-FP8 GEMMs with at most 8 rows): block-FP8 linear, tensor-FP8 linear and
WO projection decode. The ``torch`` (cuBLAS) BF16 projection backend is not
covered because it is not a b12x kernel.
"""

from __future__ import annotations

import torch

from ._base import Scenario, _generator, _rows_for_waves


def _frozen_session(device, *requests):
    """Prepare ``requests`` and freeze; the caller closes it (see gemm.py)."""
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


def _close(session):
    return lambda: session.__exit__(None, None, None)


def _randn(gen, shape, scale, dtype, device):
    return (torch.randn(shape, generator=gen) * scale).to(dtype).to(device)


# --------------------------------------------------------------------------
# Packed BF16-source weights (NVFP4, MXFP4, IQ2_XS, IQ2_XXS, Q8_0)


def _a16_override(case):
    """An explicit A16 launch geometry, or None for the production default."""
    from b12x.gemm.blockscaled import BlockscaledConfig

    if case.get("split_k") is None:
        return None
    return BlockscaledConfig(
        mode="a16",
        tile_m=case.get("tile_m"),
        tile_n=case.get("tile_n", 64),
        tile_k=case.get("tile_k", 64),
        split_k=case["split_k"],
    )


def _packed_scenario(device, case, weight, source, *, label, activation_scale=None):
    from b12x.gemm import blockscaled
    from b12x.gemm.blockscaled._a16 import _weight_parts
    from b12x.preparation import PreparedCall

    mode = case.get("mode", "quantized")
    rows = source.shape[0]
    out = torch.empty((rows, weight.out_features), dtype=torch.bfloat16, device=device)
    query = blockscaled.query_from_call(
        source,
        weight,
        activation_mode=mode,
        activation_global_scale=activation_scale,
        out=out,
    )
    plan = blockscaled.plan(
        query, override=_a16_override(case) if mode == "a16" else None
    )
    values, scales, gain, _ = _weight_parts(weight)

    def call(state):
        return PreparedCall(
            run=lambda: state.run(
                source, values, scales, gain, activation_scale=activation_scale, out=out
            )
        )

    session = _frozen_session(
        device, plan.request(name=f"race.{label}", prepare_call=call)
    )
    options = (
        {}
        if activation_scale is None
        else {"activation_global_scale": activation_scale}
    )

    def forward():
        blockscaled.mm(source, weight, out, plan=plan, **options)

    split = "" if case.get("split_k") is None else f" split_k={case['split_k']}"
    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, weight, source, activation_scale],
        description=f"packed {label} {mode}{split} {rows}x{weight.in_features}->{weight.out_features}",
    )


def gemm_packed_nvfp4(device, case) -> Scenario:
    """Packed NVFP4 weight with BF16 activations: quantized (activation
    quantization into workspace, then the dense engine) or A16 (inline weight
    dequantization; ``split_k`` > 1 adds the fixed-order partial reduce)."""
    from b12x.gemm import blockscaled
    from b12x._lib.intrinsics import swizzle_block_scale

    n, k = case.get("n", 4096), case.get("k", 2048)
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    codes = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    local = (torch.rand((n, k // 16), generator=gen) * 2 + 0.0625).to(
        torch.float8_e4m3fn
    )
    weight = blockscaled.pack_weight(
        codes.to(device),
        swizzle_block_scale(local.to(device)),
        recipe="nvfp4",
        global_scale=torch.tensor([0.125], dtype=torch.float32, device=device),
    )
    source = _randn(gen, (rows, k), 0.5, torch.bfloat16, device)
    activation_scale = None
    if case.get("mode", "quantized") == "quantized":
        # The activation quantizer multiplier: map the source amax to E4M3 x E2M1 max.
        amax = source.float().abs().max()
        activation_scale = (448.0 * 6.0 / amax).reshape(1).to(torch.float32)
    return _packed_scenario(
        device, case, weight, source, label="nvfp4", activation_scale=activation_scale
    )


def gemm_packed_mxfp4(device, case) -> Scenario:
    """Packed MXFP4 weight; BF16 rows are quantized to MXFP4 in workspace."""
    from b12x.gemm import blockscaled
    from b12x._lib.intrinsics import swizzle_block_scale

    n, k = case.get("n", 4096), case.get("k", 2048)
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    codes = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
    exponents = torch.randint(118, 126, (n, k // 32), generator=gen, dtype=torch.uint8)
    weight = blockscaled.pack_weight(
        codes.to(device), swizzle_block_scale(exponents.to(device)), recipe="mxfp4"
    )
    source = _randn(gen, (rows, k), 0.5, torch.bfloat16, device)
    return _packed_scenario(
        device, dict(case, mode="quantized"), weight, source, label="mxfp4"
    )


def _codec_blocks(gen, n, k, codec):
    """Random IQ2_XS/IQ2_XXS/Q8_0 payloads with finite FP16 block scales."""
    block = 32 if codec == "q8_0" else 256
    width = {"q8_0": 34, "iq2_xxs": 66}.get(codec, 74)
    raw = torch.randint(
        0, 256, (n, k // block, width), generator=gen, dtype=torch.uint8
    )
    base = (torch.randn((n, k // block, 1), generator=gen) / 128).half()
    raw[..., :2] = base.contiguous().view(torch.uint8)
    return raw


def gemm_block_codec(device, case) -> Scenario:
    """IQ2_XS / IQ2_XXS / Q8_0 weights: A16 with inline codebook decoding."""
    from b12x.gemm import blockscaled

    codec = case.get("codec", "iq2_xs")
    n, k = case.get("n", 4096), case.get("k", 2048)
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    weight = blockscaled.pack_weight(
        _codec_blocks(gen, n, k, codec).to(device), recipe=codec
    )
    source = _randn(gen, (rows, k), 1.0, torch.bfloat16, device)
    return _packed_scenario(device, dict(case, mode="a16"), weight, source, label=codec)


# --------------------------------------------------------------------------
# Raw NVFP4 operands: the only recipe with a non-TMA (cp.async) load path


def _nvfp4_operand(gen, rows, k, device):
    from b12x._lib.intrinsics import quantize_grouped_nvfp4_torch

    source = _randn(gen, (1, rows, k), 0.25, torch.bfloat16, device)
    global_scale = (448.0 * 6.0 / source.float().abs().max()).reshape(1)
    counts = torch.full((1,), rows, dtype=torch.int32, device=device)
    packed, scales = quantize_grouped_nvfp4_torch(source, counts, global_scale)
    return (packed, scales), global_scale


def gemm_dense_nvfp4(device, case) -> Scenario:
    """Raw NVFP4 dense GEMM with an explicit load path and tile.

    ``load_path="cpasync"`` loads A/B/scales with cp.async instead of TMA. The
    dense GEMM issues its stage-reuse proxy fence only before TMA refills;
    cp.async refills go through the generic proxy and rely on the pipeline
    barrier alone, so this path is covered separately.
    """
    from b12x.gemm import blockscaled
    from b12x.preparation import PreparedCall

    n, k = case.get("n", 4096), case.get("k", 2048)
    tile = tuple(case.get("tile", (64, 128)))
    swap = bool(case.get("swap_ab", False))
    rows = case.get("rows") or _rows_for_waves(
        device,
        n,
        case.get("waves", 3),
        tile_m=tile[1] if swap else tile[0],
        tile_n=tile[0] if swap else tile[1],
    )
    gen = _generator(case)
    lhs, lhs_scale = _nvfp4_operand(gen, rows, k, device)
    rhs, rhs_scale = _nvfp4_operand(gen, n, k, device)
    alpha = (1.0 / (lhs_scale * rhs_scale)).reshape(1).to(torch.float32)
    out = torch.empty((rows, n, 1), dtype=torch.bfloat16, device=device)
    query = blockscaled.query_from_call(
        lhs,
        rhs,
        out=out,
        alpha=alpha,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        load_path=case.get("load_path", "tma"),
        mma_tiler_mn=tile,
        swap_ab=swap,
    )
    plan = blockscaled.plan(query)

    def call(state):
        return PreparedCall(
            run=lambda: state.run(*lhs, *rhs, alpha, out, None, torch.bfloat16, None)
        )

    session = _frozen_session(
        device, plan.request(name="race.dense_nvfp4", prepare_call=call)
    )

    def forward():
        blockscaled.mm(lhs, rhs, out, plan=plan, alpha=alpha, out_dtype=torch.bfloat16)

    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, lhs, rhs, alpha],
        description=(
            f"dense NVFP4 {case.get('load_path', 'tma')} tile={tile} swap_ab={swap} "
            f"{rows}x{k}->{n}"
        ),
    )


# --------------------------------------------------------------------------
# Block-FP8 linear and WO projection (dense engine behind fused quantizers)


def gemm_block_fp8_linear(device, case) -> Scenario:
    """Block-FP8 linear: MXFP8 row quantization, then the block-FP8 dense GEMM
    with a bias epilogue. Up to 8 rows the policy may split K with atomics."""
    from b12x.gemm import block_fp8_linear as bfl
    from b12x.gemm._shared.wo_mxfp8 import empty_dense_gemm_mnl_view
    from b12x.preparation import PreparedCall

    n, k = case.get("n", 4096), case.get("k", 2048)
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    source = _randn(gen, (rows, k), 0.25, torch.bfloat16, device)
    weight = _randn(gen, (n, k), 0.25, torch.float8_e4m3fn, device)
    weight_scale = (torch.rand((n // 128, k // 128), generator=gen) * 0.2 + 0.05).to(
        device
    )
    packed = bfl.pack_weight(weight, weight_scale)
    output = empty_dense_gemm_mnl_view(rows, n, 1, device=device, dtype=torch.bfloat16)
    bias = _randn(gen, (n,), 0.125, torch.bfloat16, device)
    caps = bfl.Caps(
        device=device,
        max_tokens=rows,
        in_features=k,
        out_features=n,
        source_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    plan = bfl.plan(caps)
    owned = {}

    def call(state):
        (spec,) = state.scratch.scratch_specs()
        owned["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            scratch=owned["scratch"],
            source=source,
            packed_weight=packed,
            output=output,
            bias=bias,
        )
        return PreparedCall(run=lambda: state.run_binding(binding), output=output)

    session = _frozen_session(
        device, plan.request(name="race.block_fp8", prepare_call=call)
    )
    binding = bfl.bind(
        plan,
        scratch=owned["scratch"],
        source=source,
        packed_weight=packed,
        output=output,
        bias=bias,
    )

    def forward():
        bfl.run(binding=binding)

    return Scenario(
        forward=forward,
        outputs={"out": output},
        reset=lambda: output.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, binding, owned, packed, source, bias],
        description=f"block-FP8 linear {rows}x{k}->{n}",
    )


def gemm_tensor_fp8_linear(device, case) -> Scenario:
    """Tensor-scaled FP8 linear (prequantized E4M3 source, plain-FP8 dense
    engine). The API is functional, so ``forward`` copies its graph-pool
    result into a fixed output."""
    from b12x.gemm import tensor_fp8_linear as tfl
    from b12x.preparation import PreparedCall

    n, k = case.get("n", 4096), case.get("k", 2048)
    rows = case.get("rows") or _rows_for_waves(device, n, case.get("waves", 3))
    gen = _generator(case)
    source = (
        _randn(gen, (rows, k), 1.0, torch.float32, device)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    weight = (
        _randn(gen, (n, k), 1.0, torch.float32, device)
        .clamp(-4, 4)
        .to(torch.float8_e4m3fn)
    )
    output_scale = torch.tensor([0.0002], dtype=torch.float32, device=device)
    packed = tfl.pack_weight(weight, output_scale)
    plan = tfl.plan(tfl.query_from_call(source, packed))

    def call(state):
        return PreparedCall(
            run=lambda: state.run_tensor_fp8(
                source,
                packed.values,
                packed.scale_mma,
                packed.block_scale,
                packed.output_scale,
                out_dtype=torch.bfloat16,
                stream=None,
            )
        )

    session = _frozen_session(
        device, plan.request(name="race.tensor_fp8", prepare_call=call)
    )
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)

    def forward():
        out.copy_(tfl.mm(source, packed, plan=plan))

    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, packed, source],
        description=f"tensor-FP8 linear {rows}x{k}->{n}",
    )


def gemm_wo_projection(device, case) -> Scenario:
    """Grouped WO-A MXFP8 GEMM, MXFP8 requantization, then the WO-B GEMM.

    Decode rows (1-8) of the 2x4096/1024/5120 geometry take the fused WO-B
    path with its split-K BF16 atomic reduction.
    """
    from b12x.gemm import wo_projection as wo
    from b12x.gemm._shared.wo_mxfp8 import quantize_wo_projection_weights_mxfp8_torch
    from b12x.preparation import PreparedCall

    groups, width = case.get("groups", 4), case.get("group_width", 4096)
    rank, hidden = case.get("rank", 1024), case.get("hidden", 4096)
    rows = case.get("rows") or _rows_for_waves(
        device, rank * groups, case.get("waves", 3)
    )
    gen = _generator(case)
    source = _randn(gen, (rows, groups, width), 0.125, torch.bfloat16, device)
    weights = quantize_wo_projection_weights_mxfp8_torch(
        _randn(gen, (groups, rank, width), width**-0.5, torch.bfloat16, device),
        _randn(
            gen,
            (hidden, groups * rank),
            (groups * rank) ** -0.5,
            torch.bfloat16,
            device,
        ),
    )
    caps = wo.Caps(
        device=device,
        max_tokens=rows,
        groups=groups,
        group_width=width,
        rank=rank,
        hidden=hidden,
    )
    plan = wo.plan(caps)
    specs = []

    def call(state):
        specs[:] = state._scratch_state.scratch_specs()
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device) for s in specs
        )
        binding = state.bind(scratch=scratch, source_tgd=source, weights=weights)
        return PreparedCall(run=lambda: state.run(binding))

    session = _frozen_session(device, plan.request(name="race.wo", prepare_call=call))
    scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in specs)
    binding = wo.bind(plan, scratch=scratch, source_tgd=source, weights=weights)
    output = binding.output

    def forward():
        wo.run(binding=binding, plan=plan)

    return Scenario(
        forward=forward,
        outputs={"out": output},
        reset=lambda: output.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, binding, scratch, weights, source],
        description=f"WO projection {rows}x{groups}x{width} rank {rank} -> {hidden}",
    )


# --------------------------------------------------------------------------
# MXFP8 BMM and the fused MLA query projection (their own cp.async kernels)


def _mxfp8_pack(gen, batch, rows, k, device):
    values = _randn(gen, (batch * rows, k), 0.1, torch.float8_e4m3fn, device)
    scales = torch.randint(
        118, 132, (batch * rows, k // 32), generator=gen, dtype=torch.uint8
    )
    return values, scales.to(device)


def gemm_bmm(device, case) -> Scenario:
    """BF16 x rowwise-MXFP8 batched GEMM on its qualified geometries."""
    from b12x import gemm
    from b12x.gemm._bmm._tuning import BmmQuery
    from b12x.preparation import PreparedCall

    major, batch, m = case.get("major", "n"), case.get("batch", 16), case.get("m", 65)
    k, n = (192, 512) if major == "n" else (512, 256)
    gen = _generator(case)
    # Physical storage of both layouts: N-major rows follow K, K-major rows follow N.
    physical_rows, physical_k = (k, n) if major == "n" else (n, k)
    values, scales = _mxfp8_pack(gen, batch, physical_rows, physical_k, device)
    rhs = (
        values.view(batch, physical_rows, physical_k),
        scales.view(batch, physical_rows, physical_k // 32),
    )
    lhs = _randn(gen, (batch, m, k), 1.0, torch.bfloat16, device)
    out = torch.empty((batch, m, n), dtype=torch.bfloat16, device=device)
    plan = gemm.plan_bmm(
        BmmQuery(
            batch=batch,
            max_rows=m,
            in_features=k,
            out_features=n,
            b_major=major,
            sf_axis=major,
        )
    )
    spec = dict(
        a_dtype="bfloat16",
        b_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        b_major=major,
        sf_axis=major,
    )

    def call(state):
        return PreparedCall(
            run=lambda: state.run(lhs, rhs, out, b_major=major, sf_axis=major)
        )

    session = _frozen_session(device, plan.request(name="race.bmm", prepare_call=call))

    def forward():
        gemm.bmm(lhs, rhs, out, plan=plan, **spec)

    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, lhs, rhs],
        description=f"MXFP8 BMM {major}-major B={batch} M={m} K={k} N={n}",
    )


def gemm_mla_query_projection(device, case) -> Scenario:
    """Absorbed MLA query projection (MXFP8 BMM with the fused RoPE-concat and
    optional FP8 output epilogue)."""
    from b12x.gemm import mla_query_projection as mla
    from b12x.gemm.mla_query_projection._tuning import ProjectionQuery
    from b12x.preparation import PreparedCall

    heads, m = case.get("heads", 16), case.get("m", 32)
    fp8 = bool(case.get("fp8_out", False))
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    gen = _generator(case)
    values, scales = _mxfp8_pack(gen, heads, 192, 512, device)
    weight = (values.view(heads, 192, 512), scales.view(heads, 192, 512 // 32))
    q_nope = _randn(gen, (heads, m, 192), 1.0, torch.bfloat16, device)
    q_pe = _randn(gen, (m, heads, 576), 1.0, torch.bfloat16, device)[..., 512:]
    q_scale = torch.tensor([0.037], dtype=torch.float32, device=device) if fp8 else None
    out = torch.empty((m, heads, 576), dtype=dtype, device=device)
    plan = mla.plan(
        ProjectionQuery(
            heads=heads,
            max_rows=m,
            weight_format="mxfp8",
            output_dtype=str(dtype).removeprefix("torch."),
            b_major="n",
            sf_axis="n",
        )
    )

    def call(state):
        return PreparedCall(
            run=lambda: state.run(q_nope, weight, q_pe, out, q_scale=q_scale)
        )

    session = _frozen_session(device, plan.request(name="race.mla", prepare_call=call))

    def forward():
        mla.run(q_nope, weight, q_pe, out, plan=plan, q_scale=q_scale)

    raw = out.view(torch.uint8) if fp8 else out
    return Scenario(
        forward=forward,
        outputs={"out": raw},
        reset=lambda: raw.fill_(0x7F if fp8 else float("nan")),
        close=_close(session),
        keepalive=[session, plan, weight, q_nope, q_pe, q_scale],
        description=f"MLA query projection heads={heads} M={m} out={dtype}",
    )


# --------------------------------------------------------------------------
# BF16 projections (SIMT GEMV, MMA, tensor-core GEMV, TMA-pipelined prefill)


def gemm_bf16_gemv(device, case) -> Scenario:
    """Unquantized BF16 projection with a pinned b12x backend."""
    from b12x.gemm import bf16_gemv
    from b12x.preparation import PreparedCall

    backend = case.get("backend", "simt")
    rows, n, k = case.get("rows", 4), case.get("n", 8192), case.get("k", 4096)
    gen = _generator(case)
    x = _randn(gen, (rows, k), 0.125, torch.bfloat16, device)
    weight = _randn(gen, (n, k), 0.125, torch.bfloat16, device)
    bias = _randn(gen, (n,), 0.5, torch.bfloat16, device) if case.get("bias") else None
    out = torch.empty((rows, n), dtype=torch.bfloat16, device=device)
    config = {"backend": backend}
    if backend == "simt":
        config.update(
            rows_per_tile=case.get("rows_per_tile", 8),
            block_threads=case.get("block_threads", 128),
        )
    plan = bf16_gemv.plan(
        bf16_gemv.query_from_call(x, weight, out=out, bias=bias),
        override=bf16_gemv.GemvConfig(**config),
    )

    def call(state):
        return PreparedCall(run=lambda: state.run(x, weight, out=out, bias=bias))

    session = _frozen_session(device, plan.request(name="race.gemv", prepare_call=call))

    def forward():
        bf16_gemv.mm(x, weight, out=out, bias=bias, plan=plan)

    return Scenario(
        forward=forward,
        outputs={"out": out},
        reset=lambda: out.fill_(float("nan")),
        close=_close(session),
        keepalive=[session, plan, x, weight, bias],
        description=f"BF16 projection {backend} {rows}x{k}->{n}"
        + (" +bias" if bias is not None else ""),
    )


FAMILIES = {
    "gemm_packed_nvfp4": gemm_packed_nvfp4,
    "gemm_packed_mxfp4": gemm_packed_mxfp4,
    "gemm_block_codec": gemm_block_codec,
    "gemm_dense_nvfp4": gemm_dense_nvfp4,
    "gemm_block_fp8_linear": gemm_block_fp8_linear,
    "gemm_tensor_fp8_linear": gemm_tensor_fp8_linear,
    "gemm_wo_projection": gemm_wo_projection,
    "gemm_bmm": gemm_bmm,
    "gemm_mla_query_projection": gemm_mla_query_projection,
    "gemm_bf16_gemv": gemm_bf16_gemv,
}

CASES = {
    "gemm_packed_nvfp4": [
        {"n": 4096, "k": 2048, "mode": "quantized"},
        {"n": 2560, "k": 2560, "mode": "a16"},
        # A16 split-K: partial slices into workspace, then the fixed-order reduce.
        {
            "rows": 64,
            "n": 2048,
            "k": 4096,
            "mode": "a16",
            "tile_m": 16,
            "tile_n": 64,
            "tile_k": 64,
            "split_k": 4,
        },
    ],
    "gemm_packed_mxfp4": [
        {"n": 4096, "k": 2048},
        {"rows": 384, "n": 6144, "k": 2560},
    ],
    "gemm_block_codec": [
        {"codec": "iq2_xs", "n": 4096, "k": 2048},
        {"codec": "q8_0", "n": 4096, "k": 2048},
        {
            "codec": "iq2_xxs",
            "rows": 32,
            "n": 4096,
            "k": 4096,
            "tile_m": 16,
            "tile_n": 64,
            "tile_k": 128,
            "split_k": 4,
        },
    ],
    "gemm_dense_nvfp4": [
        {"load_path": "cpasync", "n": 4096, "k": 2048},
        {"load_path": "cpasync", "n": 2048, "k": 4096, "tile": (128, 128)},
        {"load_path": "tma", "n": 4096, "k": 2048},
    ],
    "gemm_block_fp8_linear": [
        {"n": 4096, "k": 2048},
        {"rows": 768, "n": 4096, "k": 8192},
        # Decode: split-K slices reduce with BF16 atomics.
        {"rows": 4, "n": 8192, "k": 4096},
    ],
    "gemm_tensor_fp8_linear": [
        {"n": 4096, "k": 2048},
        # Decode: split-K slices reduce with BF16 atomics.
        {"rows": 8, "n": 8192, "k": 4096},
    ],
    "gemm_wo_projection": [
        {"rows": 1024},
        {"rows": 640, "groups": 2, "rank": 1024, "hidden": 5120},
        # Decode tile: fused WO-B with split-K BF16 atomics.
        {"rows": 6, "groups": 2, "rank": 1024, "hidden": 5120},
    ],
    "gemm_bmm": [
        {"major": "n", "batch": 32, "m": 300},
        {"major": "k", "batch": 16, "m": 513},
    ],
    "gemm_mla_query_projection": [
        # The qualified geometry caps this at 64 CTAs (heads x N/128).
        {"heads": 16, "m": 32},
        {"heads": 16, "m": 24, "fp8_out": True},
    ],
    "gemm_bf16_gemv": [
        # Bf16PrefillKernel's consumers read a stage with ldmatrix (generic
        # proxy) before its producer refills it with TMA (async proxy); these
        # cases depend on the fence.proxy.async between them
        # (local-inference-lab/flashinfer#25). Without it, contention corrupts
        # 345 and 208 of 2000 replays on GB10; warp delays do not expose it.
        {"backend": "prefill", "rows": 1536, "n": 1024, "k": 5120},
        {"backend": "prefill", "rows": 2048, "n": 384, "k": 5120},
        {"backend": "mma", "rows": 384, "n": 4096, "k": 2048, "bias": True},
        {"backend": "tc", "rows": 8, "n": 8192, "k": 2048},
        {"backend": "simt", "rows": 4, "n": 8192, "k": 4096},
    ],
}


def _seed(rng):
    return rng.getrandbits(31)


def _fuzz_packed_nvfp4(rng):
    case = {
        "rows": rng.choice((17, 32, 64, 96, 128, 200, 256, None)),
        "n": 128 * rng.randint(4, 48),
        "k": 128 * rng.randint(4, 32),
        "mode": rng.choice(("quantized", "a16")),
        "seed": _seed(rng),
    }
    if case["mode"] == "a16" and rng.random() < 0.5:
        case.update(
            tile_m=16,
            tile_n=rng.choice((64, 128)),
            tile_k=64,
            split_k=rng.choice((2, 4, 8)),
            rows=rng.choice((16, 32, 64)),
        )
    return case


def _fuzz_packed_mxfp4(rng):
    return {
        "rows": rng.choice((17, 64, 96, 128, 256, None)),
        "n": 128 * rng.randint(4, 48),
        "k": 256 * rng.randint(2, 16),
        "seed": _seed(rng),
    }


def _fuzz_block_codec(rng):
    codec = rng.choice(("iq2_xs", "iq2_xxs", "q8_0"))
    case = {
        "codec": codec,
        "rows": rng.choice((17, 32, 65, 128, None)),
        "n": 128 * rng.randint(4, 40),
        "k": 256 * rng.randint(2, 16),
        "seed": _seed(rng),
    }
    if rng.random() < 0.4:
        case.update(
            rows=rng.choice((16, 32, 64)),
            tile_m=16,
            tile_n=rng.choice((64, 128)),
            tile_k=rng.choice((64, 128, 256)),
            split_k=rng.choice((2, 4, 8)),
        )
    return case


def _fuzz_dense_nvfp4(rng):
    load_path = rng.choice(("cpasync", "tma"))
    tile, swap = rng.choice(
        (
            ((64, 128), False),
            ((128, 128), False),
            ((64, 64), False),
            ((64, 32), True),
            ((64, 16), True),
        )
    )
    return {
        "load_path": load_path,
        "tile": tile,
        "swap_ab": swap,
        "rows": rng.choice((32, 64, 96, 200, 256, None)),
        "n": 128 * rng.randint(2, 40),
        "k": 128 * rng.randint(2, 32),
        "seed": _seed(rng),
    }


def _fuzz_block_fp8_linear(rng):
    return {
        "rows": rng.choice((1, 2, 4, 8, 9, 17, 64, 96, 128, 256, None)),
        "n": 128 * rng.randint(2, 40),
        "k": 128 * rng.randint(2, 64),
        "seed": _seed(rng),
    }


def _fuzz_tensor_fp8_linear(rng):
    return {
        "rows": rng.choice((1, 3, 8, 17, 64, 200, None)),
        "n": 64 * rng.randint(2, 128),
        "k": 128 * rng.randint(1, 48),
        "seed": _seed(rng),
    }


def _fuzz_wo_projection(rng):
    return {
        "rows": rng.choice((1, 6, 8, 9, 33, 129, 256, 512)),
        "groups": rng.choice((1, 2, 4)),
        "group_width": 512 * rng.randint(1, 8),
        "rank": 128 * rng.randint(1, 8),
        "hidden": 512 * rng.randint(1, 10),
        "seed": _seed(rng),
    }


def _fuzz_bmm(rng):
    return {
        "major": rng.choice(("n", "k")),
        "batch": rng.choice((8, 16, 32)),
        "m": rng.choice((1, 5, 16, 17, 32, 65, 128, 300)),
        "seed": _seed(rng),
    }


def _fuzz_mla(rng):
    return {
        "heads": rng.choice((8, 16)),
        "m": rng.randint(1, 32),
        "fp8_out": rng.random() < 0.5,
        "seed": _seed(rng),
    }


def _fuzz_bf16_gemv(rng):
    backend = rng.choice(("prefill", "mma", "tc", "simt"))
    if backend == "prefill":
        n = rng.choice((384, 512, 1024))
        return {
            "backend": backend,
            "rows": rng.choice((64, 128, 300, 512)),
            "n": n,
            "k": 5120,
            "seed": _seed(rng),
        }
    case = {
        "backend": backend,
        "rows": rng.randint(1, 8)
        if backend in ("tc", "simt")
        else rng.choice((24, 64, 128, 256)),
        "n": 256 * rng.randint(1, 32),
        "k": 64 * rng.randint(2, 64),
        "seed": _seed(rng),
    }
    if backend == "mma":
        case["bias"] = rng.random() < 0.5
    if backend == "simt":
        case.update(
            rows_per_tile=rng.choice((1, 2, 4, 8)),
            block_threads=rng.choice((128, 256)),
            bias=rng.random() < 0.5,
        )
    return case


FUZZ = {
    "gemm_packed_nvfp4": _fuzz_packed_nvfp4,
    "gemm_packed_mxfp4": _fuzz_packed_mxfp4,
    "gemm_block_codec": _fuzz_block_codec,
    "gemm_dense_nvfp4": _fuzz_dense_nvfp4,
    "gemm_block_fp8_linear": _fuzz_block_fp8_linear,
    "gemm_tensor_fp8_linear": _fuzz_tensor_fp8_linear,
    "gemm_wo_projection": _fuzz_wo_projection,
    "gemm_bmm": _fuzz_bmm,
    "gemm_mla_query_projection": _fuzz_mla,
    "gemm_bf16_gemv": _fuzz_bf16_gemv,
}
