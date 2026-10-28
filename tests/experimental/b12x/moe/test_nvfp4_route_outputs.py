"""Unweighted NVFP4 expert outputs and physical FC1 half gains."""

from __future__ import annotations

import math

import pytest
import torch

from b12x._lib.intrinsics.reference import (
    fp4_quantize_values_torch,
    swizzle_block_scale,
)
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.moe import fused_moe as moe
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x


def _quantized_interval(lower, upper):
    lo = lower.float().reshape(lower.shape[0], -1, 16)
    hi = upper.float().reshape_as(lo)
    minabs = torch.minimum(lo.abs(), hi.abs())
    minabs = torch.where((lo <= 0) & (hi >= 0), 0, minabs)
    maxabs = torch.maximum(lo.abs(), hi.abs())
    # rcp.approx and two FP32 multiplies each contribute at most two ULPs.
    epsilon = 8 * torch.finfo(torch.float32).eps
    sflo = (
        (minabs.amax(-1, keepdim=True) / 6 * (1 - epsilon))
        .to(torch.float8_e4m3fn)
        .float()
    )
    sfhi = (
        (maxabs.amax(-1, keepdim=True) / 6 * (1 + epsilon))
        .to(torch.float8_e4m3fn)
        .float()
    )
    # BF16 input intervals span at most adjacent E4M3 scale encodings.
    assert torch.all(
        sfhi.to(torch.float8_e4m3fn).view(torch.uint8).int()
        - sflo.to(torch.float8_e4m3fn).view(torch.uint8).int()
        <= 1
    )
    lows, highs = [], []
    for scale in (sflo, sfhi):
        quotient_lo = lo / scale.clamp_min(1e-30)
        quotient_hi = hi / scale.clamp_min(1e-30)
        qlo = fp4_quantize_values_torch(quotient_lo - quotient_lo.abs() * epsilon)
        qhi = fp4_quantize_values_torch(quotient_hi + quotient_hi.abs() * epsilon)
        lows.append(qlo * scale)
        highs.append(qhi * scale)
    return torch.minimum(*lows).reshape_as(lower), torch.maximum(*highs).reshape_as(
        upper
    )


def _decode(packed):
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=packed.device,
    )
    codes = torch.stack((packed & 15, packed >> 4), -1).reshape(packed.shape[0], -1)
    return values[codes.long()] * 0.125


def _reference(x, ids, w13, w2, gains, gain2, *, gate_first, raw_routes=False):
    tokens, topk = ids.shape
    hidden, intermediate = x.shape[1], w2.shape[-1] * 2
    reference = torch.zeros(tokens, topk, hidden, device=x.device)
    magnitude = torch.zeros_like(reference)
    uncertainty = torch.zeros_like(reference)
    xq = x.float()
    for expert in range(w13.shape[0]):
        rows, routes = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        projected = xq[rows] @ _decode(w13[expert]).T
        first, second = projected.chunk(2, -1)
        up, gate = (second, first) if gate_first else (first, second)
        expert_gains = gains if gains.numel() == 1 else gains[expert]
        if expert_gains.numel() == 2:
            up_gain, gate_gain = expert_gains.flip(0) if gate_first else expert_gains
        else:
            up_gain = gate_gain = expert_gains
        activated = torch.nn.functional.silu(gate * gate_gain) * (up * up_gain)
        # FC1 dyadic operands sum exactly in FP32 for this fixture. Scaling,
        # sigmoid exp/reciprocal and multiplication contribute at most 16 ULPs.
        activation_error = activated.abs() * (16 * torch.finfo(torch.float32).eps)
        lower, upper = _quantized_interval(
            (activated - activation_error).to(torch.bfloat16),
            (activated + activation_error).to(torch.bfloat16),
        )
        activations = (lower + upper) * 0.5
        quantization_error = (upper - lower) * 0.5
        down = _decode(w2[expert])
        total = torch.zeros(rows.numel(), hidden, device=x.device)
        absolute = torch.zeros_like(total)
        for start in range(0, intermediate, 128):
            part = (
                activations[:, start : start + 128] @ down[:, start : start + 128].T
            ) * gain2[expert]
            total += part
            if raw_routes:
                absolute += (
                    activations[:, start : start + 128].abs()
                    @ down[:, start : start + 128].abs().T
                ) * gain2[expert]
            else:
                absolute += part.abs()
        reference[rows, routes] = total
        magnitude[rows, routes] = absolute
        uncertainty[rows, routes] = (quantization_error @ down.abs().T) * gain2[expert]
    return reference, magnitude, uncertainty


def _reference_bound(
    reference, magnitude, intermediate, uncertainty=0, *, raw_routes=False
):
    slices = math.ceil(intermediate / 128)
    if raw_routes:
        # FP32 dot products, gain multiplication and slice accumulation precede
        # the one BF16 output conversion. Absolute products cover cancellation.
        rounds = 128 + slices + 2
        unit = 2.0**-24
        gamma = rounds * unit / (1 - rounds * unit)
        arithmetic = gamma * magnitude
        bound = uncertainty + arithmetic
        bound += 2.0**-8 * (reference.abs() + bound)
    else:
        rounds = 2 * slices
        unit = 2.0**-8
        gamma = rounds * unit / (1 - rounds * unit)
        bound = uncertainty * (1 + gamma) + gamma * magnitude
    return bound + torch.finfo(torch.float32).eps


def _assert_reference(
    output, reference, magnitude, intermediate, uncertainty=0, *, raw_routes=False
):
    bound = _reference_bound(
        reference, magnitude, intermediate, uncertainty, raw_routes=raw_routes
    )
    error = (output.float() - reference).abs()
    assert torch.all(error <= bound), (error.max().item(), bound.max().item())


@pytest.mark.parametrize(
    "experts,hidden,intermediate,tokens,gate_first,paired,fast_math,layout,tile_m",
    [
        (8, 256, 512, 1, False, True, True, "routes", None),
        (8, 256, 512, 1, False, True, False, "routes", None),
        (8, 256, 512, 17, True, True, True, "routes", None),
        (256, 4096, 512, 1, True, True, True, "routes", None),
        (256, 4096, 512, 4096, True, True, True, "routes", None),
        (256, 4096, 4096, 3, True, True, True, "routes", None),
        (8, 256, 512, 3, False, False, True, "routes", None),
        (8, 256, 512, 17, False, False, True, "token", None),
        (8, 256, 512, 17, False, None, True, "token", None),
        *[
            (8, 256, 512, tokens, True, True, True, "routes", tile)
            for tokens in (1, 4, 17, 152)
            for tile in (16, 32, 64, 128)
        ],
    ],
)
def test_expert_routes_reference_and_graph_replay(
    experts, hidden, intermediate, tokens, gate_first, paired, fast_math, layout, tile_m
):
    require_b12x()
    torch.manual_seed(218)
    topk = min(8, experts)
    codebook = torch.tensor(
        [
            -0.375,
            -0.25,
            -0.1875,
            -0.125,
            -0.09375,
            -0.0625,
            -0.03125,
            0.0,
            0.03125,
            0.0625,
            0.09375,
            0.125,
            0.1875,
            0.25,
            0.375,
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = codebook[torch.randint(0, codebook.numel(), (tokens, hidden), device="cuda")]
    x[:, ::16] = 0.375
    w13 = torch.randint(
        0,
        256,
        (experts, intermediate * 2, hidden // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0, 256, (experts, hidden, intermediate // 2), device="cuda", dtype=torch.uint8
    )
    if intermediate == 4096:
        # Every output-row group observes exactly one intermediate slice.
        w2.zero_()
        for part in range(intermediate // 128):
            w2[:, part :: (intermediate // 128), part * 64 : (part + 1) * 64] = 0x22
    gains = torch.linspace(0.03, 0.05, experts, device="cuda")
    if paired:
        gains = torch.stack((gains, gains * 2.5), -1)
    elif paired is None:
        gains = gains[:1].reshape(())
    gain2 = torch.linspace(0.04, 0.07, experts, device="cuda")
    plan = moe.plan_weights(
        source=moe.PackedSource(
            format=moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=moe.W13Layout("w31" if gate_first else "w13"),
        ),
        activation=moe.ActivationSpec(
            mode=moe.ActivationMode.A4, nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=intermediate
        ),
    )
    prepared = moe.prepare_weights(
        plan=plan,
        weights=moe.PackedWeights(
            w13=w13.clone(),
            w2=w2,
            w13_block_scales=swizzle_block_scale(
                torch.full(
                    (experts, 2 * intermediate, hidden // 16),
                    0.125,
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
            ),
            w2_block_scales=swizzle_block_scale(
                torch.full(
                    (experts, hidden, intermediate // 16),
                    0.125,
                    device="cuda",
                    dtype=torch.float8_e4m3fn,
                )
            ),
            w13_global_scales=gains,
            w2_global_scales=gain2,
            input_scale=torch.ones(experts, device="cuda"),
            intermediate_scale=torch.ones(experts, device="cuda"),
        ),
    )
    ids = torch.randint(
        0,
        experts,
        (tokens, topk),
        device="cuda",
        dtype=torch.int32 if fast_math else torch.int64,
    )
    ids[-1, -1] = -1
    weights = torch.rand(tokens, topk, device="cuda")
    output_shape = (tokens, topk, hidden) if layout == "routes" else (tokens, hidden)
    output = torch.empty(output_shape, device="cuda", dtype=torch.bfloat16)
    override = None
    if tile_m is not None:
        from b12x.moe.fused_moe._tuning import MoeDecodeConfig

        override = MoeDecodeConfig(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=tile_m,
            dynamic_route_mode="grouped",
        )
    declaration = moe.plan_execution(
        experts=prepared,
        override=override,
        capacity=moe.ExecutionCapacity(
            max_tokens=tokens, top_k=topk, output_layout=layout
        ),
        invocation=FrozenMapping({"fast_math": fast_math}),
    )
    retained = []

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in state.scratch.scratch_specs()
        )
        binding = state.bind(
            scratch=scratch, a=x, topk_ids=ids, topk_weights=weights, output=output
        )
        retained[:] = [binding, scratch]
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        if experts == 256 and intermediate == 512:
            from b12x.moe.fused_moe._tuning import TUNING

            configuration = TUNING.configure(
                declaration.query, device=session.device.identity
            )
            candidates = [config for _, config in TUNING.iterate(configuration)]
            assert candidates and all(
                config.backend == "dynamic"
                and not config.nvfp4_materialize_intermediate
                for config in candidates
            )
            assert len({config.dynamic_tile_m for config in candidates}) > 1
        session.prepare(
            (declaration.request(name="expert-routes", prepare_call=prepare),)
        )
        binding, scratch = retained
        with kernel_resolution_guard("unweighted expert route capture"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                binding.run()
            for epoch in range(2):
                ids.copy_(
                    torch.randint(0, experts, ids.shape, device="cuda", dtype=ids.dtype)
                )
                ids[-1, -1] = -1 if ids.dtype == torch.int32 else 2**40
                if tokens > 1:
                    ids[-1] = -1
                    ids[0, -1] = experts
                weights.uniform_(0.01, 3)
                if layout == "routes":
                    weights[0, 0] = 0
                for storage in scratch:
                    storage.view(torch.uint8).fill_(0xA5)
                output.fill_(float("nan"))
                graph.replay()
                reference, magnitude, uncertainty = _reference(
                    x,
                    ids,
                    w13,
                    w2,
                    gains,
                    gain2,
                    gate_first=gate_first,
                    raw_routes=layout == "routes",
                )
                reduction_width = intermediate
                if layout == "token":
                    reference = (reference * weights[..., None]).sum(1)
                    magnitude = (magnitude * weights.abs()[..., None]).sum(1)
                    uncertainty = (uncertainty * weights.abs()[..., None]).sum(1)
                    reduction_width *= topk
                _assert_reference(
                    output,
                    reference,
                    magnitude,
                    reduction_width,
                    uncertainty,
                    raw_routes=layout == "routes",
                )
                if epoch == 0 and paired and experts == 256 and tokens == 1:
                    for incorrect_gains in (
                        gains.flip(1),
                        gains[:, :1].expand_as(gains),
                    ):
                        wrong, wrong_magnitude, wrong_uncertainty = _reference(
                            x,
                            ids,
                            w13,
                            w2,
                            incorrect_gains,
                            gain2,
                            gate_first=gate_first,
                            raw_routes=layout == "routes",
                        )
                        with pytest.raises(AssertionError):
                            _assert_reference(
                                output,
                                wrong,
                                wrong_magnitude,
                                intermediate,
                                wrong_uncertainty,
                                raw_routes=layout == "routes",
                            )
                if intermediate == 4096:
                    bound = _reference_bound(
                        reference,
                        magnitude,
                        intermediate,
                        uncertainty,
                        raw_routes=layout == "routes",
                    )
                    for part in range(intermediate // 128):
                        assert torch.any(
                            reference[..., part :: (intermediate // 128)].abs()
                            > bound[..., part :: (intermediate // 128)]
                        )
                if layout == "routes":
                    assert torch.count_nonzero(output[-1, -1]) == 0
                    if tokens > 1:
                        assert torch.count_nonzero(output[-1]) == 0
                        assert torch.count_nonzero(output[0, -1]) == 0
                first = output.clone()
                graph.replay()
                second = output.clone()
                if layout == "routes":
                    assert torch.equal(first, second)
                else:
                    _assert_reference(second, first.float(), magnitude, reduction_width)
            if layout == "routes" and tile_m is not None and tokens == 152:
                for live_rows in (1, 4, tokens):
                    short = moe.bind(
                        declaration,
                        scratch=scratch,
                        a=x[:live_rows],
                        topk_ids=ids[:live_rows],
                        topk_weights=weights[:live_rows],
                        output=output[:live_rows],
                    )
                    short_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(short_graph):
                        short.run()
                    x.neg_()
                    for storage in scratch:
                        storage.view(torch.uint8).fill_(0xFF)
                    output.fill_(float("nan"))
                    short_graph.replay()
                    ref, mag, unc = _reference(
                        x[:live_rows],
                        ids[:live_rows],
                        w13,
                        w2,
                        gains,
                        gain2,
                        gate_first=gate_first,
                        raw_routes=True,
                    )
                    _assert_reference(
                        output[:live_rows], ref, mag, intermediate, unc, raw_routes=True
                    )
                    assert torch.isnan(output[live_rows:]).all()
                    saved = output[:live_rows].clone()
                    short_graph.replay()
                    assert torch.equal(saved, output[:live_rows])
                    short_graph.reset()
        del graph


def test_raw_route_sum_large_flattened_domain():
    """Invalid routes still overwrite BF16 outputs beyond the signed int32 boundary."""
    require_b12x()
    from b12x.moe.fused_moe._impl import _raw_route_sum_op

    routes, hidden = (1 << 19) + 1, 4096
    elements = routes * hidden
    required = elements * 6 + routes * 4
    free, _ = torch.cuda.mem_get_info()
    if free < required + (2 << 30):
        pytest.skip("requires 12 GiB of reducer storage plus 2 GiB device headroom")
    parts = torch.empty((routes, hidden), dtype=torch.float32, device="cuda")
    output = torch.full(
        (routes, hidden), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    ids = torch.full((routes,), -1, dtype=torch.int32, device="cuda")
    _raw_route_sum_op(parts, ids, output, routes, 1, 1)
    torch.cuda.synchronize()
    assert torch.count_nonzero(output).item() == 0
    output.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _raw_route_sum_op(parts, ids, output, routes, 1, 1)
    before = torch.cuda.memory_allocated()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before
    assert torch.count_nonzero(output).item() == 0
    graph.reset()
