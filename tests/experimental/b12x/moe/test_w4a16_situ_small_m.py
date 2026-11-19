"""Small-M MXFP4 W4A16 experts apply the SiTU nonlinearity of Kimi-K3."""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from b12x.moe._shared.kernels.reference import moe_reference_w4a16_fp4_e8m0_k32
from b12x.testing.reference.w4a16_reference import (
    compare_to_reference,
    moe_reference_w4a16,
)
from b12x._lib.intrinsics.reference import swizzle_block_scale
from ..conftest import require_b12x


def _e8m0(shape, *, offset=0, base=119):
    numel = 1
    for dim in shape:
        numel *= int(dim)
    storage = (
        torch.arange(numel, device="cuda", dtype=torch.int64) + offset
    ) % 4 + base
    return storage.to(torch.uint8).reshape(shape)


@pytest.mark.parametrize(
    "route_mode,direct_schedule",
    [("direct", "fused"), ("direct", "split"), ("packed", "fused")],
)
@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("nonlinearity", ["situ", "silu"])
def test_mxfp4_a16_small_m_matches_reference(
    rows, route_mode, direct_schedule, nonlinearity
):
    _check_small_m(rows, route_mode, direct_schedule, nonlinearity)


@pytest.mark.parametrize("source_format", ["fp4_e8m0_k32", "modelopt_nvfp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nonlinearity", ["silu", "situ", "relu2"])
@pytest.mark.parametrize("layout", ["w13", "w31"])
def test_split_direct_neighbor_arithmetic(source_format, dtype, nonlinearity, layout):
    eligibility = (
        pytest.raises(
            ValueError,
            match="direct routing does not support|A16 token cutoff requires BF16",
        )
        if dtype == torch.float16
        else nullcontext()
    )
    with eligibility:
        _check_small_m(
            2,
            "direct",
            "split",
            nonlinearity,
            source_format=source_format,
            dtype=dtype,
            layout=layout,
            geometry=(8, 256, 256, 4),
        )


@pytest.mark.parametrize(
    "hidden,nonlinearity,layout",
    [
        (3584, "silu", "w31"),
        (3584, "situ", "w31"),
        (3584, "relu2", "w13"),
        (3584, "relu2", "w31"),
        (1536, "relu2", "w31"),
    ],
)
def test_split_direct_generic_fc1_layouts(hidden, nonlinearity, layout):
    _check_small_m(
        2,
        "direct",
        "split",
        nonlinearity,
        layout=layout,
        geometry=(8, hidden, 192, 4),
    )


@pytest.mark.parametrize("source_format", ["fp4_e8m0_k32", "modelopt_nvfp4"])
@pytest.mark.parametrize("rows", [1, 2])
def test_split_direct_wide_intermediate(source_format, rows):
    _check_small_m(
        rows,
        "direct",
        "split",
        "silu",
        source_format=source_format,
        layout="w31",
        geometry=(8, 256, 512, 4),
    )


def test_split_direct_compiled_prepared_graph():
    _check_small_m(2, "direct", "split", "situ", layout="w31", compiled=True)


def test_split_direct_odd_topk_compiled_graph():
    _check_small_m(
        2,
        "direct",
        "split",
        "situ",
        layout="w31",
        geometry=(8, 3584, 192, 3),
        compiled=True,
    )


@pytest.mark.parametrize("fc1_stages", [2, 3, 4, 5])
@pytest.mark.parametrize("geometry", [(16, 2048, 256, 4), (16, 5120, 576, 6)])
@pytest.mark.parametrize("nonlinearity", ["situ", "silu"])
def test_independent_fc1_pipeline_matches_neighboring_experts(
    fc1_stages, geometry, nonlinearity
):
    _check_small_m(
        4,
        "packed",
        "fused",
        nonlinearity,
        geometry=geometry,
        fc1_pipeline_stages=fc1_stages,
    )


@pytest.mark.parametrize("rows", [2, 8])
@pytest.mark.parametrize("pipeline_stages", [4, 5])
def test_native_mxfp4_i256_graph_with_deep_pipelines(rows, pipeline_stages):
    _check_small_m(
        rows,
        "packed",
        "fused",
        "situ",
        geometry=(64, 3584, 256, 16),
        fc1_pipeline_stages=5,
        pipeline_stages=pipeline_stages,
        required_packing="source_native",
        replay_graph=True,
    )


@pytest.mark.parametrize(
    "rows,activation,geometry,layout",
    [
        (1, "silu", (8, 256, 192, 2), "w13"),
        (4, "situ", (8, 384, 192, 3), "w31"),
        (8, "silu", (16, 256, 256, 4), "w31"),
        (16, "situ", (8, 256, 192, 2), "w13"),
    ],
)
def test_packed_three_stage_graph_matches_reference(rows, activation, geometry, layout):
    _check_small_m(
        rows,
        "packed",
        "fused",
        activation,
        geometry=geometry,
        layout=layout,
        packed_schedule="split",
        required_packing="mma_packed",
        replay_graph=True,
    )


def _check_small_m(
    rows,
    route_mode,
    direct_schedule,
    nonlinearity,
    *,
    source_format="fp4_e8m0_k32",
    dtype=torch.bfloat16,
    layout="w13",
    geometry=(64, 3584, 192, 16),
    compiled=False,
    fc1_pipeline_stages=None,
    pipeline_stages=3,
    required_packing=None,
    replay_graph=False,
    packed_schedule="fused",
):
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig
    from b12x.preparation import PreparationSession, PreparedCall

    device = require_b12x()
    torch.manual_seed(3584 + rows)
    experts, hidden, intermediate, topk = geometry
    fc1_rows = intermediate * (1 if nonlinearity == "relu2" else 2)
    raw = (
        torch.randint(
            256,
            (experts, fc1_rows, hidden // 2),
            dtype=torch.uint8,
            device=device,
        ),
        # Large FC1 scales drive SiTU's tanh terms into saturation, where it departs from SiLU.
        _e8m0((experts, fc1_rows, hidden // 32), base=124),
        torch.ones(experts, device=device),
        torch.randint(
            256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device=device
        ),
        _e8m0((experts, hidden, intermediate // 32), offset=1),
        torch.ones(experts, device=device),
    )
    if source_format == "modelopt_nvfp4":
        raw = (
            raw[0],
            swizzle_block_scale(
                (
                    torch.rand(experts, fc1_rows, hidden // 16, device=device) * 0.25
                    + 0.03125
                ).to(torch.float8_e4m3fn)
            ),
            torch.full((experts,), 0.1, device=device),
            raw[3],
            swizzle_block_scale(
                (
                    torch.rand(experts, hidden, intermediate // 16, device=device)
                    * 0.25
                    + 0.03125
                ).to(torch.float8_e4m3fn)
            ),
            torch.full((experts,), 0.1, device=device),
        )
    oracle_raw = raw
    if layout == "w31" and nonlinearity != "relu2":
        if source_format == "modelopt_nvfp4":
            from b12x.moe._shared.kernels.reference import unswizzle_block_scale

            sf = torch.stack(
                [unswizzle_block_scale(v, fc1_rows, hidden // 16) for v in raw[1]]
            )
            swapped_sf = swizzle_block_scale(
                torch.cat((sf[:, intermediate:], sf[:, :intermediate]), dim=1).to(
                    torch.float8_e4m3fn
                )
            )
        else:
            swapped_sf = torch.cat(
                (raw[1][:, intermediate:], raw[1][:, :intermediate]), dim=1
            )
        raw = (
            torch.cat((raw[0][:, intermediate:], raw[0][:, :intermediate]), dim=1),
            swapped_sf,
            *raw[2:],
        )
    x = (torch.randn(rows, hidden, device=device) * 0.125).to(dtype)
    ids = torch.stack(
        [torch.randperm(experts, device=device)[:topk] for _ in range(rows)]
    ).to(torch.int32)
    weights = torch.softmax(torch.randn(rows, topk, device=device), dim=-1)
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format=source_format, w13_layout=layout),
        activation=fused_moe.ActivationSpec(
            mode="a4" if source_format == "modelopt_nvfp4" else "a16",
            nonlinearity=nonlinearity,
            io_dtype=dtype,
            a16_max_tokens=rows if source_format == "modelopt_nvfp4" else 0,
        ),
        constraints=fused_moe.WeightPlanConstraints(
            required_packing=(
                required_packing
                if required_packing is not None
                else (
                    fused_moe.WeightPacking.SOURCE_NATIVE
                    if direct_schedule == "split"
                    else None
                )
            ),
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=experts, hidden_size=hidden, intermediate_size=intermediate
        ),
    )
    prepared = fused_moe.prepare_weights(
        plan=plan,
        weights=fused_moe.PackedWeights(
            w13=raw[0].clone(),
            w13_block_scales=raw[1].clone(),
            w13_global_scales=raw[2].clone(),
            w2=raw[3].clone(),
            w2_block_scales=raw[4].clone(),
            w2_global_scales=raw[5].clone(),
            input_scale=torch.ones(experts, device=device)
            if source_format == "modelopt_nvfp4"
            else None,
            intermediate_scale=torch.ones(experts, device=device)
            if source_format == "modelopt_nvfp4"
            else None,
        ),
    )
    launch = (
        {}
        if fc1_pipeline_stages is None
        else dict(
            w4a16_tile_config=(128, 64, 64, 128),
            w4a16_block_size_m=16,
            w4a16_pipeline_stages=pipeline_stages,
            w4a16_fc1_pipeline_stages=fc1_pipeline_stages,
        )
    )
    execution = fused_moe.plan_execution(
        experts=prepared,
        capacity=fused_moe.ExecutionCapacity(max_tokens=rows, top_k=topk),
        routing=fused_moe.RoutingSpec(),
        override=MoeDecodeConfig(
            backend="w4a16",
            route_planner="internal",
            max_active_clusters=None,
            w4a16_route_mode=route_mode,
            w4a16_direct_schedule=direct_schedule,
            w4a16_packed_schedule=packed_schedule,
            **launch,
        ),
    )
    output = torch.empty_like(x)
    bound = {}

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in state.scratch.scratch_specs()
        )
        bound["scratch"] = scratch
        bound["state"] = state
        bound["binding"] = state.bind(
            scratch=scratch,
            a=x,
            experts=prepared,
            topk_weights=weights,
            topk_ids=ids,
            output=output,
        )
        return PreparedCall(run=lambda: state.run(bound["binding"]), owners=scratch)

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                execution.request(
                    name=f"w4a16-{route_mode}-{nonlinearity}-{rows}",
                    prepare_call=prepare,
                ),
            )
        )
        reference = (
            moe_reference_w4a16_fp4_e8m0_k32
            if source_format == "fp4_e8m0_k32"
            else moe_reference_w4a16
        )
        reference_kwargs = (
            {"w13_layout": "w13"} if source_format == "fp4_e8m0_k32" else {}
        )
        expected = reference(
            x,
            *oracle_raw,
            ids,
            weights,
            experts,
            hidden,
            intermediate,
            activation=nonlinearity,
            **reference_kwargs,
        )
        run = lambda: bound["state"].run(bound["binding"])
        if compiled:
            binding = fused_moe.bind(
                execution,
                scratch=bound["scratch"],
                a=x,
                experts=prepared,
                topk_weights=weights,
                topk_ids=ids,
                output=output,
            )

            def consume():
                return fused_moe.run(binding=binding)

            run = torch.compile(consume, fullgraph=True)
            run()
        metrics = compare_to_reference(output, expected)
        assert metrics.cos >= 0.9975, (route_mode, metrics)
        if source_format == "fp4_e8m0_k32" and direct_schedule == "fused":
            reference_rms = expected.float().square().mean().sqrt().item()
            assert metrics.rmse / max(reference_rms, 1e-12) <= 0.015, metrics
        if direct_schedule == "split" or replay_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            allocated = torch.cuda.memory_allocated(device)
            for tensor in bound["scratch"]:
                if tensor.is_floating_point():
                    tensor.fill_(float("nan"))
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            metrics = compare_to_reference(output, expected)
            assert metrics.cos >= 0.9975, metrics
            if source_format == "fp4_e8m0_k32" and direct_schedule == "fused":
                assert metrics.rmse / max(reference_rms, 1e-12) <= 0.015, metrics
            stable = output.clone()
            ids[0].fill_(-1)
            graph.replay()
            torch.cuda.synchronize(device)
            assert bool(output[0].eq(0).all())
            if replay_graph and rows > 1:
                metrics = compare_to_reference(output[1:], expected[1:])
                assert metrics.cos >= 0.9975, metrics
                active_reference_rms = (
                    expected[1:].float().square().mean().sqrt().item()
                )
                assert metrics.rmse / max(active_reference_rms, 1e-12) <= 0.015, metrics
                replayed = output.clone()
                run()
                torch.cuda.synchronize(device)
                torch.testing.assert_close(output, replayed, atol=0, rtol=0)
            else:
                torch.testing.assert_close(output[1:], stable[1:], atol=0, rtol=0)
            weights.zero_()
            graph.replay()
            torch.cuda.synchronize(device)
            assert bool(output.eq(0).all())


@pytest.mark.parametrize("rows", [4, 16])
@pytest.mark.parametrize(
    "geometry", [(16, 2048, 352, 4), (16, 3584, 192, 16), (16, 5120, 576, 6)]
)
@pytest.mark.parametrize(
    "tiles", [(64, 256, 64, 256), (128, 64, 128, 64), (64, 256, 128, 128)]
)
@pytest.mark.parametrize("nonlinearity", ["situ", "silu"])
def test_native_fc1_n_and_fc2_k_tail_tactics_match_reference(
    rows, geometry, tiles, nonlinearity
):
    _check_native_pinned_tail(rows, nonlinearity, geometry=geometry, tile_config=tiles)


def test_native_fc1_tail_preserves_scratch_boundaries():
    _check_native_pinned_tail(
        2,
        "situ",
        geometry=(128, 3584, 192, 16),
        tile_config=(64, 256, 64, 256),
        written_scratch=("fc1_c_tmp",),
    )


def _check_native_pinned_tail(
    rows,
    nonlinearity,
    *,
    geometry,
    tile_config,
    block_size=16,
    distinct_routes=False,
    written_scratch=(),
    seed=None,
):
    from b12x.moe._shared.kernels.w4a16.host import make_w4a16_packed_buffers
    from b12x.moe._shared.kernels.w4a16.kernel.execution import run_w4a16_moe
    from b12x.moe._shared.kernels.w4a16.kernel.fused_launch import (
        compile_w4a16_fused_moe,
    )
    from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import compile_w4a16_topk_sum
    from b12x.moe._shared.kernels.w4a16.prepare import (
        prepare_w4a16_e8m0_native_weights,
    )
    from b12x.moe._shared.kernels.w4a16.route_pack import (
        compile_w4a16_route_pack_launches,
    )

    device = require_b12x()
    torch.manual_seed(3584 + rows if seed is None else seed)
    experts, hidden, intermediate, topk = geometry
    raw = (
        torch.randint(
            256,
            (experts, 2 * intermediate, hidden // 2),
            dtype=torch.uint8,
            device=device,
        ),
        _e8m0((experts, 2 * intermediate, hidden // 32), base=124),
        torch.ones(experts, device=device),
        torch.randint(
            256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device=device
        ),
        _e8m0((experts, hidden, intermediate // 32), offset=1),
        torch.ones(experts, device=device),
    )
    x = (torch.randn(rows, hidden, device=device) * 0.125).to(torch.bfloat16)
    if distinct_routes:
        ids = torch.arange(rows * topk, dtype=torch.int32, device=device).reshape(
            rows, topk
        )
    else:
        ids = torch.stack(
            [torch.randperm(experts, device=device)[:topk] for _ in range(rows)]
        ).to(torch.int32)
    weights = torch.softmax(torch.randn(rows, topk, device=device), dim=-1)
    expected = moe_reference_w4a16_fp4_e8m0_k32(
        x,
        raw[0],
        raw[1].view(torch.float8_e8m0fnu),
        raw[2],
        raw[3],
        raw[4].view(torch.float8_e8m0fnu),
        raw[5],
        ids,
        weights,
        experts,
        hidden,
        intermediate,
        activation=nonlinearity,
        w13_layout="w13",
    )
    prepared = prepare_w4a16_e8m0_native_weights(
        *(tensor.clone() for tensor in raw),
        activation=nonlinearity,
        params_dtype=torch.bfloat16,
        w13_layout="w13",
    )
    buffers = make_w4a16_packed_buffers(
        prepared,
        m=rows,
        topk=topk,
        dtype=torch.bfloat16,
        device=device,
        block_size_m=block_size,
    )
    props = torch.cuda.get_device_properties(device)
    fused = compile_w4a16_fused_moe(
        size_m=rows,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_experts=experts,
        top_k=topk,
        activation=nonlinearity,
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        moe_block_size=block_size,
        max_m_blocks=buffers.block_expert_ids.numel(),
        element_dtype="bf16",
        sms=props.multi_processor_count,
        max_shared_mem=props.shared_memory_per_block_optin,
        weight_layout="modelopt",
        scale_format="e8m0_k32",
        w13_layout="w13",
        pipeline_stages=3,
        fc1_pipeline_stages=3,
        force_tile_config=tile_config,
    )
    if tile_config[-1] == 512:
        assert fused.blocks_per_sm <= 2
        if written_scratch:
            assert (
                rows * topk * ((hidden + 511) // 512)
                > props.multi_processor_count * fused.blocks_per_sm
            )
    route_pack = compile_w4a16_route_pack_launches(
        tokens=rows,
        topk=topk,
        block_size=block_size,
        num_experts=experts,
        ordinal=torch.cuda.current_device(),
    )
    topk_sum = compile_w4a16_topk_sum(
        m=rows, topk=topk, hidden_size=hidden, element_dtype="bf16"
    )
    scratch, guards = {}, {}
    for name in ("fc1_c_tmp", "fc2_c_tmp"):
        tensor = getattr(buffers, name)
        storage = torch.full(
            (2 * tensor.numel(),), 123456.0, dtype=tensor.dtype, device=device
        )
        scratch[name] = storage[: tensor.numel()].view_as(tensor)
        guards[name] = storage[tensor.numel() :]
    actual = run_w4a16_moe(
        x,
        prepared,
        weights,
        ids,
        activation=nonlinearity,
        route_block_size_m=block_size,
        route_mode="packed",
        intermediate_cache13=buffers.intermediate_cache13,
        intermediate_cache2=buffers.intermediate_cache2,
        output=buffers.output,
        **scratch,
        packed_route_indices=buffers.packed_route_indices,
        block_expert_ids=buffers.block_expert_ids,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        fused_launch=fused,
        topk_sum_launch=topk_sum,
        route_pack_launches=route_pack,
    )
    torch.cuda.synchronize()
    metrics = compare_to_reference(actual, expected)
    assert metrics.cos >= 0.9975, metrics
    reference_rms = expected.float().square().mean().sqrt().item()
    assert metrics.rmse / max(reference_rms, 1e-12) <= 0.015, metrics
    for name, tensor in scratch.items():
        if name in written_scratch:
            assert bool(tensor.ne(123456.0).any()), f"MoE did not write {name}"
        assert bool(guards[name].eq(123456.0).all()), f"MoE wrote beyond planned {name}"


@pytest.mark.parametrize("topk", [12, 24])
def test_native_fc2_n512_tail_preserves_written_scratch_boundaries(topk):
    _check_native_pinned_tail(
        2,
        "situ",
        geometry=(64, 3840, 576, topk),
        tile_config=(128, 128, 32, 512),
        block_size=8,
        distinct_routes=True,
        written_scratch=("fc1_c_tmp", "fc2_c_tmp") if topk == 24 else (),
        seed=3840,
    )
