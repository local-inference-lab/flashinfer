"""Direct typed stores preserve full-rotation FP32 reduction and final rounding."""

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig,
    write_exl3_checkpoint,
)
from b12x.moe._shared.kernels.w4a16.kernel.sum_launch import (
    _w4a16_topk_sum_launch_flat,
    compile_w4a16_topk_sum,
)
from b12x.moe._shared.kernels.w4a16.kernel.types import W4A16TopKSumCompileResult


def _helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/experimental/b12x/benchmark_trellis_final_store.py"
    )
    spec = importlib.util.spec_from_file_location("trellis_store_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retained_sum_rejects_destination_dtype_mismatch():
    launch = W4A16TopKSumCompileResult(
        compiled=None, m=1, topk=4, hidden_size=512, output_dtype=torch.bfloat16
    )
    with pytest.raises(ValueError, match="output dtype"):
        _w4a16_topk_sum_launch_flat(
            torch.zeros(4, 512),
            torch.zeros(1, 512, dtype=torch.float16),
            1,
            4,
            512,
            "fp16",
            0,
            launcher=launch,
        )


def test_full_rotation_launch_without_dtype_keeps_fp32_contract():
    launch = W4A16TopKSumCompileResult(
        compiled=None, m=1, topk=4, hidden_size=512, full_rotation=True
    )
    with pytest.raises(ValueError, match="output dtype"):
        _w4a16_topk_sum_launch_flat(
            torch.zeros(4, 512),
            torch.zeros(1, 512, dtype=torch.bfloat16),
            1,
            4,
            512,
            "fp16",
            0,
            launcher=launch,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("rows", [1, 8, 128])
def test_typed_store_matches_fp32_result_copy(tmp_path, dtype, offset, rows):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    helper = _helpers()
    shared = helper.shared
    manifest = write_exl3_checkpoint(
        tmp_path,
        Exl3SynthConfig(
            codebook="lut_e4m3",
            num_experts=17,
            hidden_size=512,
            intermediate_size=512,
            moe_layer_indices=(0,),
            bits=2,
            intermediate_hadamard=True,
            pre_block=512,
            post_block=128,
            seed=2917,
        ),
    )
    layer = shared.read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    source, weights = shared.trellis_from_exl3(layer)
    geometry = shared.fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    experts = shared.prepare_experts(source, weights, geometry, dtype)
    x, ids, routes = shared.inputs(rows, 512, 17, 4, dtype, seed=6103 + rows)
    with (
        shared.prepared_runner(
            experts, x, ids, routes, output_dtype=torch.float32
        ) as fp32,
        shared.prepared_runner(experts, x, ids, routes, output_offset=offset) as direct,
    ):
        reference = helper.reference_runner(fp32, x, output_offset=offset)
        assert direct.output.data_ptr() % 16 == 2 * offset
        assert direct.binding.output.data_ptr() == direct.output.data_ptr()
        assert direct.binding.output_cast_target is None
        assert direct.binding.topk_sum_launch.output_dtype == dtype
        assert fp32.binding.topk_sum_launch.output_dtype == torch.float32
        shared.qualify_pair(reference, direct, x, ids, routes, 17, seed=6109 + rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_terminal_conversion_preserves_exceptional_values(dtype):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    routes = torch.empty((4, 512), dtype=torch.float16, device="cuda")
    ids = torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4)
    weights = torch.full((1, 4), 0.25, dtype=torch.float32, device="cuda")
    scales = torch.ones((1, 512), dtype=torch.float16, device="cuda")
    full = torch.empty((1, 512), dtype=torch.float32, device="cuda")
    expected = torch.empty((1, 512), dtype=dtype, device="cuda")
    actual = torch.empty(513, dtype=dtype, device="cuda")[1:].view(1, 512)
    options = dict(
        m=1,
        topk=4,
        hidden_size=512,
        element_dtype="fp16",
        full_rotation=True,
        intermediate_hadamard=True,
        num_experts=4,
        broadcast_svh=True,
    )
    fp32_launch = compile_w4a16_topk_sum(**options, output_dtype=torch.float32)
    typed_launch = compile_w4a16_topk_sum(**options, output_dtype=dtype)

    def execute(output, launch):
        _w4a16_topk_sum_launch_flat(
            routes,
            output,
            1,
            4,
            512,
            "fp16",
            torch.cuda.current_stream().cuda_stream,
            full_rotation=True,
            intermediate_hadamard=True,
            num_experts=4,
            topk_weights=weights,
            route_expert_ids=ids,
            svh_table=scales,
            launcher=launch,
        )

    routes.zero_()
    execute(full, fp32_launch)
    execute(actual, typed_launch)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute(full, fp32_launch)
        expected.copy_(full)
        execute(actual, typed_launch)
    for case in ("finite", "infinite", "nan", "invalid-nan"):
        routes.copy_(
            torch.linspace(-1.0, 1.0, routes.numel(), device="cuda").view_as(routes)
        )
        ids.copy_(torch.arange(4, dtype=torch.int32, device="cuda").view(1, 4))
        if case != "finite":
            routes[0].fill_(float("inf") if case == "infinite" else float("nan"))
        if case == "invalid-nan":
            ids[0, 0] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), case
