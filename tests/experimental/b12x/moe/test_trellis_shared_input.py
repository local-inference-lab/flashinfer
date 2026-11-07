"""Shared-scale input transforms preserve the per-expert FP16 arithmetic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from b12x.moe import fused_moe
from b12x.moe.checkpoints.exl3 import read_exl3_layer, trellis_from_exl3
from b12x.moe._shared.kernels.w4a16.exl3_synth import (
    Exl3SynthConfig,
    write_exl3_checkpoint,
)
from b12x.moe._shared.kernels.w4a16.kernel.fused import W4A16FusedMoeKernel


def _helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "benchmarks/experimental/b12x/benchmark_trellis_shared_input.py"
    )
    spec = importlib.util.spec_from_file_location("trellis_input_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("broadcast", [False, True])
@pytest.mark.parametrize("intermediate", [False, True])
def test_shared_transform_requires_expert_independent_scales(broadcast, intermediate):
    kernel = W4A16FusedMoeKernel(
        size_m=8,
        hidden_size=512,
        intermediate_size=256,
        num_experts=17,
        top_k=4,
        activation="situ",
        apply_router_weight_on_input=False,
        zero_fc2_output=False,
        fc1_tile_n=128,
        fc1_tile_k=128,
        fc2_tile_n=128,
        fc2_tile_k=128,
        moe_block_size=8,
        max_m_blocks=32,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="trellis_t256_proj",
        trellis_bits=2,
        full_rotation=True,
        intermediate_rotation=True,
        intermediate_hadamard=intermediate,
        rotation_input_dtype="bf16",
        broadcast_suh=broadcast,
    )
    assert kernel.shared_input_rotation == (broadcast and intermediate)
    assert kernel.fc1.route_major_a == (not kernel.shared_input_rotation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 8, 16, 128])
def test_shared_input_matches_expanded_scale_rows(tmp_path, rows, dtype):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    helper = _helpers()
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
            seed=1921,
        ),
    )
    layer = read_exl3_layer(tmp_path, manifest, 0, first_slot=0, slot_count=8)
    source, weights = trellis_from_exl3(layer)
    geometry = fused_moe.MoEGeometry(
        num_experts=17, hidden_size=512, intermediate_size=256
    )
    expanded_source, expanded_weights = helper.expanded_input_scales(
        source, weights, 17
    )
    shared = helper.prepare_experts(source, weights, geometry, dtype)
    expanded = helper.prepare_experts(
        expanded_source, expanded_weights, geometry, dtype
    )
    x, ids, routes = helper.inputs(rows, 512, 17, 4, dtype, seed=7723 + rows)
    with (
        helper.prepared_runner(expanded, x, ids, routes) as reference,
        helper.prepared_runner(shared, x, ids, routes) as candidate,
    ):
        assert not reference.binding.fused_launch.broadcast_suh
        assert candidate.binding.fused_launch.broadcast_suh
        helper.qualify_pair(
            reference,
            candidate,
            x,
            ids,
            routes,
            17,
            seed=9813 + rows,
            scratch_saving_bytes=rows * 3 * 512 * 2,
        )
