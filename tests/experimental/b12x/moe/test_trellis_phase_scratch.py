"""Trellis scratch follows FC1, FC2, and output-reconstruction lifetimes."""

from dataclasses import replace
import math

import pytest
import torch

from b12x.moe.fused_moe import _impl


def disjoint_nbytes(plan):
    size = 0
    for spec in plan.tensor_specs:
        size = _impl.align_up(size, max(16, spec.dtype.itemsize))
        size += math.prod(spec.shape) * spec.dtype.itemsize
    return size


def phase_saving(plan):
    sizes = {s.name: math.prod(s.shape) * s.dtype.itemsize for s in plan.tensor_specs}
    before = sum(
        sizes[name]
        for name in (
            "fc1_c_tmp",
            "fc2_c_tmp",
            "rotation_a_gate",
            "full_rotation_output",
        )
    )
    after = max(
        _impl.align_up(sizes["fc1_c_tmp"], 16) + sizes["rotation_a_gate"],
        sizes["fc2_c_tmp"],
        sizes["full_rotation_output"],
    )
    return before - after


def plan_for_rows(monkeypatch, rows):
    monkeypatch.setattr(_impl, "get_num_sm", lambda _: 188)
    return _impl._plan_core_workspace(
        "w4a16",
        "w4a16",
        896,
        896,
        3584,
        384,
        16,
        torch.device("cpu"),
        torch.bfloat16,
        routed_rows=rows * 16,
        max_rows=rows * 16,
        activation="situ",
        source_format="b12x_trellis",
        w13_layout="trellis_t256_proj",
        w4a16_weight_layout="trellis_t256",
        trellis_bits=2,
        trellis_tile_config=(128, 128, 128, 128),
        trellis_codebook="lut_e4m3",
        intermediate_hadamard=True,
        trellis_input_scale_rows=1,
    )


@pytest.mark.parametrize("rows", [1, 8, 32, 128, 1024, 3072])
def test_phase_layout_preserves_simultaneous_operands(monkeypatch, rows):
    plan = plan_for_rows(monkeypatch, rows)
    offsets, size = _impl._core_workspace_offsets(plan)
    assert size == disjoint_nbytes(plan) - phase_saving(plan)
    arena = torch.full((size,), 0x6A, dtype=torch.uint8)
    tensors = _impl._map_core_workspace_views(plan, arena, do_init=False)
    fc1, rotation = tensors["fc1_c_tmp"], tensors["rotation_a_gate"]
    assert rotation.data_ptr() >= fc1.data_ptr() + fc1.nbytes
    fc1.fill_(3)
    rotation.fill_(5)
    assert bool((fc1 == 3).all())
    assert (
        offsets["fc2_c_tmp"] == offsets["fc1_c_tmp"] == offsets["full_rotation_output"]
    )
    tensors["fc2_c_tmp"].fill_(7)
    tensors["full_rotation_output"].fill_(11)
    for name, tensor in tensors.items():
        if name not in {
            "fc1_c_tmp",
            "fc2_c_tmp",
            "rotation_a_gate",
            "full_rotation_output",
        }:
            assert bool((tensor.view(torch.uint8) == 0x6A).all()), name


@pytest.mark.parametrize(
    "changes",
    [
        dict(full_rotation=False),
        dict(intermediate_hadamard=False),
        dict(projection_mixed_trellis=True),
        dict(trellis_pair_kinds=frozenset({"k3"})),
    ],
)
def test_other_execution_contracts_keep_disjoint_layout(monkeypatch, changes):
    plan = replace(plan_for_rows(monkeypatch, 32), **changes)
    assert _impl._core_workspace_nbytes(plan) == disjoint_nbytes(plan)
