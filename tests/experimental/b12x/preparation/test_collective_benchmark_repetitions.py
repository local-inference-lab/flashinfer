"""Reduction timing graphs scale invocation counts by logical token rows."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie._benchmark import _ReductionInputs, _reduction_repetitions


@pytest.mark.parametrize(
    "shape, count",
    [
        ((2560,), 100),
        ((1, 2560), 100),
        ((4, 2560), 25),
        ((16, 2560), 7),
        ((2, 5, 2560), 10),
        ((100, 2560), 1),
        ((1024, 2560), 1),
        ((4096, 4096), 1),
    ],
)
def test_reduction_count_scales_with_logical_rows(shape, count):
    assert _reduction_repetitions(shape) == count


@pytest.mark.parametrize("inplace", [False, True])
def test_grouped_fused_count_uses_residual_rows(inplace):
    query = SimpleNamespace(
        surface="OneshotAllReduce.fused_add_rmsnorm",
        world_size=2,
        rank=0,
        setup={"shape": (4, 3, 8), "logical_shape": (4, 8)},
        call={
            "dtype": "bfloat16",
            "residual_dtype": "bfloat16",
            "normalize": False,
            "input_addend": False,
            "group_size": 3,
            "group_ids_dtype": "int32",
            "group_id_limit": 1,
            "reduction_dtype": None,
            "output_dtype": "bfloat16",
            "zero_centered": False,
        },
    )
    inputs = _ReductionInputs(
        SimpleNamespace(
            query=query, runtime=SimpleNamespace(device=torch.device("cpu"))
        ),
        inplace=inplace,
    )
    assert inputs.repetitions == 25
    assert inputs.residual.shape == (4, 8)


@pytest.mark.parametrize("inplace", [False, True])
def test_plain_alias_does_not_change_timing_count(inplace):
    query = SimpleNamespace(
        surface="OneshotAllReduce.all_reduce",
        world_size=2,
        rank=0,
        setup={"shape": (16, 8)},
        call={"dtype": "bfloat16"},
    )
    inputs = _ReductionInputs(
        SimpleNamespace(
            query=query, runtime=SimpleNamespace(device=torch.device("cpu"))
        ),
        inplace=inplace,
    )
    assert inputs.repetitions == 7
    assert (inputs.source is inputs.out) is inplace
