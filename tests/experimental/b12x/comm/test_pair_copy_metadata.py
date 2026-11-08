"""Pair-copy declarations reject unsafe extents and overlapping destinations."""

import pytest
import torch

from b12x.comm.pcie.pair_copy import Query, _State


@pytest.mark.parametrize(
    "arguments",
    [
        ("unknown", 1, 16, 16, 8),
        ("pack", 0, 16, 16, 8),
        ("pack", 1, 3, 16, 8),
        ("pack", 1, 16, 0, 8),
        ("pull", 7, 16, 16, 8),
        ("pack", 1, 16, 16, 3),
    ],
)
def test_invalid_declarations(arguments):
    with pytest.raises(ValueError):
        Query(*arguments)


def test_overlapping_pack_output_rejected_before_launch():
    state = _State(Query("pack", 2, 16, 16, 8), torch.device("cpu"), None)
    output = torch.empty((2, 8), dtype=torch.float32)
    first = output.flatten()[:8].view(2, 4)
    second = torch.empty_like(first)
    with pytest.raises(ValueError, match="disjoint"):
        state.run(first, second, output, None)


def test_wrong_pack_extent_rejected_before_launch():
    state = _State(Query("pack", 2, 16, 16, 8), torch.device("cpu"), None)
    first = torch.empty((2, 4), dtype=torch.float32)
    with pytest.raises(ValueError, match="extents"):
        state.run(first, first.clone(), torch.empty((2, 9), dtype=torch.float32), None)
