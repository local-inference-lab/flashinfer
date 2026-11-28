"""Geometry accepted by the circular-window staging kernel."""

import pytest

pytest.importorskip("cutlass")

from b12x.attention._window_stage import WindowStageKernel


def test_lookback_defaults_to_the_history_length():
    kernel = WindowStageKernel(heads=2, dim=128, capacity=8, window=1152)
    assert kernel.lookback == kernel.window == 1152


def test_lookback_may_be_shorter_than_the_history():
    kernel = WindowStageKernel(heads=2, dim=128, capacity=8, window=1152, lookback=1024)
    assert (kernel.window, kernel.lookback) == (1152, 1024)


@pytest.mark.parametrize("lookback", [0, -1, 1153, 1024.0])
def test_lookback_outside_the_history_is_rejected(lookback):
    with pytest.raises(ValueError, match="looks back"):
        WindowStageKernel(heads=2, dim=128, capacity=8, window=1152, lookback=lookback)
