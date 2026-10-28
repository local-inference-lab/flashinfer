"""Capture contexts protect the lifetime of the fixed DMA channel."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce


def channel():
    runtime = PCIeDmaAllReduce.__new__(PCIeDmaAllReduce)
    runtime.device = torch.device("cuda:0")
    runtime._closed = False
    runtime._capture_context_depth = 0
    return runtime


def test_capture_rejects_closed_channel_and_foreign_stream():
    runtime = channel()
    with pytest.raises(ValueError, match="channel device"):
        with runtime.capture(stream=SimpleNamespace(device=torch.device("cuda:1"))):
            pass
    runtime._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        with runtime.capture():
            pass


def test_capture_retains_owner_and_releases_guard_after_failure():
    runtime = channel()
    stream = SimpleNamespace(device=runtime.device)
    with pytest.raises(ValueError, match="recording failed"):
        with runtime.capture(stream=stream) as owner:
            assert owner is runtime
            with pytest.raises(RuntimeError, match="overlapping"):
                with runtime.capture(stream=stream):
                    pass
            with pytest.raises(RuntimeError, match="during capture"):
                runtime.close()
            assert not runtime._closed
            raise ValueError("recording failed")
    with runtime.capture(stream=stream) as owner:
        assert owner is runtime
