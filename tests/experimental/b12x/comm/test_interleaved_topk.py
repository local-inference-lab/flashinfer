# SPDX-License-Identifier: Apache-2.0
"""A direct selector borrows only the initialized full-size IPC channel."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import interleaved_topk as selection
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A, _SIGNAL_BYTES


def channel():
    runtime = object.__new__(PCIeDCPA2A)
    runtime.device = torch.device("cuda", 3)
    runtime.rank, runtime.world_size, runtime.max_batch_size = 3, 16, 4
    runtime._closed = False
    runtime._legacy_ext_module = None
    runtime._signal_bytes, runtime._slot_bytes = _SIGNAL_BYTES, 4 * 2112 * 8
    runtime._owned_buffers = ()
    return runtime


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", 8),
        ("max_batch_size", 3),
        ("_signal_bytes", 0),
        ("_signal_bytes", _SIGNAL_BYTES - 1),
        ("_slot_bytes", 4 * 2112 * 8 - 1),
        ("_legacy_ext_module", object()),
        ("_closed", True),
    ],
)
def test_requires_initialized_owned_channel(field, value):
    runtime = channel()
    assert selection.supports_channel(runtime)
    setattr(runtime, field, value)
    assert not selection.supports_channel(runtime)
    with pytest.raises(ValueError, match="owned 16-rank"):
        selection.plan(runtime, cache_pages=1)
    assert not selection.supports_channel(SimpleNamespace())


@pytest.mark.parametrize("pages", [0, -1, 1 << 24, True, 1.5])
def test_page_bounds(pages):
    with pytest.raises(ValueError, match="int32 token slots"):
        selection.plan(channel(), cache_pages=pages)


def test_source_and_resident_ownership_are_part_of_declaration():
    runtime = channel()
    runtime._owned_buffers = [SimpleNamespace(nbytes=811_008)]
    try:
        plan = selection.plan(runtime, cache_pages=(2**31 - 1) // 128)
        assert plan.query.rank == 3 and plan.query.rows == 4
        assert len(plan.invocation["program"]) == 64
        memory = plan._memory_requirements(None, None)
        assert not memory.scratch
        (owner,) = memory.persistent
        assert owner.key == ("pcie-dcp-runtime", id(runtime))
        assert owner.required_nbytes == owner.resident_nbytes == 811_008
    finally:
        runtime._owned_buffers.clear()
