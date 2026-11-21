# SPDX-License-Identifier: Apache-2.0
"""A direct selector borrows only the initialized full-size IPC channel."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm import interleaved_topk as selection
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A, _SIGNAL_BYTES


def test_pcie_import_keeps_the_same_selector_api():
    from b12x.comm.pcie import interleaved_topk

    assert interleaved_topk.InterleavedTopKQuery is selection.InterleavedTopKQuery
    assert interleaved_topk.plan is selection.plan
    assert interleaved_topk.supports_channel is selection.supports_channel


def channel(*, world_size=16, rows=4):
    runtime = object.__new__(PCIeDCPA2A)
    runtime.device = torch.device("cuda", 3)
    runtime.rank, runtime.world_size, runtime.max_batch_size = min(3, world_size - 1), world_size, rows
    runtime._closed = False
    runtime._legacy_ext_module = None
    runtime._signal_bytes, runtime._slot_bytes = _SIGNAL_BYTES, rows * 2112 * 8
    runtime._owned_buffers = ()
    return runtime


@pytest.mark.parametrize("world_size", [4, 8, 16])
@pytest.mark.parametrize("rows", [1, 4, 32, 64])
def test_supported_worlds_and_row_counts(world_size, rows):
    runtime = channel(world_size=world_size, rows=rows)
    assert selection.supports_channel(runtime, rows=rows)
    plan = selection.plan(runtime, cache_pages=16, rows=rows)
    assert plan.query.world == world_size
    assert plan.query.rows == rows


@pytest.mark.parametrize("rows", [0, -1, 65, True, 4.0])
def test_invalid_row_counts_rejected(rows):
    runtime = channel(rows=64)
    assert not selection.supports_channel(runtime, rows=rows)
    with pytest.raises(ValueError, match="owned 4/8/16-rank"):
        selection.plan(runtime, cache_pages=1, rows=rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("world_size", 2),
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
    with pytest.raises(ValueError, match="owned 4/8/16-rank"):
        selection.plan(runtime, cache_pages=1)
    assert not selection.supports_channel(SimpleNamespace())


@pytest.mark.parametrize("pages", [0, -1, 1 << 24, True, 1.5])
def test_page_bounds(pages):
    with pytest.raises(ValueError, match="int32 token slots"):
        selection.plan(channel(), cache_pages=pages)


def test_compile_job_and_resident_ownership_are_part_of_declaration():
    runtime = channel()
    runtime._owned_buffers = [SimpleNamespace(nbytes=811_008)]
    try:
        plan = selection.plan(runtime, cache_pages=(2**31 - 1) // 128)
        assert plan.query.rank == 3 and plan.query.rows == 4
        assert plan.contract.default_config(plan.query, None).backend == "cute"
        (job,) = plan._compile_jobs(None, None)
        assert job.factory == "b12x.comm.interleaved_topk._cute:compile_interleaved_topk_slots"
        assert job.args[0] == 16
        memory = plan._memory_requirements(None, None)
        assert not memory.scratch
        (owner,) = memory.persistent
        assert owner.key == ("pcie-dcp-runtime", id(runtime))
        assert owner.required_nbytes == owner.resident_nbytes == 811_008
    finally:
        runtime._owned_buffers.clear()
