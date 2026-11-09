from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie.pcie_dcp_topk import (
    PCIeDCPTopKOwnerExchange,
    _SIGNAL_BYTES,
    _candidate_staging_layout,
    owner_stage_reference,
)
from b12x.comm.pcie._dcp_cute_common import signal_bytes


class _PreparedCPUState:
    def __init__(self, owner, rows, threads=512, block_limit=128):
        self.owner = self.runtime = owner
        self.query = SimpleNamespace(
            surface="DcpTopKOwnerExchange.stage_candidates",
            call=dict(rows=rows, threads=threads, block_limit=block_limit),
        )

    def require_runtime(self, owner):
        assert owner is self.owner

    def launcher(self):
        return None


@pytest.fixture(autouse=True)
def prepared_cpu_dispatch(monkeypatch):
    # Geometry/slot tests use CPU planes. Actual compilation and plan binding
    # are exercised by the CUDA integration test.
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_topk.require_prepared", lambda plan, *_: plan
    )


class _FakeOwner(PCIeDCPTopKOwnerExchange):
    def __init__(self) -> None:
        super().__init__(
            rank=1,
            world_size=2,
            device="cpu",
            signal_ptrs=(10, 20),
            staging0_ptrs=(30, 40),
            staging1_ptrs=(62, 72),
            max_rows=8,
            topk=4,
        )
        shape = (self.max_owner_rows, self.world_size * self.topk)
        self._candidate_views = tuple(
            (
                torch.empty(shape, dtype=torch.int32),
                torch.empty(shape, dtype=torch.float32),
            )
            for _ in range(2)
        )
        self.stage_calls: list[tuple] = []

    def stage_candidates(
        self, indices, scores, *, plan=None, threads=None, block_limit=None
    ):
        if plan is None:
            plan = _PreparedCPUState(
                self, indices.shape[0], threads or 512, block_limit or 128
            )
        return super().stage_candidates(
            indices, scores, plan=plan, threads=threads, block_limit=block_limit
        )

    def capture(self, *, plan=None):
        return super().capture(plan=plan or _PreparedCPUState(self, self.max_rows))

    def _launch_stage(
        self,
        launcher,
        local_indices,
        local_scores,
        *,
        slot,
        rows,
        threads,
        blocks,
        wait_for_prior_consumer,
    ):
        owner_rows = -(-rows // self.world_size)
        indices, scores = owner_stage_reference(
            torch.stack([local_indices] * self.world_size),
            torch.stack([local_scores] * self.world_size),
            self.rank,
        )
        views = self._candidate_views[slot]
        views[0][:owner_rows].copy_(indices)
        views[1][:owner_rows].copy_(scores)
        self.stage_calls.append((slot, threads, blocks, wait_for_prior_consumer))


def _make_owner() -> PCIeDCPTopKOwnerExchange:
    return _FakeOwner()


def test_candidate_staging_layout_is_aligned():
    owner = _candidate_staging_layout(
        signal_bytes=513,
        max_rows=8,
        topk=4,
        world_size=2,
    )
    assert owner.staging0_offset == 768
    assert owner.plane_bytes == 128
    assert owner.slot_bytes == 256
    assert owner.staging1_offset == 1024
    assert owner.slab_bytes == 1280


def test_topk_signal_layout_matches_the_shared_barrier_abi():
    assert signal_bytes(128) == _SIGNAL_BYTES


def test_owner_stage_reference_preserves_rank_major_order_and_score_bits():
    world_size, rows, topk = 4, 8, 4
    indices = torch.empty(world_size, rows, topk, dtype=torch.int32)
    score_bits = torch.empty(world_size, rows, topk, dtype=torch.int32)
    for rank in range(world_size):
        for row in range(rows):
            indices[rank, row].fill_(1000 * rank + 10 * row)
            score_bits[rank, row].fill_(0x3F000000 + rank * 16 + row)
    scores = score_bits.view(torch.float32)

    owner_indices, owner_scores = owner_stage_reference(indices, scores, 2)

    assert owner_indices.shape == (2, world_size * topk)
    assert owner_scores.shape == owner_indices.shape
    for owner_row, global_row in enumerate((4, 5)):
        for rank in range(world_size):
            rank_slice = slice(rank * topk, (rank + 1) * topk)
            assert torch.equal(
                owner_indices[owner_row, rank_slice], indices[rank, global_row]
            )
            assert torch.equal(
                owner_scores[owner_row, rank_slice].view(torch.int32),
                score_bits[rank, global_row],
            )


def test_owner_dispatches_and_disposes():
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10

    candidate_indices, candidate_scores = owner.stage_candidates(
        indices,
        scores,
        threads=128,
        block_limit=32,
    )
    assert torch.equal(candidate_indices, indices[4:].repeat(1, 2))
    assert torch.equal(candidate_scores, scores[4:].repeat(1, 2))
    assert owner.stage_calls == [(0, 128, 1, False)]

    owner.close()
    assert owner._closed


@pytest.mark.parametrize("eager_calls, expected_slot", [(0, 0), (1, 1), (2, 0)])
def test_graph_capture_pins_the_next_staging_slot_and_enables_prior_wait(
    monkeypatch,
    eager_calls,
    expected_slot,
):
    owner = _make_owner()
    indices = torch.arange(32, dtype=torch.int32).reshape(8, 4)
    scores = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    capture_state = False
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_topk._is_current_stream_capturing",
        lambda _device: capture_state,
    )

    for _ in range(eager_calls):
        owner.stage_candidates(indices, scores)
    capture_state = True
    with owner.capture():
        first_indices, first_scores = owner.stage_candidates(indices, scores)
    capture_state = False
    second_indices, second_scores = owner.stage_candidates(indices + 100, scores + 1)
    capture_state = True
    with owner.capture():
        third_indices, third_scores = owner.stage_candidates(indices + 200, scores + 2)

    assert owner._graph_slot == expected_slot
    assert first_indices.data_ptr() == second_indices.data_ptr()
    assert second_indices.data_ptr() == third_indices.data_ptr()
    assert first_scores.data_ptr() == second_scores.data_ptr()
    assert second_scores.data_ptr() == third_scores.data_ptr()
    eager_stages = [(slot % 2, 512, 1, False) for slot in range(eager_calls)]
    assert owner.stage_calls == eager_stages + [
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
        (expected_slot, 512, 1, True),
    ]
    assert torch.equal(third_indices, (indices + 200)[4:].repeat(1, 2))


def test_owner_rejects_invalid_contracts():
    owner = _make_owner()
    indices = torch.zeros(8, 4, dtype=torch.int32)
    scores = torch.zeros(8, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices, scores[:4])
    with pytest.raises(ValueError, match="matching"):
        owner.stage_candidates(indices.flatten(), scores.flatten())
    with pytest.raises(ValueError, match="local_indices must be int32"):
        owner.stage_candidates(indices.float(), scores)
    with pytest.raises(ValueError, match="local_scores must be float32"):
        owner.stage_candidates(indices, scores.bfloat16())
    with pytest.raises(ValueError, match="rows must"):
        owner.stage_candidates(indices[:0], scores[:0])
    with pytest.raises(ValueError, match="threads"):
        owner.stage_candidates(indices, scores, threads=31)
    with pytest.raises(ValueError, match="block_limit"):
        owner.stage_candidates(indices, scores, block_limit=129)


def test_configuration_rejects_invalid_capacity_and_topk():
    with pytest.raises(ValueError, match="multiple of 4"):
        _candidate_staging_layout(
            signal_bytes=256,
            max_rows=8,
            topk=6,
            world_size=2,
        )
    with pytest.raises(ValueError, match="max_rows must be positive"):
        PCIeDCPTopKOwnerExchange(
            rank=0,
            world_size=4,
            device="cpu",
            signal_ptrs=(1, 2, 3, 4),
            staging0_ptrs=(5, 6, 7, 8),
            staging1_ptrs=(9, 10, 11, 12),
            max_rows=0,
            topk=4,
        )


@pytest.mark.parametrize("world", [2, 3, 4, 6, 8, 16])
@pytest.mark.parametrize("rows", [1, 3, 7, 8, 17, 32])
def test_ragged_owner_rows_preserve_rank_order_and_pad_only_missing_rows(world, rows):
    topk = 4
    indices = torch.arange(world * rows * topk, dtype=torch.int32).view(
        world, rows, topk
    )
    bits = (indices + 0x3F000000).to(torch.int32)
    owner_rows = -(-rows // world)
    for owner in range(world):
        got_ids, got_scores = owner_stage_reference(
            indices, bits.view(torch.float32), owner
        )
        assert got_ids.shape == (owner_rows, world * topk)
        for local in range(owner_rows):
            global_row = owner * owner_rows + local
            if global_row < rows:
                assert torch.equal(
                    got_ids[local].view(world, topk), indices[:, global_row]
                )
                assert torch.equal(
                    got_scores[local].view(torch.int32).view(world, topk),
                    bits[:, global_row],
                )
            else:
                assert (got_ids[local] == -1).all()
                assert (got_scores[local] == -torch.inf).all()


def test_graph_outputs_are_stable_before_capture_and_do_not_execute_transport():
    owner = _make_owner()
    indices = torch.arange(28, dtype=torch.int32).view(7, 4)
    scores = indices.float()
    reserved = owner.graph_candidate_views(7)
    assert owner.stage_calls == []
    result = owner.stage_candidates(indices, scores)
    assert tuple(t.data_ptr() for t in reserved) == tuple(t.data_ptr() for t in result)
    assert owner.stage_calls == [(0, 512, 1, True)]
    assert (result[0][-1] == -1).all() and (result[1][-1] == -torch.inf).all()
    owner.close()
    with pytest.raises(RuntimeError, match="open CUDA"):
        owner.graph_candidate_views(7)


def test_prepared_call_captures_materialized_state_before_plan_selection(monkeypatch):
    from b12x.comm.pcie._owner_preparation import prepared_call

    owner = _make_owner()
    ids = torch.arange(28, dtype=torch.int32).view(7, 4)
    scores = ids.float()
    owner.graph_candidate_views(7)
    state = _PreparedCPUState(owner, 7)
    call = prepared_call(state, local_indices=ids, local_scores=scores)
    monkeypatch.setattr(
        "b12x.comm.pcie.pcie_dcp_topk._is_current_stream_capturing", lambda _: True
    )
    assert owner._capture_context_depth == 0
    with call.capture_context():
        assert owner._capture_context_depth == 1
        call.run()
    assert owner._capture_context_depth == 0
    assert owner.stage_calls == [(0, 512, 1, True)]
