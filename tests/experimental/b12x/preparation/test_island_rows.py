"""A four-row epilogue borrows exactly its selected native PCIe mappings."""

from types import SimpleNamespace
import pytest
import torch
from b12x.comm.pcie.island_row import prepared_rows
from b12x.comm.pcie.pcie_island_rs import PCIeIslandRSAllReduce


@pytest.mark.parametrize("rank", range(16))
def test_bind_only_peers_used_by_the_two_switch_dimensions(rank):
    owner = object.__new__(PCIeIslandRSAllReduce)
    owner._closed = False
    owner.rank = rank
    owner.world_size = 16
    owner.blocks = 64
    owner.max_elements = 28672
    owner.quarter_capacity = 16384
    owner._group_timing_records = None
    owner._slab_ptrs = tuple(0x1000000000 + 0x2000000 * i for i in range(16))
    owner.device = torch.device("cpu")
    owner.stage_offset = 98816
    owner.part_offset = 0x200000
    owner.final_offset = 0x400000
    owner.threads = 256
    operand = dict(
        alignment=256, dtype="torch.bfloat16", shape=(4, 3584), stride=(3584, 1)
    )
    q = SimpleNamespace(setup={}, call={"inp": operand, "output": operand})
    state = SimpleNamespace(runtime=owner, query=q)
    plan = SimpleNamespace(
        prepared=SimpleNamespace(
            state=state,
            device=SimpleNamespace(
                identity=SimpleNamespace(
                    vendor="nvidia", compute_capability=(12, 0), sm_count=188
                )
            ),
        )
    )
    layout = prepared_rows(plan, rows=4, width=3584, threads=256)
    src = torch.empty(4, 3584, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    assert tuple(layout.arguments(src, dst)[:4]) == tuple(
        owner._slab_ptrs[(rank // 4) * 4 + i] for i in range(4)
    )
    assert tuple(layout.arguments(src, dst)[4:8]) == tuple(
        owner._slab_ptrs[(rank % 4) + 4 * i] for i in range(4)
    )
    assert layout.rank_coordinates == (rank % 4, rank // 4)
    assert prepared_rows(plan, rows=1, width=3584, threads=256) is None
    operand["alignment"] = 8
    assert prepared_rows(plan, rows=4, width=3584, threads=256) is None
    operand["alignment"] = 256
    q.setup["threads_override"] = True
    assert prepared_rows(plan, rows=4, width=3584, threads=128) is None
    q.setup.clear()
    owner.blocks = 2
    assert prepared_rows(plan, rows=4, width=3584, threads=256) is None
