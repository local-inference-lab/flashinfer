"""Word exchange reserves counters independently of signals and receive banks."""

import pytest

from b12x.comm.pcie.pcie_dcp_a2a import _staging_layout


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    "rows,local_heads", [(1, 1), (8, 1), (11, 4), (8, 8), (128, 1)]
)
def test_counter_tail_and_receive_banks_are_disjoint(world, rows, local_heads):
    layout = _staging_layout(
        signal_bytes=8192,
        world_size=world,
        max_batch_size=rows,
        total_heads=world * local_heads,
        head_dim=64,
    )
    receive_bytes = layout.exchange_chunks * world * 64 * 8
    assert layout.exchange_offset >= layout.lse_offset + layout.lse_capacity * 4
    assert layout.exchange_offset + receive_bytes == layout.slot_bytes
    assert layout.staging1_offset - layout.staging0_offset == layout.slot_bytes
    counters = layout.staging0_offset + layout.exchange_counter_offset
    assert counters == layout.staging1_offset + layout.slot_bytes
    assert counters + layout.exchange_chunks * 4 <= layout.slab_bytes
    assert layout.slab_bytes - counters == 256
    assert 1 <= layout.exchange_chunks <= 64


def test_unsupported_world_does_not_reserve_packet_state():
    layout = _staging_layout(
        signal_bytes=8192, world_size=8, max_batch_size=8, total_heads=8, head_dim=64
    )
    assert layout.exchange_chunks == 0
    assert layout.slab_bytes == layout.staging1_offset + layout.slot_bytes
