"""Selected gather plans validate exact layouts and channel-owned receive storage."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.comm.pcie._gather_preparation import plan, selected_gather_profile
from b12x.comm.pcie._tuning import PcieQuery, TUNING
from b12x.comm.pcie.pcie_dcp_a2a import _SIGNAL_BYTES
from b12x.preparation import FrozenMapping


def query():
    return PcieQuery(
        surface="DcpAllToAll.all_gather_heads",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                shape=(64, 1, 64),
                stride=(64, 64, 1),
                out_shape=(64, 4, 64),
                out_stride=(256, 64, 1),
                dtype="torch.bfloat16",
                threads=256,
                block_limit=32,
                active_blocks=4,
                submode="barrier",
                gather_transport="joint",
                disjoint_output=True,
                pack_aligned=True,
            )
        ),
        setup=FrozenMapping(
            dict(
                max_batch_size=256,
                total_heads=4,
                query_head_dim=64,
                exchange_chunks=64,
                exchange_offset=131072,
                slot_bytes=262144,
                exchange_counter_offset=524288,
                signal_bytes=_SIGNAL_BYTES,
            )
        ),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"shape": (64, 64)},
        {"shape": (0, 1, 64)},
        {"dtype": "torch.float32"},
        {"out_stride": (256, 1, 4)},
        {"pack_aligned": False},
        {"disjoint_output": False},
        {"active_blocks": 33},
        {"submode": "pull"},
        {"threads": 96},
        {"shape": (512, 1, 64)},
        {"shape": (64, 2, 64)},
    ],
)
def test_selected_gather_rejects_incompatible_call(changes):
    q = query()
    with pytest.raises(ValueError):
        TUNING.validate_query(
            replace(q, call=FrozenMapping(dict(q.call, **changes))), None
        )


def test_selected_gather_requires_matching_receive_owner():
    q = query()
    TUNING.validate_query(q, None)
    runtime = SimpleNamespace(
        max_batch_size=256,
        total_heads=4,
        query_head_dim=64,
        _exchange_chunks=32,
        _exchange_offset=131072,
        _slot_bytes=262144,
        _exchange_counter_offset=524288,
    )
    with pytest.raises(ValueError, match="exchange_chunks differs"):
        plan(q, runtime=runtime, invocation=FrozenMapping(), override=None)


@pytest.mark.parametrize(
    "transport,mode",
    [
        ("joint", "barrier"),
        ("joint", "word_exchange"),
        ("selected_barrier", "barrier"),
        ("selected_word_exchange", "word_exchange"),
    ],
)
def test_selected_gather_profiles_encode_mode_and_active_grid(transport, mode):
    assert (
        selected_gather_profile(transport, 32, mode, 8)
        == (int(mode == "word_exchange") << 16) | 8
    )
    with pytest.raises(ValueError):
        selected_gather_profile(transport, 32, mode, 33)


@pytest.mark.parametrize("world", [2, 4, 8, 9, 10, 12, 16])
@pytest.mark.parametrize("threads", [32, 64, 128, 256, 512])
@pytest.mark.parametrize(
    "transport", ["selected_phased_barrier", "selected_phased_push"]
)
def test_phased_gather_requires_a_complete_cluster(world, threads, transport):
    q = query()
    q = replace(
        q,
        world_size=world,
        setup=FrozenMapping(dict(q.setup, total_heads=world)),
        call=FrozenMapping(
            dict(
                q.call,
                threads=threads,
                block_limit=8,
                active_blocks=8,
                out_shape=(64, world, 64),
                out_stride=(world * 64, 64, 1),
                gather_transport=transport,
            )
        ),
    )
    TUNING.validate_query(q, None)
    assert selected_gather_profile(transport, 8, "barrier", 8) == 8
    for blocks, active in ((8, 1), (8, 4), (16, 8), (64, 8), (7, 7), (8, 9)):
        with pytest.raises(ValueError):
            TUNING.validate_query(
                replace(
                    q,
                    call=FrozenMapping(
                        dict(q.call, block_limit=blocks, active_blocks=active)
                    ),
                ),
                None,
            )


@pytest.mark.parametrize("origin", [0, 1])
@pytest.mark.parametrize("entered", [False, True])
@pytest.mark.parametrize(
    "transport",
    [
        "selected_barrier",
        "selected_phased_barrier",
        "selected_phased_push",
        "joint",
        "selected_word_exchange",
    ],
)
def test_selected_calls_preserve_the_channel_bank_origin(origin, entered, transport):
    import torch
    from b12x.comm.pcie._gather_preparation import SelectedGatherState

    runtime = SimpleNamespace(
        _device_slot_selection=entered,
        _next_slot=1 - origin if entered else origin,
        _graph_base_slot=origin if entered else 0,
        _staging_ptrs=((0x10000, 0x20000), (0x11000, 0x21000)),
        _signal_ptrs=(0x30000, 0x40000),
        _slot_bytes=4096,
        _exchange_offset=1024,
        _exchange_counter_offset=8192,
    )
    source = torch.zeros((1, 1, 64), dtype=torch.bfloat16)
    output = torch.empty((1, 2, 64), dtype=source.dtype)
    q = replace(
        query(),
        call=FrozenMapping(
            dict(
                query().call,
                shape=tuple(source.shape),
                dtype=str(source.dtype),
                gather_transport=transport,
            )
        ),
    )
    launches = []
    state = SelectedGatherState(
        q,
        runtime,
        lambda *args: launches.append(args),
        torch.zeros(1, dtype=torch.uint32),
    )
    state.run_selected(source, output)
    runtime._next_slot ^= 1
    state.run_selected(source, output)
    bank = 0 if transport == "selected_word_exchange" else origin
    assert len(launches) == 2
    for args in launches:
        assert args[3] == runtime._staging_ptrs[bank]
        assert args[6] == (4096 if bank == 0 else -4096)
        assert args[7:] == (1024, 8192)
    if transport != "selected_word_exchange":
        assert runtime._device_slot_selection and runtime._graph_base_slot == origin


@pytest.mark.parametrize("transport", ["selected_ring", "selected_recursive", "selected_recursive_ready"])
@pytest.mark.parametrize("world", [8, 16])
def test_pairwise_gather_requires_its_explicit_geometry(transport, world):
    q = query()
    q = replace(
        q,
        world_size=world,
        setup=FrozenMapping(dict(q.setup, total_heads=world)),
        call=FrozenMapping(
            dict(
                q.call,
                gather_transport=transport,
                threads=128,
                block_limit=8,
                active_blocks=8,
                out_shape=(64, world, 64),
                out_stride=(64 * world, 64, 1),
            )
        ),
    )
    TUNING.validate_query(q, None)
    assert selected_gather_profile(transport, 8, "barrier", 8) == 8
    for change in (
        {"threads": 32},
        {"threads": 512},
        {"active_blocks": 7},
        {"block_limit": 16},
        {"submode": "word_exchange"},
    ):
        with pytest.raises(ValueError):
            TUNING.validate_query(
                replace(q, call=FrozenMapping(dict(q.call, **change))), None
            )
    for world in (2, 4, 12):
        with pytest.raises(ValueError):
            TUNING.validate_query(replace(q, world_size=world), None)
