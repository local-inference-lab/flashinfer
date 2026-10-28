"""Default collective search excludes only extremely serialized plain grids."""

from dataclasses import replace

import pytest

from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
from b12x.preparation import FrozenMapping, autotune_effort


_PROTOCOLS = (
    "pull",
    "tp2_remote_push",
    "tp2_remote_push_simple",
    "remote_push_ll",
    "remote_push_simple",
    "remote_push_full",
    "remote_push_rsag",
)


def query(packs, *, world=2, rank=0, stage_input=True, dtype="bfloat16"):
    names = (
        _PROTOCOLS
        if world == 2
        else tuple(n for n in _PROTOCOLS if not n.startswith("tp2_"))
    )
    if not stage_input or dtype == "float32":
        names = ("pull",)
    return CollectiveQuery(
        operation="all_reduce",
        world_size=world,
        rank=rank,
        alternatives=FrozenMapping(
            {
                name: dict(
                    surface="OneshotAllReduce.all_reduce",
                    world_size=world,
                    rank=rank,
                    topology="pcie_ipc",
                    setup={},
                    call=dict(
                        threads=128,
                        blocks=4,
                        dtype=dtype,
                        stage_input=stage_input,
                        size_packs=packs,
                    ),
                )
                for name in names
            }
        ),
        default=CollectiveConfig(implementation="pull", threads=128, blocks=4),
    )


def candidates(value, effort):
    with autotune_effort(effort):
        return {
            config for _, config in TUNING.iterate(TUNING.configure(value, device=None))
        }


@pytest.mark.parametrize("name", _PROTOCOLS)
def test_iteration_bound_preserves_boundary_and_audit_and_pinned_configs(name):
    factor = 2 if name == "tp2_remote_push" else 1
    packs = 1024 * 128 * 4 // factor
    config = CollectiveConfig(implementation=name, threads=128, blocks=4)
    assert config in candidates(query(packs), "default")
    larger = query(packs + 1)
    assert config not in candidates(larger, "default")
    assert config in candidates(larger, "audit")
    with autotune_effort("default"):
        assert TUNING.configure(larger, device=None, override=config).pinned == config
    assert candidates(replace(larger, default=config), "off") == {config}


@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
def test_maximum_grids_keep_every_protocol_and_thread_count_at_large_sizes(world):
    from b12x.comm.pcie.pcie_oneshot import _MAX_BLOCKS

    value = query(1024 * 512 * _MAX_BLOCKS + 1, world=world)
    default = candidates(value, "default")
    audit = candidates(value, "audit")
    assert default < audit
    assert {config.implementation for config in default} == set(value.alternatives)
    for name in value.alternatives:
        assert {
            config.threads for config in default if config.implementation == name
        } == {32, 64, 128, 256, 512}
    assert all(config.blocks == _MAX_BLOCKS for config in default)
    assert (
        candidates(
            replace(
                value,
                rank=world - 1,
                alternatives=FrozenMapping(
                    {
                        name: {**payload, "rank": world - 1}
                        for name, payload in value.alternatives.items()
                    }
                ),
            ),
            "default",
        )
        == default
    )


@pytest.mark.parametrize("packs", (1, 320, 640, 20480))
def test_small_and_decode_payloads_keep_the_complete_search(packs):
    assert candidates(query(packs, world=4), "default") == candidates(
        query(packs, world=4), "audit"
    )


@pytest.mark.parametrize("changes", ({"stage_input": False}, {"dtype": "float32"}))
def test_nonexpanded_plain_surfaces_are_unaffected(changes):
    value = query(1 << 26, **changes)
    assert candidates(value, "default") == candidates(value, "audit")
