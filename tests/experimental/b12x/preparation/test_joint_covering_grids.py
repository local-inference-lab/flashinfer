"""Joint active grids include the shape-covering native work partition."""

import pytest

from b12x.comm.pcie._collective_tuning import CollectiveConfig, CollectiveQuery, TUNING
from b12x.preparation import FrozenMapping, autotune_effort


@pytest.mark.parametrize(
    "mode,active",
    [("bulk_push", 3), ("bulk_pull", 3), ("payload_rsag", 5), ("pull", 5)],
)
def test_shape_covering_joint_grid_is_raced_and_accepts_override(mode, active):
    family = "joint_bandwidth"
    query = CollectiveQuery(
        operation="all_reduce",
        world_size=2,
        rank=0,
        alternatives=FrozenMapping(
            {
                family: dict(
                    surface="OneshotAllReduce.all_reduce",
                    world_size=2,
                    rank=0,
                    topology="pcie_ipc",
                    setup={},
                    call=dict(
                        threads=256,
                        blocks=64,
                        submode="payload_rsag",
                        active_blocks=16,
                        size_packs=1025,
                        stage_input=True,
                        dtype="bfloat16",
                    ),
                )
            }
        ),
        default=CollectiveConfig(
            implementation=family,
            threads=256,
            blocks=64,
            submode="payload_rsag",
            active_blocks=16,
        ),
    )
    expected = CollectiveConfig(
        implementation=family,
        threads=256,
        blocks=64,
        submode=mode,
        active_blocks=active,
    )
    with autotune_effort("default"):
        candidates = {
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        }
        assert expected in candidates
        assert (
            TUNING.configure(query, device=None, override=expected).pinned == expected
        )
    assert {
        config.active_blocks
        for config in candidates
        if config.submode == mode and config.threads == 256
    } == {1, 2, 4, 8, 16, 32, 36, 64, active}
