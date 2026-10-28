"""Prepared mode retention uses the established finalist cache contract."""

from b12x.comm.pcie import joint_finalist_policy, joint_startup_spec
import pytest


def test_joint_finalist_policy_preserves_mode_and_compiled_geometry():
    policy = joint_finalist_policy()
    assert policy.family_fields == ("implementation", "threads", "blocks", "submode")
    assert policy.per_family == 1
    assert policy.max_total == 45


@pytest.mark.parametrize("family", ("joint_latency", "joint_bandwidth"))
def test_serial_push_profiles_require_multiple_peers(family):
    from b12x.comm.pcie._joint_profile import (
        JointMode,
        profile_records,
        selected_profile,
    )

    assert all(
        mode != JointMode.SERIAL_PUSH
        for mode, _ in profile_records(family, 64, world_size=2)
    )
    with pytest.raises(ValueError, match="peer count"):
        selected_profile(family, 64, "serial_push", 1, world_size=2)
    for world in (3, 4, 6, 8, 10):
        assert (JointMode.SERIAL_PUSH, 1) in profile_records(
            family, 64, world_size=world
        )
        assert (
            selected_profile(family, 64, "serial_push", 1, world_size=world)
            == (int(JointMode.SERIAL_PUSH) << 16) | 1
        )
        for blocks in (2, 64):
            with pytest.raises(ValueError, match="exactly one active CTA"):
                selected_profile(family, 64, "serial_push", blocks, world_size=world)


def test_unprepared_joint_startup_rejected():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="prepared joint"):
        joint_startup_spec(SimpleNamespace(prepared=None))
