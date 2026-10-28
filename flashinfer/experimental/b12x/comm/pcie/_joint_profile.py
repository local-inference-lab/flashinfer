"""Prepared profile geometry for joint BF16 PCIe all-reduce kernels."""

from enum import IntEnum


class JointMode(IntEnum):
    FULL = 0
    PAYLOAD_RSAG = 1
    BULK_PUSH = 2
    BULK_PULL = 3
    PULL = 4
    SERIAL_PUSH = 5


JOINT_MODES = {mode.name.lower(): int(mode) for mode in JointMode}
JOINT_FAMILIES = {
    "joint_latency": ("full", "payload_rsag", "pull", "serial_push"),
    "joint_bandwidth": (
        "payload_rsag",
        "bulk_push",
        "bulk_pull",
        "pull",
        "serial_push",
    ),
}


def joint_modes(implementation, world_size):
    """Admit destination serialization only when a rank has multiple peers."""
    return tuple(
        mode
        for mode in JOINT_FAMILIES.get(implementation, ())
        if world_size > 2 or mode != "serial_push"
    )


def profile_records(implementation, max_blocks, *, world_size):
    if (
        implementation not in JOINT_FAMILIES
        or type(max_blocks) is not int
        or not 1 <= max_blocks <= 64
    ):
        raise ValueError(
            "joint profiles require a latency/bandwidth family and 1–64 CTAs"
        )
    return tuple(
        (JOINT_MODES[mode], blocks)
        for mode in joint_modes(implementation, world_size)
        for blocks in ((1,) if mode == "serial_push" else range(1, max_blocks + 1))
    )


def profile_index(implementation, max_blocks, submode, active_blocks, *, world_size):
    if type(active_blocks) is not int or not 1 <= active_blocks <= max_blocks:
        raise ValueError("active CTA count must fit the prepared maximum grid")
    if submode not in joint_modes(implementation, world_size):
        raise ValueError("submode is not admitted for the joint family and peer count")
    if submode == "serial_push" and active_blocks != 1:
        raise ValueError("serial-destination push requires exactly one active CTA")
    return profile_records(implementation, max_blocks, world_size=world_size).index(
        (JOINT_MODES[submode], active_blocks)
    )


def selected_profile(implementation, max_blocks, submode, active_blocks, *, world_size):
    """Encode an admitted profile for one device load without changing its identity."""
    index = profile_index(
        implementation, max_blocks, submode, active_blocks, world_size=world_size
    )
    mode, blocks = profile_records(implementation, max_blocks, world_size=world_size)[
        index
    ]
    return (mode << 16) | blocks


JOINT_MAX_BLOCKS = 64
JOINT_MAX_RANKS = 16
JOINT_SIGNAL_BYTES = 266248


def joint_slot_packs(capacity, world):
    """Reserve staging, payload-ready, bulk scatter, and bulk gather regions."""
    if type(capacity) is not int or capacity <= 0 or world not in (2, 3, 4, 6, 8, 10):
        raise ValueError(
            "joint storage requires positive capacity and a supported peer count"
        )
    return (world + 1) * capacity + 2 * world * ((capacity + world - 1) // world)
