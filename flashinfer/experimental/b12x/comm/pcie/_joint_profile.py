"""Prepared profile geometry for joint BF16 PCIe all-reduce kernels."""

JOINT_MODES = {"full": 0, "payload_rsag": 1, "bulk_push": 2, "bulk_pull": 3, "pull": 4}
JOINT_FAMILIES = {
    "joint_latency": ("full", "payload_rsag", "pull"),
    "joint_bandwidth": ("payload_rsag", "bulk_push", "bulk_pull", "pull"),
}


def profile_records(implementation, max_blocks):
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
        for mode in JOINT_FAMILIES[implementation]
        for blocks in range(1, max_blocks + 1)
    )


def profile_index(implementation, max_blocks, submode, active_blocks):
    if type(active_blocks) is not int or not 1 <= active_blocks <= max_blocks:
        raise ValueError("active CTA count must fit the prepared maximum grid")
    if submode not in JOINT_FAMILIES.get(implementation, ()):
        raise ValueError("submode does not belong to the prepared joint family")
    return profile_records(implementation, max_blocks).index(
        (JOINT_MODES[submode], active_blocks)
    )


def selected_profile(implementation, max_blocks, submode, active_blocks):
    """Encode an admitted profile for one device load without changing its identity."""
    index = profile_index(implementation, max_blocks, submode, active_blocks)
    mode, blocks = profile_records(implementation, max_blocks)[index]
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
