"""Prepared profile geometry for joint BF16 PCIe all-reduce kernels."""

JOINT_MODES = {
    "full": 0,
    "payload_rsag": 1,
    "bulk_push": 2,
    "bulk_pull": 3,
    "pull": 4,
    "full_phased": 5,
    "payload_rsag_phased": 6,
    "bulk_push_phased": 7,
}
# Thread counts a joint entrypoint is compiled and tuned for.
JOINT_THREADS = (32, 64, 128, 256, 512)
# Phased modes move the same data as their base modes, but every CTA of a rank
# writes to one peer at a time, in rank-staggered order. Where peers sit behind
# different CPU host bridges, writes toward several peers at once collapse,
# while one destination per rank keeps link rate; peer reads there stall in
# bursts. Interleaving peers per pack stays faster for small messages on
# switched fabrics, so both orders are prepared profiles.
JOINT_FAMILIES = {
    "joint_latency": (
        "full",
        "payload_rsag",
        "pull",
        "full_phased",
        "payload_rsag_phased",
    ),
    "joint_bandwidth": (
        "payload_rsag",
        "bulk_push",
        "bulk_pull",
        "pull",
        "payload_rsag_phased",
        "bulk_push_phased",
    ),
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
# Barrier counters and flags, local phase arrival and release, epoch and arrival.
JOINT_SIGNAL_BYTES = 266256


def joint_slot_packs(capacity, world):
    """Reserve staging, payload-ready, bulk scatter, and bulk gather regions."""
    if type(capacity) is not int or capacity <= 0 or world not in (2, 3, 4, 6, 8, 10):
        raise ValueError(
            "joint storage requires positive capacity and a supported peer count"
        )
    return (world + 1) * capacity + 2 * world * ((capacity + world - 1) // world)
