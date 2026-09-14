"""Validate subordinate-ID ranges before configuring a rootless Docker user."""

import argparse
from pathlib import Path


def validate_ranges(text: str, user: str, *, candidate: bool = False) -> None:
    """Require non-overlapping mappings and 65,536 IDs within Docker's five ranges.

    Candidate mode checks whether the provisioning range is unused without
    modifying the host. Invalid mappings raise ValueError before daemon startup.
    """
    entries = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        owner, first, count = line.split(":")
        start, size = int(first), int(count)
        if start < 0 or size <= 0 or start + size > 2**32:
            raise ValueError("invalid subordinate-ID range")
        entries.append((owner, start, start + size))
    if candidate:
        entries.append((user, 200000, 265536))
    selected = [entry for entry in entries if entry[0] == user]
    if sum(end - start for _, start, end in selected[:5]) < 65536:
        raise ValueError(
            "Docker requires at least 65536 subordinate IDs within its first five mappings"
        )
    for index, (owner, start, end) in enumerate(entries):
        for other_owner, other_start, other_end in entries[index + 1 :]:
            if user in (owner, other_owner) and start < other_end and other_start < end:
                raise ValueError(
                    "subordinate-ID mappings overlap; assign an unused range"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args()
    validate_ranges(args.file.read_text(), args.user, candidate=args.candidate)
