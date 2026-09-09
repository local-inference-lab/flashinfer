"""Verify snapshot mappings and explicitly selected follow-on source imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--update-manifest",
        type=Path,
        action="append",
        default=[],
        help="Apply a recorded source update after verifying its predecessor hashes.",
    )
    args = parser.parse_args()
    root = args.repo_root.resolve()
    manifest = json.loads((root / "validation/b12x/snapshot_manifest.json").read_text())
    entries = manifest["entries"]
    errors = []
    sources = [entry["source"] for entry in entries]
    destinations = [entry["destination"] for entry in entries]
    if len(set(sources)) != len(entries) or len(set(destinations)) != len(entries):
        errors.append("snapshot mappings must be one-to-one")
    by_source = {entry["source"]: dict(entry) for entry in entries}
    revision = manifest["source_revision"]
    for update_path in args.update_manifest:
        update = json.loads(update_path.read_text())
        if update["parent_source_revision"] != revision:
            errors.append(f"source update has the wrong predecessor: {update_path}")
            continue
        seen = set()
        for replacement in update["entries"]:
            source = replacement["source"]
            previous = by_source.get(source)
            if source in seen or previous is None:
                errors.append(f"duplicate or unknown source update: {source}")
                continue
            seen.add(source)
            if (
                previous["destination"] != replacement["destination"]
                or previous["source_sha256"] != replacement["previous_source_sha256"]
                or previous["destination_sha256"]
                != replacement["previous_destination_sha256"]
            ):
                errors.append(f"source update predecessor mismatch: {source}")
                continue
            by_source[source] = {
                key: replacement[key]
                for key in (
                    "source",
                    "destination",
                    "source_sha256",
                    "destination_sha256",
                    "source_mode",
                )
            }
        revision = update["source_revision"]
    entries = list(by_source.values())
    if args.source_root is not None:
        tracked = (
            subprocess.check_output(["git", "ls-files", "-z"], cwd=args.source_root)
            .decode()
            .split("\0")[:-1]
        )
        if set(tracked) != set(sources):
            errors.append("source tracked-file inventory differs from snapshot")
    for entry in entries:
        destination = root / entry["destination"]
        if not destination.is_file():
            errors.append(f"missing destination: {entry['destination']}")
            continue
        if not destination.resolve().is_relative_to(root):
            errors.append(f"destination escapes checkout: {entry['destination']}")
        expected = entry.get("destination_sha256")
        if (
            expected is not None
            and hashlib.sha256(destination.read_bytes()).hexdigest() != expected
        ):
            errors.append(f"destination changed: {entry['destination']}")
        if (destination.stat().st_mode & 0o777) != int(entry["source_mode"], 8):
            errors.append(f"destination mode changed: {entry['destination']}")
        if args.source_root is not None:
            source = args.source_root / entry["source"]
            if (
                not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest()
                != entry["source_sha256"]
            ):
                errors.append(f"source changed: {entry['source']}")
    if errors:
        for error in errors:
            print(error)
        return 1
    print(
        f"Verified {len(entries)} mappings at {revision}; "
        f"source checked: {args.source_root is not None}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
