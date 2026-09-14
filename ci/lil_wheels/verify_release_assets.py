"""Verify the complete immutable asset contract of a FlashInfer release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FIXED_ASSETS = {
    "SHA256SUMS",
    "install.sh",
    "manifest.json",
    "requirements-github.txt",
    "runtime.lock",
}


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_entries(path: Path) -> dict[str, str]:
    """Read sha256sum output and normalize every entry to its basename."""
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, declared_path = line.split(maxsplit=1)
        name = Path(declared_path.lstrip("* ")).name
        if name in entries:
            raise ValueError(f"duplicate checksum basename: {name}")
        entries[name] = digest
    return entries


def verify_release(
    directory: Path,
    source_commit: str,
    beta_tag: str,
    promotion: bool,
    reference_directory: Path | None = None,
) -> None:
    """Verify identity, asset membership, and content hashes for one release."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["source"]["commit"] != source_commit:
        raise ValueError("manifest source commit does not match the requested commit")
    if manifest["release_tag"] != beta_tag:
        raise ValueError("manifest release tag does not match the requested beta tag")

    packages = manifest["packages"]
    package_files = {package["file"] for package in packages}
    if len(packages) != 2 or len(package_files) != 2:
        raise ValueError("manifest must declare exactly two package files")
    for prefix in ("flashinfer_python-", "flashinfer_jit_cache-"):
        matches = [name for name in package_files if name.startswith(prefix)]
        if (
            len(matches) != 1
            or not matches[0].endswith(".whl")
            or not all(
                char.isascii() and (char.isalnum() or char in "_.+-")
                for char in matches[0]
            )
        ):
            raise ValueError(
                "manifest must declare distinct FlashInfer wheel filenames"
            )
    archive = f"flashinfer-cu134-sm120-{source_commit}.tar.zst"
    expected = FIXED_ASSETS | package_files | {archive, f"{archive}.sha256"}
    beta_assets = expected.copy()
    if promotion:
        expected.add("stable-promotion.json")
    actual = {path.name for path in directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"release asset set mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    if any(not path.is_file() or path.is_symlink() for path in directory.iterdir()):
        raise ValueError("release assets must be regular files")

    package_hashes = {
        package["file"]: package["sha256"] for package in manifest["packages"]
    }
    for name, expected_digest in package_hashes.items():
        if sha256(directory / name) != expected_digest:
            raise ValueError(f"package digest mismatch: {name}")

    checksums = checksum_entries(directory / "SHA256SUMS")
    checksummed_assets = package_files | {
        "install.sh",
        "manifest.json",
        "requirements-github.txt",
        "runtime.lock",
    }
    if checksums.keys() != checksummed_assets:
        raise ValueError("SHA256SUMS membership does not match the bundle contract")
    for name, expected_digest in checksums.items():
        if sha256(directory / name) != expected_digest:
            raise ValueError(f"bundle digest mismatch: {name}")

    archive_checksum = checksum_entries(directory / f"{archive}.sha256")
    if archive_checksum.keys() != {archive}:
        raise ValueError("archive checksum must identify exactly the release archive")
    if sha256(directory / archive) != archive_checksum[archive]:
        raise ValueError("release archive digest mismatch")

    if reference_directory is not None:
        if {path.name for path in reference_directory.iterdir()} != beta_assets:
            raise ValueError("reference asset set mismatch")
        for name in beta_assets:
            reference = reference_directory / name
            if not reference.is_file() or reference.is_symlink():
                raise ValueError("reference assets must be regular files")
            if sha256(directory / name) != sha256(reference):
                raise ValueError(f"independent reference mismatch: {name}")
    if promotion:
        if reference_directory is None:
            raise ValueError("stable promotion requires the source beta reference")
        promotion_record = json.loads(
            (directory / "stable-promotion.json").read_text(encoding="utf-8")
        )
        expected_promotion = {
            "schema": "local-inference-flashinfer-promotion/v1",
            "status": "byte-identical-promotion",
            "source_release": beta_tag,
            "source_commit": source_commit,
            "source_manifest_sha256": sha256(manifest_path),
            "invariant": (
                "Wheel files and wheel SHA-256 digests are unchanged from the "
                "source beta release."
            ),
        }
        if promotion_record != expected_promotion:
            raise ValueError("stable promotion record does not match the beta assets")


def main() -> None:
    """Parse the release contract and fail if any declared invariant is false."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--beta-tag", required=True)
    parser.add_argument("--promotion", action="store_true")
    parser.add_argument("--reference-directory", type=Path)
    args = parser.parse_args()
    verify_release(
        args.directory,
        args.source_commit,
        args.beta_tag,
        args.promotion,
        reference_directory=args.reference_directory,
    )


if __name__ == "__main__":
    main()
