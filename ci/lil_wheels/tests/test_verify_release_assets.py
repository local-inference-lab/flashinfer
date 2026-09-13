from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ci.lil_wheels.verify_release_assets import verify_release


COMMIT = "1" * 40
BETA_TAG = f"flashinfer-cu133-sm120-beta-{COMMIT}"


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_release(directory: Path, promotion: bool = False) -> None:
    wheels = {
        "flashinfer_python-0.6.18-py3-none-any.whl": b"python wheel",
        "flashinfer_jit_cache-0.6.18-py3-none-any.whl": b"jit wheel",
    }
    for name, payload in wheels.items():
        (directory / name).write_bytes(payload)
    manifest = {
        "source": {"commit": COMMIT},
        "release_tag": BETA_TAG,
        "packages": [
            {"file": name, "sha256": digest(payload)}
            for name, payload in wheels.items()
        ],
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    for name in ("install.sh", "requirements-github.txt", "runtime.lock"):
        (directory / name).write_text(name)
    checksum_names = list(wheels) + [
        "install.sh",
        "manifest.json",
        "requirements-github.txt",
        "runtime.lock",
    ]
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{digest((directory / name).read_bytes())}  {name}\n"
            for name in checksum_names
        )
    )
    archive = f"flashinfer-cu133-sm120-{COMMIT}.tar.zst"
    (directory / archive).write_bytes(b"archive")
    (directory / f"{archive}.sha256").write_text(
        f"{digest(b'archive')}  /build/{archive}\n"
    )
    if promotion:
        record = {
            "schema": "local-inference-flashinfer-promotion/v1",
            "status": "qualified",
            "source_release": BETA_TAG,
            "source_commit": COMMIT,
            "source_manifest_sha256": digest(
                (directory / "manifest.json").read_bytes()
            ),
            "invariant": (
                "Wheel files and wheel SHA-256 digests are unchanged from the "
                "source beta release."
            ),
        }
        (directory / "stable-promotion.json").write_text(json.dumps(record))


def test_accepts_complete_beta_release(tmp_path):
    write_release(tmp_path)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_rejects_extra_asset(tmp_path):
    write_release(tmp_path)
    (tmp_path / "unexpected").write_text("untrusted")

    with pytest.raises(ValueError, match="asset set mismatch"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_accepts_complete_stable_promotion(tmp_path):
    write_release(tmp_path, promotion=True)

    verify_release(tmp_path, COMMIT, BETA_TAG, promotion=True)
