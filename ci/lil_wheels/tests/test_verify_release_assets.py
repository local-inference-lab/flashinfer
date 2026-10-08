from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ci.lil_wheels.verify_release_assets import verify_release


COMMIT = "1" * 40
BETA_TAG = f"flashinfer-cu134-sm120-beta-{COMMIT}"


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
    archive = f"flashinfer-cu134-sm120-{COMMIT}.tar.zst"
    (directory / archive).write_bytes(b"archive")
    (directory / f"{archive}.sha256").write_text(
        f"{digest(b'archive')}  /build/{archive}\n"
    )
    if promotion:
        record = {
            "schema": "local-inference-flashinfer-promotion/v1",
            "status": "byte-identical-promotion",
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
    release = tmp_path / "release"
    release.mkdir()
    write_release(release)
    reference = tmp_path / "reference"
    shutil.copytree(release, reference)
    write_release(release, promotion=True)

    verify_release(
        release, COMMIT, BETA_TAG, promotion=True, reference_directory=reference
    )


@pytest.mark.parametrize(
    "names",
    [
        ["install.sh", "runtime.lock"],
        ["flashinfer_python-a.whl", "flashinfer_python-b.whl"],
        ["flashinfer_python-a.txt", "flashinfer_jit_cache-a.whl"],
        ["flashinfer_python-../a.whl", "flashinfer_jit_cache-a.whl"],
    ],
)
def test_rejects_nonwheel_package_inventory(tmp_path, names):
    write_release(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    for package, name in zip(manifest["packages"], names, strict=True):
        package["file"] = name
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="wheel filenames"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)


def test_promotion_requires_independent_beta(tmp_path):
    write_release(tmp_path, promotion=True)
    with pytest.raises(ValueError, match="requires the source beta reference"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=True)


def test_coordinated_stable_substitution_is_rejected(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    write_release(release)
    reference = tmp_path / "reference"
    shutil.copytree(release, reference)
    write_release(release, promotion=True)
    path = release / "install.sh"
    original = digest(path.read_bytes())
    path.write_text("substituted installer")
    sums = release / "SHA256SUMS"
    sums.write_text(sums.read_text().replace(original, digest(path.read_bytes())))
    with pytest.raises(ValueError, match="independent reference mismatch"):
        verify_release(
            release, COMMIT, BETA_TAG, promotion=True, reference_directory=reference
        )


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_nonregular_assets_are_rejected(tmp_path, kind):
    write_release(tmp_path)
    path = tmp_path / "install.sh"
    path.unlink()
    if kind == "symlink":
        path.symlink_to(tmp_path / "runtime.lock")
    else:
        path.mkdir()
    with pytest.raises(ValueError, match="regular files"):
        verify_release(tmp_path, COMMIT, BETA_TAG, promotion=False)
