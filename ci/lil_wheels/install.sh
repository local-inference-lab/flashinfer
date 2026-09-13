#!/usr/bin/env bash
# Install an immutable FlashInfer release into an existing Python environment.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin=${1:-python}

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${python_bin}" -m pip install --no-deps --force-reinstall "${bundle_dir}"/wheels/*.whl
BUNDLE_DIR="${bundle_dir}" "${python_bin}" -I - <<'PY'
import importlib.metadata
import json
import os
from pathlib import Path

import flashinfer
import flashinfer_jit_cache

manifest = json.loads((Path(os.environ["BUNDLE_DIR"]) / "manifest.json").read_text())
expected_version = manifest["package_version"]
expected_commit = manifest["source"]["commit"]
assert flashinfer.__version__ == expected_version
assert flashinfer.__git_commit__ == expected_commit
assert flashinfer_jit_cache.__version__ == expected_version
assert flashinfer_jit_cache.__git_version__ == expected_commit
assert importlib.metadata.version("flashinfer-python") == expected_version
assert importlib.metadata.version("flashinfer-jit-cache") == expected_version
print(f"FlashInfer {flashinfer.__version__}")
print(f"source commit {flashinfer.__git_commit__}")
PY
