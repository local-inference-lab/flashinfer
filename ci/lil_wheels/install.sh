#!/usr/bin/env bash
# Install an immutable FlashInfer release into an existing Python environment.
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin=${1:-python}

(cd "${bundle_dir}" && sha256sum --check SHA256SUMS)
"${python_bin}" -m pip install --no-deps --force-reinstall "${bundle_dir}"/wheels/*.whl
"${python_bin}" - <<'PY'
import flashinfer
import flashinfer_jit_cache

assert flashinfer.__version__ == flashinfer_jit_cache.__version__
print(f"FlashInfer {flashinfer.__version__}")
print(f"source commit {flashinfer.__git_commit__}")
PY
