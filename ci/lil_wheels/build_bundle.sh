#!/usr/bin/env bash
# Build and verify source-addressed FlashInfer wheels for CUDA 13.3 and SM120.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tool_dir="${repo_root}/ci/lil_wheels"
lock_path="${tool_dir}/runtime.lock"
output_dir=${1:-"${repo_root}/dist/lil-flashinfer-wheels"}

lock_value() {
  local key=$1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

source_commit=$(git -C "${repo_root}" rev-parse HEAD)
source_tree=$(git -C "${repo_root}" rev-parse 'HEAD^{tree}')
source_date_epoch=$(git -C "${repo_root}" show -s --format=%ct HEAD)
base_version=$(tr -d '[:space:]' < "${repo_root}/version.txt")
local_version="lil.cu133.sm120.g${source_commit:0:12}"
package_version="${base_version}+${local_version}"
builder=$(lock_value buildx.builder)
repository=${GITHUB_REPOSITORY:-local-inference-lab/flashinfer}
release_tag=${FLASHINFER_RELEASE_TAG:-"flashinfer-cu133-sm120-beta-${source_commit}"}

test -z "$(git -C "${repo_root}" status --porcelain)"
"${tool_dir}/ensure_builder.sh"

mkdir -p "$(dirname "${output_dir}")"
if ! mkdir "${output_dir}"; then
  printf 'Output path already exists or is being built: %s\n' \
    "${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}/raw" "${output_dir}/bundle/wheels"

docker buildx build \
  --builder "${builder}" \
  --file "${tool_dir}/Dockerfile" \
  --build-arg "BUILDER_IMAGE=$(lock_value builder.image)" \
  --build-arg "FLASHINFER_LOCAL_VERSION=${local_version}" \
  --build-arg "FLASHINFER_SOURCE_COMMIT=${source_commit}" \
  --build-arg "FLASHINFER_SOURCE_DATE_EPOCH=${source_date_epoch}" \
  --build-arg "MAX_JOBS=$(lock_value build.max-jobs)" \
  --build-arg "NVCC_THREADS=$(lock_value build.nvcc-threads)" \
  --target wheels \
  --output "type=local,dest=${output_dir}/raw" \
  "${repo_root}"

cp -a "${output_dir}/raw/wheels/." "${output_dir}/bundle/wheels/"
python_wheel=$(find "${output_dir}/bundle/wheels" -maxdepth 1 -name 'flashinfer_python-*.whl' -print -quit)
jit_wheel=$(find "${output_dir}/bundle/wheels" -maxdepth 1 -name 'flashinfer_jit_cache-*.whl' -print -quit)
test -n "${python_wheel}"
test -n "${jit_wheel}"

wheel_metadata() {
  python3 - "$1" "$2" <<'PY'
import email
import sys
import zipfile

wheel, field = sys.argv[1:]
with zipfile.ZipFile(wheel) as archive:
    metadata_paths = [
        name
        for name in archive.namelist()
        if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_paths) != 1:
        raise RuntimeError(f"Expected one METADATA file in {wheel}, found {metadata_paths}")
    message = email.message_from_bytes(archive.read(metadata_paths[0]))
print(message[field])
PY
}

test "$(wheel_metadata "${python_wheel}" Name)" = flashinfer-python
test "$(wheel_metadata "${python_wheel}" Version)" = "${package_version}"
test "$(wheel_metadata "${jit_wheel}" Name)" = flashinfer-jit-cache
test "$(wheel_metadata "${jit_wheel}" Version)" = "${package_version}"

packages_json='[]'
requirements="${output_dir}/bundle/requirements-github.txt"
: > "${requirements}"
while IFS= read -r wheel; do
  name=$(wheel_metadata "${wheel}" Name)
  version=$(wheel_metadata "${wheel}" Version)
  file=$(basename "${wheel}")
  digest=$(sha256sum "${wheel}" | awk '{print $1}')
  url="https://github.com/${repository}/releases/download/${release_tag}/${file}"
  printf '%s @ %s --hash=sha256:%s\n' "${name}" "${url}" "${digest}" >> "${requirements}"
  packages_json=$(jq \
    --arg name "${name}" --arg version "${version}" --arg file "${file}" \
    --arg sha256 "${digest}" --arg url "${url}" \
    '. + [{name: $name, version: $version, file: $file, sha256: $sha256, url: $url}]' \
    <<<"${packages_json}")
done < <(find "${output_dir}/bundle/wheels" -maxdepth 1 -name '*.whl' | sort)

submodules_json=$(git -C "${repo_root}" submodule status --recursive | python3 -c '
import json
import sys

entries = []
for line in sys.stdin:
    state = line[0]
    commit, path, *_ = line[1:].split()
    entries.append({"path": path, "commit": commit, "status": state})
print(json.dumps(entries))
')

jq -n \
  --arg schema local-inference-flashinfer-wheel-release/v1 \
  --arg status research-only \
  --arg repository "https://github.com/${repository}.git" \
  --arg commit "${source_commit}" \
  --arg tree "${source_tree}" \
  --arg package_version "${package_version}" \
  --arg release_tag "${release_tag}" \
  --arg builder_image "$(lock_value builder.image)" \
  --arg python_version "$(lock_value python.version)" \
  --arg cuda_version "$(lock_value cuda.version)" \
  --arg pytorch_version "$(lock_value pytorch.version)" \
  --arg pytorch_commit "$(lock_value pytorch.commit)" \
  --arg cutlass_dsl_version "$(lock_value cutlass-dsl.version)" \
  --arg cuda_arch_list "$(lock_value cuda.arch-list)" \
  --argjson submodules "${submodules_json}" \
  --argjson packages "${packages_json}" \
  '{
    schema: $schema,
    status: $status,
    scope: "FlashInfer Python sources and precompiled JIT modules for the declared ABI and GPU architecture",
    source: {repository: $repository, commit: $commit, tree: $tree, submodules: $submodules},
    package_version: $package_version,
    release_tag: $release_tag,
    runtime: {
      builder_image: $builder_image,
      python: $python_version,
      cuda: $cuda_version,
      pytorch: $pytorch_version,
      pytorch_commit: $pytorch_commit,
      cutlass_dsl: $cutlass_dsl_version,
      cuda_arch_list: $cuda_arch_list
    },
    packages: $packages,
    exclusions: ["flashinfer-cubin", "CUDA runtime", "PyTorch runtime"]
  }' > "${output_dir}/bundle/manifest.json"

cp "${lock_path}" "${tool_dir}/install.sh" "${output_dir}/bundle/"
chmod 0755 "${output_dir}/bundle/install.sh"
(
  cd "${output_dir}/bundle"
  find wheels -maxdepth 1 -name '*.whl' -print0 | sort -z | xargs -0 sha256sum
  sha256sum manifest.json requirements-github.txt runtime.lock install.sh
) > "${output_dir}/bundle/SHA256SUMS"

archive="${output_dir}/flashinfer-cu133-sm120-${source_commit}.tar.zst"
tar --sort=name \
  --mtime="@${source_date_epoch}" \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --zstd \
  -C "${output_dir}/bundle" \
  -cf "${archive}" .
(
  cd "${output_dir}"
  sha256sum "$(basename "${archive}")"
) > "${archive}.sha256"

cat > "${output_dir}/release-notes.md" <<EOF
Status: **research-only**

These wheels contain FlashInfer ${package_version} for Python 3.12, CUDA 13.3,
PyTorch 2.13.0, and SM120. The release does not contain CUDA or PyTorch.

- Source commit: \`${source_commit}\`
- Source tree: \`${source_tree}\`
- Precompiled architecture: \`12.0f\`

Install both release assets in a compatible runtime with:

\`uv pip install --no-deps --require-hashes -r requirements-github.txt\`
EOF

printf '%s\n' "${output_dir}/bundle"
