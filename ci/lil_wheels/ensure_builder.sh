#!/usr/bin/env bash
# Create or verify the resource-bounded BuildKit worker used on frank2.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lock_path="${repo_root}/ci/lil_wheels/runtime.lock"

lock_value() {
  local key=$1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

builder=$(lock_value buildx.builder)
memory=$(lock_value buildx.memory-bytes)
cpuset=$(lock_value buildx.cpuset)
quota=$(lock_value buildx.cpu-quota)
period=$(lock_value buildx.cpu-period)
container="buildx_buildkit_${builder}0"

if ! docker buildx inspect "${builder}" >/dev/null 2>&1; then
  docker buildx create \
    --name "${builder}" \
    --driver docker-container \
    --driver-opt "memory=${memory}" \
    --driver-opt "memory-swap=${memory}" \
    --driver-opt "cpuset-cpus=${cpuset}" \
    --driver-opt "cpu-quota=${quota}" \
    --driver-opt "cpu-period=${period}" \
    --bootstrap >/dev/null
fi

docker buildx inspect --bootstrap "${builder}" >/dev/null
actual_memory=$(docker inspect --format '{{.HostConfig.Memory}}' "${container}")
actual_memory_swap=$(docker inspect --format '{{.HostConfig.MemorySwap}}' "${container}")
actual_cpuset=$(docker inspect --format '{{.HostConfig.CpusetCpus}}' "${container}")
actual_quota=$(docker inspect --format '{{.HostConfig.CpuQuota}}' "${container}")
actual_period=$(docker inspect --format '{{.HostConfig.CpuPeriod}}' "${container}")

test "${actual_memory}" = "${memory}"
test "${actual_memory_swap}" = "${memory}"
test "${actual_cpuset}" = "${cpuset}"
test "${actual_quota}" = "${quota}"
test "${actual_period}" = "${period}"

printf 'builder=%s memory=%s cpuset=%s cpu_quota=%s cpu_period=%s\n' \
  "${builder}" "${actual_memory}" "${actual_cpuset}" \
  "${actual_quota}" "${actual_period}"
