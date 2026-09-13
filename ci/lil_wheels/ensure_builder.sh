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
rootless=0
if docker info --format '{{json .SecurityOptions}}' | grep -q 'name=rootless'; then
  rootless=1
fi

if ! docker buildx inspect "${builder}" >/dev/null 2>&1; then
  if (( rootless )); then
    docker buildx create \
      --name "${builder}" \
      --driver docker-container \
      --bootstrap >/dev/null
  else
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
fi

docker buildx inspect --bootstrap "${builder}" >/dev/null

if (( rootless )); then
  service=lil-flashinfer-rootless-docker.service
  control_group=$(systemctl show --value --property ControlGroup "${service}")
  cgroup_path="/sys/fs/cgroup${control_group}"
  daemon_pid=$(systemctl show --value --property MainPID "${service}")
  actual_memory_high=$(<"${cgroup_path}/memory.high")
  actual_memory=$(<"${cgroup_path}/memory.max")
  actual_swap=$(<"${cgroup_path}/memory.swap.max")
  read -r actual_quota actual_period < "${cgroup_path}/cpu.max"
  actual_cpuset=$(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/${daemon_pid}/status")
  test "${actual_memory_high}" = "$(lock_value buildx.memory-high-bytes)"
  test "${actual_memory}" = "${memory}"
  test "${actual_swap}" = "$(lock_value buildx.swap-max-bytes)"
  test "${actual_cpuset}" = "${cpuset}"
  test "${actual_quota}" = "${quota}"
  test "${actual_period}" = "${period}"
  printf 'builder=%s rootless_service=%s memory_high=%s memory_max=%s swap_max=%s cpuset=%s cpu_quota=%s cpu_period=%s\n' \
    "${builder}" "${service}" "${actual_memory_high}" "${actual_memory}" \
    "${actual_swap}" "${actual_cpuset}" "${actual_quota}" "${actual_period}"
  exit 0
fi

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
