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
  user_slice="user-$(id -u).slice"
  user_control_group=$(systemctl show --value --property ControlGroup "${user_slice}")
  user_cgroup_path="/sys/fs/cgroup${user_control_group}"
  user_runtime="/run/user/$(id -u)"
  daemon_pid=$(XDG_RUNTIME_DIR="${user_runtime}" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=${user_runtime}/bus" \
    systemctl --user show --value --property MainPID "${service}")
  actual_cpuset=$(awk '/^Cpus_allowed_list:/ {print $2}' "/proc/${daemon_pid}/status")
  test "${actual_cpuset}" = "${cpuset}"
  actual_memory_high=$(<"${user_cgroup_path}/memory.high")
  actual_memory=$(<"${user_cgroup_path}/memory.max")
  actual_swap=$(<"${user_cgroup_path}/memory.swap.max")
  test "${actual_memory_high}" = "$(lock_value buildx.memory-high-bytes)"
  test "${actual_memory}" = "${memory}"
  test "${actual_swap}" = "$(lock_value buildx.swap-max-bytes)"
  read -r user_quota user_period < "${user_cgroup_path}/cpu.max"
  test "${user_quota}" = "${quota}"
  test "${user_period}" = "${period}"
  test "$(<"${user_cgroup_path}/pids.max")" = "$(lock_value buildx.tasks-max)"
  test "$(<"${user_cgroup_path}/cpuset.cpus.effective")" = "${cpuset}"
  container_pid=$(docker inspect --format '{{.State.Pid}}' "${container}")
  container_cgroup=$(awk -F: '$1 == "0" {print $3}' "/proc/${container_pid}/cgroup")
  case "${container_cgroup}" in
    "${user_control_group}"/*) ;;
    *)
      printf 'BuildKit cgroup %s is outside bounded user slice %s.\n' \
        "${container_cgroup}" "${user_control_group}" >&2
      exit 1
      ;;
  esac
  printf 'builder=%s rootless_service=%s build_slice=%s memory_high=%s memory_max=%s swap_max=%s cpuset=%s cpu_quota=%s cpu_period=%s\n' \
    "${builder}" "${service}" "${user_slice}" "${actual_memory_high}" \
    "${actual_memory}" "${actual_swap}" "${actual_cpuset}" "${user_quota}" \
    "${user_period}"
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
