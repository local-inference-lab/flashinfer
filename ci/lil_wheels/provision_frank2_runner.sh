#!/usr/bin/env bash
# Provision the repository-scoped rootless GitHub Actions runner on frank2.
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  printf 'Run this provisioning command as root.\n' >&2
  exit 1
fi
if [[ $(hostname -s) != frank2 ]]; then
  printf 'This resource profile is qualified only for host frank2.\n' >&2
  exit 1
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
lock_path="${repo_root}/ci/lil_wheels/runtime.lock"
runner_user=github-flashinfer
runner_home=/var/lib/github-flashinfer
runner_dir=/opt/actions-runner-flashinfer
runner_name=frank2-flashinfer-cu133-sm120
repository_url=https://github.com/local-inference-lab/flashinfer

lock_value() {
  local key=$1
  awk -F= -v key="${key}" '$1 == key {sub(/^[^=]*=/, ""); print; found=1} END {exit !found}' \
    "${lock_path}"
}

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install --yes --no-install-recommends \
  ca-certificates \
  curl \
  fuse-overlayfs \
  git \
  jq \
  libicu74 \
  libssl3t64 \
  slirp4netns \
  uidmap \
  unzip \
  zstd

if ! id "${runner_user}" >/dev/null 2>&1; then
  useradd --system --create-home --home-dir "${runner_home}" \
    --shell /usr/sbin/nologin "${runner_user}"
fi
if ! grep -q "^${runner_user}:" /etc/subuid; then
  usermod --add-subuids 200000-265535 "${runner_user}"
fi
if ! grep -q "^${runner_user}:" /etc/subgid; then
  usermod --add-subgids 200000-265535 "${runner_user}"
fi
runner_uid=$(id -u "${runner_user}")
user_slice="user-${runner_uid}.slice"
user_slice_dropin="/etc/systemd/system/${user_slice}.d"
install -d -m 0755 "${user_slice_dropin}"
install -m 0644 \
  "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-build-user-slice.conf" \
  "${user_slice_dropin}/lil-flashinfer-build.conf"
systemctl daemon-reload
loginctl enable-linger "${runner_user}"
systemctl start "user@${runner_uid}.service"
cpu_percent=$((
  $(lock_value buildx.cpu-quota) * 100 / $(lock_value buildx.cpu-period)
))
systemctl set-property --runtime "${user_slice}" \
  "AllowedCPUs=$(lock_value buildx.cpuset)" \
  "CPUQuota=${cpu_percent}%" \
  "MemoryHigh=$(lock_value buildx.memory-high-bytes)" \
  "MemoryMax=$(lock_value buildx.memory-bytes)" \
  "MemorySwapMax=$(lock_value buildx.swap-max-bytes)" \
  "TasksMax=$(lock_value buildx.tasks-max)"
user_runtime="/run/user/${runner_uid}"
user_systemctl=(
  runuser -u "${runner_user}" -- env
  "HOME=${runner_home}"
  "XDG_RUNTIME_DIR=${user_runtime}"
  "DBUS_SESSION_BUS_ADDRESS=unix:path=${user_runtime}/bus"
  systemctl --user
)

# A user service gives rootless runc access to the delegated user manager.
# The bounded user slice remains the resource-control parent of every container
# scope even though Docker creates those scopes as siblings of the daemon.
systemctl disable --now lil-flashinfer-rootless-docker.service 2>/dev/null || true
rm -f /etc/systemd/system/lil-flashinfer-rootless-docker.service
install -m 0644 \
  "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-rootless-docker.tmpfiles" \
  /etc/tmpfiles.d/lil-flashinfer-rootless-docker.conf
systemd-tmpfiles --create /etc/tmpfiles.d/lil-flashinfer-rootless-docker.conf
user_unit_dir="${runner_home}/.config/systemd/user"
install -d -o "${runner_user}" -g "${runner_user}" "${user_unit_dir}"
install -o "${runner_user}" -g "${runner_user}" -m 0644 \
  "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-rootless-docker.service" \
  "${user_unit_dir}/lil-flashinfer-rootless-docker.service"
"${user_systemctl[@]}" daemon-reload
"${user_systemctl[@]}" enable lil-flashinfer-rootless-docker.service
"${user_systemctl[@]}" restart lil-flashinfer-rootless-docker.service

install -d -o "${runner_user}" -g "${runner_user}" "${runner_dir}"
if [[ ! -e ${runner_dir}/config.sh ]]; then
  archive=$(mktemp --tmpdir actions-runner.XXXXXX.tar.gz)
  trap 'rm -f "${archive}"' EXIT
  curl --fail --location --retry 3 "$(lock_value runner.archive.url)" --output "${archive}"
  printf '%s  %s\n' "$(lock_value runner.archive.sha256)" "${archive}" | sha256sum --check -
  tar -xzf "${archive}" -C "${runner_dir}"
  chown -R "${runner_user}:${runner_user}" "${runner_dir}"
fi
test "$(runuser -u "${runner_user}" -- "${runner_dir}/bin/Runner.Listener" --version)" = \
  "$(lock_value runner.version)"

if [[ ! -e ${runner_dir}/.runner ]]; then
  if [[ -z ${GITHUB_RUNNER_TOKEN:-} ]]; then
    printf 'GITHUB_RUNNER_TOKEN must contain a repository registration token.\n' >&2
    exit 1
  fi
  runuser -u "${runner_user}" -- env HOME="${runner_home}" \
    "${runner_dir}/config.sh" \
      --unattended \
      --url "${repository_url}" \
      --token "${GITHUB_RUNNER_TOKEN}" \
      --name "${runner_name}" \
      --labels lil-flashinfer-builder \
      --work "${runner_home}/work" \
      --disableupdate \
      --replace
fi

install -m 0644 "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-actions-runner.service" \
  /etc/systemd/system/lil-flashinfer-actions-runner.service
systemctl daemon-reload

for _ in $(seq 1 60); do
  if runuser -u "${runner_user}" -- env \
    DOCKER_HOST=unix:///run/lil-flashinfer-docker/docker.sock \
    docker info >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
runuser -u "${runner_user}" -- env \
  DOCKER_HOST=unix:///run/lil-flashinfer-docker/docker.sock \
  docker info >/dev/null

systemctl enable lil-flashinfer-actions-runner.service
systemctl restart lil-flashinfer-actions-runner.service
"${user_systemctl[@]}" --no-pager --full status \
  lil-flashinfer-rootless-docker.service
systemctl --no-pager --full status lil-flashinfer-actions-runner.service
