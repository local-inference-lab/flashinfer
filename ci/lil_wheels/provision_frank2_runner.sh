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
if [[ -z ${GITHUB_RUNNER_TOKEN:-} ]]; then
  printf 'GITHUB_RUNNER_TOKEN must contain a repository registration token.\n' >&2
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

install -d -o "${runner_user}" -g "${runner_user}" "${runner_dir}"
archive=$(mktemp --tmpdir actions-runner.XXXXXX.tar.gz)
trap 'rm -f "${archive}"' EXIT
curl --fail --location --retry 3 "$(lock_value runner.archive.url)" --output "${archive}"
printf '%s  %s\n' "$(lock_value runner.archive.sha256)" "${archive}" | sha256sum --check -

if [[ ! -e ${runner_dir}/config.sh ]]; then
  tar -xzf "${archive}" -C "${runner_dir}"
  chown -R "${runner_user}:${runner_user}" "${runner_dir}"
fi

if [[ ! -e ${runner_dir}/.runner ]]; then
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

install -m 0644 "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-rootless-docker.service" \
  /etc/systemd/system/lil-flashinfer-rootless-docker.service
install -m 0644 "${repo_root}/ci/lil_wheels/systemd/lil-flashinfer-actions-runner.service" \
  /etc/systemd/system/lil-flashinfer-actions-runner.service
systemctl daemon-reload
systemctl enable --now lil-flashinfer-rootless-docker.service

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

systemctl enable --now lil-flashinfer-actions-runner.service
systemctl --no-pager --full status lil-flashinfer-rootless-docker.service
systemctl --no-pager --full status lil-flashinfer-actions-runner.service
