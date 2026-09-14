#!/usr/bin/env bash
# Restrict the shared native-wheel runner group to repositories that publish wheels.
set -euo pipefail

organization=local-inference-lab
group_name='LIL CUDA 13.4 SM120 wheel builders'
repositories=(
  flashinfer
  vllm
  b12x
  LMCache
  InstantTensor
  nccl-canonical
  blackwell-llm-docker
)
workflows=(
  local-inference-lab/flashinfer/.github/workflows/lil-cu134-sm120-wheel-release.yml@community/jovian-judgement-cu134-sm120
  local-inference-lab/vllm/.github/workflows/jovian-judgement-wheel-release.yml@dev/jovian-judgement
  local-inference-lab/b12x/.github/workflows/lil-cu134-sm120-wheel-release.yml@master
  local-inference-lab/LMCache/.github/workflows/lil-cu134-sm120-wheel-release.yml@integration/local-inference-lab
  local-inference-lab/InstantTensor/.github/workflows/lil-cu134-wheel-release.yml@main
  local-inference-lab/nccl-canonical/.github/workflows/lil-cu134-sm120-release.yml@canonical/cu134-nccl2312-amd-turin
  local-inference-lab/blackwell-llm-docker/.github/workflows/jovian-wheel-runtime-release.yml@main
  local-inference-lab/blackwell-llm-docker/.github/workflows/community-container-release.yml@main
)
workflow_json=$(printf '%s\n' "${workflows[@]}" | jq -Rsc 'split("\n")[:-1]')

group_id=$(gh api "orgs/${organization}/actions/runner-groups" \
  --jq ".runner_groups[] | select(.name == \"${group_name}\") | .id" | head -n1)
if [[ -z ${group_id} ]]; then
  group_id=$(jq -n \
    --arg name "${group_name}" \
    --argjson workflows "${workflow_json}" \
    '{name: $name, visibility: "selected", allows_public_repositories: true,
      restricted_to_workflows: true, selected_workflows: $workflows}' |
    gh api --method POST "orgs/${organization}/actions/runner-groups" \
      --input - --jq .id)
fi

repository_ids='[]'
for repository in "${repositories[@]}"; do
  repository_id=$(gh api "repos/${organization}/${repository}" --jq .id)
  repository_ids=$(jq --argjson id "${repository_id}" '. + [$id]' \
    <<<"${repository_ids}")
done
# Restrict eligible workflows before granting access to additional repositories.
jq -n \
  --arg name "${group_name}" \
  --argjson workflows "${workflow_json}" \
  '{name: $name, visibility: "selected", allows_public_repositories: true,
    restricted_to_workflows: true, selected_workflows: $workflows}' |
  gh api --method PATCH \
    "orgs/${organization}/actions/runner-groups/${group_id}" \
    --input - >/dev/null

jq -n --argjson ids "${repository_ids}" '{selected_repository_ids: $ids}' |
  gh api --method PUT \
    "orgs/${organization}/actions/runner-groups/${group_id}/repositories" \
    --input - >/dev/null

printf 'runner_group=%s id=%s repositories=%s workflows=%s\n' \
  "${group_name}" "${group_id}" "${repositories[*]}" "${workflows[*]}"
