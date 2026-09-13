#!/usr/bin/env bash
# Restrict the shared native-wheel runner group to repositories that publish wheels.
set -euo pipefail

organization=local-inference-lab
group_name='LIL CUDA 13.3 SM120 wheel builders'
repositories=(
  flashinfer
  vllm
  b12x
  LMCache
  nccl-canonical
  blackwell-llm-docker
)

group_id=$(gh api "orgs/${organization}/actions/runner-groups" \
  --jq ".runner_groups[] | select(.name == \"${group_name}\") | .id" | head -n1)
if [[ -z ${group_id} ]]; then
  group_id=$(jq -n \
    --arg name "${group_name}" \
    '{name: $name, visibility: "selected", allows_public_repositories: true}' |
    gh api --method POST "orgs/${organization}/actions/runner-groups" \
      --input - --jq .id)
fi

repository_ids='[]'
for repository in "${repositories[@]}"; do
  repository_id=$(gh api "repos/${organization}/${repository}" --jq .id)
  repository_ids=$(jq --argjson id "${repository_id}" '. + [$id]' \
    <<<"${repository_ids}")
done
jq -n --argjson ids "${repository_ids}" '{selected_repository_ids: $ids}' |
  gh api --method PUT \
    "orgs/${organization}/actions/runner-groups/${group_id}/repositories" \
    --input - >/dev/null

gh api --method PATCH \
  "orgs/${organization}/actions/runner-groups/${group_id}" \
  -f visibility=selected \
  -F allows_public_repositories=true >/dev/null

printf 'runner_group=%s id=%s repositories=%s\n' \
  "${group_name}" "${group_id}" "${repositories[*]}"
