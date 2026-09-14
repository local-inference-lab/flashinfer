# FlashInfer CUDA 13.4 and SM120 wheel releases

Status: **research-only**

The `LIL FlashInfer CUDA 13.4 SM120 wheels` GitHub Actions workflow produces
source-addressed `flashinfer-python` and `flashinfer-jit-cache` wheels for the
CUDA 13.4, NVIDIA PyTorch 26.08, Python 3.12, and SM120 runtime declared in
`ci/lil_wheels/runtime.lock`. The wheels are application packages; they do not
contain CUDA, PyTorch, or an NVIDIA driver.

The source branch `community/jovian-judgement-cu134-sm120` contains the
CUDA 13.4 wheel publisher and excludes B12X from the FlashInfer package.
Each push produces a source-addressed GitHub prerelease whose tag contains
the full source commit. Repeated workflow runs verify existing releases;
published release assets are not overwritten.

## Release channels

Beta releases are built once by the resource-bounded self-hosted runner. A beta
release is identified by:

```text
flashinfer-cu134-sm120-beta-<full-source-commit>
```

Stable promotion does not compile source. After correctness and performance
qualification, a maintainer creates a tag at the qualified source commit:

```bash
git tag flashinfer-cu134-sm120-stable-v0.6.18-g<short-source-commit> <full-source-commit>
git push origin flashinfer-cu134-sm120-stable-v0.6.18-g<short-source-commit>
```

The promotion job downloads the corresponding beta release, verifies its
checksums, and attaches the same wheel bytes to a non-prerelease GitHub release.
The stable release also contains an attestation that identifies the beta
manifest and its SHA-256 digest.

## Installation

Every release exposes individual wheel files as direct GitHub downloads and a
hash-pinned `requirements-github.txt`. Download the requirements file and run:

```bash
uv pip install --no-deps --require-hashes -r requirements-github.txt
```

The `--no-deps` requirement is intentional. Install the wheels only in a
runtime providing the exact CUDA, PyTorch, Python, and CUTLASS DSL versions in
`manifest.json`. The included `install.sh` supports an offline bundle containing
the two wheels and verifies `SHA256SUMS` before installation.

GitHub Releases are an artifact host, not a PEP 503 package index. Direct URL
requirements work with `uv` without an index. A static package index can be
added later if installing only by package name is required.

## frank2 resource isolation

Repository-scoped runners for `flashinfer`, `vllm`, `b12x`, `LMCache`,
`InstantTensor`, `nccl-canonical`, and `blackwell-llm-docker` use the
`lil-wheel-builder` label. Their native-wheel jobs share one BuildKit worker,
foundation layers, and build caches. The file lock
`/var/lib/github-flashinfer/locks/cu134-sm120-build.lock` serializes builds
across repositories. These runners must not execute untrusted pull-request
jobs. An organization-scoped runner group is also supported by the
provisioning scripts; it is not required for this deployment.

The runner opens outbound TLS connections to GitHub on TCP port 443; GitHub
does not connect to frank2 and no inbound firewall rule is required.
`ci/lil_wheels/provision_frank2_runner.sh` installs the pinned GitHub Actions
runner release and verifies its published SHA-256 digest.
Automatic runner updates are disabled so the executable remains pinned. Update
`runner.version`, its URL, and its digest before GitHub's 30-day disabled-update
grace period expires.

Compilation runs inside a dedicated Docker BuildKit worker named
`lil-wheel-cu134-sm120`. The rootless Docker daemon and its containers
inherit the build limits from a dedicated systemd user slice. The
`ci/lil_wheels/ensure_builder.sh` command verifies the parent cgroup and the
BuildKit worker's membership in the bounded user slice before building:

- CPU affinity: logical CPUs 64 through 127;
- CPU quota: 64 logical CPUs;
- memory: 256 GiB;
- swap: disabled;
- tasks: 8,192 across the rootless daemon and its build children.

The AOT compiler starts at most 48 concurrent CUDA compilation jobs. Each `nvcc`
process uses one frontend thread because device compilation is predominantly
single-threaded for this wheel. The BuildKit CPU quota remains the absolute
64-CPU ceiling. If the build exceeds 256 GiB, the kernel selects processes from
the bounded build slice instead of reclaiming unbounded host memory. Swap is
disabled for the slice independently of frank2's host configuration.

The BuildKit worker owns persistent pip and AOT object caches keyed by the CUDA,
Python, and architecture identity. A FlashInfer source change reuses unchanged
objects. A CUDA, PyTorch, CUTLASS DSL, Python ABI, or target-architecture change
requires new cache identifiers in the Dockerfile.

The runner and BuildKit worker use a dedicated rootless Docker daemon whose
storage, socket, container namespace, and build cache are separate from the
rootful Docker daemon used for model serving. The dedicated user slice has a
256 GiB memory ceiling. The runner process has separate ceilings of 4 GiB and
4,096 tasks. Do not enable workflows from untrusted pull requests on the
self-hosted label even with this isolation.

For an organization-scoped deployment, create the selected-repository runner group with an authenticated
organization administrator whose GitHub token has runner-group permission:

```bash
ci/lil_wheels/configure_org_runner_group.sh
```

Stream a short-lived organization registration token over SSH standard input.
The token is not placed in a local or remote process argument and is not stored
in Git:

```bash
gh api --method POST \
  orgs/local-inference-lab/actions/runners/registration-token \
  --jq .token |
  ssh root@192.168.66.14 \
    'IFS= read -r GITHUB_RUNNER_TOKEN; export GITHUB_RUNNER_TOKEN;
     exec /path/to/flashinfer/ci/lil_wheels/provision_frank2_runner.sh'
```

Migrating an existing repository-scoped registration also requires its
short-lived removal token:

```bash
{
  gh api --method POST \
    repos/local-inference-lab/flashinfer/actions/runners/remove-token \
    --jq .token
  gh api --method POST \
    orgs/local-inference-lab/actions/runners/registration-token \
    --jq .token
} | ssh root@192.168.66.14 \
  'IFS= read -r GITHUB_RUNNER_REMOVE_TOKEN;
   IFS= read -r GITHUB_RUNNER_TOKEN;
   export GITHUB_RUNNER_REMOVE_TOKEN GITHUB_RUNNER_TOKEN;
   exec /path/to/flashinfer/ci/lil_wheels/provision_frank2_runner.sh'
```

The registration token expires after one hour. The runner receives its own
organization-scoped credentials during registration; neither short-lived token
is retained by the service definition.
## B12X package ownership

Status: **implemented**.

The `community/jovian-judgement-cu134-sm120` branch builds FlashInfer wheels
for composition with the independently released `local-inference-lab/b12x`
package. B12X is built from a resolved `master` commit. Its wheel owns the
`b12x` import namespace, GPU profiles, command-line tools, and vLLM plugins.

FlashInfer wheels exclude both `b12x/` and `flashinfer/b12x/`, and publish no
B12X entry points. This prevents installation order from replacing the B12X
package with FlashInfer's migration snapshot. The build inspects the wheel
archive and rejects any overlapping package payload or entry point.
The migration sources remain in the checkout for provenance; executing that
checkout through `PYTHONPATH` is not the community wheel deployment contract.
