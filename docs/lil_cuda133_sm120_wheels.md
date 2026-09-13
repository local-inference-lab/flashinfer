# FlashInfer CUDA 13.3 and SM120 wheel releases

Status: **research-only**

The `LIL FlashInfer CUDA 13.3 SM120 wheels` GitHub Actions workflow produces
source-addressed `flashinfer-python` and `flashinfer-jit-cache` wheels for the
CUDA 13.3, PyTorch 2.13.0, Python 3.12, and SM120 runtime declared in
`ci/lil_wheels/runtime.lock`. The wheels are application packages; they do not
contain CUDA, PyTorch, or an NVIDIA driver.

The source branch `community/jovian-judgement-cu133-sm120` preserves the
FlashInfer implementation used by the Jovian Judgement CUDA 13.3 community
containers. Each push to that branch produces one immutable GitHub prerelease.
Its tag contains the full source commit. A source commit is never rebuilt under
the same tag and release assets are never overwritten.

## Release channels

Beta releases are built once by the resource-bounded self-hosted runner. A beta
release is identified by:

```text
flashinfer-cu133-sm120-beta-<full-source-commit>
```

Stable promotion does not compile source. After correctness and performance
qualification, a maintainer creates a tag at the qualified source commit:

```bash
git tag flashinfer-cu133-sm120-stable-v0.6.18-g<short-source-commit> <full-source-commit>
git push origin flashinfer-cu133-sm120-stable-v0.6.18-g<short-source-commit>
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

The repository-scoped runner uses the unique `lil-flashinfer-builder` label and
accepts no pull-request jobs. It opens outbound TLS connections to GitHub on TCP
port 443; GitHub does not connect to frank2 and no inbound firewall rule is
required. `ci/lil_wheels/provision_frank2_runner.sh` installs the pinned GitHub
Actions runner release and verifies its published SHA-256 digest.
Automatic runner updates are disabled so the executable remains pinned. Update
`runner.version`, its URL, and its digest before GitHub's 30-day disabled-update
grace period expires.

Compilation runs inside a dedicated Docker BuildKit worker named
`lil-flashinfer-cu133-sm120`. `ci/lil_wheels/ensure_builder.sh` creates the worker
with these hard container limits and refuses to build if an existing worker has
different limits:

- CPU affinity: logical CPUs 64 through 127;
- CPU quota: 64 logical CPUs;
- memory: 256 GiB;
- memory plus swap: 256 GiB.

The AOT compiler starts at most 16 concurrent jobs with four CUDA compiler
frontend threads per job. This uses no more than 64 compiler threads and budgets
128 GiB for compiler jobs, leaving 128 GiB inside the BuildKit limit for Python,
linkers, and filesystem cache. If the build exceeds 256 GiB, the kernel kills
the BuildKit container rather than reclaiming unbounded host memory. frank2 has
no swap, so the memory-plus-swap limit also prevents hidden swap pressure.

The BuildKit worker owns persistent pip and AOT object caches keyed by the CUDA,
Python, and architecture identity. A FlashInfer source change reuses unchanged
objects. A CUDA, PyTorch, CUTLASS DSL, Python ABI, or target-architecture change
requires new cache identifiers in the Dockerfile.

The runner and BuildKit worker use a dedicated rootless Docker daemon whose
storage, socket, container namespace, and build cache are separate from the
rootful Docker daemon used for model serving. The Docker daemon's systemd unit
has a 300 GiB memory ceiling in addition to the BuildKit worker's 256 GiB
ceiling. The runner process has a 4 GiB memory ceiling. Do not enable workflows
from untrusted pull requests on the self-hosted label even with this isolation.

Generate a short-lived repository registration token on an authenticated admin
machine and pass it to the provisioning command without storing it in Git:

```bash
GITHUB_RUNNER_TOKEN=$(gh api --method POST \
  repos/local-inference-lab/flashinfer/actions/runners/registration-token \
  --jq .token)
ssh root@192.168.66.14 env GITHUB_RUNNER_TOKEN="${GITHUB_RUNNER_TOKEN}" \
  /path/to/flashinfer/ci/lil_wheels/provision_frank2_runner.sh
```

The registration token expires after one hour. The runner receives its own
repository-scoped credentials during registration; the token is not retained
by the service definition.
