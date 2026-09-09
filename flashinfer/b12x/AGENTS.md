# Agent guide: b12x inside FlashInfer

## What this directory is

`flashinfer.b12x` is the imported b12x implementation inside the
[`local-inference-lab/flashinfer`](https://github.com/local-inference-lab/flashinfer)
fork. It contains SM120/SM121-oriented inference kernels and their runtime,
planning, quantization, loading, and serving integration code: GEMM, MoE,
attention, normalization, recurrent/sequence operations, and collectives.

This directory is a **migration staging area**, not the intended permanent home
of every API. Moving code here did not integrate it into FlashInfer's existing
APIs, upstream it, or qualify every implementation for every FlashInfer GPU.
Some imported code and tools are experimental or historical; inspect their
contracts and evidence rather than inferring support from their presence.

The initial import is a snapshot of b12x revision
`75ffee6375b0577ce2c8d6931ffacefda3ecbdd6`, without b12x Git ancestry. The user
explicitly retained that tested cutoff when the original checkout received
concurrent loader/PLE changes. Do not silently resynchronize from that checkout.
New work for this migration belongs in this fork; the original b12x repository
is not an automatically synchronized second implementation.

The user subsequently authorized importing commit
`4774ca8e68186ff25b5a1adafc142c4af092bb40` (demand-paged mmap PLE table storage).
That follow-on import retains the namespace adaptations and is recorded in the
[source update ledger](../../validation/b12x/source_updates/4774ca8e6.json).
The initial snapshot and its serving evidence remain historical records; they
have not been rewritten to claim qualification of these newer changes.

## Current package and compatibility contract

- The repository-root `pyproject.toml` builds **one** `flashinfer-python`
  distribution. Use `flashinfer.b12x` in new in-repo imports.
- The top-level [`b12x/__init__.py`](../../b12x/__init__.py) is an automatic, lazy
  compatibility importer. `b12x` and `b12x.*` resolve to the corresponding
  `flashinfer.b12x` modules, not separate copies of their source.
- Both import spellings must share module objects, classes, registries, compiler
  caches, and freeze state. Preserve canonical module metadata and avoid eagerly
  importing every optional component. Merely sharing `__path__` can execute the
  same code twice and is not an adequate replacement.
- Keep these shims in the FlashInfer fork for now. Moving them into the original
  b12x project is a deferred user decision, not part of routine cleanup. Do not
  require a separately installed b12x distribution.
- Stable backend names, `B12X_*` controls, torch custom-op names, checkpoint
  formats, and policy IDs are contracts, not Python import paths. Do not rename
  them merely because the package moved.
- Preserve CLI commands, vLLM plugin entry points, embedded profiles, and native
  C-source package data when changing packaging. Verify a built wheel as well as
  an editable install; a checkout can conceal missing wheel contents.

## The plan

The reason for placing b12x in this fork is to use Git to move pieces into
FlashInfer incrementally, upstream them, and keep rebasing onto upstream as those
pieces land. This is not a mandate for a wholesale rewrite.

### 1. Maintain the imported baseline

Keep the existing b12x APIs usable while the next component is selected. Keep
snapshot/import changes separate from API integration changes. Do not mix a
compiler upgrade, new quantization semantics, or unrelated kernel tuning into a
namespace or packaging change.

### 2. Integrate one bounded component at a time

Before editing, identify the component's consumers, public contract, supporting
kernels/helpers, planner decisions, tests, and intended FlashInfer API. Inspect
current upstream conventions and any existing equivalent implementation first.
No fixed component ordering has been chosen; choose the next slice with the user
rather than inventing a GEMM/attention/MoE roadmap.

Adapt that slice into an upstream-appropriate FlashInfer API and implementation
location. Migrate affected callers, tests, documentation, and packaging together.
Preserve promised legacy behavior deliberately while eliminating obsolete code;
do not maintain a second independent kernel implementation just to keep two
namespaces working. An API wrapper around this entire staging package is not,
by itself, completed FlashInfer integration.

### 3. Upstream focused, independently usable changes

Prepare small commits and PRs against
[`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer), following
its current contribution and API conventions. A contribution must stand on its
upstream base without requiring the wholesale b12x import or unrelated fork-only
code. Include the necessary helpers and qualification with the component.

Do not submit this whole directory as one upstream PR or merge the snapshot
branch into an upstream contribution merely to obtain its dependencies. Keep
provenance and license attribution even though the original Git ancestry was
not imported. Publishing PRs or pushing branches still requires task authority.

### 4. Rebase and remove the upstreamed duplicate

Once upstream contains an accepted component, update the fork from upstream and
rebase the remaining fork-only work. Verify that the updated base supplies the
required behavior before removing the local duplicate. Migrate any remaining
callers and compatibility surface deliberately; an upstream merge alone does
not prove downstream compatibility.

Repeat with the next component so this staging area shrinks instead of becoming
a permanent parallel FlashInfer. Keep topic changes separable from the initial
snapshot, and do not rewrite shared branches or discard user changes without
explicit authorization.

## Engineering invariants during the transition

- **Core compute stays CuTe DSL.** Triton is for supporting metadata/packing
  kernels, not new core GEMM, attention, or MoE kernels or their prototypes.
- **Preserve serving behavior.** Warmup, CUDA graph capture/replay, stable device
  addresses, and fixed or preplanned scratch capacity are requirements. Planned
  ops normally follow `Caps -> plan -> bind -> run`; binding maps caller-owned
  storage into views and must not allocate a new workspace. Inspect the actual
  component and integration contract before changing lifetime or ownership.
- **Keep live quantities out of compile/cache keys.** Token, row, batch,
  sequence, expert, page, block, and occupancy counts are runtime inputs. Static
  model geometry, planned capacity, device, and toolchain identity may select
  specializations. Warm those specializations before capture and exercise
  multiple live counts under frozen kernel resolution.
- **Preserve planner ownership.** Integrations provide metadata and capacities;
  they must not duplicate b12x planner policy. Resolve policy at planning time,
  not bind/replay. New or migrated planned ops must keep the typed query/config,
  [component catalog](policy/catalog.py), generators, schemas, and embedded
  profiles consistent. Invalid matching profiles fail closed.
- **Preserve numerical contracts.** W4A16 uses BF16 activations and inline FP4
  weight dequantization, without activation-scale math. Compressed MLA and GLM
  MLA/NSA are different contracts; verify layouts, dimensions, roles, and TP
  axes rather than merging assumptions.
- **Use Int64 for pool-scaled offsets.** Widen page/block/row IDs before stride
  multiplication. Paged-kernel repros must include high IDs beyond the Int32
  offset boundary; small sequential page IDs do not cover this failure mode.
- **Treat compiler changes as their own project.** Read the current root
  dependency pins and CI policy. Do not independently upgrade CuTe/compiler
  libraries to satisfy a new component. Isolate and qualify a compiler migration
  without changing the known-good baseline underneath serving work.
- **Prove the real path.** Run the affected GPU oracle and serving path before
  performance claims. Verify current hardware before selecting architecture
  settings. Preserve raw commands, source/artifact identity, correctness state,
  GPU mode, timings, and ratio direction when benchmarking. Reference or fallback
  execution is not proof of native/fused integration.

## Where to look and how to qualify a change

Paths below are relative to this guide; commands run from the repository root.

- [Package overview](../../docs/b12x/README.md): operation families and usage.
- [GPU policy contracts](../../docs/b12x/gpu-profiles.md): resolution,
  qualification modes, profiles, and generators.
- [MoE execution model](../../docs/b12x/moe-execution-model.md): planner and
  execution ownership for MoE work.
- [Imported engineering guidance](../../docs/b12x/source/AGENTS.md): detailed
  original kernel, policy, addressing, and compiler-migration requirements.
  Its old paths must be interpreted through the new layout.
- [Tests](../../tests/b12x/), [benchmarks](../../benchmarks/b12x/), and
  [scripts](../../scripts/b12x/) retain the imported support code.
- [Snapshot manifest](../../validation/b12x/snapshot_manifest.json) records the
  original-to-destination mapping and frozen hashes.
- [Migration verification](../../validation/b12x/migration_verification.json)
  records the initial install checks and four-model vLLM serving qualification,
  including exact commands, overrides, and known limitations.

For import/packaging changes, exercise both import orders, nested module identity,
shared runtime state, CLI/plugin discovery, and wheel data. The focused starting
check is:

```bash
python -m pytest -q tests/test_b12x_compat.py tests/b12x/test_registry.py
```

For a component integration, also run its behavioral tests, relevant existing
FlashInfer tests, GPU correctness checks, and affected real serving scripts.
The import was qualified with GLM 5.3 Flash, GLM 5.3, Qwen 3.8 Flash Next, and
DeepSeek V4 Flash. That evidence is a baseline, not blanket acceptance of later
changes. External serving checkouts are consumers, not implicit edit targets or
source-code fallbacks.

The imported catalog test has a recorded baseline defect, some historical tools
require absent sources, and the initial vLLM environment has a recorded
FlashInfer version-pin mismatch. Read the verification record before diagnosing
failures. These are not exemptions for new defects: distinguish baseline issues
from regressions without blanket skips, fake fallbacks, or weakened tests.

`python validation/b12x/verify_snapshot.py` checks the original frozen import.
To verify the explicitly authorized follow-on import, run:

```bash
python validation/b12x/verify_snapshot.py \
  --update-manifest validation/b12x/source_updates/4774ca8e6.json
```

The update ledger binds both predecessor and replacement hashes; the original
snapshot hashes remain intact. Pass update manifests in source-revision order
when qualifying subsequent imports. Do not rewrite old hashes merely to make a
check pass or claim that old serving evidence qualifies changed kernels; record
the new change and its own verification explicitly. The archived
`docs/b12x/source/pyproject.toml` is historical provenance, not another package
to build.
