# PCIe collective preparation

`AllReduce.plan` declares the eligible transports for an established channel.
`PreparationSession` races them jointly across the communicator and retains the
selected native execution state before graph capture. Each callsite submits its
actual shape, dtype, layout, arithmetic options, and aliasing contract. Equivalent
declarations coalesce; request names and layer numbers do not create separate
tuning keys. Every caller still binds its own serving operands.

## Candidate operations

| Declaration | Candidates |
| --- | --- |
| Staged `AllReduce.all_reduce`, direct peer groups | Native pull; LL and Simple peer push; BF16 payload-ready full reduction and reduce-scatter/all-gather; provisioned BF16 two-shot |
| Staged fused residual addition and RMSNorm | Native staged transports compatible with declared input, residual, normalization, grouping, and output arithmetic |
| DCP `all_gather_heads` and `lse_reduce_scatter`, TP2/4/8 | Thread count and block limit; equivalent effective grids are measured once |

Payload-ready transports require their reserved incoming slots, BF16 support,
and sufficient SMs to keep their grid resident. Two-shot retains its native
capacity and alignment constraints.
Unsupported alternatives are absent, rather than failing over during execution.
Plain tuning admits FP16/BF16 tensors with positive 16-byte-packed payloads
within the prepared staging capacity. LL requires two capacity shards per remote
source; Simple additionally keeps payload bytes disjoint from its 36 CTA flag
records. Full reduction and reduce-scatter/all-gather use separate BF16 payload
shards. Widths, row counts, and divisibility by rank count do not restrict this
measured candidate catalog.
The peer-push transports require the prepared alternating graph slots. Thread counts
32, 64, 128, 256, and 512 are measured over grids up to 36 CTAs, including the
payload-covering grid. Pull remains a measured alternative. Optimized TP2 LL and
Simple implementations remain separate entrants alongside generic peer push.
Two-shot measures warp-aligned thread counts through 512 and block limits through
64, coalescing limits that produce the same effective grid.

Native peer-push kernels support rank counts 2, 3, 4, 6, 8, and 10 with fully
mapped peers. Single-channel TP10 auto preparation provisions the actual direct
pool and retains it on success. Only a coordinated peer-mapping rejection with
complete rollback selects hierarchy; unrelated setup or cleanup failures remain
fatal. Explicit hierarchical choices and multichannel TP10 declarations retain
hierarchical routing. Pairwise reachability alone does not establish simultaneous
full-peer mapping capacity. Inputs and outputs must be 16-byte aligned, use
compatible physical traversal, and be either exact aliases or disjoint.

Tuning races the native alternatives over their launch geometry; explicitly
disabled tuning retains the static dispatch policy. Explicit
`CollectiveConfig(implementation=..., threads=..., blocks=...)` pins a legal
configuration. Native low-level plans still use `PcieConfig`.

Hierarchical collectives, DMA rings, FP8 sequence-parallel transports, paired gathers,
and owner/top-k exchanges retain their native fixed declarations. Transport
codecs with different arithmetic are not interchangeable tuning candidates.

`PCIeTwoShotBF16.all_reduce` accepts a disjoint output or the exact input
storage span. Reduce-scatter consumes the input into a fixed private shard
before same-stream all-gather writes the output; peers consume IPC copies.
Partial overlaps remain rejected. Standalone reduce-scatter and all-gather
require disjoint outputs. All operations and graph replays sharing one runtime
must remain serialized.

## Benchmark and synchronization contract

The `_benchmark.py` fixtures allocate private operands, generate and reset inputs,
capture candidate calls, and check arithmetic before timing. They preserve
declared tensor layouts and residual aliasing. A caller can supply an explicit
`benchmark_call` for a different representative execution context.

All ranks compile and bind a candidate before any rank primes it. Each rank
queues the same captured calls behind a stream gate; the coordinator releases
the sample block after every rank reports readiness. An untimed collective
aligns GPU entry. Corresponding event samples reduce with **maximum across
ranks** before repeat sizing, elimination, and winner selection. The tournament
retains a champion and one challenger, bounding candidate operand residency to
two fixtures plus graph/event and cache-eviction storage.

`PreparationProgress.ready_collective_tuning` carries
`CollectiveTuningRequirement(key, ranks, step, payload)`. An embedding coordinator
exchanges these contributions in communicator rank order and resumes
`PreparationJob.advance(collective_tuning=contributions)`. All members must report
the same key, ranks, and step. Query signatures reject inconsistent shapes or
arithmetic before candidate entry. Cancellation is agreed at drained group
boundaries. Coordinators must propagate preparation exceptions to every member.

Collective candidates execute on the whole group; they are not divided among
the individual-kernel tuner's candidate shards. Independent TP replicas tune
their own communication group. The selection key includes the ordered GPU UUIDs,
rank-local query/program identities, and measurement policy. A valid cached
record from any member is adopted by the group. Cache-only mode fails if none
has a record. Interrupted collective races restart; completed selections reuse
the ordinary selection cache.

## Validation

Run the standalone group test with two or four visible peer-accessible SM120
GPUs:

```sh
B12X_RUN_COLLECTIVE_TUNING_TEST=1 python -m pytest -q \
  tests/experimental/b12x/comm/test_collective_tuning_gpu.py
```

The test exercises duplicate declarations, cache reuse with a missing member
cache, one-rank cancellation, inconsistent shapes, cache-only misses, and graph
replay. `benchmarks/experimental/b12x/benchmark_pcie_allreduce_transports.py`
compares explicit transports and the static dispatch policy with CUDA graphs.
