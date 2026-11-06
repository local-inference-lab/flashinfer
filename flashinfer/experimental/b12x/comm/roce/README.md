# RoCEnante: experimental neighbor rings

The switchless DGX Spark topology is inspired by
[FujitsuPolycom/SparkRing](https://github.com/FujitsuPolycom/sparkring), which
describes directly cabled GB10 clusters and collective forwarding between
non-neighbor hosts. Credit to that project for the topology and design
inspiration. This RoCEnante experiment uses native CPU proxies and RDMA
mailboxes for relaying; it does not implement SparkRing's ConnectX hardware
forwarding or establish equivalent performance.

Both the API and this transport implementation are experimental. The explicit
`ring=True` constructor argument enables it; it is never selected automatically.
Omitting the argument keeps the direct all-peer transport.

## Configuration

Construct `b12x.comm.roce.AllReduce` as usual, with the additional keyword:

```python
runtime = roce.AllReduce(
    exchange_group=control_group,
    device="cuda:0",
    max_size=1 << 20,
    pipe_bytes=8 << 20,
    ring=True,
)
```

The exchange group's rank order defines the cycle `0,1,...,N-1,0`, with
2 through 16 ranks. All ranks must select the same mode and capacities. Only
the two neighbors' queue pairs connect (one for a two-rank cycle). This also
works on a switched fabric, allowing neighbor-only connectivity to be emulated
without changing host routes, firewall rules, or cabling.

On a physical ring, enumerate the active RDMA functions on both cabled ports
using `hca_names` and set the appropriate `gid_index`. The existing route planner
pairs endpoints on the same subnet for each neighbor, with up to two rails per
link. It trusts device order for endpoints without IPv4 GIDs, so verify those
links explicitly. A TCP control network is still needed for setup. A Gloo group
can perform that exchange without requiring an NCCL data transport.

One-shot reductions, two-shot reductions, fixed gathers, and variable-length
gathers share the same native write path. Non-neighbor transfers use the
shorter direction around the ring; equal distances choose increasing ranks.
Relays split each rail into at most 64 KiB packets. Each original sender owns
eight mailbox slots per rail at every rank, so up to eight packets per sender
and rail are in flight, and it reuses a slot only after the end-to-end
acknowledgment of the packet that last used it. Acknowledgments propagate back
only after the downstream transfer has finished reading its source, preserving
staging lifetimes without circular credit dependencies between senders. The
sender's payload is read only by its first-hop write, so its lifetime follows
the same completion rules as a direct write.

The proxy owns about 1 MiB of extra registered host memory per rank in the
world (8.0 MiB at eight ranks, 16.1 MiB at sixteen), allocated before connection
exchange. No relay allocation or Python callback occurs during replay. Existing
captured buffers and kernel layouts are retained. Proxy ABI 10 extends the
connection blob and sizes the mailbox region; native consumers must use the
matching C source and call `roce_enable_ring` before exchanging blobs. The
returned `native_description()` records `ring`; private mailbox addresses are
exchanged fresh at startup rather than restored from a CUDA graph.

The two halves of the pipelined two-shot path are also separate operations.
`reduce_scatter` reduces a whole message and keeps this rank's shard of the
sum (rows `[r R / world, (r + 1) R / world)` of an `R`-row operand), and
`gather_shards` concatenates every rank's shard along dimension 0, both over
the two-shot buffers, chunk flags and grid, so they interleave with one-shot
and two-shot operations in one graph. A reduce-scatter followed by a shard
gather moves the bytes of one two-shot all-reduce and reproduces its bits,
which lets a caller run row-local work on its shard between the halves.
`should_reduce_scatter` and `should_gather_shards` admit operands whose rows
hold whole 16-byte packs, whose row count (or gathered row count) divides by
the world size, and whose whole message fits `pipe_bytes`. The control record
carries them as op kinds 3 and 4; proxy ABI 11 adds the two kinds and changes
no region layout.

Each pipelined kind reuses one receive row and one flag per source, phase and
chunk, and a kernel waits until a flag equals its op's sequence number. Two
transitions let a rank write a peer's rows before that peer finished reading
the previous op: a reduce-scatter or two-shot after a reduce-scatter (phase 0),
and a shard gather after a two-shot or shard gather (phase 1). For those, every
rank writes its start-of-op sequence into a credit word of each peer's control
record, and the proxy holds the hazardous phase until every peer's credit
reached the op (proxy ABI 14). Without the credits, a write overtaking a slow
peer's read ends in that peer's wait timeout or in silently mixed rows. Other
transitions are ordered by data dependencies and carry no credits. On the ring
transports, a credit to a rank that is not a neighbor travels as an empty
mailbox write.

Collective generations range from 1 through 2,147,483,647. The proxy fails
with `RoCE sequence space exhausted` before dispatching a larger generation;
recreate every rank's runtime to reset the sequence. The bound preserves the
meaning of zero-initialized chunk-ready words and the healthy error sentinel.
It applies to the direct and ring transports and does not change the proxy ABI.

## Rails, lanes and the all-gather relay

Each peer is reached over up to two rails, each a pair of RDMA devices on one
IPv4 subnet. The direct transport routes a peer over the narrowest subnet it
shares: on DGX Sparks whose first port joins a switch and whose second port is
cabled to one partner, the partner's rails use the cable's two functions and
every other peer's use the switch's. A peer's two rails use devices behind
different PCIe links when possible, because the two ConnectX-7 functions behind
one PCIe x4 link share its bandwidth.

The two ranks of a group on one cabled pair hold every subnet in common, so
subnet breadth cannot tell their cable from the switch. Each endpoint therefore
also records its netdev's MAC address and, when lldpd runs on the host, the port
MAC addresses that `lldpctl` reports for its neighbors. LLDP frames do not cross
a switch, so devices of two ranks that list each other share a cable, and the
planner prefers such links. Without lldpd on both hosts, a two-rank group on a
cabled pair uses the switch.

Each rail is `lanes` reliable queue pairs (constructor `lanes`, else
`B12X_ROCE_LANES`, else 4; at most 4, equal on every rank). Two-shot and
variable-length chunk `c` uses lane `c % lanes`; one-shot and ring relays use
lane 0. A stripe and its flag always share a lane, so the flag-after-payload
ordering holds per stripe. The connection blob carries every lane's queue-pair
number and the lane count (proxy ABI 12); the region layout is unchanged.

On four DGX Sparks with both a switch and partner cables, a CPU-driven 16 MiB
two-shot (every chunk marked ready at once, 128 KiB chunks, 60 operations) sent
192.5 Gb/s per rank with one lane and 206.8 Gb/s with four; over the switch
alone it sent 190.8 Gb/s with either. A single queue pair per peer and device
does not keep both PCIe links of a host busy once a cable adds wire capacity.

When the ranks form pairs joined by direct cables (every rail to one peer uses
subnets only the two of them hold), the two-shot all-gather is relayed
(constructor `relay`, else `B12X_ROCE_RELAY`, else on; equal on every rank). A
rank sends its reduced shard to its partner and to one rank of every other
pair, which forwards each chunk over its own cable to its partner as soon as
the chunk's flags arrived. The receiver sees the same bytes and flags as from
a direct write, so results are unchanged bit for bit. The ConnectX-7 splits
each PCIe link evenly between its cable and switch functions, so a flat
two-shot, which sends one third of a rank's bytes to its partner, leaves cable
capacity unused while the switch traffic finishes. With two pairs the relay
puts two of a rank's three all-gather shards on its cable, and the whole
two-shot is half cable and half switch. The relay applies to two-shot and
shard-gather operations, not to the variable-length all-gather;
`native_description()["relay_plan"]` is the plan a native runtime passes to
`roce_set_relay`. Proxy ABI 13 adds `roce_set_relay`.

## Runnable checks

The transport validator uses real libibverbs registrations and RDMA writes,
with CPU-produced doorbells and byte-exact reference checks. It needs Python,
a C compiler, libibverbs development files, and at least two active RDMA functions per
rank. It creates no CUDA context. Run one command on each host, changing `R`
to that host's rank and `N` to the total rank count:

```sh
python3 tests/experimental/b12x/comm/validate_roce_ring_native.py \
  --rank R --world-size N --rendezvous RANK_ZERO_HOST:29783 \
  --hca-names roceP2p1s0f0,rocep1s0f0 --gid-index 3 --iterations 120
```

The default source directory is this checkout. Use `--source PATH` to validate
another proxy source directory, `--skew-rank R` to delay one CPU driver between
operations, `--consume-delay-ms D` to make that rank read peers' rows late,
`--scatter-gather` to add reduce-scatter and shard-gather ops in every order
between pipelined kinds, `--mesh` to check the ordinary direct transport, `--lanes L` to
select the queue pairs per rail, or `--mesh --relay` to relay two-shot
all-gather chunks through cable partners, which requires ranks that pair up over
direct cables and all four functions in `--hca-names`. The validator
alternates one-shot, two-shot, and variable-length operations, including changing
payload sizes, uneven shards, and empty contributions. Each rank reports routes,
connected-peer count, forwarding count, and completed writes as JSON.

The GPU test is a separate torchrun target. From each host, with its rank `R`:

```sh
torchrun --nnodes=N --nproc-per-node=1 --node-rank=R \
  --master-addr=RANK_ZERO_HOST --master-port=29650 \
  -m pytest -x tests/experimental/b12x/comm/test_roce_ring_gpu.py
```

This uses a Gloo control group, captures alternating two-shot/one-shot reductions,
and checks changed BF16 inputs over repeated graph replay. It skips without a
2..16-rank torchrun launch. Use the existing explicit `hca_names` and `gid_index`
constructor arguments if the machine's automatic discovery is insufficient.

The protocol and generation-limit tests use the same launch, with these pytest
targets:

```sh
torchrun --nnodes=N --nproc-per-node=1 --node-rank=R \
  --master-addr=RANK_ZERO_HOST --master-port=29650 \
  -m pytest -x tests/experimental/b12x/comm/test_roce_protocol_gpu.py \
  tests/experimental/b12x/comm/test_roce_sequence_exhaustion_gpu.py
```

The protocol tests alternate 2, 8, 4, and 16 MiB reductions within one captured
graph, change inputs between replays, mix variable-length gathers with both
reduction paths, and delay individual proxies. Both direct and ring transports
must preserve every operation's payload. The generation-limit tests initialize
stopped proxies through a test-only C wrapper: a two-shot graph completes at
generation 2,147,483,647, then the next replay must report sequence exhaustion
without advancing the epoch or posting another transfer.

## Validation limits

Validation used four physical DGX Sparks through a switch, with two RoCE rails
per host and non-neighbor queue pairs absent in ring mode:

| Check | Configuration | Result |
| --- | --- | --- |
| GPU collective and failure checks | Two ranks, sequence-limited proxy | 110 passed and one capacity-related skip per rank |
| GPU collective and failure checks | Four ranks, proxy and CuTe source at revision `c46a0b5` | 105 collective cases plus four protocol cases passed, with one capacity-related skip per rank |
| CPU-driven RDMA ring | Four ranks, sequence-limited proxy | 120 mixed operations per rank, with driver skew; every byte matched |
| CPU-driven RDMA ring | Six logical ranks on three hosts, sequence-limited proxy | 90 mixed operations per rank, with driver skew; every byte matched |

Revision `c46a0b5` also passed CPU-driven ring checks with four ranks and 240
operations per rank, seven ranks with 90, eight ranks with 120, and sixteen
ranks with 30. Larger worlds used multiple processes on each of the four hosts.
All payloads matched; its four-rank mesh passed 60 operations per rank.

The GPU checks cover changed-input graph replay, BF16 reductions, changing
variable-length counts, temporary proxy pauses, stopped-proxy timeouts, invalid
counts, and fail-stop handling. The ABI 9 proxy and CuTe source at revision
`c46a0b5`, including the two-shot `pipe_done` wait, passed four-rank graph replay.
The sequence-limited proxy passed generation-boundary GPU checks with two ranks
in both direct and ring modes, plus four- and six-rank CPU transport checks.
Every rank in those native ring checks forwarded traffic and connected only its
two neighbors.

A separate four-rank GPU negative control removed only the two-shot
`pipe_done` wait. Its changing-shard ring graph produced corrupt output; the
matching graph passed with the wait present. The wait therefore remains part
of the send-buffer lifetime contract.

On eight DGX Sparks cabled directly in a ring (each ConnectX-7 port to one
neighbor, two PCIe functions per port), the native validator passed 1000 mixed
operations per rank with one rank delayed, and the GPU graph replay test above
passed with eight ranks. Every byte and BF16 sum matched, each rank connected
only its two neighbors, and every rank forwarded traffic.

On the same ring, a CPU-driven timing loop (each rank rings the doorbell and
waits for every peer's flags; median over 500 operations, slowest rank) measured:

| Operation | Payload | Time |
| --- | ---: | ---: |
| One-shot | 32 B | 27 us |
| One-shot | 64 KiB | 72 us |
| One-shot | 1 MiB | 1.3 ms |
| Two-shot | 16 MiB | 5.0 ms |

Full-model ring serving and ring performance under a model remain unqualified.
These finite tests do not establish progress under every possible failure or
scheduling pattern. CPU relays add overhead that depends on host load, topology
and workload.
