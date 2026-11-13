// What a relayed packet needs and where it may go, driven against the production
// proxy with a fake verbs provider: the queue pair must declare the two send SGEs
// a per-destination packet uses, the startup mailbox has to be sized, a context that failed or
// was stopped inside an operation must never restart, and a fanout packet may only
// replace a per-destination relay where some destination is not a neighbor. The
// provider enforces the declared cap and copies every accepted write where a NIC
// would, so no RDMA hardware is needed.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <stdint.h>
#include ROCE_PROXY_SOURCE

#define WORLD 4
#define ORIGIN 0
#define RELAY 1
#define NEXT 2
#define TARGET 3
#define PAYLOAD 16
#define SCRATCH_BYTES 4096
#define REGION_BYTES (256u << 10)
#define SEQ 9u
#define TOTAL_PACKS 4096u  // 64 KiB two-shot or shard-gather message
#define CHUNK_PACKS 512u   // 8 KiB per pipelined chunk
#define CHUNK_BYTES (CHUNK_PACKS * 16u)
#define SHARD_CAP 32768u
#define ROW_BYTES (2u << 20)
// Startup ring budget for a four-rank line: 768 mailboxes (four origins, two
// rails, 32 slots per destination) of a 256 KiB mailbox, an 8-byte acknowledgment
// word and a 128-byte outgoing descriptor. The same group on a closed ring keeps a
// shared eight-slot window, so it reserves 64 of those mailboxes at 64 KiB.
#define RING_BYTES_WORLD4 UINT64_C(201529344)
#define RING_BYTES_RING4 UINT64_C(4211200)
#define RING_BYTES(world, open)                                               \
  ((uint64_t)(world) * ROCE_MAX_RAILS * ring_lane_slots_at((world), (open)) * \
   (uint64_t)(ring_box_at((world), (open)) + sizeof(uint64_t) + sizeof(ring_descriptor_t)))

static struct ibv_context device;
static struct ibv_qp qps[ROCE_MAX_PEERS];
static struct ibv_mr mr;
static struct ibv_cq cq;
static uint8_t region[REGION_BYTES];
static uint8_t scratch[SCRATCH_BYTES];
static uint8_t payload[PAYLOAD];
static uint8_t* rows[ROCE_MAX_PEERS];
static uint8_t* mirror;
static uint8_t* ring;
static size_t mirror_bytes;
static roce_ctx_t relay;
static roce_ctx_t terminal;
static int provider_sge = ROCE_MAX_SEND_SGE, saw_two_sge, rejected, test_slot;
static uint64_t serial = 77;
static int row_payload[ROCE_MAX_PEERS];
static int row_flags[ROCE_MAX_PEERS];
// Writes that reached a next hop's mailbox: descriptor bodies, commit serials and
// payload bytes.
static int box_descriptors, box_commits, box_bodies, unexpected, failures;

static void require(int condition, const char* what) {
  if (condition) return;
  fprintf(stderr, "failed: %s\n", what);
  failures++;
}

// Account every write by where it lands, then copy it the way a NIC would, so the
// mailbox contents can be read back afterwards. Writes to the relay's own region or
// to a peer's row outside a mailbox are not a relay packet and go unaccounted.
static void account(uintptr_t dst, uint32_t len) {
  for (int p = 0; p < relay.world; p++) {
    uintptr_t base = (uintptr_t)rows[p];
    if (dst < base || dst >= base + ROW_BYTES) continue;
    if (len >= CHUNK_BYTES)
      row_payload[p] += (int)len;
    else if (len == 4)
      row_flags[p]++;
    else
      unexpected++;
    return;
  }
  if (dst < (uintptr_t)mirror || dst >= (uintptr_t)mirror + mirror_bytes) {
    if (len > sizeof(ring_descriptor_t) - 8) unexpected++;
  } else if (len == sizeof(ring_descriptor_t) - 8) {
    box_descriptors++;
  } else if (len == 8) {
    box_commits++;
  } else if (len > 8) {
    box_bodies += (int)len;
  } else {
    unexpected++;
  }
}

static int post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad) {
  (void)qp;
  for (; wr; wr = wr->next) {
    if ((int)wr->num_sge > provider_sge) {
      rejected++;
      *bad = wr;
      return ENOMEM;
    }
    if (wr->num_sge == 2) saw_two_sge = 1;
    uintptr_t dst = (uintptr_t)wr->wr.rdma.remote_addr;
    for (int i = 0; i < wr->num_sge; i++) {
      const void* src = (const void*)(uintptr_t)wr->sg_list[i].addr;
      uint32_t len = wr->sg_list[i].length;
      account(dst, len);
      memcpy((void*)dst, src, len);
      dst += len;
    }
  }
  return 0;
}

static int poll_cq(struct ibv_cq* ibcq, int count, struct ibv_wc* wc) {
  (void)ibcq;
  (void)count;
  (void)wc;
  return 0;
}

// Every peer's mailbox, and every peer's row, is a block of this process, so one
// rank's writes are read back by another's relay. Every mailbox starts out empty.
static void reset(int world, int rank, int n_lane, int ring_open) {
  memset(&relay, 0, sizeof relay);
  for (int p = 0; p < ROCE_MAX_PEERS; p++) row_payload[p] = row_flags[p] = 0;
  mirror_bytes = ring_lane_slots_at(world, ring_open) * ring_box_at(world, ring_open);
  memset(mirror, 0, mirror_bytes);
  memset(ring, 0, roce_ring_bytes(world, ring_open));
  box_descriptors = box_commits = box_bodies = 0;
  device.ops.post_send = post_send;
  device.ops.poll_cq = poll_cq;
  cq.context = &device;
  relay.world = world;
  relay.rank = rank;
  relay.n_rail = 1;
  relay.n_hca = 1;
  relay.n_lane = n_lane;
  relay.ring_open = ring_open;
  relay.region = region;
  relay.region_bytes = REGION_BYTES;
  relay.pipe_send_off = 0;
  relay.pipe_recv_off = 0;
  relay.pipe_flag_off = 1u << 20;
  relay.shard_cap = SHARD_CAP;
  relay.ring = ring;
  relay.hca[0].cq = &cq;
  relay.hca[0].mr = &mr;
  relay.ring_mr[0] = &mr;
  for (int p = 0; p < world; p++) {
    qps[p].context = &device;
    for (int lane = 0; lane < ROCE_MAX_LANES; lane++) relay.hca[0].qp[p][lane] = &qps[p];
    relay.peer_addr[p] = (uintptr_t)rows[p];
    relay.ring_addr[p] = (uintptr_t)mirror;
    for (int k = 0; k < ROCE_MAX_RAILS; k++) relay.rail_hca[p][k] = 0;
  }
}

// The mailbox holding origin ORIGIN's first packet toward `target`: the first slot
// of that destination's dense lane, or the first slot of the shared window.
static ring_mailbox_t* inbox(int target) {
  int slot = ring_slot_of(&relay, ORIGIN, target, 1);
  return (ring_mailbox_t*)(mirror + ring_inbox_offset(&relay, ring_slot(&relay, ORIGIN, 0, slot)));
}

static void post_chunk(void) {
  memset(region, 0x5a, REGION_BYTES);
  require(post_pipe_chunk(&relay, SEQ, 1, 0, TOTAL_PACKS, CHUNK_PACKS) == 0,
          "the all-gather chunk posts");
}

// A direct write reaches the peer's receive row and its ready flag.
static void require_direct(int target, const char* what) {
  require(row_payload[target] == (int)CHUNK_BYTES, what);
  require(row_flags[target] == 1, "the direct write carries its ready flag");
}

// A per-destination relay packet holds descriptor and payload in the mailbox.
static void require_relayed(int target, const char* what) {
  ring_mailbox_t* box = inbox(target);
  require(box->descriptor.serial == 1, what);
  require(box->descriptor.target == (uint64_t)target && box->descriptor.previous == ORIGIN,
          "the relay packet names its destination and its previous hop");
  require(box->descriptor.fanout == 0 && box->descriptor.nbytes == CHUNK_BYTES,
          "a per-destination packet carries one whole chunk");
  require(box_commits >= 1, "the relay packet commits a serial");
  require(memcmp(box->data, region, CHUNK_BYTES) == 0, "the relay packet carries the chunk bytes");
}

int main(void) {
  for (int p = 0; p < ROCE_MAX_PEERS; p++) {
    rows[p] = calloc(1, ROW_BYTES);
    if (rows[p] == NULL) abort();
  }
  mirror = calloc(1, ring_lane_slots_at(ROCE_MAX_PEERS, 1) * ring_box_at(ROCE_MAX_PEERS, 1));
  ring = calloc(1, roce_ring_bytes(ROCE_MAX_PEERS, 1));
  if (mirror == NULL || ring == NULL) abort();
  for (int i = 0; i < PAYLOAD; i++) payload[i] = (uint8_t)(0xa0 + i);

  struct ibv_qp_cap cap;
  roce_qp_cap(&cap);
  require(ROCE_MAX_SEND_SGE == 2 && cap.max_send_sge == 2,
          "a relay packet is one two-SGE write and the queue pair declares it");
  require(cap.max_send_wr == ROCE_SEND_DEPTH, "the declared cap carries the send depth");
  require(cap.max_recv_wr == 1 && cap.max_recv_sge == 1, "the receive side stays minimal");
  require(cap.max_inline_data == 16, "the inline cap covers the commit serial");

  require(
      roce_ring_bytes(WORLD, 1) == RING_BYTES_WORLD4 && RING_BYTES(WORLD, 1) == RING_BYTES_WORLD4,
      "the four-rank broadcasting mailbox is 192.2 MiB");
  require(roce_ring_bytes(WORLD, 0) == RING_BYTES_RING4 && RING_BYTES(WORLD, 0) == RING_BYTES_RING4,
          "a four-rank ring that only relays per destination reserves 4.0 MiB");
  require(roce_ring_bytes(8, 1) == RING_BYTES(8, 1) && roce_ring_bytes(16, 1) == RING_BYTES(16, 1),
          "the broadcasting mailbox grows with the square of the world size");
  require(roce_ring_bytes(8, 0) == RING_BYTES(8, 0) && roce_ring_bytes(16, 0) == RING_BYTES(16, 0),
          "a closed ring's mailbox grows with the world size alone");
  require(roce_ring_bytes(2, 1) == RING_BYTES(2, 0) && roce_ring_bytes(3, 1) == RING_BYTES(3, 0),
          "a line too short to broadcast keeps the small geometry");
  require(roce_ring_bytes(1, 0) == 0 && roce_ring_bytes(ROCE_MAX_PEERS + 1, 0) == 0,
          "an invalid world size is refused");

  // One relayed unicast packet from ORIGIN toward TARGET, forwarded by RELAY to its
  // neighbour NEXT. NEXT's mailbox is biased onto a private block, so the provider
  // can read back what a real peer would have received.
  reset(WORLD, RELAY, 1, 1);
  test_slot = ring_slot_of(&relay, ORIGIN, TARGET, serial);
  relay.ring_addr[NEXT] =
      (uintptr_t)scratch - ring_inbox_offset(&relay, ring_slot(&relay, ORIGIN, 0, test_slot));
  ring_descriptor_t descriptor = {
      .serial = serial, .target = TARGET, .previous = ORIGIN, .flag_offset = UINT64_MAX};
  descriptor.nbytes = PAYLOAD;
  require(ring_submit(&relay, ORIGIN, 0, test_slot, NEXT, payload, mr.lkey, &descriptor) == 0,
          "the relayed 16-byte write posts on a two-SGE provider");
  require(saw_two_sge, "the packet really used descriptor plus payload in one work request");
  require(relay.hca[0].outstanding[NEXT][0] == 1 && relay.hca[0].bytes_posted == PAYLOAD,
          "only the signaled commit is outstanding and the payload is counted once");
  require(unexpected == 0, "the relayed packet wrote only the mailbox it was addressed to");
  require(*(uint64_t*)scratch == serial, "the commit serial reached the next hop's mailbox");
  require(memcmp(scratch + sizeof(ring_descriptor_t), payload, PAYLOAD) == 0,
          "the payload reached the next hop's mailbox");

  // Without the capacity the same packet must fail loudly and account nothing as
  // accepted: a half-posted operation can never be retried or released.
  provider_sge = 1;
  saw_two_sge = rejected = 0;
  relay.hca[0].outstanding[NEXT][0] = 0;
  relay.hca[0].bytes_posted = 0;
  memset(scratch, 0, SCRATCH_BYTES);
  require(ring_submit(&relay, ORIGIN, 0, test_slot, NEXT, payload, mr.lkey, &descriptor) == -1,
          "a one-SGE provider refuses the relayed packet");
  require(rejected == 1 && !saw_two_sge, "the rejection happens at the first wide request");
  require(strstr(relay.err, "ring batch submit") != NULL, "the refusal names the submit path");
  require(relay.hca[0].outstanding[NEXT][0] == 0 && relay.hca[0].bytes_posted == 0 &&
              *(uint64_t*)scratch == 0,
          "a rejected packet counts no bytes, releases no owner and writes nothing");
  provider_sge = ROCE_MAX_SEND_SGE;

  // A quiescent context restarts; one that failed, or was stopped inside an
  // operation, refuses to start again.
  terminal.world = WORLD;
  terminal.region = region;
  terminal.region_bytes = REGION_BYTES;
  require(roce_start(&terminal) == 0, "a healthy context starts");
  roce_stop(&terminal);
  require(atomic_load(&terminal.failed) == 0, "a quiescent stop is not a failure");
  require(roce_start(&terminal) == 0, "a quiescent context starts again");
  roce_stop(&terminal);
  atomic_store(&terminal.failed, 1);
  require(roce_start(&terminal) == -1, "a failed context refuses to start");
  require(strstr(terminal.err, "failed or was stopped") != NULL,
          "the refusal explains that the context is terminal");
  atomic_store(&terminal.failed, 0);
  terminal.err[0] = '\0';
  terminal.in_op = 1;
  roce_stop(&terminal);
  require(atomic_load(&terminal.failed) == 1, "a stopped-in-operation context is terminal");
  require(strstr(terminal.err, "stopped during an operation") != NULL,
          "the stop records why the context is terminal");
  require(roce_start(&terminal) == -1, "a terminal context refuses to start again");

  // Two- and three-rank open chains retain per-destination delivery. A three-rank
  // chain has a non-neighbor endpoint pair, but broadcast is enabled only from
  // four ranks; this check verifies that threshold rather than a speedup claim.
  reset(2, ORIGIN, 4, 1);
  require(ring_fanout(&relay) == 0, "a two-rank line does not fan out");
  post_chunk();
  require_direct(1, "a two-rank line writes its peer's row");
  require(box_descriptors == 0 && box_commits == 0 && box_bodies == 0,
          "a two-rank line posts no relay packet");

  reset(3, ORIGIN, 4, 1);
  require(ring_fanout(&relay) == 0, "a three-rank line does not fan out");
  post_chunk();
  require_direct(1, "a three-rank line writes its near peer's row");
  require_relayed(2, "a three-rank line relays to its far peer");
  require(box_descriptors == 1, "a three-rank line posts one relay packet");

  // Four ranks: one fanout packet, its payload written into the next hop's final
  // receive row and only its descriptor left in the mailbox.
  reset(4, ORIGIN, 4, 1);
  require(ring_fanout(&relay) == 1, "a four-rank line fans out");
  post_chunk();
  ring_mailbox_t* box = inbox(3);
  require(box->descriptor.serial == 1, "a four-rank line posts one fanout packet");
  require(box->descriptor.target == 3 && box->descriptor.fanout == 1,
          "the fanout packet names the far endpoint and is marked as one");
  require(box->descriptor.nbytes == CHUNK_BYTES, "the fanout packet carries one chunk");
  require(box_descriptors == 1 && box_commits == 1 && box_bodies == 0,
          "the fanout packet leaves one descriptor and no payload in the mailbox");
  require(row_payload[1] == (int)CHUNK_BYTES, "the fanout payload lands in the next hop's row");
  require(row_flags[1] == 0, "a fanout payload publishes no local flag");
  require(row_payload[2] == 0 && row_payload[3] == 0, "a fanout writes no other peer's row");

  // Four ranks on a closed ring: the far peer uses per-destination relay delivery.
  reset(4, ORIGIN, 4, 0);
  require(ring_fanout(&relay) == 0, "a closed ring does not fan out");
  post_chunk();
  require_direct(1, "a closed ring writes its neighbor's row");
  require_direct(3, "a closed ring writes its other neighbor's row");
  require_relayed(2, "a closed ring relays to the rank across the ring");
  require(box_descriptors == 1, "a closed ring posts one relay packet");
  require(unexpected == 0, "every write landed in a receive row or the next hop's mailbox");

  if (!failures) puts("relay guard: ok");
  return failures != 0;
}
