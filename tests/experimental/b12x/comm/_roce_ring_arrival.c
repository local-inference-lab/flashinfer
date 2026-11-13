// One lane window of the production relay, with the later packet of the lane
// already visible and its predecessor absent. Two test-only hooks inject the
// predecessor: one after a selected acquire returned its old value, one at every
// ready-flag publication the relay performs itself. Both let the ordered remainder
// of the lane appear exactly where a live slot scan would have moved past it. Three
// receive roles take three different paths through ring_service_slot, and both rails
// are driven because serials and acknowledgments are per rail. The proxy names its
// worker threads, so the feature macro has to precede every system header here.
#define _GNU_SOURCE
#include <infiniband/verbs.h>
#include <stdarg.h>
#include <stdint.h>

static void after_load(const void* address);
static void after_store(const void* address);
#define __atomic_load_n(ptr, order)                            \
  ({                                                           \
    __auto_type test_ptr = (ptr);                              \
    __auto_type test_value = __atomic_load_n(test_ptr, order); \
    after_load((const void*)test_ptr);                         \
    test_value;                                                \
  })
#define __atomic_store_n(ptr, value, order)    \
  ({                                           \
    __atomic_store_n((ptr), (value), (order)); \
    after_store((const void*)(ptr));           \
  })
#include ROCE_PROXY_SOURCE
#undef __atomic_load_n
#undef __atomic_store_n

#define BOX_BYTES (sizeof(ring_descriptor_t) + RING_OPEN_CHUNK_BYTES)
#define TEST_BYTES (2 * RING_OPEN_CHUNK_BYTES + 128)
#define TEST_FLAG (2 * RING_OPEN_CHUNK_BYTES)
#define TEST_SEQ 123u
#define TEST_LANE RING_OPEN_WINDOW
#define ORIGIN 0
#define TARGET 3
#define WORLD 4
#define SINK_BYTES 8192
#define CQ_SIZE 64
#define EARLY_FILL 0x11
#define LATE_FILL 0x22
// The two payload writes of the lane, in the order the relay must forward them.
static const uint64_t lane_offset[2] = {0, RING_OPEN_CHUNK_BYTES};
static const uint32_t lane_bytes[2] = {RING_OPEN_CHUNK_BYTES, RING_OPEN_CHUNK_BYTES / 2};
static const uint8_t lane_fill[2] = {EARLY_FILL, LATE_FILL};
// mode 0: last unicast hop toward TARGET, payload taken from the mailbox.
// mode 1: chain endpoint, payload in the final receive row, flag published locally.
// mode 2: intermediate fanout hop, payload in the receive row, forwarded downstream.
enum { MODE_FINAL_HOP = 0, MODE_ENDPOINT = 1, MODE_INTERMEDIATE = 2 };

static roce_ctx_t relay;
static struct ibv_context device;
static struct ibv_qp qps[WORLD];
static struct ibv_mr mr;
static struct ibv_cq cq;
static struct ibv_wc completions[CQ_SIZE];
static unsigned cq_head, cq_tail;
static uint8_t output[TEST_BYTES];
// mode 2 forwards downstream, so rank 2 gets a private receive row and a private
// mailbox set, biased so slot `lane + k` lands at hop_boxes[k].
static uint8_t hop_data[TEST_BYTES];
static uint8_t hop_boxes[TEST_LANE * BOX_BYTES];
// Acknowledgments go to the upstream neighbour's acknowledgment words. Point that
// neighbour's ring base below a private sink so the address stays in this process.
static uint8_t ack_sink[SINK_BYTES];
static const uint8_t* sink = ack_sink + SINK_BYTES / 2;
static int mode, rail, trigger, pass_number, injected, early_flag, early_forward;
static int flags, flags_before, n_events, wanted_events, failed;
static uint64_t first_serial;
static const void* watched;
static int lane_base, upstream_rank;

static void fail(const char* what, ...) {
  fprintf(stderr, "serial=%llu trigger=%d mode=%d rail=%d: ", (unsigned long long)first_serial,
          trigger, mode, rail);
  va_list ap;
  va_start(ap, what);
  vfprintf(stderr, what, ap);
  va_end(ap);
  fputc('\n', stderr);
  failed = 1;
}

static int filled(const uint8_t* p, size_t n, uint8_t v) {
  for (size_t i = 0; i < n; i++)
    if (p[i] != v) return 0;
  return 1;
}

// One packet of the lane. Submitting hops write the descriptor and commit into a
// mailbox; fanout and the last hop send the payload into a final receive row, so only
// the descriptor stays in the mailbox. Each chunk has its own fill, so a crossed,
// stale or double forward is visible.
static void arrive(uint64_t serial, uint64_t offset, uint64_t bytes, int last, uint64_t predecessor,
                   uint8_t fill) {
  int slot = lane_base + (int)((serial - 1) % TEST_LANE);
  ring_mailbox_t* box = ring_inbox(&relay, ORIGIN, rail, slot);
  if (mode == MODE_FINAL_HOP)
    memset(box->data, fill, bytes);
  else
    memset(relay.region + offset, fill, bytes);
  box->descriptor =
      (ring_descriptor_t){.serial = serial,
                          .target = TARGET,
                          .previous = (uint64_t)upstream_rank,
                          .offset = offset,
                          .nbytes = bytes,
                          .flag_offset = last ? TEST_FLAG : UINT64_MAX,
                          .flag_value = TEST_SEQ,
                          .fanout = mode == MODE_FINAL_HOP ? 0 : 1,
                          .fanout_previous = mode == MODE_FINAL_HOP ? 0 : predecessor};
}

// The later packet of the lane is already committed. Its predecessor appears now, in
// source order, after the relay has looked at the predecessor's own slot.
static void inject(void) {
  injected = 1;
  flags_before = flags;
  arrive(first_serial, 0, RING_OPEN_CHUNK_BYTES, 0, first_serial - 1, EARLY_FILL);
}

static void after_load(const void* address) {
  if (trigger != 1 || address != watched || injected) return;
  inject();
}

// The relay publishes a fanout receipt with a local release store rather than an RDMA
// write, so both publication paths are checked at the publication itself.
static int payload_ready(void) {
  return filled(output, RING_OPEN_CHUNK_BYTES, EARLY_FILL) &&
         filled(output + RING_OPEN_CHUNK_BYTES, RING_OPEN_CHUNK_BYTES / 2, LATE_FILL);
}

static void after_store(const void* address) {
  if (address != (const void*)(output + TEST_FLAG)) return;
  flags++;
  // A GPU may consume at this exact point, so the final output alone would miss a
  // late write repairing the corruption.
  if (!payload_ready()) early_flag = 1;
}

// The payload rows of this process: the relay's own region, plus the downstream row
// that mode 2 forwards into.
static int row_offset(uintptr_t dst, uint64_t* offset) {
  if (dst >= (uintptr_t)output && dst - (uintptr_t)output < TEST_BYTES) {
    *offset = dst - (uintptr_t)output;
    return 1;
  }
  if (dst >= (uintptr_t)hop_data && dst - (uintptr_t)hop_data < TEST_BYTES) {
    *offset = dst - (uintptr_t)hop_data;
    return 1;
  }
  return 0;
}

static int post_send(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad) {
  (void)qp;
  (void)bad;
  for (; wr; wr = wr->next) {
    uintptr_t dst = (uintptr_t)wr->wr.rdma.remote_addr;
    for (int i = 0; i < wr->num_sge; i++) {
      uintptr_t src = (uintptr_t)wr->sg_list[i].addr;
      uint32_t len = wr->sg_list[i].length;
      uint64_t offset;
      if (len > 8 && row_offset(dst, &offset)) {
        // Each forwarded chunk must carry its own bytes, in lane order.
        if (n_events >= 2 || offset != lane_offset[n_events] || len != lane_bytes[n_events] ||
            !filled((const uint8_t*)src, len, lane_fill[n_events]))
          fail("unexpected payload write: offset=%llu bytes=%u", (unsigned long long)offset, len);
        n_events++;
        // Nothing may reach a receive row before the lane predecessor, except in the
        // ack-reuse trigger, which delivers the predecessor first.
        if (pass_number == 0) early_forward = 1;
      }
      memcpy((void*)dst, (const void*)src, len);
      dst += len;
    }
    if (wr->wr.rdma.remote_addr == (uintptr_t)(output + TEST_FLAG)) {
      flags++;
      if (!payload_ready()) early_flag = 1;
    }
    // An intermediate fanout hop commits the descriptor into the next hop's mailbox. A
    // correct next hop accepts that serial and writes it back into the upstream
    // acknowledgment word, which is what lets this relay retire the slot.
    if (mode == MODE_INTERMEDIATE && wr->num_sge == 1 && wr->sg_list[0].length == 8 &&
        wr->wr.rdma.remote_addr - (uintptr_t)hop_boxes < sizeof hop_boxes &&
        (wr->wr.rdma.remote_addr - (uintptr_t)hop_boxes) % BOX_BYTES == 0) {
      size_t k = (size_t)(wr->wr.rdma.remote_addr - (uintptr_t)hop_boxes) / BOX_BYTES;
      memcpy(ring_ack(&relay, ORIGIN, rail, lane_base + (int)k), (const void*)wr->sg_list[0].addr,
             sizeof(uint64_t));
    }
    if (trigger == 3 && !injected && wr->wr.rdma.remote_addr >= (uintptr_t)ack_sink &&
        wr->wr.rdma.remote_addr < (uintptr_t)ack_sink + SINK_BYTES) {
      // Returning the retired slot's credit lets the source refill it at once.
      inject();
    }
    if (wr->send_flags & IBV_SEND_SIGNALED) {
      if (cq_tail - cq_head == CQ_SIZE) abort();
      completions[cq_tail++ % CQ_SIZE] =
          (struct ibv_wc){.wr_id = wr->wr_id, .status = IBV_WC_SUCCESS};
    }
  }
  return 0;
}

static int poll_cq(struct ibv_cq* ibcq, int count, struct ibv_wc* wc) {
  (void)ibcq;
  int n = 0;
  while (n < count && cq_head != cq_tail) wc[n++] = completions[cq_head++ % CQ_SIZE];
  return n;
}

int main(int argc, char** argv) {
  if (argc != 5) return 2;
  first_serial = strtoull(argv[1], NULL, 0);
  trigger = atoi(argv[2]);
  mode = atoi(argv[3]);
  rail = atoi(argv[4]);
  if (first_serial < TEST_LANE || trigger < 1 || trigger > 3 || mode < 0 ||
      mode > MODE_INTERMEDIATE || rail < 0 || rail >= ROCE_MAX_RAILS)
    return 2;
  // The ack-reuse trigger needs a retired predecessor serial, so the very first window
  // cannot supply one.
  if (trigger == 3 && first_serial == TEST_LANE) return 2;

  device.ops.post_send = post_send;
  device.ops.poll_cq = poll_cq;
  cq.context = &device;
  relay.world = WORLD;
  // 0 -> 1 -> 2 -> 3 on an open chain: the relay sits at its upstream neighbour plus
  // one, so the upstream rank is always rank - 1.
  relay.rank = mode == MODE_FINAL_HOP ? 2 : mode == MODE_ENDPOINT ? 3 : 1;
  upstream_rank = relay.rank - 1;
  relay.n_rail = ROCE_MAX_RAILS;
  relay.n_hca = 1;
  relay.ring_open = 1;
  relay.region = output;
  relay.region_bytes = TEST_BYTES;
  relay.ring = calloc(1, roce_ring_bytes(WORLD, 1));
  if (!relay.ring) abort();
  relay.ring_sources = ring_source_mask(&relay);
  for (int target = 0; target < WORLD; target++)
    relay.ring_destinations[target] = ring_destination_mask(&relay, target);
  relay.hca[0].cq = &cq;
  relay.hca[0].mr = &mr;
  relay.ring_mr[0] = &mr;
  lane_base = ring_lane_base(ORIGIN, TARGET);
  relay.peer_addr[TARGET] = (uintptr_t)output;
  if (mode == MODE_INTERMEDIATE) {
    relay.peer_addr[2] = (uintptr_t)hop_data;
    relay.ring_addr[2] = (uintptr_t)hop_boxes -
                         ring_inbox_offset(&relay, ring_slot(&relay, ORIGIN, rail, lane_base));
  }
  for (int p = 0; p < WORLD; p++) {
    qps[p].context = &device;
    relay.hca[0].qp[p][0] = &qps[p];
    for (int r = 0; r < ROCE_MAX_RAILS; r++) relay.rail_hca[p][r] = 0;
  }
  int first_slot = lane_base + (int)((first_serial - 1) % TEST_LANE);
  relay.ring_addr[upstream_rank] =
      (uintptr_t)sink - ring_ack_offset(&relay, ring_slot(&relay, ORIGIN, rail, first_slot));
  // Prior packets were acknowledged, so their slots are reusable, and the lane's
  // predecessor chain is already past them.
  for (int slot = 0; slot < TEST_LANE; slot++) {
    uint64_t prior = (uint64_t)slot + 1;
    while (prior + TEST_LANE < first_serial) prior += TEST_LANE;
    // A window that has not filled yet has no earlier occupant for its last slot.
    if (prior >= first_serial) prior = 0;
    relay.ring_seen[ORIGIN][rail][lane_base + slot] = prior;
    ring_inbox(&relay, ORIGIN, rail, lane_base + slot)->descriptor.serial = prior;
  }
  relay.ring_order[ORIGIN][TARGET][rail] = first_serial - 1;
  // The later packet is committed before the relay looks at anything, and names the
  // packet that has not arrived yet as its own predecessor.
  arrive(first_serial + 1, RING_OPEN_CHUNK_BYTES, RING_OPEN_CHUNK_BYTES / 2, 1, first_serial,
         LATE_FILL);
  watched = &ring_inbox(&relay, ORIGIN, rail, first_slot)->descriptor.serial;
  if (trigger == 3) {
    // The predecessor's slot is still owed an end-to-end path: the relay forwarded its
    // previous occupant and has not been acknowledged yet.
    relay.ring_seen[ORIGIN][rail][first_slot] = 0;
    relay.ring_pending[ORIGIN][rail][first_slot] = 2;
    relay.ring_previous[ORIGIN][rail][first_slot] = (uint8_t)upstream_rank;
    relay.ring_inflight[ORIGIN][rail][first_slot] = first_serial - TEST_LANE;
    *ring_ack(&relay, ORIGIN, rail, first_slot) = first_serial - TEST_LANE;
  }
  int flags_in_first_pass = -1;
  for (int pass = 0; pass < 8; pass++) {
    pass_number = pass;
    // The second timing lets a whole pass observe the later packet with its
    // predecessor still absent, which is the state a live slot scan forwards in.
    if (trigger == 2 && pass == 1) inject();
    if (drain_all(&relay)) {
      fail("%s", relay.err);
      break;
    }
    if (pass == 0) flags_in_first_pass = flags;
  }
  uint32_t observed_flag;
  memcpy(&observed_flag, output + TEST_FLAG, sizeof(observed_flag));
  // Every host a fanout packet passes publishes its own row's flag, so even the
  // intermediate role publishes exactly one, before it forwards.
  if (!injected || flags != 1 || early_flag || observed_flag != TEST_SEQ)
    fail("injected=%d flags=%d early_flag=%d flag=%u", injected, flags, early_flag, observed_flag);
  if (trigger != 3 && flags_in_first_pass)
    fail("published %d flags in the pass that saw only the later packet", flags_in_first_pass);
  // An endpoint consumes locally, so it forwards nothing. Every other role must ship
  // the predecessor's chunk before the later chunk.
  wanted_events = mode == MODE_ENDPOINT ? 0 : 2;
  if (n_events != wanted_events) fail("forwarded %d payloads, wanted %d", n_events, wanted_events);
  if (trigger != 3 && early_forward) fail("wrote a receive row before ordering");
  if (relay.ring_order[ORIGIN][TARGET][rail] != first_serial + 1)
    fail("chain reached %llu", (unsigned long long)relay.ring_order[ORIGIN][TARGET][rail]);
  if (flags_before) fail("published %d flags before the predecessor", flags_before);
  for (int slot = lane_base; slot < lane_base + TEST_LANE; slot++)
    if (relay.ring_pending[ORIGIN][rail][slot]) failed = 1;
  if (relay.hca[0].outstanding[upstream_rank][0]) failed = 1;
  if (mode == MODE_INTERMEDIATE && relay.hca[0].outstanding[2][0]) failed = 1;
  free(relay.ring);
  if (!failed) puts("ordered relay arrival: ok");
  return failed;
}
