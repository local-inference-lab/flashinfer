// p2pstall: PCIe peer-to-peer collective stall harness (diagnostic tool, not production code).
//
// One process drives all selected GPUs. Each GPU runs one persistent kernel that executes `iters`
// iterations of a communication pattern without any CPU involvement. Every CTA stamps %globaltimer at
// iteration start, after the cross-GPU start barrier, and after its transfer (including delivery
// confirmation for push patterns). Per iteration the harness keeps min(start), max(ready), max(end)
// over CTAs, so a stall in any CTA of any GPU is visible with its absolute time.
//
// Patterns (S = --bytes per rank and destination/source):
//   read_ring     rank r reads S from r+1                       (one reader per target)
//   read_all      rank r reads S from every peer               (one-shot pull traffic)
//   read_incast   ranks != 0 read S from rank 0
//   read_pair     rank 0 reads S from rank 1; others idle
//   write_ring    rank r writes S to r+1, flags it, waits for r-1
//   write_all     rank r writes S to every peer, interleaved per vector (fan-out)
//   write_phased  rank r writes S to r+1, r+2, ... one destination per GPU-wide phase (rotation)
//   write_incast  ranks != 0 write S to rank 0
//   write_pair    rank 0 writes S to rank 1
//   pull          one-shot all-reduce: read all peers' inputs, reduce (u32 add), store
//   push          one-shot all-reduce: fan-out write to every peer's inbox, reduce local inbox
//   push_phased   one-shot all-reduce: rotation write, one destination per GPU-wide phase (Fraser BRRR style)
//   ring          ring reduce-scatter + all-gather, one destination per rank (NCCL Simple style)
//   rsag_push     reduce-scatter by fan-out shard writes, all-gather by fan-out writes
//   rsag_phased   rsag_push with one destination per GPU-wide phase in both halves
//   rsag_pull     reduce-scatter by reading own shard from peers, all-gather by reading reduced shards
//   host_read     every rank reads S from pinned host memory
//   host_write    every rank writes S to pinned host memory
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <time.h>
#include <atomic>
#include <thread>
#include <chrono>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));             \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

constexpr int MAXR = 8;
constexpr int MAXCTA = 128;
constexpr int KINDS = 6;
constexpr int kWarm = 16;  // iterations excluded from statistics (launch skew)
enum Kind { K_START, K_DATA, K_RS, K_AG, K_STEP, K_RED };

enum Pattern {
  P_READ_RING, P_READ_ALL, P_READ_INCAST, P_READ_PAIR,
  P_WRITE_RING, P_WRITE_ALL, P_WRITE_PHASED, P_WRITE_INCAST, P_WRITE_PAIR,
  P_PULL, P_PUSH, P_PUSH_PHASED, P_RING, P_RSAG_PUSH, P_RSAG_PHASED, P_RSAG_PULL,
  P_HOST_READ, P_HOST_WRITE, P_COUNT
};
static const char* kPatternNames[P_COUNT] = {
  "read_ring", "read_all", "read_incast", "read_pair",
  "write_ring", "write_all", "write_phased", "write_incast", "write_pair",
  "pull", "push", "push_phased", "ring", "rsag_push", "rsag_phased", "rsag_pull",
  "host_read", "host_write"};

struct Args {
  int rank, nranks, pattern, iters, sync, kernel_sync;
  long vecs;                       // S / 16
  unsigned long long gap_ns, timeout_ns;
  const uint4* in;                 // local input, vecs
  uint4* out;                      // local output, vecs
  uint4* work;                     // local scratch, vecs
  const uint4* peer_in[MAXR];      // every rank's input (pull patterns)
  uint4* peer_red[MAXR];           // every rank's reduced-shard buffer (rsag_pull)
  uint4* peer_inbox[MAXR];         // every rank's inbox: [parity 2][slot MAXR*2][vecs]
  unsigned* peer_flags[MAXR];      // every rank's flags: [kind][src][cta]
  unsigned* gpu_bar;               // local GPU-wide phase counter
  uint4* host;                     // pinned host buffer for this rank, vecs
  unsigned long long *t_start, *t_ready, *t_end;
  unsigned* err;
};

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned ld_volatile_u32(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint4 ld_v4(const uint4* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_v4(uint4* p, uint4 v) {
  asm volatile("st.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}
__device__ __forceinline__ void add4(uint4& a, const uint4& b) {
  a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

__device__ __forceinline__ unsigned* flag_at(unsigned* base, int kind, int src, int cta) {
  return base + (kind * MAXR + src) * MAXCTA + cta;
}

// Waits until *p >= want (monotonic counters). Records a timeout instead of hanging the GPU.
__device__ __forceinline__ void wait_geq(const unsigned* p, unsigned want, const Args& a) {
  unsigned long long t0 = 0;
  int n = 0;
  while ((int)(ld_acquire_sys(p) - want) < 0) {
    if (++n == 1024) {
      n = 0;
      unsigned long long t = gtimer();
      if (t0 == 0) t0 = t;
      else if (t - t0 > a.timeout_ns) { atomicExch(a.err, 1u); return; }
    }
  }
}

__device__ __forceinline__ uint4* inbox_slot(uint4* inbox, long vecs, int parity, int slot) {
  return inbox + ((long)parity * (2 * MAXR) + slot) * vecs;
}

// Cross-GPU start barrier: thread t signals peer t, then waits for peer t.
__device__ void start_barrier(const Args& a, unsigned seq) {
  int t = threadIdx.x;
  if (t < a.nranks && t != a.rank) {
    st_release_sys(flag_at(a.peer_flags[t], K_START, a.rank, blockIdx.x), seq);
    wait_geq(flag_at(a.peer_flags[a.rank], K_START, t, blockIdx.x), seq, a);
  }
  __syncthreads();
}

// Publishes "this CTA's writes for `kind` are complete" to destination `dst`.
__device__ __forceinline__ void signal(const Args& a, int dst, int kind, unsigned seq) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    st_release_sys(flag_at(a.peer_flags[dst], kind, a.rank, blockIdx.x), seq);
  }
}

// Waits for `src`'s CTA-matching signal.
__device__ __forceinline__ void await(const Args& a, int src, int kind, unsigned seq) {
  if (threadIdx.x == 0) wait_geq(flag_at(a.peer_flags[a.rank], kind, src, blockIdx.x), seq, a);
  __syncthreads();
}

// GPU-wide phase barrier across all CTAs of this kernel (they are co-resident by construction).
__device__ __forceinline__ void gpu_phase(const Args& a, unsigned phase) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(a.gpu_bar, 1u);
    unsigned want = phase * gridDim.x;
    unsigned long long t0 = gtimer();
    while ((int)(ld_volatile_u32(a.gpu_bar) - want) < 0) {
      if (gtimer() - t0 > a.timeout_ns) { atomicExch(a.err, 1u); break; }
    }
  }
  __syncthreads();
}

// Range of 16-byte vectors owned by this CTA within [0, n).
__device__ __forceinline__ void cta_range(long n, long& lo, long& hi) {
  long per = (n + gridDim.x - 1) / gridDim.x;
  lo = min(n, (long)blockIdx.x * per);
  hi = min(n, lo + per);
}

// Reads [lo,hi) from `count` sources concurrently (all loads in flight before the adds), adds into dst.
template <int U>
__device__ __forceinline__ void gather_add(const uint4* const* srcs, int count, long lo, long hi,
                                           const uint4* local, uint4* dst) {
  for (long base = lo + threadIdx.x; base < hi; base += (long)blockDim.x * U) {
    uint4 acc[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      long i = base + (long)u * blockDim.x;
      acc[u] = (local && i < hi) ? local[i] : make_uint4(0, 0, 0, 0);
    }
    uint4 v[MAXR][U];
#pragma unroll
    for (int s = 0; s < MAXR; ++s) {
      if (s < count) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          long i = base + (long)u * blockDim.x;
          if (i < hi) v[s][u] = ld_v4(srcs[s] + i);
        }
      }
    }
#pragma unroll
    for (int s = 0; s < MAXR; ++s) {
      if (s < count) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
          long i = base + (long)u * blockDim.x;
          if (i < hi) add4(acc[u], v[s][u]);
        }
      }
    }
    if (dst) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        long i = base + (long)u * blockDim.x;
        if (i < hi) dst[i] = acc[u];
      }
    } else {
      // Keep the loads alive without a store: fold into a never-true branch.
#pragma unroll
      for (int u = 0; u < U; ++u)
        if (acc[u].x == 0xdeadbeefu && acc[u].y == 0x12345678u) asm volatile("trap;");
    }
  }
}

// Writes local [lo,hi) to `count` destinations. Interleaved: every vector goes to all destinations
// before the next vector (fan-out). Otherwise destination by destination.
__device__ __forceinline__ void scatter(uint4* const* dsts, int count, long lo, long hi, const uint4* src) {
  for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    uint4 v = src[i];
#pragma unroll
    for (int s = 0; s < MAXR; ++s)
      if (s < count) st_v4(dsts[s] + i, v);
  }
}

__global__ void __launch_bounds__(512) run(Args a) {
  const int R = a.nranks, r = a.rank;
  const long V = a.vecs;
  unsigned long long prev_end = 0;
  unsigned phase = 0;
  long lo, hi;
  cta_range(V, lo, hi);
  // Shard ranges for RS/AG patterns: shard j = [j*V/R, (j+1)*V/R), split across CTAs.
  for (int it = 0; it < a.iters; ++it) {
    if (a.gap_ns && prev_end) {
      while (gtimer() < prev_end + a.gap_ns) {}
    }
    // Kernel-boundary emulation: every CTA of this GPU starts the operation together.
    if (a.kernel_sync) gpu_phase(a, ++phase);
    unsigned long long ts = gtimer();
    const unsigned seq = (unsigned)it + 1;
    const int par = it & 1;
    if (a.sync) start_barrier(a, seq);
    unsigned long long tr = gtimer();
    switch (a.pattern) {
      case P_READ_RING: case P_READ_ALL: case P_READ_INCAST: case P_READ_PAIR: {
        const uint4* srcs[MAXR];
        int n = 0;
        if (a.pattern == P_READ_RING) srcs[n++] = a.peer_in[(r + 1) % R];
        else if (a.pattern == P_READ_ALL) { for (int k = 1; k < R; ++k) srcs[n++] = a.peer_in[(r + k) % R]; }
        else if (a.pattern == P_READ_INCAST) { if (r != 0) srcs[n++] = a.peer_in[0]; }
        else if (r == 0) srcs[n++] = a.peer_in[1];
        if (n) gather_add<2>(srcs, n, lo, hi, nullptr, a.work);
        __syncthreads();
        break;
      }
      case P_WRITE_RING: case P_WRITE_ALL: case P_WRITE_INCAST: case P_WRITE_PAIR: {
        uint4* dsts[MAXR];
        int dr[MAXR], n = 0;
        if (a.pattern == P_WRITE_RING) dr[n++] = (r + 1) % R;
        else if (a.pattern == P_WRITE_ALL) { for (int k = 1; k < R; ++k) dr[n++] = (r + k) % R; }
        else if (a.pattern == P_WRITE_INCAST) { if (r != 0) dr[n++] = 0; }
        else if (r == 0) dr[n++] = 1;
        for (int k = 0; k < n; ++k) dsts[k] = inbox_slot(a.peer_inbox[dr[k]], V, par, r);
        scatter(dsts, n, lo, hi, a.in);
        for (int k = 0; k < n; ++k) signal(a, dr[k], K_DATA, seq);
        // Receivers wait for the sources that write to them.
        if (a.pattern == P_WRITE_RING) await(a, (r + R - 1) % R, K_DATA, seq);
        else if (a.pattern == P_WRITE_ALL) { for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_DATA, seq); }
        else if (a.pattern == P_WRITE_INCAST) { if (r == 0) for (int k = 1; k < R; ++k) await(a, k, K_DATA, seq); }
        else if (r == 1) await(a, 0, K_DATA, seq);
        __syncthreads();
        break;
      }
      case P_WRITE_PHASED: {
        for (int k = 1; k < R; ++k) {
          int d = (r + k) % R;
          uint4* dst = inbox_slot(a.peer_inbox[d], V, par, r);
          scatter(&dst, 1, lo, hi, a.in);
          signal(a, d, K_DATA, seq);
          if (k < R - 1) gpu_phase(a, ++phase);
        }
        for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_DATA, seq);
        break;
      }
      case P_PULL: {
        const uint4* srcs[MAXR];
        int n = 0;
        for (int k = 1; k < R; ++k) srcs[n++] = a.peer_in[(r + k) % R];
        gather_add<2>(srcs, n, lo, hi, a.in, a.out);
        __syncthreads();
        break;
      }
      case P_PUSH: case P_PUSH_PHASED: {
        if (a.pattern == P_PUSH) {
          uint4* dsts[MAXR];
          int n = 0;
          for (int k = 1; k < R; ++k) dsts[n++] = inbox_slot(a.peer_inbox[(r + k) % R], V, par, r);
          scatter(dsts, n, lo, hi, a.in);
          for (int k = 1; k < R; ++k) signal(a, (r + k) % R, K_DATA, seq);
        } else {
          for (int k = 1; k < R; ++k) {
            int d = (r + k) % R;
            uint4* dst = inbox_slot(a.peer_inbox[d], V, par, r);
            scatter(&dst, 1, lo, hi, a.in);
            signal(a, d, K_DATA, seq);
            if (k < R - 1) gpu_phase(a, ++phase);
          }
        }
        for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_DATA, seq);
        const uint4* srcs[MAXR];
        int n = 0;
        uint4* inbox = a.peer_inbox[r];
        for (int k = 1; k < R; ++k) srcs[n++] = inbox_slot(inbox, V, par, (r + R - k) % R);
        gather_add<2>(srcs, n, lo, hi, a.in, a.out);
        __syncthreads();
        break;
      }
      case P_RING: {
        // Per CTA ring over its own [lo,hi) range, split into R chunks.
        long len = hi - lo, c = (len + R - 1) / R;
        auto chunk = [&](int j, long& b, long& e) { b = lo + min(len, j * c); e = lo + min(len, (j + 1) * c); };
        int nx = (r + 1) % R, pv = (r + R - 1) % R;
        uint4* inbox_next = a.peer_inbox[nx];
        uint4* inbox_me = a.peer_inbox[r];
        // work = in
        for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) a.work[i] = a.in[i];
        __syncthreads();
        const unsigned steps = 2 * (R - 1);
        for (int s = 0; s < (int)steps; ++s) {
          long b, e;
          int send = s < R - 1 ? (r - s + 2 * R) % R : (r + 1 - (s - (R - 1)) + 2 * R) % R;
          chunk(send, b, e);
          uint4* dst = inbox_slot(inbox_next, V, par, s);
          for (long i = b + threadIdx.x; i < e; i += blockDim.x) st_v4(dst + i, a.work[i]);
          signal(a, nx, K_STEP, (unsigned)it * steps + s + 1);
          await(a, pv, K_STEP, (unsigned)it * steps + s + 1);
          int recv = s < R - 1 ? (r - s - 1 + 2 * R) % R : (r - (s - (R - 1)) + 2 * R) % R;
          chunk(recv, b, e);
          const uint4* src = inbox_slot(inbox_me, V, par, s);
          for (long i = b + threadIdx.x; i < e; i += blockDim.x) {
            uint4 v = ld_v4(src + i);
            if (s < R - 1) { uint4 w = a.work[i]; add4(w, v); a.work[i] = w; }
            else a.work[i] = v;
          }
          __syncthreads();
        }
        for (long i = lo + threadIdx.x; i < hi; i += blockDim.x) a.out[i] = a.work[i];
        __syncthreads();
        break;
      }
      case P_RSAG_PUSH: case P_RSAG_PHASED: {
        // Shard j of the whole message belongs to rank j. Each CTA handles its slice of every shard.
        long sh = (V + R - 1) / R;
        long slo = min(sh, (long)blockIdx.x * ((sh + gridDim.x - 1) / gridDim.x));
        long shi = min(sh, slo + (sh + gridDim.x - 1) / gridDim.x);
        auto off = [&](int j) { return (long)j * sh; };
        bool phased = a.pattern == P_RSAG_PHASED;
        // RS: write shard d of my input to rank d's inbox slot [r].
        if (!phased) {
          for (long i = slo + threadIdx.x; i < shi; i += blockDim.x) {
            for (int k = 1; k < R; ++k) {
              int d = (r + k) % R;
              if (off(d) + i < V) st_v4(inbox_slot(a.peer_inbox[d], V, par, r) + i, a.in[off(d) + i]);
            }
          }
          for (int k = 1; k < R; ++k) signal(a, (r + k) % R, K_RS, seq);
        } else {
          for (int k = 1; k < R; ++k) {
            int d = (r + k) % R;
            uint4* dst = inbox_slot(a.peer_inbox[d], V, par, r);
            for (long i = slo + threadIdx.x; i < shi; i += blockDim.x)
              if (off(d) + i < V) st_v4(dst + i, a.in[off(d) + i]);
            signal(a, d, K_RS, seq);
            if (k < R - 1) gpu_phase(a, ++phase);
          }
        }
        for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_RS, seq);
        // Reduce my shard.
        for (long i = slo + threadIdx.x; i < shi; i += blockDim.x) {
          if (off(r) + i >= V) continue;
          uint4 acc = a.in[off(r) + i];
          for (int k = 1; k < R; ++k) add4(acc, ld_v4(inbox_slot(a.peer_inbox[r], V, par, (r + R - k) % R) + i));
          a.out[off(r) + i] = acc;
        }
        __syncthreads();
        // AG: write my reduced shard into every peer's output-gather slot [R + r].
        if (!phased) {
          for (long i = slo + threadIdx.x; i < shi; i += blockDim.x) {
            if (off(r) + i >= V) continue;
            uint4 v = a.out[off(r) + i];
            for (int k = 1; k < R; ++k) st_v4(inbox_slot(a.peer_inbox[(r + k) % R], V, par, R + r) + i, v);
          }
          for (int k = 1; k < R; ++k) signal(a, (r + k) % R, K_AG, seq);
        } else {
          for (int k = 1; k < R; ++k) {
            int d = (r + k) % R;
            uint4* dst = inbox_slot(a.peer_inbox[d], V, par, R + r);
            for (long i = slo + threadIdx.x; i < shi; i += blockDim.x)
              if (off(r) + i < V) st_v4(dst + i, a.out[off(r) + i]);
            signal(a, d, K_AG, seq);
            if (k < R - 1) gpu_phase(a, ++phase);
          }
        }
        for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_AG, seq);
        for (int k = 1; k < R; ++k) {
          int s = (r + R - k) % R;
          const uint4* src = inbox_slot(a.peer_inbox[r], V, par, R + s);
          for (long i = slo + threadIdx.x; i < shi; i += blockDim.x)
            if (off(s) + i < V) a.out[off(s) + i] = ld_v4(src + i);
        }
        __syncthreads();
        break;
      }
      case P_RSAG_PULL: {
        long sh = (V + R - 1) / R;
        long slo = min(sh, (long)blockIdx.x * ((sh + gridDim.x - 1) / gridDim.x));
        long shi = min(sh, slo + (sh + gridDim.x - 1) / gridDim.x);
        long o = (long)r * sh;
        long e = min(V, o + shi);
        // RS: read my shard from every peer's input.
        const uint4* srcs[MAXR];
        int n = 0;
        for (int k = 1; k < R; ++k) srcs[n++] = a.peer_in[(r + k) % R] ;
        if (o + slo < e) gather_add<2>(srcs, n, o + slo, e, a.in, a.peer_red[r]);
        // Publish the reduced shard to every peer, then read theirs.
        for (int k = 1; k < R; ++k) signal(a, (r + k) % R, K_RED, seq);
        for (int k = 1; k < R; ++k) await(a, (r + R - k) % R, K_RED, seq);
        for (int k = 1; k < R; ++k) {
          int s = (r + k) % R;
          long so = (long)s * sh, se = min(V, so + shi);
          const uint4* src = a.peer_red[s];
          for (long i = so + slo + threadIdx.x; i < se; i += blockDim.x) a.out[i] = ld_v4(src + i);
        }
        for (long i = o + slo + threadIdx.x; i < e; i += blockDim.x) a.out[i] = a.peer_red[r][i];
        __syncthreads();
        break;
      }
      case P_HOST_READ: {
        const uint4* src = a.host;
        gather_add<2>(&src, 1, lo, hi, nullptr, a.work);
        __syncthreads();
        break;
      }
      case P_HOST_WRITE: {
        uint4* dst = a.host;
        scatter(&dst, 1, lo, hi, a.in);
        __syncthreads();
        if (threadIdx.x == 0) __threadfence_system();
        __syncthreads();
        break;
      }
    }
    unsigned long long te = gtimer();
    if (threadIdx.x == 0) {
      atomicMin(a.t_start + it, ts);
      atomicMax(a.t_ready + it, tr);
      atomicMax(a.t_end + it, te);
    }
    prev_end = te;
  }
}

__global__ void fill(uint4* p, long n, unsigned v) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    p[i] = make_uint4(v, v + 1, v + 2, v + 3);
}

__global__ void timer_resolution(unsigned long long* out) {
  unsigned long long a = gtimer(), b, minstep = ~0ull;
  for (int k = 0; k < 200; ++k) {
    do { b = gtimer(); } while (b == a);
    minstep = min(minstep, b - a);
    a = b;
  }
  *out = minstep;
}

// ---------------------------------------------------------------------------------------------------
struct Gpu {
  int dev;
  cudaStream_t stream;
  uint4 *in, *out, *work, *inbox, *red;
  unsigned *flags, *gpu_bar, *err;
  unsigned long long *t_start, *t_ready, *t_end;
  uint4* host;
};

static std::vector<int> parse_ints(const std::string& s) {
  std::vector<int> v;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    v.push_back(atoi(s.substr(p, q - p).c_str()));
    p = q + 1;
  }
  return v;
}
static std::vector<std::string> parse_strs(const std::string& s) {
  std::vector<std::string> v;
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(',', p);
    if (q == std::string::npos) q = s.size();
    v.push_back(s.substr(p, q - p));
    p = q + 1;
  }
  return v;
}
static double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = (size_t)std::min<double>(v.size() - 1, q * (v.size() - 1) + 0.5);
  return v[i];
}

int main(int argc, char** argv) {
  std::vector<int> gpus = {0, 1, 2, 3};
  std::vector<std::string> patterns = {"pull"};
  std::vector<int> ctas_list = {16};
  std::vector<int> kib_list = {80};
  int threads = 512, iters = 20000, sync = 1, kernel_sync = 1;
  double gap_us = 0, stall_us = 100;
  std::string dump;
  bool csv = false;
  double bg_mbps = 0;  // background host->device copy traffic per GPU (MB/s), on a separate stream
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    auto val = [&]() { return std::string(argv[++i]); };
    if (k == "--gpus") gpus = parse_ints(val());
    else if (k == "--patterns") patterns = parse_strs(val());
    else if (k == "--ctas") ctas_list = parse_ints(val());
    else if (k == "--kib") kib_list = parse_ints(val());
    else if (k == "--threads") threads = atoi(argv[++i]);
    else if (k == "--iters") iters = atoi(argv[++i]);
    else if (k == "--gap-us") gap_us = atof(argv[++i]);
    else if (k == "--stall-us") stall_us = atof(argv[++i]);
    else if (k == "--no-sync") sync = 0;
    else if (k == "--no-kernel-sync") kernel_sync = 0;
    else if (k == "--dump") dump = val();
    else if (k == "--csv") csv = true;
    else if (k == "--bg-host-mbps") bg_mbps = atof(argv[++i]);
    else { fprintf(stderr, "unknown argument %s\n", k.c_str()); return 2; }
  }
  const int R = (int)gpus.size();
  if (R < 2 || R > MAXR) { fprintf(stderr, "need 2..8 GPUs\n"); return 2; }
  int max_kib = *std::max_element(kib_list.begin(), kib_list.end());
  long max_vecs = (long)max_kib * 1024 / 16;
  std::vector<Gpu> g(R);
  uint4* host_all = nullptr;
  CK(cudaSetDevice(gpus[0]));
  CK(cudaHostAlloc(&host_all, (size_t)R * max_vecs * 16, cudaHostAllocMapped | cudaHostAllocPortable));
  memset(host_all, 1, (size_t)R * max_vecs * 16);
  for (int r = 0; r < R; ++r) {
    Gpu& x = g[r];
    x.dev = gpus[r];
    CK(cudaSetDevice(x.dev));
    for (int p = 0; p < R; ++p) {
      if (p == r) continue;
      int ok = 0;
      CK(cudaDeviceCanAccessPeer(&ok, x.dev, gpus[p]));
      if (!ok) { fprintf(stderr, "no P2P %d -> %d\n", x.dev, gpus[p]); return 1; }
      cudaError_t e = cudaDeviceEnablePeerAccess(gpus[p], 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
      cudaGetLastError();
    }
    CK(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
    CK(cudaMalloc(&x.in, max_vecs * 16));
    CK(cudaMalloc(&x.out, max_vecs * 16));
    CK(cudaMalloc(&x.work, max_vecs * 16));
    CK(cudaMalloc(&x.red, max_vecs * 16));
    CK(cudaMalloc(&x.inbox, (size_t)2 * 2 * MAXR * max_vecs * 16));
    CK(cudaMalloc(&x.flags, (size_t)KINDS * MAXR * MAXCTA * 4));
    CK(cudaMalloc(&x.gpu_bar, 4));
    CK(cudaMalloc(&x.err, 4));
    CK(cudaMalloc(&x.t_start, (size_t)iters * 8));
    CK(cudaMalloc(&x.t_ready, (size_t)iters * 8));
    CK(cudaMalloc(&x.t_end, (size_t)iters * 8));
    fill<<<64, 256>>>(x.in, max_vecs, (unsigned)(r + 1) * 16);
    x.host = host_all + (long)r * max_vecs;
    CK(cudaDeviceSynchronize());
  }
  {
    CK(cudaSetDevice(g[0].dev));
    unsigned long long* d;
    unsigned long long res = 0;
    CK(cudaMalloc(&d, 8));
    timer_resolution<<<1, 1>>>(d);
    CK(cudaMemcpy(&res, d, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(d));
    printf("# globaltimer min step %llu ns; gpus", res);
    for (int v : gpus) printf(" %d", v);
    printf("; iters %d threads %d gap %.1f us sync %d kernel_sync %d stall>%.0f us\n", iters, threads, gap_us, sync,
           kernel_sync, stall_us);
  }
  if (csv) printf("pattern,kib,ctas,rank,p50,p90,p99,p999,max,mean,xfer_p50,xfer_p99,xfer_max,n_stall,n_1ms,global_p50,global_p99,global_max,global_mean,err\n");
  std::vector<unsigned long long> hs(iters), hr(iters), he(iters);
  for (const std::string& pname : patterns) {
    int pat = -1;
    for (int p = 0; p < P_COUNT; ++p)
      if (pname == kPatternNames[p]) pat = p;
    if (pat < 0) { fprintf(stderr, "unknown pattern %s\n", pname.c_str()); return 2; }
    for (int kib : kib_list) {
      for (int ctas : ctas_list) {
        if (ctas > MAXCTA) { fprintf(stderr, "ctas > %d\n", MAXCTA); return 2; }
        long vecs = (long)kib * 1024 / 16;
        for (int r = 0; r < R; ++r) {
          Gpu& x = g[r];
          CK(cudaSetDevice(x.dev));
          CK(cudaMemset(x.flags, 0, (size_t)KINDS * MAXR * MAXCTA * 4));
          CK(cudaMemset(x.gpu_bar, 0, 4));
          CK(cudaMemset(x.err, 0, 4));
          CK(cudaMemset(x.t_start, 0xff, (size_t)iters * 8));
          CK(cudaMemset(x.t_ready, 0, (size_t)iters * 8));
          CK(cudaMemset(x.t_end, 0, (size_t)iters * 8));
          CK(cudaMemset(x.out, 0, vecs * 16));
          CK(cudaDeviceSynchronize());
        }
        {
          timespec ts_;
          clock_gettime(CLOCK_REALTIME, &ts_);
          printf("# launch %s host_ns %lld\n", pname.c_str(), (long long)ts_.tv_sec * 1000000000LL + ts_.tv_nsec);
          fflush(stdout);
        }
        for (int r = 0; r < R; ++r) {
          Gpu& x = g[r];
          CK(cudaSetDevice(x.dev));
          Args a{};
          a.rank = r; a.nranks = R; a.pattern = pat; a.iters = iters; a.sync = sync; a.kernel_sync = kernel_sync;
          a.vecs = vecs;
          a.gap_ns = (unsigned long long)(gap_us * 1000);
          a.timeout_ns = 5ull * 1000 * 1000 * 1000;
          a.in = x.in; a.out = x.out; a.work = x.work;
          for (int p = 0; p < R; ++p) {
            a.peer_in[p] = g[p].in;
            a.peer_red[p] = g[p].red;
            a.peer_inbox[p] = g[p].inbox;
            a.peer_flags[p] = g[p].flags;
          }
          a.gpu_bar = x.gpu_bar;
          a.host = x.host;
          a.t_start = x.t_start; a.t_ready = x.t_ready; a.t_end = x.t_end;
          a.err = x.err;
          // inbox strides use the run's vecs; inbox capacity was sized for the maximum.
          run<<<ctas, threads, 0, x.stream>>>(a);
          CK(cudaGetLastError());
        }
        std::atomic<bool> bg_stop{false};
        std::thread bg;
        if (bg_mbps > 0) {
          bg = std::thread([&]() {
            const size_t chunk = 256 << 10;
            std::vector<cudaStream_t> st(R);
            std::vector<void*> dbuf(R);
            void* hbuf = nullptr;
            CK(cudaSetDevice(g[0].dev));
            CK(cudaHostAlloc(&hbuf, chunk, cudaHostAllocPortable));
            for (int r = 0; r < R; ++r) {
              CK(cudaSetDevice(g[r].dev));
              CK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
              CK(cudaMalloc(&dbuf[r], chunk));
            }
            auto period = std::chrono::microseconds((long)(chunk / (bg_mbps * 1e6) * 1e6));
            auto next = std::chrono::steady_clock::now();
            while (!bg_stop.load()) {
              for (int r = 0; r < R; ++r) {
                CK(cudaSetDevice(g[r].dev));
                CK(cudaMemcpyAsync(dbuf[r], hbuf, chunk, cudaMemcpyHostToDevice, st[r]));
              }
              next += period;
              std::this_thread::sleep_until(next);
            }
            for (int r = 0; r < R; ++r) { CK(cudaSetDevice(g[r].dev)); CK(cudaStreamSynchronize(st[r])); }
          });
        }
        for (int r = 0; r < R; ++r) { CK(cudaSetDevice(g[r].dev)); CK(cudaStreamSynchronize(g[r].stream)); }
        if (bg.joinable()) { bg_stop = true; bg.join(); }
        // Collect.
        std::vector<std::vector<unsigned long long>> S(R), Rd(R), E(R);
        unsigned errs = 0;
        bool correct = true;
        for (int r = 0; r < R; ++r) {
          Gpu& x = g[r];
          CK(cudaSetDevice(x.dev));
          S[r].resize(iters); Rd[r].resize(iters); E[r].resize(iters);
          CK(cudaMemcpy(S[r].data(), x.t_start, (size_t)iters * 8, cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(Rd[r].data(), x.t_ready, (size_t)iters * 8, cudaMemcpyDeviceToHost));
          CK(cudaMemcpy(E[r].data(), x.t_end, (size_t)iters * 8, cudaMemcpyDeviceToHost));
          unsigned e = 0;
          CK(cudaMemcpy(&e, x.err, 4, cudaMemcpyDeviceToHost));
          errs |= e;
          if (pat == P_PULL || pat == P_PUSH || pat == P_PUSH_PHASED || pat == P_RING || pat == P_RSAG_PUSH ||
              pat == P_RSAG_PHASED || pat == P_RSAG_PULL) {
            std::vector<uint4> o(vecs);
            CK(cudaMemcpy(o.data(), x.out, vecs * 16, cudaMemcpyDeviceToHost));
            unsigned want = 0;
            for (int p = 0; p < R; ++p) want += (unsigned)(p + 1) * 16;
            for (long i = 0; i < vecs; ++i)
              if (o[i].x != want || o[i].w != want + 3 * R) { correct = false; break; }
          }
        }
        // Align GPU clocks: barrier exits are simultaneous to within the flag latency.
        std::vector<long long> off(R, 0);
        if (sync) {
          for (int r = 1; r < R; ++r) {
            std::vector<double> d;
            for (int it = 0; it < iters; ++it) d.push_back((double)(long long)(Rd[r][it] - Rd[0][it]));
            off[r] = (long long)pct(d, 0.5);
          }
        }
        for (int r = 0; r < R; ++r)
          for (int it = 0; it < iters; ++it) { S[r][it] -= off[r]; Rd[r][it] -= off[r]; E[r][it] -= off[r]; }
        // Per-rank statistics.
        std::vector<double> gl;
        double gsum = 0;
        for (int it = kWarm; it < iters; ++it) {
          unsigned long long s0 = ~0ull, e1 = 0;
          for (int r = 0; r < R; ++r) { s0 = std::min(s0, S[r][it]); e1 = std::max(e1, E[r][it]); }
          gl.push_back((e1 - s0) / 1000.0);
          gsum += (e1 - s0) / 1000.0;
        }
        unsigned long long t_first = ~0ull;
        for (int r = 0; r < R; ++r) t_first = std::min(t_first, S[r][0]);
        double gp50 = pct(gl, 0.5), gp99 = pct(gl, 0.99), gmax = pct(gl, 1.0);
        printf("== %-12s %4d KiB ctas %3d  global p50 %8.2f p99 %8.2f max %9.1f mean %8.2f us  %s%s\n",
               pname.c_str(), kib, ctas, gp50, gp99, gmax, gsum / (iters - kWarm), errs ? "TIMEOUT " : "",
               correct ? "" : "WRONG");
        for (int r = 0; r < R; ++r) {
          std::vector<double> tot, xf, br;
          double sum = 0;
          int nst = 0, n1 = 0;
          for (int it = kWarm; it < iters; ++it) {
            double t = (E[r][it] - S[r][it]) / 1000.0, x = (E[r][it] - Rd[r][it]) / 1000.0;
            tot.push_back(t); xf.push_back(x); br.push_back((Rd[r][it] - S[r][it]) / 1000.0); sum += t;
            if (x > stall_us) ++nst;
            if (x > 1000) ++n1;
          }
          if (csv)
            printf("%s,%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%.1f,%.2f,%.2f,%.2f,%.1f,%d,%d,%.2f,%.2f,%.1f,%.2f,%u\n",
                   pname.c_str(), kib, ctas, gpus[r], pct(tot, .5), pct(tot, .9), pct(tot, .99), pct(tot, .999),
                   pct(tot, 1), sum / (iters - kWarm), pct(xf, .5), pct(xf, .99), pct(xf, 1), nst, n1, gp50, gp99, gmax,
                   gsum / (iters - kWarm), errs);
          else
            printf("   gpu %2d  total p50 %7.2f p99 %8.2f p99.9 %8.1f max %8.1f mean %7.2f | bar p50 %6.2f p99 %7.2f max %8.1f | xfer p50 %7.2f p99 %8.2f max %8.1f | >%.0fus %5d >1ms %4d\n",
                   gpus[r], pct(tot, .5), pct(tot, .99), pct(tot, .999), pct(tot, 1), sum / (iters - kWarm),
                   pct(br, .5), pct(br, .99), pct(br, 1), pct(xf, .5), pct(xf, .99), pct(xf, 1), stall_us, nst, n1);
        }
        // Stall events: transfer phase longer than the threshold, per rank, with absolute time.
        if (!csv) {
          int shown = 0;
          double last = -1;
          std::vector<double> gaps;
          for (int it = kWarm; it < iters; ++it) {
            for (int r = 0; r < R; ++r) {
              double x = (E[r][it] - Rd[r][it]) / 1000.0;
              if (x > stall_us) {
                double at = (Rd[r][it] - t_first) / 1e6;
                if (last >= 0 && at - last > 1.0) gaps.push_back(at - last);
                if (last < 0 || at - last > 1.0) {
                  if (shown < 12) {
                    printf("     stall it %6d t %9.3f ms gpu %2d xfer %8.1f us", it, at, gpus[r], x);
                    for (int q = 0; q < R; ++q) printf(" [g%d %.1f]", gpus[q], (E[q][it] - Rd[q][it]) / 1000.0);
                    printf("\n");
                  }
                  ++shown;
                }
                last = at;
              }
            }
          }
          if (shown) {
            double span = (E[0][iters - 1] - t_first) / 1e6;
            printf("     stall episodes %d over %.1f ms", shown, span);
            if (!gaps.empty()) printf("; gap between episodes p50 %.2f ms min %.2f max %.2f", pct(gaps, .5), pct(gaps, 0), pct(gaps, 1));
            printf("\n");
          }
        }
        if (!dump.empty()) {
          std::string fn = dump + "-" + pname + "-" + std::to_string(kib) + "k-c" + std::to_string(ctas) + ".bin";
          FILE* f = fopen(fn.c_str(), "wb");
          int hdr[4] = {R, iters, kib, ctas};
          fwrite(hdr, 4, 4, f);
          fwrite(gpus.data(), 4, R, f);
          for (int r = 0; r < R; ++r) {
            fwrite(S[r].data(), 8, iters, f);
            fwrite(Rd[r].data(), 8, iters, f);
            fwrite(E[r].data(), 8, iters, f);
          }
          fclose(f);
        }
        fflush(stdout);
        if (errs) { fprintf(stderr, "timeout in %s; stopping\n", pname.c_str()); return 3; }
      }
    }
  }
  return 0;
}
