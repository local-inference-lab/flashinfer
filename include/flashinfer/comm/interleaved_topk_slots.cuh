// SPDX-License-Identifier: Apache-2.0
#pragma once

// Select the attention records consumed by one interleaved DCP shard.
// Input is the captured rank-major candidate contract: each rank contributes
// up to 2048 (FP32 score bits, global position) pairs and pads with (-inf,-1).
#include <cuda_runtime.h>

#include <cstdint>
#include <cub/block/block_scan.cuh>

#include "interleaved_topk_cutoff.cuh"

namespace flashinfer::interleaved_topk {
constexpr int world = 16, k = 2048, threads = 256;
constexpr unsigned pad = 0x007fffffu;  // ordered(-infinity)
struct Peers {
  unsigned char* stage[world];
  unsigned* flags[world];
};
__device__ __forceinline__ unsigned key(unsigned raw) {
  return (raw & 0x80000000u) ? ~raw : (raw ^ 0x80000000u);
}
__device__ __forceinline__ unsigned load_gpu(unsigned* p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void publish(unsigned long long* p, unsigned long long value) {
  asm volatile("st.release.sys.global.u64 [%0],%1;" ::"l"(p), "l"(value) : "memory");
}
__device__ __forceinline__ unsigned long long receive(unsigned long long* p, unsigned gen) {
  unsigned long long v;
  do {
    asm volatile("ld.acquire.sys.global.u64 %0,[%1];" : "=l"(v) : "l"(p) : "memory");
  } while (unsigned(v >> 32) != gen);
  return v;
}
__device__ __forceinline__ unsigned long long* mailbox(unsigned* base, int bank, int row,
                                                       int sender) {
  // The common DCP signal layout uses word zero in each 32-word record.
  // The graph epoch and block arrival counter use words 1 and 2 of record 0.
  return reinterpret_cast<unsigned long long*>(base + 1024 +
                                               ((bank * 64 + row) * 16 + sender) * 32 + 8);
}
__device__ __forceinline__ int visible(long long total, int owner) {
  return int(min((long long)k, (total + 15 - owner) / 16));
}

__device__ __forceinline__ uint2 local_pair(const int* ids, const unsigned* scores, int j) {
  const int i = ids[j];
  return i < 0 ? make_uint2(0xff800000u, 0xffffffffu)
               : make_uint2(scores[j], unsigned(i));
}

__global__ void local_candidate_slots(const int* local_ids, const unsigned* local_scores,
                                      const long long* positions, const int* tables, int* owners,
                                      int* slots, int* counts, Peers peers, long long stride,
                                      int rank, long long table_stride, long long table_columns,
                                      int cache_pages) {
  using Scan = cub::BlockScan<int, threads>;
  __shared__ typename Scan::TempStorage scan;
  __shared__ unsigned gen, graph_epoch;
  __shared__ int above, below, local_ties, remaining, preceding, nout, staging;
  __shared__ unsigned cutoff, peer_counts[16];
  __shared__ sampled::Storage boundary;
  sampled::Key threshold = 0;
  int tid = threadIdx.x, row = blockIdx.x;
  const long long position = positions[row];
  // Each peer supplies at most k entries, including for an int64 absolute position.
  const long long total = max(0ll, min(position, static_cast<long long>(world * k)) + 1);
  const int local_count = visible(total, rank);
  const bool short_row = total <= k;
  unsigned* header = peers.flags[rank];
  if (tid == 0) {
    graph_epoch = load_gpu(header + 1025);
    above = below = local_ties = nout = preceding = 0;
    // Short invocations never borrow a payload bank. Leave its parity
    // unchanged so two regular bank users always have an intervening
    // all-rank arrival before either can overwrite an earlier payload.
    staging = 0;
    for (unsigned r = 0; r < gridDim.x; ++r) staging |= positions[r] >= k;
    // Short summaries and payload exchanges each advance their own sequence.
    // A channel can go arbitrarily long without using the other protocol.
    // Both sequences retire every mailbox before a generation can repeat.
    // Wrap at a multiple of 128 to retain ring and bank alignment and reserve
    // zero for an empty mailbox. The four rows fit in each counter group.
    auto& sequence = header[1024 + (63 * 16 + 15) * 32 + 4 + row + (staging ? 4 : 0)];
    gen = sequence == 0xffffff80u ? 1u : sequence + 1u;
    sequence = gen;
    cutoff = 0;
    remaining = k;
  }
  __syncthreads();
  const long long bank_offset = (graph_epoch & 1u) * stride;
  sampled::Key* local_stage =
      reinterpret_cast<sampled::Key*>(peers.stage[rank] + bank_offset) + row * sampled::stage_items;
  int hi = 0, lo = 0, special = 0;
  for (int j = tid; j < local_count; j += threads) {
    const uint2 p = local_pair(local_ids, local_scores, row * k + j);
    const unsigned v = int(p.y) < 0 ? pad : key(p.x);
    hi += (v > pad);
    lo += (v < pad);
    special += (int(p.y) >= 0 && v == pad);
  }
  if (!short_row) {
    sampled::Key vals[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      int j = tid * 8 + i;
      uint2 p = j < local_count ? local_pair(local_ids, local_scores, row * k + j)
                                : make_uint2(0xff800000u, 0xffffffffu);
      vals[i] = sampled::tie_key(int(p.y) < 0 ? pad : key(p.x), rank, j);
    }
    sampled::publish_sorted(boundary, local_stage, vals);
  }
  if (hi) atomicAdd(&above, hi);
  if (lo) atomicAdd(&below, lo);
  if (special) atomicAdd(&local_ties, special);
  for (int j = tid; j < k; j += threads) {
    owners[row * k + j] = -1;
    slots[row * k + j] = -1;
  }
  __syncthreads();
  const unsigned summary = unsigned(above) | (unsigned(below) << 16);
  if (!staging) {
    // Keep 64 generations of local metadata in the channel's existing
    // signal padding. A rendezvous before each ring wrap retires every
    // previous reader. Ordinary finite rows can return without waiting.
    const unsigned seq = gen - 1;
    if (!(seq & 63u) && tid < world) {
      int bank = (seq >> 6) & 1u;
      auto wrap = [&](unsigned* base, int sender) {
        return reinterpret_cast<unsigned long long*>(base + 1024 +
                                                     ((bank * 64 + row) * 16 + sender) * 32 + 16);
      };
      publish(wrap(peers.flags[tid], rank), static_cast<unsigned long long>(gen) << 32);
      receive(wrap(header, tid), gen);
    }
    __syncthreads();
    const int index = 1024 + ((seq & 63u) * 16 + row) * 32 + 20;
    if (!tid)
      publish(reinterpret_cast<unsigned long long*>(header + index),
              (static_cast<unsigned long long>(gen) << 32) | summary);
    __syncthreads();
    if (local_ties && tid < world)
      peer_counts[tid] =
          unsigned(receive(reinterpret_cast<unsigned long long*>(peers.flags[tid] + index), gen));
  } else if (tid < world) {
    publish(mailbox(peers.flags[tid], gen & 1, row, rank),
            (static_cast<unsigned long long>(gen) << 32) | summary);
    peer_counts[tid] = unsigned(receive(mailbox(header, gen & 1, row, tid), gen));
  }
  __syncthreads();
  if (short_row) {
    // With at most k valid candidates, every key above padding survives.
    // The unused slots go to the first padding-key entries in input order,
    // including invalid entries, exactly as in the full-width stable fold.
    if (tid == 0) {
      cutoff = pad;
      preceding = rank * k;
      if (!staging && !local_ties)
        remaining = 0;
      else
        for (int p = 0; p < world; ++p) {
          int greater = int(peer_counts[p] & 65535u), lower = int(peer_counts[p] >> 16);
          remaining -= greater;
          if (p < rank) preceding -= greater + lower;
        }
    }
  } else {
    threshold = sampled::select(boundary, peers, bank_offset, row);
  }
  __syncthreads();
  // Keep the original local indexer's order, including its cutoff ties.
  // Page filtering follows score selection, so an invalid page still
  // occupies its score rank in the cutoff calculation.
  for (int j0 = 0; j0 < local_count; j0 += threads) {
    int j = j0 + tid;
    uint2 pair = j < local_count ? local_pair(local_ids, local_scores, row * k + j)
                                 : make_uint2(0xff800000u, 0xffffffffu);
    unsigned v = int(pair.y) < 0 ? pad : key(pair.x);
    int ties_before, ties;
    Scan(scan).ExclusiveSum(int(j < local_count && v == cutoff), ties_before, ties);
    __syncthreads();
    bool chosen = short_row ? (v > cutoff || (v == cutoff && preceding + ties_before < remaining))
                            : sampled::tie_key(v, rank, j) >= threshold;
    const int local = int(pair.y);
    const long long token = static_cast<long long>(local) * world + rank;
    int mapped = -1;
    if (j < local_count && chosen && local >= 0 && token <= position) {
      const int col = local / 128;
      if (col < table_columns) {
        int page = tables[row * table_stride + col];
        if (page >= 0 && page < cache_pages) mapped = page * 128 + local % 128;
      }
    }
    int prior, count;
    Scan(scan).ExclusiveSum(int(mapped >= 0), prior, count);
    __syncthreads();
    if (mapped >= 0) {
      slots[row * k + nout + prior] = mapped;
      owners[row * k + nout + prior] = rank;
    }
    __syncthreads();
    if (tid == 0) {
      nout += count;
      preceding += ties;
    }
    __syncthreads();
  }
  if (tid == 0) {
    counts[row] = nout;
    if (staging && atomicAdd(header + 1026, 1u) + 1 == gridDim.x) {
      atomicExch(header + 1026, 0u);
      atomicAdd(header + 1025, 1u);
    }
  }
}

}  // namespace flashinfer::interleaved_topk
