// SPDX-License-Identifier: Apache-2.0
#pragma once

// Exact distributed cutoff for World (4, 8, or 16) sorted lists of 2048 entries.
// The low word uniquely identifies a rank-major input position for cutoff ties.
#include <cub/block/block_radix_sort.cuh>

namespace flashinfer::interleaved_topk::sampled {
using Key = unsigned long long;
constexpr int items = 2048, block_items = 32, samples = items / block_items, threads = 256;
constexpr int stage_items = items + samples;
using LocalSort = cub::BlockRadixSort<Key, threads, 8>;
using BoundarySort = cub::BlockRadixSort<Key, threads, 4>;

template <int Ranks>
struct Storage {
  static_assert(Ranks == 4 || Ranks == 8 || Ranks == 16);
  union {
    typename LocalSort::TempStorage local;
    typename BoundarySort::TempStorage boundaries;
  } sort;
  Key keys[1024], upper, lower, cutoff;
  int first[Ranks], stop[Ranks], offset[Ranks], peer_for_block[2 * Ranks], local_block[2 * Ranks];
};

__device__ __forceinline__ Key tie_key(unsigned ordered_score, int owner, int local_index) {
  return (Key(ordered_score) << 32) | unsigned(~(owner * items + local_index));
}
__device__ __forceinline__ Key read_peer(const Key* p) {
  Key value;
  asm volatile("ld.global.cg.u64 %0,[%1];" : "=l"(value) : "l"(p) : "memory");
  return value;
}

template <int Ranks>
__device__ void publish_sorted(Storage<Ranks>& s, Key* stage, Key (&keys)[8]) {
  LocalSort(s.sort.local).SortDescending(keys);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) stage[8 * threadIdx.x + i] = keys[i];
  if ((threadIdx.x & 3) == 3) stage[items + threadIdx.x / 4] = keys[7];
}

template <int Ranks, class Pointers>
__device__ Key select(Storage<Ranks>& s, const Pointers& peers, long long bank_offset, int row) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int j = tid; j < 1024; j += threads) {
    Key value = 0;
    if (j < Ranks * samples) {
      const int peer = j / samples;
      const Key* stage =
          reinterpret_cast<const Key*>(peers.stage[peer] + bank_offset) + row * stage_items;
      value = read_peer(stage + items + (j % samples));
    }
    s.keys[j] = value;
  }
  __syncthreads();
  Key ordered[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ordered[i] = s.keys[tid * 4 + i];
  BoundarySort(s.sort.boundaries).SortDescending(ordered);
  // At least k entries are >= the 64th block minimum (index 63).
  // Strictly fewer than k are >= the (64 - Ranks)th block minimum (index 63 - Ranks),
  // since each rank adds at most 31 entries from its first partial block.
  constexpr int upper_idx = samples - Ranks - 1;
  constexpr int lower_idx = samples - 1;
  if (tid == upper_idx / 4) s.upper = ordered[upper_idx % 4];
  if (tid == lower_idx / 4) s.lower = ordered[lower_idx % 4];
  __syncthreads();
  if (tid < Ranks) {
    int first = 0, stop = 0;
    for (int i = 0; i < samples; ++i) {
      Key v = s.keys[tid * samples + i];
      first += v >= s.upper;
      stop += v >= s.lower;
    }
    s.first[tid] = first;
    s.stop[tid] = min(stop + 1, samples);
  }
  __syncthreads();
  if (tid == 0) {
    int offset = 0;
    for (int p = 0; p < Ranks; ++p) {
      s.offset[p] = offset;
      offset += s.stop[p] - s.first[p];
    }
    for (int i = offset; i < 2 * Ranks; ++i) s.peer_for_block[i] = -1;
  }
  __syncthreads();
  if (tid < Ranks) {
    int offset = s.offset[tid];
    for (int i = s.first[tid]; i < s.stop[tid]; ++i, ++offset) {
      s.peer_for_block[offset] = tid;
      s.local_block[offset] = i;
    }
  }
  __syncthreads();
  // The (64 - Ranks) complete upper blocks account for (64 - Ranks) * 32 winners.
  // The remaining Ranks * 32 winners are in at most 2 * Ranks blocks.
  for (int j = tid; j < 1024; j += threads) {
    Key value = 0;
    if (j < 2 * Ranks * block_items) {
      const int block = j / block_items, peer = s.peer_for_block[block];
      if (peer >= 0) {
        const Key* stage =
            reinterpret_cast<const Key*>(peers.stage[peer] + bank_offset) + row * stage_items;
        value = read_peer(stage + s.local_block[block] * block_items + (j % block_items));
      }
    }
    s.keys[j] = value;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) ordered[i] = s.keys[tid * 4 + i];
  BoundarySort(s.sort.boundaries).SortDescending(ordered);
  constexpr int cutoff_idx = Ranks * block_items - 1;
  if (tid == cutoff_idx / 4) s.cutoff = ordered[cutoff_idx % 4];
  __syncthreads();
  return s.cutoff;
}
}  // namespace flashinfer::interleaved_topk::sampled
