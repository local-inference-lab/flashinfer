// SPDX-License-Identifier: Apache-2.0
#pragma once

// Exact distributed cutoff for sixteen sorted lists of 2048 entries.
// The low word uniquely identifies a rank-major input position for cutoff ties.
#include <cub/block/block_radix_sort.cuh>

namespace flashinfer::interleaved_topk::sampled {
using Key = unsigned long long;
constexpr int ranks = 16, items = 2048, block_items = 32, samples = items / block_items,
              threads = 256;
constexpr int stage_items = items + samples;
using LocalSort = cub::BlockRadixSort<Key, threads, 8>;
using BoundarySort = cub::BlockRadixSort<Key, threads, 4>;

struct Storage {
  union {
    typename LocalSort::TempStorage local;
    typename BoundarySort::TempStorage boundaries;
  } sort;
  Key keys[ranks * samples], upper, lower, cutoff;
  int first[ranks], stop[ranks], offset[ranks], peer_for_block[2 * ranks], local_block[2 * ranks];
};

__device__ __forceinline__ Key tie_key(unsigned ordered_score, int owner, int local_index) {
  return (Key(ordered_score) << 32) | unsigned(~(owner * items + local_index));
}
__device__ __forceinline__ Key read_peer(const Key* p) {
  Key value;
  asm volatile("ld.global.cg.u64 %0,[%1];" : "=l"(value) : "l"(p) : "memory");
  return value;
}

__device__ void publish_sorted(Storage& s, Key* stage, Key (&keys)[8]) {
  LocalSort(s.sort.local).SortDescending(keys);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) stage[8 * threadIdx.x + i] = keys[i];
  if ((threadIdx.x & 3) == 3) stage[items + threadIdx.x / 4] = keys[7];
}

template <class Pointers>
__device__ Key select(Storage& s, const Pointers& peers, long long bank_offset, int row) {
  const int tid = threadIdx.x;
#pragma unroll
  for (int j = tid; j < ranks * samples; j += threads) {
    const int peer = j / samples;
    const Key* stage =
        reinterpret_cast<const Key*>(peers.stage[peer] + bank_offset) + row * stage_items;
    s.keys[j] = read_peer(stage + items + (j % samples));
  }
  __syncthreads();
  Key ordered[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) ordered[i] = s.keys[tid * 4 + i];
  BoundarySort(s.sort.boundaries).SortDescending(ordered);
  // At least k entries are >= the 64th block minimum. Strictly fewer
  // than k are >= the 48th, since each rank adds at most 31 more entries.
  if (tid == 11) s.upper = ordered[3];
  if (tid == 15) s.lower = ordered[3];
  __syncthreads();
  if (tid < ranks) {
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
    for (int p = 0; p < ranks; ++p) {
      s.offset[p] = offset;
      offset += s.stop[p] - s.first[p];
    }
    for (int i = offset; i < 2 * ranks; ++i) s.peer_for_block[i] = -1;
  }
  __syncthreads();
  if (tid < ranks) {
    int offset = s.offset[tid];
    for (int i = s.first[tid]; i < s.stop[tid]; ++i, ++offset) {
      s.peer_for_block[offset] = tid;
      s.local_block[offset] = i;
    }
  }
  __syncthreads();
  // The 48 complete upper blocks account for exactly 1536 winners.
  // The remaining 512 are in at most 32 blocks. A unique input tie key
  // handles equal scores and padding with the ordinary global fold's order.
  for (int j = tid; j < 2 * ranks * block_items; j += threads) {
    const int block = j / block_items, peer = s.peer_for_block[block];
    Key value = 0;
    if (peer >= 0) {
      const Key* stage =
          reinterpret_cast<const Key*>(peers.stage[peer] + bank_offset) + row * stage_items;
      value = read_peer(stage + s.local_block[block] * block_items + (j % block_items));
    }
    s.keys[j] = value;
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; ++i) ordered[i] = s.keys[tid * 4 + i];
  BoundarySort(s.sort.boundaries).SortDescending(ordered);
  if (tid == 127) s.cutoff = ordered[3];
  __syncthreads();
  return s.cutoff;
}
}  // namespace flashinfer::interleaved_topk::sampled
