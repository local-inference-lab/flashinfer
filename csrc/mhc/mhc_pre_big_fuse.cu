/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <initializer_list>

#include "tvm_ffi_utils.h"

namespace flashinfer::mhc {
namespace {

static constexpr int kHc4 = 4;
static constexpr int kMixHc4 = kHc4 * (2 + kHc4);
static constexpr int kWarpThreads = 32;
static constexpr int kVecWidth = 8;

struct Bf16x8 {
  uint4 raw;
};

__device__ inline Bf16x8 load_bf16x8(const __nv_bfloat16* ptr) {
  Bf16x8 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(v.raw.x), "=r"(v.raw.y), "=r"(v.raw.z), "=r"(v.raw.w)
               : "l"(ptr));
  return v;
}

__device__ inline void store_bf16x8(__nv_bfloat16* ptr, const Bf16x8& v) {
  asm volatile("st.global.v4.u32 [%4], {%0, %1, %2, %3};\n"
               :
               : "r"(v.raw.x), "r"(v.raw.y), "r"(v.raw.z), "r"(v.raw.w), "l"(ptr));
}

__device__ inline __nv_bfloat162 bf16x8_get_pair(const Bf16x8& v, int i) {
  const uint32_t bits = i == 0 ? v.raw.x : (i == 1 ? v.raw.y : (i == 2 ? v.raw.z : v.raw.w));
  __nv_bfloat162 out;
  *reinterpret_cast<uint32_t*>(&out) = bits;
  return out;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_sum(float v, float* shared) {
  constexpr int kWarps = BLOCK_SIZE / kWarpThreads;
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const int warp = threadIdx.x / kWarpThreads;

  v = warp_sum(v);
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();

  v = threadIdx.x < kWarps ? shared[lane] : 0.0f;
  if (warp == 0) {
    v = warp_sum(v);
  }
  if (threadIdx.x == 0) {
    shared[0] = v;
  }
  __syncthreads();
  return shared[0];
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float residual_square_sum_vec8(const __nv_bfloat16* residual, int K) {
  float sum = 0.0f;
  for (int idx = threadIdx.x * kVecWidth; idx < K; idx += BLOCK_SIZE * kVecWidth) {
    const Bf16x8 raw = load_bf16x8(residual + idx);
#pragma unroll
    for (int i = 0; i < kVecWidth / 2; ++i) {
      const float2 f = __bfloat1622float2(bf16x8_get_pair(raw, i));
      sum = fmaf(f.x, f.x, sum);
      sum = fmaf(f.y, f.y, sum);
    }
  }
  return sum;
}

template <bool WRITE_PRE = true>
__device__ __forceinline__ void write_token_metadata(
    const float* y_local, float rstd, const float* mhc_scale, const float* mhc_base,
    float* post_mix, float* comb_mix, float* pre_mix, float mhc_pre_eps, float mhc_sinkhorn_eps,
    float mhc_post_mult_value, int sinkhorn_repeat) {
  const int lane = threadIdx.x & (kWarpThreads - 1);
  float cm[kHc4];

  const float scale_pre = mhc_scale[0];
  const float scale_post = mhc_scale[1];
  const float scale_comb = mhc_scale[2];

  float v = y_local[lane] * rstd * scale_pre + mhc_base[lane];
  if constexpr (WRITE_PRE) {
    pre_mix[lane] = 1.0f / (1.0f + expf(-v)) + mhc_pre_eps;
  }

  v = y_local[kHc4 + lane] * rstd * scale_post + mhc_base[kHc4 + lane];
  post_mix[lane] = 1.0f / (1.0f + expf(-v)) * mhc_post_mult_value;

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = y_local[2 * kHc4 + lane * kHc4 + k] * rstd * scale_comb +
            mhc_base[2 * kHc4 + lane * kHc4 + k];
  }

  constexpr unsigned kLaneMask = (1u << kHc4) - 1;
  float row_max = fmaxf(fmaxf(cm[0], cm[1]), fmaxf(cm[2], cm[3]));
#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = expf(cm[k] - row_max);
  }
  float row_sum = cm[0] + cm[1] + cm[2] + cm[3];
#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    cm[k] = cm[k] / row_sum + mhc_sinkhorn_eps;
  }

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    float col_sum = cm[k];
    col_sum += __shfl_xor_sync(kLaneMask, col_sum, 1);
    col_sum += __shfl_xor_sync(kLaneMask, col_sum, 2);
    cm[k] /= (col_sum + mhc_sinkhorn_eps);
  }

  for (int it = 1; it < sinkhorn_repeat; ++it) {
    row_sum = cm[0] + cm[1] + cm[2] + cm[3] + mhc_sinkhorn_eps;
#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      cm[k] /= row_sum;
    }

#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      float col_sum = cm[k];
      col_sum += __shfl_xor_sync(kLaneMask, col_sum, 1);
      col_sum += __shfl_xor_sync(kLaneMask, col_sum, 2);
      cm[k] /= (col_sum + mhc_sinkhorn_eps);
    }
  }

#pragma unroll
  for (int k = 0; k < kHc4; ++k) {
    comb_mix[lane * kHc4 + k] = cm[k];
  }
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void write_layer_input(const __nv_bfloat16* residual,
                                                  const float* pre_mix, __nv_bfloat16* layer_input,
                                                  int H) {
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpThreads;
  if (warp_id == 0) {
    return;
  }

  const int p2_tid = tid - kWarpThreads;
  constexpr int p2_threads = BLOCK_SIZE - kWarpThreads;
  for (int h = p2_tid * kVecWidth; h < H; h += p2_threads * kVecWidth) {
    float acc[kVecWidth] = {};

#pragma unroll
    for (int j = 0; j < kHc4; ++j) {
      const Bf16x8 raw = load_bf16x8(residual + j * H + h);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const float2 f = __bfloat1622float2(bf16x8_get_pair(raw, v));
        acc[2 * v + 0] += pre_mix[j] * f.x;
        acc[2 * v + 1] += pre_mix[j] * f.y;
      }
    }

    Bf16x8 out;
    __nv_bfloat162* pairs = reinterpret_cast<__nv_bfloat162*>(&out.raw);
#pragma unroll
    for (int v = 0; v < kVecWidth / 2; ++v) {
      pairs[v] = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
    }
    store_bf16x8(layer_input + h, out);
  }
}

template <int NUM_SPLITS, int BLOCK_SIZE, bool COMPUTE_SQRSUM>
__launch_bounds__(BLOCK_SIZE) __global__ void mhc_pre_big_fuse_kernel(
    const float* __restrict__ dot_mix, const float* __restrict__ sqrsum,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base, float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ layer_input, int total_tokens, int K, int H, float rms_eps,
    float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value, int sinkhorn_repeat) {
  static_assert(NUM_SPLITS == 1 || !COMPUTE_SQRSUM,
                "internal sqrsum path only supports NUM_SPLITS=1");

  const int token = blockIdx.x;
  if (token >= total_tokens) {
    return;
  }

  const int warp_id = threadIdx.x / kWarpThreads;
  const int lane = threadIdx.x & (kWarpThreads - 1);
  const __nv_bfloat16* residual_token = residual + static_cast<long long>(token) * kHc4 * H;

  __shared__ float pre_mix[kHc4];
  __shared__ float sq_partials[BLOCK_SIZE / kWarpThreads];

  float sq_total = 0.0f;
  if constexpr (COMPUTE_SQRSUM) {
    const float local_sq = residual_square_sum_vec8<BLOCK_SIZE>(residual_token, kHc4 * H);
    sq_total = block_sum<BLOCK_SIZE>(local_sq, sq_partials);
  }

  if (warp_id == 0 && lane < kHc4) {
    float y_local[kMixHc4];
    if constexpr (NUM_SPLITS == 1) {
      if constexpr (!COMPUTE_SQRSUM) {
        sq_total = sqrsum[token];
      }
      const float* y_row = dot_mix + static_cast<long long>(token) * kMixHc4;
#pragma unroll
      for (int c = 0; c < kMixHc4; ++c) {
        y_local[c] = y_row[c];
      }
    } else {
      sq_total = 0.0f;
#pragma unroll
      for (int c = 0; c < kMixHc4; ++c) {
        y_local[c] = 0.0f;
      }
#pragma unroll
      for (int s = 0; s < NUM_SPLITS; ++s) {
        sq_total += sqrsum[s * total_tokens + token];
        const float* y_row = dot_mix + (static_cast<long long>(s) * total_tokens + token) * kMixHc4;
#pragma unroll
        for (int c = 0; c < kMixHc4; ++c) {
          y_local[c] += y_row[c];
        }
      }
    }

    const float rstd = rsqrtf(sq_total / static_cast<float>(K) + rms_eps);
    write_token_metadata(y_local, rstd, mhc_scale, mhc_base, post_mix + token * kHc4,
                         comb_mix + token * kHc4 * kHc4, pre_mix, mhc_pre_eps, mhc_sinkhorn_eps,
                         mhc_post_mult_value, sinkhorn_repeat);
  }

  __syncthreads();
  write_layer_input<BLOCK_SIZE>(residual_token, pre_mix,
                                layer_input + static_cast<long long>(token) * H, H);
}

// Warp zero publishes pre coefficients, then computes Sinkhorn while the other
// warps collapse and normalize the residual. Worker-only barriers must exclude
// warp zero, which can finish before the normalization reduction.
template <int H, int BLOCK_SIZE, typename Norm>
__launch_bounds__(BLOCK_SIZE) __global__ void mhc_pre_big_fuse_with_norm_kernel(
    const float* __restrict__ dot_mix, const float* __restrict__ sqrsum,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base, const Norm* __restrict__ norm_weight,
    float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ normalized, int total_tokens, int num_splits,
    float rms_eps, float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value,
    int sinkhorn_repeat, float norm_eps) {
  constexpr int kWorkers = BLOCK_SIZE - kWarpThreads;
  constexpr int kRounds = (H + kWorkers * kVecWidth - 1) / (kWorkers * kVecWidth);
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpThreads - 1);
  __shared__ float pre_mix[kHc4];
  __shared__ float partials[kWorkers / kWarpThreads];
  float y_local[kMixHc4];
  float rstd = 0.0f;
  if (tid < kHc4) {
    float sq_total = 0.0f;
#pragma unroll
    for (int c = 0; c < kMixHc4; ++c) y_local[c] = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
      sq_total += sqrsum[s * total_tokens + token];
#pragma unroll
      for (int c = 0; c < kMixHc4; ++c) {
        y_local[c] += dot_mix[(static_cast<long long>(s) * total_tokens + token) * kMixHc4 + c];
      }
    }
    rstd = rsqrtf(sq_total / static_cast<float>(kHc4 * H) + rms_eps);
    const float v = y_local[tid] * rstd * mhc_scale[0] + mhc_base[tid];
    pre_mix[tid] = 1.0f / (1.0f + expf(-v)) + mhc_pre_eps;
  }
  __syncthreads();
  if (tid < kWarpThreads) {
    if (tid < kHc4) {
      write_token_metadata<false>(y_local, rstd, mhc_scale, mhc_base,
          post_mix + token * kHc4, comb_mix + token * kHc4 * kHc4, pre_mix,
          mhc_pre_eps, mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat);
    }
    return;
  }

  const int worker = tid - kWarpThreads;
  const int worker_warp = worker / kWarpThreads;
  const auto* row = residual + static_cast<long long>(token) * kHc4 * H;
  Bf16x8 collapsed[kRounds];
  float sum = 0.0f;
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int h = (worker + r * kWorkers) * kVecWidth;
    if (h < H) {
      float acc[kVecWidth] = {};
#pragma unroll
      for (int j = 0; j < kHc4; ++j) {
        const Bf16x8 raw = load_bf16x8(row + j * H + h);
#pragma unroll
        for (int v = 0; v < kVecWidth / 2; ++v) {
          const float2 f = __bfloat1622float2(bf16x8_get_pair(raw, v));
          acc[2 * v] = fmaf(pre_mix[j], f.x, acc[2 * v]);
          acc[2 * v + 1] = fmaf(pre_mix[j], f.y, acc[2 * v + 1]);
        }
      }
      auto* pairs = reinterpret_cast<__nv_bfloat162*>(&collapsed[r].raw);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const auto packed = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
        pairs[v] = packed;
        // Both the variance and normalized numerator consume the BF16 collapse.
        const float2 rounded = __bfloat1622float2(packed);
        sum = fmaf(rounded.x, rounded.x, sum);
        sum = fmaf(rounded.y, rounded.y, sum);
      }
    }
  }
  sum = warp_sum(sum);
  if (lane == 0) partials[worker_warp] = sum;
  asm volatile("bar.sync 1, %0;" : : "n"(kWorkers) : "memory");
  if (worker_warp == 0) {
    sum = lane < kWorkers / kWarpThreads ? partials[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) partials[0] = rsqrtf(sum / static_cast<float>(H) + norm_eps);
  }
  asm volatile("bar.sync 1, %0;" : : "n"(kWorkers) : "memory");
  const float inverse = partials[0];
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const int h = (worker + r * kWorkers) * kVecWidth;
    if (h < H) {
      Bf16x8 output;
      auto* pairs = reinterpret_cast<__nv_bfloat162*>(&output.raw);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const float2 rounded = __bfloat1622float2(bf16x8_get_pair(collapsed[r], v));
        const float a = __fmul_rn(__fmul_rn(rounded.x, inverse),
                                 static_cast<float>(norm_weight[h + 2 * v]));
        const float b = __fmul_rn(__fmul_rn(rounded.y, inverse),
                                 static_cast<float>(norm_weight[h + 2 * v + 1]));
        pairs[v] = __float22bfloat162_rn(make_float2(a, b));
      }
      store_bf16x8(normalized + static_cast<long long>(token) * H + h, output);
    }
  }
}

// Named barrier `id` over `count` participating threads.
template <int ID, int COUNT>
__device__ __forceinline__ void named_barrier_sync() {
  asm volatile("bar.sync %0, %1;" : : "n"(ID), "n"(COUNT) : "memory");
}

template <int ID, int COUNT>
__device__ __forceinline__ void named_barrier_arrive() {
  asm volatile("bar.arrive %0, %1;" : : "n"(ID), "n"(COUNT) : "memory");
}

// Norm weights of one eight-element vector, held in registers until the
// normalization reads them.
template <typename Norm>
struct NormWeights8;

template <>
struct NormWeights8<__nv_bfloat16> {
  Bf16x8 packed;
  __device__ __forceinline__ void load(const __nv_bfloat16* ptr) { packed = load_bf16x8(ptr); }
  __device__ __forceinline__ float get(int i) const {
    const __nv_bfloat162 pair = bf16x8_get_pair(packed, i >> 1);
    return (i & 1) ? __high2float(pair) : __low2float(pair);
  }
};

template <>
struct NormWeights8<float> {
  float4 lo, hi;
  __device__ __forceinline__ void load(const float* ptr) {
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(lo.x), "=f"(lo.y), "=f"(lo.z), "=f"(lo.w)
                 : "l"(ptr));
    asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
                 : "=f"(hi.x), "=f"(hi.y), "=f"(hi.z), "=f"(hi.w)
                 : "l"(ptr + 4));
  }
  __device__ __forceinline__ float get(int i) const {
    return i < 4 ? (i == 0 ? lo.x : (i == 1 ? lo.y : (i == 2 ? lo.z : lo.w)))
                 : (i == 4 ? hi.x : (i == 5 ? hi.y : (i == 6 ? hi.z : hi.w)));
  }
};

// Finalizer over TOKENS tokens per CTA. Thread layout: warp zero is the
// coordinator, then TOKENS groups of (BLOCK_SIZE - 32) workers, one group per
// token. A worker issues all of its residual and norm-weight loads before the
// first arithmetic operation, so each token costs one memory round trip. The
// coordinator lanes 4g..4g+3 produce token g's mix coefficients; every lane
// runs the Sinkhorn iterations (lanes without a token compute replicated data
// that is never stored). Per-element arithmetic matches
// mhc_pre_big_fuse_with_norm_kernel, so the outputs are bitwise identical.
template <int H, int BLOCK_SIZE, typename Norm, int TOKENS>
__launch_bounds__(kWarpThreads + TOKENS*(BLOCK_SIZE - kWarpThreads),
                  TOKENS) __global__ void mhc_pre_big_fuse_with_norm_batched_kernel(
    const float* __restrict__ dot_mix, const float* __restrict__ sqrsum,
    const __nv_bfloat16* __restrict__ residual, const float* __restrict__ mhc_scale,
    const float* __restrict__ mhc_base, const Norm* __restrict__ norm_weight,
    float* __restrict__ post_mix, float* __restrict__ comb_mix,
    __nv_bfloat16* __restrict__ normalized, int total_tokens, int num_splits, float rms_eps,
    float mhc_pre_eps, float mhc_sinkhorn_eps, float mhc_post_mult_value, int sinkhorn_repeat,
    float norm_eps) {
  static_assert(TOKENS == 1 || TOKENS == 2, "tokens per CTA must be 1 or 2");
  constexpr int kWorkers = BLOCK_SIZE - kWarpThreads;
  constexpr int kThreads = kWarpThreads + TOKENS * kWorkers;
  constexpr int kRounds = (H + kWorkers * kVecWidth - 1) / (kWorkers * kVecWidth);
  static_assert(kRounds * kWorkers * kVecWidth == H, "every round must be full");
  const int tid = threadIdx.x;
  const int lane = tid & (kWarpThreads - 1);
  const int first_token = blockIdx.x * TOKENS;
  __shared__ float pre_mix[TOKENS][kHc4];
  __shared__ float partials[TOKENS][kWorkers / kWarpThreads];

  if (tid < kWarpThreads) {
    // Coordinator: lane l serves token first_token + (l >> 2) % TOKENS, column l & 3.
    const int group = (lane >> 2) & (TOKENS - 1);
    const int column = lane & (kHc4 - 1);
    const int token = first_token + group;
    const bool valid = token < total_tokens;
    float sq_total = 0.0f;
    float y_pre = 0.0f, y_post = 0.0f;
    float y_comb[kHc4];
#pragma unroll
    for (int k = 0; k < kHc4; ++k) y_comb[k] = 0.0f;
    if (valid) {
      for (int s = 0; s < num_splits; ++s) {
        sq_total += sqrsum[s * total_tokens + token];
        const float* y_row = dot_mix + (static_cast<long long>(s) * total_tokens + token) * kMixHc4;
        y_pre += y_row[column];
        y_post += y_row[kHc4 + column];
#pragma unroll
        for (int k = 0; k < kHc4; ++k) y_comb[k] += y_row[2 * kHc4 + column * kHc4 + k];
      }
    }
    const float rstd = rsqrtf(sq_total / static_cast<float>(kHc4 * H) + rms_eps);
    const float scale_pre = mhc_scale[0];
    const float scale_post = mhc_scale[1];
    const float scale_comb = mhc_scale[2];

    float v = y_pre * rstd * scale_pre + mhc_base[column];
    if (lane < TOKENS * kHc4) {
      pre_mix[group][column] = 1.0f / (1.0f + expf(-v)) + mhc_pre_eps;
    }
    named_barrier_arrive<0, kThreads>();

    v = y_post * rstd * scale_post + mhc_base[kHc4 + column];
    const float post = 1.0f / (1.0f + expf(-v)) * mhc_post_mult_value;
    float cm[kHc4];
#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      cm[k] = y_comb[k] * rstd * scale_comb + mhc_base[2 * kHc4 + column * kHc4 + k];
    }
    constexpr unsigned kAllLanes = 0xffffffffu;
    float row_max = fmaxf(fmaxf(cm[0], cm[1]), fmaxf(cm[2], cm[3]));
#pragma unroll
    for (int k = 0; k < kHc4; ++k) cm[k] = expf(cm[k] - row_max);
    float row_sum = cm[0] + cm[1] + cm[2] + cm[3];
#pragma unroll
    for (int k = 0; k < kHc4; ++k) cm[k] = cm[k] / row_sum + mhc_sinkhorn_eps;
#pragma unroll
    for (int k = 0; k < kHc4; ++k) {
      float col_sum = cm[k];
      col_sum += __shfl_xor_sync(kAllLanes, col_sum, 1);
      col_sum += __shfl_xor_sync(kAllLanes, col_sum, 2);
      cm[k] /= (col_sum + mhc_sinkhorn_eps);
    }
    for (int it = 1; it < sinkhorn_repeat; ++it) {
      row_sum = cm[0] + cm[1] + cm[2] + cm[3] + mhc_sinkhorn_eps;
#pragma unroll
      for (int k = 0; k < kHc4; ++k) cm[k] /= row_sum;
#pragma unroll
      for (int k = 0; k < kHc4; ++k) {
        float col_sum = cm[k];
        col_sum += __shfl_xor_sync(kAllLanes, col_sum, 1);
        col_sum += __shfl_xor_sync(kAllLanes, col_sum, 2);
        cm[k] /= (col_sum + mhc_sinkhorn_eps);
      }
    }
    if (valid && lane < TOKENS * kHc4) {
      post_mix[token * kHc4 + column] = post;
#pragma unroll
      for (int k = 0; k < kHc4; ++k) comb_mix[token * kHc4 * kHc4 + column * kHc4 + k] = cm[k];
    }
    return;
  }

  const int worker = tid - kWarpThreads;
  const int group = worker / kWorkers;
  const int in_group = worker - group * kWorkers;
  const int worker_warp = in_group / kWarpThreads;
  const int token = first_token + group;
  const bool valid = token < total_tokens;
  const auto* row = residual + static_cast<long long>(token) * kHc4 * H;

  // BF16 weights (16 registers) load first, then every residual load of the token,
  // before any arithmetic. FP32 weights need 32 registers; together with the 64
  // residual registers they exceed the 112-register budget and spill, so they load
  // right after the collapse, overlapping the reduction barriers.
  constexpr bool kWeightsAtEntry = sizeof(Norm) == 2;
  NormWeights8<Norm> weights[kRounds];
  Bf16x8 raw[kRounds][kHc4];
  if (valid) {
    if constexpr (kWeightsAtEntry) {
#pragma unroll
      for (int r = 0; r < kRounds; ++r) {
        weights[r].load(norm_weight + (in_group + r * kWorkers) * kVecWidth);
      }
    }
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
#pragma unroll
      for (int j = 0; j < kHc4; ++j) {
        raw[r][j] = load_bf16x8(row + j * H + (in_group + r * kWorkers) * kVecWidth);
      }
    }
  }
  named_barrier_sync<0, kThreads>();
  float pre[kHc4];
#pragma unroll
  for (int j = 0; j < kHc4; ++j) pre[j] = pre_mix[group][j];

  Bf16x8 collapsed[kRounds];
  float sum = 0.0f;
  if (valid) {
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
      float acc[kVecWidth] = {};
#pragma unroll
      for (int j = 0; j < kHc4; ++j) {
#pragma unroll
        for (int v = 0; v < kVecWidth / 2; ++v) {
          const float2 f = __bfloat1622float2(bf16x8_get_pair(raw[r][j], v));
          acc[2 * v] = fmaf(pre[j], f.x, acc[2 * v]);
          acc[2 * v + 1] = fmaf(pre[j], f.y, acc[2 * v + 1]);
        }
      }
      auto* pairs = reinterpret_cast<__nv_bfloat162*>(&collapsed[r].raw);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const auto packed = __float22bfloat162_rn(make_float2(acc[2 * v], acc[2 * v + 1]));
        pairs[v] = packed;
        const float2 rounded = __bfloat1622float2(packed);
        sum = fmaf(rounded.x, rounded.x, sum);
        sum = fmaf(rounded.y, rounded.y, sum);
      }
    }
  }
  if constexpr (!kWeightsAtEntry) {
    if (valid) {
#pragma unroll
      for (int r = 0; r < kRounds; ++r) {
        weights[r].load(norm_weight + (in_group + r * kWorkers) * kVecWidth);
      }
    }
  }
  // Each group reduces on its own barrier; inactive workers contribute zero.
  sum = warp_sum(sum);
  if (lane == 0) partials[group][worker_warp] = sum;
  if (group == 0) {
    named_barrier_sync<1, kWorkers>();
  } else {
    named_barrier_sync<2, kWorkers>();
  }
  if (worker_warp == 0) {
    sum = lane < kWorkers / kWarpThreads ? partials[group][lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) partials[group][0] = rsqrtf(sum / static_cast<float>(H) + norm_eps);
  }
  if (group == 0) {
    named_barrier_sync<1, kWorkers>();
  } else {
    named_barrier_sync<2, kWorkers>();
  }
  const float inverse = partials[group][0];
  if (valid) {
#pragma unroll
    for (int r = 0; r < kRounds; ++r) {
      Bf16x8 output;
      auto* pairs = reinterpret_cast<__nv_bfloat162*>(&output.raw);
#pragma unroll
      for (int v = 0; v < kVecWidth / 2; ++v) {
        const float2 rounded = __bfloat1622float2(bf16x8_get_pair(collapsed[r], v));
        const float a = __fmul_rn(__fmul_rn(rounded.x, inverse), weights[r].get(2 * v));
        const float b = __fmul_rn(__fmul_rn(rounded.y, inverse), weights[r].get(2 * v + 1));
        pairs[v] = __float22bfloat162_rn(make_float2(a, b));
      }
      store_bf16x8(normalized + static_cast<long long>(token) * H + (in_group + r * kWorkers) * kVecWidth,
                   output);
    }
  }
}

static int select_pre_big_fuse_block_size(int /* total_tokens */) { return 256; }

static int select_pre_big_fuse_with_prenorm_block_size(int /* total_tokens */) { return 128; }

template <int NUM_SPLITS, bool COMPUTE_SQRSUM>
void dispatch_pre_big_fuse(const float* dot_mix, const float* sqrsum, const __nv_bfloat16* residual,
                           const float* mhc_scale, const float* mhc_base, float* post_mix,
                           float* comb_mix, __nv_bfloat16* layer_input, int total_tokens, int K,
                           int H, float rms_eps, float mhc_pre_eps, float mhc_sinkhorn_eps,
                           float mhc_post_mult_value, int sinkhorn_repeat, int block_size,
                           cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(total_tokens));

#define LAUNCH_PRE_BIG_FUSE(BS)                                                                  \
  mhc_pre_big_fuse_kernel<NUM_SPLITS, BS, COMPUTE_SQRSUM>                                        \
      <<<grid, BS, 0, stream>>>(dot_mix, sqrsum, residual, mhc_scale, mhc_base, post_mix,        \
                                comb_mix, layer_input, total_tokens, K, H, rms_eps, mhc_pre_eps, \
                                mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat)

  if (block_size >= 512) {
    LAUNCH_PRE_BIG_FUSE(512);
  } else if (block_size >= 256) {
    LAUNCH_PRE_BIG_FUSE(256);
  } else {
    LAUNCH_PRE_BIG_FUSE(128);
  }

#undef LAUNCH_PRE_BIG_FUSE
}

void launch_pre_big_fuse(TensorView dot_mix, TensorView sqrsum, TensorView residual,
                         TensorView mhc_scale, TensorView mhc_base, TensorView post_mix,
                         TensorView comb_mix, TensorView layer_input, int64_t total_tokens,
                         int64_t K, int64_t H, double rms_eps, double mhc_pre_eps,
                         double mhc_sinkhorn_eps, double mhc_post_mult_value,
                         int64_t sinkhorn_repeat, int64_t num_splits, int64_t block_size,
                         cudaStream_t stream) {
  const int bs = block_size > 0 ? static_cast<int>(block_size)
                                : select_pre_big_fuse_block_size(static_cast<int>(total_tokens));
  const auto* dot_ptr = static_cast<const float*>(dot_mix.data_ptr());
  const auto* sq_ptr = static_cast<const float*>(sqrsum.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(mhc_scale.data_ptr());
  const auto* base_ptr = static_cast<const float*>(mhc_base.data_ptr());
  auto* post_ptr = static_cast<float*>(post_mix.data_ptr());
  auto* comb_ptr = static_cast<float*>(comb_mix.data_ptr());
  auto* layer_ptr = reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr());

#define DISPATCH_SPLITS(NS)                                                              \
  dispatch_pre_big_fuse<NS, false>(                                                      \
      dot_ptr, sq_ptr, residual_ptr, scale_ptr, base_ptr, post_ptr, comb_ptr, layer_ptr, \
      static_cast<int>(total_tokens), static_cast<int>(K), static_cast<int>(H),          \
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),                      \
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),     \
      static_cast<int>(sinkhorn_repeat), bs, stream)

  switch (num_splits) {
    case 1:
      DISPATCH_SPLITS(1);
      break;
    case 2:
      DISPATCH_SPLITS(2);
      break;
    case 4:
      DISPATCH_SPLITS(4);
      break;
    case 8:
      DISPATCH_SPLITS(8);
      break;
    case 16:
      DISPATCH_SPLITS(16);
      break;
  }
#undef DISPATCH_SPLITS
}

void launch_pre_big_fuse_with_prenorm(TensorView dot_mix, TensorView residual, TensorView mhc_scale,
                                      TensorView mhc_base, TensorView post_mix, TensorView comb_mix,
                                      TensorView layer_input, int64_t total_tokens, int64_t H,
                                      double rms_eps, double mhc_pre_eps, double mhc_sinkhorn_eps,
                                      double mhc_post_mult_value, int64_t sinkhorn_repeat,
                                      int64_t block_size, cudaStream_t stream) {
  const int bs = block_size > 0
                     ? static_cast<int>(block_size)
                     : select_pre_big_fuse_with_prenorm_block_size(static_cast<int>(total_tokens));
  const auto* dot_ptr = static_cast<const float*>(dot_mix.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* scale_ptr = static_cast<const float*>(mhc_scale.data_ptr());
  const auto* base_ptr = static_cast<const float*>(mhc_base.data_ptr());
  auto* post_ptr = static_cast<float*>(post_mix.data_ptr());
  auto* comb_ptr = static_cast<float*>(comb_mix.data_ptr());
  auto* layer_ptr = reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr());

  dispatch_pre_big_fuse<1, true>(
      dot_ptr, nullptr, residual_ptr, scale_ptr, base_ptr, post_ptr, comb_ptr, layer_ptr,
      static_cast<int>(total_tokens), static_cast<int>(kHc4 * H), static_cast<int>(H),
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),
      static_cast<int>(sinkhorn_repeat), bs, stream);
}

void check_common_shapes(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                         TensorView residual, TensorView mhc_scale, TensorView mhc_base,
                         int64_t* total_tokens, int64_t* H) {
  CHECK_INPUT_AND_TYPE(post_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(comb_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(layer_input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(residual, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(mhc_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(mhc_base, dl_float32);
  CHECK_DEVICE(residual, post_mix);
  CHECK_DEVICE(residual, comb_mix);
  CHECK_DEVICE(residual, layer_input);
  CHECK_DEVICE(residual, mhc_scale);
  CHECK_DEVICE(residual, mhc_base);
  CHECK_DIM(3, residual);
  CHECK_DIM(2, post_mix);
  CHECK_DIM(3, comb_mix);
  CHECK_DIM(2, layer_input);
  CHECK_DIM(1, mhc_scale);
  CHECK_DIM(1, mhc_base);

  *total_tokens = residual.size(0);
  const int64_t HC = residual.size(1);
  *H = residual.size(2);
  TVM_FFI_ICHECK_EQ(HC, kHc4) << "residual.shape[1] / HC must be 4";
  TVM_FFI_ICHECK_GT(*H, 0) << "hidden size must be positive";
  TVM_FFI_ICHECK_EQ(*H % kVecWidth, 0) << "hidden size must be divisible by 8";
  TVM_FFI_ICHECK_EQ(mhc_scale.size(0), 3) << "mhc_scale must have shape [3]";
  TVM_FFI_ICHECK_EQ(mhc_base.size(0), kMixHc4) << "mhc_base must have shape [24]";
  TVM_FFI_ICHECK_EQ(post_mix.size(0), *total_tokens)
      << "post_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(post_mix.size(1), kHc4) << "post_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_mix.size(0), *total_tokens)
      << "comb_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(comb_mix.size(1), kHc4) << "comb_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_mix.size(2), kHc4) << "comb_mix.shape[2] must be 4";
  TVM_FFI_ICHECK_EQ(layer_input.size(0), *total_tokens)
      << "layer_input.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(layer_input.size(1), *H) << "layer_input.shape[1] must match hidden size";
}

}  // namespace

void mhc_pre_big_fuse(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                      TensorView dot_mix, TensorView sqrsum, TensorView residual,
                      TensorView mhc_scale, TensorView mhc_base, int64_t K, double rms_eps,
                      double mhc_pre_eps, double mhc_sinkhorn_eps, double mhc_post_mult_value,
                      int64_t sinkhorn_repeat, int64_t num_splits, int64_t block_size) {
  CHECK_INPUT_AND_TYPE(dot_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(sqrsum, dl_float32);
  int64_t total_tokens = 0;
  int64_t H = 0;
  check_common_shapes(post_mix, comb_mix, layer_input, residual, mhc_scale, mhc_base, &total_tokens,
                      &H);
  CHECK_DEVICE(residual, dot_mix);
  CHECK_DEVICE(residual, sqrsum);

  TVM_FFI_ICHECK_GT(K, 0) << "K must be positive";
  TVM_FFI_ICHECK_GE(sinkhorn_repeat, 1) << "sinkhorn_repeat must be >= 1";
  TVM_FFI_ICHECK(num_splits == 1 || num_splits == 2 || num_splits == 4 || num_splits == 8 ||
                 num_splits == 16)
      << "num_splits must be one of {1,2,4,8,16}";

  if (num_splits == 1) {
    CHECK_DIM(2, dot_mix);
    CHECK_DIM(1, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), total_tokens)
        << "dot_mix.shape[0] must match residual.shape[0]";
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), kMixHc4) << "dot_mix.shape[1] must be 24";
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), total_tokens)
        << "sqrsum.shape[0] must match residual.shape[0]";
  } else {
    CHECK_DIM(3, dot_mix);
    CHECK_DIM(2, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), num_splits) << "dot_mix.shape[0] must match num_splits";
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), total_tokens)
        << "dot_mix.shape[1] must match residual.shape[0]";
    TVM_FFI_ICHECK_EQ(dot_mix.size(2), kMixHc4) << "dot_mix.shape[2] must be 24";
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), num_splits) << "sqrsum.shape[0] must match num_splits";
    TVM_FFI_ICHECK_EQ(sqrsum.size(1), total_tokens)
        << "sqrsum.shape[1] must match residual.shape[0]";
  }

  if (total_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());
  launch_pre_big_fuse(dot_mix, sqrsum, residual, mhc_scale, mhc_base, post_mix, comb_mix,
                      layer_input, total_tokens, K, H, rms_eps, mhc_pre_eps, mhc_sinkhorn_eps,
                      mhc_post_mult_value, sinkhorn_repeat, num_splits, block_size, stream);
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse kernel launch failed with error code " << cudaGetErrorString(status);
}

void mhc_pre_big_fuse_with_prenorm(TensorView post_mix, TensorView comb_mix, TensorView layer_input,
                                   TensorView dot_mix, TensorView residual, TensorView mhc_scale,
                                   TensorView mhc_base, double rms_eps, double mhc_pre_eps,
                                   double mhc_sinkhorn_eps, double mhc_post_mult_value,
                                   int64_t sinkhorn_repeat, int64_t block_size) {
  CHECK_INPUT_AND_TYPE(dot_mix, dl_float32);
  int64_t total_tokens = 0;
  int64_t H = 0;
  check_common_shapes(post_mix, comb_mix, layer_input, residual, mhc_scale, mhc_base, &total_tokens,
                      &H);
  CHECK_DEVICE(residual, dot_mix);
  CHECK_DIM(2, dot_mix);
  TVM_FFI_ICHECK_EQ(dot_mix.size(0), total_tokens)
      << "dot_mix.shape[0] must match residual.shape[0]";
  TVM_FFI_ICHECK_EQ(dot_mix.size(1), kMixHc4) << "dot_mix.shape[1] must be 24";
  TVM_FFI_ICHECK_GE(sinkhorn_repeat, 1) << "sinkhorn_repeat must be >= 1";

  if (total_tokens == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());
  launch_pre_big_fuse_with_prenorm(dot_mix, residual, mhc_scale, mhc_base, post_mix, comb_mix,
                                   layer_input, total_tokens, H, rms_eps, mhc_pre_eps,
                                   mhc_sinkhorn_eps, mhc_post_mult_value, sinkhorn_repeat,
                                   block_size, stream);
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse_with_prenorm kernel launch failed with error code "
      << cudaGetErrorString(status);
}

void mhc_pre_big_fuse_with_norm(
    TensorView post_mix, TensorView comb_mix, TensorView normalized, TensorView dot_mix,
    TensorView sqrsum, TensorView residual, TensorView mhc_scale, TensorView mhc_base,
    TensorView norm_weight, double rms_eps, double mhc_pre_eps, double mhc_sinkhorn_eps,
    double mhc_post_mult_value, int64_t sinkhorn_repeat, double norm_eps,
    int64_t num_splits, int64_t block_size, int64_t tokens_per_cta) {
  int64_t total_tokens = 0, H = 0;
  check_common_shapes(post_mix, comb_mix, normalized, residual, mhc_scale, mhc_base,
                      &total_tokens, &H);
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16, 0)
      << "residual must have a 16-byte aligned base";
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(normalized.data_ptr()) % 16, 0)
      << "normalized must have a 16-byte aligned base";
  CHECK_INPUT_AND_TYPE(dot_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(sqrsum, dl_float32);
  CHECK_INPUT(norm_weight);
  CHECK_DEVICE(residual, dot_mix);
  CHECK_DEVICE(residual, sqrsum);
  CHECK_DEVICE(residual, norm_weight);
  CHECK_DIM(1, norm_weight);
  TVM_FFI_ICHECK_EQ(norm_weight.size(0), H) << "norm_weight must have shape [H]";
  TVM_FFI_ICHECK(norm_weight.dtype() == dl_bfloat16 || norm_weight.dtype() == dl_float32)
      << "norm_weight must be BF16 or FP32";
  TVM_FFI_ICHECK(H == 4096 || H == 5120 || H == 7168)
      << "weighted mHC finalization requires hidden size 4096, 5120, or 7168";
  TVM_FFI_ICHECK(num_splits == 1 || num_splits == 2 || num_splits == 4 ||
                num_splits == 8 || num_splits == 16)
      << "num_splits must be one of {1,2,4,8,16}";
  TVM_FFI_ICHECK_GE(sinkhorn_repeat, 1);
  TVM_FFI_ICHECK_LE(sinkhorn_repeat, INT32_MAX);
  TVM_FFI_ICHECK_GT(rms_eps, 0.0);
  TVM_FFI_ICHECK_GT(mhc_pre_eps, 0.0);
  TVM_FFI_ICHECK_GT(mhc_sinkhorn_eps, 0.0);
  TVM_FFI_ICHECK_GT(norm_eps, 0.0);
  if (num_splits == 1) {
    CHECK_DIM(2, dot_mix);
    CHECK_DIM(1, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), total_tokens);
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), kMixHc4);
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), total_tokens);
  } else {
    CHECK_DIM(3, dot_mix);
    CHECK_DIM(2, sqrsum);
    TVM_FFI_ICHECK_EQ(dot_mix.size(0), num_splits);
    TVM_FFI_ICHECK_EQ(dot_mix.size(1), total_tokens);
    TVM_FFI_ICHECK_EQ(dot_mix.size(2), kMixHc4);
    TVM_FFI_ICHECK_EQ(sqrsum.size(0), num_splits);
    TVM_FFI_ICHECK_EQ(sqrsum.size(1), total_tokens);
  }
  const int64_t selected_block = block_size == 0 ? 160 : block_size;
  TVM_FFI_ICHECK(selected_block == 96 || selected_block == 160 || selected_block == 288)
      << "weighted mHC finalization requires block size 96, 160, or 288";
  const int block = static_cast<int>(selected_block);
  TVM_FFI_ICHECK(tokens_per_cta == 1 || tokens_per_cta == 2)
      << "tokens_per_cta must be 1 or 2";
  if (tokens_per_cta == 2) {
    TVM_FFI_ICHECK_EQ(block, 160) << "tokens_per_cta 2 requires block size 160";
    TVM_FFI_ICHECK_EQ(H, 4096) << "tokens_per_cta 2 requires hidden size 4096";
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(norm_weight.data_ptr()) % 16, 0)
        << "norm_weight must have a 16-byte aligned base for tokens_per_cta 2";
  }
  // The row index and the split-square index use signed 32-bit arithmetic.
  TVM_FFI_ICHECK_LE(total_tokens, static_cast<int64_t>(INT32_MAX) / (kHc4 * kHc4));
  if (total_tokens == 0) return;
  auto overlaps = [](TensorView a, TensorView b) {
    const auto begin_a = reinterpret_cast<uintptr_t>(a.data_ptr());
    const auto begin_b = reinterpret_cast<uintptr_t>(b.data_ptr());
    const auto bytes_a = a.numel() * (a.dtype().bits / 8);
    const auto bytes_b = b.numel() * (b.dtype().bits / 8);
    return bytes_a && bytes_b && begin_a < begin_b + bytes_b && begin_b < begin_a + bytes_a;
  };
  for (auto output : {post_mix, comb_mix, normalized}) {
    for (auto input : {dot_mix, sqrsum, residual, mhc_scale, mhc_base, norm_weight}) {
      TVM_FFI_ICHECK(!overlaps(output, input)) << "mHC outputs must not overlap inputs";
    }
  }
  TVM_FFI_ICHECK(!overlaps(post_mix, comb_mix) && !overlaps(post_mix, normalized) &&
                !overlaps(comb_mix, normalized))
      << "mHC outputs must not overlap each other";
  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());

#define LAUNCH_NORM(HIDDEN, BLOCK, NORM)                                                  \
  mhc_pre_big_fuse_with_norm_kernel<HIDDEN, BLOCK, NORM><<<total_tokens, BLOCK, 0, stream>>>( \
      static_cast<const float*>(dot_mix.data_ptr()),                                     \
      static_cast<const float*>(sqrsum.data_ptr()),                                      \
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),                        \
      static_cast<const float*>(mhc_scale.data_ptr()),                                   \
      static_cast<const float*>(mhc_base.data_ptr()),                                     \
      static_cast<const NORM*>(norm_weight.data_ptr()),                                  \
      static_cast<float*>(post_mix.data_ptr()), static_cast<float*>(comb_mix.data_ptr()), \
      reinterpret_cast<__nv_bfloat16*>(normalized.data_ptr()),                            \
      static_cast<int>(total_tokens), static_cast<int>(num_splits),                       \
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),                       \
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),       \
      static_cast<int>(sinkhorn_repeat), static_cast<float>(norm_eps))
#define DISPATCH_NORM(HIDDEN, BLOCK)                 \
  if (norm_weight.dtype() == dl_float32) {           \
    LAUNCH_NORM(HIDDEN, BLOCK, float);               \
  } else {                                         \
    LAUNCH_NORM(HIDDEN, BLOCK, __nv_bfloat16);        \
  }
#define DISPATCH_NORM_BLOCK(HIDDEN)                 \
  if (block == 96) {                               \
    DISPATCH_NORM(HIDDEN, 96);                     \
  } else if (block == 160) {                       \
    DISPATCH_NORM(HIDDEN, 160);                    \
  } else {                                        \
    DISPATCH_NORM(HIDDEN, 288);                    \
  }
#define LAUNCH_NORM_BATCHED(NORM)                                                          \
  mhc_pre_big_fuse_with_norm_batched_kernel<4096, 160, NORM, 2>                            \
      <<<(total_tokens + 1) / 2, 32 + 2 * (160 - 32), 0, stream>>>(                        \
      static_cast<const float*>(dot_mix.data_ptr()),                                     \
      static_cast<const float*>(sqrsum.data_ptr()),                                      \
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),                        \
      static_cast<const float*>(mhc_scale.data_ptr()),                                   \
      static_cast<const float*>(mhc_base.data_ptr()),                                     \
      static_cast<const NORM*>(norm_weight.data_ptr()),                                  \
      static_cast<float*>(post_mix.data_ptr()), static_cast<float*>(comb_mix.data_ptr()), \
      reinterpret_cast<__nv_bfloat16*>(normalized.data_ptr()),                            \
      static_cast<int>(total_tokens), static_cast<int>(num_splits),                       \
      static_cast<float>(rms_eps), static_cast<float>(mhc_pre_eps),                       \
      static_cast<float>(mhc_sinkhorn_eps), static_cast<float>(mhc_post_mult_value),       \
      static_cast<int>(sinkhorn_repeat), static_cast<float>(norm_eps))
  if (tokens_per_cta == 2) {
    if (norm_weight.dtype() == dl_float32) {
      LAUNCH_NORM_BATCHED(float);
    } else {
      LAUNCH_NORM_BATCHED(__nv_bfloat16);
    }
  } else if (H == 4096) {
    DISPATCH_NORM_BLOCK(4096);
  } else if (H == 5120) {
    DISPATCH_NORM_BLOCK(5120);
  } else {
    DISPATCH_NORM_BLOCK(7168);
  }
#undef LAUNCH_NORM_BATCHED
#undef DISPATCH_NORM_BLOCK
#undef DISPATCH_NORM
#undef LAUNCH_NORM
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse_with_norm kernel launch failed with error code "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_big_fuse, flashinfer::mhc::mhc_pre_big_fuse);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_big_fuse_with_prenorm,
                              flashinfer::mhc::mhc_pre_big_fuse_with_prenorm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mhc_pre_big_fuse_with_norm,
                             flashinfer::mhc::mhc_pre_big_fuse_with_norm);

}  // namespace flashinfer::mhc
