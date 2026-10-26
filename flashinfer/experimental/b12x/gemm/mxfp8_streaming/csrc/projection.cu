// SPDX-License-Identifier: Apache-2.0
// Streaming E4M3/g32 MXFP8 projection, with FP32 accumulation and BF16 output.
// Derived from MoET dsv41_sm120/sm120_gemv/mxfp8_gemv_sm120.cu,
// revision 5d25cff77495. The reachable device helpers and arithmetic body
// are retained from the source used to qualify the SM120 projection.
// clang-format off
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
namespace vllm_moet_sm120 {
constexpr int kBlockSize = 32;

constexpr int kWarps = 8;

constexpr int kThreads = kWarps * 32;

constexpr int kMmaCols = 8;

constexpr int kV3MaxM = 8;

// Exponent zero is reserved for encoded blocks whose values are all zero.
__device__ __forceinline__ float ue8m0_to_float(uint32_t e) {
  return e == 0u ? 0.0f : __uint_as_float(e << 23);
}

__device__ __forceinline__ void griddep_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void griddep_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint64_t evict_first_policy() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}

__device__ __forceinline__ uint4 ld_w_v4(const void* ptr, bool pred) {
    uint4 v = make_uint4(0, 0, 0, 0);
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n @p ld.global" ".nc"
                 ".v4.u32 {%0, %1, %2, %3}, [%4];\n}"
                 : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w)
                 : "l"(ptr), "r"((int)pred));
    return v;
  }

__device__ __forceinline__ uint4 ld_act_v4(const void* ptr, bool pred) {
    uint4 v = make_uint4(0, 0, 0, 0);
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n @p ld.global" ""
                 ".v4.u32 {%0, %1, %2, %3}, [%4];\n}"
                 : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w)
                 : "l"(ptr), "r"((int)pred));
    return v;
  }

__device__ __forceinline__ uint32_t ld_w_u32(const void* ptr, bool pred) {
    uint32_t v = 0;
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n @p ld.global" ".nc"
                 ".u32 %0, [%1];\n}"
                 : "+r"(v)
                 : "l"(ptr), "r"((int)pred));
    return v;
  }

__device__ __forceinline__ uint32_t ld_act_u32(const void* ptr, bool pred) {
    uint32_t v = 0;
    asm volatile("{\n .reg .pred p;\n setp.ne.b32 p, %2, 0;\n @p ld.global" ""
                 ".u32 %0, [%1];\n}"
                 : "+r"(v)
                 : "l"(ptr), "r"((int)pred));
    return v;
  }

template <int EV>
__device__ __forceinline__ uint4 ld_weight_v4(const void* ptr, bool pred, uint64_t pol) {
  if constexpr (EV == 0) return ld_w_v4(ptr, pred);
  uint4 v = make_uint4(0, 0, 0, 0);
  asm volatile(
      "{\n .reg .pred p;\n setp.ne.b32 p, %5, 0;\n"
      " @p ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %6;\n}"
      : "+r"(v.x), "+r"(v.y), "+r"(v.z), "+r"(v.w)
      : "l"(ptr), "r"((int)pred), "l"(pol));
  return v;
}

struct GemvArgs {
  const uint8_t* A;
  const uint8_t* A_sf;
  const uint8_t* B;
  const uint8_t* B_sf;
  __nv_bfloat16* C;
  int M, N, K, num_k_tiles;
  int ldc;                 // C row stride (elements)
  long long a_gstride;     // bytes between groups
  long long a_sf_gstride;  // bytes between groups
  long long b_gstride;     // bytes between groups
  long long b_sf_gstride;  // bytes between groups
  long long c_gstride;     // elements between groups
  int a_sf_t_al;           // SF_MN_PACKED: T_aligned (int32 words per K-block column)
  int b_sf_ld;             // SF_ROWMAJOR: bytes per row
  int lda_s;               // v2 SA: shared-memory row stride of the staged A slice (bytes)
};

struct SplitKArgs {
  float* partials;  // [S][groups][M][N] fp32
  int* counters;    // [groups * col_blocks], zero at rest
  int S;
};

__device__ __forceinline__ void mma_m16n8k32_e4m3(float* d, const uint32_t* a, uint32_t b0,
                                                  uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "f"(0.0f),
        "f"(0.0f), "f"(0.0f));
}

__device__ __forceinline__ uint32_t sf_word_index(int row, int kt, int num_k_tiles) {
  const int mt = row >> 7, r = row & 127, a = r >> 5, b = r & 31;
  return ((mt * num_k_tiles + kt) * 32 + b) * 4 + a;  // the 4 bytes kb = 4kt..4kt+3
}

template <int R>
struct V3Smem {
  static constexpr int kRoundK = R * 128;                // k per round
  static constexpr int kRoundKB = kRoundK / kBlockSize;  // MX blocks per round
  static constexpr int kWTile = kMmaCols * kRoundK;      // weight bytes per warp per round
  static constexpr int kATile = kV3MaxM * kRoundK;       // A bytes per warp per round
  static constexpr int kSfWords = 2 * R * kMmaCols;      // per operand, 2 buffers
  static constexpr int kPerWarp = kWTile + kATile + 2 * kSfWords * 4;
  static constexpr size_t kBytes = (size_t)kWarps * kPerWarp;
};

template <int R, int PDL = 0, int EV = 0>
__device__ __forceinline__ void mxfp8_mma_gemv_v3_body(const GemvArgs p, const SplitKArgs sk) {
  using S = V3Smem<R>;
  extern __shared__ __align__(16) uint8_t v3_smem[];
  __shared__ float s_red[kWarps][32][2];
  __shared__ int s_last;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tig = lane & 3;
  const int M = p.M, N = p.N, K = p.K, nkt = p.num_k_tiles;
  const int n0 = blockIdx.x * kMmaCols;
  // Split-K over 128-byte units: block (x, s) owns units [u_begin, u_begin + units).
  const int units_all = K >> 7;
  const int ups = (units_all + gridDim.y - 1) / gridDim.y;
  const int u_begin = blockIdx.y * ups;
  const int units = max(0, min(units_all - u_begin, ups));

  uint8_t* s_w = v3_smem + (size_t)warp * S::kPerWarp;             // [8 rows][kRoundK]
  uint8_t* s_a = s_w + S::kWTile;                                  // [kV3MaxM rows][kRoundK]
  uint32_t* s_bsf = reinterpret_cast<uint32_t*>(s_a + S::kATile);  // [2][R][8]
  uint32_t* s_asf = s_bsf + S::kSfWords;                           // [2][R][8]

  const int nu = units > warp ? (units - warp + kWarps - 1) / kWarps : 0;
  const int rounds = (nu + R - 1) / R;

  // Weight chunks of this lane per unit: rows lr0 / lr1 (0..7), 16-byte offset lo.
  const int lr0 = lane >> 3, lr1 = lr0 + 4, lo = (lane & 7) << 4;
  const bool r0_ok = n0 + lr0 < N, r1_ok = n0 + lr1 < N;
  const uint8_t* b_row0 = p.B + (size_t)(n0 + (r0_ok ? lr0 : 0)) * K + lo;
  const uint8_t* b_row1 = p.B + (size_t)(n0 + (r1_ok ? lr1 : 0)) * K + lo;
  // A chunks of this lane per unit: the same (row, offset) pattern over kV3MaxM rows.
  const bool a0_ok = lr0 < M, a1_ok = lr1 < M;
  const uint8_t* a_row0 = p.A + (size_t)(a0_ok ? lr0 : 0) * K + lo;
  const uint8_t* a_row1 = p.A + (size_t)(a1_ok ? lr1 : 0) * K + lo;
  // Scale words: lanes 0..8R-1 load weight words (row lane&7, unit lane>>3), lanes
  // 16..16+8R-1 load A words (row (lane-16)&7, unit (lane-16)>>3). R <= 2 keeps both
  // inside one warp instruction each; for R = 4 the two sets alias, so two loads.
  const uint32_t* b_sf32 = reinterpret_cast<const uint32_t*>(p.B_sf);
  const uint32_t* a_sf32 = reinterpret_cast<const uint32_t*>(p.A_sf);
  const int sl_row = lane & 7, sl_j = (lane >> 3) % R;
  const bool sl_b = lane < 8 * R && n0 + sl_row < N;
  const bool sl_a = lane < 8 * R && sl_row < M;

  uint4 wreg[R][2], areg[R][2];
  uint32_t sbw = 0u, saw = 0u;

  auto issue_round = [&](int r) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int i = r * R + j;
      const bool ok = i < nu;
      const size_t k0 = (size_t)(u_begin + warp + kWarps * i) << 7;
      wreg[j][0] = (ok && r0_ok) ? __ldg(reinterpret_cast<const uint4*>(b_row0 + k0))
                                 : make_uint4(0, 0, 0, 0);
      wreg[j][1] = (ok && r1_ok) ? __ldg(reinterpret_cast<const uint4*>(b_row1 + k0))
                                 : make_uint4(0, 0, 0, 0);
      areg[j][0] = (ok && a0_ok) ? __ldg(reinterpret_cast<const uint4*>(a_row0 + k0))
                                 : make_uint4(0, 0, 0, 0);
      areg[j][1] = (ok && a1_ok) ? __ldg(reinterpret_cast<const uint4*>(a_row1 + k0))
                                 : make_uint4(0, 0, 0, 0);
    }
    const int i = r * R + sl_j;
    const int kt = u_begin + warp + kWarps * i;
    const bool ok = i < nu;
    sbw = (ok && sl_b) ? __ldg(b_sf32 + sf_word_index(n0 + sl_row, kt, nkt)) : 0u;
    saw = (ok && sl_a) ? __ldg(a_sf32 + sf_word_index(sl_row, kt, nkt)) : 0u;
  };
  // With PDL a round is split: the weight half may be issued before griddepcontrol.wait ...
  const uint64_t pol = (PDL != 0 && EV != 0) ? evict_first_policy() : 0;
  auto issue_round_w = [&](int r) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int i = r * R + j;
      const bool ok = i < nu;
      const size_t k0 = (size_t)(u_begin + warp + kWarps * i) << 7;
      wreg[j][0] = ld_weight_v4<EV>(b_row0 + k0, ok && r0_ok, pol);
      wreg[j][1] = ld_weight_v4<EV>(b_row1 + k0, ok && r1_ok, pol);
    }
    const int i = r * R + sl_j;
    const int kt = u_begin + warp + kWarps * i;
    sbw = ld_w_u32(b_sf32 + sf_word_index(n0 + sl_row, kt, nkt), i < nu && sl_b);
  };
  // ... the activation half (written by the kernel before) only after it, through L2.
  auto issue_round_a = [&](int r) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      const int i = r * R + j;
      const bool ok = i < nu;
      const size_t k0 = (size_t)(u_begin + warp + kWarps * i) << 7;
      areg[j][0] = ld_act_v4(a_row0 + k0, ok && a0_ok);
      areg[j][1] = ld_act_v4(a_row1 + k0, ok && a1_ok);
    }
    const int i = r * R + sl_j;
    const int kt = u_begin + warp + kWarps * i;
    saw = ld_act_u32(a_sf32 + sf_word_index(sl_row, kt, nkt), i < nu && sl_a);
  };

  if constexpr (PDL == 0) {
    if (rounds > 0) issue_round(0);
  } else {
    if (rounds > 0) issue_round_w(0);
    griddep_wait();
    if constexpr (PDL == 1) griddep_launch_dependents();
    if (rounds > 0) issue_round_a(0);
  }

  float acc0 = 0.0f, acc1 = 0.0f;
  const int c0 = n0 + tig * 2, c1 = c0 + 1;
  const bool g_ok = g < M;

  for (int r = 0; r < rounds; ++r) {
#pragma unroll
    for (int j = 0; j < R; ++j) {
      *reinterpret_cast<uint4*>(s_w + (size_t)lr0 * S::kRoundK + j * 128 + lo) = wreg[j][0];
      *reinterpret_cast<uint4*>(s_w + (size_t)lr1 * S::kRoundK + j * 128 + lo) = wreg[j][1];
      *reinterpret_cast<uint4*>(s_a + (size_t)lr0 * S::kRoundK + j * 128 + lo) = areg[j][0];
      *reinterpret_cast<uint4*>(s_a + (size_t)lr1 * S::kRoundK + j * 128 + lo) = areg[j][1];
    }
    const int buf = r & 1;
    if (lane < 8 * R) {
      s_bsf[(buf * R + sl_j) * kMmaCols + sl_row] = sbw;
      s_asf[(buf * R + sl_j) * kMmaCols + sl_row] = saw;
    }
    __syncwarp();
    if (r + 1 < rounds) {
      if constexpr (PDL == 0) {
        issue_round(r + 1);
      } else {
        issue_round_w(r + 1);
        issue_round_a(r + 1);
      }
    }

    uint32_t sb0w[R], sb1w[R], saw_g[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
      sb0w[j] = s_bsf[(buf * R + j) * kMmaCols + tig * 2];
      sb1w[j] = s_bsf[(buf * R + j) * kMmaCols + tig * 2 + 1];
      saw_g[j] = s_asf[(buf * R + j) * kMmaCols + g];
    }
    const int nb = min(S::kRoundKB, (nu - r * R) * 4);
#pragma unroll
    for (int kb = 0; kb < S::kRoundKB; ++kb) {
      if (kb >= nb) break;
      const int j = kb >> 2, c = kb & 3;
      const uint2 wb =
          *reinterpret_cast<const uint2*>(s_w + (size_t)g * S::kRoundK + kb * 32 + tig * 8);
      const uint2 wa =
          g_ok ? *reinterpret_cast<const uint2*>(s_a + (size_t)g * S::kRoundK + kb * 32 + tig * 8)
               : make_uint2(0, 0);
      const uint32_t afrag[4] = {wa.x, 0u, wa.y, 0u};
      float d[4];
      mma_m16n8k32_e4m3(d, afrag, wb.x, wb.y);
      const float fb0 = ue8m0_to_float((sb0w[j] >> (8 * c)) & 0xffu);
      const float fb1 = ue8m0_to_float((sb1w[j] >> (8 * c)) & 0xffu);
      const float fa = ue8m0_to_float((saw_g[j] >> (8 * c)) & 0xffu);
      acc0 = fmaf(d[0], fa * fb0, acc0);
      acc1 = fmaf(d[1], fa * fb1, acc1);
    }
    __syncwarp();
  }
  if constexpr (PDL == 3) griddep_launch_dependents();

  s_red[warp][lane][0] = acc0;
  s_red[warp][lane][1] = acc1;
  __syncthreads();
  if (gridDim.y == 1) {
    if (warp == 0) {
      float r0 = 0.0f, r1 = 0.0f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        r0 += s_red[w][lane][0];
        r1 += s_red[w][lane][1];
      }
      if (g < M) {
        if (c0 < N) p.C[(size_t)g * p.ldc + c0] = __float2bfloat16(r0);
        if (c1 < N) p.C[(size_t)g * p.ldc + c1] = __float2bfloat16(r1);
      }
    }
    if constexpr (PDL == 2) griddep_launch_dependents();
    return;
  }
  // Split-K: publish this block's [kV3MaxM][8] fp32 partial, the last-arriving block sums.
  float* part = sk.partials + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * (kV3MaxM * kMmaCols);
  if (warp == 0) {
    float r0 = 0.0f, r1 = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      r0 += s_red[w][lane][0];
      r1 += s_red[w][lane][1];
    }
    part[g * kMmaCols + tig * 2] = r0;
    part[g * kMmaCols + tig * 2 + 1] = r1;
    __threadfence();
  }
  __syncthreads();
  if (threadIdx.x == 0)
    s_last = (atomicAdd(sk.counters + blockIdx.x, 1) == (int)gridDim.y - 1) ? 1 : 0;
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (threadIdx.x < kV3MaxM * kMmaCols) {
    const int m = threadIdx.x >> 3, col = threadIdx.x & 7;
    float sum = 0.0f;
    for (int q = 0; q < (int)gridDim.y; ++q)
      sum += __ldcg(sk.partials + ((size_t)q * gridDim.x + blockIdx.x) * (kV3MaxM * kMmaCols) +
                    m * kMmaCols + col);
    if (m < M && n0 + col < N) p.C[(size_t)m * p.ldc + n0 + col] = __float2bfloat16(sum);
  }
  if (threadIdx.x == 0) sk.counters[blockIdx.x] = 0;  // ready for the next launch / graph replay
  if constexpr (PDL == 2) griddep_launch_dependents();
}

extern "C" __global__ void __launch_bounds__(kThreads)
b12x_mxfp8_stream_p0_c0(const uint8_t* a, const uint8_t* asf, const uint8_t* w,
    const uint8_t* wsf, __nv_bfloat16* out, int rows, int columns, int inner) {
  GemvArgs p{};
  p.A = a;
  p.A_sf = asf;
  p.B = w;
  p.B_sf = wsf;
  p.C = out;
  p.M = rows;
  p.N = columns;
  p.K = inner;
  p.num_k_tiles = inner / 128;
  p.ldc = columns;
  mxfp8_mma_gemv_v3_body<1, 0, 0>(p, SplitKArgs{nullptr, nullptr, 1});
}

extern "C" __global__ void __launch_bounds__(kThreads)
b12x_mxfp8_stream_p1_c0(const uint8_t* a, const uint8_t* asf, const uint8_t* w,
    const uint8_t* wsf, __nv_bfloat16* out, int rows, int columns, int inner) {
  GemvArgs p{};
  p.A = a;
  p.A_sf = asf;
  p.B = w;
  p.B_sf = wsf;
  p.C = out;
  p.M = rows;
  p.N = columns;
  p.K = inner;
  p.num_k_tiles = inner / 128;
  p.ldc = columns;
  mxfp8_mma_gemv_v3_body<1, 1, 0>(p, SplitKArgs{nullptr, nullptr, 1});
}

extern "C" __global__ void __launch_bounds__(kThreads)
b12x_mxfp8_stream_p1_c1(const uint8_t* a, const uint8_t* asf, const uint8_t* w,
    const uint8_t* wsf, __nv_bfloat16* out, int rows, int columns, int inner) {
  GemvArgs p{};
  p.A = a;
  p.A_sf = asf;
  p.B = w;
  p.B_sf = wsf;
  p.C = out;
  p.M = rows;
  p.N = columns;
  p.K = inner;
  p.num_k_tiles = inner / 128;
  p.ldc = columns;
  mxfp8_mma_gemv_v3_body<1, 1, 1>(p, SplitKArgs{nullptr, nullptr, 1});
}
}  // namespace vllm_moet_sm120
// clang-format on
