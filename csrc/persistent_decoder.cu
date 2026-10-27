// SPDX-License-Identifier: Apache-2.0
#include <flashinfer/experimental/persistent_decoder/decoder.cuh>

#include "tvm_ffi_utils.h"
using tvm::ffi::Array;
using tvm::ffi::Tensor;
using namespace flashinfer::persistent_decoder;

int64_t persistent_decoder_scratch_bytes(int64_t index) {
  switch (index) {
    case 0:
      return (1) * sizeof(unsigned);
    case 1:
      return (COUNTERS) * sizeof(unsigned);
    case 2:
      return (1) * sizeof(Table);
    case 3:
      return (long(MAXM) * OA) * sizeof(float);
    case 4:
      return (long(MAXM) * OA) * sizeof(uint8_t);
    case 5:
      return (long(MAXM) * 64) * sizeof(uint8_t);
    case 6:
      return (long(MAXM) * HID) * sizeof(float);
    case 7:
      return (long(MAXM) * HID) * sizeof(__nv_bfloat16);
    case 8:
      return (long(SLICES) * MAXM * MIXW) * sizeof(float);
    case 9:
      return (long(MAXM) * (MIX + 1)) * sizeof(float);
    case 10:
      return (long(MAXM) * HID) * sizeof(__nv_bfloat16);
    case 11:
      return (long(SLICES) * MAXM) * sizeof(float);
    case 12:
      return (long(MAXM) * HID) * sizeof(__nv_bfloat16);
    case 13:
      return (long(MAXM) * HID) * sizeof(uint8_t);
    case 14:
      return (long(MAXM) * SLICES) * sizeof(uint8_t);
    case 15:
      return (long(MAXM) * EXPERTS) * sizeof(long long);
    case 16:
      return (ROUTES) * sizeof(float);
    case 17:
      return (long(MAXM) * 2 * INTER) * sizeof(float);
    case 18:
      return (2L * MAXM * INTER) * sizeof(uint8_t);
    case 19:
      return (long(MAXM) * HQ_LD) * sizeof(uint8_t);
    case 20:
      return (long(MAXM) * HID) * sizeof(float);
    case 21:
      return (long(ROUTES) * 2 * INTER) * sizeof(float);
    case 22:
      return (long(ROUTES) * INTER) * sizeof(uint8_t);
    case 23:
      return (long(ROUTES) * HQ_LD) * sizeof(uint8_t);
    case 24:
      return (long(MAXM) * HID) * sizeof(float);
    case 25:
      return (long(MAXM) * HID) * sizeof(uint8_t);
    case 26:
      return (long(SLICES / 4) * 16 * MAXM) * sizeof(uint8_t);
    case 27:
      return (long(KSPLIT) * MAXM * QKV) * sizeof(float);
    case 28:
      return (long(MAXM) * QKV) * sizeof(float);
    case 29:
      return (long(QKV_TILES) * MAXM) * sizeof(float);
    case 30:
      return (QKV_TILES) * sizeof(unsigned);
    case 31:
      return (PCOUNTERS) * sizeof(unsigned);
    case 32:
      return (long(MAXM) * 2 * OAK) * sizeof(uint8_t);
    case 33:
      return (long(MAXM) * 2 * OAK / 32) * sizeof(uint8_t);
    default:
      TVM_FFI_ICHECK(false) << "invalid persistent decoder scratch index";
      return 0;
  }
}

int64_t persistent_decoder_ctas(int64_t device, int64_t blocks_per_sm) {
  TVM_FFI_ICHECK(blocks_per_sm == 1 || blocks_per_sm == 2);
  ffi::CUDADeviceGuard guard(device);
  int sms = 0, per = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) ==
                 cudaSuccess);
  TVM_FFI_ICHECK(cudaFuncSetAttribute(layer_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      sizeof(Smem)) == cudaSuccess);
  TVM_FFI_ICHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_kernel, THREADS,
                                                               sizeof(Smem)) == cudaSuccess);
  TVM_FFI_ICHECK(per >= blocks_per_sm) << "persistent decoder exceeds resident CTA capacity";
  return sms * blocks_per_sm;
}

void persistent_decoder_run(Array<Tensor> io, Array<Tensor> weights, Array<Tensor> scratch,
                            Array<Tensor> peers, int64_t rank, int64_t ctas, int64_t shared_a16) {
  TVM_FFI_ICHECK_EQ(io.size(), 13);
  TVM_FFI_ICHECK_EQ(weights.size(), 19);
  TVM_FFI_ICHECK_EQ(scratch.size(), 34);
  TVM_FFI_ICHECK_EQ(peers.size(), RANKS);
  TVM_FFI_ICHECK(rank >= 0 && rank < RANKS);
  Params p{};
  p.rank = rank;
  p.shared_a16 = shared_a16;
  TVM_FFI_ICHECK(shared_a16 >= 0 && shared_a16 <= 3);
  p.m = io[4].size(0);
  TVM_FFI_ICHECK(p.m > 0 && p.m <= MAXM);
  TVM_FFI_ICHECK(shared_a16 == 0 || p.m <= 8);
  p.o_stride = io[0].stride(0);
  p.ids64 = true;
  p.image_lo = 129280;
  p.attn_mix = false;
  p.o = reinterpret_cast<const __nv_bfloat16*>(io[0].data_ptr());
  p.positions = reinterpret_cast<const long*>(io[1].data_ptr());
  p.input_ids = reinterpret_cast<const long*>(io[2].data_ptr());
  p.cos_sin = reinterpret_cast<const float*>(io[3].data_ptr());
  p.resid0 = reinterpret_cast<const __nv_bfloat16*>(io[4].data_ptr());
  p.attn_post = reinterpret_cast<float*>(io[5].data_ptr());
  p.attn_comb = reinterpret_cast<float*>(io[6].data_ptr());
  p.attn_pre = reinterpret_cast<float*>(io[7].data_ptr());
  p.y_ffn = reinterpret_cast<__nv_bfloat16*>(io[8].data_ptr());
  p.resid1 = reinterpret_cast<__nv_bfloat16*>(io[9].data_ptr());
  p.ffn_post = reinterpret_cast<float*>(io[10].data_ptr());
  p.ffn_comb = reinterpret_cast<float*>(io[11].data_ptr());
  p.ffn_pre = reinterpret_cast<float*>(io[12].data_ptr());
  p.w.woa = weights[0].numel() ? reinterpret_cast<const uint8_t*>(weights[0].data_ptr()) : nullptr;
  p.w.woa_s =
      weights[1].numel() ? reinterpret_cast<const uint8_t*>(weights[1].data_ptr()) : nullptr;
  p.w.wob = weights[2].numel() ? reinterpret_cast<const uint8_t*>(weights[2].data_ptr()) : nullptr;
  p.w.wob_s =
      weights[3].numel() ? reinterpret_cast<const uint8_t*>(weights[3].data_ptr()) : nullptr;
  p.w.gate = weights[4].numel() ? reinterpret_cast<const uint8_t*>(weights[4].data_ptr()) : nullptr;
  p.w.gate_b = weights[5].numel() ? reinterpret_cast<const float*>(weights[5].data_ptr()) : nullptr;
  p.w.bias_vl =
      weights[6].numel() ? reinterpret_cast<const float*>(weights[6].data_ptr()) : nullptr;
  p.w.sh13 = weights[7].numel() ? reinterpret_cast<const uint8_t*>(weights[7].data_ptr()) : nullptr;
  p.w.sh13_s =
      weights[8].numel() ? reinterpret_cast<const uint8_t*>(weights[8].data_ptr()) : nullptr;
  p.w.sh2 = weights[9].numel() ? reinterpret_cast<const uint8_t*>(weights[9].data_ptr()) : nullptr;
  p.w.sh2_s =
      weights[10].numel() ? reinterpret_cast<const uint8_t*>(weights[10].data_ptr()) : nullptr;
  p.w.w13 = weights[11].numel() ? reinterpret_cast<const uint4*>(weights[11].data_ptr()) : nullptr;
  p.w.s13 =
      weights[12].numel() ? reinterpret_cast<const unsigned*>(weights[12].data_ptr()) : nullptr;
  p.w.w2 = weights[13].numel() ? reinterpret_cast<const uint4*>(weights[13].data_ptr()) : nullptr;
  p.w.s2 =
      weights[14].numel() ? reinterpret_cast<const unsigned*>(weights[14].data_ptr()) : nullptr;
  p.w.fn = weights[15].numel() ? reinterpret_cast<const float*>(weights[15].data_ptr()) : nullptr;
  p.w.hc_scale =
      weights[16].numel() ? reinterpret_cast<const float*>(weights[16].data_ptr()) : nullptr;
  p.w.hc_base =
      weights[17].numel() ? reinterpret_cast<const float*>(weights[17].data_ptr()) : nullptr;
  p.w.norm = weights[18].numel() ? reinterpret_cast<const __nv_bfloat16*>(weights[18].data_ptr())
                                 : nullptr;
  p.w.w13_stride = weights[11].stride(0) * weights[11].dtype().bits / 8 / sizeof(uint4);
  p.w.s13_stride = weights[12].stride(0) * weights[12].dtype().bits / 8 / sizeof(unsigned);
  p.w.w2_stride = weights[13].stride(0) * weights[13].dtype().bits / 8 / sizeof(uint4);
  p.w.s2_stride = weights[14].stride(0) * weights[14].dtype().bits / 8 / sizeof(unsigned);
  TVM_FFI_ICHECK_GE(scratch[0].numel(), persistent_decoder_scratch_bytes(0));
  p.s.epoch = reinterpret_cast<unsigned*>(scratch[0].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[1].numel(), persistent_decoder_scratch_bytes(1));
  p.s.counters = reinterpret_cast<unsigned*>(scratch[1].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[2].numel(), persistent_decoder_scratch_bytes(2));
  p.s.table = reinterpret_cast<Table*>(scratch[2].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[3].numel(), persistent_decoder_scratch_bytes(3));
  p.s.zacc = reinterpret_cast<float*>(scratch[3].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[4].numel(), persistent_decoder_scratch_bytes(4));
  p.s.zq = reinterpret_cast<uint8_t*>(scratch[4].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[5].numel(), persistent_decoder_scratch_bytes(5));
  p.s.zse = reinterpret_cast<uint8_t*>(scratch[5].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[6].numel(), persistent_decoder_scratch_bytes(6));
  p.s.yacc = reinterpret_cast<float*>(scratch[6].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[7].numel(), persistent_decoder_scratch_bytes(7));
  p.s.yattn = reinterpret_cast<__nv_bfloat16*>(scratch[7].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[8].numel(), persistent_decoder_scratch_bytes(8));
  p.s.mixp = reinterpret_cast<float*>(scratch[8].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[9].numel(), persistent_decoder_scratch_bytes(9));
  p.s.mixq = reinterpret_cast<float*>(scratch[9].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[10].numel(), persistent_decoder_scratch_bytes(10));
  p.s.xc = reinterpret_cast<__nv_bfloat16*>(scratch[10].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[11].numel(), persistent_decoder_scratch_bytes(11));
  p.s.xsq = reinterpret_cast<float*>(scratch[11].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[12].numel(), persistent_decoder_scratch_bytes(12));
  p.s.xbf = reinterpret_cast<__nv_bfloat16*>(scratch[12].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[13].numel(), persistent_decoder_scratch_bytes(13));
  p.s.xq = reinterpret_cast<uint8_t*>(scratch[13].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[14].numel(), persistent_decoder_scratch_bytes(14));
  p.s.xse = reinterpret_cast<uint8_t*>(scratch[14].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[15].numel(), persistent_decoder_scratch_bytes(15));
  p.s.gacc = reinterpret_cast<long long*>(scratch[15].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[16].numel(), persistent_decoder_scratch_bytes(16));
  p.s.topk_w = reinterpret_cast<float*>(scratch[16].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[17].numel(), persistent_decoder_scratch_bytes(17));
  p.s.sacc = reinterpret_cast<float*>(scratch[17].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[18].numel(), persistent_decoder_scratch_bytes(18));
  p.s.hsq = reinterpret_cast<uint8_t*>(scratch[18].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[19].numel(), persistent_decoder_scratch_bytes(19));
  p.s.hse = reinterpret_cast<uint8_t*>(scratch[19].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[20].numel(), persistent_decoder_scratch_bytes(20));
  p.s.shacc = reinterpret_cast<float*>(scratch[20].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[21].numel(), persistent_decoder_scratch_bytes(21));
  p.s.hacc = reinterpret_cast<float*>(scratch[21].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[22].numel(), persistent_decoder_scratch_bytes(22));
  p.s.hq = reinterpret_cast<uint8_t*>(scratch[22].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[23].numel(), persistent_decoder_scratch_bytes(23));
  p.s.hqe = reinterpret_cast<uint8_t*>(scratch[23].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[24].numel(), persistent_decoder_scratch_bytes(24));
  p.s.yracc = reinterpret_cast<float*>(scratch[24].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[25].numel(), persistent_decoder_scratch_bytes(25));
  p.s.aq = reinterpret_cast<uint8_t*>(scratch[25].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[26].numel(), persistent_decoder_scratch_bytes(26));
  p.s.as = reinterpret_cast<uint8_t*>(scratch[26].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[27].numel(), persistent_decoder_scratch_bytes(27));
  p.s.qkp = reinterpret_cast<float*>(scratch[27].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[28].numel(), persistent_decoder_scratch_bytes(28));
  p.s.qkv = reinterpret_cast<float*>(scratch[28].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[29].numel(), persistent_decoder_scratch_bytes(29));
  p.s.qsq = reinterpret_cast<float*>(scratch[29].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[30].numel(), persistent_decoder_scratch_bytes(30));
  p.s.tiles = reinterpret_cast<unsigned*>(scratch[30].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[31].numel(), persistent_decoder_scratch_bytes(31));
  p.s.pc = reinterpret_cast<unsigned*>(scratch[31].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[32].numel(), persistent_decoder_scratch_bytes(32));
  p.s.oq = reinterpret_cast<uint8_t*>(scratch[32].data_ptr());
  TVM_FFI_ICHECK_GE(scratch[33].numel(), persistent_decoder_scratch_bytes(33));
  p.s.ose = reinterpret_cast<uint8_t*>(scratch[33].data_ptr());
  for (int r = 0; r < RANKS; ++r) {
    TVM_FFI_ICHECK_GE(peers[r].numel(), FLAG_BYTES + 8L * PUB_SLOT * sizeof(__nv_bfloat16));
    auto* base = static_cast<uint8_t*>(peers[r].data_ptr());
    p.peers.flags[r] = reinterpret_cast<unsigned*>(base);
    p.peers.pub[r] = reinterpret_cast<__nv_bfloat16*>(base + FLAG_BYTES);
  }
  ffi::CUDADeviceGuard guard(io[0].device().device_id);
  auto stream = get_stream(io[0].device());
  quantize_woa_input<<<(p.m * 2 * OAK / 32 + 3) / 4, 128, 0, stream>>>(p);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = sizeof(Smem);
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  auto status = cudaLaunchKernelEx(&config, layer_kernel, p);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(scratch_bytes, persistent_decoder_scratch_bytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ctas, persistent_decoder_ctas);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, persistent_decoder_run);

// Own only the DLPack descriptor. The coordinated channel owns the CUDA mapping.
Tensor persistent_decoder_view(int64_t address, int64_t nbytes, int64_t device) {
  struct View {
    DLManagedTensor managed{};
    int64_t extent;
  };
  auto* view = new View;
  view->extent = nbytes;
  view->managed.dl_tensor = DLTensor{reinterpret_cast<void*>(address),
                                     {kDLCUDA, int(device)},
                                     1,
                                     dl_uint8,
                                     &view->extent,
                                     nullptr,
                                     0};
  view->managed.manager_ctx = view;
  view->managed.deleter = [](DLManagedTensor* tensor) {
    delete static_cast<View*>(tensor->manager_ctx);
  };
  return Tensor::FromDLPack(&view->managed);
}

void persistent_decoder_quantize(TensorView source, TensorView values, TensorView scales) {
  Params p{};
  p.m = source.size(0);
  TVM_FFI_ICHECK(p.m > 0 && p.m <= MAXM);
  p.o = static_cast<const __nv_bfloat16*>(source.data_ptr());
  p.o_stride = source.stride(0);
  p.s.oq = static_cast<uint8_t*>(values.data_ptr());
  p.s.ose = static_cast<uint8_t*>(scales.data_ptr());
  ffi::CUDADeviceGuard guard(source.device().device_id);
  quantize_woa_input<<<(p.m * 2 * OAK / 32 + 3) / 4, 128, 0, get_stream(source.device())>>>(p);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(view, persistent_decoder_view);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize, persistent_decoder_quantize);

int64_t persistent_prologue_ctas(int64_t device) {
  ffi::CUDADeviceGuard guard(device);
  int sms = 0, per = 0;
  TVM_FFI_ICHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) ==
                 cudaSuccess);
  TVM_FFI_ICHECK(cudaFuncSetAttribute(prologue_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      PRO_SMEM) == cudaSuccess);
  TVM_FFI_ICHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, prologue_kernel, GT,
                                                               PRO_SMEM) == cudaSuccess);
  TVM_FFI_ICHECK(sms * per >= QKV_CTAS && sms * per >= WQB_CTAS);
  return sms * per;
}

__global__ void persistent_attention_mix(Prologue a, const float* scale, const float* base,
                                         float* post, float* comb, float* pre) {
  const int tok = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (tok >= a.in.m) return;
  const float mine = lane <= MIX ? a.s.mixq[tok * (MIX + 1) + lane] : 0.0f;
  mix_finish(lane, tok, mine, scale, base, pre, post, comb);
}

void persistent_prologue_run(Array<Tensor> io, Array<Tensor> weights, Array<Tensor> scratch,
                             int64_t ctas) {
  TVM_FFI_ICHECK_EQ(io.size(), 15);
  TVM_FFI_ICHECK_EQ(weights.size(), 10);
  TVM_FFI_ICHECK_EQ(scratch.size(), 34);
  Prologue a{};
  a.in.m = io[1].size(0);
  TVM_FFI_ICHECK(a.in.m > 0 && a.in.m <= 8);
  a.in.y = static_cast<const __nv_bfloat16*>(io[0].data_ptr());
  a.in.resid = static_cast<const __nv_bfloat16*>(io[1].data_ptr());
  a.in.post = static_cast<const float*>(io[2].data_ptr());
  a.in.comb = static_cast<const float*>(io[3].data_ptr());
  a.in.pre = static_cast<const float*>(io[4].data_ptr());
  a.positions = static_cast<const long*>(io[5].data_ptr());
  a.cos_sin = static_cast<const float*>(io[6].data_ptr());
  a.in.resid_out = static_cast<__nv_bfloat16*>(io[7].data_ptr());
  a.in.input_out = static_cast<__nv_bfloat16*>(io[8].data_ptr());
  a.qr = static_cast<__nv_bfloat16*>(io[9].data_ptr());
  a.kv = static_cast<__nv_bfloat16*>(io[10].data_ptr());
  a.q = static_cast<__nv_bfloat16*>(io[11].data_ptr());
  a.in.fn = static_cast<const float*>(weights[0].data_ptr());
  a.in.norm = static_cast<const __nv_bfloat16*>(weights[1].data_ptr());
  a.wqa = static_cast<const uint8_t*>(weights[2].data_ptr());
  a.wqa_s = static_cast<const uint8_t*>(weights[3].data_ptr());
  a.q_norm = static_cast<const __nv_bfloat16*>(weights[4].data_ptr());
  a.kv_norm = static_cast<const __nv_bfloat16*>(weights[5].data_ptr());
  a.wqb = static_cast<const uint8_t*>(weights[6].data_ptr());
  a.wqb_s = static_cast<const uint8_t*>(weights[7].data_ptr());
  a.eps = 1e-20f;
  a.s.mixp = static_cast<float*>(scratch[8].data_ptr());
  a.s.mixq = static_cast<float*>(scratch[9].data_ptr());
  a.s.xc = static_cast<__nv_bfloat16*>(scratch[10].data_ptr());
  a.s.xsq = static_cast<float*>(scratch[11].data_ptr());
  a.s.aq = static_cast<uint8_t*>(scratch[25].data_ptr());
  a.s.as = static_cast<uint8_t*>(scratch[26].data_ptr());
  a.s.qkp = static_cast<float*>(scratch[27].data_ptr());
  a.s.qkv = static_cast<float*>(scratch[28].data_ptr());
  a.s.qsq = static_cast<float*>(scratch[29].data_ptr());
  a.s.tiles = static_cast<unsigned*>(scratch[30].data_ptr());
  a.s.pc = static_cast<unsigned*>(scratch[31].data_ptr());
  a.qr_q = static_cast<uint8_t*>(scratch[32].data_ptr());
  a.qr_s = static_cast<uint8_t*>(scratch[33].data_ptr());
  ffi::CUDADeviceGuard guard(io[0].device().device_id);
  auto stream = get_stream(io[0].device());
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(ctas);
  config.blockDim = dim3(GT);
  config.dynamicSmemBytes = PRO_SMEM;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  auto status = cudaLaunchKernelEx(&config, prologue_kernel, a);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  persistent_attention_mix<<<(a.in.m + 3) / 4, 128, 0, stream>>>(
      a, static_cast<const float*>(weights[8].data_ptr()),
      static_cast<const float*>(weights[9].data_ptr()), static_cast<float*>(io[12].data_ptr()),
      static_cast<float*>(io[13].data_ptr()), static_cast<float*>(io[14].data_ptr()));
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prologue_ctas, persistent_prologue_ctas);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(prologue, persistent_prologue_run);
