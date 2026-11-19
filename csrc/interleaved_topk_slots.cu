// SPDX-License-Identifier: Apache-2.0
#include <tvm/ffi/container/array.h>

#include <flashinfer/comm/interleaved_topk_slots.cuh>

#include "tvm_ffi_utils.h"

namespace {
using tvm::ffi::Array;
using namespace flashinfer::interleaved_topk;

void run(TensorView ids, TensorView scores, TensorView positions, TensorView tables,
         TensorView owners, TensorView slots, TensorView counts, Array<int64_t> stages,
         Array<int64_t> flags, int64_t bank_stride, int64_t rank, int64_t cache_pages) {
  CHECK_INPUT_AND_TYPE(ids, dl_int32);
  CHECK_INPUT_AND_TYPE(scores, dl_float32);
  CHECK_INPUT_AND_TYPE(positions, dl_int64);
  CHECK_CUDA(tables);
  CHECK_INPUT_TYPE(tables, dl_int32);
  CHECK_INPUT_AND_TYPE(owners, dl_int32);
  CHECK_INPUT_AND_TYPE(slots, dl_int32);
  CHECK_INPUT_AND_TYPE(counts, dl_int32);
  CHECK_DIM(2, ids);
  CHECK_DIM(1, positions);
  CHECK_DIM(1, counts);
  CHECK_DIM(2, tables);
  CHECK_SHAPE(ids, scores);
  CHECK_SHAPE(ids, owners);
  CHECK_SHAPE(ids, slots);
  for (auto tensor : {scores, positions, tables, owners, slots, counts}) {
    CHECK_DEVICE(ids, tensor);
  }
  TVM_FFI_ICHECK_EQ(ids.size(0), 4);
  TVM_FFI_ICHECK_EQ(ids.size(1), k);
  TVM_FFI_ICHECK_EQ(positions.size(0), 4);
  TVM_FFI_ICHECK_EQ(counts.size(0), 4);
  TVM_FFI_ICHECK_EQ(tables.size(0), 4);
  TVM_FFI_ICHECK_EQ(tables.stride(1), 1);
  // A single request broadcasts its page-table row to all four verification positions.
  TVM_FFI_ICHECK(tables.size(1) > 0 && tables.stride(0) >= 0);
  TVM_FFI_ICHECK(cache_pages > 0 && cache_pages <= INT_MAX / 128);
  TVM_FFI_ICHECK(rank >= 0 && rank < world);
  TVM_FFI_ICHECK_EQ(stages.size(), world);
  TVM_FFI_ICHECK_EQ(flags.size(), world);
  constexpr int64_t min_bank = 4 * sampled::stage_items * sizeof(sampled::Key);
  TVM_FFI_ICHECK(bank_stride >= min_bank || bank_stride <= -min_bank);
  Peers peers{};
  for (int peer = 0; peer < world; ++peer) {
    TVM_FFI_ICHECK(stages[peer] && (stages[peer] % 8 == 0) && flags[peer] &&
                   (flags[peer] % 8 == 0));
    peers.stage[peer] = reinterpret_cast<unsigned char*>(stages[peer]);
    peers.flags[peer] = reinterpret_cast<unsigned*>(flags[peer]);
  }
  ffi::CUDADeviceGuard guard(ids.device().device_id);
  local_candidate_slots<<<4, threads, 0, get_stream(ids.device())>>>(
      static_cast<const int*>(ids.data_ptr()), static_cast<const unsigned*>(scores.data_ptr()),
      static_cast<const long long*>(positions.data_ptr()),
      static_cast<const int*>(tables.data_ptr()), static_cast<int*>(owners.data_ptr()),
      static_cast<int*>(slots.data_ptr()), static_cast<int*>(counts.data_ptr()), peers, bank_stride,
      rank, tables.stride(0), tables.size(1), cache_pages);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK_EQ(status, cudaSuccess)
      << "interleaved top-k slot selection: " << cudaGetErrorString(status);
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);
