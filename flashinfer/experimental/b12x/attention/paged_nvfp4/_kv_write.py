# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120 vectorized paged NVFP4 (4-or-6 MAE K, max/6 V) K/V writer.

Each 16-element channel block is owned by one thread (two 16-byte BF16 loads,
zero warp shuffles, zero shared memory). For Hkv=8, D=128 there are 64 K blocks
and 64 V blocks = 128 threads = one pass per token.

Layouts:
  - NVFP4 data tensor : ``[pages, page_size, Hkv, D // 2]`` uint8 (even col in
    low nibble, odd col in high nibble).
  - NVFP4 scale tensor: ``[pages, page_size, Hkv, D // 16]`` uint8 (raw E4M3 bits).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.intrinsics.conversion import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    f16x2_to_f32x2,
    fp4_decode_4bytes,
    fp8_e4m3_to_f32,
    fp8_e4m3_to_f32_and_rcp,
)
from b12x._lib.intrinsics.math import fabs_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_u8,
    st_global_v2_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.packed import (
    bfloat2_habs2,
    bfloat2_hmax2,
    bfloat2_hmax_to_f32,
)

KV_WRITE_THREADS = 128


@cute.jit
def _unpack_bf16x2(word: Uint32):
    return u32_as_f32(word << Uint32(16)), u32_as_f32(word & Uint32(0xFFFF0000))


@cute.jit
def _quant_block16_nvfp4(xs, amax: Float32, denom: Float32):
    """Quantize 16 FP32 values using block scale = ``e4m3(amax * denom)``."""
    sf_u32 = cvt_f32_to_e4m3(amax * denom)
    # Prevent underflow to 0 on tiny nonzero blocks (smallest positive E4M3 subnormal is 0x01).
    if amax > Float32(0.0) and (sf_u32 & Uint32(0x7F)) == Uint32(0):
        sf_u32 = Uint32(1)
    inv_sf = fp8_e4m3_to_f32_and_rcp(sf_u32)
    w0 = cvt_e2m1x8_f32(
        xs[0] * inv_sf,
        xs[1] * inv_sf,
        xs[2] * inv_sf,
        xs[3] * inv_sf,
        xs[4] * inv_sf,
        xs[5] * inv_sf,
        xs[6] * inv_sf,
        xs[7] * inv_sf,
    )
    w1 = cvt_e2m1x8_f32(
        xs[8] * inv_sf,
        xs[9] * inv_sf,
        xs[10] * inv_sf,
        xs[11] * inv_sf,
        xs[12] * inv_sf,
        xs[13] * inv_sf,
        xs[14] * inv_sf,
        xs[15] * inv_sf,
    )
    return w0, w1, sf_u32


@cute.jit
def _block16_mae(xs, w0: Uint32, w1: Uint32, sf_u32: Uint32) -> Float32:
    """Compute exact 16-element L1 reconstruction error using ``fp4_decode_4bytes``."""
    sf = fp8_e4m3_to_f32(sf_u32)
    h0, h1, h2, h3 = fp4_decode_4bytes(w0)
    h4, h5, h6, h7 = fp4_decode_4bytes(w1)
    err = Float32(0.0)
    pairs = (h0, h1, h2, h3, h4, h5, h6, h7)
    for i in cutlass.range_constexpr(8):
        d0, d1 = f16x2_to_f32x2(pairs[i])
        err = err + fabs_f32(xs[2 * i] - d0 * sf) + fabs_f32(xs[2 * i + 1] - d1 * sf)
    return err


class PagedNVFP4KVWriteKernel:
    """One CTA per token; each thread quantizes one 16-element block of K (4o6 MAE) or V (max/6)."""

    def __init__(
        self,
        *,
        num_kv_heads: int,
        head_dim_k: int,
        head_dim_v: int,
        page_size: int,
    ):
        if head_dim_k % 16 or head_dim_v % 16:
            raise ValueError("head_dim_k and head_dim_v must be multiples of 16")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim_k = int(head_dim_k)
        self.head_dim_v = int(head_dim_v)
        self.page_size = int(page_size)
        self.k_blocks = self.head_dim_k // 16
        self.v_blocks = self.head_dim_v // 16
        self.blocks_per_head = self.k_blocks + self.v_blocks
        self.total_blocks = self.num_kv_heads * self.blocks_per_head
        self.iters = -(-self.total_blocks // KV_WRITE_THREADS)

    @cute.jit
    def __call__(
        self,
        key: cute.Tensor,  # [T, Hkv, Dk] bf16
        value: cute.Tensor,  # [T, Hkv, Dv] bf16
        k_cache: cute.Tensor,  # [pages, P, Hkv, Dk//2] u8
        k_scales: cute.Tensor,  # [pages, P, Hkv, Dk//16] u8
        v_cache: cute.Tensor,  # [pages, P, Hkv, Dv//2] u8
        v_scales: cute.Tensor,  # [pages, P, Hkv, Dv//16] u8
        slot_mapping: cute.Tensor,  # [T] i64
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            key, value, k_cache, k_scales, v_cache, v_scales, slot_mapping
        ).launch(
            grid=(num_tokens, 1, 1),
            block=[KV_WRITE_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        key: cute.Tensor,
        value: cute.Tensor,
        k_cache: cute.Tensor,
        k_scales: cute.Tensor,
        v_cache: cute.Tensor,
        v_scales: cute.Tensor,
        slot_mapping: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, _ = cute.arch.block_idx()
        token = Int64(token_idx)
        slot = Int64(slot_mapping[token_idx])
        capacity = Int64(k_cache.shape[0]) * Int64(self.page_size)
        if slot >= Int64(0) and slot < capacity:
            page = slot // Int64(self.page_size)
            offset = slot - page * Int64(self.page_size)

            k_src_base = get_ptr_as_int64(key, 0) + token * Int64(
                key.stride[0]
            ) * Int64(2)
            v_src_base = get_ptr_as_int64(value, 0) + token * Int64(
                value.stride[0]
            ) * Int64(2)
            k_dst_base = (
                get_ptr_as_int64(k_cache, 0)
                + page * Int64(k_cache.stride[0])
                + offset * Int64(k_cache.stride[1])
            )
            ks_dst_base = (
                get_ptr_as_int64(k_scales, 0)
                + page * Int64(k_scales.stride[0])
                + offset * Int64(k_scales.stride[1])
            )
            v_dst_base = (
                get_ptr_as_int64(v_cache, 0)
                + page * Int64(v_cache.stride[0])
                + offset * Int64(v_cache.stride[1])
            )
            vs_dst_base = (
                get_ptr_as_int64(v_scales, 0)
                + page * Int64(v_scales.stride[0])
                + offset * Int64(v_scales.stride[1])
            )

            for it in cutlass.range_constexpr(self.iters):
                bidx = Int32(tidx) + Int32(it * KV_WRITE_THREADS)
                if bidx < Int32(self.total_blocks):
                    head = bidx // Int32(self.blocks_per_head)
                    j = bidx - head * Int32(self.blocks_per_head)
                    is_k = j < Int32(self.k_blocks)
                    blk = cutlass.select_(is_k, j, j - Int32(self.k_blocks))
                    col = Int64(blk) * Int64(16)
                    head64 = Int64(head)
                    src = cutlass.select_(
                        is_k,
                        k_src_base + (head64 * Int64(key.stride[1]) + col) * Int64(2),
                        v_src_base + (head64 * Int64(value.stride[1]) + col) * Int64(2),
                    )
                    a0, a1, a2, a3 = ld_global_v4_u32(src)
                    b0, b1, b2, b3 = ld_global_v4_u32(src + Int64(16))

                    m01 = bfloat2_hmax2(bfloat2_habs2(a0), bfloat2_habs2(a1))
                    m23 = bfloat2_hmax2(bfloat2_habs2(a2), bfloat2_habs2(a3))
                    m45 = bfloat2_hmax2(bfloat2_habs2(b0), bfloat2_habs2(b1))
                    m67 = bfloat2_hmax2(bfloat2_habs2(b2), bfloat2_habs2(b3))
                    amax = bfloat2_hmax_to_f32(
                        bfloat2_hmax2(bfloat2_hmax2(m01, m23), bfloat2_hmax2(m45, m67))
                    )

                    x0, x1 = _unpack_bf16x2(a0)
                    x2, x3 = _unpack_bf16x2(a1)
                    x4, x5 = _unpack_bf16x2(a2)
                    x6, x7 = _unpack_bf16x2(a3)
                    x8, x9 = _unpack_bf16x2(b0)
                    x10, x11 = _unpack_bf16x2(b1)
                    x12, x13 = _unpack_bf16x2(b2)
                    x14, x15 = _unpack_bf16x2(b3)
                    xs = (
                        x0,
                        x1,
                        x2,
                        x3,
                        x4,
                        x5,
                        x6,
                        x7,
                        x8,
                        x9,
                        x10,
                        x11,
                        x12,
                        x13,
                        x14,
                        x15,
                    )

                    w0, w1, sf_u32 = _quant_block16_nvfp4(xs, amax, Float32(1.0 / 6.0))
                    if is_k:
                        w0_4, w1_4, sf_4 = _quant_block16_nvfp4(
                            xs, amax, Float32(1.0 / 4.0)
                        )
                        use4 = _block16_mae(xs, w0_4, w1_4, sf_4) < _block16_mae(
                            xs, w0, w1, sf_u32
                        )
                        w0 = Uint32(cutlass.select_(use4, w0_4, w0))
                        w1 = Uint32(cutlass.select_(use4, w1_4, w1))
                        sf_u32 = Uint32(cutlass.select_(use4, sf_4, sf_u32))

                    dst_data = cutlass.select_(
                        is_k,
                        k_dst_base
                        + head64 * Int64(k_cache.stride[2])
                        + Int64(blk) * Int64(8),
                        v_dst_base
                        + head64 * Int64(v_cache.stride[2])
                        + Int64(blk) * Int64(8),
                    )
                    dst_sf = cutlass.select_(
                        is_k,
                        ks_dst_base + head64 * Int64(k_scales.stride[2]) + Int64(blk),
                        vs_dst_base + head64 * Int64(v_scales.stride[2]) + Int64(blk),
                    )
                    st_global_v2_u32(dst_data, w0, w1)
                    st_global_u8(dst_sf, Uint8(sf_u32))


__all__ = ["KV_WRITE_THREADS", "PagedNVFP4KVWriteKernel"]
