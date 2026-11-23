"""Ultra-fast 1-sync CuTeDSL outer boundary kernels for Kimi EXL3 2-bit MoE.

Implements the shared outer Hadamard boundaries for ``intermediate_hadamard=True``
with shared ``suh`` / ``down_svh`` vectors (``H % 512 == 0``, e.g., Kimi H=3584):

  1) ``Exl3InputBoundaryKernel``:
       ``x [M, H] (bf16) -> fp16 -> H512 -> * suh -> H128``
       - ``mode="fp16"``: writes ``out_fp16 [M, H]`` (unquantized W2A16 decode)
       - ``mode="bf16"``: writes ``out_bf16 [M, H]``
       - ``mode="mxfp8"``: quantizes each K32 block in-register with
         ``_W4A8_TRELLIS_A_K32_PERM`` and UE8M0 block scale, writing directly
         to ``packed_a [M, H] (u8)`` and ``scale_storage [M, H // 32] (u8)``
         for W2A8-MXFP8 (``w4a8_trellis``) with ZERO extra passes!

  2) ``Exl3OutputBoundaryKernel``:
       ``y_rot [M, H] (bf16/f32) -> H128 -> * down_svh -> H512 -> out [M, H] (bf16)``

Each CTA owns one 512-element chunk using 4 warps (128 threads), performing
all H128 transforms via warp butterfly shuffles in registers and needing only
ONE ``sync_threads()`` for the cross-warp H4 (512 = 4 x 128) step!
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_f32x4_to_e4m3x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _W4A8_TRELLIS_A_K32_PERM,
    _w4a8_had128_quad,
)

# Inverse permutation: for source element i in [0..31], dst_pos[i] is where it
# lands in the permuted K32 output (so out[k] = in[PERM[k]]).
_INV_K32_PERM = tuple(
    _W4A8_TRELLIS_A_K32_PERM.index(i) for i in range(32)
)


@cute.jit
def _had4_norm(a: cutlass.Float32, b: cutlass.Float32, c: cutlass.Float32, d: cutlass.Float32):
    s0 = a + b
    d0 = a - b
    s1 = c + d
    d1 = c - d
    half = cutlass.Float32(0.5)
    return (s0 + s1) * half, (d0 + d1) * half, (s0 - s1) * half, (d0 - d1) * half


class Exl3InputBoundaryKernel:
    """1-launch H512 -> * suh -> H128 -> (fp16 | bf16 | mxfp8-trellis-k32)."""

    def __init__(self, *, hidden_size: int, mode: str = "mxfp8"):
        if hidden_size <= 0 or hidden_size % 512 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 512, got {hidden_size}")
        if mode not in ("mxfp8", "fp16", "bf16"):
            raise ValueError(f"unsupported mode {mode!r}")
        self.hidden_size = int(hidden_size)
        self.chunks_per_row = self.hidden_size // 512
        self.mode = mode

    @cute.jit
    def __call__(
        self,
        x_in: cute.Tensor,         # [M, H] bf16
        suh: cute.Tensor,          # [H] fp16
        out_u32: cute.Tensor,      # flat u32 view of packed_a [M*H//4] (mxfp8) or out [M*H//2] (fp16/bf16)
        out_scales: cute.Tensor,   # [M, H // 32] u8 (used when mode == "mxfp8")
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        grid_x = num_tokens * Int32(self.chunks_per_row)
        self.kernel(x_in, suh, out_u32, out_scales, num_tokens).launch(
            grid=(grid_x, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_in: cute.Tensor,
        suh: cute.Tensor,
        out_u32: cute.Tensor,
        out_scales: cute.Tensor,
        num_tokens: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        unit = Int32(bidx)
        token = unit // Int32(self.chunks_per_row)
        chunk = unit - token * Int32(self.chunks_per_row)

        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(512), byte_alignment=16)

        col_chunk = chunk * Int32(512)
        col_warp = col_chunk + warp * Int32(128) + lane * Int32(4)

        # Load 4 elements per lane -> round via FP16 (EXL3 contract) -> H128
        x0 = cutlass.Float16(x_in[token, col_warp + Int32(0)].to(cutlass.Float32)).to(cutlass.Float32)
        x1 = cutlass.Float16(x_in[token, col_warp + Int32(1)].to(cutlass.Float32)).to(cutlass.Float32)
        x2 = cutlass.Float16(x_in[token, col_warp + Int32(2)].to(cutlass.Float32)).to(cutlass.Float32)
        x3 = cutlass.Float16(x_in[token, col_warp + Int32(3)].to(cutlass.Float32)).to(cutlass.Float32)
        h0, h1, h2, h3 = _w4a8_had128_quad(x0, x1, x2, x3, lane)

        s_off = warp * Int32(128) + lane * Int32(4)
        buf[s_off + Int32(0)] = h0
        buf[s_off + Int32(1)] = h1
        buf[s_off + Int32(2)] = h2
        buf[s_off + Int32(3)] = h3
        cute.arch.sync_threads()

        # Cross-warp H4 over the 4 H128 blocks -> completes H512!
        # Each warp computes its own sub-block's H4 linear combination directly
        # without a second SMEM round-trip!
        l_off = lane * Int32(4)
        a0, a1, a2, a3 = buf[l_off], buf[l_off + Int32(1)], buf[l_off + Int32(2)], buf[l_off + Int32(3)]
        b0, b1, b2, b3 = buf[Int32(128) + l_off], buf[Int32(128) + l_off + Int32(1)], buf[Int32(128) + l_off + Int32(2)], buf[Int32(128) + l_off + Int32(3)]
        c0, c1, c2, c3 = buf[Int32(256) + l_off], buf[Int32(256) + l_off + Int32(1)], buf[Int32(256) + l_off + Int32(2)], buf[Int32(256) + l_off + Int32(3)]
        d0, d1, d2, d3 = buf[Int32(384) + l_off], buf[Int32(384) + l_off + Int32(1)], buf[Int32(384) + l_off + Int32(2)], buf[Int32(384) + l_off + Int32(3)]

        # Select this warp's H4 output row (0: ++++, 1: +-+-, 2: ++--, 3: +--+)
        half = cutlass.Float32(0.5)
        sign_b = cutlass.Float32(-1.0) if (warp & Int32(1)) != Int32(0) else cutlass.Float32(1.0)
        sign_c = cutlass.Float32(-1.0) if (warp & Int32(2)) != Int32(0) else cutlass.Float32(1.0)
        sign_d = sign_b * sign_c
        v0 = (a0 + sign_b * b0 + sign_c * c0 + sign_d * d0) * half
        v1 = (a1 + sign_b * b1 + sign_c * c1 + sign_d * d1) * half
        v2 = (a2 + sign_b * b2 + sign_c * c2 + sign_d * d2) * half
        v3 = (a3 + sign_b * b3 + sign_c * c3 + sign_d * d3) * half

        # Multiply by shared suh (fp16) -> second H128 in registers!
        v0 = v0 * cutlass.Float32(suh[col_warp + Int32(0)])
        v1 = v1 * cutlass.Float32(suh[col_warp + Int32(1)])
        v2 = v2 * cutlass.Float32(suh[col_warp + Int32(2)])
        v3 = v3 * cutlass.Float32(suh[col_warp + Int32(3)])
        if cutlass.const_expr(self.mode == "fp16"):
            v0 = cutlass.Float16(v0).to(cutlass.Float32)
            v1 = cutlass.Float16(v1).to(cutlass.Float32)
            v2 = cutlass.Float16(v2).to(cutlass.Float32)
            v3 = cutlass.Float16(v3).to(cutlass.Float32)
        o0, o1, o2, o3 = _w4a8_had128_quad(v0, v1, v2, v3, lane)

        if cutlass.const_expr(self.mode == "mxfp8"):
            # 8 lanes own one K32 block (lanes 0..7, 8..15, 16..23, 24..31).
            # Compute exact block_max across 8 lanes via 3 butterfly shuffles.
            m01 = fmax_f32(fabs_f32(o0), fabs_f32(o1))
            m23 = fmax_f32(fabs_f32(o2), fabs_f32(o3))
            bmax = fmax_f32(m01, m23)
            bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=1))
            bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=2))
            bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=4))
            _, scale_byte = pow2_ceil_ue8m0(bmax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX))
            inv_scale = ue8m0_to_output_scale(scale_byte)

            # Reuse buf[s_off..] (needs 1 sync so all warps finished reading H512 buf!)
            cute.arch.sync_threads()
            k32_local = lane >> Int32(3)       # 0..3 within warp's 128
            sub_lane = lane & Int32(7)         # 0..7 within K32
            k32_smem_base = warp * Int32(128) + k32_local * Int32(32)
            buf[k32_smem_base + sub_lane * Int32(4) + Int32(0)] = o0 * inv_scale
            buf[k32_smem_base + sub_lane * Int32(4) + Int32(1)] = o1 * inv_scale
            buf[k32_smem_base + sub_lane * Int32(4) + Int32(2)] = o2 * inv_scale
            buf[k32_smem_base + sub_lane * Int32(4) + Int32(3)] = o3 * inv_scale
            cute.arch.sync_warp()

            # Read permuted 4-tuple for this sub_lane!
            # sub_lane s owns output positions [4*s, 4*s+1, 4*s+2, 4*s+3].
            # Note from _W4A8_TRELLIS_A_K32_PERM:
            #   s=0: (0,1,8,9), s=1: (4,5,12,13), s=2: (2,3,10,11), s=3: (6,7,14,15)
            #   s=4: (20,21,28,29), s=5: (16,17,24,25), s=6: (22,23,30,31), s=7: (18,19,26,27)
            # Closed-form formula (ZERO table lookup!):
            #   half16 = (sub_lane >> 2) * 16
            #   base_in_16 = (((sub_lane & 1) << 2) | ((sub_lane & 2) >> 1) << 1) ^ ((sub_lane & 4))
            # Let's verify:
            #   s=0: (0<<2)|(0<<1)=0 ^ 0 = 0  -> (0,1,8,9)
            #   s=1: (1<<2)|(0<<1)=4 ^ 0 = 4  -> (4,5,12,13)
            #   s=2: (0<<2)|(1<<1)=2 ^ 0 = 2  -> (2,3,10,11)
            #   s=3: (1<<2)|(1<<1)=6 ^ 0 = 6  -> (6,7,14,15)
            #   s=4: (0<<2)|(0<<1)=0 ^ 4 = 4  -> 16+4=20 -> (20,21,28,29)
            #   s=5: (1<<2)|(0<<1)=4 ^ 4 = 0  -> 16+0=16 -> (16,17,24,25)
            #   s=6: (0<<2)|(1<<1)=2 ^ 4 = 6  -> 16+6=22 -> (22,23,30,31)
            #   s=7: (1<<2)|(1<<1)=6 ^ 4 = 2  -> 16+2=18 -> (18,19,26,27)
            # EXACT MATCH FOR ALL 32 ELEMENTS!!
            p_base = (
                ((sub_lane >> Int32(2)) << Int32(4))
                + ((((sub_lane & Int32(1)) << Int32(2)) | (sub_lane & Int32(2))) ^ (sub_lane & Int32(4)))
            )
            q0 = buf[k32_smem_base + p_base + Int32(0)]
            q1 = buf[k32_smem_base + p_base + Int32(1)]
            q2 = buf[k32_smem_base + p_base + Int32(8)]
            q3 = buf[k32_smem_base + p_base + Int32(9)]
            packed = cvt_f32x4_to_e4m3x4(q0, q1, q2, q3)
            word_idx = (
                Int64(token) * Int64(self.hidden_size // 4)
                + Int64(col_warp >> Int32(2))
            )
            out_u32[word_idx] = packed
            if sub_lane == Int32(0):
                k32_global = (col_chunk >> Int32(5)) + warp * Int32(4) + k32_local
                out_scales[token, k32_global] = Uint8(scale_byte & Uint32(0xFF))
        else:
            from b12x._lib.intrinsics.conversion import pack_f32x2_to_f16x2
            from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2

            word_idx = (
                Int64(token) * Int64(self.hidden_size // 2)
                + Int64(col_warp >> Int32(1))
            )
            if cutlass.const_expr(self.mode == "fp16"):
                out_u32[word_idx] = pack_f32x2_to_f16x2(o0, o1)
                out_u32[word_idx + Int64(1)] = pack_f32x2_to_f16x2(o2, o3)
            else:
                out_u32[word_idx] = pack_f32x2_to_bfloat2(o0, o1)
                out_u32[word_idx + Int64(1)] = pack_f32x2_to_bfloat2(o2, o3)


class Exl3OutputBoundaryKernel:
    """1-launch H128 -> * down_svh -> H512 -> (bf16 | f32) output stage (1 sync/CTA)."""

    def __init__(self, *, hidden_size: int, out_dtype: str = "bf16"):
        if hidden_size <= 0 or hidden_size % 512 != 0:
            raise ValueError(f"hidden_size must be a positive multiple of 512, got {hidden_size}")
        if out_dtype not in ("bf16", "f32"):
            raise ValueError(f"unsupported out_dtype {out_dtype!r}")
        self.hidden_size = int(hidden_size)
        self.chunks_per_row = self.hidden_size // 512
        self.out_dtype = out_dtype

    @cute.jit
    def __call__(
        self,
        y_rot: cute.Tensor,     # [M, H] bf16 or f32
        down_svh: cute.Tensor,  # [H] fp16
        out_buf: cute.Tensor,   # flat u32 view of out_bf16 [M*H//2] if bf16 else [M, H] f32
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        grid_x = num_tokens * Int32(self.chunks_per_row)
        self.kernel(y_rot, down_svh, out_buf, num_tokens).launch(
            grid=(grid_x, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        y_rot: cute.Tensor,
        down_svh: cute.Tensor,
        out_buf: cute.Tensor,
        num_tokens: cutlass.Int32,
    ):
        from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        unit = Int32(bidx)
        token = unit // Int32(self.chunks_per_row)
        chunk = unit - token * Int32(self.chunks_per_row)

        smem = cutlass.utils.SmemAllocator()
        buf = smem.allocate_tensor(cutlass.Float32, cute.make_layout(512), byte_alignment=16)

        col_chunk = chunk * Int32(512)
        col_warp = col_chunk + warp * Int32(128) + lane * Int32(4)

        y0 = y_rot[token, col_warp + Int32(0)].to(cutlass.Float32)
        y1 = y_rot[token, col_warp + Int32(1)].to(cutlass.Float32)
        y2 = y_rot[token, col_warp + Int32(2)].to(cutlass.Float32)
        y3 = y_rot[token, col_warp + Int32(3)].to(cutlass.Float32)

        # H128 -> * down_svh -> H128 (all in registers!)
        h0, h1, h2, h3 = _w4a8_had128_quad(y0, y1, y2, y3, lane)
        h0 = h0 * cutlass.Float32(down_svh[col_warp + Int32(0)])
        h1 = h1 * cutlass.Float32(down_svh[col_warp + Int32(1)])
        h2 = h2 * cutlass.Float32(down_svh[col_warp + Int32(2)])
        h3 = h3 * cutlass.Float32(down_svh[col_warp + Int32(3)])
        h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, lane)

        s_off = warp * Int32(128) + lane * Int32(4)
        buf[s_off + Int32(0)] = h0
        buf[s_off + Int32(1)] = h1
        buf[s_off + Int32(2)] = h2
        buf[s_off + Int32(3)] = h3
        cute.arch.sync_threads()

        # Cross-warp H4 completes H512 -> packed BF16 or direct FP32 store
        l_off = lane * Int32(4)
        a0, a1, a2, a3 = buf[l_off], buf[l_off + Int32(1)], buf[l_off + Int32(2)], buf[l_off + Int32(3)]
        b0, b1, b2, b3 = buf[Int32(128) + l_off], buf[Int32(128) + l_off + Int32(1)], buf[Int32(128) + l_off + Int32(2)], buf[Int32(128) + l_off + Int32(3)]
        c0, c1, c2, c3 = buf[Int32(256) + l_off], buf[Int32(256) + l_off + Int32(1)], buf[Int32(256) + l_off + Int32(2)], buf[Int32(256) + l_off + Int32(3)]
        d0, d1, d2, d3 = buf[Int32(384) + l_off], buf[Int32(384) + l_off + Int32(1)], buf[Int32(384) + l_off + Int32(2)], buf[Int32(384) + l_off + Int32(3)]

        half = cutlass.Float32(0.5)
        sign_b = cutlass.Float32(-1.0) if (warp & Int32(1)) != Int32(0) else cutlass.Float32(1.0)
        sign_c = cutlass.Float32(-1.0) if (warp & Int32(2)) != Int32(0) else cutlass.Float32(1.0)
        sign_d = sign_b * sign_c
        o0 = (a0 + sign_b * b0 + sign_c * c0 + sign_d * d0) * half
        o1 = (a1 + sign_b * b1 + sign_c * c1 + sign_d * d1) * half
        o2 = (a2 + sign_b * b2 + sign_c * c2 + sign_d * d2) * half
        o3 = (a3 + sign_b * b3 + sign_c * c3 + sign_d * d3) * half

        if cutlass.const_expr(self.out_dtype == "bf16"):
            word_idx = (
                Int64(token) * Int64(self.hidden_size // 2)
                + Int64(col_warp >> Int32(1))
            )
            out_buf[word_idx] = pack_f32x2_to_bfloat2(o0, o1)
            out_buf[word_idx + Int64(1)] = pack_f32x2_to_bfloat2(o2, o3)
        else:
            out_buf[token, col_warp + Int32(0)] = o0
            out_buf[token, col_warp + Int32(1)] = o1
            out_buf[token, col_warp + Int32(2)] = o2
            out_buf[token, col_warp + Int32(3)] = o3


__all__ = ["Exl3InputBoundaryKernel", "Exl3OutputBoundaryKernel"]
