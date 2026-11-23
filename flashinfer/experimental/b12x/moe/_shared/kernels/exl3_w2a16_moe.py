"""Hardware-roofline Lossless W2A16 (FP16 Tensor Core) EXL3 2-bit MoE kernels.

Zero-B-SMEM, 64 KiB SMEM-Direct-LUT architecture sharing the exact canonical
``trellis_t256`` weight allocation and 1-sync ``Exl3InputBoundaryKernel(mode="fp16")``
/ ``Exl3OutputBoundaryKernel`` boundary stages.

Two specialized engines cover the entire token-count spectrum at SM120 roofline:

1. **Decode Engine (M = 1..32, block_size = 8, 8 warps = 256 threads/CTA, 0 spills)**:
   - ``Exl3W2A16DecodeFC1Kernel8W``: 4N x 2K (N128, K128), transposed
     ``f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32``, 255-reg/thread headroom (0 spills),
     fast 8-warp barriers, dynamic Split-K (``split_k=3`` at M=1 -> 144 CTAs = 1 full wave!),
     fused SiTU+Hadamard epilogue when ``split_k == 1``, and zero-route-pack ``direct_m1``.
   - ``Exl3ActHadF16Kernel``: lightweight warp-per-(route, block) activation+Hadamard for ``split_k > 1``.
   - ``Exl3W2A16DecodeFC2Kernel8W``: **8N x 1K (N256, K128)** — ZERO K-split across warps,
     ZERO K-loop barriers (preloads all <=3 K128 slices in 1 ``cp.async``), **ZERO epilogue SMEM
     and ZERO epilogue barriers** (4-instruction ``shfl.bfly(4)`` register transpose directly into
     coalesced ``red.relaxed.gpu.global.add.v2.f32``)!

2. **Prefill Engine (M = 64..2048, block_size = 64, 16 warps = 512 threads/CTA)**:
   - ``Exl3W2A16PrefillFC1Kernel16W``: 8N x 2K (N128, K64, 3-stage FP16 A), bank-free
     ``ldmatrix.m8n8.x4``, writing **contiguous physical-row** ``intermediate_f16``.
   - ``Exl3W2A16PrefillFC2Kernel16W``: 8N x 2K (N256, K64, 4-stage FP16 A), contiguous
     physical-row reads, ``preload_all`` for ``compute_m <= 32``, bank-free K-reduce
     + ``scatter_add_bf16x2``.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics.atomics import (
    ld_shared_f32,
    ld_shared_i32_relaxed,
    ld_shared_v4_f32,
    scatter_add_bf16x2,
    st_shared_f32,
    st_shared_i32,
)
from b12x._lib.intrinsics.conversion import pack_f32x2_to_f16x2
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    get_ptr_as_int64,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
)
from b12x._lib.intrinsics.mma import (
    f16_mma_m16n8k16_f32,
    f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32,
)
from b12x._lib.intrinsics.packed import fp8x4_e4m3_to_half2x2
from b12x._lib.intrinsics.trellis import packed_decode_lut_e4m3_direct_to_e4m3x8
from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.exl3_mxfp8_moe import (
    _ld_shared_v2_f32,
    _load_direct_lut_smem,
    _load_trellis_k32_raw_u32,
    _red_add_global_v2_f32,
    _st_shared_v2_f32,
    _warp_sum_i32,
)
from b12x.moe._shared.kernels.trellis_ring import trellis256_lane_geom_bits
from b12x.moe._shared.kernels.w4a8_trellis_decode import _w4a8_had128_quad


@cute.jit
def _decode_trellis_k16_to_f16x4(
    raw_b: Uint32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    lut_addr: Int64,
):
    a = Uint32(cute.arch.shuffle_sync(raw_b, ia))
    b = Uint32(cute.arch.shuffle_sync(raw_b, ib))
    m = (Int64(a) << Int64(32)) | Int64(b)
    wa = Uint32(m >> Int64(s2))
    wb = Uint32(m >> Int64(s2 + Int32(8)))
    e_lo, e_hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
        wa, wb, lut_addr, 2, rate_indexed=False, in_shared=True
    )
    o0, o1 = fp8x4_e4m3_to_half2x2(e_lo)
    o2, o3 = fp8x4_e4m3_to_half2x2(e_hi)
    return o0, o1, o2, o3


@cute.jit
def _activated_f32(
    gate: Float32,
    up: Float32,
    is_situ: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
) -> Float32:
    sigmoid = cute.arch.rcp_approx(
        Float32(1.0) + cute.math.exp(-gate, fastmath=fast_math)
    )
    if cutlass.const_expr(is_situ):
        beta = Float32(SITU_DEFAULT_BETA)
        linear_beta = Float32(SITU_DEFAULT_LINEAR_BETA)
        situ_gate = beta * cute.math.tanh(gate / beta, fastmath=fast_math) * sigmoid
        situ_up = linear_beta * cute.math.tanh(up / linear_beta, fastmath=fast_math)
        return situ_gate * situ_up
    return gate * sigmoid * up


@cute.jit
def _load_gu_rot4(
    trellis_rotations: cute.Tensor,
    trellis_rotations_u16: cute.Tensor,
    expert_idx: Int32,
    tr_slot: Int32,
    col: Int32,
    tr_isz: Int32,
    packed_rot: Int32,
):
    s0 = Float32(0.0)
    s1 = Float32(0.0)
    s2 = Float32(0.0)
    s3 = Float32(0.0)
    if packed_rot != Int32(0):
        stride = tr_isz + (tr_isz >> Int32(3)) + Int32(8)
        row_base = expert_idx * stride
        gain = Float32(trellis_rotations[row_base + tr_isz + (tr_isz >> Int32(3)) + tr_slot])
        neg_gain = -gain
        bits = Int32(
            trellis_rotations_u16[
                row_base + tr_isz + tr_slot * (tr_isz >> Int32(4)) + (col >> Int32(4))
            ]
        )
        sh = col & Int32(15)
        s0 = neg_gain if ((bits >> sh) & Int32(1)) != Int32(0) else gain
        s1 = neg_gain if ((bits >> (sh + Int32(1))) & Int32(1)) != Int32(0) else gain
        s2 = neg_gain if ((bits >> (sh + Int32(2))) & Int32(1)) != Int32(0) else gain
        s3 = neg_gain if ((bits >> (sh + Int32(3))) & Int32(1)) != Int32(0) else gain
    else:
        sc_idx = expert_idx * (Int32(3) * tr_isz) + tr_slot * tr_isz + col
        s0 = Float32(trellis_rotations[sc_idx])
        s1 = Float32(trellis_rotations[sc_idx + Int32(1)])
        s2 = Float32(trellis_rotations[sc_idx + Int32(2)])
        s3 = Float32(trellis_rotations[sc_idx + Int32(3)])
    return s0, s1, s2, s3


@cute.jit
def _apply_intermediate_hadamard_and_store_f16(
    gate_row_smem: Int32,
    up_row_smem: Int32,
    row_swiz: Int32,
    trellis_rotations: cute.Tensor,
    trellis_rotations_u16: cute.Tensor,
    sign_patterns: cute.Tensor,
    sign_table: cute.Tensor,
    intermediate_u32: cute.Tensor,
    expert_idx: Int32,
    output_tile: Int32,
    dst_row: Int64,
    intermediate_tiles: Int32,
    packed_rot: Int32,
    lane: Int32,
    is_situ: cutlass.Constexpr[bool],
    fast_math: cutlass.Constexpr[bool],
):
    tr_isz = intermediate_tiles * Int32(128)
    tr_col = output_tile * Int32(128) + lane * Int32(4)
    tr_chunk = lane >> Int32(3)
    tr_chunk_lane = lane & Int32(7)
    tr_slot = tr_chunk & Int32(1)
    tr_sign_base = sign_patterns[expert_idx].to(Int32) * (Int32(3) * tr_isz)
    tr_ci0 = ((tr_chunk >> Int32(1)) << Int32(5)) + tr_chunk_lane * Int32(4)
    tr_ci1 = Int32(64) + tr_ci0
    tr_sg0_idx = tr_sign_base + output_tile * Int32(256) + lane * Int32(4)
    tr_sg1_idx = tr_sg0_idx + Int32(128)
    tr_ub_idx = tr_sign_base + Int32(2) * tr_isz + tr_col
    tr_dn_idx = (
        expert_idx * (tr_isz + (tr_isz >> Int32(3)) + Int32(8)) + tr_col
        if packed_rot != Int32(0)
        else expert_idx * (Int32(3) * tr_isz) + Int32(2) * tr_isz + tr_col
    )

    aa0 = Float32(0.0)
    aa1 = Float32(0.0)
    bb0 = Float32(0.0)
    bb1 = Float32(0.0)
    for _pb in cutlass.range_constexpr(2):
        ci_raw = tr_ci0 if cutlass.const_expr(_pb == 0) else tr_ci1
        tr_ci = ci_raw ^ row_swiz
        base_src = gate_row_smem if tr_slot == Int32(0) else up_row_smem
        p0, p1, p2, p3 = ld_shared_v4_f32(base_src + (tr_ci << Int32(2)))
        h0, h1, h2, h3 = _w4a8_had128_quad(p0, p1, p2, p3, lane)
        r0, r1, r2, r3 = _load_gu_rot4(
            trellis_rotations,
            trellis_rotations_u16,
            expert_idx,
            tr_slot,
            output_tile * Int32(128) + ci_raw,
            tr_isz,
            packed_rot,
        )
        h0 = h0 * r0
        h1 = h1 * r1
        h2 = h2 * r2
        h3 = h3 * r3
        h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, lane)
        sg_idx = tr_sg0_idx if cutlass.const_expr(_pb == 0) else tr_sg1_idx
        h0 = h0 * Float32(sign_table[sg_idx])
        h1 = h1 * Float32(sign_table[sg_idx + Int32(1)])
        h2 = h2 * Float32(sign_table[sg_idx + Int32(2)])
        h3 = h3 * Float32(sign_table[sg_idx + Int32(3)])
        s0 = _activated_f32(h0, h1, is_situ, fast_math)
        s1 = _activated_f32(h2, h3, is_situ, fast_math)
        if cutlass.const_expr(_pb == 0):
            aa0, aa1 = s0, s1
        else:
            bb0, bb1 = s0, s1

    src0 = (lane & Int32(15)) << Int32(1)
    src1 = src0 + Int32(1)
    av0 = cute.arch.shuffle_sync(aa0, src0)
    av1 = cute.arch.shuffle_sync(aa1, src0)
    av2 = cute.arch.shuffle_sync(aa0, src1)
    av3 = cute.arch.shuffle_sync(aa1, src1)
    bv0 = cute.arch.shuffle_sync(bb0, src0)
    bv1 = cute.arch.shuffle_sync(bb1, src0)
    bv2 = cute.arch.shuffle_sync(bb0, src1)
    bv3 = cute.arch.shuffle_sync(bb1, src1)
    o0, o1, o2, o3 = av0, av1, av2, av3
    if lane >= Int32(16):
        o0, o1, o2, o3 = bv0, bv1, bv2, bv3

    o0 = o0 * Float32(sign_table[tr_ub_idx])
    o1 = o1 * Float32(sign_table[tr_ub_idx + Int32(1)])
    o2 = o2 * Float32(sign_table[tr_ub_idx + Int32(2)])
    o3 = o3 * Float32(sign_table[tr_ub_idx + Int32(3)])
    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)
    o0 = o0 * Float32(trellis_rotations[tr_dn_idx])
    o1 = o1 * Float32(trellis_rotations[tr_dn_idx + Int32(1)])
    o2 = o2 * Float32(trellis_rotations[tr_dn_idx + Int32(2)])
    o3 = o3 * Float32(trellis_rotations[tr_dn_idx + Int32(3)])
    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)

    words_per_row = intermediate_tiles * Int32(64)
    dst_word = (
        dst_row * Int64(words_per_row)
        + Int64(output_tile * Int32(64) + lane * Int32(2))
    )
    intermediate_u32[dst_word] = pack_f32x2_to_f16x2(o0, o1)
    intermediate_u32[dst_word + Int64(1)] = pack_f32x2_to_f16x2(o2, o3)


# =============================================================================
# 1. DECODE ENGINE (M = 1..32, block_size = 8, 8 Warps = 256 Threads/CTA)
# =============================================================================


class Exl3W2A16DecodeFC1Kernel16W:
    """16-warp (8N x 2K) Spill-Free, Zero-B-SMEM, 64 KiB Direct-LUT Lossless W2A16 Decode FC1.

    Hits the SM120 sweet spot: 512 threads/CTA (4 warps/sub-partition saturating the 32-bank
    SMEM Direct LUT pipe), 128 regs/thread budget (~68 used -> 0 spills!), only 2 K-warps
    (16 KiB epilogue SMEM), fused SiTU+Hadamard when ``split_k == 1``, and ``direct_m1``.
    """

    tile_m = 8
    tile_n = 128
    tile_k = 128
    num_warps = 16
    threads_per_cta = 512

    def __init__(
        self,
        *,
        top_k: int,
        trellis_bits: int = 2,
        activation: str = SITU,
        stages: int = 3,
        split_k_mode: bool = False,
        direct_m1: bool = False,
        fast_math: bool = True,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3W2A16DecodeFC1Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3):
            raise ValueError(f"stages must be 2 or 3, got {stages}")
        if activation not in ("silu", SITU):
            raise ValueError(f"unsupported activation {activation!r}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.is_situ = activation == SITU
        self.stages = int(stages)
        self.split_k_mode = bool(split_k_mode)
        self.direct_m1 = bool(direct_m1)
        self.fast_math = bool(fast_math)

        self.stage_bytes = self.tile_m * self.tile_k * 2  # 2,048 B
        self.pipeline_bytes = self.stages * self.stage_bytes  # 6,144 B
        self.epilogue_half_bytes = 2 * self.tile_m * self.tile_n * 4  # 8,192 B
        self.epilogue_bytes = 2 * self.epilogue_half_bytes  # 16,384 B
        self.work_bytes = max(self.pipeline_bytes, self.epilogue_bytes)

        self.tok_smem_offset = self.work_bytes
        self.route_smem_offset = self.tok_smem_offset + self.tile_m * 4
        self.meta_smem_offset = self.route_smem_offset + self.tile_m * 4
        self.trellis_lut_offset = ((self.meta_smem_offset + 16 + 255) // 256) * 256
        self.total_smem_bytes = self.trellis_lut_offset + 65536
        if self.total_smem_bytes > 101376:
            raise ValueError(f"SMEM footprint {self.total_smem_bytes} exceeds 101,376 B")
        self.shared_words = (self.total_smem_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        a_f16_u32: cute.Tensor,
        w13_u32: cute.Tensor,
        intermediate_u32: cute.Tensor,
        fc1_accum_f32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        live_routes: cutlass.Int32,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        split_k: cutlass.Int32,
        packed_rot: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_f16_u32,
            w13_u32,
            intermediate_u32,
            fc1_accum_f32,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            trellis_lut,
            trellis_rotations,
            trellis_rotations_u16,
            sign_patterns,
            sign_table,
            live_routes,
            input_k128_tiles,
            intermediate_tiles,
            split_k,
            packed_rot,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            min_blocks_per_mp=1,
            preferred_smem_carveout=100,
            stream=stream,
        )

    @cute.jit
    def _stage_a_slice(
        self,
        a_f16_u32: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k128_slice: Int32,
        input_k128_tiles: Int32,
        stage_idx: Int32,
    ):
        stage_base = smem_base + stage_idx * Int32(self.stage_bytes)
        tok_smem = smem_base + Int32(self.tok_smem_offset)
        words_per_token = input_k128_tiles * Int32(64)
        if tid < Int32(128):
            row = tid >> Int32(4)
            vec = tid & Int32(15)
            tok = ld_shared_i32_relaxed(tok_smem + (row << Int32(2)))
            phys_vec = vec ^ row
            src_word = Int64(tok) * Int64(words_per_token) + Int64(
                k128_slice * Int32(64) + (vec << Int32(2))
            )
            cp_async4_shared_global(
                stage_base + (row << Int32(8)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(a_f16_u32, src_word),
            )

    @cute.kernel
    def kernel(
        self,
        a_f16_u32: cute.Tensor,
        w13_u32: cute.Tensor,
        intermediate_u32: cute.Tensor,
        fc1_accum_f32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        live_routes: cutlass.Int32,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        split_k: cutlass.Int32,
        packed_rot: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(tid >> Int32(5))
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        warp_n = warp_idx & Int32(7)  # 0..7 (16 cols = 1 N16 tile each)
        warp_k = warp_idx >> Int32(3)  # 0..1 (K64 = 2 K32 pairs each)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        _load_direct_lut_smem(
            trellis_lut,
            smem_base + Int32(self.trellis_lut_offset),
            self.trellis_bits,
            tid,
            self.threads_per_cta,
        )

        tok_smem = smem_base + Int32(self.tok_smem_offset)
        route_smem = smem_base + Int32(self.route_smem_offset)
        meta_smem = smem_base + Int32(self.meta_smem_offset)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))
        ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)

        tr_n16_cnt = intermediate_tiles * Int32(8)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(input_k128_tiles * Int32(8)) * tr_k16_stride
        tr_w13_half = Int64(w13_u32.shape[0]) >> Int64(1)
        tr_k128_step = Int64(8) * tr_k16_stride
        tr_k32_step = Int64(2) * tr_k16_stride

        ldm_row = lane & Int32(7)
        ldm_k8 = (lane >> Int32(3)) & Int32(1)
        wk_vec_base = warp_k * Int32(8)
        a_off_k0 = (ldm_row << Int32(8)) + (((wk_vec_base + Int32(0) + ldm_k8) ^ ldm_row) << Int32(4))
        a_off_k1 = (ldm_row << Int32(8)) + (((wk_vec_base + Int32(2) + ldm_k8) ^ ldm_row) << Int32(4))
        a_off_k2 = (ldm_row << Int32(8)) + (((wk_vec_base + Int32(4) + ldm_k8) ^ ldm_row) << Int32(4))
        a_off_k3 = (ldm_row << Int32(8)) + (((wk_vec_base + Int32(6) + ldm_k8) ^ ldm_row) << Int32(4))

        num_m_tiles = (
            Int32(self.top_k)
            if cutlass.const_expr(self.direct_m1)
            else (packed_route_count[Int32(0)].to(Int32) // Int32(self.tile_m))
        )
        eff_split_k = split_k if cutlass.const_expr(self.split_k_mode) else Int32(1)
        tasks_per_m = intermediate_tiles * eff_split_k
        task_tail = num_m_tiles * tasks_per_m
        task_slot = Int32(bidx)

        while task_slot < task_tail:
            m_tile = task_slot // tasks_per_m
            rem = task_slot - m_tile * tasks_per_m
            if cutlass.const_expr(self.split_k_mode):
                k_split = rem % eff_split_k
                output_tile = rem // eff_split_k
            else:
                k_split = Int32(0)
                output_tile = rem

            expert_idx = (
                packed_route_indices[m_tile].to(Int32)
                if cutlass.const_expr(self.direct_m1)
                else block_expert_ids[m_tile].to(Int32)
            )
            if expert_idx >= Int32(0):
                valid_rows = Int32(1)
                if cutlass.const_expr(self.direct_m1):
                    if tid < Int32(8):
                        st_shared_i32(tok_smem + (tid << Int32(2)), Int32(0))
                        st_shared_i32(route_smem + (tid << Int32(2)), m_tile)
                    cute.arch.sync_threads()
                else:
                    if tid < Int32(32):
                        is_v = Int32(0)
                        if tid < Int32(8):
                            r = packed_route_indices[m_tile * Int32(8) + tid].to(Int32)
                            tok = Int32(0)
                            rr = Int32(0)
                            if r >= Int32(0) and r < live_routes:
                                is_v = Int32(1)
                                tok = r // Int32(self.top_k)
                                rr = r
                            st_shared_i32(tok_smem + (tid << Int32(2)), tok)
                            st_shared_i32(route_smem + (tid << Int32(2)), rr)
                        wsum = _warp_sum_i32(is_v)
                        if tid == Int32(0):
                            st_shared_i32(meta_smem, wsum)
                    cute.arch.sync_threads()
                    valid_rows = ld_shared_i32_relaxed(meta_smem)

                if valid_rows > Int32(0):
                    if cutlass.const_expr(self.split_k_mode):
                        k_start = (k_split * input_k128_tiles) // eff_split_k
                        k_end = ((k_split + Int32(1)) * input_k128_tiles) // eff_split_k
                    else:
                        k_start = Int32(0)
                        k_end = input_k128_tiles

                    k_count = k_end - k_start
                    if k_count > Int32(0):
                        for preload in cutlass.range_constexpr(self.stages - 1):
                            if Int32(preload) < k_count:
                                self._stage_a_slice(
                                    a_f16_u32,
                                    smem_base,
                                    tid,
                                    k_start + Int32(preload),
                                    input_k128_tiles,
                                    Int32(preload),
                                )
                            cute.arch.cp_async_commit_group()

                        g_slice_base = (
                            Int64(expert_idx) * tr_eu
                            + Int64(output_tile * Int32(8) + warp_n) * Int64(16)
                            + Int64(k_start * Int32(8) + warp_k * Int32(4)) * tr_k16_stride
                        )
                        # Full K128-slice-ahead 4-reg B prefetch!
                        rg0 = _load_trellis_k32_raw_u32(w13_u32, g_slice_base, tr_k16_stride, lane)
                        ru0 = _load_trellis_k32_raw_u32(
                            w13_u32, g_slice_base + tr_w13_half, tr_k16_stride, lane
                        )
                        rg1 = _load_trellis_k32_raw_u32(
                            w13_u32, g_slice_base + tr_k32_step, tr_k16_stride, lane
                        )
                        ru1 = _load_trellis_k32_raw_u32(
                            w13_u32, g_slice_base + tr_w13_half + tr_k32_step, tr_k16_stride, lane
                        )

                        g0 = Float32(0.0)
                        g1 = Float32(0.0)
                        g2 = Float32(0.0)
                        g3 = Float32(0.0)
                        u0 = Float32(0.0)
                        u1 = Float32(0.0)
                        u2 = Float32(0.0)
                        u3 = Float32(0.0)

                        step = Int32(0)
                        while step < k_count:
                            stage = step % Int32(self.stages)
                            stage_base = smem_base + stage * Int32(self.stage_bytes)

                            next_step = step + Int32(self.stages - 1)
                            if next_step < k_count:
                                self._stage_a_slice(
                                    a_f16_u32,
                                    smem_base,
                                    tid,
                                    k_start + next_step,
                                    input_k128_tiles,
                                    next_step % Int32(self.stages),
                                )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(self.stages - 1)
                            cute.arch.sync_threads()

                            nxt_g0, nxt_u0, nxt_g1, nxt_u1 = rg0, ru0, rg1, ru1
                            next_b_base = g_slice_base + tr_k128_step
                            if step + Int32(1) < k_count:
                                nxt_g0 = _load_trellis_k32_raw_u32(w13_u32, next_b_base, tr_k16_stride, lane)
                                nxt_u0 = _load_trellis_k32_raw_u32(
                                    w13_u32, next_b_base + tr_w13_half, tr_k16_stride, lane
                                )
                                nxt_g1 = _load_trellis_k32_raw_u32(
                                    w13_u32, next_b_base + tr_k32_step, tr_k16_stride, lane
                                )
                                nxt_u1 = _load_trellis_k32_raw_u32(
                                    w13_u32, next_b_base + tr_w13_half + tr_k32_step, tr_k16_stride, lane
                                )

                            # k16 = 0
                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off_k0)
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(rg0, ia, ib, s2, trellis_lut_addr)
                            g0, g1, g2, g3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                g0, g1, g2, g3, b0, b2, b1, b3, a0, a1
                            )
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(ru0, ia, ib, s2, trellis_lut_addr)
                            u0, u1, u2, u3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                u0, u1, u2, u3, b0, b2, b1, b3, a0, a1
                            )

                            # k16 = 1
                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off_k1)
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                rg0, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                            )
                            g0, g1, g2, g3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                g0, g1, g2, g3, b0, b2, b1, b3, a0, a1
                            )
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                ru0, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                            )
                            u0, u1, u2, u3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                u0, u1, u2, u3, b0, b2, b1, b3, a0, a1
                            )

                            # k16 = 2
                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off_k2)
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(rg1, ia, ib, s2, trellis_lut_addr)
                            g0, g1, g2, g3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                g0, g1, g2, g3, b0, b2, b1, b3, a0, a1
                            )
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(ru1, ia, ib, s2, trellis_lut_addr)
                            u0, u1, u2, u3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                u0, u1, u2, u3, b0, b2, b1, b3, a0, a1
                            )

                            # k16 = 3
                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off_k3)
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                rg1, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                            )
                            g0, g1, g2, g3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                g0, g1, g2, g3, b0, b2, b1, b3, a0, a1
                            )
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                ru1, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                            )
                            u0, u1, u2, u3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                u0, u1, u2, u3, b0, b2, b1, b3, a0, a1
                            )

                            rg0, ru0, rg1, ru1 = nxt_g0, nxt_u0, nxt_g1, nxt_u1
                            if step + Int32(self.stages) < k_count:
                                cute.arch.sync_threads()
                            step += Int32(1)
                            g_slice_base = next_b_base

                        cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_threads()

                        gate_smem = smem_base
                        up_smem = smem_base + Int32(self.epilogue_half_bytes)
                        wk_off = warp_k * Int32(8 * 128 * 4)
                        row0 = c << Int32(1)
                        row1 = row0 + Int32(1)
                        c_swiz = c << Int32(3)
                        col_lo = (warp_n * Int32(16) + q) ^ c_swiz
                        col_hi = (warp_n * Int32(16) + q + Int32(8)) ^ c_swiz
                        off_r0_lo = wk_off + ((row0 * Int32(128) + col_lo) << Int32(2))
                        off_r1_lo = wk_off + ((row1 * Int32(128) + col_lo) << Int32(2))
                        off_r0_hi = wk_off + ((row0 * Int32(128) + col_hi) << Int32(2))
                        off_r1_hi = wk_off + ((row1 * Int32(128) + col_hi) << Int32(2))

                        st_shared_f32(gate_smem + off_r0_lo, g0)
                        st_shared_f32(gate_smem + off_r1_lo, g1)
                        st_shared_f32(gate_smem + off_r0_hi, g2)
                        st_shared_f32(gate_smem + off_r1_hi, g3)
                        st_shared_f32(up_smem + off_r0_lo, u0)
                        st_shared_f32(up_smem + off_r1_lo, u1)
                        st_shared_f32(up_smem + off_r0_hi, u2)
                        st_shared_f32(up_smem + off_r1_hi, u3)
                        cute.arch.sync_threads()

                        if warp_idx < valid_rows:
                            row_swiz = (warp_idx >> Int32(1)) << Int32(3)
                            col4 = (lane * Int32(4)) ^ row_swiz
                            r_off = (warp_idx * Int32(128) + col4) << Int32(2)
                            gs0, gs1, gs2, gs3 = ld_shared_v4_f32(gate_smem + r_off)
                            a0, a1, a2, a3 = ld_shared_v4_f32(gate_smem + Int32(8 * 128 * 4) + r_off)
                            gs0 += a0
                            gs1 += a1
                            gs2 += a2
                            gs3 += a3
                            us0, us1, us2, us3 = ld_shared_v4_f32(up_smem + r_off)
                            b0, b1, b2, b3 = ld_shared_v4_f32(up_smem + Int32(8 * 128 * 4) + r_off)
                            us0 += b0
                            us1 += b1
                            us2 += b2
                            us3 += b3

                            route_id = ld_shared_i32_relaxed(route_smem + (warp_idx << Int32(2)))
                            if cutlass.const_expr(self.split_k_mode):
                                isz = intermediate_tiles * Int32(128)
                                base_g = (
                                    Int64(route_id) * Int64(Int32(2) * isz)
                                    + Int64(output_tile * Int32(128) + lane * Int32(4))
                                )
                                base_u = base_g + Int64(isz)
                                _red_add_global_v2_f32(
                                    get_ptr_as_int64(fc1_accum_f32, base_g), gs0, gs1
                                )
                                _red_add_global_v2_f32(
                                    get_ptr_as_int64(fc1_accum_f32, base_g + Int64(2)), gs2, gs3
                                )
                                _red_add_global_v2_f32(
                                    get_ptr_as_int64(fc1_accum_f32, base_u), us0, us1
                                )
                                _red_add_global_v2_f32(
                                    get_ptr_as_int64(fc1_accum_f32, base_u + Int64(2)), us2, us3
                                )
                            else:
                                row_base_g = gate_smem + (warp_idx * Int32(128 * 4))
                                row_base_u = up_smem + (warp_idx * Int32(128 * 4))
                                _st_shared_v2_f32(row_base_g + (col4 << Int32(2)), gs0, gs1)
                                _st_shared_v2_f32(row_base_g + ((col4 + Int32(2)) << Int32(2)), gs2, gs3)
                                _st_shared_v2_f32(row_base_u + (col4 << Int32(2)), us0, us1)
                                _st_shared_v2_f32(row_base_u + ((col4 + Int32(2)) << Int32(2)), us2, us3)
                                cute.arch.sync_warp()
                                _apply_intermediate_hadamard_and_store_f16(
                                    row_base_g,
                                    row_base_u,
                                    row_swiz,
                                    trellis_rotations,
                                    trellis_rotations_u16,
                                    sign_patterns,
                                    sign_table,
                                    intermediate_u32,
                                    expert_idx,
                                    output_tile,
                                    Int64(route_id),
                                    intermediate_tiles,
                                    packed_rot,
                                    lane,
                                    self.is_situ,
                                    self.fast_math,
                                )
                        cute.arch.sync_threads()
            task_slot += Int32(gdimx)


class Exl3ActHadF16Kernel:
    """Lightweight warp-per-(route, post_block) activation + intermediate Hadamard (for Split-K)."""

    def __init__(self, *, activation: str = SITU, fast_math: bool = True):
        if activation not in ("silu", SITU):
            raise ValueError(f"unsupported activation {activation!r}")
        self.is_situ = activation == SITU
        self.fast_math = bool(fast_math)

    @cute.jit
    def __call__(
        self,
        fc1_accum_f32: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        intermediate_u32: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_rot: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        total_warps = live_routes * intermediate_tiles
        grid_x = (total_warps + Int32(3)) >> Int32(2)
        self.kernel(
            fc1_accum_f32,
            topk_ids_flat,
            trellis_rotations,
            trellis_rotations_u16,
            sign_patterns,
            sign_table,
            intermediate_u32,
            live_routes,
            intermediate_tiles,
            packed_rot,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        fc1_accum_f32: cute.Tensor,
        topk_ids_flat: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        intermediate_u32: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_rot: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        warp_id = Int32(bidx) * Int32(4) + (tid >> Int32(5))
        lane = tid & Int32(31)
        total_warps = live_routes * intermediate_tiles
        if warp_id < total_warps:
            route_id = warp_id // intermediate_tiles
            output_tile = warp_id - route_id * intermediate_tiles
            expert_idx = topk_ids_flat[route_id].to(Int32)
            if expert_idx >= Int32(0):
                tr_isz = intermediate_tiles * Int32(128)
                tr_col = output_tile * Int32(128) + lane * Int32(4)
                tr_chunk = lane >> Int32(3)
                tr_chunk_lane = lane & Int32(7)
                tr_slot = tr_chunk & Int32(1)
                tr_sign_base = sign_patterns[expert_idx].to(Int32) * (Int32(3) * tr_isz)
                tr_ci0 = ((tr_chunk >> Int32(1)) << Int32(5)) + tr_chunk_lane * Int32(4)
                tr_ci1 = Int32(64) + tr_ci0
                tr_sg0_idx = tr_sign_base + output_tile * Int32(256) + lane * Int32(4)
                tr_sg1_idx = tr_sg0_idx + Int32(128)
                tr_ub_idx = tr_sign_base + Int32(2) * tr_isz + tr_col
                tr_dn_idx = (
                    expert_idx * (tr_isz + (tr_isz >> Int32(3)) + Int32(8)) + tr_col
                    if packed_rot != Int32(0)
                    else expert_idx * (Int32(3) * tr_isz) + Int32(2) * tr_isz + tr_col
                )

                fc1_row_base = Int64(route_id) * Int64(Int32(2) * tr_isz) + Int64(tr_slot * tr_isz)
                aa0 = Float32(0.0)
                aa1 = Float32(0.0)
                bb0 = Float32(0.0)
                bb1 = Float32(0.0)
                for _pb in cutlass.range_constexpr(2):
                    tr_ci = tr_ci0 if cutlass.const_expr(_pb == 0) else tr_ci1
                    src_idx = fc1_row_base + Int64(output_tile * Int32(128) + tr_ci)
                    p0 = fc1_accum_f32[src_idx]
                    p1 = fc1_accum_f32[src_idx + Int64(1)]
                    p2 = fc1_accum_f32[src_idx + Int64(2)]
                    p3 = fc1_accum_f32[src_idx + Int64(3)]
                    h0, h1, h2, h3 = _w4a8_had128_quad(p0, p1, p2, p3, lane)
                    r0, r1, r2, r3 = _load_gu_rot4(
                        trellis_rotations,
                        trellis_rotations_u16,
                        expert_idx,
                        tr_slot,
                        output_tile * Int32(128) + tr_ci,
                        tr_isz,
                        packed_rot,
                    )
                    h0 = h0 * r0
                    h1 = h1 * r1
                    h2 = h2 * r2
                    h3 = h3 * r3
                    h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, lane)
                    sg_idx = tr_sg0_idx if cutlass.const_expr(_pb == 0) else tr_sg1_idx
                    h0 = h0 * Float32(sign_table[sg_idx])
                    h1 = h1 * Float32(sign_table[sg_idx + Int32(1)])
                    h2 = h2 * Float32(sign_table[sg_idx + Int32(2)])
                    h3 = h3 * Float32(sign_table[sg_idx + Int32(3)])
                    s0 = _activated_f32(h0, h1, self.is_situ, self.fast_math)
                    s1 = _activated_f32(h2, h3, self.is_situ, self.fast_math)
                    if cutlass.const_expr(_pb == 0):
                        aa0, aa1 = s0, s1
                    else:
                        bb0, bb1 = s0, s1

                src0 = (lane & Int32(15)) << Int32(1)
                src1 = src0 + Int32(1)
                av0 = cute.arch.shuffle_sync(aa0, src0)
                av1 = cute.arch.shuffle_sync(aa1, src0)
                av2 = cute.arch.shuffle_sync(aa0, src1)
                av3 = cute.arch.shuffle_sync(aa1, src1)
                bv0 = cute.arch.shuffle_sync(bb0, src0)
                bv1 = cute.arch.shuffle_sync(bb1, src0)
                bv2 = cute.arch.shuffle_sync(bb0, src1)
                bv3 = cute.arch.shuffle_sync(bb1, src1)
                o0, o1, o2, o3 = av0, av1, av2, av3
                if lane >= Int32(16):
                    o0, o1, o2, o3 = bv0, bv1, bv2, bv3
                o0 = o0 * Float32(sign_table[tr_ub_idx])
                o1 = o1 * Float32(sign_table[tr_ub_idx + Int32(1)])
                o2 = o2 * Float32(sign_table[tr_ub_idx + Int32(2)])
                o3 = o3 * Float32(sign_table[tr_ub_idx + Int32(3)])
                o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)
                o0 = o0 * Float32(trellis_rotations[tr_dn_idx])
                o1 = o1 * Float32(trellis_rotations[tr_dn_idx + Int32(1)])
                o2 = o2 * Float32(trellis_rotations[tr_dn_idx + Int32(2)])
                o3 = o3 * Float32(trellis_rotations[tr_dn_idx + Int32(3)])
                o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)

                words_per_row = intermediate_tiles * Int32(64)
                dst_word = (
                    Int64(route_id) * Int64(words_per_row)
                    + Int64(output_tile * Int32(64) + lane * Int32(2))
                )
                intermediate_u32[dst_word] = pack_f32x2_to_f16x2(o0, o1)
                intermediate_u32[dst_word + Int64(1)] = pack_f32x2_to_f16x2(o2, o3)


class Exl3W2A16DecodeFC2Kernel16W:
    """16-warp (16N x 1K) Zero-B-SMEM, Zero-K-Barrier, Zero-Epilogue-SMEM Lossless W2A16 Decode FC2.

    - 16 warps (512 threads, 4 warps/sub-partition) with ONLY 4 acc regs/thread (~44 regs -> 0 spills!).
    - N256 output tile (14 tiles/expert for Kimi H=3584) OR N512 (7 tiles/expert -> 112 CTAs = 1 wave at M=1!).
    - Preloads ALL <=3 K128 slices (6 KiB) in ONE cp.async -> ZERO K-loop barriers!
    - 16N x 1K -> ZERO K-reduce SMEM, ZERO epilogue barriers (shfl.bfly(4) directly into red.add.v2.f32)!
    """

    tile_m = 8
    tile_k = 128
    num_warps = 16
    threads_per_cta = 512

    def __init__(
        self,
        *,
        top_k: int,
        trellis_bits: int = 2,
        stages: int = 3,
        tile_n: int = 512,
        direct_m1: bool = False,
        use_bf16_scatter: bool = True,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3W2A16DecodeFC2Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3):
            raise ValueError(f"stages must be 2 or 3, got {stages}")
        if tile_n not in (256, 512):
            raise ValueError(f"tile_n must be 256 or 512, got {tile_n}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.stages = int(stages)
        self.tile_n = int(tile_n)
        self.n_per_warp = self.tile_n // 16  # 16 (1 tile) or 32 (2 tiles)
        self.direct_m1 = bool(direct_m1)
        self.use_bf16_scatter = bool(use_bf16_scatter)

        self.stage_bytes = self.tile_m * self.tile_k * 2  # 2,048 B
        self.pipeline_bytes = self.stages * self.stage_bytes  # 6,144 B
        self.work_bytes = self.pipeline_bytes

        self.tok_smem_offset = self.work_bytes
        self.route_smem_offset = self.tok_smem_offset + self.tile_m * 4
        self.wt_smem_offset = self.route_smem_offset + self.tile_m * 4
        self.meta_smem_offset = self.wt_smem_offset + self.tile_m * 4
        self.trellis_lut_offset = ((self.meta_smem_offset + 16 + 255) // 256) * 256
        self.total_smem_bytes = self.trellis_lut_offset + 65536
        if self.total_smem_bytes > 101376:
            raise ValueError(f"SMEM footprint {self.total_smem_bytes} exceeds 101,376 B")
        self.shared_words = (self.total_smem_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        intermediate_u32: cute.Tensor,
        w2_u32: cute.Tensor,
        scatter_f32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        trellis_lut: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        output_n_tiles: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            intermediate_u32,
            w2_u32,
            scatter_f32,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            trellis_lut,
            live_routes,
            intermediate_tiles,
            output_n_tiles,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            min_blocks_per_mp=1,
            preferred_smem_carveout=100,
            stream=stream,
        )

    @cute.jit
    def _stage_a_slice(
        self,
        intermediate_u32: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k128_slice: Int32,
        intermediate_tiles: Int32,
        stage_idx: Int32,
    ):
        stage_base = smem_base + stage_idx * Int32(self.stage_bytes)
        route_smem = smem_base + Int32(self.route_smem_offset)
        words_per_row = intermediate_tiles * Int32(64)
        if tid < Int32(128):
            row = tid >> Int32(4)
            vec = tid & Int32(15)
            r = ld_shared_i32_relaxed(route_smem + (row << Int32(2)))
            phys_vec = vec ^ row
            src_word = Int64(r) * Int64(words_per_row) + Int64(
                k128_slice * Int32(64) + (vec << Int32(2))
            )
            cp_async4_shared_global(
                stage_base + (row << Int32(8)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(intermediate_u32, src_word),
            )

    @cute.jit
    def _scatter_pair_from_reg(
        self,
        scatter_f32: cute.Tensor,
        scatter_n: Int32,
        tok: Int32,
        wt: Float32,
        col_g: Int32,
        is_odd: Int32,
        v_even: Float32,
        v_odd: Float32,
        valid: cutlass.Boolean,
    ):
        send = v_even if is_odd != Int32(0) else v_odd
        peer = cute.arch.shuffle_sync_bfly(send, offset=4)
        out0 = peer if is_odd != Int32(0) else v_even
        out1 = v_odd if is_odd != Int32(0) else peer
        if valid:
            dst_ptr = get_ptr_as_int64(scatter_f32, Int64(tok) * Int64(scatter_n) + Int64(col_g))
            if cutlass.const_expr(self.use_bf16_scatter):
                scatter_add_bf16x2(dst_ptr, out0 * wt, out1 * wt)
            else:
                _red_add_global_v2_f32(dst_ptr, out0 * wt, out1 * wt)

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        w2_u32: cute.Tensor,
        scatter_f32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        trellis_lut: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        output_n_tiles: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_n = cute.arch.make_warp_uniform(tid >> Int32(5))  # 0..15
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        _load_direct_lut_smem(
            trellis_lut,
            smem_base + Int32(self.trellis_lut_offset),
            self.trellis_bits,
            tid,
            self.threads_per_cta,
        )

        tok_smem = smem_base + Int32(self.tok_smem_offset)
        route_smem = smem_base + Int32(self.route_smem_offset)
        wt_smem = smem_base + Int32(self.wt_smem_offset)
        meta_smem = smem_base + Int32(self.meta_smem_offset)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))
        ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)

        n16_per_cta = self.tile_n // 16  # 16 or 32
        n16_per_warp = self.n_per_warp // 16  # 1 or 2
        tr_n16_cnt = output_n_tiles * Int32(n16_per_cta)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(intermediate_tiles * Int32(8)) * tr_k16_stride
        tr_k32_step = Int64(2) * tr_k16_stride
        scatter_n = Int32(scatter_f32.shape[1])

        ldm_row = lane & Int32(7)
        ldm_k8 = (lane >> Int32(3)) & Int32(1)
        a_off = tuple(
            (ldm_row << Int32(8)) + (((Int32(k * 2) + ldm_k8) ^ ldm_row) << Int32(4))
            for k in range(8)
        )

        is_odd = q & Int32(1)
        out_row = (c << Int32(1)) + is_odd
        col_pair = q & Int32(6)

        num_m_tiles = (
            Int32(self.top_k)
            if cutlass.const_expr(self.direct_m1)
            else (packed_route_count[Int32(0)].to(Int32) // Int32(self.tile_m))
        )
        task_tail = num_m_tiles * output_n_tiles
        task_slot = Int32(bidx)
        preload_all = intermediate_tiles <= Int32(self.stages)

        while task_slot < task_tail:
            output_tile = task_slot % output_n_tiles
            m_tile = task_slot // output_n_tiles
            expert_idx = (
                packed_route_indices[m_tile].to(Int32)
                if cutlass.const_expr(self.direct_m1)
                else block_expert_ids[m_tile].to(Int32)
            )
            if expert_idx >= Int32(0):
                valid_rows = Int32(1)
                if cutlass.const_expr(self.direct_m1):
                    if tid < Int32(8):
                        st_shared_i32(tok_smem + (tid << Int32(2)), Int32(0))
                        st_shared_i32(route_smem + (tid << Int32(2)), m_tile)
                        wt = topk_weights_flat[m_tile].to(Float32) if tid == Int32(0) else Float32(0.0)
                        st_shared_f32(wt_smem + (tid << Int32(2)), wt)
                    cute.arch.sync_threads()
                else:
                    if tid < Int32(32):
                        is_v = Int32(0)
                        if tid < Int32(8):
                            r = packed_route_indices[m_tile * Int32(8) + tid].to(Int32)
                            tok = Int32(0)
                            rr = Int32(0)
                            wt = Float32(0.0)
                            if r >= Int32(0) and r < live_routes:
                                is_v = Int32(1)
                                tok = r // Int32(self.top_k)
                                rr = r
                                wt = topk_weights_flat[r].to(Float32)
                            st_shared_i32(tok_smem + (tid << Int32(2)), tok)
                            st_shared_i32(route_smem + (tid << Int32(2)), rr)
                            st_shared_f32(wt_smem + (tid << Int32(2)), wt)
                        wsum = _warp_sum_i32(is_v)
                        if tid == Int32(0):
                            st_shared_i32(meta_smem, wsum)
                    cute.arch.sync_threads()
                    valid_rows = ld_shared_i32_relaxed(meta_smem)

                if valid_rows > Int32(0):
                    if preload_all:
                        for s in cutlass.range_constexpr(self.stages):
                            if Int32(s) < intermediate_tiles:
                                self._stage_a_slice(
                                    intermediate_u32, smem_base, tid, Int32(s),
                                    intermediate_tiles, Int32(s),
                                )
                        cute.arch.cp_async_commit_group()
                    else:
                        for s in cutlass.range_constexpr(self.stages - 1):
                            if Int32(s) < intermediate_tiles:
                                self._stage_a_slice(
                                    intermediate_u32, smem_base, tid, Int32(s),
                                    intermediate_tiles, Int32(s),
                                )
                            cute.arch.cp_async_commit_group()

                    w_k32_base = (
                        Int64(expert_idx) * tr_eu
                        + Int64(output_tile * Int32(n16_per_cta) + warp_n * Int32(n16_per_warp)) * Int64(16)
                    )
                    raw_n0 = _load_trellis_k32_raw_u32(w2_u32, w_k32_base, tr_k16_stride, lane)
                    raw_n1 = (
                        _load_trellis_k32_raw_u32(w2_u32, w_k32_base + Int64(16), tr_k16_stride, lane)
                        if cutlass.const_expr(n16_per_warp == 2)
                        else Uint32(0)
                    )

                    if preload_all:
                        cute.arch.cp_async_wait_group(0)
                        cute.arch.sync_threads()

                    fa0 = Float32(0.0)
                    fa1 = Float32(0.0)
                    fa2 = Float32(0.0)
                    fa3 = Float32(0.0)
                    fb0 = Float32(0.0)
                    fb1 = Float32(0.0)
                    fb2 = Float32(0.0)
                    fb3 = Float32(0.0)

                    total_k32 = intermediate_tiles * Int32(4)
                    k128_slice = Int32(0)
                    k32_global = Int32(0)
                    while k128_slice < intermediate_tiles:
                        stage = k128_slice % Int32(self.stages)
                        stage_base = smem_base + stage * Int32(self.stage_bytes)
                        if not preload_all:
                            next_a = k128_slice + Int32(self.stages - 1)
                            if next_a < intermediate_tiles:
                                self._stage_a_slice(
                                    intermediate_u32, smem_base, tid, next_a,
                                    intermediate_tiles, next_a % Int32(self.stages),
                                )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(self.stages - 1)
                            cute.arch.sync_threads()

                        for q32 in cutlass.range_constexpr(4):
                            nxt_n0 = raw_n0
                            nxt_n1 = raw_n1
                            next_w_base = w_k32_base + tr_k32_step
                            if k32_global + Int32(1) < total_k32:
                                nxt_n0 = _load_trellis_k32_raw_u32(
                                    w2_u32, next_w_base, tr_k16_stride, lane
                                )
                                if cutlass.const_expr(n16_per_warp == 2):
                                    nxt_n1 = _load_trellis_k32_raw_u32(
                                        w2_u32, next_w_base + Int64(16), tr_k16_stride, lane
                                    )

                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off[q32 * 2])
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                raw_n0, ia, ib, s2, trellis_lut_addr
                            )
                            fa0, fa1, fa2, fa3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                fa0, fa1, fa2, fa3, b0, b2, b1, b3, a0, a1
                            )
                            if cutlass.const_expr(n16_per_warp == 2):
                                b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                    raw_n1, ia, ib, s2, trellis_lut_addr
                                )
                                fb0, fb1, fb2, fb3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                    fb0, fb1, fb2, fb3, b0, b2, b1, b3, a0, a1
                                )

                            a0, a1 = ldmatrix_m8n8x2_b16(stage_base + a_off[q32 * 2 + 1])
                            b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                raw_n0, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                            )
                            fa0, fa1, fa2, fa3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                fa0, fa1, fa2, fa3, b0, b2, b1, b3, a0, a1
                            )
                            if cutlass.const_expr(n16_per_warp == 2):
                                b0, b1, b2, b3 = _decode_trellis_k16_to_f16x4(
                                    raw_n1, ia + Int32(16), ib + Int32(16), s2, trellis_lut_addr
                                )
                                fb0, fb1, fb2, fb3 = f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
                                    fb0, fb1, fb2, fb3, b0, b2, b1, b3, a0, a1
                                )

                            raw_n0 = nxt_n0
                            raw_n1 = nxt_n1
                            w_k32_base = next_w_base
                            k32_global += Int32(1)

                        if not preload_all and k128_slice + Int32(self.stages) < intermediate_tiles:
                            cute.arch.sync_threads()
                        k128_slice += Int32(1)

                    if not preload_all:
                        cute.arch.cp_async_wait_group(0)

                    valid = out_row < valid_rows
                    tok = ld_shared_i32_relaxed(tok_smem + (out_row << Int32(2)))
                    wt = ld_shared_f32(wt_smem + (out_row << Int32(2)))
                    col_g0 = output_tile * Int32(self.tile_n) + warp_n * Int32(self.n_per_warp) + col_pair
                    self._scatter_pair_from_reg(
                        scatter_f32, scatter_n, tok, wt, col_g0, is_odd, fa0, fa1, valid
                    )
                    self._scatter_pair_from_reg(
                        scatter_f32, scatter_n, tok, wt, col_g0 + Int32(8), is_odd, fa2, fa3, valid
                    )
                    if cutlass.const_expr(n16_per_warp == 2):
                        self._scatter_pair_from_reg(
                            scatter_f32, scatter_n, tok, wt, col_g0 + Int32(16), is_odd, fb0, fb1, valid
                        )
                        self._scatter_pair_from_reg(
                            scatter_f32, scatter_n, tok, wt, col_g0 + Int32(24), is_odd, fb2, fb3, valid
                        )
                    cute.arch.sync_threads()
            task_slot += Int32(gdimx)


# =============================================================================
# 2. PREFILL ENGINE (M = 64..2048, block_size = 64, 16 Warps = 512 Threads/CTA)
# =============================================================================


class Exl3W2A16PrefillFC1Kernel16W:
    """16-warp (8N x 2K) Zero-B-SMEM, 64 KiB Direct-LUT Lossless W2A16 Prefill FC1.

    Writes **contiguous physical-row** ``intermediate_f16 [m_tile*64 + abs_row, I]``
    so FC2 streams sequential cachelines with zero scattered DRAM/L2 stalls.
    """

    tile_m = 64
    tile_n = 128
    tile_k = 64
    num_warps = 16
    threads_per_cta = 512

    def __init__(
        self,
        *,
        top_k: int,
        trellis_bits: int = 2,
        activation: str = SITU,
        stages: int = 3,
        fast_math: bool = True,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3W2A16PrefillFC1Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3):
            raise ValueError(f"stages must be 2 or 3, got {stages}")
        if activation not in ("silu", SITU):
            raise ValueError(f"unsupported activation {activation!r}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.is_situ = activation == SITU
        self.stages = int(stages)
        self.fast_math = bool(fast_math)

        self.stage_bytes = self.tile_m * self.tile_k * 2  # 8,192 B
        self.pipeline_bytes = self.stages * self.stage_bytes  # 24,576 B
        self.epilogue_half_bytes = 32 * self.tile_n * 4  # 16,384 B
        self.epilogue_bytes = 2 * self.epilogue_half_bytes  # 32,768 B
        self.work_bytes = max(self.pipeline_bytes, self.epilogue_bytes)

        self.tok_smem_offset = self.work_bytes
        self.meta_smem_offset = self.tok_smem_offset + self.tile_m * 4
        self.trellis_lut_offset = ((self.meta_smem_offset + 16 + 255) // 256) * 256
        self.total_smem_bytes = self.trellis_lut_offset + 65536
        if self.total_smem_bytes > 101376:
            raise ValueError(f"SMEM footprint {self.total_smem_bytes} exceeds 101,376 B")
        self.shared_words = (self.total_smem_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        a_f16_u32: cute.Tensor,
        w13_u32: cute.Tensor,
        intermediate_u32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        live_routes: cutlass.Int32,
        input_k64_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_rot: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_f16_u32,
            w13_u32,
            intermediate_u32,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            trellis_lut,
            trellis_rotations,
            trellis_rotations_u16,
            sign_patterns,
            sign_table,
            live_routes,
            input_k64_tiles,
            intermediate_tiles,
            packed_rot,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            min_blocks_per_mp=1,
            preferred_smem_carveout=100,
            stream=stream,
        )

    @cute.jit
    def _stage_a_slice(
        self,
        a_f16_u32: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k64_slice: Int32,
        input_k64_tiles: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        stage = k64_slice % Int32(self.stages)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        tok_smem = smem_base + Int32(self.tok_smem_offset)
        words_per_token = input_k64_tiles * Int32(32)
        if tid < Int32(int(compute_m) * 8):
            row = tid >> Int32(3)
            vec = tid & Int32(7)
            tok = ld_shared_i32_relaxed(tok_smem + (row << Int32(2)))
            phys_vec = vec ^ (row & Int32(7))
            src_word = Int64(tok) * Int64(words_per_token) + Int64(
                k64_slice * Int32(32) + (vec << Int32(2))
            )
            cp_async4_shared_global(
                stage_base + (row << Int32(7)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(a_f16_u32, src_word),
            )

    @cute.jit
    def _run_task(
        self,
        a_f16_u32: cute.Tensor,
        w13_u32: cute.Tensor,
        intermediate_u32: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        m_tile: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        input_k64_tiles: Int32,
        intermediate_tiles: Int32,
        packed_rot: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        warp_n = warp_idx & Int32(7)
        warp_k = warp_idx >> Int32(3)

        ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))

        tr_n16_cnt = intermediate_tiles * Int32(8)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(input_k64_tiles * Int32(4)) * tr_k16_stride
        tr_w13_half = Int64(w13_u32.shape[0]) >> Int64(1)
        tr_warp_col_base = (
            Int64(expert_idx) * tr_eu
            + Int64(output_tile * Int32(8) + warp_n) * Int64(16)
            + Int64(warp_k * Int32(2)) * tr_k16_stride
        )
        tr_k64_step = Int64(4) * tr_k16_stride

        for preload in cutlass.range_constexpr(self.stages - 1):
            if Int32(preload) < input_k64_tiles:
                self._stage_a_slice(
                    a_f16_u32, smem_base, tid, Int32(preload), input_k64_tiles, compute_m
                )
            cute.arch.cp_async_commit_group()

        g_slice_base = tr_warp_col_base
        raw_g = _load_trellis_k32_raw_u32(w13_u32, g_slice_base, tr_k16_stride, lane)
        raw_u = _load_trellis_k32_raw_u32(
            w13_u32, g_slice_base + tr_w13_half, tr_k16_stride, lane
        )

        m_blocks = int(compute_m) // 16
        gate_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), Float32) for _nt in range(2))
            for _blk in range(m_blocks)
        )
        up_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), Float32) for _nt in range(2))
            for _blk in range(m_blocks)
        )
        for blk in cutlass.range_constexpr(m_blocks):
            for nt in cutlass.range_constexpr(2):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)

        ldm_row = lane & Int32(15)
        ldm_k8 = lane >> Int32(4)
        ldm_swiz = ldm_row & Int32(7)
        a_off_k0 = (ldm_row << Int32(7)) + (((warp_k * Int32(4) + ldm_k8) ^ ldm_swiz) << Int32(4))
        a_off_k1 = (ldm_row << Int32(7)) + (
            ((warp_k * Int32(4) + Int32(2) + ldm_k8) ^ ldm_swiz) << Int32(4)
        )

        k64_slice = Int32(0)
        while k64_slice < input_k64_tiles:
            stage = k64_slice % Int32(self.stages)
            stage_base = smem_base + stage * Int32(self.stage_bytes)

            next_a = k64_slice + Int32(self.stages - 1)
            if next_a < input_k64_tiles:
                self._stage_a_slice(
                    a_f16_u32, smem_base, tid, next_a, input_k64_tiles, compute_m
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            nxt_g = raw_g
            nxt_u = raw_u
            next_b_base = g_slice_base + tr_k64_step
            if k64_slice + Int32(1) < input_k64_tiles:
                nxt_g = _load_trellis_k32_raw_u32(w13_u32, next_b_base, tr_k16_stride, lane)
                nxt_u = _load_trellis_k32_raw_u32(
                    w13_u32, next_b_base + tr_w13_half, tr_k16_stride, lane
                )

            for k16_local in cutlass.range_constexpr(2):
                ia_k = ia if cutlass.const_expr(k16_local == 0) else (ia + Int32(16))
                ib_k = ib if cutlass.const_expr(k16_local == 0) else (ib + Int32(16))
                a_off_k = a_off_k0 if cutlass.const_expr(k16_local == 0) else a_off_k1

                g0, g1, g2, g3 = _decode_trellis_k16_to_f16x4(
                    raw_g, ia_k, ib_k, s2, trellis_lut_addr
                )
                u0, u1, u2, u3 = _decode_trellis_k16_to_f16x4(
                    raw_u, ia_k, ib_k, s2, trellis_lut_addr
                )

                for blk in cutlass.range_constexpr(m_blocks):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        stage_base + Int32(blk * 16 * 128) + a_off_k
                    )
                    ga0 = gate_acc[blk][0]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        ga0[0], ga0[1], ga0[2], ga0[3], a0, a1, a2, a3, g0, g1
                    )
                    ga0[0], ga0[1], ga0[2], ga0[3] = r0, r1, r2, r3

                    ga1 = gate_acc[blk][1]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        ga1[0], ga1[1], ga1[2], ga1[3], a0, a1, a2, a3, g2, g3
                    )
                    ga1[0], ga1[1], ga1[2], ga1[3] = r0, r1, r2, r3

                    ua0 = up_acc[blk][0]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        ua0[0], ua0[1], ua0[2], ua0[3], a0, a1, a2, a3, u0, u1
                    )
                    ua0[0], ua0[1], ua0[2], ua0[3] = r0, r1, r2, r3

                    ua1 = up_acc[blk][1]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        ua1[0], ua1[1], ua1[2], ua1[3], a0, a1, a2, a3, u2, u3
                    )
                    ua1[0], ua1[1], ua1[2], ua1[3] = r0, r1, r2, r3

            raw_g = nxt_g
            raw_u = nxt_u
            if k64_slice + Int32(self.stages) < input_k64_tiles:
                cute.arch.sync_threads()
            k64_slice += Int32(1)
            g_slice_base = next_b_base

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        gate_smem = smem_base
        up_smem = smem_base + Int32(self.epilogue_half_bytes)
        col_base = warp_n * Int32(16) + (c << Int32(1))
        swiz_q = (q & Int32(3)) << Int32(3)
        physical_row_base = Int64(m_tile) * Int64(self.tile_m)

        num_halves = 2 if int(compute_m) == 64 else 1
        blks_per_half = 1 if int(compute_m) == 16 else 2
        rows_per_warp = 1 if int(compute_m) == 16 else 2

        for half in cutlass.range_constexpr(num_halves):
            row_start = Int32(half * 32)
            if warp_k == Int32(0):
                for hb in cutlass.range_constexpr(blks_per_half):
                    blk = half * 2 + hb
                    row_lo = Int32(hb * 16) + q
                    row_hi = row_lo + Int32(8)
                    for nt in cutlass.range_constexpr(2):
                        swiz_col = (col_base + Int32(nt * 8)) ^ swiz_q
                        off_lo = (row_lo * Int32(128) + swiz_col) << Int32(2)
                        off_hi = (row_hi * Int32(128) + swiz_col) << Int32(2)
                        gf = gate_acc[blk][nt]
                        uf = up_acc[blk][nt]
                        _st_shared_v2_f32(gate_smem + off_lo, gf[0], gf[1])
                        _st_shared_v2_f32(gate_smem + off_hi, gf[2], gf[3])
                        _st_shared_v2_f32(up_smem + off_lo, uf[0], uf[1])
                        _st_shared_v2_f32(up_smem + off_hi, uf[2], uf[3])
            cute.arch.sync_threads()
            if warp_k == Int32(1):
                for hb in cutlass.range_constexpr(blks_per_half):
                    blk = half * 2 + hb
                    row_lo = Int32(hb * 16) + q
                    row_hi = row_lo + Int32(8)
                    for nt in cutlass.range_constexpr(2):
                        swiz_col = (col_base + Int32(nt * 8)) ^ swiz_q
                        off_lo = (row_lo * Int32(128) + swiz_col) << Int32(2)
                        off_hi = (row_hi * Int32(128) + swiz_col) << Int32(2)
                        gf = gate_acc[blk][nt]
                        uf = up_acc[blk][nt]
                        g0, g1 = _ld_shared_v2_f32(gate_smem + off_lo)
                        _st_shared_v2_f32(gate_smem + off_lo, g0 + gf[0], g1 + gf[1])
                        g2, g3 = _ld_shared_v2_f32(gate_smem + off_hi)
                        _st_shared_v2_f32(gate_smem + off_hi, g2 + gf[2], g3 + gf[3])
                        u0, u1 = _ld_shared_v2_f32(up_smem + off_lo)
                        _st_shared_v2_f32(up_smem + off_lo, u0 + uf[0], u1 + uf[1])
                        u2, u3 = _ld_shared_v2_f32(up_smem + off_hi)
                        _st_shared_v2_f32(up_smem + off_hi, u2 + uf[2], u3 + uf[3])
            cute.arch.sync_threads()

            for r_it in cutlass.range_constexpr(rows_per_warp):
                local_row = warp_idx + Int32(r_it * 16)
                abs_row = row_start + local_row
                if abs_row < valid_rows:
                    row_smem_off = local_row * Int32(128 * 4)
                    row_swiz = (local_row & Int32(3)) << Int32(3)
                    _apply_intermediate_hadamard_and_store_f16(
                        gate_smem + row_smem_off,
                        up_smem + row_smem_off,
                        row_swiz,
                        trellis_rotations,
                        trellis_rotations_u16,
                        sign_patterns,
                        sign_table,
                        intermediate_u32,
                        expert_idx,
                        output_tile,
                        physical_row_base + Int64(abs_row),
                        intermediate_tiles,
                        packed_rot,
                        lane,
                        self.is_situ,
                        self.fast_math,
                    )
            cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        a_f16_u32: cute.Tensor,
        w13_u32: cute.Tensor,
        intermediate_u32: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
        trellis_rotations_u16: cute.Tensor,
        sign_patterns: cute.Tensor,
        sign_table: cute.Tensor,
        live_routes: cutlass.Int32,
        input_k64_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_rot: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(tid >> Int32(5))

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        _load_direct_lut_smem(
            trellis_lut,
            smem_base + Int32(self.trellis_lut_offset),
            self.trellis_bits,
            tid,
            self.threads_per_cta,
        )

        tok_smem = smem_base + Int32(self.tok_smem_offset)
        meta_smem = smem_base + Int32(self.meta_smem_offset)

        num_m_tiles = packed_route_count[Int32(0)].to(Int32) // Int32(self.tile_m)
        task_tail = num_m_tiles * intermediate_tiles
        task_slot = Int32(bidx)
        while task_slot < task_tail:
            output_tile = task_slot % intermediate_tiles
            m_tile = task_slot // intermediate_tiles
            expert_idx = block_expert_ids[m_tile].to(Int32)
            if expert_idx >= Int32(0):
                if tid < Int32(64):
                    r = packed_route_indices[m_tile * Int32(64) + tid].to(Int32)
                    is_v = Int32(0)
                    tok = Int32(0)
                    if r >= Int32(0) and r < live_routes:
                        is_v = Int32(1)
                        tok = r // Int32(self.top_k)
                    st_shared_i32(tok_smem + (tid << Int32(2)), tok)
                    wsum = _warp_sum_i32(is_v)
                    if (tid & Int32(31)) == Int32(0):
                        st_shared_i32(meta_smem + ((tid >> Int32(5)) << Int32(2)), wsum)
                cute.arch.sync_threads()
                valid_rows = (
                    ld_shared_i32_relaxed(meta_smem)
                    + ld_shared_i32_relaxed(meta_smem + Int32(4))
                )
                if valid_rows > Int32(0):
                    if valid_rows <= Int32(16):
                        self._run_task(
                            a_f16_u32, w13_u32, intermediate_u32, trellis_rotations,
                            trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile, expert_idx, output_tile,
                            valid_rows, input_k64_tiles, intermediate_tiles, packed_rot, 16,
                        )
                    elif valid_rows <= Int32(32):
                        self._run_task(
                            a_f16_u32, w13_u32, intermediate_u32, trellis_rotations,
                            trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile, expert_idx, output_tile,
                            valid_rows, input_k64_tiles, intermediate_tiles, packed_rot, 32,
                        )
                    else:
                        self._run_task(
                            a_f16_u32, w13_u32, intermediate_u32, trellis_rotations,
                            trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile, expert_idx, output_tile,
                            valid_rows, input_k64_tiles, intermediate_tiles, packed_rot, 64,
                        )
            task_slot += Int32(gdimx)


class Exl3W2A16PrefillFC2Kernel16W:
    """16-warp (8N x 2K) Contiguous-A, Zero-B-SMEM, 64 KiB Direct-LUT, N256 Lossless W2A16 Prefill FC2."""

    tile_m = 64
    tile_n = 256
    tile_k = 64
    num_warps = 16
    threads_per_cta = 512

    def __init__(
        self,
        *,
        top_k: int,
        trellis_bits: int = 2,
        stages: int = 4,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3W2A16PrefillFC2Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3, 4):
            raise ValueError(f"stages must be 2, 3, or 4, got {stages}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.stages = int(stages)

        self.stage_bytes = self.tile_m * self.tile_k * 2  # 8,192 B (for M=64)
        self.pipeline_bytes = self.stages * self.stage_bytes  # 32,768 B
        self.epilogue_bytes = 32 * self.tile_n * 4  # 32,768 B
        self.work_bytes = max(self.pipeline_bytes, self.epilogue_bytes)

        self.tok_smem_offset = self.work_bytes
        self.wt_smem_offset = self.tok_smem_offset + self.tile_m * 4
        self.meta_smem_offset = self.wt_smem_offset + self.tile_m * 4
        self.trellis_lut_offset = ((self.meta_smem_offset + 16 + 255) // 256) * 256
        self.total_smem_bytes = self.trellis_lut_offset + 65536
        if self.total_smem_bytes > 101376:
            raise ValueError(f"SMEM footprint {self.total_smem_bytes} exceeds 101,376 B")
        self.shared_words = (self.total_smem_bytes + 3) // 4

    @cute.jit
    def __call__(
        self,
        intermediate_u32: cute.Tensor,
        w2_u32: cute.Tensor,
        scatter_output: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        trellis_lut: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_k64_tiles: cutlass.Int32,
        output_n256_tiles: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            intermediate_u32,
            w2_u32,
            scatter_output,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            trellis_lut,
            live_routes,
            intermediate_k64_tiles,
            output_n256_tiles,
        ).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            min_blocks_per_mp=1,
            preferred_smem_carveout=100,
            stream=stream,
        )

    @cute.jit
    def _stage_a_slice(
        self,
        intermediate_u32: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        m_tile: Int32,
        k64_slice: Int32,
        intermediate_k64_tiles: Int32,
        stage_bytes_eff: cutlass.Constexpr[int],
        stages_eff: cutlass.Constexpr[int],
        compute_m: cutlass.Constexpr[int],
    ):
        stage = k64_slice % Int32(stages_eff)
        stage_base = smem_base + stage * Int32(stage_bytes_eff)
        words_per_row = intermediate_k64_tiles * Int32(32)
        physical_row_base = m_tile * Int32(self.tile_m)
        if tid < Int32(int(compute_m) * 8):
            row = tid >> Int32(3)
            vec = tid & Int32(7)
            phys_vec = vec ^ (row & Int32(7))
            src_word = (
                Int64(physical_row_base + row) * Int64(words_per_row)
                + Int64(k64_slice * Int32(32) + (vec << Int32(2)))
            )
            cp_async4_shared_global(
                stage_base + (row << Int32(7)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(intermediate_u32, src_word),
            )

    @cute.jit
    def _run_task(
        self,
        intermediate_u32: cute.Tensor,
        w2_u32: cute.Tensor,
        scatter_output: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        m_tile: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        intermediate_k64_tiles: Int32,
        output_n256_tiles: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        warp_n = warp_idx & Int32(7)
        warp_k = warp_idx >> Int32(3)

        ia, ib, s2, _ = trellis256_lane_geom_bits(lane, 0, 8, 2)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))

        tr_n16_cnt = output_n256_tiles * Int32(16)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(intermediate_k64_tiles * Int32(4)) * tr_k16_stride
        tr_warp_base = (
            Int64(expert_idx) * tr_eu
            + Int64(output_tile * Int32(16) + warp_n * Int32(2)) * Int64(16)
            + Int64(warp_k * Int32(2)) * tr_k16_stride
        )
        tr_k64_step = Int64(4) * tr_k16_stride

        # Compact per-slice bytes let compute_m <= 32 preload ALL 6 K64 slices (ZERO K-loop barriers!)
        stage_bytes_eff = int(compute_m) * self.tile_k * 2
        stages_eff = min(6, self.work_bytes // stage_bytes_eff)
        preload_all = intermediate_k64_tiles <= Int32(stages_eff)
        if preload_all:
            for s in cutlass.range_constexpr(stages_eff):
                if Int32(s) < intermediate_k64_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, Int32(s),
                        intermediate_k64_tiles, stage_bytes_eff, stages_eff, compute_m,
                    )
            cute.arch.cp_async_commit_group()
        else:
            for s in cutlass.range_constexpr(stages_eff - 1):
                if Int32(s) < intermediate_k64_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, Int32(s),
                        intermediate_k64_tiles, stage_bytes_eff, stages_eff, compute_m,
                    )
                cute.arch.cp_async_commit_group()

        w_slice_base = tr_warp_base
        raw_n0 = _load_trellis_k32_raw_u32(w2_u32, w_slice_base, tr_k16_stride, lane)
        raw_n1 = _load_trellis_k32_raw_u32(
            w2_u32, w_slice_base + Int64(16), tr_k16_stride, lane
        )

        if preload_all:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        m_blocks = int(compute_m) // 16
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), Float32) for _nt in range(4))
            for _blk in range(m_blocks)
        )
        for blk in cutlass.range_constexpr(m_blocks):
            for nt in cutlass.range_constexpr(4):
                facc[blk][nt].fill(0.0)

        ldm_row = lane & Int32(15)
        ldm_k8 = lane >> Int32(4)
        ldm_swiz = ldm_row & Int32(7)
        a_off_k0 = (ldm_row << Int32(7)) + (((warp_k * Int32(4) + ldm_k8) ^ ldm_swiz) << Int32(4))
        a_off_k1 = (ldm_row << Int32(7)) + (
            ((warp_k * Int32(4) + Int32(2) + ldm_k8) ^ ldm_swiz) << Int32(4)
        )

        k64_slice = Int32(0)
        while k64_slice < intermediate_k64_tiles:
            stage = k64_slice % Int32(stages_eff)
            stage_base = smem_base + stage * Int32(stage_bytes_eff)

            if not preload_all:
                next_a = k64_slice + Int32(stages_eff - 1)
                if next_a < intermediate_k64_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, next_a,
                        intermediate_k64_tiles, stage_bytes_eff, stages_eff, compute_m,
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(stages_eff - 1)
                cute.arch.sync_threads()

            nxt_n0 = raw_n0
            nxt_n1 = raw_n1
            next_b_base = w_slice_base + tr_k64_step
            if k64_slice + Int32(1) < intermediate_k64_tiles:
                nxt_n0 = _load_trellis_k32_raw_u32(w2_u32, next_b_base, tr_k16_stride, lane)
                nxt_n1 = _load_trellis_k32_raw_u32(
                    w2_u32, next_b_base + Int64(16), tr_k16_stride, lane
                )

            for k16_local in cutlass.range_constexpr(2):
                ia_k = ia if cutlass.const_expr(k16_local == 0) else (ia + Int32(16))
                ib_k = ib if cutlass.const_expr(k16_local == 0) else (ib + Int32(16))
                a_off_k = a_off_k0 if cutlass.const_expr(k16_local == 0) else a_off_k1

                d0, d1, d2, d3 = _decode_trellis_k16_to_f16x4(
                    raw_n0, ia_k, ib_k, s2, trellis_lut_addr
                )
                e0, e1, e2, e3 = _decode_trellis_k16_to_f16x4(
                    raw_n1, ia_k, ib_k, s2, trellis_lut_addr
                )

                for blk in cutlass.range_constexpr(m_blocks):
                    a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                        stage_base + Int32(blk * 16 * 128) + a_off_k
                    )
                    f0 = facc[blk][0]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        f0[0], f0[1], f0[2], f0[3], a0, a1, a2, a3, d0, d1
                    )
                    f0[0], f0[1], f0[2], f0[3] = r0, r1, r2, r3

                    f1 = facc[blk][1]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        f1[0], f1[1], f1[2], f1[3], a0, a1, a2, a3, d2, d3
                    )
                    f1[0], f1[1], f1[2], f1[3] = r0, r1, r2, r3

                    f2 = facc[blk][2]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        f2[0], f2[1], f2[2], f2[3], a0, a1, a2, a3, e0, e1
                    )
                    f2[0], f2[1], f2[2], f2[3] = r0, r1, r2, r3

                    f3 = facc[blk][3]
                    r0, r1, r2, r3 = f16_mma_m16n8k16_f32(
                        f3[0], f3[1], f3[2], f3[3], a0, a1, a2, a3, e2, e3
                    )
                    f3[0], f3[1], f3[2], f3[3] = r0, r1, r2, r3

            raw_n0 = nxt_n0
            raw_n1 = nxt_n1
            if not preload_all and k64_slice + Int32(stages_eff) < intermediate_k64_tiles:
                cute.arch.sync_threads()
            k64_slice += Int32(1)
            w_slice_base = next_b_base

        if not preload_all:
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        scatter_n = Int32(scatter_output.shape[1])
        tok_smem = smem_base + Int32(self.tok_smem_offset)
        wt_smem = smem_base + Int32(self.wt_smem_offset)
        col_base_local = warp_n * Int32(32) + (c << Int32(1))
        col_base_global = output_tile * Int32(self.tile_n) + col_base_local
        swiz_q = (q & Int32(3)) << Int32(3)
        num_halves = 2 if int(compute_m) == 64 else 1
        blks_per_half = 1 if int(compute_m) == 16 else 2

        for half in cutlass.range_constexpr(num_halves):
            if warp_k == Int32(0):
                for hb in cutlass.range_constexpr(blks_per_half):
                    blk = half * 2 + hb
                    row_lo = Int32(hb * 16) + q
                    row_hi = row_lo + Int32(8)
                    for nt in cutlass.range_constexpr(4):
                        swiz_col = (col_base_local + Int32(nt * 8)) ^ swiz_q
                        off_lo = (row_lo * Int32(256) + swiz_col) << Int32(2)
                        off_hi = (row_hi * Int32(256) + swiz_col) << Int32(2)
                        f = facc[blk][nt]
                        _st_shared_v2_f32(smem_base + off_lo, f[0], f[1])
                        _st_shared_v2_f32(smem_base + off_hi, f[2], f[3])
            cute.arch.sync_threads()
            if warp_k == Int32(1):
                for hb in cutlass.range_constexpr(blks_per_half):
                    blk = half * 2 + hb
                    row_lo = Int32(hb * 16) + q
                    row_hi = row_lo + Int32(8)
                    abs_lo = Int32(blk * 16) + q
                    abs_hi = abs_lo + Int32(8)
                    if abs_lo < valid_rows:
                        tok_lo = ld_shared_i32_relaxed(tok_smem + (abs_lo << Int32(2)))
                        wt_lo = ld_shared_f32(wt_smem + (abs_lo << Int32(2)))
                        for nt in cutlass.range_constexpr(4):
                            swiz_col = (col_base_local + Int32(nt * 8)) ^ swiz_q
                            col_g = col_base_global + Int32(nt * 8)
                            off_lo = (row_lo * Int32(256) + swiz_col) << Int32(2)
                            f = facc[blk][nt]
                            s0, s1 = _ld_shared_v2_f32(smem_base + off_lo)
                            scatter_add_bf16x2(
                                get_ptr_as_int64(
                                    scatter_output,
                                    Int64(tok_lo) * Int64(scatter_n) + Int64(col_g),
                                ),
                                (s0 + f[0]) * wt_lo,
                                (s1 + f[1]) * wt_lo,
                            )
                    if abs_hi < valid_rows:
                        tok_hi = ld_shared_i32_relaxed(tok_smem + (abs_hi << Int32(2)))
                        wt_hi = ld_shared_f32(wt_smem + (abs_hi << Int32(2)))
                        for nt in cutlass.range_constexpr(4):
                            swiz_col = (col_base_local + Int32(nt * 8)) ^ swiz_q
                            col_g = col_base_global + Int32(nt * 8)
                            off_hi = (row_hi * Int32(256) + swiz_col) << Int32(2)
                            f = facc[blk][nt]
                            sv2, sv3 = _ld_shared_v2_f32(smem_base + off_hi)
                            scatter_add_bf16x2(
                                get_ptr_as_int64(
                                    scatter_output,
                                    Int64(tok_hi) * Int64(scatter_n) + Int64(col_g),
                                ),
                                (sv2 + f[2]) * wt_hi,
                                (sv3 + f[3]) * wt_hi,
                            )
            cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        w2_u32: cute.Tensor,
        scatter_output: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        trellis_lut: cute.Tensor,
        live_routes: cutlass.Int32,
        intermediate_k64_tiles: cutlass.Int32,
        output_n256_tiles: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(tid >> Int32(5))

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())
        _load_direct_lut_smem(
            trellis_lut,
            smem_base + Int32(self.trellis_lut_offset),
            self.trellis_bits,
            tid,
            self.threads_per_cta,
        )

        tok_smem = smem_base + Int32(self.tok_smem_offset)
        wt_smem = smem_base + Int32(self.wt_smem_offset)
        meta_smem = smem_base + Int32(self.meta_smem_offset)

        num_m_tiles = packed_route_count[Int32(0)].to(Int32) // Int32(self.tile_m)
        task_tail = num_m_tiles * output_n256_tiles
        task_slot = Int32(bidx)
        while task_slot < task_tail:
            output_tile = task_slot % output_n256_tiles
            m_tile = task_slot // output_n256_tiles
            expert_idx = block_expert_ids[m_tile].to(Int32)
            if expert_idx >= Int32(0):
                if tid < Int32(64):
                    r = packed_route_indices[m_tile * Int32(64) + tid].to(Int32)
                    is_v = Int32(0)
                    tok = Int32(0)
                    wt = Float32(0.0)
                    if r >= Int32(0) and r < live_routes:
                        is_v = Int32(1)
                        tok = r // Int32(self.top_k)
                        wt = topk_weights_flat[r].to(Float32)
                    st_shared_i32(tok_smem + (tid << Int32(2)), tok)
                    st_shared_f32(wt_smem + (tid << Int32(2)), wt)
                    wsum = _warp_sum_i32(is_v)
                    if (tid & Int32(31)) == Int32(0):
                        st_shared_i32(meta_smem + ((tid >> Int32(5)) << Int32(2)), wsum)
                cute.arch.sync_threads()
                valid_rows = (
                    ld_shared_i32_relaxed(meta_smem)
                    + ld_shared_i32_relaxed(meta_smem + Int32(4))
                )
                if valid_rows > Int32(0):
                    if valid_rows <= Int32(16):
                        self._run_task(
                            intermediate_u32, w2_u32, scatter_output, smem_base, tid,
                            warp_idx, m_tile, expert_idx, output_tile, valid_rows,
                            intermediate_k64_tiles, output_n256_tiles, 16,
                        )
                    elif valid_rows <= Int32(32):
                        self._run_task(
                            intermediate_u32, w2_u32, scatter_output, smem_base, tid,
                            warp_idx, m_tile, expert_idx, output_tile, valid_rows,
                            intermediate_k64_tiles, output_n256_tiles, 32,
                        )
                    else:
                        self._run_task(
                            intermediate_u32, w2_u32, scatter_output, smem_base, tid,
                            warp_idx, m_tile, expert_idx, output_tile, valid_rows,
                            intermediate_k64_tiles, output_n256_tiles, 64,
                        )
            task_slot += Int32(gdimx)


__all__ = [
    "Exl3ActHadF16Kernel",
    "Exl3W2A16DecodeFC1Kernel16W",
    "Exl3W2A16DecodeFC2Kernel16W",
    "Exl3W2A16PrefillFC1Kernel16W",
    "Exl3W2A16PrefillFC2Kernel16W",
]
