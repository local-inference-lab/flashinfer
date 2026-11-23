"""16-warp (8N x 2K) SMEM-Direct-LUT W2A8-MXFP8 MoE kernels for EXL3 2-bit Kimi.

v4 Hardware-Roofline Architecture:
  - 64 KiB ``lut_e4m3`` direct table resident in SMEM (proven SM120 floor: 788 us).
  - ZERO 2-bit weight staging through SMEM: coalesced 1-LDG ``ld.global.nc.u32`` +
    ``shfl.sync`` directly into registers.
  - ZERO-bank-conflict A staging/loading using ``vec ^ ((row & 3) << 1)`` (fixes the
    2-way half-warp bank conflict of ``vec ^ (row & 7)`` on 64-bit ``ld.shared.v2.u32``).
  - Spill-free FC1 (<= 128 regs/thread, 0 local-memory sectors): 2-reg K64-half B
    prefetching + per-M16-block A streaming (4 regs).
  - FC2 N256 (H=3584 = 14 x 256): preloads all ``<= stages`` K128 slices of A before
    the K-loop (ZERO K-loop barriers for Kimi I_local=384!), plus optional direct
    FP32 atomic reduction (``direct_f32_atomic=True``) eliminating ALL epilogue SMEM
    and epilogue barriers while upgrading cross-expert sum precision from BF16 to FP32.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint32, dsl_user_op

from b12x._lib.intrinsics.atomics import (
    ld_shared_f32,
    ld_shared_i32_relaxed,
    ld_shared_u32,
    ld_shared_v4_f32,
    scatter_add_bf16x2,
    st_shared_f32,
    st_shared_i32,
    st_shared_v4_f32,
)
from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_f32x4_to_e4m3x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_shared_v2_u32,
    shared_ptr_to_u32,
)
from b12x._lib.intrinsics.mma import mxfp8_mma_m16n8k32_f32_e4m3
from b12x._lib.intrinsics.trellis import packed_decode_lut_e4m3_direct_to_e4m3x8
from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_trellis_lane_geom,
)


@dsl_user_op
def _st_shared_v2_f32(addr: Int32, v0: Float32, v1: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_v2_f32(addr: Int32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _red_add_global_v2_f32(ptr: Int64, v0: Float32, v1: Float32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.v2.f32 [$0], {$1, $2};",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _warp_sum_i32(val: Int32) -> Int32:
    val = val + cute.arch.shuffle_sync_bfly(val, offset=16)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=8)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=4)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=2)
    val = val + cute.arch.shuffle_sync_bfly(val, offset=1)
    return val


@cute.jit
def _load_direct_lut_smem(
    trellis_lut: cute.Tensor,
    dst_smem_base: Int32,
    trellis_bits: cutlass.Constexpr[int],
    tid: Int32,
    threads_per_cta: cutlass.Constexpr[int],
):
    trellis_lut_u32 = cute.recast_tensor(trellis_lut, cutlass.Uint32)
    rate_word_off = Int64(((int(trellis_bits) - 2) << 16) // 4)
    for i in cutlass.range_constexpr(4096 // int(threads_per_cta)):
        chunk = tid + Int32(i * int(threads_per_cta))
        cp_async4_shared_global(
            dst_smem_base + (chunk << Int32(4)),
            get_ptr_as_int64(
                trellis_lut_u32, rate_word_off + Int64(chunk * Int32(4))
            ),
        )
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()


@cute.jit
def _load_trellis_k32_raw_u32(
    w_u32: cute.Tensor,
    base0_gmem: Int64,
    k16_stride_u32: Int64,
    lane: Int32,
) -> Uint32:
    half_lane = lane & Int32(15)
    is_k1 = lane >> Int32(4)
    idx = base0_gmem + Int64(is_k1) * k16_stride_u32 + Int64(half_lane)
    return ld_global_nc_u32(get_ptr_as_int64(w_u32, idx))


@cute.jit
def _decode_reg_k32_pair_both(
    raw_b: Uint32,
    lane: Int32,
    ia: Int32,
    ib: Int32,
    s2: Int32,
    lut_addr: Int64,
):
    a0 = Uint32(cute.arch.shuffle_sync(raw_b, ia))
    b0 = Uint32(cute.arch.shuffle_sync(raw_b, ib))
    m0 = (Int64(a0) << Int64(32)) | Int64(b0)
    wa0 = Uint32(m0 >> Int64(s2))
    wb0 = Uint32(m0 >> Int64(s2 + Int32(8)))
    e0_lo, e0_hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
        wa0, wb0, lut_addr, 2, rate_indexed=False, in_shared=True
    )

    a1 = Uint32(cute.arch.shuffle_sync(raw_b, ia + Int32(16)))
    b1 = Uint32(cute.arch.shuffle_sync(raw_b, ib + Int32(16)))
    m1 = (Int64(a1) << Int64(32)) | Int64(b1)
    wa1 = Uint32(m1 >> Int64(s2))
    wb1 = Uint32(m1 >> Int64(s2 + Int32(8)))
    e1_lo, e1_hi = packed_decode_lut_e4m3_direct_to_e4m3x8(
        wa1, wb1, lut_addr, 2, rate_indexed=False, in_shared=True
    )

    c = lane & Int32(3)
    own_lo = e0_lo
    own_hi = e0_hi
    send_lo = e1_lo
    send_hi = e1_hi
    if c >= Int32(2):
        own_lo = e1_lo
        own_hi = e1_hi
        send_lo = e0_lo
        send_hi = e0_hi
    peer_lo = Uint32(cute.arch.shuffle_sync_bfly(send_lo, offset=2))
    peer_hi = Uint32(cute.arch.shuffle_sync_bfly(send_hi, offset=2))
    return own_lo, peer_lo, own_hi, peer_hi


class Exl3MxFC1Kernel16W:
    """16-warp (8N x 2K) Spill-Free, Bank-Free-A, Zero-B-SMEM W2A8-MXFP8 FC1."""

    tile_m = 64
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
        intermediate_hadamard: bool = True,
        stages: int = 3,
        fast_math: bool = True,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3MxFC1Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3):
            raise ValueError(f"stages must be 2 or 3, got {stages}")
        if activation not in ("silu", SITU):
            raise ValueError(f"unsupported activation {activation!r}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.is_situ = activation == SITU
        self.intermediate_hadamard = bool(intermediate_hadamard)
        self.stages = int(stages)
        self.fast_math = bool(fast_math)

        self.a_offset = 0
        self.a_payload_bytes = self.tile_m * self.tile_k  # 8,192 B
        self.sfa_offset = self.a_offset + self.a_payload_bytes
        self.a_scale_bytes = self.tile_m * 4  # 256 B
        self.stage_bytes = self.sfa_offset + self.a_scale_bytes  # 8,448 B
        self.pipeline_bytes = self.stages * self.stage_bytes

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
        packed_a_u32: cute.Tensor,
        a_scales_u32: cute.Tensor,
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
        rows_capacity: cutlass.Int32,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_rot: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            packed_a_u32,
            a_scales_u32,
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
            rows_capacity,
            input_k128_tiles,
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
    def _activated(self, gate: cutlass.Float32, up: cutlass.Float32) -> cutlass.Float32:
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        if cutlass.const_expr(self.is_situ):
            beta = cutlass.Float32(SITU_DEFAULT_BETA)
            linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
            situ_gate = beta * cute.math.tanh(gate / beta, fastmath=self.fast_math) * sigmoid
            situ_up = linear_beta * cute.math.tanh(up / linear_beta, fastmath=self.fast_math)
            return situ_gate * situ_up
        return gate * sigmoid * up

    @cute.jit
    def _stage_a_slice(
        self,
        packed_a_u32: cute.Tensor,
        a_scales_u32: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k128_slice: Int32,
        input_k128_tiles: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        stage = k128_slice % Int32(self.stages)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        tok_smem = smem_base + Int32(self.tok_smem_offset)

        words_per_token = input_k128_tiles * Int32(32)
        if tid < Int32(int(compute_m) * 8):
            row = tid >> Int32(3)
            vec = tid & Int32(7)
            tok = ld_shared_i32_relaxed(tok_smem + (row << Int32(2)))
            # Zero bank conflicts for 64-bit ld_shared_v2_u32 (q=0..3 hit banks 0..31!)
            phys_vec = vec ^ ((row & Int32(3)) << Int32(1))
            src_word = Int64(tok) * Int64(words_per_token) + Int64(
                k128_slice * Int32(32) + (vec << Int32(2))
            )
            cp_async4_shared_global(
                a_base + (row << Int32(7)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(packed_a_u32, src_word),
            )
        if tid < Int32(compute_m):
            tok = ld_shared_i32_relaxed(tok_smem + (tid << Int32(2)))
            sf_src = Int64(tok) * Int64(input_k128_tiles) + Int64(k128_slice)
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(a_scales_u32, sf_src),
            )

    @cute.jit
    def _run_task(
        self,
        packed_a_u32: cute.Tensor,
        a_scales_u32: cute.Tensor,
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
        rows_capacity: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        packed_rot: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        warp_n = warp_idx & Int32(7)
        warp_k = warp_idx >> Int32(3)

        tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(lane, self.trellis_bits)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))

        tr_n16_cnt = intermediate_tiles * Int32(8)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(input_k128_tiles * Int32(8)) * tr_k16_stride
        tr_w13_half = Int64(w13_u32.shape[0]) >> Int64(1)
        tr_warp_col_base = (
            Int64(expert_idx) * tr_eu
            + Int64(output_tile * Int32(8) + warp_n) * Int64(16)
            + Int64(warp_k * Int32(4)) * tr_k16_stride
        )
        tr_k128_step = Int64(8) * tr_k16_stride
        tr_k32_step = Int64(2) * tr_k16_stride

        for preload in cutlass.range_constexpr(self.stages - 1):
            if Int32(preload) < input_k128_tiles:
                self._stage_a_slice(
                    packed_a_u32,
                    a_scales_u32,
                    smem_base,
                    tid,
                    Int32(preload),
                    input_k128_tiles,
                    compute_m,
                )
            cute.arch.cp_async_commit_group()

        # 4-register K128-slice-ahead software pipeline (hides full 400-cycle GDDR7 latency!)
        g_slice_base = tr_warp_col_base
        raw_g0 = _load_trellis_k32_raw_u32(w13_u32, g_slice_base, tr_k16_stride, lane)
        raw_u0 = _load_trellis_k32_raw_u32(w13_u32, g_slice_base + tr_w13_half, tr_k16_stride, lane)
        raw_g1 = _load_trellis_k32_raw_u32(w13_u32, g_slice_base + tr_k32_step, tr_k16_stride, lane)
        raw_u1 = _load_trellis_k32_raw_u32(w13_u32, g_slice_base + tr_w13_half + tr_k32_step, tr_k16_stride, lane)

        m_blocks = int(compute_m) // 16
        gate_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(2))
            for _blk in range(m_blocks)
        )
        up_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(2))
            for _blk in range(m_blocks)
        )
        for blk in cutlass.range_constexpr(m_blocks):
            for nt in cutlass.range_constexpr(2):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)

        unit_sfb = Uint32(0x7F7F7F7F)
        warp_k_scale_shift = Uint32(warp_k * Int32(16))
        sf_lane_off = q + ((lane & Int32(1)) << Int32(3))
        q_swiz = (q & Int32(3)) << Int32(1)
        c_sub = (c & Int32(1)) << Int32(3)

        k128_slice = Int32(0)
        while k128_slice < input_k128_tiles:
            stage = k128_slice % Int32(self.stages)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)

            next_a_slice = k128_slice + Int32(self.stages - 1)
            if next_a_slice < input_k128_tiles:
                self._stage_a_slice(
                    packed_a_u32,
                    a_scales_u32,
                    smem_base,
                    tid,
                    next_a_slice,
                    input_k128_tiles,
                    compute_m,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()

            nxt_g0, nxt_u0, nxt_g1, nxt_u1 = raw_g0, raw_u0, raw_g1, raw_u1
            next_b_base = g_slice_base + tr_k128_step
            if k128_slice + Int32(1) < input_k128_tiles:
                nxt_g0 = _load_trellis_k32_raw_u32(w13_u32, next_b_base, tr_k16_stride, lane)
                nxt_u0 = _load_trellis_k32_raw_u32(w13_u32, next_b_base + tr_w13_half, tr_k16_stride, lane)
                nxt_g1 = _load_trellis_k32_raw_u32(w13_u32, next_b_base + tr_k32_step, tr_k16_stride, lane)
                nxt_u1 = _load_trellis_k32_raw_u32(w13_u32, next_b_base + tr_w13_half + tr_k32_step, tr_k16_stride, lane)

            for kb_local in cutlass.range_constexpr(2):
                kb = warp_k * Int32(2) + Int32(kb_local)
                u_phys = (kb * Int32(2) + (c >> Int32(1))) ^ q_swiz
                a_row_base = a_base + (q << Int32(7)) + (u_phys << Int32(4)) + c_sub
                rg = raw_g0 if cutlass.const_expr(kb_local == 0) else raw_g1
                ru = raw_u0 if cutlass.const_expr(kb_local == 0) else raw_u1

                g_lo0, g_lo1, g_hi0, g_hi1 = _decode_reg_k32_pair_both(
                    rg, lane, tr_ia, tr_ib, tr_s2, trellis_lut_addr
                )
                u_lo0, u_lo1, u_hi0, u_hi1 = _decode_reg_k32_pair_both(
                    ru, lane, tr_ia, tr_ib, tr_s2, trellis_lut_addr
                )
                for blk in cutlass.range_constexpr(m_blocks):
                    asc_v = ld_shared_u32(sfa_base + ((Int32(blk * 16) + sf_lane_off) << Int32(2))) >> warp_k_scale_shift
                    a_lo = a_row_base + Int32(blk * 16 * 128)
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * 128))

                    ga0 = gate_acc[blk][0]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        ga0[0], ga0[1], ga0[2], ga0[3],
                        a0, a1, a2, a3, g_lo0, g_lo1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    ga0[0], ga0[1], ga0[2], ga0[3] = r0, r1, r2, r3

                    ga1 = gate_acc[blk][1]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        ga1[0], ga1[1], ga1[2], ga1[3],
                        a0, a1, a2, a3, g_hi0, g_hi1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    ga1[0], ga1[1], ga1[2], ga1[3] = r0, r1, r2, r3

                    ua0 = up_acc[blk][0]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        ua0[0], ua0[1], ua0[2], ua0[3],
                        a0, a1, a2, a3, u_lo0, u_lo1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    ua0[0], ua0[1], ua0[2], ua0[3] = r0, r1, r2, r3

                    ua1 = up_acc[blk][1]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        ua1[0], ua1[1], ua1[2], ua1[3],
                        a0, a1, a2, a3, u_hi0, u_hi1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    ua1[0], ua1[1], ua1[2], ua1[3] = r0, r1, r2, r3

            raw_g0, raw_u0, raw_g1, raw_u1 = nxt_g0, nxt_u0, nxt_g1, nxt_u1
            if k128_slice + Int32(self.stages) < input_k128_tiles:
                cute.arch.sync_threads()
            k128_slice += Int32(1)
            g_slice_base = next_b_base

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Epilogue: Bank-free K-reduce + intermediate_hadamard + warp-parallel MXFP8 K32 store
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

        gate_smem = smem_base
        up_smem = smem_base + Int32(self.epilogue_half_bytes)
        col_base = warp_n * Int32(16) + (c << Int32(1))
        swiz_q = (q & Int32(3)) << Int32(3)
        physical_row_base = Int64(m_tile) * Int64(self.tile_m)
        words_per_row = intermediate_tiles * Int32(32)
        sf_base = Int64(rows_capacity) * Int64(words_per_row)

        k32_local = lane >> Int32(3)
        sub_lane = lane & Int32(7)
        p_base = (
            ((sub_lane >> Int32(2)) << Int32(4))
            + ((((sub_lane & Int32(1)) << Int32(2)) | (sub_lane & Int32(2))) ^ (sub_lane & Int32(4)))
        )

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
                    aa0 = cutlass.Float32(0.0)
                    aa1 = cutlass.Float32(0.0)
                    bb0 = cutlass.Float32(0.0)
                    bb1 = cutlass.Float32(0.0)
                    for _pb in cutlass.range_constexpr(2):
                        ci_raw = tr_ci0 if cutlass.const_expr(_pb == 0) else tr_ci1
                        tr_ci = ci_raw ^ row_swiz
                        base_src = gate_smem if tr_slot == Int32(0) else up_smem
                        p0, p1, p2, p3 = ld_shared_v4_f32(base_src + row_smem_off + (tr_ci << Int32(2)))
                        h0, h1, h2, h3 = _w4a8_had128_quad(p0, p1, p2, p3, lane)
                        col_gu = output_tile * Int32(128) + ci_raw
                        r0 = cutlass.Float32(0.0)
                        r1 = cutlass.Float32(0.0)
                        r2 = cutlass.Float32(0.0)
                        r3 = cutlass.Float32(0.0)
                        if packed_rot != Int32(0):
                            stride = tr_isz + (tr_isz >> Int32(3)) + Int32(8)
                            row_b = expert_idx * stride
                            gain = cutlass.Float32(
                                trellis_rotations[row_b + tr_isz + (tr_isz >> Int32(3)) + tr_slot]
                            )
                            ng = -gain
                            bits = Int32(
                                trellis_rotations_u16[
                                    row_b + tr_isz + tr_slot * (tr_isz >> Int32(4)) + (col_gu >> Int32(4))
                                ]
                            )
                            sh = col_gu & Int32(15)
                            r0 = ng if ((bits >> sh) & Int32(1)) != Int32(0) else gain
                            r1 = ng if ((bits >> (sh + Int32(1))) & Int32(1)) != Int32(0) else gain
                            r2 = ng if ((bits >> (sh + Int32(2))) & Int32(1)) != Int32(0) else gain
                            r3 = ng if ((bits >> (sh + Int32(3))) & Int32(1)) != Int32(0) else gain
                        else:
                            sc_idx = expert_idx * (Int32(3) * tr_isz) + tr_slot * tr_isz + col_gu
                            r0 = cutlass.Float32(trellis_rotations[sc_idx])
                            r1 = cutlass.Float32(trellis_rotations[sc_idx + Int32(1)])
                            r2 = cutlass.Float32(trellis_rotations[sc_idx + Int32(2)])
                            r3 = cutlass.Float32(trellis_rotations[sc_idx + Int32(3)])
                        h0 = h0 * r0
                        h1 = h1 * r1
                        h2 = h2 * r2
                        h3 = h3 * r3
                        h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, lane)
                        sg_idx = tr_sg0_idx if cutlass.const_expr(_pb == 0) else tr_sg1_idx
                        h0 = h0 * cutlass.Float32(sign_table[sg_idx])
                        h1 = h1 * cutlass.Float32(sign_table[sg_idx + Int32(1)])
                        h2 = h2 * cutlass.Float32(sign_table[sg_idx + Int32(2)])
                        h3 = h3 * cutlass.Float32(sign_table[sg_idx + Int32(3)])
                        s0 = self._activated(h0, h1)
                        s1 = self._activated(h2, h3)
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
                    o0 = o0 * cutlass.Float32(sign_table[tr_ub_idx])
                    o1 = o1 * cutlass.Float32(sign_table[tr_ub_idx + Int32(1)])
                    o2 = o2 * cutlass.Float32(sign_table[tr_ub_idx + Int32(2)])
                    o3 = o3 * cutlass.Float32(sign_table[tr_ub_idx + Int32(3)])
                    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)
                    o0 = o0 * cutlass.Float32(trellis_rotations[tr_dn_idx])
                    o1 = o1 * cutlass.Float32(trellis_rotations[tr_dn_idx + Int32(1)])
                    o2 = o2 * cutlass.Float32(trellis_rotations[tr_dn_idx + Int32(2)])
                    o3 = o3 * cutlass.Float32(trellis_rotations[tr_dn_idx + Int32(3)])
                    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)

                    bmax = fmax_f32(
                        fmax_f32(fabs_f32(o0), fabs_f32(o1)),
                        fmax_f32(fabs_f32(o2), fabs_f32(o3)),
                    )
                    bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=1))
                    bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=2))
                    bmax = fmax_f32(bmax, cute.arch.shuffle_sync_bfly(bmax, offset=4))
                    _, scale_byte = pow2_ceil_ue8m0(
                        bmax * cutlass.Float32(1.0 / FLOAT8_E4M3_MAX)
                    )
                    inv_scale = ue8m0_to_output_scale(scale_byte)
                    k32_row_base = gate_smem + row_smem_off + (k32_local << Int32(7))
                    st_shared_v4_f32(
                        k32_row_base + ((sub_lane << Int32(2)) << Int32(2)),
                        o0 * inv_scale,
                        o1 * inv_scale,
                        o2 * inv_scale,
                        o3 * inv_scale,
                    )
                    cute.arch.sync_warp()

                    q0, q1 = _ld_shared_v2_f32(k32_row_base + (p_base << Int32(2)))
                    q2, q3 = _ld_shared_v2_f32(k32_row_base + ((p_base + Int32(8)) << Int32(2)))
                    packed = cvt_f32x4_to_e4m3x4(q0, q1, q2, q3)

                    phys_row = physical_row_base + Int64(abs_row)
                    dst_word = (
                        phys_row * Int64(words_per_row)
                        + Int64(output_tile * Int32(32) + lane)
                    )
                    intermediate_u32[dst_word] = packed

                    sb = scale_byte & Uint32(0xFF)
                    sb1 = Uint32(cute.arch.shuffle_sync(sb, Int32(8)))
                    sb2 = Uint32(cute.arch.shuffle_sync(sb, Int32(16)))
                    sb3 = Uint32(cute.arch.shuffle_sync(sb, Int32(24)))
                    if lane == Int32(0):
                        scale_word = (
                            sb
                            | (sb1 << Uint32(8))
                            | (sb2 << Uint32(16))
                            | (sb3 << Uint32(24))
                        )
                        intermediate_u32[
                            sf_base + Int64(output_tile) * Int64(rows_capacity) + phys_row
                        ] = scale_word

            cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        packed_a_u32: cute.Tensor,
        a_scales_u32: cute.Tensor,
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
        rows_capacity: cutlass.Int32,
        input_k128_tiles: cutlass.Int32,
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
                            packed_a_u32, a_scales_u32, w13_u32, intermediate_u32,
                            trellis_rotations, trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile,
                            expert_idx, output_tile, valid_rows, rows_capacity,
                            input_k128_tiles, intermediate_tiles, packed_rot, 16,
                        )
                    elif valid_rows <= Int32(32):
                        self._run_task(
                            packed_a_u32, a_scales_u32, w13_u32, intermediate_u32,
                            trellis_rotations, trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile,
                            expert_idx, output_tile, valid_rows, rows_capacity,
                            input_k128_tiles, intermediate_tiles, packed_rot, 32,
                        )
                    else:
                        self._run_task(
                            packed_a_u32, a_scales_u32, w13_u32, intermediate_u32,
                            trellis_rotations, trellis_rotations_u16, sign_patterns, sign_table,
                            smem_base, tid, warp_idx, m_tile,
                            expert_idx, output_tile, valid_rows, rows_capacity,
                            input_k128_tiles, intermediate_tiles, packed_rot, 64,
                        )
            task_slot += Int32(gdimx)


class Exl3MxFC2Kernel16W:
    """16-warp (8N x 2K) Spill-Free, Bank-Free-A, Zero-B-SMEM, N256 W2A8-MXFP8 FC2."""

    tile_m = 64
    tile_n = 256
    tile_k = 128
    num_warps = 16
    threads_per_cta = 512

    def __init__(
        self,
        *,
        top_k: int,
        trellis_bits: int = 2,
        stages: int = 3,
        direct_f32_atomic: bool = False,
    ):
        if trellis_bits != 2:
            raise ValueError(f"Exl3MxFC2Kernel16W requires bits=2, got {trellis_bits}")
        if stages not in (2, 3):
            raise ValueError(f"stages must be 2 or 3, got {stages}")
        self.top_k = int(top_k)
        self.trellis_bits = int(trellis_bits)
        self.stages = int(stages)
        self.direct_f32_atomic = bool(direct_f32_atomic)

        self.a_offset = 0
        self.a_payload_bytes = self.tile_m * self.tile_k  # 8,192 B
        self.sfa_offset = self.a_offset + self.a_payload_bytes
        self.a_scale_bytes = self.tile_m * 4  # 256 B
        self.stage_bytes = self.sfa_offset + self.a_scale_bytes  # 8,448 B
        self.pipeline_bytes = self.stages * self.stage_bytes
        self.epilogue_bytes = 0 if self.direct_f32_atomic else (32 * self.tile_n * 4)
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
        scatter_output: cute.Tensor,  # [M, H] bf16 (if not direct_f32_atomic) or f32 (if direct_f32_atomic)
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        trellis_lut: cute.Tensor,
        live_routes: cutlass.Int32,
        rows_capacity: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
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
            rows_capacity,
            intermediate_tiles,
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
        k128_slice: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        stage = k128_slice % Int32(self.stages)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)

        words_per_row = intermediate_tiles * Int32(32)
        physical_row_base = m_tile * Int32(self.tile_m)

        if tid < Int32(int(compute_m) * 8):
            row = tid >> Int32(3)
            vec = tid & Int32(7)
            # Zero bank conflicts for 64-bit ld_shared_v2_u32!
            phys_vec = vec ^ ((row & Int32(3)) << Int32(1))
            src_word = (
                Int64(physical_row_base + row) * Int64(words_per_row)
                + Int64(k128_slice * Int32(32) + (vec << Int32(2)))
            )
            cp_async4_shared_global(
                a_base + (row << Int32(7)) + (phys_vec << Int32(4)),
                get_ptr_as_int64(intermediate_u32, src_word),
            )
        if tid < Int32(compute_m):
            sf_src = (
                Int64(rows_capacity) * Int64(words_per_row)
                + Int64(k128_slice) * Int64(rows_capacity)
                + Int64(physical_row_base + tid)
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_src),
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
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        output_n256_tiles: Int32,
        compute_m: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        warp_n = warp_idx & Int32(7)
        warp_k = warp_idx >> Int32(3)

        tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(lane, self.trellis_bits)
        trellis_lut_addr = Int64(smem_base + Int32(self.trellis_lut_offset))

        tr_n16_cnt = output_n256_tiles * Int32(16)
        tr_k16_stride = Int64(tr_n16_cnt * Int32(16))
        tr_eu = Int64(intermediate_tiles * Int32(8)) * tr_k16_stride
        tr_warp_base = (
            Int64(expert_idx) * tr_eu
            + Int64(output_tile * Int32(16) + warp_n * Int32(2)) * Int64(16)
            + Int64(warp_k * Int32(4)) * tr_k16_stride
        )
        tr_k128_step = Int64(8) * tr_k16_stride
        tr_k32_step = Int64(2) * tr_k16_stride

        # Preload ALL stages up to self.stages at once!
        # When intermediate_tiles <= self.stages (e.g., Kimi I=384 -> 3 <= 3),
        # ONE wait_group(0) + ONE sync_threads() covers the ENTIRE K-loop with ZERO barriers!
        preload_all = intermediate_tiles <= Int32(self.stages)
        if preload_all:
            for s in cutlass.range_constexpr(self.stages):
                if Int32(s) < intermediate_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, Int32(s),
                        rows_capacity, intermediate_tiles, compute_m,
                    )
            cute.arch.cp_async_commit_group()
        else:
            for s in cutlass.range_constexpr(self.stages - 1):
                if Int32(s) < intermediate_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, Int32(s),
                        rows_capacity, intermediate_tiles, compute_m,
                    )
                cute.arch.cp_async_commit_group()

        # Preload slice 0 B weights (all 4 K32 halves) while A stages are in flight!
        w_slice_base = tr_warp_base
        raw_n0_k0 = _load_trellis_k32_raw_u32(w2_u32, w_slice_base, tr_k16_stride, lane)
        raw_n1_k0 = _load_trellis_k32_raw_u32(w2_u32, w_slice_base + Int64(16), tr_k16_stride, lane)
        raw_n0_k1 = _load_trellis_k32_raw_u32(w2_u32, w_slice_base + tr_k32_step, tr_k16_stride, lane)
        raw_n1_k1 = _load_trellis_k32_raw_u32(w2_u32, w_slice_base + tr_k32_step + Int64(16), tr_k16_stride, lane)

        if preload_all:
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        m_blocks = int(compute_m) // 16
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(m_blocks)
        )
        for blk in cutlass.range_constexpr(m_blocks):
            for nt in cutlass.range_constexpr(4):
                facc[blk][nt].fill(0.0)

        unit_sfb = Uint32(0x7F7F7F7F)
        warp_k_shift = Uint32(warp_k * Int32(16))
        sf_lane_off = q + ((lane & Int32(1)) << Int32(3))
        q_swiz = (q & Int32(3)) << Int32(1)
        c_sub = (c & Int32(1)) << Int32(3)

        k128_slice = Int32(0)
        while k128_slice < intermediate_tiles:
            stage = k128_slice % Int32(self.stages)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)

            if not preload_all:
                next_a_slice = k128_slice + Int32(self.stages - 1)
                if next_a_slice < intermediate_tiles:
                    self._stage_a_slice(
                        intermediate_u32, smem_base, tid, m_tile, next_a_slice,
                        rows_capacity, intermediate_tiles, compute_m,
                    )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(self.stages - 1)
                cute.arch.sync_threads()

            nxt_n0_k0, nxt_n1_k0, nxt_n0_k1, nxt_n1_k1 = raw_n0_k0, raw_n1_k0, raw_n0_k1, raw_n1_k1
            next_b_base = w_slice_base + tr_k128_step
            if k128_slice + Int32(1) < intermediate_tiles:
                nxt_n0_k0 = _load_trellis_k32_raw_u32(w2_u32, next_b_base, tr_k16_stride, lane)
                nxt_n1_k0 = _load_trellis_k32_raw_u32(w2_u32, next_b_base + Int64(16), tr_k16_stride, lane)
                nxt_n0_k1 = _load_trellis_k32_raw_u32(w2_u32, next_b_base + tr_k32_step, tr_k16_stride, lane)
                nxt_n1_k1 = _load_trellis_k32_raw_u32(w2_u32, next_b_base + tr_k32_step + Int64(16), tr_k16_stride, lane)

            for kb_local in cutlass.range_constexpr(2):
                kb = warp_k * Int32(2) + Int32(kb_local)
                u_phys = (kb * Int32(2) + (c >> Int32(1))) ^ q_swiz
                a_row_base = a_base + (q << Int32(7)) + (u_phys << Int32(4)) + c_sub
                rn0 = raw_n0_k0 if cutlass.const_expr(kb_local == 0) else raw_n0_k1
                rn1 = raw_n1_k0 if cutlass.const_expr(kb_local == 0) else raw_n1_k1

                d0, d1, d2, d3 = _decode_reg_k32_pair_both(
                    rn0, lane, tr_ia, tr_ib, tr_s2, trellis_lut_addr
                )
                e0, e1, e2, e3 = _decode_reg_k32_pair_both(
                    rn1, lane, tr_ia, tr_ib, tr_s2, trellis_lut_addr
                )
                for blk in cutlass.range_constexpr(m_blocks):
                    asc_v = ld_shared_u32(sfa_base + ((Int32(blk * 16) + sf_lane_off) << Int32(2))) >> warp_k_shift
                    a_lo = a_row_base + Int32(blk * 16 * 128)
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * 128))

                    f0 = facc[blk][0]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        f0[0], f0[1], f0[2], f0[3],
                        a0, a1, a2, a3, d0, d1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    f0[0], f0[1], f0[2], f0[3] = r0, r1, r2, r3

                    f1 = facc[blk][1]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        f1[0], f1[1], f1[2], f1[3],
                        a0, a1, a2, a3, d2, d3, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    f1[0], f1[1], f1[2], f1[3] = r0, r1, r2, r3

                    f2 = facc[blk][2]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        f2[0], f2[1], f2[2], f2[3],
                        a0, a1, a2, a3, e0, e1, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    f2[0], f2[1], f2[2], f2[3] = r0, r1, r2, r3

                    f3 = facc[blk][3]
                    r0, r1, r2, r3 = mxfp8_mma_m16n8k32_f32_e4m3(
                        f3[0], f3[1], f3[2], f3[3],
                        a0, a1, a2, a3, e2, e3, asc_v, unit_sfb, bid_a=kb_local, bid_b=0,
                    )
                    f3[0], f3[1], f3[2], f3[3] = r0, r1, r2, r3

            raw_n0_k0, raw_n1_k0, raw_n0_k1, raw_n1_k1 = nxt_n0_k0, nxt_n1_k0, nxt_n0_k1, nxt_n1_k1
            if not preload_all and k128_slice + Int32(self.stages) < intermediate_tiles:
                cute.arch.sync_threads()
            k128_slice += Int32(1)
            w_slice_base = next_b_base

        scatter_n = Int32(scatter_output.shape[1])
        tok_smem = smem_base + Int32(self.tok_smem_offset)
        wt_smem = smem_base + Int32(self.wt_smem_offset)
        col_base_local = warp_n * Int32(32) + (c << Int32(1))
        col_base_global = output_tile * Int32(self.tile_n) + col_base_local

        if cutlass.const_expr(self.direct_f32_atomic):
            # ZERO epilogue SMEM and ZERO epilogue barriers! Both warp_k in {0, 1}
            # reduce directly in FP32 global memory via red.relaxed.gpu.global.add.v2.f32!
            if not preload_all:
                cute.arch.cp_async_wait_group(0)
            for blk in cutlass.range_constexpr(m_blocks):
                abs_lo = Int32(blk * 16) + q
                abs_hi = abs_lo + Int32(8)
                if abs_lo < valid_rows:
                    tok_lo = ld_shared_i32_relaxed(tok_smem + (abs_lo << Int32(2)))
                    wt_lo = ld_shared_f32(wt_smem + (abs_lo << Int32(2)))
                    for nt in cutlass.range_constexpr(4):
                        col_g = col_base_global + Int32(nt * 8)
                        f = facc[blk][nt]
                        _red_add_global_v2_f32(
                            get_ptr_as_int64(
                                scatter_output,
                                Int64(tok_lo) * Int64(scatter_n) + Int64(col_g),
                            ),
                            f[0] * wt_lo,
                            f[1] * wt_lo,
                        )
                if abs_hi < valid_rows:
                    tok_hi = ld_shared_i32_relaxed(tok_smem + (abs_hi << Int32(2)))
                    wt_hi = ld_shared_f32(wt_smem + (abs_hi << Int32(2)))
                    for nt in cutlass.range_constexpr(4):
                        col_g = col_base_global + Int32(nt * 8)
                        f = facc[blk][nt]
                        _red_add_global_v2_f32(
                            get_ptr_as_int64(
                                scatter_output,
                                Int64(tok_hi) * Int64(scatter_n) + Int64(col_g),
                            ),
                            f[2] * wt_hi,
                            f[3] * wt_hi,
                        )
            cute.arch.sync_threads()
        else:
            if not preload_all:
                cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
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
                                s2, s3 = _ld_shared_v2_f32(smem_base + off_hi)
                                scatter_add_bf16x2(
                                    get_ptr_as_int64(
                                        scatter_output,
                                        Int64(tok_hi) * Int64(scatter_n) + Int64(col_g),
                                    ),
                                    (s2 + f[2]) * wt_hi,
                                    (s3 + f[3]) * wt_hi,
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
        rows_capacity: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
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
                    wt = cutlass.Float32(0.0)
                    if r >= Int32(0) and r < live_routes:
                        is_v = Int32(1)
                        tok = r // Int32(self.top_k)
                        wt = topk_weights_flat[r].to(cutlass.Float32)
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
                            rows_capacity, intermediate_tiles, output_n256_tiles, 16,
                        )
                    elif valid_rows <= Int32(32):
                        self._run_task(
                            intermediate_u32, w2_u32, scatter_output, smem_base, tid,
                            warp_idx, m_tile, expert_idx, output_tile, valid_rows,
                            rows_capacity, intermediate_tiles, output_n256_tiles, 32,
                        )
                    else:
                        self._run_task(
                            intermediate_u32, w2_u32, scatter_output, smem_base, tid,
                            warp_idx, m_tile, expert_idx, output_tile, valid_rows,
                            rows_capacity, intermediate_tiles, output_n256_tiles, 64,
                        )
            task_slot += Int32(gdimx)


__all__ = ["Exl3MxFC1Kernel16W", "Exl3MxFC2Kernel16W"]
