# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM120/SM121 CuTe-DSL four-or-six NVFP4 / MXFP8 activation quantizers and route helpers."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T

from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_u32
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_global_nc_v4_u32,
    shared_ptr_to_u32,
    st_global_u32,
    st_shared_u8,
    st_shared_v4_u32,
)
from b12x._lib.intrinsics.packed import pack_f32x2_to_bfloat2

from ._math import (
    FP32_MAX,
    div_rn_f32,
    e2m1x2_pack_and_roundtrip,
    e4m3_byte_to_f32,
    f32_to_e4m3_byte,
    rcp_rn_f32,
    shfl_xor_f32,
    ue8m0_byte_to_f32,
)


@dsl_user_op
def _bf16x2_to_f32x2(word: Uint32, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(word).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.bf16 $0, lo;
            cvt.f32.bf16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
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
def _f32_to_e4m3x2_u16(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 out16;
                cvt.rn.satfinite.e4m3x2.f32 out16, $2, $1;
                cvt.u32.u16 $0, out16;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_as_u32(x: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _u32_as_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def partition_csf_slab_offsets(
    exceptions: torch.Tensor,
    starts: torch.Tensor,
    num_experts: int,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Partition expert-major CSF exceptions into per-16-row-slab contiguous ranges.

    Executed once at weight-load time on GPU. Returns ``(slab_exceptions, slab_offsets)``
    where ``slab_offsets`` has shape ``[E * (rows // 16) + 1]`` (int32) and
    ``slab_exceptions`` has the exact same ``uint32`` exception records sorted by
    ``(expert, slab)``.
    """
    if rows % 16:
        raise ValueError(f"CSF rows must be a multiple of 16, got {rows}")
    slabs = rows // 16
    total_slabs = num_experts * slabs
    device = exceptions.device
    if exceptions.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.uint32, device=device),
            torch.zeros((total_slabs + 1,), dtype=torch.int32, device=device),
        )
    exc_u32 = exceptions.contiguous()
    starts_i64 = starts.to(torch.int64)
    counts_per_expert = starts_i64[1:] - starts_i64[:-1]
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int64, device=device), counts_per_expert
    )
    flats = exc_u32.to(torch.int64) & 0xFFFFFF
    slab_in_expert = flats // (16 * cols)
    global_slab = expert_ids * slabs + slab_in_expert
    order = torch.argsort(global_slab, stable=True)
    sorted_exc = exc_u32.index_select(0, order).contiguous()
    sorted_slab = global_slab.index_select(0, order)
    slab_counts = torch.bincount(sorted_slab, minlength=total_slabs).to(torch.int32)
    slab_offsets = torch.zeros((total_slabs + 1,), dtype=torch.int32, device=device)
    slab_offsets[1:] = torch.cumsum(slab_counts, dim=0, dtype=torch.int32)
    return sorted_exc, slab_offsets


class SituInputQuantNVFP4Kernel:
    """Exact Inkling-2-Mini four-or-six MAE block-16 NVFP4 row quantizer.

    Grid ``(M, 1, 1)``, block ``(128, 1, 1)`` (4 warps).
    Matches ``situ_experts._quantize(..., FP4=True)`` bit-for-bit on SM120/SM121.
    """

    def __init__(self, hidden: int):
        if hidden % 16 != 0:
            raise ValueError(f"hidden must be a multiple of 16, got {hidden}")
        self.hidden = int(hidden)
        self.num_blocks = self.hidden // 16
        self.threads = 128

    def _get_shared_storage_cls(self):
        row_bytes = self.hidden * 2

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "warp_max": cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 8], 16
            ],
            "row_bf16": cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint8, row_bytes], 16
            ],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # [M, K] bf16
        mQ: cute.Tensor,  # [M, K//2] u8
        mS: cute.Tensor,  # [M, K//16] u8 (E4M3FN bits)
        mOuter: cute.Tensor,  # [M] f32 (stores amax, matching situ_experts)
        stream: cuda.CUstream,
    ):
        m_rows = mX.shape[0]
        self.kernel(mX, mQ, mS, mOuter).launch(
            grid=(m_rows, 1, 1),
            block=[self.threads, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        mOuter: cute.Tensor,
    ):
        row = Int32(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        warp = tid // Int32(32)
        lane = tid % Int32(32)

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        warp_max_s = storage.warp_max.get_tensor(cute.make_layout((8,)))
        row_s_addr = shared_ptr_to_u32(storage.row_bf16.data_ptr())

        row_x_ptr = get_ptr_as_int64(mX, Int64(row) * Int64(self.hidden))

        # Pass 1: 16B vectorized global load -> SMEM stage + exact row amax (reads HBM ONCE).
        local_max = Float32(0.0)
        vecs_per_row: cutlass.Constexpr = self.hidden // 8
        iters_max: cutlass.Constexpr = (
            vecs_per_row + self.threads - 1
        ) // self.threads
        for it in cutlass.range_constexpr(iters_max):
            v = tid + Int32(it * self.threads)
            if v < Int32(vecs_per_row):
                w0, w1, w2, w3 = ld_global_nc_v4_u32(row_x_ptr + Int64(v) * Int64(16))
                st_shared_v4_u32(row_s_addr + v * Int32(16), w0, w1, w2, w3)
                for w in (w0, w1, w2, w3):
                    a0, a1 = _bf16x2_to_f32x2(w)
                    local_max = fmax_f32(local_max, fmax_f32(fabs_f32(a0), fabs_f32(a1)))

        local_max = fmax_f32(local_max, shfl_xor_f32(local_max, 16))
        local_max = fmax_f32(local_max, shfl_xor_f32(local_max, 8))
        local_max = fmax_f32(local_max, shfl_xor_f32(local_max, 4))
        local_max = fmax_f32(local_max, shfl_xor_f32(local_max, 2))
        local_max = fmax_f32(local_max, shfl_xor_f32(local_max, 1))
        if lane == Int32(0):
            warp_max_s[warp] = local_max
        cute.arch.barrier()

        if tid == Int32(0):
            m0 = fmax_f32(warp_max_s[0], warp_max_s[1])
            m1 = fmax_f32(warp_max_s[2], warp_max_s[3])
            row_amax = fmax_f32(m0, m1)
            warp_max_s[4] = row_amax
            mOuter[row] = row_amax
        cute.arch.barrier()

        amax = warp_max_s[4]
        encode = div_rn_f32(Float32(1536.0), amax)
        if amax == Float32(0.0) or encode > Float32(FP32_MAX):
            encode = Float32(FP32_MAX)
        decoder = div_rn_f32(Float32(1.0), encode)
        outer = amax * Float32(1.0 / 1536.0)

        # Pass 2: one thread per 16-element block (reads from SMEM, zero second HBM pass!).
        row_q_ptr = get_ptr_as_int64(mQ, Int64(row) * Int64(self.hidden // 2))
        blk_iters: cutlass.Constexpr = (
            self.num_blocks + self.threads - 1
        ) // self.threads
        for it in cutlass.range_constexpr(blk_iters):
            blk = tid + Int32(it * self.threads)
            if blk < Int32(self.num_blocks):
                vals = cute.make_rmem_tensor(cute.make_layout((16,)), Float32)
                bmax = Float32(0.0)
                blk_s_addr = row_s_addr + blk * Int32(32)
                for p in cutlass.range_constexpr(8):
                    bf2 = ld_shared_u32(blk_s_addr + Int32(p * 4))
                    v0, v1 = _bf16x2_to_f32x2(bf2)
                    vals[2 * p] = v0
                    vals[2 * p + 1] = v1
                    bmax = fmax_f32(bmax, fmax_f32(fabs_f32(v0), fabs_f32(v1)))

                raw6 = bmax * (encode * Float32(1.0 / 6.0))
                if raw6 > Float32(448.0):
                    raw6 = Float32(448.0)
                raw4 = bmax * (encode * Float32(0.25))
                if raw4 > Float32(448.0):
                    raw4 = Float32(448.0)
                s6_u8 = f32_to_e4m3_byte(raw6)
                s4_u8 = f32_to_e4m3_byte(raw4)
                s6_f = e4m3_byte_to_f32(s6_u8)
                s4_f = e4m3_byte_to_f32(s4_u8)

                inv6 = div_rn_f32(Float32(1.0), s6_f * decoder)
                if s6_f * decoder == Float32(0.0) or inv6 > Float32(FP32_MAX):
                    inv6 = Float32(FP32_MAX)
                inv4 = div_rn_f32(Float32(1.0), s4_f * decoder)
                if s4_f * decoder == Float32(0.0) or inv4 > Float32(FP32_MAX):
                    inv4 = Float32(FP32_MAX)

                p6_lo = Uint32(0)
                p6_hi = Uint32(0)
                p4_lo = Uint32(0)
                p4_hi = Uint32(0)
                err6 = Float32(0.0)
                err4 = Float32(0.0)
                s6_out = s6_f * outer
                s4_out = s4_f * outer
                for p in cutlass.range_constexpr(8):
                    x0 = vals[2 * p]
                    x1 = vals[2 * p + 1]
                    b6, r6_0, r6_1 = e2m1x2_pack_and_roundtrip(x0 * inv6, x1 * inv6)
                    b4, r4_0, r4_1 = e2m1x2_pack_and_roundtrip(x0 * inv4, x1 * inv4)
                    err6 = (
                        err6
                        + fabs_f32(r6_0 * s6_out - x0)
                        + fabs_f32(r6_1 * s6_out - x1)
                    )
                    err4 = (
                        err4
                        + fabs_f32(r4_0 * s4_out - x0)
                        + fabs_f32(r4_1 * s4_out - x1)
                    )
                    if const_expr(p < 4):
                        p6_lo = p6_lo | (b6 << Uint32(p * 8))
                        p4_lo = p4_lo | (b4 << Uint32(p * 8))
                    else:
                        p6_hi = p6_hi | (b6 << Uint32((p - 4) * 8))
                        p4_hi = p4_hi | (b4 << Uint32((p - 4) * 8))

                use4 = err4 < err6
                out_lo = Uint32(cutlass.select_(use4, p4_lo, p6_lo))
                out_hi = Uint32(cutlass.select_(use4, p4_hi, p6_hi))
                out_s = Uint32(cutlass.select_(use4, s4_u8, s6_u8))
                st_global_u32(row_q_ptr + Int64(blk) * Int64(8), out_lo)
                st_global_u32(row_q_ptr + Int64(blk) * Int64(8) + Int64(4), out_hi)
                mS[row, blk] = out_s.to(Uint8)


class SituInputQuantMXFP8Kernel:
    """Exact Inkling-2-Mini MXFP8 block-32 UE8M0 row quantizer."""

    def __init__(self, hidden: int):
        if hidden % 32 != 0:
            raise ValueError(f"MXFP8 hidden must be a multiple of 32, got {hidden}")
        self.hidden = int(hidden)
        self.num_blocks = self.hidden // 32
        self.threads = 128

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # [M, K] bf16
        mQ: cute.Tensor,  # [M, K] u8 (E4M3FN bits)
        mS: cute.Tensor,  # [M, K//32] u8 (UE8M0)
        stream: cuda.CUstream,
    ):
        self.kernel(mX, mQ, mS).launch(
            grid=(mX.shape[0], 1, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor):
        row = Int32(cute.arch.block_idx()[0])
        tid = Int32(cute.arch.thread_idx()[0])
        row_x_ptr = get_ptr_as_int64(mX, Int64(row) * Int64(self.hidden))
        row_q_ptr = get_ptr_as_int64(mQ, Int64(row) * Int64(self.hidden))

        blk_iters: cutlass.Constexpr = (
            self.num_blocks + self.threads - 1
        ) // self.threads
        for it in cutlass.range_constexpr(blk_iters):
            blk = tid + Int32(it * self.threads)
            if blk < Int32(self.num_blocks):
                vals = cute.make_rmem_tensor(cute.make_layout((32,)), Float32)
                bmax = Float32(0.0)
                bx_ptr = row_x_ptr + Int64(blk) * Int64(64)
                for p in cutlass.range_constexpr(16):
                    bf2 = ld_global_nc_u32(bx_ptr + Int64(p * 4))
                    v0, v1 = _bf16x2_to_f32x2(bf2)
                    vals[2 * p] = v0
                    vals[2 * p + 1] = v1
                    bmax = fmax_f32(bmax, fmax_f32(fabs_f32(v0), fabs_f32(v1)))

                raw = bmax * Float32(1.0 / 448.0)
                min_f = Float32(2.0**-127)
                raw_bits = _f32_as_u32(raw)
                exp = ((raw_bits + Uint32(0x007FFFFF)) & Uint32(0x7F800000)) >> Uint32(
                    23
                )
                if raw <= min_f:
                    exp = Uint32(0)
                elif exp > Uint32(254):
                    exp = Uint32(254)
                scale = ue8m0_byte_to_f32(exp)
                inv = rcp_rn_f32(scale)
                if exp == Uint32(0):
                    inv = Float32(2.0**127)

                bq_ptr = row_q_ptr + Int64(blk) * Int64(32)
                for q4 in cutlass.range_constexpr(8):
                    e01 = _f32_to_e4m3x2_u16(vals[4 * q4] * inv, vals[4 * q4 + 1] * inv)
                    e23 = _f32_to_e4m3x2_u16(
                        vals[4 * q4 + 2] * inv, vals[4 * q4 + 3] * inv
                    )
                    st_global_u32(bq_ptr + Int64(q4 * 4), e01 | (e23 << Uint32(16)))
                mS[row, blk] = exp.to(Uint8)


class SituSmallRoutePackKernel:
    """Single-CTA deterministic route grouper for decode (M <= 16, R <= 128).

    Replaces three Triton launches (``_clear``, ``_pack``, ``_tasks``) with one
    single-CTA launch producing:
      - ``active_experts[slot]``: expert id (or -1 if unused slot), ``slot < max_slots``
      - ``active_counts[slot]``: number of routes assigned to that expert
      - ``active_routes[slot, k]``: route index ``0..R-1`` in deterministic ascending order
    """

    def __init__(self, num_experts: int, num_routes: int):
        self.num_experts = int(num_experts)
        self.num_routes = int(num_routes)
        self.max_slots = min(self.num_experts, self.num_routes)

    def _get_shared_storage_cls(self):
        E = self.num_experts
        R = self.num_routes

        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "ids": cute.struct.MemRange[cutlass.Int32, R],
            "counts": cute.struct.MemRange[cutlass.Int32, E],
            "expert_slot": cute.struct.MemRange[cutlass.Int32, E],
            "cursor": cute.struct.MemRange[cutlass.Int32, E],
            "num_active": cute.struct.MemRange[cutlass.Int32, 1],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        mIds: cute.Tensor,  # [R] i32
        mActiveExperts: cute.Tensor,  # [max_slots] i32
        mActiveCounts: cute.Tensor,  # [max_slots] i32
        mActiveRoutes: cute.Tensor,  # [max_slots, R] i32
        stream: cuda.CUstream,
    ):
        self.kernel(mIds, mActiveExperts, mActiveCounts, mActiveRoutes).launch(
            grid=(1, 1, 1),
            block=[128, 1, 1],
            smem=self._get_shared_storage_cls().size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mIds: cute.Tensor,
        mActiveExperts: cute.Tensor,
        mActiveCounts: cute.Tensor,
        mActiveRoutes: cute.Tensor,
    ):
        tid = Int32(cute.arch.thread_idx()[0])
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(self._get_shared_storage_cls())
        s_ids = storage.ids.get_tensor(cute.make_layout((self.num_routes,)))
        s_counts = storage.counts.get_tensor(cute.make_layout((self.num_experts,)))
        s_slot = storage.expert_slot.get_tensor(cute.make_layout((self.num_experts,)))
        s_cursor = storage.cursor.get_tensor(cute.make_layout((self.num_experts,)))
        s_num_active = storage.num_active.get_tensor(cute.make_layout((1,)))

        # Parallel load & clear
        e_iters: cutlass.Constexpr = (self.num_experts + 127) // 128
        for it in cutlass.range_constexpr(e_iters):
            e = tid + Int32(it * 128)
            if e < Int32(self.num_experts):
                s_counts[e] = Int32(0)
                s_slot[e] = Int32(-1)
                s_cursor[e] = Int32(0)
        r_iters: cutlass.Constexpr = (self.num_routes + 127) // 128
        for it in cutlass.range_constexpr(r_iters):
            r = tid + Int32(it * 128)
            if r < Int32(self.num_routes):
                s_ids[r] = mIds[r]
        s_iters: cutlass.Constexpr = (self.max_slots + 127) // 128
        for it in cutlass.range_constexpr(s_iters):
            sl = tid + Int32(it * 128)
            if sl < Int32(self.max_slots):
                mActiveExperts[sl] = Int32(-1)
                mActiveCounts[sl] = Int32(0)
        cute.arch.barrier()

        # Thread 0 packs <= 128 routes deterministically in ~150 cycles (zero atomics).
        if tid == Int32(0):
            n_act = Int32(0)
            for r in range(self.num_routes):
                e = s_ids[r]
                if e >= Int32(0) and e < Int32(self.num_experts):
                    sl = s_slot[e]
                    if sl < Int32(0):
                        sl = n_act
                        s_slot[e] = sl
                        mActiveExperts[sl] = e
                        n_act = n_act + Int32(1)
                    pos = s_cursor[e]
                    mActiveRoutes[sl, pos] = Int32(r)
                    s_cursor[e] = pos + Int32(1)
                    s_counts[e] = pos + Int32(1)
            s_num_active[0] = n_act
        cute.arch.barrier()

        n_act = s_num_active[0]
        for it in cutlass.range_constexpr(s_iters):
            sl = tid + Int32(it * 128)
            if sl < n_act:
                e = mActiveExperts[sl]
                mActiveCounts[sl] = s_counts[e]


class SituRouteSumKernel:
    """Deterministic FP32 reduction across top-K route outputs -> BF16 ``[M, H]``."""

    def __init__(self, hidden: int, top_k: int):
        if hidden % 2 != 0:
            raise ValueError("hidden must be even")
        self.hidden = int(hidden)
        self.top_k = int(top_k)
        self.threads = 128
        self.pairs_per_cta = 128  # 256 BF16 elements per CTA

    @cute.jit
    def __call__(
        self,
        mRouteOut: cute.Tensor,  # [M * top_k, H] bf16
        mIds: cute.Tensor,  # [M * top_k] i32
        mOut: cute.Tensor,  # [M, H] bf16
        stream: cuda.CUstream,
    ):
        m_rows = mOut.shape[0]
        h_tiles = (self.hidden // 2 + self.pairs_per_cta - 1) // self.pairs_per_cta
        self.kernel(mRouteOut, mIds, mOut).launch(
            grid=(h_tiles, m_rows, 1),
            block=[self.threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mRouteOut: cute.Tensor, mIds: cute.Tensor, mOut: cute.Tensor):
        tile = Int32(cute.arch.block_idx()[0])
        tok = Int32(cute.arch.block_idx()[1])
        tid = Int32(cute.arch.thread_idx()[0])
        pair_col = tile * Int32(self.pairs_per_cta) + tid
        if pair_col < Int32(self.hidden // 2):
            acc0 = Float32(0.0)
            acc1 = Float32(0.0)
            base_route = tok * Int32(self.top_k)
            for k in cutlass.range_constexpr(self.top_k):
                r = base_route + Int32(k)
                if mIds[r] >= Int32(0):
                    ptr = get_ptr_as_int64(
                        mRouteOut,
                        Int64(r) * Int64(self.hidden) + Int64(pair_col) * Int64(2),
                    )
                    bf2 = ld_global_nc_u32(ptr)
                    v0, v1 = _bf16x2_to_f32x2(bf2)
                    acc0 = acc0 + v0
                    acc1 = acc1 + v1
            out_ptr = get_ptr_as_int64(
                mOut, Int64(tok) * Int64(self.hidden) + Int64(pair_col) * Int64(2)
            )
            st_global_u32(out_ptr, pack_f32x2_to_bfloat2(acc0, acc1))
