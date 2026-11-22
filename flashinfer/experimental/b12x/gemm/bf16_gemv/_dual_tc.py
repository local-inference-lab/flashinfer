# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused dual-output BF16 tensor-core GEMV (e.g. latent_down_rows BF16 + router FP32).

Projects one shared BF16 input ``x[M, K]`` against two disjoint BF16 weight
matrices ``w_a[N_a, K]`` -> ``out_a[M, N_a]`` (BF16) and ``w_b[N_b, K]`` ->
``out_b[M, N_b]`` (FP32) in a single split-K launch with coalesced 128-bit
L2-coherent reduction.  Zero weight duplication and zero auxiliary-stream
fork/join overhead.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_nc_v4_u32
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

from ._tensor_core import (
    _LANES,
    _atomic_add_gpu_u32,
    _flat,
    _ld_global_cg_v4_f32,
    _st_global_release_u32,
    _st_global_wb_v4_f32,
)

def dual_tc_split_k(n_a: int, n_b: int, k: int, max_rows: int) -> int:
    """Pick wave-balanced split-K for dual BF16 TC-GEMV."""
    chunks = int(k) // 64
    preferred = (7, 8, 4, 2, 1) if max_rows <= 16 else (8, 7, 4, 2, 1)
    for sk in preferred:
        if chunks % sk == 0:
            return sk
    return 1


def dual_tc_scratch_numel(n_a: int, n_b: int, max_rows: int, split_k: int) -> int:
    total_blocks = (int(n_a) + 15) // 16 + (int(n_b) + 15) // 16
    m_tiles = (int(max_rows) + 7) // 8
    return total_blocks * int(split_k) * m_tiles * 128


def dual_tc_flags_numel(n_a: int, n_b: int) -> int:
    return (int(n_a) + 15) // 16 + (int(n_b) + 15) // 16


class DualBf16TCGemvKernel:
    """Fused ``out_a[m, n_a] (BF16)`` and ``out_b[m, n_b] (FP32)`` from shared ``x[m, k]``."""

    def __init__(self, n_a: int, n_b: int, k: int, max_rows: int, split_k: int):
        if k <= 0 or k % 64 != 0:
            raise ValueError("dual BF16 TC-GEMV requires K to be a positive multiple of 64")
        if not 1 <= max_rows <= 32:
            raise ValueError("dual BF16 TC-GEMV supports 1..32 rows")
        chunks = int(k) // 64
        if split_k <= 0 or chunks % int(split_k) != 0:
            raise ValueError(f"split_k={split_k} must divide K/64={chunks}")
        self.n_a, self.n_b, self.k = int(n_a), int(n_b), int(k)
        self.blocks_a = (self.n_a + 15) // 16
        self.blocks_b = (self.n_b + 15) // 16
        self.total_blocks = self.blocks_a + self.blocks_b
        self.max_m_tiles = (int(max_rows) + 7) // 8
        self.split_k = int(split_k)
        self.chunks_per_split = chunks // self.split_k
        w = min(8, self.chunks_per_split)
        while self.chunks_per_split % w:
            w -= 1
        self.warps = w
        self.chunks_per_warp = self.chunks_per_split // self.warps

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.warps * _LANES * 4 * self.max_m_tiles
                ],
                16,
            ],
            "last": cute.struct.Align[cute.struct.MemRange[Uint32, 1], 4],
        }
        return cute.struct(SharedStorage)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        w_a: cute.Pointer,
        w_b: cute.Pointer,
        out_a: cute.Pointer,
        out_b: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        wa_stride: Int64,
        wb_stride: Int64,
        oa_stride: Int64,
        ob_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(w_a),
            _flat(w_b),
            _flat(out_a),
            _flat(out_b),
            scratch,
            flags,
            rows,
            x_stride,
            wa_stride,
            wb_stride,
            oa_stride,
            ob_stride,
        ).launch(
            grid=(self.total_blocks, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        w_a: cute.Tensor,
        w_b: cute.Tensor,
        out_a: cute.Tensor,
        out_b: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride: Int64,
        wa_stride: Int64,
        wb_stride: Int64,
        oa_stride: Int64,
        ob_stride: Int64,
    ):
        thread, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(thread) // _LANES
        lane = Int32(thread) % _LANES
        group = lane // 4
        quad = lane % 4
        is_a = Int32(n_tile) < Int32(self.blocks_a)
        local_tile = Int32(n_tile) if is_a else (Int32(n_tile) - Int32(self.blocks_a))
        cur_n = Int32(self.n_a) if is_a else Int32(self.n_b)
        first = local_tile * 16
        row_a = cutlass.min(first + group, cur_n - 1)
        row_b = cutlass.min(first + group + 8, cur_n - 1)
        w_base = get_ptr_as_int64(w_a, Int64(0)) if is_a else get_ptr_as_int64(w_b, Int64(0))
        w_stride = wa_stride if is_a else wb_stride
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        split_chunk_base = Int32(k_split) * self.chunks_per_split
        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = split_chunk_base + warp * self.chunks_per_warp + step
            offset = Int64(chunk * 128 + quad * 32)
            pa = w_base + Int64(row_a) * w_stride * 2 + offset
            pb = w_base + Int64(row_b) * w_stride * 2 + offset
            wa0, wa1, wa2, wa3 = ld_global_nc_v4_u32(pa)
            wa4, wa5, wa6, wa7 = ld_global_nc_v4_u32(pa + Int64(16))
            wb0, wb1, wb2, wb3 = ld_global_nc_v4_u32(pb)
            wb4, wb5, wb6, wb7 = ld_global_nc_v4_u32(pb + Int64(16))
            wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
            wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                px = x_base + Int64(token) * x_stride * 2 + offset
                x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                x4, x5, x6, x7 = ld_global_nc_v4_u32(px + Int64(16))
                xs = (x0, x1, x2, x3, x4, x5, x6, x7)
                a0, a1, a2, a3 = acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3]
                for s in cutlass.range_constexpr(4):
                    a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                        a0,
                        a1,
                        a2,
                        a3,
                        wa[2 * s],
                        wb[2 * s],
                        wa[2 * s + 1],
                        wb[2 * s + 1],
                        xs[2 * s],
                        xs[2 * s + 1],
                    )
                acc[mt, 0], acc[mt, 1], acc[mt, 2], acc[mt, 3] = a0, a1, a2, a3

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(
            cute.make_layout((self.warps * _LANES * 4 * self.max_m_tiles,))
        )
        if cutlass.const_expr(self.warps > 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                base = ((mt * self.warps + warp) * _LANES + lane) * 4
                for e in cutlass.range_constexpr(4):
                    partials[base + e] = acc[mt, e]
            cute.arch.sync_threads()

        if cutlass.const_expr(self.split_k == 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    t0 = acc[mt, 0] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t1 = acc[mt, 1] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t2 = acc[mt, 2] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t3 = acc[mt, 3] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    if cutlass.const_expr(self.warps > 1):
                        for w in cutlass.range_constexpr(self.warps):
                            pbase = ((mt * self.warps + w) * _LANES + lane) * 4
                            t0 = t0 + partials[pbase + 0]
                            t1 = t1 + partials[pbase + 1]
                            t2 = t2 + partials[pbase + 2]
                            t3 = t3 + partials[pbase + 3]
                    f_lo = first + group
                    f_hi = first + group + 8
                    r0 = Int32(mt * 8) + quad * 2
                    r1 = r0 + 1
                    if is_a:
                        if f_lo < cur_n:
                            if r0 < rows:
                                out_a[Int64(r0) * oa_stride + Int64(f_lo)] = t0.to(BFloat16)
                            if r1 < rows:
                                out_a[Int64(r1) * oa_stride + Int64(f_lo)] = t1.to(BFloat16)
                        if f_hi < cur_n:
                            if r0 < rows:
                                out_a[Int64(r0) * oa_stride + Int64(f_hi)] = t2.to(BFloat16)
                            if r1 < rows:
                                out_a[Int64(r1) * oa_stride + Int64(f_hi)] = t3.to(BFloat16)
                    else:
                        if f_lo < cur_n:
                            if r0 < rows:
                                out_b[Int64(r0) * ob_stride + Int64(f_lo)] = t0
                            if r1 < rows:
                                out_b[Int64(r1) * ob_stride + Int64(f_lo)] = t1
                        if f_hi < cur_n:
                            if r0 < rows:
                                out_b[Int64(r0) * ob_stride + Int64(f_hi)] = t2
                            if r1 < rows:
                                out_b[Int64(r1) * ob_stride + Int64(f_hi)] = t3
        else:
            tile_elems = self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    t0 = acc[mt, 0] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t1 = acc[mt, 1] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t2 = acc[mt, 2] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    t3 = acc[mt, 3] if cutlass.const_expr(self.warps == 1) else Float32(0.0)
                    if cutlass.const_expr(self.warps > 1):
                        for w in cutlass.range_constexpr(self.warps):
                            pbase = ((mt * self.warps + w) * _LANES + lane) * 4
                            t0 = t0 + partials[pbase + 0]
                            t1 = t1 + partials[pbase + 1]
                            t2 = t2 + partials[pbase + 2]
                            t3 = t3 + partials[pbase + 3]
                    vaddr = scratch_base + (sbase + Int64((mt * _LANES + lane) * 4)) * Int64(4)
                    _st_global_wb_v4_f32(vaddr, t0, t1, t2, t3)
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(n_tile) * Int64(4)
            if Int32(thread) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(n_tile) * self.split_k * tile_elems
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    if warp == Int32(mt % self.warps):
                        s0 = Float32(0.0)
                        s1 = Float32(0.0)
                        s2 = Float32(0.0)
                        s3 = Float32(0.0)
                        lane_off = Int64((mt * _LANES + lane) * 4)
                        for sp in cutlass.range_constexpr(self.split_k):
                            raddr = (
                                scratch_base
                                + (tbase + Int64(sp) * tile_elems + lane_off) * Int64(4)
                            )
                            v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                            s0 = s0 + v0
                            s1 = s1 + v1
                            s2 = s2 + v2
                            s3 = s3 + v3
                        f_lo = first + group
                        f_hi = first + group + 8
                        r0 = Int32(mt * 8) + quad * 2
                        r1 = r0 + 1
                        if is_a:
                            if f_lo < cur_n:
                                if r0 < rows:
                                    out_a[Int64(r0) * oa_stride + Int64(f_lo)] = s0.to(BFloat16)
                                if r1 < rows:
                                    out_a[Int64(r1) * oa_stride + Int64(f_lo)] = s1.to(BFloat16)
                            if f_hi < cur_n:
                                if r0 < rows:
                                    out_a[Int64(r0) * oa_stride + Int64(f_hi)] = s2.to(BFloat16)
                                if r1 < rows:
                                    out_a[Int64(r1) * oa_stride + Int64(f_hi)] = s3.to(BFloat16)
                        else:
                            if f_lo < cur_n:
                                if r0 < rows:
                                    out_b[Int64(r0) * ob_stride + Int64(f_lo)] = s0
                                if r1 < rows:
                                    out_b[Int64(r1) * ob_stride + Int64(f_lo)] = s1
                            if f_hi < cur_n:
                                if r0 < rows:
                                    out_b[Int64(r0) * ob_stride + Int64(f_hi)] = s2
                                if r1 < rows:
                                    out_b[Int64(r1) * ob_stride + Int64(f_hi)] = s3
                cute.arch.sync_threads()
                if Int32(thread) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))


@program_cache
def compile_dual_tc_projection(
    ordinal: int,
    max_rows: int,
    split_k: int,
    n_a: int,
    n_b: int,
    k: int,
):
    max_m_tiles = (int(max_rows) + 7) // 8
    kernel = DualBf16TCGemvKernel(n_a, n_b, k, max_rows=max_m_tiles * 8, split_k=split_k)
    key = (max_m_tiles, int(split_k), int(n_a), int(n_b), int(k))
    fake_x = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_wa = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_wb = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_oa = make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2)
    fake_ob = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4)
    fake_sc = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    fake_fl = make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4)
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            fake_x,
            fake_wa,
            fake_wb,
            fake_oa,
            fake_ob,
            fake_sc,
            fake_fl,
            Int32(1),
            Int64(k),
            Int64(k),
            Int64(k),
            Int64(n_a),
            Int64(n_b),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.bf16_projection.dual_tc_multirow", 1, key
            ),
        )

    def run(x, w_a, w_b, out_a, out_b, *, scratch, flags):
        with torch.cuda.device(ordinal):
            run_compiled(
                raw,
                (
                    make_ptr(BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, w_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, w_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(BFloat16, out_a.data_ptr(), cute.AddressSpace.gmem, assumed_align=2),
                    make_ptr(Float32, out_b.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
                    make_ptr(Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                    make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4),
                    Int32(x.shape[0]),
                    Int64(x.stride(0)),
                    Int64(w_a.stride(0)),
                    Int64(w_b.stride(0)),
                    Int64(out_a.stride(0)),
                    Int64(out_b.stride(0)),
                    current_cuda_stream(),
                ),
            )

    return attach_programs(run, raw)


@torch.library.custom_op(
    "b12x::bf16_dual_gemv_out", mutates_args=("out_a", "out_b", "scratch", "flags")
)
def bf16_dual_gemv_out(
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    scratch: torch.Tensor,
    flags: torch.Tensor,
    max_rows: int,
    split_k: int,
) -> None:
    if x.shape[0] == 0:
        return
    ordinal = int(x.device.index or 0)
    n_a, k = int(w_a.shape[0]), int(w_a.shape[1])
    n_b = int(w_b.shape[0])
    launcher = compile_dual_tc_projection(ordinal, max_rows, split_k, n_a, n_b, k)
    launcher(x, w_a, w_b, out_a, out_b, scratch=scratch, flags=flags)


@bf16_dual_gemv_out.register_fake
def _bf16_dual_gemv_out_fake(
    x: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    scratch: torch.Tensor,
    flags: torch.Tensor,
    max_rows: int,
    split_k: int,
) -> None:
    return None
