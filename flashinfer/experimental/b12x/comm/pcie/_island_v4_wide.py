# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Fused TP16 V4 Island AllReduce + Kimi epilogues for wide verify/draft batches (rows=33..64).

Isolated in its own module so source-dependency hashing of `_island_rs_cute.py`
and `_island_v4_mix.py` never invalidates Wave 15 autotune caches.
"""

from __future__ import annotations

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics.memory import ld_global_v4_u32, st_global_v4_u32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import (
    cuda_stream_from_int_or_current,
    cuda_stream_to_int,
    current_cuda_stream,
)
from b12x._lib.utils.pointers import make_ptr
from ._cute_intrinsics import pack_f32x2_to_bf16x2, unpack_bf16x2
from ._island_rs_cute import (
    MAX_BLOCKS,
    _ACTIVE_EPOCH_OFFSET,
    _AG_ARRIVED,
    _ISLAND_SIZE,
    _LOCAL_BAR_CNT_OFFSET,
    _LOCAL_BAR_DONE_OFFSET,
    _MAX_WORLD_SIZE,
    _RS_ARRIVED,
    _XS_ARRIVED,
    _accumulate_payload_words,
    _add_rn_f32,
    _atom_acqrel_gpu_add_u32,
    _atomic_add_global_u32,
    _div_full_f32,
    _fence_sc_sys,
    _flag_address,
    _fma_rn_f32,
    _island_v4_allpush_to_final,
    _ld_cg_v4_u32,
    _ld_nc_v4_u32,
    _load_acquire_gpu_u32,
    _mul_rn_f32,
    _pack_payload_words,
    _pair_tree_dot_fma_f32,
    _store_relaxed_gpu_u32,
    _store_relaxed_sys_u32,
    _store_release_gpu_u32,
    _triton_bfly_reduce_8warps,
    _triton_rsqrt_3584,
    _triton_rsqrt_7168,
    _wait_nosleep,
)
from ._island_v4_mix import (
    _even_odd_sum_f32,
    _max_f32,
    _neg_f32,
    _sub_rn_f32,
    _tree_max_f32,
    _triton_bfly_reduce1_warps,
    _triton_bfly_reduce2_warps,
    _triton_exp_f32,
)


class _IslandV4ResNorm7168WideLaunch:
    """Fused TP16 V4 AllReduce + ResidualAdd + RMSNorm(7168) for 33 <= rows <= 64."""

    def __init__(self, rank: int, rows: int, active_blocks: int) -> None:
        self._rank = int(rank)
        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE
        self._rows = int(rows)
        self._ab = int(active_blocks)

    @cute.jit
    def _sync3(
        self,
        slabs,
        self_base: Int64,
        region: cutlass.Constexpr[int],
        cross: cutlass.Constexpr[bool],
        bidx: Int32,
        tidx: Int32,
        tag: Uint32,
    ):
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for d in cutlass.range_constexpr(1, 4):
                peer = (
                    ((self._island ^ d) * 4 + self._lane)
                    if cutlass.const_expr(cross)
                    else (self._island * 4 + (self._lane ^ d))
                )
                sender = self._island if cutlass.const_expr(cross) else self._lane
                _store_relaxed_sys_u32(
                    _flag_address(
                        Int64(slabs[peer].toint()), region, bidx, Int32(sender)
                    ),
                    tag,
                )
            for d in cutlass.range_constexpr(1, 4):
                w = (self._island ^ d) if cutlass.const_expr(cross) else (self._lane ^ d)
                _wait_nosleep(_flag_address(self_base, region, bidx, Int32(w)), tag)
        cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer,
        residual: cute.Pointer,
        weight: cute.Pointer,
        res_out: cute.Pointer,
        norm_out: cute.Pointer,
        stage_off: Int64,
        part_off: Int64,
        final_off: Int64,
        qcap: Int64,
        res_stride: Int64,
        res_out_stride: Int64,
        norm_out_stride: Int64,
        eps: Float32,
        blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
            inp, residual, weight, res_out, norm_out,
            stage_off, part_off, final_off, qcap,
            res_stride, res_out_stride, norm_out_stride, eps,
        ).launch(grid=(blocks, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer,
        residual: cute.Pointer,
        weight: cute.Pointer,
        res_out: cute.Pointer,
        norm_out: cute.Pointer,
        stage_off: Int64,
        part_off: Int64,
        final_off: Int64,
        qcap: Int64,
        res_stride: Int64,
        res_out_stride: Int64,
        norm_out_stride: Int64,
        eps: Float32,
    ) -> None:
        slabs = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)
        self_base = Int64(slabs[self._rank].toint())
        alloc = cutlass.utils.SmemAllocator()
        sh_gen = alloc.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout((2,)), byte_alignment=4
        )
        sh_red = alloc.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((32,)), byte_alignment=16
        )
        if tidx == Int32(0):
            sh_gen[0] = (
                _atomic_add_global_u32(
                    self_base + Int64(bidx) * Int64(4), Uint32(1)
                )
                + Uint32(1)
            )
            ep = Uint32(0)
            if bidx < Int32(self._ab):
                ep = (
                    _atomic_add_global_u32(
                        self_base
                        + Int64(_ACTIVE_EPOCH_OFFSET)
                        + Int64(bidx) * Int64(4),
                        Uint32(1),
                    )
                    + Uint32(1)
                )
            sh_gen[1] = ep
        cute.arch.barrier()
        call_no = Uint32(sh_gen[0])
        if call_no % Uint32(2) != Uint32(0):
            stage_off += qcap * Int64(16)
            part_off += qcap * Int64(4)
            final_off += qcap * Int64(16)

        if bidx < Int32(self._ab):
            _island_v4_allpush_to_final(
                slabs,
                self_base,
                inp,
                inp,
                stage_off,
                part_off,
                final_off,
                qcap,
                bidx,
                tidx,
                Uint32(sh_gen[1]),
                self._island,
                self._lane,
                self._rows,
                7168,
                self._ab,
                False,
                False,
            )

        if bidx < Int32(max(self._ab, self._rows)):
            if tidx == Int32(0):
                acnt = self_base + Int64(_LOCAL_BAR_CNT_OFFSET)
                adone = self_base + Int64(_LOCAL_BAR_DONE_OFFSET)
                if bidx < Int32(self._ab):
                    old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                    if old + Uint32(1) == Uint32(self._ab):
                        _store_relaxed_gpu_u32(acnt, Uint32(0))
                        _store_release_gpu_u32(adone, call_no)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

        if bidx < Int32(self._rows):
            red_addr = self_base + final_off + Int64(bidx) * Int64(7168 * 2)
            res_addr = Int64(residual.toint()) + Int64(bidx) * res_stride * Int64(2)
            rout_addr = Int64(res_out.toint()) + Int64(bidx) * res_out_stride * Int64(2)
            nout_addr = Int64(norm_out.toint()) + Int64(bidx) * norm_out_stride * Int64(2)
            w_addr = Int64(weight.toint())
            ss = Float32(0.0)
            u = cute.make_rmem_tensor((32,), Float32)

            if tidx < Int32(256):
                for seg in cutlass.range_constexpr(3):
                    pidx = tidx + Int32(seg * 256)
                    rd = _ld_cg_v4_u32(red_addr + Int64(pidx) * Int64(16))
                    rs = ld_global_v4_u32(res_addr + Int64(pidx) * Int64(16))
                    pk = []
                    for j in cutlass.range_constexpr(4):
                        rl, rh = unpack_bf16x2(rd[j])
                        sl, sh = unpack_bf16x2(rs[j])
                        p = pack_f32x2_to_bf16x2(_add_rn_f32(sl, rl), _add_rn_f32(sh, rh))
                        pk.append(p)
                        ul, uh = unpack_bf16x2(p)
                        u[seg * 8 + j * 2] = ul
                        u[seg * 8 + j * 2 + 1] = uh
                    st_global_v4_u32(
                        rout_addr + Int64(pidx) * Int64(16), pk[0], pk[1], pk[2], pk[3]
                    )
                for k in cutlass.range_constexpr(24, 32):
                    u[k] = Float32(0.0)
                if tidx < Int32(128):
                    pidx = tidx + Int32(768)
                    rd = _ld_cg_v4_u32(red_addr + Int64(pidx) * Int64(16))
                    rs = ld_global_v4_u32(res_addr + Int64(pidx) * Int64(16))
                    pk = []
                    for j in cutlass.range_constexpr(4):
                        rl, rh = unpack_bf16x2(rd[j])
                        sl, sh = unpack_bf16x2(rs[j])
                        p = pack_f32x2_to_bf16x2(_add_rn_f32(sl, rl), _add_rn_f32(sh, rh))
                        pk.append(p)
                        ul, uh = unpack_bf16x2(p)
                        u[24 + j * 2] = ul
                        u[24 + j * 2 + 1] = uh
                    st_global_v4_u32(
                        rout_addr + Int64(pidx) * Int64(16), pk[0], pk[1], pk[2], pk[3]
                    )
                ss = _pair_tree_dot_fma_f32(u, u, 32)

            tot = _triton_bfly_reduce_8warps(ss, sh_red, tidx)
            rstd = _triton_rsqrt_7168(tot, eps)

            if tidx < Int32(256):
                for seg in cutlass.range_constexpr(3):
                    pidx = tidx + Int32(seg * 256)
                    wt = ld_global_v4_u32(w_addr + Int64(pidx) * Int64(16))
                    ow = []
                    for j in cutlass.range_constexpr(4):
                        wl, wh = unpack_bf16x2(wt[j])
                        np = pack_f32x2_to_bf16x2(
                            _mul_rn_f32(u[seg * 8 + j * 2], rstd),
                            _mul_rn_f32(u[seg * 8 + j * 2 + 1], rstd),
                        )
                        nl, nh = unpack_bf16x2(np)
                        ow.append(
                            pack_f32x2_to_bf16x2(
                                _mul_rn_f32(nl, wl), _mul_rn_f32(nh, wh)
                            )
                        )
                    st_global_v4_u32(
                        nout_addr + Int64(pidx) * Int64(16), ow[0], ow[1], ow[2], ow[3]
                    )
                if tidx < Int32(128):
                    pidx = tidx + Int32(768)
                    wt = ld_global_v4_u32(w_addr + Int64(pidx) * Int64(16))
                    ow = []
                    for j in cutlass.range_constexpr(4):
                        wl, wh = unpack_bf16x2(wt[j])
                        np = pack_f32x2_to_bf16x2(
                            _mul_rn_f32(u[24 + j * 2], rstd),
                            _mul_rn_f32(u[24 + j * 2 + 1], rstd),
                        )
                        nl, nh = unpack_bf16x2(np)
                        ow.append(
                            pack_f32x2_to_bf16x2(
                                _mul_rn_f32(nl, wl), _mul_rn_f32(nh, wh)
                            )
                        )
                    st_global_v4_u32(
                        nout_addr + Int64(pidx) * Int64(16), ow[0], ow[1], ow[2], ow[3]
                    )


class _IslandV4Norm3584WideLaunch:
    """Fused TP16 V4 AllReduce + inline RMSNorm(3584 -> 224) for 33 <= rows <= 64."""

    def __init__(self, rank: int, rows: int, active_blocks: int) -> None:
        self._rank = int(rank)
        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE
        self._rows = int(rows)
        self._ab = int(active_blocks)

    @cute.jit
    def _sync3(
        self,
        slabs,
        self_base: Int64,
        region: cutlass.Constexpr[int],
        cross: cutlass.Constexpr[bool],
        bidx: Int32,
        tidx: Int32,
        tag: Uint32,
    ):
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for d in cutlass.range_constexpr(1, 4):
                peer = (
                    ((self._island ^ d) * 4 + self._lane)
                    if cutlass.const_expr(cross)
                    else (self._island * 4 + (self._lane ^ d))
                )
                sender = self._island if cutlass.const_expr(cross) else self._lane
                _store_relaxed_sys_u32(
                    _flag_address(
                        Int64(slabs[peer].toint()), region, bidx, Int32(sender)
                    ),
                    tag,
                )
            for d in cutlass.range_constexpr(1, 4):
                w = (self._island ^ d) if cutlass.const_expr(cross) else (self._lane ^ d)
                _wait_nosleep(_flag_address(self_base, region, bidx, Int32(w)), tag)
        cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer,
        weight: cute.Pointer,
        out: cute.Pointer,
        begin_ptr: cute.Pointer,
        stage_off: Int64,
        part_off: Int64,
        final_off: Int64,
        qcap: Int64,
        eps: Float32,
        blocks: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
            inp, weight, out, begin_ptr, stage_off, part_off, final_off, qcap, eps,
        ).launch(grid=(blocks, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer,
        weight: cute.Pointer,
        out: cute.Pointer,
        begin_ptr: cute.Pointer,
        stage_off: Int64,
        part_off: Int64,
        final_off: Int64,
        qcap: Int64,
        eps: Float32,
    ) -> None:
        slabs = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)
        self_base = Int64(slabs[self._rank].toint())
        alloc = cutlass.utils.SmemAllocator()
        sh_gen = alloc.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout((2,)), byte_alignment=4
        )
        sh_red = alloc.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((32,)), byte_alignment=16
        )
        if tidx == Int32(0):
            sh_gen[0] = (
                _atomic_add_global_u32(
                    self_base + Int64(bidx) * Int64(4), Uint32(1)
                )
                + Uint32(1)
            )
            ep = Uint32(0)
            if bidx < Int32(self._ab):
                ep = (
                    _atomic_add_global_u32(
                        self_base
                        + Int64(_ACTIVE_EPOCH_OFFSET)
                        + Int64(bidx) * Int64(4),
                        Uint32(1),
                    )
                    + Uint32(1)
                )
            sh_gen[1] = ep
        cute.arch.barrier()
        call_no = Uint32(sh_gen[0])
        if call_no % Uint32(2) != Uint32(0):
            stage_off += qcap * Int64(16)
            part_off += qcap * Int64(4)
            final_off += qcap * Int64(16)

        if bidx < Int32(self._ab):
            _island_v4_allpush_to_final(
                slabs,
                self_base,
                inp,
                inp,
                stage_off,
                part_off,
                final_off,
                qcap,
                bidx,
                tidx,
                Uint32(sh_gen[1]),
                self._island,
                self._lane,
                self._rows,
                3584,
                self._ab,
                False,
                False,
            )

        if bidx < Int32(max(self._ab, self._rows)):
            if tidx == Int32(0):
                acnt = self_base + Int64(_LOCAL_BAR_CNT_OFFSET)
                adone = self_base + Int64(_LOCAL_BAR_DONE_OFFSET)
                if bidx < Int32(self._ab):
                    old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                    if old + Uint32(1) == Uint32(self._ab):
                        _store_relaxed_gpu_u32(acnt, Uint32(0))
                        _store_release_gpu_u32(adone, call_no)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

        if bidx < Int32(self._rows):
            row_addr = self_base + final_off + Int64(bidx) * Int64(3584 * 2)
            ss = Float32(0.0)
            if tidx < Int32(256):
                u = cute.make_rmem_tensor((16,), Float32)
                w0 = _ld_cg_v4_u32(row_addr + Int64(tidx) * Int64(16))
                for j in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16x2(w0[j])
                    u[j * 2] = lo
                    u[j * 2 + 1] = hi
                for k in cutlass.range_constexpr(8, 16):
                    u[k] = Float32(0.0)
                if tidx < Int32(192):
                    w1 = _ld_cg_v4_u32(
                        row_addr + Int64(tidx + Int32(256)) * Int64(16)
                    )
                    for j in cutlass.range_constexpr(4):
                        lo, hi = unpack_bf16x2(w1[j])
                        u[8 + j * 2] = lo
                        u[8 + j * 2 + 1] = hi
                ss = _pair_tree_dot_fma_f32(u, u, 16)
            tot = _triton_bfly_reduce_8warps(ss, sh_red, tidx)
            rstd = _triton_rsqrt_3584(tot, eps)
            if tidx < Int32(28):
                begin_pack = Int32(cute.arch.load(begin_ptr, Int32)) // Int32(8)
                col_pack = begin_pack + tidx
                x = _ld_cg_v4_u32(row_addr + Int64(col_pack) * Int64(16))
                wt = ld_global_v4_u32(Int64(weight.toint()) + Int64(tidx) * Int64(16))
                ow = []
                for j in cutlass.range_constexpr(4):
                    xl, xh = unpack_bf16x2(x[j])
                    wl, wh = unpack_bf16x2(wt[j])
                    np = pack_f32x2_to_bf16x2(_mul_rn_f32(xl, rstd), _mul_rn_f32(xh, rstd))
                    nl, nh = unpack_bf16x2(np)
                    ow.append(pack_f32x2_to_bf16x2(_mul_rn_f32(nl, wl), _mul_rn_f32(nh, wh)))
                st_global_v4_u32(
                    Int64(out.toint()) + (Int64(bidx) * Int64(28) + Int64(tidx)) * Int64(16),
                    ow[0], ow[1], ow[2], ow[3],
                )


class _IslandV4Mix7168WideLaunch:
    """Two-stage TP16 V4 AllReduce (+ optional inline addend) + L2-direct bitwise-exact _mix
    for 33 <= rows <= 64 (1 CTA per row in stage 2, fitting inside MAX_BLOCKS=64)."""

    def __init__(
        self,
        rank: int,
        rows: int,
        active_blocks: int,
        sources: int,
        write: int,
        mode: int,
        aux: bool,
        has_addend: bool,
    ):
        self._rank = int(rank)
        self._island = self._rank // _ISLAND_SIZE
        self._lane = self._rank % _ISLAND_SIZE
        self._rows = int(rows)
        self._ab = int(active_blocks)
        self._sources = int(sources)
        self._write = int(write)
        self._mode = int(mode)
        self._aux = bool(aux)
        self._has_addend = bool(has_addend)
        self._tile = 2 if self._sources <= 1 else 4
        self._warps = 4 if self._sources <= 1 else 8
        self._m_threads = self._warps * 32
        self._segs = 8192 // (self._m_threads * 8)
        self._nelem = self._segs * 8
        self._nwords = self._segs * 4

    @cute.jit
    def _sync3(
        self,
        slabs,
        self_base: Int64,
        region: cutlass.Constexpr[int],
        cross: cutlass.Constexpr[bool],
        bidx: Int32,
        tidx: Int32,
        tag: Uint32,
    ):
        cute.arch.barrier()
        if tidx == Int32(0):
            _fence_sc_sys()
            for d in cutlass.range_constexpr(1, 4):
                peer = (
                    ((self._island ^ d) * 4 + self._lane)
                    if cutlass.const_expr(cross)
                    else (self._island * 4 + (self._lane ^ d))
                )
                sender = self._island if cutlass.const_expr(cross) else self._lane
                _store_relaxed_sys_u32(
                    _flag_address(Int64(slabs[peer].toint()), region, bidx, Int32(sender)),
                    tag,
                )
            for d in cutlass.range_constexpr(1, 4):
                w = (self._island ^ d) if cutlass.const_expr(cross) else (self._lane ^ d)
                _wait_nosleep(_flag_address(self_base, region, bidx, Int32(w)), tag)
        cute.arch.barrier()

    @cute.jit
    def __call__(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer, addend: cute.Pointer, pref: cute.Pointer, blocks_t: cute.Pointer,
        score: cute.Pointer, norm: cute.Pointer, out: cute.Pointer, mix_out: cute.Pointer,
        stage_off: Int64, part_off: Int64, final_off: Int64, qcap: Int64,
        p_s: Int64, br_s: Int64, b_s: Int64, o_s: Int64, m_s: Int64,
        eps: Float32, oeps: Float32, grid_blocks: Int32, stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15,
            inp, addend, pref, blocks_t, score, norm, out, mix_out,
            stage_off, part_off, final_off, qcap, p_s, br_s, b_s, o_s, m_s, eps, oeps,
        ).launch(grid=(grid_blocks, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        s0: cute.Pointer, s1: cute.Pointer, s2: cute.Pointer, s3: cute.Pointer,
        s4: cute.Pointer, s5: cute.Pointer, s6: cute.Pointer, s7: cute.Pointer,
        s8: cute.Pointer, s9: cute.Pointer, s10: cute.Pointer, s11: cute.Pointer,
        s12: cute.Pointer, s13: cute.Pointer, s14: cute.Pointer, s15: cute.Pointer,
        inp: cute.Pointer, addend: cute.Pointer, pref: cute.Pointer, blocks_t: cute.Pointer,
        score: cute.Pointer, norm: cute.Pointer, out: cute.Pointer, mix_out: cute.Pointer,
        stage_off: Int64, part_off: Int64, final_off: Int64, qcap: Int64,
        p_s: Int64, br_s: Int64, b_s: Int64, o_s: Int64, m_s: Int64,
        eps: Float32, oeps: Float32,
    ) -> None:
        slabs = (s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tidx = Int32(tidx)
        bidx = Int32(bidx)
        self_base = Int64(slabs[self._rank].toint())
        alloc = cutlass.utils.SmemAllocator()
        sh_gen = alloc.allocate_tensor(Uint32, cute.make_layout((2,)), byte_alignment=4)
        sh_red = alloc.allocate_tensor(Float32, cute.make_layout((32,)), byte_alignment=16)

        if tidx == Int32(0):
            sh_gen[0] = (
                _atomic_add_global_u32(self_base + Int64(bidx) * Int64(4), Uint32(1))
                + Uint32(1)
            )
            ep = Uint32(0)
            if bidx < Int32(self._ab):
                ep = (
                    _atomic_add_global_u32(
                        self_base + Int64(_ACTIVE_EPOCH_OFFSET) + Int64(bidx) * Int64(4),
                        Uint32(1),
                    )
                    + Uint32(1)
                )
            sh_gen[1] = ep
        cute.arch.barrier()
        call_no = Uint32(sh_gen[0])
        if call_no % Uint32(2) != Uint32(0):
            stage_off += qcap * Int64(16)
            part_off += qcap * Int64(4)
            final_off += qcap * Int64(16)

        if bidx < Int32(self._ab):
            _island_v4_allpush_to_final(
                slabs,
                self_base,
                inp,
                addend,
                stage_off,
                part_off,
                final_off,
                qcap,
                bidx,
                tidx,
                Uint32(sh_gen[1]),
                self._island,
                self._lane,
                self._rows,
                7168,
                self._ab,
                self._has_addend,
                False,
            )

        if bidx < Int32(max(self._ab, self._rows)):
            if tidx == Int32(0):
                acnt = self_base + Int64(_LOCAL_BAR_CNT_OFFSET)
                adone = self_base + Int64(_LOCAL_BAR_DONE_OFFSET)
                if bidx < Int32(self._ab):
                    old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                    if old + Uint32(1) == Uint32(self._ab):
                        _store_relaxed_gpu_u32(acnt, Uint32(0))
                        _store_release_gpu_u32(adone, call_no)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

        if bidx < Int32(self._rows):
            row = Int64(bidx)
            p_addr = Int64(pref.toint()) + row * p_s * Int64(2)
            d_addr = self_base + final_off + row * Int64(7168 * 2)
            b_base = Int64(blocks_t.toint()) + row * br_s * Int64(2)
            nm_addr = Int64(norm.toint())
            o_addr = Int64(out.toint()) + row * o_s * Int64(2)
            m_addr = Int64(mix_out.toint()) + row * m_s * Int64(2)

            sw = cute.make_rmem_tensor((self._nelem,), Float32)
            acc = cute.make_rmem_tensor((self._nelem,), Float32)
            for k in cutlass.range_constexpr(self._nelem):
                sw[k] = Float32(0.0)
                acc[k] = Float32(0.0)

            maximum = Float32(-float("inf"))
            total = Float32(0.0)
            last_scores = []
            last_sd = []
            last_sc = []
            num_tiles = (self._sources + self._tile) // self._tile
            last_tile = num_tiles - 1

            if cutlass.const_expr(self._sources > 0):
                if tidx < Int32(self._m_threads):
                    st_t = cute.make_tensor(score, cute.make_layout((Int64(7168),)))
                    for seg in cutlass.range_constexpr(self._segs):
                        pidx = tidx + Int32(seg * self._m_threads)
                        if pidx < Int32(896):
                            for j in cutlass.range_constexpr(8):
                                sw[seg * 8 + j] = st_t[Int64(pidx) * Int64(8) + Int64(j)]

                for t in cutlass.range_constexpr(last_tile):
                    t_scores = []
                    for kk in cutlass.range_constexpr(self._tile):
                        src_id = t * self._tile + kk
                        rb = cute.make_rmem_tensor((self._nelem,), Float32)
                        for k in cutlass.range_constexpr(self._nelem):
                            rb[k] = Float32(0.0)
                        if tidx < Int32(self._m_threads):
                            s_addr = b_base + Int64(src_id) * b_s * Int64(2)
                            for seg in cutlass.range_constexpr(self._segs):
                                pidx = tidx + Int32(seg * self._m_threads)
                                if pidx < Int32(896):
                                    bw = ld_global_v4_u32(s_addr + Int64(pidx) * Int64(16))
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        rb[seg * 8 + j * 2] = vl
                                        rb[seg * 8 + j * 2 + 1] = vh
                        ss_l = _pair_tree_dot_fma_f32(rb, rb, self._nelem)
                        sd_l = _pair_tree_dot_fma_f32(rb, sw, self._nelem)
                        ss, sd = _triton_bfly_reduce2_warps(ss_l, sd_l, sh_red, tidx, self._warps)
                        sc_v = _triton_rsqrt_7168(ss, eps)
                        t_scores.append(_mul_rn_f32(sd, sc_v))

                    tile_max = _tree_max_f32(t_scores)
                    peak = _max_f32(maximum, tile_max)
                    rescale = _triton_exp_f32(_sub_rn_f32(maximum, peak))
                    exps = [_triton_exp_f32(_sub_rn_f32(sv, peak)) for sv in t_scores]
                    total = _fma_rn_f32(total, rescale, _even_odd_sum_f32(exps))
                    if tidx < Int32(self._m_threads):
                        for seg in cutlass.range_constexpr(self._segs):
                            pidx = tidx + Int32(seg * self._m_threads)
                            if pidx < Int32(896):
                                raw_v = [
                                    cute.make_rmem_tensor((8,), Float32)
                                    for _ in range(self._tile)
                                ]
                                for kk in cutlass.range_constexpr(self._tile):
                                    src_id = t * self._tile + kk
                                    s_addr = b_base + Int64(src_id) * b_s * Int64(2)
                                    bw = ld_global_v4_u32(s_addr + Int64(pidx) * Int64(16))
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        raw_v[kk][j * 2] = vl
                                        raw_v[kk][j * 2 + 1] = vh
                                for j in cutlass.range_constexpr(8):
                                    p01 = _fma_rn_f32(
                                        exps[0], raw_v[0][j], _mul_rn_f32(exps[1], raw_v[1][j])
                                    )
                                    p23 = _fma_rn_f32(
                                        exps[2], raw_v[2][j], _mul_rn_f32(exps[3], raw_v[3][j])
                                    )
                                    s_tile = _add_rn_f32(p01, p23)
                                    acc[seg * 8 + j] = _fma_rn_f32(
                                        acc[seg * 8 + j], rescale, s_tile
                                    )
                    maximum = peak

                for kk in cutlass.range_constexpr(self._tile):
                    src_id = last_tile * self._tile + kk
                    if cutlass.const_expr(src_id < self._sources):
                        rb = cute.make_rmem_tensor((self._nelem,), Float32)
                        for k in cutlass.range_constexpr(self._nelem):
                            rb[k] = Float32(0.0)
                        if tidx < Int32(self._m_threads):
                            s_addr = b_base + Int64(src_id) * b_s * Int64(2)
                            for seg in cutlass.range_constexpr(self._segs):
                                pidx = tidx + Int32(seg * self._m_threads)
                                if pidx < Int32(896):
                                    bw = ld_global_v4_u32(s_addr + Int64(pidx) * Int64(16))
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        rb[seg * 8 + j * 2] = vl
                                        rb[seg * 8 + j * 2 + 1] = vh
                        ss_l = _pair_tree_dot_fma_f32(rb, rb, self._nelem)
                        sd_l = _pair_tree_dot_fma_f32(rb, sw, self._nelem)
                        ss, sd = _triton_bfly_reduce2_warps(ss_l, sd_l, sh_red, tidx, self._warps)
                        sc_v = _triton_rsqrt_7168(ss, eps)
                        last_scores.append(_mul_rn_f32(sd, sc_v))
                        last_sd.append(sd)
                        last_sc.append(sc_v)
                    else:
                        last_scores.append(Float32(-float("inf")))
                        last_sd.append(Float32(0.0))
                        last_sc.append(Float32(0.0))

            cur_f = cute.make_rmem_tensor((self._nelem,), Float32)
            for k in cutlass.range_constexpr(self._nelem):
                cur_f[k] = Float32(0.0)
            if tidx < Int32(self._m_threads):
                for seg in cutlass.range_constexpr(self._segs):
                    pidx = tidx + Int32(seg * self._m_threads)
                    if pidx < Int32(896):
                        if cutlass.const_expr(self._mode == 2):
                            pk = _ld_cg_v4_u32(d_addr + Int64(pidx) * Int64(16))
                        elif cutlass.const_expr(self._mode == 1):
                            pw = ld_global_v4_u32(p_addr + Int64(pidx) * Int64(16))
                            dw = _ld_cg_v4_u32(d_addr + Int64(pidx) * Int64(16))
                            pk = []
                            for j in cutlass.range_constexpr(4):
                                pl, ph = unpack_bf16x2(pw[j])
                                dl, dh = unpack_bf16x2(dw[j])
                                pk.append(
                                    pack_f32x2_to_bf16x2(
                                        _add_rn_f32(pl, dl), _add_rn_f32(ph, dh)
                                    )
                                )
                        else:
                            pk = ld_global_v4_u32(p_addr + Int64(pidx) * Int64(16))
                        if cutlass.const_expr(self._mode != 0):
                            st_global_v4_u32(
                                p_addr + Int64(pidx) * Int64(16),
                                pk[0], pk[1], pk[2], pk[3],
                            )
                        if cutlass.const_expr(self._write >= 0):
                            st_global_v4_u32(
                                b_base
                                + Int64(self._write) * b_s * Int64(2)
                                + Int64(pidx) * Int64(16),
                                pk[0], pk[1], pk[2], pk[3],
                            )
                        for j in cutlass.range_constexpr(4):
                            ul, uh = unpack_bf16x2(pk[j])
                            cur_f[seg * 8 + j * 2] = ul
                            cur_f[seg * 8 + j * 2 + 1] = uh

            mixed = cute.make_rmem_tensor((self._nelem,), Float32)
            for k in cutlass.range_constexpr(self._nelem):
                mixed[k] = Float32(0.0)

            if cutlass.const_expr(self._sources == 0):
                for k in cutlass.range_constexpr(self._nelem):
                    mixed[k] = cur_f[k]
                if cutlass.const_expr(self._aux):
                    if tidx < Int32(self._m_threads):
                        for seg in cutlass.range_constexpr(self._segs):
                            pidx = tidx + Int32(seg * self._m_threads)
                            if pidx < Int32(896):
                                pk = []
                                for j in cutlass.range_constexpr(4):
                                    pk.append(
                                        pack_f32x2_to_bf16x2(
                                            mixed[seg * 8 + j * 2],
                                            mixed[seg * 8 + j * 2 + 1],
                                        )
                                    )
                                st_global_v4_u32(
                                    m_addr + Int64(pidx) * Int64(16),
                                    pk[0], pk[1], pk[2], pk[3],
                                )
            else:
                ss_c_l = _pair_tree_dot_fma_f32(cur_f, cur_f, self._nelem)
                sd_c_l = _pair_tree_dot_fma_f32(cur_f, sw, self._nelem)
                ss_c, sd_c = _triton_bfly_reduce2_warps(
                    ss_c_l, sd_c_l, sh_red, tidx, self._warps
                )
                sc_c = _triton_rsqrt_7168(ss_c, eps)
                cur_slot = self._sources - last_tile * self._tile
                last_scores[cur_slot] = _mul_rn_f32(sd_c, sc_c)
                last_sd[cur_slot] = sd_c
                last_sc[cur_slot] = sc_c

                tile_max = _tree_max_f32(last_scores)
                peak = _max_f32(maximum, tile_max)
                rescale = _triton_exp_f32(_sub_rn_f32(maximum, peak))
                neg_peak = _neg_f32(peak)
                valid_in_last = self._sources + 1 - last_tile * self._tile
                use_fma_score = (self._tile == 2) or (valid_in_last < 4)
                exps = []
                for kk in cutlass.range_constexpr(self._tile):
                    src_id = last_tile * self._tile + kk
                    if cutlass.const_expr(src_id > self._sources):
                        exps.append(_triton_exp_f32(_sub_rn_f32(Float32(-float("inf")), peak)))
                    elif cutlass.const_expr(use_fma_score):
                        exps.append(
                            _triton_exp_f32(_fma_rn_f32(last_sd[kk], last_sc[kk], neg_peak))
                        )
                    else:
                        exps.append(_triton_exp_f32(_sub_rn_f32(last_scores[kk], peak)))
                total = _fma_rn_f32(total, rescale, _even_odd_sum_f32(exps))

                if tidx < Int32(self._m_threads):
                    for seg in cutlass.range_constexpr(self._segs):
                        pidx = tidx + Int32(seg * self._m_threads)
                        if pidx < Int32(896):
                            raw_v = [
                                cute.make_rmem_tensor((8,), Float32)
                                for _ in range(self._tile)
                            ]
                            for kk in cutlass.range_constexpr(self._tile):
                                src_id = last_tile * self._tile + kk
                                if cutlass.const_expr(src_id > self._sources):
                                    for j in cutlass.range_constexpr(8):
                                        raw_v[kk][j] = Float32(0.0)
                                elif cutlass.const_expr(src_id == self._sources):
                                    for j in cutlass.range_constexpr(8):
                                        raw_v[kk][j] = cur_f[seg * 8 + j]
                                else:
                                    s_addr = b_base + Int64(src_id) * b_s * Int64(2)
                                    bw = ld_global_v4_u32(s_addr + Int64(pidx) * Int64(16))
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        raw_v[kk][j * 2] = vl
                                        raw_v[kk][j * 2 + 1] = vh
                            pk = []
                            for j in cutlass.range_constexpr(4):
                                s_pair = []
                                for sub in cutlass.range_constexpr(2):
                                    jj = j * 2 + sub
                                    if cutlass.const_expr(self._tile == 2):
                                        s_tile = _fma_rn_f32(
                                            exps[0],
                                            raw_v[0][jj],
                                            _mul_rn_f32(exps[1], raw_v[1][jj]),
                                        )
                                    else:
                                        p01 = _fma_rn_f32(
                                            exps[0],
                                            raw_v[0][jj],
                                            _mul_rn_f32(exps[1], raw_v[1][jj]),
                                        )
                                        p23 = _fma_rn_f32(
                                            exps[2],
                                            raw_v[2][jj],
                                            _mul_rn_f32(exps[3], raw_v[3][jj]),
                                        )
                                        s_tile = _add_rn_f32(p01, p23)
                                    a_v = _fma_rn_f32(acc[seg * 8 + jj], rescale, s_tile)
                                    s_pair.append(_div_full_f32(a_v, total))
                                p = pack_f32x2_to_bf16x2(s_pair[0], s_pair[1])
                                pk.append(p)
                                ml, mh = unpack_bf16x2(p)
                                mixed[seg * 8 + j * 2] = ml
                                mixed[seg * 8 + j * 2 + 1] = mh
                            if cutlass.const_expr(self._aux):
                                st_global_v4_u32(
                                    m_addr + Int64(pidx) * Int64(16),
                                    pk[0], pk[1], pk[2], pk[3],
                                )

            tot_m = _triton_bfly_reduce1_warps(
                _pair_tree_dot_fma_f32(mixed, mixed, self._nelem),
                sh_red,
                tidx,
                self._warps,
            )
            rstd = _triton_rsqrt_7168(tot_m, oeps)
            if tidx < Int32(self._m_threads):
                for seg in cutlass.range_constexpr(self._segs):
                    pidx = tidx + Int32(seg * self._m_threads)
                    if pidx < Int32(896):
                        nw = ld_global_v4_u32(nm_addr + Int64(pidx) * Int64(16))
                        ow = []
                        for j in cutlass.range_constexpr(4):
                            wl, wh = unpack_bf16x2(nw[j])
                            np_v = pack_f32x2_to_bf16x2(
                                _mul_rn_f32(mixed[seg * 8 + j * 2], rstd),
                                _mul_rn_f32(mixed[seg * 8 + j * 2 + 1], rstd),
                            )
                            nl, nh = unpack_bf16x2(np_v)
                            ow.append(
                                pack_f32x2_to_bf16x2(
                                    _mul_rn_f32(wl, nl), _mul_rn_f32(wh, nh)
                                )
                            )
                        st_global_v4_u32(
                            o_addr + Int64(pidx) * Int64(16),
                            ow[0], ow[1], ow[2], ow[3],
                        )


@program_cache
def get_island_resnorm7168_wide_launcher(
    rank: int,
    rows: int,
    device_index: int,
    *,
    blocks: int = MAX_BLOCKS,
    active_blocks: int | None = None,
) -> Callable[..., None]:
    del device_index
    req_ab = (16 if int(rows) >= 60 else 8) if active_blocks is None else int(active_blocks)
    ab = max(1, min(req_ab, int(blocks)))
    launch = _IslandV4ResNorm7168WideLaunch(rank, rows, ab)
    slab_ph = make_ptr(cutlass.Uint8, 256, cute.AddressSpace.gmem, assumed_align=256)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    raw = b12x_compile(
        launch,
        *(slab_ph for _ in range(_MAX_WORLD_SIZE)),
        bf_ph, bf_ph, bf_ph, bf_ph, bf_ph,
        1, 1, 1, 1, 1, 1, 1,
        Float32(1e-6),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_v4_resnorm7168_wide.bf16",
            2,
            (int(rank), int(rows), int(ab)),
            labels=("rank", "rows", "active_blocks"),
        ),
    )

    def run(
        slab_addresses, input_address, residual_address, weight_address,
        res_out_address, norm_out_address, stage_offset, part_offset, final_offset,
        quarter_capacity, res_stride, res_out_stride, norm_out_stride, eps, blocks, stream=None,
    ):
        raw(
            *(make_ptr(cutlass.Uint8, int(a), cute.AddressSpace.gmem, assumed_align=256) for a in slab_addresses),
            make_ptr(cutlass.BFloat16, int(input_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(residual_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(weight_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(res_out_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(norm_out_address), cute.AddressSpace.gmem, assumed_align=16),
            int(stage_offset), int(part_offset), int(final_offset), int(quarter_capacity),
            int(res_stride), int(res_out_stride), int(norm_out_stride),
            Float32(float(eps)), int(blocks),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


@program_cache
def get_island_norm3584_wide_launcher(
    rank: int,
    rows: int,
    device_index: int,
    *,
    blocks: int = MAX_BLOCKS,
    active_blocks: int | None = None,
) -> Callable[..., None]:
    del device_index
    req_ab = (16 if int(rows) >= 60 else 8) if active_blocks is None else int(active_blocks)
    ab = max(1, min(req_ab, int(blocks)))
    launch = _IslandV4Norm3584WideLaunch(rank, rows, ab)
    slab_ph = make_ptr(cutlass.Uint8, 256, cute.AddressSpace.gmem, assumed_align=256)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    i32_ph = make_ptr(Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    raw = b12x_compile(
        launch,
        *(slab_ph for _ in range(_MAX_WORLD_SIZE)),
        bf_ph, bf_ph, bf_ph, i32_ph,
        1, 1, 1, 1,
        Float32(1e-6),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_v4_norm3584_wide.bf16",
            2,
            (int(rank), int(rows), int(ab)),
            labels=("rank", "rows", "active_blocks"),
        ),
    )

    def run(
        slab_addresses, input_address, weight_address, out_address, begin_address,
        stage_offset, part_offset, final_offset, quarter_capacity, eps, blocks, stream=None,
    ):
        raw(
            *(make_ptr(cutlass.Uint8, int(a), cute.AddressSpace.gmem, assumed_align=256) for a in slab_addresses),
            make_ptr(cutlass.BFloat16, int(input_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(weight_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(out_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Int32, int(begin_address), cute.AddressSpace.gmem, assumed_align=4),
            int(stage_offset), int(part_offset), int(final_offset), int(quarter_capacity),
            Float32(float(eps)), int(blocks),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


@program_cache
def get_island_mix7168_wide_launcher(
    rank: int,
    rows: int,
    sources: int,
    write: int,
    mode: int,
    aux: bool,
    has_addend: bool,
    device_index: int,
    *,
    blocks: int = MAX_BLOCKS,
    active_blocks: int | None = None,
) -> Callable[..., None]:
    del device_index
    req_ab = (16 if int(rows) >= 60 else 8) if active_blocks is None else int(active_blocks)
    ab = max(1, min(req_ab, int(blocks)))
    launch = _IslandV4Mix7168WideLaunch(
        rank, rows, ab, sources, write, mode, aux, has_addend
    )
    slab_ph = make_ptr(cutlass.Uint8, 256, cute.AddressSpace.gmem, assumed_align=256)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    f32_ph = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    raw = b12x_compile(
        launch,
        *(slab_ph for _ in range(_MAX_WORLD_SIZE)),
        bf_ph, bf_ph, bf_ph, bf_ph, f32_ph, bf_ph, bf_ph, bf_ph,
        1, 1, 1, 1, 1, 1, 1, 1, 1,
        Float32(1e-6), Float32(1e-6),
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.island_v4_mix7168_wide.bf16",
            2,
            (int(rank), int(rows), int(ab), int(sources), int(write), int(mode), bool(aux), bool(has_addend)),
            labels=("rank", "rows", "active_blocks", "sources", "write", "mode", "aux", "has_addend"),
        ),
    )

    def run(
        slab_addresses, input_address, addend_address, prefix_address, blocks_address,
        score_address, norm_address, out_address, mix_address, stage_offset, part_offset,
        final_offset, quarter_capacity, prefix_stride, block_row_stride, block_stride,
        output_stride, mixture_stride, eps, output_eps, blocks, stream=None,
    ):
        raw(
            *(make_ptr(cutlass.Uint8, int(a), cute.AddressSpace.gmem, assumed_align=256) for a in slab_addresses),
            make_ptr(cutlass.BFloat16, int(input_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(addend_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(prefix_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(blocks_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Float32, int(score_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(norm_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(out_address), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.BFloat16, int(mix_address), cute.AddressSpace.gmem, assumed_align=16),
            int(stage_offset), int(part_offset), int(final_offset), int(quarter_capacity),
            int(prefix_stride), int(block_row_stride), int(block_stride), int(output_stride), int(mixture_stride),
            Float32(float(eps)), Float32(float(output_eps)), int(blocks),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


__all__ = [
    "_IslandV4ResNorm7168WideLaunch",
    "_IslandV4Norm3584WideLaunch",
    "_IslandV4Mix7168WideLaunch",
    "get_island_resnorm7168_wide_launcher",
    "get_island_norm3584_wide_launcher",
    "get_island_mix7168_wide_launcher",
]
