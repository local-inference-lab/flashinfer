# SPDX-License-Identifier: Apache-2.0
"""Fused TP8 Relaxed-Ordering PCIe AllReduce + Kimi K3.1 Epilogues (rows=1..32).

Reuses the existing PCIeOneshotAllReduce IPC slabs (signal_ptrs + eager0/1_ptrs,
ZERO extra IPC VRAM) and matches standalone PCIeOneshotAllReduce's exact
(owner + 0..7) % 8 FP32 summation order and Triton epilogue arithmetic for
bit-exact (torch.equal) parity:
  - get_tp8_ro_mix7168_launcher:     addend + AR + concurrent committed-block _mix + RMSNorm
  - get_tp8_ro_norm3584_launcher:    AR + exact Triton-order RMSNorm (3584 -> 448 cols)
  - get_tp8_ro_resnorm7168_launcher: AR + concurrent residual/norm preload + ResAdd + RMSNorm
"""

from __future__ import annotations

from collections.abc import Callable

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

from ._cute_intrinsics import (
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_global_v4_u32,
    pack_f32x2_to_bf16x2,
    st_relaxed_sys_global_v4_u32,
    unpack_bf16x2,
)
from ._island_rs_cute import (
    _add_rn_f32,
    _atom_acqrel_gpu_add_u32,
    _div_full_f32,
    _fma_rn_f32,
    _ld_cg_v4_u32,
    _load_acquire_gpu_u32,
    _mul_rn_f32,
    _pair_tree_dot_fma_f32,
    _store_relaxed_gpu_u32,
    _store_release_gpu_u32,
    _triton_bfly_reduce_8warps,
    _triton_rsqrt_3584,
    _triton_rsqrt_7168,
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
from ._oneshot_cute import (
    _MAX_RANKS,
    _PLAIN_GRAPH_EPOCH_OFFSET,
    _SELF_COUNTER_BYTES,
)

# Unused 4-byte slots inside the first 128-byte padded peer-flag record
# (following _PLAIN_GRAPH_ARRIVED_OFFSET at +16; zero-initialized on alloc):
_FUSED_BAR_CNT_OFFSET = _SELF_COUNTER_BYTES + 20
_FUSED_BAR_DONE_OFFSET = _SELF_COUNTER_BYTES + 24
_FUSED_CALL_SEQ_OFFSET = _SELF_COUNTER_BYTES + 28
_FUSED_EXIT_CNT_OFFSET = _SELF_COUNTER_BYTES + 32


@cute.jit
def _publish_bf16_v4(address: Int64, words) -> None:
    encoded = cute.make_rmem_tensor((4,), Uint32)
    for word in cutlass.range_constexpr(4):
        value = words[word]
        encoded[word] = (
            value
            | (Uint32((value & Uint32(0xFFFF)) == Uint32(0)) << Uint32(15))
            | (Uint32((value & Uint32(0xFFFF0000)) == Uint32(0)) << Uint32(31))
        )
    st_relaxed_sys_global_v4_u32(
        address, encoded[0], encoded[1], encoded[2], encoded[3]
    )


@cute.jit
def _tp8_ro_allreduce_phase(
    peer_ptrs: cute.Pointer,
    input_base: Int64,
    addend_base: Int64,
    output_base: Int64,
    size_packs: Int32,
    plain_region_packs: Int64,
    plain_capacity_packs: Int64,
    bidx: Int32,
    tidx: Int32,
    slot: Uint32,
    rank: cutlass.Constexpr[int],
    ab: cutlass.Constexpr[int],
    threads: cutlass.Constexpr[int],
    transport: cutlass.Constexpr[str],
    has_addend: cutlass.Constexpr[bool],
) -> None:
    """Execute TP8 remote_push_full or remote_push_rsag on CTAs [0..ab-1]."""
    slot_table = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)
    local_base = Int64(slot_table[rank]) + plain_region_packs * Int64(16)
    index = Int32(bidx) * Int32(threads) + Int32(tidx)
    stride = Int32(ab) * Int32(threads)

    publish_index = index
    while publish_index < size_packs:
        words = ld_global_v4_u32(input_base + Int64(publish_index) * Int64(16))
        if cutlass.const_expr(has_addend):
            aw = ld_global_v4_u32(addend_base + Int64(publish_index) * Int64(16))
            summed = cute.make_rmem_tensor((4,), Uint32)
            for j in cutlass.range_constexpr(4):
                sl, sh = unpack_bf16x2(words[j])
                al, ah = unpack_bf16x2(aw[j])
                summed[j] = pack_f32x2_to_bf16x2(
                    _add_rn_f32(sl, al), _add_rn_f32(sh, ah)
                )
            words = summed
        if cutlass.const_expr(transport == "remote_push_full"):
            for target in cutlass.range_constexpr(8):
                addr = (
                    Int64(slot_table[target])
                    + plain_region_packs * Int64(16)
                    + (Int64(rank) * plain_capacity_packs + Int64(publish_index))
                    * Int64(16)
                )
                _publish_bf16_v4(addr, words)
        else:
            part = size_packs // Int32(8)
            owner = Int32(7)
            if part > Int32(0):
                owner = cutlass.min(publish_index // part, Int32(7))
            addr = (
                Int64(slot_table[owner])
                + plain_region_packs * Int64(16)
                + (Int64(rank) * plain_capacity_packs + Int64(publish_index))
                * Int64(16)
            )
            _publish_bf16_v4(addr, words)
        publish_index += stride

    reduce_index = index
    while reduce_index < size_packs:
        owner = Int32(rank)
        if cutlass.const_expr(transport == "remote_push_rsag"):
            part = size_packs // Int32(8)
            owner = Int32(7)
            if part > Int32(0):
                owner = cutlass.min(reduce_index // part, Int32(7))
        if owner == Int32(rank):
            incoming = cute.make_rmem_tensor((8, 4), Uint32)
            waiting = cutlass.Boolean(True)
            while waiting:
                waiting = cutlass.Boolean(False)
                for peer in cutlass.range_constexpr(8):
                    addr = (
                        local_base
                        + (Int64(peer) * plain_capacity_packs + Int64(reduce_index))
                        * Int64(16)
                    )
                    w = ld_relaxed_sys_global_v4_u32(addr)
                    for j in cutlass.range_constexpr(4):
                        incoming[peer, j] = w[j]
                        waiting = (
                            waiting
                            | ((w[j] & Uint32(0xFFFF)) == Uint32(0))
                            | ((w[j] & Uint32(0xFFFF0000)) == Uint32(0))
                        )
            acc = cute.make_rmem_tensor((8,), Float32)
            for peer_idx in cutlass.range_constexpr(8):
                peer = (rank + peer_idx) % 8
                for j in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16x2(incoming[peer, j])
                    if cutlass.const_expr(peer_idx == 0):
                        acc[j * 2] = lo
                        acc[j * 2 + 1] = hi
                    else:
                        acc[j * 2] = acc[j * 2] + lo
                        acc[j * 2 + 1] = acc[j * 2 + 1] + hi
            for peer in cutlass.range_constexpr(8):
                addr = (
                    local_base
                    + (Int64(peer) * plain_capacity_packs + Int64(reduce_index))
                    * Int64(16)
                )
                st_relaxed_sys_global_v4_u32(
                    addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                )
            res = cute.make_rmem_tensor((4,), Uint32)
            for j in cutlass.range_constexpr(4):
                res[j] = pack_f32x2_to_bf16x2(acc[j * 2], acc[j * 2 + 1])
            st_global_v4_u32(
                output_base + Int64(reduce_index) * Int64(16),
                res[0],
                res[1],
                res[2],
                res[3],
            )
            if cutlass.const_expr(transport == "remote_push_rsag"):
                for peer in cutlass.range_constexpr(8):
                    if cutlass.const_expr(peer != rank):
                        dest = (
                            Int64(slot_table[peer])
                            + plain_region_packs * Int64(16)
                            + (Int64(rank) * plain_capacity_packs + Int64(reduce_index))
                            * Int64(16)
                        )
                        _publish_bf16_v4(dest, res)
        reduce_index += stride

    if cutlass.const_expr(transport == "remote_push_rsag"):
        recv_index = index
        while recv_index < size_packs:
            part = size_packs // Int32(8)
            owner = Int32(7)
            if part > Int32(0):
                owner = cutlass.min(recv_index // part, Int32(7))
            if owner != Int32(rank):
                addr = (
                    local_base
                    + (Int64(owner) * plain_capacity_packs + Int64(recv_index))
                    * Int64(16)
                )
                g = cute.make_rmem_tensor((4,), Uint32)
                waiting = cutlass.Boolean(True)
                while waiting:
                    w = ld_relaxed_sys_global_v4_u32(addr)
                    waiting = cutlass.Boolean(False)
                    for j in cutlass.range_constexpr(4):
                        g[j] = w[j]
                        waiting = (
                            waiting
                            | ((w[j] & Uint32(0xFFFF)) == Uint32(0))
                            | ((w[j] & Uint32(0xFFFF0000)) == Uint32(0))
                        )
                st_global_v4_u32(
                    output_base + Int64(recv_index) * Int64(16),
                    g[0],
                    g[1],
                    g[2],
                    g[3],
                )
                st_relaxed_sys_global_v4_u32(
                    addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                )
            recv_index += stride


@cute.jit
def _tp8_grid_exit_advance(
    self_signal: Int64,
    tidx: Int32,
    total_ctas: cutlass.Constexpr[int],
    call_no: Uint32,
    gen: Uint32,
    device_slot_selection: cutlass.Constexpr[bool],
) -> None:
    """Advance shared sequence & graph epoch ONLY once every CTA in the grid finishes."""
    cute.arch.sync_threads()
    if tidx == Int32(0):
        ecnt = self_signal + Int64(_FUSED_EXIT_CNT_OFFSET)
        old = _atom_acqrel_gpu_add_u32(ecnt, Uint32(1))
        if old == Uint32(total_ctas - 1):
            _store_relaxed_gpu_u32(ecnt, Uint32(0))
            _store_relaxed_gpu_u32(
                self_signal + Int64(_FUSED_CALL_SEQ_OFFSET), call_no
            )
            if cutlass.const_expr(device_slot_selection):
                next_gen = Uint32(0) if gen == Uint32(0xFFFFFFFD) else gen + Uint32(1)
                _store_relaxed_gpu_u32(
                    self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET), next_gen
                )


class _TP8RoMix7168Launch:
    """Fused TP8 RO AllReduce (+ inline addend) + concurrent committed-block _mix + RMSNorm."""

    def __init__(
        self,
        rank: int,
        rows: int,
        ab: int,
        ar_threads: int,
        transport: str,
        device_slot_selection: bool,
        slot_bias: int,
        sources: int,
        write: int,
        mode: int,
        aux: bool,
        has_addend: bool,
    ):
        self._rank = int(rank)
        self._rows = int(rows)
        self._ab = int(ab)
        self._ar_threads = int(ar_threads)
        self._transport = str(transport)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._sources = int(sources)
        self._write = int(write)
        self._mode = int(mode)
        self._aux = bool(aux)
        self._has_addend = bool(has_addend)
        self._tile = 2 if self._sources <= 1 else 4
        self._warps = 4 if self._sources <= 1 else 8
        self._m_threads = self._warps * 32
        self._threads = max(self._m_threads, self._ar_threads)
        self._segs = 8192 // (self._m_threads * 8)
        self._nelem = self._segs * 8
        self._nwords = self._segs * 4

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        add_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        pref_ptr: cute.Pointer,
        blk_ptr: cute.Pointer,
        sc_ptr: cute.Pointer,
        nm_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        mix_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        p_s: Int64,
        br_s: Int64,
        b_s: Int64,
        o_s: Int64,
        m_s: Int64,
        eps: Float32,
        oeps: Float32,
        eager_slot: Int32,
        stream: object,
    ):
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            inp_ptr,
            add_ptr,
            full_ptr,
            pref_ptr,
            blk_ptr,
            sc_ptr,
            nm_ptr,
            out_ptr,
            mix_ptr,
            plain_region_packs,
            plain_capacity_packs,
            p_s,
            br_s,
            b_s,
            o_s,
            m_s,
            eps,
            oeps,
            eager_slot,
        ).launch(
            grid=(self._ab + self._rows, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        add_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        pref_ptr: cute.Pointer,
        blk_ptr: cute.Pointer,
        sc_ptr: cute.Pointer,
        nm_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        mix_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        p_s: Int64,
        br_s: Int64,
        b_s: Int64,
        o_s: Int64,
        m_s: Int64,
        eps: Float32,
        oeps: Float32,
        eager_slot: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        self_signal = Int64(signal_ptrs[self._rank])

        seq = ld_relaxed_gpu_u32(self_signal + Int64(_FUSED_CALL_SEQ_OFFSET))
        call_no = seq + Uint32(1)
        gen = Uint32(0)
        slot = Uint32(eager_slot)
        if cutlass.const_expr(self._device_slot_selection):
            gen = ld_relaxed_gpu_u32(self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET))
            slot = (gen + Uint32(self._slot_bias)) % Uint32(2)

        if bidx < Int32(self._ab):
            if cutlass.const_expr(self._ar_threads == self._threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(add_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 896),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    self._has_addend,
                )
            elif tidx < Int32(self._ar_threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(add_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 896),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    self._has_addend,
                )
            cute.arch.sync_threads()
            if tidx == Int32(0):
                acnt = self_signal + Int64(_FUSED_BAR_CNT_OFFSET)
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                if old == Uint32(self._ab - 1):
                    _store_relaxed_gpu_u32(acnt, Uint32(0))
                    _store_release_gpu_u32(adone, call_no)
        else:
            row = Int64(bidx - Int32(self._ab))
            smem = cutlass.utils.SmemAllocator()
            s_red = smem.allocate_tensor(
                Float32, cute.make_layout((20,)), byte_alignment=16
            )

            d_addr = Int64(full_ptr.toint()) + row * Int64(7168 * 2)
            p_addr = Int64(pref_ptr.toint()) + row * p_s * Int64(2)
            b_base = Int64(blk_ptr.toint()) + row * br_s * Int64(2)
            nm_addr = Int64(nm_ptr.toint())
            o_addr = Int64(out_ptr.toint()) + row * o_s * Int64(2)
            m_addr = Int64(mix_ptr.toint()) + row * m_s * Int64(2)

            sw = cute.make_rmem_tensor((self._nelem,), Float32)
            acc = cute.make_rmem_tensor((self._nelem,), Float32)
            for k in cutlass.range_constexpr(self._nelem):
                sw[k] = Float32(0.0)
                acc[k] = Float32(0.0)

            maximum = Float32(-float("inf"))
            total = Float32(0.0)
            n_tiles = (self._sources + self._tile) // self._tile
            last_tile = n_tiles - 1
            last_scores = []
            last_sd = []
            last_sc = []

            if cutlass.const_expr(self._sources > 0):
                sc_f32 = cute.make_tensor(sc_ptr, cute.make_layout((7168,)))
                if tidx < Int32(self._m_threads):
                    for seg in cutlass.range_constexpr(self._segs):
                        pidx = tidx + Int32(seg * self._m_threads)
                        if pidx < Int32(896):
                            base_c = pidx * Int32(8)
                            for j in cutlass.range_constexpr(8):
                                sw[seg * 8 + j] = sc_f32[base_c + Int32(j)]

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
                                    bw = ld_global_v4_u32(
                                        s_addr + Int64(pidx) * Int64(16)
                                    )
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        rb[seg * 8 + j * 2] = vl
                                        rb[seg * 8 + j * 2 + 1] = vh
                        ss_l = _pair_tree_dot_fma_f32(rb, rb, self._nelem)
                        sd_l = _pair_tree_dot_fma_f32(rb, sw, self._nelem)
                        ss, sd = _triton_bfly_reduce2_warps(
                            ss_l, sd_l, s_red, tidx, self._warps
                        )
                        t_scores.append(_mul_rn_f32(sd, _triton_rsqrt_7168(ss, eps)))
                    tile_max = _tree_max_f32(t_scores)
                    peak = _max_f32(maximum, tile_max)
                    rescale = _triton_exp_f32(_sub_rn_f32(maximum, peak))
                    exps = [
                        _triton_exp_f32(_sub_rn_f32(t_scores[kk], peak))
                        for kk in range(self._tile)
                    ]
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
                                    bw = ld_global_v4_u32(
                                        s_addr + Int64(pidx) * Int64(16)
                                    )
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        raw_v[kk][j * 2] = vl
                                        raw_v[kk][j * 2 + 1] = vh
                                for j in cutlass.range_constexpr(8):
                                    p01 = _fma_rn_f32(
                                        exps[0],
                                        raw_v[0][j],
                                        _mul_rn_f32(exps[1], raw_v[1][j]),
                                    )
                                    p23 = _fma_rn_f32(
                                        exps[2],
                                        raw_v[2][j],
                                        _mul_rn_f32(exps[3], raw_v[3][j]),
                                    )
                                    acc[seg * 8 + j] = _fma_rn_f32(
                                        acc[seg * 8 + j], rescale, _add_rn_f32(p01, p23)
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
                                    bw = ld_global_v4_u32(
                                        s_addr + Int64(pidx) * Int64(16)
                                    )
                                    for j in cutlass.range_constexpr(4):
                                        vl, vh = unpack_bf16x2(bw[j])
                                        rb[seg * 8 + j * 2] = vl
                                        rb[seg * 8 + j * 2 + 1] = vh
                        ss_l = _pair_tree_dot_fma_f32(rb, rb, self._nelem)
                        sd_l = _pair_tree_dot_fma_f32(rb, sw, self._nelem)
                        ss, sd = _triton_bfly_reduce2_warps(
                            ss_l, sd_l, s_red, tidx, self._warps
                        )
                        sc_v = _triton_rsqrt_7168(ss, eps)
                        last_scores.append(_mul_rn_f32(sd, sc_v))
                        last_sd.append(sd)
                        last_sc.append(sc_v)
                    else:
                        last_scores.append(Float32(-float("inf")))
                        last_sd.append(Float32(0.0))
                        last_sc.append(Float32(0.0))

            pref_w = cute.make_rmem_tensor((self._nwords,), Uint32)
            for k in cutlass.range_constexpr(self._nwords):
                pref_w[k] = Uint32(0)
            if cutlass.const_expr(self._mode != 2):
                if tidx < Int32(self._m_threads):
                    for seg in cutlass.range_constexpr(self._segs):
                        pidx = tidx + Int32(seg * self._m_threads)
                        if pidx < Int32(896):
                            pw = ld_global_v4_u32(p_addr + Int64(pidx) * Int64(16))
                            for j in cutlass.range_constexpr(4):
                                pref_w[seg * 4 + j] = pw[j]

            norm_w = cute.make_rmem_tensor((self._nwords,), Uint32)
            for k in cutlass.range_constexpr(self._nwords):
                norm_w[k] = Uint32(0)
            if tidx < Int32(self._m_threads):
                for seg in cutlass.range_constexpr(self._segs):
                    pidx = tidx + Int32(seg * self._m_threads)
                    if pidx < Int32(896):
                        nw = ld_global_v4_u32(nm_addr + Int64(pidx) * Int64(16))
                        for j in cutlass.range_constexpr(4):
                            norm_w[seg * 4 + j] = nw[j]

            if tidx == Int32(0):
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

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
                            dw = _ld_cg_v4_u32(d_addr + Int64(pidx) * Int64(16))
                            pk = []
                            for j in cutlass.range_constexpr(4):
                                pl, ph = unpack_bf16x2(pref_w[seg * 4 + j])
                                dl, dh = unpack_bf16x2(dw[j])
                                pk.append(
                                    pack_f32x2_to_bf16x2(
                                        _add_rn_f32(pl, dl), _add_rn_f32(ph, dh)
                                    )
                                )
                        else:
                            pk = (
                                pref_w[seg * 4],
                                pref_w[seg * 4 + 1],
                                pref_w[seg * 4 + 2],
                                pref_w[seg * 4 + 3],
                            )
                        if cutlass.const_expr(self._mode != 0):
                            st_global_v4_u32(
                                p_addr + Int64(pidx) * Int64(16),
                                pk[0],
                                pk[1],
                                pk[2],
                                pk[3],
                            )
                        if cutlass.const_expr(self._write >= 0):
                            st_global_v4_u32(
                                b_base
                                + Int64(self._write) * b_s * Int64(2)
                                + Int64(pidx) * Int64(16),
                                pk[0],
                                pk[1],
                                pk[2],
                                pk[3],
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
                                    pk[0],
                                    pk[1],
                                    pk[2],
                                    pk[3],
                                )
            else:
                ss_c_l = _pair_tree_dot_fma_f32(cur_f, cur_f, self._nelem)
                sd_c_l = _pair_tree_dot_fma_f32(cur_f, sw, self._nelem)
                ss_c, sd_c = _triton_bfly_reduce2_warps(
                    ss_c_l, sd_c_l, s_red, tidx, self._warps
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
                        exps.append(
                            _triton_exp_f32(_sub_rn_f32(Float32(-float("inf")), peak))
                        )
                    elif cutlass.const_expr(use_fma_score):
                        exps.append(
                            _triton_exp_f32(
                                _fma_rn_f32(last_sd[kk], last_sc[kk], neg_peak)
                            )
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
                                    bw = ld_global_v4_u32(
                                        s_addr + Int64(pidx) * Int64(16)
                                    )
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
                                    a_v = _fma_rn_f32(
                                        acc[seg * 8 + jj], rescale, s_tile
                                    )
                                    s_pair.append(_div_full_f32(a_v, total))
                                p = pack_f32x2_to_bf16x2(s_pair[0], s_pair[1])
                                pk.append(p)
                                ml, mh = unpack_bf16x2(p)
                                mixed[seg * 8 + j * 2] = ml
                                mixed[seg * 8 + j * 2 + 1] = mh
                            if cutlass.const_expr(self._aux):
                                st_global_v4_u32(
                                    m_addr + Int64(pidx) * Int64(16),
                                    pk[0],
                                    pk[1],
                                    pk[2],
                                    pk[3],
                                )

            tot_m = _triton_bfly_reduce1_warps(
                _pair_tree_dot_fma_f32(mixed, mixed, self._nelem),
                s_red,
                tidx,
                self._warps,
            )
            rstd = _triton_rsqrt_7168(tot_m, oeps)
            if tidx < Int32(self._m_threads):
                for seg in cutlass.range_constexpr(self._segs):
                    pidx = tidx + Int32(seg * self._m_threads)
                    if pidx < Int32(896):
                        ow = []
                        for j in cutlass.range_constexpr(4):
                            wl, wh = unpack_bf16x2(norm_w[seg * 4 + j])
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
                            ow[0],
                            ow[1],
                            ow[2],
                            ow[3],
                        )

        _tp8_grid_exit_advance(
            self_signal,
            tidx,
            self._ab + self._rows,
            call_no,
            gen,
            self._device_slot_selection,
        )


class _TP8RoNorm3584Launch:
    """Fused TP8 RO AllReduce(3584) + exact Triton-order RMSNorm -> 448 cols."""

    def __init__(
        self,
        rank: int,
        rows: int,
        ab: int,
        ar_threads: int,
        transport: str,
        device_slot_selection: bool,
        slot_bias: int,
    ):
        self._rank = int(rank)
        self._rows = int(rows)
        self._ab = int(ab)
        self._ar_threads = int(ar_threads)
        self._transport = str(transport)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = max(256, self._ar_threads)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        begin_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        eps: Float32,
        eager_slot: Int32,
        stream: object,
    ):
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            inp_ptr,
            full_ptr,
            weight_ptr,
            out_ptr,
            begin_ptr,
            plain_region_packs,
            plain_capacity_packs,
            eps,
            eager_slot,
        ).launch(
            grid=(self._ab + self._rows, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        begin_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        eps: Float32,
        eager_slot: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        self_signal = Int64(signal_ptrs[self._rank])

        seq = ld_relaxed_gpu_u32(self_signal + Int64(_FUSED_CALL_SEQ_OFFSET))
        call_no = seq + Uint32(1)
        gen = Uint32(0)
        slot = Uint32(eager_slot)
        if cutlass.const_expr(self._device_slot_selection):
            gen = ld_relaxed_gpu_u32(self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET))
            slot = (gen + Uint32(self._slot_bias)) % Uint32(2)

        if bidx < Int32(self._ab):
            if cutlass.const_expr(self._ar_threads == self._threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(inp_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 448),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    False,
                )
            elif tidx < Int32(self._ar_threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(inp_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 448),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    False,
                )
            cute.arch.sync_threads()
            if tidx == Int32(0):
                acnt = self_signal + Int64(_FUSED_BAR_CNT_OFFSET)
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                if old == Uint32(self._ab - 1):
                    _store_relaxed_gpu_u32(acnt, Uint32(0))
                    _store_release_gpu_u32(adone, call_no)
        else:
            row = Int64(bidx - Int32(self._ab))
            smem = cutlass.utils.SmemAllocator()
            sh_red = smem.allocate_tensor(
                Float32, cute.make_layout((20,)), byte_alignment=16
            )
            # Preload weight while AR is in flight!
            wt = cute.make_rmem_tensor((4,), Uint32)
            for j in cutlass.range_constexpr(4):
                wt[j] = Uint32(0)
            if tidx < Int32(56):
                wv = ld_global_v4_u32(
                    Int64(weight_ptr.toint()) + Int64(tidx) * Int64(16)
                )
                for j in cutlass.range_constexpr(4):
                    wt[j] = wv[j]

            if tidx == Int32(0):
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

            row_addr = Int64(full_ptr.toint()) + row * Int64(3584 * 2)
            u = cute.make_rmem_tensor((16,), Float32)
            for k in cutlass.range_constexpr(16):
                u[k] = Float32(0.0)
            if tidx < Int32(256):
                w0 = _ld_cg_v4_u32(row_addr + Int64(tidx) * Int64(16))
                for j in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16x2(w0[j])
                    u[j * 2] = lo
                    u[j * 2 + 1] = hi
            if tidx < Int32(192):
                w1 = _ld_cg_v4_u32(row_addr + Int64(tidx + Int32(256)) * Int64(16))
                for j in cutlass.range_constexpr(4):
                    lo, hi = unpack_bf16x2(w1[j])
                    u[8 + j * 2] = lo
                    u[8 + j * 2 + 1] = hi
            ss = _pair_tree_dot_fma_f32(u, u, 16)
            tot = _triton_bfly_reduce_8warps(ss, sh_red, tidx)
            rstd = _triton_rsqrt_3584(tot, eps)
            if tidx < Int32(56):
                begin_pack = Int32(cute.arch.load(begin_ptr, Int32)) // Int32(8)
                col_pack = begin_pack + tidx
                x = _ld_cg_v4_u32(row_addr + Int64(col_pack) * Int64(16))
                ow = []
                for j in cutlass.range_constexpr(4):
                    xl, xh = unpack_bf16x2(x[j])
                    wl, wh = unpack_bf16x2(wt[j])
                    np = pack_f32x2_to_bf16x2(
                        _mul_rn_f32(xl, rstd), _mul_rn_f32(xh, rstd)
                    )
                    nl, nh = unpack_bf16x2(np)
                    ow.append(
                        pack_f32x2_to_bf16x2(
                            _mul_rn_f32(nl, wl), _mul_rn_f32(nh, wh)
                        )
                    )
                st_global_v4_u32(
                    Int64(out_ptr.toint())
                    + (row * Int64(56) + Int64(tidx)) * Int64(16),
                    ow[0],
                    ow[1],
                    ow[2],
                    ow[3],
                )

        _tp8_grid_exit_advance(
            self_signal,
            tidx,
            self._ab + self._rows,
            call_no,
            gen,
            self._device_slot_selection,
        )


class _TP8RoResNorm7168Launch:
    """Fused TP8 RO AllReduce(7168) + concurrent residual/weight preload + ResAdd + RMSNorm."""

    def __init__(
        self,
        rank: int,
        rows: int,
        ab: int,
        ar_threads: int,
        transport: str,
        device_slot_selection: bool,
        slot_bias: int,
    ):
        self._rank = int(rank)
        self._rows = int(rows)
        self._ab = int(ab)
        self._ar_threads = int(ar_threads)
        self._transport = str(transport)
        self._device_slot_selection = bool(device_slot_selection)
        self._slot_bias = int(slot_bias) & 1
        self._threads = max(256, self._ar_threads)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        res_out_ptr: cute.Pointer,
        norm_out_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        res_s: Int64,
        rout_s: Int64,
        nout_s: Int64,
        eps: Float32,
        eager_slot: Int32,
        stream: object,
    ):
        self.kernel(
            peer_ptrs,
            signal_ptrs,
            inp_ptr,
            full_ptr,
            res_ptr,
            weight_ptr,
            res_out_ptr,
            norm_out_ptr,
            plain_region_packs,
            plain_capacity_packs,
            res_s,
            rout_s,
            nout_s,
            eps,
            eager_slot,
        ).launch(
            grid=(self._ab + self._rows, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        inp_ptr: cute.Pointer,
        full_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        res_out_ptr: cute.Pointer,
        norm_out_ptr: cute.Pointer,
        plain_region_packs: Int64,
        plain_capacity_packs: Int64,
        res_s: Int64,
        rout_s: Int64,
        nout_s: Int64,
        eps: Float32,
        eager_slot: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        self_signal = Int64(signal_ptrs[self._rank])

        seq = ld_relaxed_gpu_u32(self_signal + Int64(_FUSED_CALL_SEQ_OFFSET))
        call_no = seq + Uint32(1)
        gen = Uint32(0)
        slot = Uint32(eager_slot)
        if cutlass.const_expr(self._device_slot_selection):
            gen = ld_relaxed_gpu_u32(self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET))
            slot = (gen + Uint32(self._slot_bias)) % Uint32(2)

        if bidx < Int32(self._ab):
            if cutlass.const_expr(self._ar_threads == self._threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(inp_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 896),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    False,
                )
            elif tidx < Int32(self._ar_threads):
                _tp8_ro_allreduce_phase(
                    peer_ptrs,
                    Int64(inp_ptr.toint()),
                    Int64(inp_ptr.toint()),
                    Int64(full_ptr.toint()),
                    Int32(self._rows * 896),
                    plain_region_packs,
                    plain_capacity_packs,
                    bidx,
                    tidx,
                    slot,
                    self._rank,
                    self._ab,
                    self._ar_threads,
                    self._transport,
                    False,
                )
            cute.arch.sync_threads()
            if tidx == Int32(0):
                acnt = self_signal + Int64(_FUSED_BAR_CNT_OFFSET)
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                old = _atom_acqrel_gpu_add_u32(acnt, Uint32(1))
                if old == Uint32(self._ab - 1):
                    _store_relaxed_gpu_u32(acnt, Uint32(0))
                    _store_release_gpu_u32(adone, call_no)
        else:
            row = Int64(bidx - Int32(self._ab))
            smem = cutlass.utils.SmemAllocator()
            sh_red = smem.allocate_tensor(
                Float32, cute.make_layout((20,)), byte_alignment=16
            )
            red_addr = Int64(full_ptr.toint()) + row * Int64(7168 * 2)
            res_addr = Int64(res_ptr.toint()) + row * res_s * Int64(2)
            rout_addr = Int64(res_out_ptr.toint()) + row * rout_s * Int64(2)
            nout_addr = Int64(norm_out_ptr.toint()) + row * nout_s * Int64(2)
            w_addr = Int64(weight_ptr.toint())

            # Preload residual and norm weight into registers while AR is in flight!
            rs_w = cute.make_rmem_tensor((16,), Uint32)
            wt_w = cute.make_rmem_tensor((16,), Uint32)
            for k in cutlass.range_constexpr(16):
                rs_w[k] = Uint32(0)
                wt_w[k] = Uint32(0)
            if tidx < Int32(256):
                for seg in cutlass.range_constexpr(3):
                    pidx = tidx + Int32(seg * 256)
                    rv = ld_global_v4_u32(res_addr + Int64(pidx) * Int64(16))
                    wv = ld_global_v4_u32(w_addr + Int64(pidx) * Int64(16))
                    for j in cutlass.range_constexpr(4):
                        rs_w[seg * 4 + j] = rv[j]
                        wt_w[seg * 4 + j] = wv[j]
            if tidx < Int32(128):
                pidx = tidx + Int32(768)
                rv = ld_global_v4_u32(res_addr + Int64(pidx) * Int64(16))
                wv = ld_global_v4_u32(w_addr + Int64(pidx) * Int64(16))
                for j in cutlass.range_constexpr(4):
                    rs_w[12 + j] = rv[j]
                    wt_w[12 + j] = wv[j]

            if tidx == Int32(0):
                adone = self_signal + Int64(_FUSED_BAR_DONE_OFFSET)
                obs = _load_acquire_gpu_u32(adone)
                while Int32(obs - call_no) < Int32(0):
                    obs = _load_acquire_gpu_u32(adone)
            cute.arch.barrier()

            u = cute.make_rmem_tensor((32,), Float32)
            for k in cutlass.range_constexpr(32):
                u[k] = Float32(0.0)
            if tidx < Int32(256):
                for seg in cutlass.range_constexpr(3):
                    pidx = tidx + Int32(seg * 256)
                    rd = _ld_cg_v4_u32(red_addr + Int64(pidx) * Int64(16))
                    pk = []
                    for j in cutlass.range_constexpr(4):
                        rl, rh = unpack_bf16x2(rd[j])
                        sl, sh = unpack_bf16x2(rs_w[seg * 4 + j])
                        p = pack_f32x2_to_bf16x2(
                            _add_rn_f32(sl, rl), _add_rn_f32(sh, rh)
                        )
                        pk.append(p)
                        ul, uh = unpack_bf16x2(p)
                        u[seg * 8 + j * 2] = ul
                        u[seg * 8 + j * 2 + 1] = uh
                    st_global_v4_u32(
                        rout_addr + Int64(pidx) * Int64(16), pk[0], pk[1], pk[2], pk[3]
                    )
            if tidx < Int32(128):
                pidx = tidx + Int32(768)
                rd = _ld_cg_v4_u32(red_addr + Int64(pidx) * Int64(16))
                pk = []
                for j in cutlass.range_constexpr(4):
                    rl, rh = unpack_bf16x2(rd[j])
                    sl, sh = unpack_bf16x2(rs_w[12 + j])
                    p = pack_f32x2_to_bf16x2(
                        _add_rn_f32(sl, rl), _add_rn_f32(sh, rh)
                    )
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
                    ow = []
                    for j in cutlass.range_constexpr(4):
                        wl, wh = unpack_bf16x2(wt_w[seg * 4 + j])
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
                ow = []
                for j in cutlass.range_constexpr(4):
                    wl, wh = unpack_bf16x2(wt_w[12 + j])
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

        _tp8_grid_exit_advance(
            self_signal,
            tidx,
            self._ab + self._rows,
            call_no,
            gen,
            self._device_slot_selection,
        )


@program_cache
def get_tp8_ro_mix7168_launcher(
    rank: int,
    rows: int,
    ab: int,
    ar_threads: int,
    transport: str,
    device_slot_selection: bool,
    slot_bias: int,
    sources: int,
    write: int,
    mode: int,
    aux: bool,
    has_addend: bool,
    device_index: int,
) -> Callable[..., None]:
    del device_index
    launch = _TP8RoMix7168Launch(
        rank,
        rows,
        ab,
        ar_threads,
        transport,
        device_slot_selection,
        slot_bias,
        sources,
        write,
        mode,
        aux,
        has_addend,
    )
    u64_ph = make_ptr(cutlass.Uint64, 16, cute.AddressSpace.gmem, assumed_align=16)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    f32_ph = make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16)
    raw = b12x_compile(
        launch,
        u64_ph,
        u64_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        f32_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1e-6),
        Float32(1e-6),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.tp8_ro_mix7168.bf16",
            2,
            (
                int(rank),
                int(rows),
                int(ab),
                int(ar_threads),
                str(transport),
                bool(device_slot_selection),
                int(slot_bias) & 1,
                int(sources),
                int(write),
                int(mode),
                bool(aux),
                bool(has_addend),
            ),
        ),
    )

    def run(
        peer_table_addr,
        sig_table_addr,
        inp_addr,
        add_addr,
        full_addr,
        pref_addr,
        blk_addr,
        sc_addr,
        nm_addr,
        out_addr,
        mix_addr,
        plain_region_packs,
        plain_capacity_packs,
        p_s,
        br_s,
        b_s,
        o_s,
        m_s,
        eps,
        oeps,
        eager_slot,
        stream=None,
    ):
        raw(
            make_ptr(
                cutlass.Uint64, int(peer_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint64, int(sig_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(inp_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(add_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(full_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(pref_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(blk_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(Float32, int(sc_addr), cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(
                cutlass.BFloat16, int(nm_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(out_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(mix_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            Int64(int(plain_region_packs)),
            Int64(int(plain_capacity_packs)),
            Int64(int(p_s)),
            Int64(int(br_s)),
            Int64(int(b_s)),
            Int64(int(o_s)),
            Int64(int(m_s)),
            Float32(float(eps)),
            Float32(float(oeps)),
            Int32(int(eager_slot)),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


@program_cache
def get_tp8_ro_norm3584_launcher(
    rank: int,
    rows: int,
    ab: int,
    ar_threads: int,
    transport: str,
    device_slot_selection: bool,
    slot_bias: int,
    device_index: int,
) -> Callable[..., None]:
    del device_index
    launch = _TP8RoNorm3584Launch(
        rank, rows, ab, ar_threads, transport, device_slot_selection, slot_bias
    )
    u64_ph = make_ptr(cutlass.Uint64, 16, cute.AddressSpace.gmem, assumed_align=16)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    i32_ph = make_ptr(Int32, 4, cute.AddressSpace.gmem, assumed_align=4)
    raw = b12x_compile(
        launch,
        u64_ph,
        u64_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        i32_ph,
        Int64(1),
        Int64(1),
        Float32(1e-6),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.tp8_ro_norm3584.bf16",
            2,
            (
                int(rank),
                int(rows),
                int(ab),
                int(ar_threads),
                str(transport),
                bool(device_slot_selection),
                int(slot_bias) & 1,
            ),
        ),
    )

    def run(
        peer_table_addr,
        sig_table_addr,
        inp_addr,
        full_addr,
        weight_addr,
        out_addr,
        begin_addr,
        plain_region_packs,
        plain_capacity_packs,
        eps,
        eager_slot,
        stream=None,
    ):
        raw(
            make_ptr(
                cutlass.Uint64, int(peer_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint64, int(sig_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(inp_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(full_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(weight_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(out_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(Int32, int(begin_addr), cute.AddressSpace.gmem, assumed_align=4),
            Int64(int(plain_region_packs)),
            Int64(int(plain_capacity_packs)),
            Float32(float(eps)),
            Int32(int(eager_slot)),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


@program_cache
def get_tp8_ro_resnorm7168_launcher(
    rank: int,
    rows: int,
    ab: int,
    ar_threads: int,
    transport: str,
    device_slot_selection: bool,
    slot_bias: int,
    device_index: int,
) -> Callable[..., None]:
    del device_index
    launch = _TP8RoResNorm7168Launch(
        rank, rows, ab, ar_threads, transport, device_slot_selection, slot_bias
    )
    u64_ph = make_ptr(cutlass.Uint64, 16, cute.AddressSpace.gmem, assumed_align=16)
    bf_ph = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16)
    raw = b12x_compile(
        launch,
        u64_ph,
        u64_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        bf_ph,
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Int64(1),
        Float32(1e-6),
        Int32(0),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.tp8_ro_resnorm7168.bf16",
            2,
            (
                int(rank),
                int(rows),
                int(ab),
                int(ar_threads),
                str(transport),
                bool(device_slot_selection),
                int(slot_bias) & 1,
            ),
        ),
    )

    def run(
        peer_table_addr,
        sig_table_addr,
        inp_addr,
        full_addr,
        res_addr,
        weight_addr,
        res_out_addr,
        norm_out_addr,
        plain_region_packs,
        plain_capacity_packs,
        res_s,
        rout_s,
        nout_s,
        eps,
        eager_slot,
        stream=None,
    ):
        raw(
            make_ptr(
                cutlass.Uint64, int(peer_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint64, int(sig_table_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(inp_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(full_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(res_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(weight_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(res_out_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.BFloat16, int(norm_out_addr), cute.AddressSpace.gmem, assumed_align=16
            ),
            Int64(int(plain_region_packs)),
            Int64(int(plain_capacity_packs)),
            Int64(int(res_s)),
            Int64(int(rout_s)),
            Int64(int(nout_s)),
            Float32(float(eps)),
            Int32(int(eager_slot)),
            cuda_stream_from_int_or_current(cuda_stream_to_int(stream)),
        )

    return attach_programs(run, raw)


__all__ = [
    "get_tp8_ro_mix7168_launcher",
    "get_tp8_ro_norm3584_launcher",
    "get_tp8_ro_resnorm7168_launcher",
]
