"""Single-launch SM120a Grouped MXFP8 A16 Tensor-Core GEMV for WO-A (+ optional fused inv-RoPE).

Streams grouped MXFP8 WO-A weights [groups, rank, group_width] + F8_128x4 UE8M0
block scales directly from global memory, optionally inverse-rotates the trailing
``rope_dim`` elements of each head in registers via 128-bit coalesced cos/sin loads,
contracts against up to 32 BF16 token rows via ``bf16_mma_m16n8k16_f32``, and writes
contiguous ``[tokens, groups * rank]`` BF16 intermediate activations directly for
``MXFP8MultiRowTCGemvKernel`` (WO-B).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_nc_v4_u32
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import cuda_stream_from_int_or_current, current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.gemm.blockscaled._mxfp8_tc_gemv import (
    _atomic_add_gpu_u32,
    _e4m3x4_scaled_to_bf16x2x2_sm120,
    _f8_128x4_offset,
    _ld_global_cg_v4_f32,
    _ld_global_nc_u8,
    _st_global_release_u32,
    _st_global_wb_v4_f32,
    _ue8m0_to_bf16x2_sm120,
    effective_mxfp8_tc_gemv_warps,
)

_LANES = 32


def _align_up(value: int, alignment: int = 256) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def wo_a16_tc_gemv_tmp_bytes(max_tokens: int, groups: int, rank: int) -> int:
    return _align_up(int(max_tokens) * int(groups) * int(rank) * 2, 256)


def wo_a16_tc_gemv_splitk_bytes(
    max_tokens: int, groups: int, rank: int, split_k: int
) -> int:
    if int(split_k) <= 1:
        return 0
    n_blocks = int(groups) * ((int(rank) + 15) // 16)
    m_tiles = (int(max_tokens) + 7) // 8
    return _align_up(n_blocks * int(split_k) * m_tiles * 128 * 4, 256)


def wo_a16_tc_gemv_flags_bytes(groups: int, rank: int, split_k: int) -> int:
    if int(split_k) <= 1:
        return 0
    return _align_up(int(groups) * ((int(rank) + 15) // 16) * 4, 256)


def wo_a16_tc_gemv_scratch_bytes(
    max_tokens: int, groups: int, rank: int, split_k: int = 1
) -> int:
    return (
        wo_a16_tc_gemv_tmp_bytes(max_tokens, groups, rank)
        + wo_a16_tc_gemv_splitk_bytes(max_tokens, groups, rank, split_k)
        + wo_a16_tc_gemv_flags_bytes(groups, rank, split_k)
    )


def cached_wo_a16_tc_gemv_flags(
    device: torch.device, groups: int, rank: int, group_width: int, split_k: int
) -> torch.Tensor:
    """Allocate a zeroed split-K completion-counter tensor for standalone callers.

    Prepared WO projections place invocation-private completion flags inside
    their caller-owned scratch buffer rather than sharing a process-global
    tensor across concurrent streams.
    """
    n_blocks = int(groups) * ((int(rank) + 15) // 16)
    return torch.zeros((n_blocks,), dtype=torch.uint32, device=device)


@dsl_user_op
def _ld_global_nc_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _inv_rotate_bf16x2(
    packed: Uint32, cos_v: Float32, sin_v: Float32, *, loc=None, ip=None
) -> Uint32:
    """Inverse-rotate one interleaved BF16 pair (a, b) with FP32 (cos, sin) -> BF16x2."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Float32(cos_v).ir_value(loc=loc, ip=ip),
                Float32(sin_v).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .b32 lo_bits, hi_bits;
                .reg .f32 a, b, ra, rb;
                shl.b32 lo_bits, $1, 16;
                and.b32 hi_bits, $1, 0xffff0000;
                mov.b32 a, lo_bits;
                mov.b32 b, hi_bits;
                mul.f32 ra, a, $2;
                fma.rn.f32 ra, b, $3, ra;
                mul.f32 rb, b, $2;
                neg.f32 a, a;
                fma.rn.f32 rb, a, $3, rb;
                cvt.rn.bf16x2.f32 $0, rb, ra;
            }""",
            "=r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _sigmoid(value: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + cute.math.exp(-value, fastmath=True))


@cute.jit
def _apply_gate(val: Float32, gate: cute.Tensor, idx: Int64) -> BFloat16:
    v_bf16 = val.to(BFloat16)
    s_bf16 = _sigmoid(Float32(gate[idx])).to(BFloat16)
    return (Float32(v_bf16) * Float32(s_bf16)).to(BFloat16)


class GroupedWoaTCGemvKernel:
    """Single-launch grouped MXFP8 A16 TC-GEMV for WO-A with optional fused inverse RoPE and gate."""

    def __init__(
        self,
        groups: int,
        rank: int,
        group_width: int,
        max_m_tiles: int,
        split_k: int = 1,
        warps: int = 8,
        fuse_inv_rope: bool = False,
        head_dim: int = 512,
        nope_dim: int = 448,
        rope_dim: int = 64,
        positions_dtype: str = "int64",
        has_gate: bool = False,
    ):
        self.groups = int(groups)
        self.rank = int(rank)
        self.k = int(group_width)
        self.max_m_tiles = int(max_m_tiles)
        self.split_k = int(split_k)
        self.fuse_inv_rope = bool(fuse_inv_rope)
        self.has_gate = bool(has_gate)
        self.head_dim = int(head_dim)
        self.nope_dim = int(nope_dim)
        self.rope_dim = int(rope_dim)
        self.half_rope = self.rope_dim // 2
        self.positions_is_i64 = positions_dtype == "int64"
        self.n_blocks_per_group = (self.rank + 15) // 16
        self.total_n_blocks = self.groups * self.n_blocks_per_group
        self.total_out_cols = self.groups * self.rank
        self.k_blocks4 = (self.k // 32 + 3) // 4
        self.s_group_bytes = ((self.rank + 127) // 128) * self.k_blocks4 * 512
        self.w_group_bytes = self.rank * self.k
        self.chunks_per_split = (self.k // 64) // self.split_k
        self.warps = effective_mxfp8_tc_gemv_warps(self.k, self.split_k, warps)
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
        positions: cute.Pointer,
        cos_sin: cute.Pointer,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        gate: cute.Pointer,
        out: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        cos_sin_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x,
            cute.make_tensor(positions, cute.make_layout((Int64(1) << Int64(30),))),
            cos_sin,
            w_u8,
            s_u8,
            cute.make_tensor(gate, cute.make_layout((Int64(1) << Int64(50),))),
            cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(50),))),
            scratch,
            flags,
            rows,
            cos_sin_stride,
        ).launch(
            grid=(self.total_n_blocks, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        positions: cute.Tensor,
        cos_sin: cute.Pointer,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        gate: cute.Tensor,
        out: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        cos_sin_stride: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        gn_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(tid) // _LANES
        lane = Int32(tid) % _LANES
        group = lane // 4
        quad = lane % 4

        g_idx = Int32(gn_tile) // Int32(self.n_blocks_per_group)
        n_local = Int32(gn_tile) - g_idx * Int32(self.n_blocks_per_group)
        first_n = n_local * Int32(16)
        row_a = cutlass.min(first_n + group, Int32(self.rank - 1))
        row_b = cutlass.min(first_n + group + 8, Int32(self.rank - 1))

        w_base = Int64(w_u8.toint()) + Int64(g_idx) * Int64(self.w_group_bytes)
        s_base = Int64(s_u8.toint()) + Int64(g_idx) * Int64(self.s_group_bytes)
        x_base = Int64(x.toint()) + Int64(g_idx) * Int64(self.k * 2)
        x_row_bytes = Int64(self.groups * self.k * 2)
        cs_base = Int64(cos_sin.toint())

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        split_chunk_base = Int32(k_split) * self.chunks_per_split
        for step in cutlass.range_constexpr(self.chunks_per_warp):
            chunk = split_chunk_base + warp * self.chunks_per_warp + step
            k_col = chunk * Int32(64) + quad * Int32(16)
            sf_g = chunk * Int32(2) + (quad >> Int32(1))
            w_off = Int64(k_col)
            x_off = Int64(k_col) * Int64(2)

            pa0, pa1, pa2, pa3 = ld_global_nc_v4_u32(
                w_base + Int64(row_a) * Int64(self.k) + w_off
            )
            pb0, pb1, pb2, pb3 = ld_global_nc_v4_u32(
                w_base + Int64(row_b) * Int64(self.k) + w_off
            )
            sa = _ue8m0_to_bf16x2_sm120(
                _ld_global_nc_u8(s_base + _f8_128x4_offset(row_a, sf_g, self.k_blocks4))
            )
            sb = _ue8m0_to_bf16x2_sm120(
                _ld_global_nc_u8(s_base + _f8_128x4_offset(row_b, sf_g, self.k_blocks4))
            )

            for half in cutlass.range_constexpr(2):
                wa0, wa1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                    pa0 if cutlass.const_expr(half == 0) else pa2, sa
                )
                wa2, wa3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                    pa1 if cutlass.const_expr(half == 0) else pa3, sa
                )
                wb0, wb1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                    pb0 if cutlass.const_expr(half == 0) else pb2, sb
                )
                wb2, wb3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                    pb1 if cutlass.const_expr(half == 0) else pb3, sb
                )
                h_off = x_off + Int64(half * 16)
                head_d = (k_col + Int32(half * 8)) % Int32(self.head_dim)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                    px = x_base + Int64(token) * x_row_bytes + h_off
                    x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                    if cutlass.const_expr(self.fuse_inv_rope):
                        if head_d >= Int32(self.nope_dim):
                            pair0 = (head_d - Int32(self.nope_dim)) >> Int32(1)
                            pos = Int64(positions[token])
                            row_cs = cs_base + pos * cos_sin_stride * Int64(4)
                            c0, c1, c2, c3 = _ld_global_nc_v4_f32(
                                row_cs + Int64(pair0) * Int64(4)
                            )
                            s0, s1, s2, s3 = _ld_global_nc_v4_f32(
                                row_cs + Int64(self.half_rope + pair0) * Int64(4)
                            )
                            x0 = _inv_rotate_bf16x2(x0, c0, s0)
                            x1 = _inv_rotate_bf16x2(x1, c1, s1)
                            x2 = _inv_rotate_bf16x2(x2, c2, s2)
                            x3 = _inv_rotate_bf16x2(x3, c3, s3)
                    a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                        acc[mt, 0],
                        acc[mt, 1],
                        acc[mt, 2],
                        acc[mt, 3],
                        wa0,
                        wb0,
                        wa1,
                        wb1,
                        x0,
                        x1,
                    )
                    a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                        a0,
                        a1,
                        a2,
                        a3,
                        wa2,
                        wb2,
                        wa3,
                        wb3,
                        x2,
                        x3,
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

        g_col_base = g_idx * Int32(self.rank)
        if cutlass.const_expr(self.split_k == 1):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    for e in cutlass.range_constexpr(4):
                        total = (
                            acc[mt, e]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        if cutlass.const_expr(self.warps > 1):
                            for w_i in cutlass.range_constexpr(self.warps):
                                total = (
                                    total
                                    + partials[
                                        ((mt * self.warps + w_i) * _LANES + lane) * 4
                                        + e
                                    ]
                                )
                        feature = first_n + group + (8 if e >= 2 else 0)
                        row = Int32(mt * 8) + quad * 2 + (e % 2)
                        if feature < self.rank and row < rows:
                            out_idx = Int64(row) * Int64(self.total_out_cols) + Int64(
                                g_col_base + feature
                            )
                            out[out_idx] = (
                                _apply_gate(total, gate, out_idx)
                                if cutlass.const_expr(self.has_gate)
                                else total.to(BFloat16)
                            )
        else:
            tile_elems = self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(gn_tile) * self.split_k + Int64(k_split)) * tile_elems
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                if warp == Int32(mt % self.warps):
                    t0 = (
                        acc[mt, 0]
                        if cutlass.const_expr(self.warps == 1)
                        else Float32(0.0)
                    )
                    t1 = (
                        acc[mt, 1]
                        if cutlass.const_expr(self.warps == 1)
                        else Float32(0.0)
                    )
                    t2 = (
                        acc[mt, 2]
                        if cutlass.const_expr(self.warps == 1)
                        else Float32(0.0)
                    )
                    t3 = (
                        acc[mt, 3]
                        if cutlass.const_expr(self.warps == 1)
                        else Float32(0.0)
                    )
                    if cutlass.const_expr(self.warps > 1):
                        for w_i in cutlass.range_constexpr(self.warps):
                            pbase = ((mt * self.warps + w_i) * _LANES + lane) * 4
                            t0 = t0 + partials[pbase + 0]
                            t1 = t1 + partials[pbase + 1]
                            t2 = t2 + partials[pbase + 2]
                            t3 = t3 + partials[pbase + 3]
                    vaddr = scratch_base + (
                        sbase + Int64((mt * _LANES + lane) * 4)
                    ) * Int64(4)
                    _st_global_wb_v4_f32(vaddr, t0, t1, t2, t3)
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(gn_tile) * Int64(4)
            if Int32(tid) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(gn_tile) * self.split_k * tile_elems
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    if warp == Int32(mt % self.warps):
                        s0 = Float32(0.0)
                        s1 = Float32(0.0)
                        s2 = Float32(0.0)
                        s3 = Float32(0.0)
                        lane_off = Int64((mt * _LANES + lane) * 4)
                        for sp in cutlass.range_constexpr(self.split_k):
                            raddr = scratch_base + (
                                tbase + Int64(sp) * tile_elems + lane_off
                            ) * Int64(4)
                            v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                            s0 = s0 + v0
                            s1 = s1 + v1
                            s2 = s2 + v2
                            s3 = s3 + v3
                        f_lo = first_n + group
                        f_hi = first_n + group + 8
                        r0 = Int32(mt * 8) + quad * 2
                        r1 = r0 + 1
                        if f_lo < self.rank:
                            if r0 < rows:
                                i0 = Int64(r0) * Int64(self.total_out_cols) + Int64(
                                    g_col_base + f_lo
                                )
                                out[i0] = (
                                    _apply_gate(s0, gate, i0)
                                    if cutlass.const_expr(self.has_gate)
                                    else s0.to(BFloat16)
                                )
                            if r1 < rows:
                                i1 = Int64(r1) * Int64(self.total_out_cols) + Int64(
                                    g_col_base + f_lo
                                )
                                out[i1] = (
                                    _apply_gate(s1, gate, i1)
                                    if cutlass.const_expr(self.has_gate)
                                    else s1.to(BFloat16)
                                )
                        if f_hi < self.rank:
                            if r0 < rows:
                                i2 = Int64(r0) * Int64(self.total_out_cols) + Int64(
                                    g_col_base + f_hi
                                )
                                out[i2] = (
                                    _apply_gate(s2, gate, i2)
                                    if cutlass.const_expr(self.has_gate)
                                    else s2.to(BFloat16)
                                )
                            if r1 < rows:
                                i3 = Int64(r1) * Int64(self.total_out_cols) + Int64(
                                    g_col_base + f_hi
                                )
                                out[i3] = (
                                    _apply_gate(s3, gate, i3)
                                    if cutlass.const_expr(self.has_gate)
                                    else s3.to(BFloat16)
                                )
                cute.arch.sync_threads()
                if Int32(tid) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))


@program_cache
def compile_grouped_woa_tc_gemv(
    ordinal: int,
    groups: int,
    rank: int,
    group_width: int,
    max_rows: int,
    split_k: int,
    warps: int,
    fuse_inv_rope: bool = False,
    head_dim: int = 512,
    nope_dim: int = 448,
    rope_dim: int = 64,
    positions_dtype: str = "int64",
    has_gate: bool = False,
):
    max_m_tiles = (int(max_rows) + 7) // 8
    eff_warps = effective_mxfp8_tc_gemv_warps(group_width, split_k, warps)
    fuse_inv_rope = bool(fuse_inv_rope)
    has_gate = bool(has_gate)
    pos_ct = Int64 if positions_dtype == "int64" else Int32
    kernel = GroupedWoaTCGemvKernel(
        groups=groups,
        rank=rank,
        group_width=group_width,
        max_m_tiles=max_m_tiles,
        split_k=split_k,
        warps=eff_warps,
        fuse_inv_rope=fuse_inv_rope,
        head_dim=head_dim,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        positions_dtype=positions_dtype,
        has_gate=has_gate,
    )
    key = (
        int(groups),
        int(rank),
        int(group_width),
        max_m_tiles,
        int(split_k),
        eff_warps,
        fuse_inv_rope,
        int(head_dim) if fuse_inv_rope else 0,
        int(nope_dim) if fuse_inv_rope else 0,
        int(rope_dim) if fuse_inv_rope else 0,
        positions_dtype if fuse_inv_rope else "",
        has_gate,
    )
    fake = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(
            pos_ct,
            16,
            cute.AddressSpace.gmem,
            assumed_align=8 if positions_dtype == "int64" else 4,
        ),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            *fake,
            Int32(1),
            Int64(rope_dim),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.wo_projection.grouped_woa_tc_gemv", 2, key
            ),
        )

    def run(
        source,
        positions,
        cos_sin,
        w_values,
        w_scale_mma,
        tmp_out,
        scratch,
        flags,
        stream,
        gate=None,
    ):
        pos_ptr = (
            make_ptr(
                pos_ct,
                positions.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=8 if positions_dtype == "int64" else 4,
            )
            if positions is not None
            else make_ptr(
                pos_ct, tmp_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=8
            )
        )
        cs_ptr = (
            make_ptr(
                Float32, cos_sin.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            if cos_sin is not None
            else make_ptr(
                Float32, tmp_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        cs_stride = Int64(cos_sin.stride(0)) if cos_sin is not None else Int64(rope_dim)
        gate_ptr = (
            make_ptr(BFloat16, gate.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            if gate is not None
            else make_ptr(
                BFloat16, tmp_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
        )
        sc_ptr = (
            make_ptr(
                Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            if scratch is not None
            else make_ptr(
                Float32, tmp_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(
                Uint32, tmp_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        run_compiled(
            raw,
            (
                make_ptr(
                    BFloat16,
                    source.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                pos_ptr,
                cs_ptr,
                make_ptr(
                    Uint8, w_values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint8,
                    w_scale_mma.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                gate_ptr,
                make_ptr(
                    BFloat16,
                    tmp_out.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                sc_ptr,
                fl_ptr,
                Int32(source.shape[0]),
                cs_stride,
                cuda_stream_from_int_or_current(stream),
            ),
        )

    return attach_programs(run, raw)
