"""Single-launch SM120a MXFP8 A16 Tensor-Core GEMV for up to 32 BF16 rows.

Streams packed E4M3 weights + F8_128x4 UE8M0 block scales directly from global
memory, dequantizes 8 weights per half-step on-the-fly with native SM120a PTX
(``cvt.rn.bf16x2.e4m3x2`` + ``cvt.rn.bf16x2.ue8m0x2`` + ``mul.bf16x2`` —
lossless in BF16 and sharing 1 scale byte across all 16 weights), and contracts
against up to 4×8 = 32 BF16 activation rows via ``bf16_mma_m16n8k16_f32``.

When ``split_k > 1``, partials are written to caller-provided workspace scratch
with coalesced 128-bit L2-coherent vector stores/loads and reduced inside the
same launch by the last arriving CTA per 16-row N-tile, eliminating a separate
SplitKReduce kernel.
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

MXFP8_TC_GEMV_MAX_ROWS = 32
_LANES = 32

_FLAGS_CACHE: dict[tuple[int, int, int, int, int], torch.Tensor] = {}


def mxfp8_tc_gemv_workspace_bytes(
    num_tokens: int, out_features: int, split_k: int, n_per_cta: int = 16
) -> int:
    if split_k <= 1:
        return 0
    n_per_cta = int(n_per_cta)
    n_tiles = n_per_cta // 16
    n_blocks = (int(out_features) + n_per_cta - 1) // n_per_cta
    m_tiles = (int(num_tokens) + 7) // 8
    return n_blocks * int(split_k) * n_tiles * m_tiles * 128 * 4


def mxfp8_tc_gemv_flags_bytes(
    out_features: int, split_k: int, n_per_cta: int = 16
) -> int:
    if split_k <= 1:
        return 0
    n_per_cta = int(n_per_cta)
    return ((int(out_features) + n_per_cta - 1) // n_per_cta) * 4


def cached_mxfp8_tc_gemv_flags(
    device: torch.device,
    out_features: int,
    padded_k: int,
    split_k: int,
    n_per_cta: int = 16,
) -> torch.Tensor:
    n_per_cta = int(n_per_cta)
    key = (
        int(device.index or 0),
        int(out_features),
        int(padded_k),
        int(split_k),
        n_per_cta,
    )
    cached = _FLAGS_CACHE.get(key)
    if cached is not None:
        return cached
    flags = torch.zeros(
        ((int(out_features) + n_per_cta - 1) // n_per_cta,),
        dtype=torch.uint32,
        device=device,
    )
    _FLAGS_CACHE[key] = flags
    return flags


def effective_mxfp8_tc_gemv_warps(padded_k: int, split_k: int, mma_warps: int) -> int:
    chunks_per_split = (int(padded_k) // 64) // int(split_k)
    w = min(int(mma_warps), chunks_per_split)
    while chunks_per_split % w != 0:
        w -= 1
    return w


@dsl_user_op
def _ue8m0_to_bf16x2_sm120(scale_byte: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(scale_byte).ir_value(loc=loc, ip=ip)],
            """{
                .reg .b16 sf;
                .reg .b32 sf_pair;
                shl.b32 sf_pair, $1, 8;
                or.b32 sf_pair, $1, sf_pair;
                cvt.u16.u32 sf, sf_pair;
                cvt.rn.bf16x2.ue8m0x2 $0, sf;
            }""",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x4_scaled_to_bf16x2x2_sm120(
    packed4: Uint32, factors_bf16x2: Uint32, *, loc=None, ip=None
):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(packed4).ir_value(loc=loc, ip=ip),
            Uint32(factors_bf16x2).ir_value(loc=loc, ip=ip),
        ],
        """{
            .reg .b16 e01, e23;
            .reg .b32 v0, v1;
            mov.b32 {e01, e23}, $2;
            cvt.rn.bf16x2.e4m3x2 v0, e01;
            cvt.rn.bf16x2.e4m3x2 v1, e23;
            mul.bf16x2 $0, v0, $3;
            mul.bf16x2 $1, v1, $3;
        }""",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_nc_u8(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_wb_v4_f32(
    addr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.wb.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_cg_v4_f32(addr: Int64, *, loc=None, ip=None):
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
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
def _atomic_add_gpu_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_release_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        "st.global.release.gpu.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _f8_128x4_offset(row: Int32, g: Int32, k_blocks4: cutlass.Constexpr[int]) -> Int64:
    tile_r = row >> Int32(7)
    tile_k = g >> Int32(2)
    r_in = row & Int32(127)
    return (
        (Int64(tile_r) * Int64(k_blocks4) + Int64(tile_k)) * Int64(32)
        + Int64(r_in & Int32(31))
    ) * Int64(16) + Int64((r_in >> Int32(5)) * Int32(4) + (g & Int32(3)))


class MXFP8MultiRowTCGemvKernel:
    """Single-launch SM120a MXFP8 A16 Tensor-Core GEMV for M <= 32 (N=16 or N=32 per CTA)."""

    def __init__(
        self,
        n: int,
        in_features: int,
        padded_k: int,
        max_m_tiles: int,
        split_k: int = 1,
        warps: int = 8,
        n_per_cta: int = 16,
        rope_head_dim: int = 0,
        rope_dim: int = 64,
    ):
        self.n = int(n)
        self.in_features = int(in_features)
        self.k = int(padded_k)
        self.max_m_tiles = int(max_m_tiles)
        self.split_k = int(split_k)
        self.n_per_cta = int(n_per_cta)
        if self.n_per_cta not in (16, 32):
            raise ValueError(
                f"MXFP8 TC-GEMV requires n_per_cta in (16, 32), got {self.n_per_cta}"
            )
        self.rope_head_dim = int(rope_head_dim)
        self.rope_dim = int(rope_dim)
        if self.rope_head_dim != 0:
            if (
                self.rope_head_dim < 128
                or (self.rope_head_dim & (self.rope_head_dim - 1)) != 0
                or self.rope_dim != 64
            ):
                raise ValueError(
                    f"MXFP8 TC-GEMV fused RoPE requires power-of-two head_dim >= 128 and rope_dim == 64, "
                    f"got head_dim={self.rope_head_dim}, rope_dim={self.rope_dim}"
                )
            if self.n <= 0 or self.n % 2 != 0:
                raise ValueError(
                    f"MXFP8 TC-GEMV fused RoPE requires positive even n so rotary pairs are complete, got n={self.n}"
                )
        self.nope_dim = self.rope_head_dim - self.rope_dim
        self.n_tiles = self.n_per_cta // 16
        self.k_blocks4 = (self.k // 32 + 3) // 4
        if self.k % (64 * self.split_k) != 0:
            raise ValueError(
                f"MXFP8 TC-GEMV requires padded_k={self.k} divisible by 64*split_k"
            )
        if self.in_features % 16 != 0 or self.in_features > self.k:
            raise ValueError(
                "MXFP8 TC-GEMV requires 16-aligned in_features <= padded_k"
            )
        self.chunks_per_split = (self.k // 64) // self.split_k
        self.warps = effective_mxfp8_tc_gemv_warps(self.k, self.split_k, warps)
        self.chunks_per_warp = self.chunks_per_split // self.warps

    def _shared_storage_cls(self):
        class SharedStorage:
            pass

        SharedStorage.__annotations__ = {
            "partials": cute.struct.Align[
                cute.struct.MemRange[
                    Float32, self.warps * _LANES * 4 * self.max_m_tiles * self.n_tiles
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
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        out: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.make_tensor(x, cute.make_layout((Int64(1) << Int64(50),))),
            w_u8,
            s_u8,
            cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(50),))),
            scratch,
            flags,
            rows,
            x_stride_elems,
            None,
        ).launch(
            grid=((self.n + self.n_per_cta - 1) // self.n_per_cta, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.jit
    def launch_rope(
        self,
        x: cute.Pointer,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        out: cute.Pointer,
        cos_sin: cute.Pointer,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.make_tensor(x, cute.make_layout((Int64(1) << Int64(50),))),
            w_u8,
            s_u8,
            cute.make_tensor(out, cute.make_layout((Int64(1) << Int64(50),))),
            scratch,
            flags,
            rows,
            x_stride_elems,
            cute.make_tensor(cos_sin, cute.make_layout((Int64(1) << Int64(30),))),
        ).launch(
            grid=((self.n + self.n_per_cta - 1) // self.n_per_cta, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _maybe_rope(
        self,
        total: Float32,
        feature: Int32,
        row: Int32,
        rows: Int32,
        is_odd,
        in_rope,
        cos_sin,
    ) -> BFloat16:
        bf_val = total.to(BFloat16)
        if cutlass.const_expr(self.rope_head_dim > 0):
            if in_rope:
                val_f = Float32(bf_val)
                partner_f = cute.arch.shuffle_sync_bfly(val_f, offset=4)
                pair_idx = (
                    (feature & Int32(self.rope_head_dim - 1)) - Int32(self.nope_dim)
                ) >> Int32(1)
                r_safe = cutlass.min(row, rows - 1)
                c = Float32(cos_sin[Int64(r_safe) * Int64(64) + Int64(pair_idx)])
                s = Float32(
                    cos_sin[Int64(r_safe) * Int64(64) + Int64(32) + Int64(pair_idx)]
                )
                rot = (
                    (partner_f * s + val_f * c)
                    if is_odd
                    else (val_f * c - partner_f * s)
                )
                bf_val = rot.to(BFloat16)
        return bf_val

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        w_u8: cute.Pointer,
        s_u8: cute.Pointer,
        out: cute.Tensor,
        scratch: cute.Pointer,
        flags: cute.Pointer,
        rows: Int32,
        x_stride_elems: Int64,
        cos_sin,
    ):
        tid, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(tid) // _LANES
        lane = Int32(tid) % _LANES
        group = lane // 4
        quad = lane % 4
        first_n = Int32(n_tile) * self.n_per_cta
        w_base = Int64(w_u8.toint())
        s_base = Int64(s_u8.toint())
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.n_tiles, self.max_m_tiles, 4), Float32)
        for nt in cutlass.range_constexpr(self.n_tiles):
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                for e in cutlass.range_constexpr(4):
                    acc[nt, mt, e] = Float32(0.0)

        split_chunk_base = Int32(k_split) * self.chunks_per_split
        if cutlass.const_expr(self.n_tiles == 1):
            row_a = cutlass.min(first_n + group, Int32(self.n - 1))
            row_b = cutlass.min(first_n + group + 8, Int32(self.n - 1))
            for step in cutlass.range_constexpr(self.chunks_per_warp):
                chunk = split_chunk_base + warp * self.chunks_per_warp + step
                k_col = chunk * Int32(64) + quad * Int32(16)
                g = chunk * Int32(2) + (quad >> Int32(1))
                w_off = Int64(k_col)
                x_off = Int64(k_col) * Int64(2)
                pa0, pa1, pa2, pa3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_a) * Int64(self.k) + w_off
                )
                pb0, pb1, pb2, pb3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_b) * Int64(self.k) + w_off
                )
                sa = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_a, g, self.k_blocks4)
                    )
                )
                sb = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_b, g, self.k_blocks4)
                    )
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
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                        px = x_base + Int64(token) * (x_stride_elems * Int64(2)) + h_off
                        x0 = Uint32(0)
                        x1 = Uint32(0)
                        x2 = Uint32(0)
                        x3 = Uint32(0)
                        if cutlass.const_expr(
                            self.in_features == self.k
                        ) or k_col < Int32(self.in_features):
                            x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
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
                        acc[0, mt, 0], acc[0, mt, 1], acc[0, mt, 2], acc[0, mt, 3] = (
                            a0,
                            a1,
                            a2,
                            a3,
                        )
        else:
            row_a0 = cutlass.min(first_n + group, Int32(self.n - 1))
            row_b0 = cutlass.min(first_n + group + 8, Int32(self.n - 1))
            row_a1 = cutlass.min(first_n + 16 + group, Int32(self.n - 1))
            row_b1 = cutlass.min(first_n + 16 + group + 8, Int32(self.n - 1))
            x_buf = cute.make_rmem_tensor((self.max_m_tiles, 4), Uint32)
            for step in cutlass.range_constexpr(self.chunks_per_warp):
                chunk = split_chunk_base + warp * self.chunks_per_warp + step
                k_col = chunk * Int32(64) + quad * Int32(16)
                g = chunk * Int32(2) + (quad >> Int32(1))
                w_off = Int64(k_col)
                x_off = Int64(k_col) * Int64(2)
                pa0, pa1, pa2, pa3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_a0) * Int64(self.k) + w_off
                )
                pb0, pb1, pb2, pb3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_b0) * Int64(self.k) + w_off
                )
                pc0, pc1, pc2, pc3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_a1) * Int64(self.k) + w_off
                )
                pd0, pd1, pd2, pd3 = ld_global_nc_v4_u32(
                    w_base + Int64(row_b1) * Int64(self.k) + w_off
                )
                sa0 = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_a0, g, self.k_blocks4)
                    )
                )
                sb0 = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_b0, g, self.k_blocks4)
                    )
                )
                sa1 = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_a1, g, self.k_blocks4)
                    )
                )
                sb1 = _ue8m0_to_bf16x2_sm120(
                    _ld_global_nc_u8(
                        s_base + _f8_128x4_offset(row_b1, g, self.k_blocks4)
                    )
                )
                for half in cutlass.range_constexpr(2):
                    h_off = x_off + Int64(half * 16)
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                        px = x_base + Int64(token) * (x_stride_elems * Int64(2)) + h_off
                        x0 = Uint32(0)
                        x1 = Uint32(0)
                        x2 = Uint32(0)
                        x3 = Uint32(0)
                        if cutlass.const_expr(
                            self.in_features == self.k
                        ) or k_col < Int32(self.in_features):
                            x0, x1, x2, x3 = ld_global_nc_v4_u32(px)
                        x_buf[mt, 0], x_buf[mt, 1], x_buf[mt, 2], x_buf[mt, 3] = (
                            x0,
                            x1,
                            x2,
                            x3,
                        )
                    # N-tile 0
                    wa0, wa1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pa0 if cutlass.const_expr(half == 0) else pa2, sa0
                    )
                    wa2, wa3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pa1 if cutlass.const_expr(half == 0) else pa3, sa0
                    )
                    wb0, wb1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pb0 if cutlass.const_expr(half == 0) else pb2, sb0
                    )
                    wb2, wb3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pb1 if cutlass.const_expr(half == 0) else pb3, sb0
                    )
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[0, mt, 0],
                            acc[0, mt, 1],
                            acc[0, mt, 2],
                            acc[0, mt, 3],
                            wa0,
                            wb0,
                            wa1,
                            wb1,
                            x_buf[mt, 0],
                            x_buf[mt, 1],
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
                            x_buf[mt, 2],
                            x_buf[mt, 3],
                        )
                        acc[0, mt, 0], acc[0, mt, 1], acc[0, mt, 2], acc[0, mt, 3] = (
                            a0,
                            a1,
                            a2,
                            a3,
                        )
                    # N-tile 1
                    wc0, wc1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pc0 if cutlass.const_expr(half == 0) else pc2, sa1
                    )
                    wc2, wc3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pc1 if cutlass.const_expr(half == 0) else pc3, sa1
                    )
                    wd0, wd1 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pd0 if cutlass.const_expr(half == 0) else pd2, sb1
                    )
                    wd2, wd3 = _e4m3x4_scaled_to_bf16x2x2_sm120(
                        pd1 if cutlass.const_expr(half == 0) else pd3, sb1
                    )
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            acc[1, mt, 0],
                            acc[1, mt, 1],
                            acc[1, mt, 2],
                            acc[1, mt, 3],
                            wc0,
                            wd0,
                            wc1,
                            wd1,
                            x_buf[mt, 0],
                            x_buf[mt, 1],
                        )
                        a0, a1, a2, a3 = bf16_mma_m16n8k16_f32(
                            a0,
                            a1,
                            a2,
                            a3,
                            wc2,
                            wd2,
                            wc3,
                            wd3,
                            x_buf[mt, 2],
                            x_buf[mt, 3],
                        )
                        acc[1, mt, 0], acc[1, mt, 1], acc[1, mt, 2], acc[1, mt, 3] = (
                            a0,
                            a1,
                            a2,
                            a3,
                        )

        storage = cutlass.utils.SmemAllocator().allocate(self._shared_storage_cls())
        partials = storage.partials.get_tensor(
            cute.make_layout(
                (self.warps * _LANES * 4 * self.max_m_tiles * self.n_tiles,)
            )
        )
        if cutlass.const_expr(self.warps > 1):
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    base = ((slot_idx * self.warps + warp) * _LANES + lane) * 4
                    for e in cutlass.range_constexpr(4):
                        partials[base + e] = acc[nt, mt, e]
            cute.arch.sync_threads()

        is_odd = (group & Int32(1)) != Int32(0)
        if cutlass.const_expr(self.split_k == 1):
            for nt in cutlass.range_constexpr(self.n_tiles):
                in_rope = False
                if cutlass.const_expr(self.rope_head_dim > 0):
                    in_rope = (
                        (first_n + nt * 16) & Int32(self.rope_head_dim - 1)
                    ) >= Int32(self.nope_dim)
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    if warp == Int32(slot_idx % self.warps):
                        for e in cutlass.range_constexpr(4):
                            total = (
                                acc[nt, mt, e]
                                if cutlass.const_expr(self.warps == 1)
                                else Float32(0.0)
                            )
                            if cutlass.const_expr(self.warps > 1):
                                for w_i in cutlass.range_constexpr(self.warps):
                                    total = (
                                        total
                                        + partials[
                                            (
                                                (slot_idx * self.warps + w_i) * _LANES
                                                + lane
                                            )
                                            * 4
                                            + e
                                        ]
                                    )
                            feature = first_n + nt * 16 + group + (8 if e >= 2 else 0)
                            row = Int32(mt * 8) + quad * 2 + (e % 2)
                            bf_val = self._maybe_rope(
                                total, feature, row, rows, is_odd, in_rope, cos_sin
                            )
                            if feature < self.n and row < rows:
                                out[Int64(row) * Int64(self.n) + Int64(feature)] = (
                                    bf_val
                                )
        else:
            tile_elems = self.n_tiles * self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
            for nt in cutlass.range_constexpr(self.n_tiles):
                for mt in cutlass.range_constexpr(self.max_m_tiles):
                    slot_idx = nt * self.max_m_tiles + mt
                    if warp == Int32(slot_idx % self.warps):
                        t0 = (
                            acc[nt, mt, 0]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t1 = (
                            acc[nt, mt, 1]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t2 = (
                            acc[nt, mt, 2]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        t3 = (
                            acc[nt, mt, 3]
                            if cutlass.const_expr(self.warps == 1)
                            else Float32(0.0)
                        )
                        if cutlass.const_expr(self.warps > 1):
                            for w_i in cutlass.range_constexpr(self.warps):
                                pbase = (
                                    (slot_idx * self.warps + w_i) * _LANES + lane
                                ) * 4
                                t0 = t0 + partials[pbase + 0]
                                t1 = t1 + partials[pbase + 1]
                                t2 = t2 + partials[pbase + 2]
                                t3 = t3 + partials[pbase + 3]
                        vaddr = scratch_base + (
                            sbase + Int64((slot_idx * _LANES + lane) * 4)
                        ) * Int64(4)
                        _st_global_wb_v4_f32(vaddr, t0, t1, t2, t3)
            cute.arch.sync_threads()
            last_flag = storage.last.get_tensor(cute.make_layout((1,)))
            flag_addr = Int64(flags.toint()) + Int64(n_tile) * Int64(4)
            if Int32(tid) == 0:
                old = _atomic_add_gpu_u32(flag_addr, Uint32(1))
                last_flag[Int32(0)] = (
                    Uint32(1) if old == Uint32(self.split_k - 1) else Uint32(0)
                )
            cute.arch.sync_threads()
            if last_flag[Int32(0)] == Uint32(1):
                tbase = Int64(n_tile) * self.split_k * tile_elems
                for nt in cutlass.range_constexpr(self.n_tiles):
                    in_rope = False
                    if cutlass.const_expr(self.rope_head_dim > 0):
                        in_rope = (
                            (first_n + nt * 16) & Int32(self.rope_head_dim - 1)
                        ) >= Int32(self.nope_dim)
                    for mt in cutlass.range_constexpr(self.max_m_tiles):
                        slot_idx = nt * self.max_m_tiles + mt
                        if warp == Int32(slot_idx % self.warps):
                            s0 = Float32(0.0)
                            s1 = Float32(0.0)
                            s2 = Float32(0.0)
                            s3 = Float32(0.0)
                            lane_off = Int64((slot_idx * _LANES + lane) * 4)
                            for sp in cutlass.range_constexpr(self.split_k):
                                raddr = scratch_base + (
                                    tbase + Int64(sp) * tile_elems + lane_off
                                ) * Int64(4)
                                v0, v1, v2, v3 = _ld_global_cg_v4_f32(raddr)
                                s0 = s0 + v0
                                s1 = s1 + v1
                                s2 = s2 + v2
                                s3 = s3 + v3
                            f_lo = first_n + nt * 16 + group
                            f_hi = first_n + nt * 16 + group + 8
                            r0 = Int32(mt * 8) + quad * 2
                            r1 = r0 + 1
                            b0 = self._maybe_rope(
                                s0, f_lo, r0, rows, is_odd, in_rope, cos_sin
                            )
                            b1 = self._maybe_rope(
                                s1, f_lo, r1, rows, is_odd, in_rope, cos_sin
                            )
                            b2 = self._maybe_rope(
                                s2, f_hi, r0, rows, is_odd, in_rope, cos_sin
                            )
                            b3 = self._maybe_rope(
                                s3, f_hi, r1, rows, is_odd, in_rope, cos_sin
                            )
                            if f_lo < self.n:
                                if r0 < rows:
                                    out[Int64(r0) * Int64(self.n) + Int64(f_lo)] = b0
                                if r1 < rows:
                                    out[Int64(r1) * Int64(self.n) + Int64(f_lo)] = b1
                            if f_hi < self.n:
                                if r0 < rows:
                                    out[Int64(r0) * Int64(self.n) + Int64(f_hi)] = b2
                                if r1 < rows:
                                    out[Int64(r1) * Int64(self.n) + Int64(f_hi)] = b3
                cute.arch.sync_threads()
                if Int32(tid) == 0:
                    _st_global_release_u32(flag_addr, Uint32(0))


@program_cache
def compile_mxfp8_tc_gemv(
    ordinal: int,
    n: int,
    in_features: int,
    padded_k: int,
    max_rows: int,
    split_k: int,
    warps: int,
    n_per_cta: int = 16,
):
    max_m_tiles = (int(max_rows) + 7) // 8
    eff_warps = effective_mxfp8_tc_gemv_warps(padded_k, split_k, warps)
    n_per_cta = int(n_per_cta)
    kernel = MXFP8MultiRowTCGemvKernel(
        n,
        in_features,
        padded_k,
        max_m_tiles,
        split_k=split_k,
        warps=eff_warps,
        n_per_cta=n_per_cta,
    )
    key = (
        int(n),
        int(in_features),
        int(padded_k),
        max_m_tiles,
        int(split_k),
        eff_warps,
        n_per_cta,
    )
    fake = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            *fake,
            Int32(1),
            Int64(in_features),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.blockscaled.mxfp8_tc_gemv", 5, key
            ),
        )

    def run(source, values, scale_bytes, output, scratch, flags, stream):
        sc_ptr = (
            make_ptr(
                Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            if scratch is not None
            else make_ptr(
                Float32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(
                Uint32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
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
                make_ptr(
                    Uint8, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint8,
                    scale_bytes.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    BFloat16,
                    output.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                sc_ptr,
                fl_ptr,
                Int32(source.shape[0]),
                Int64(source.stride(0)),
                cuda_stream_from_int_or_current(stream),
            ),
        )

    return attach_programs(run, raw)


@program_cache
def compile_mxfp8_tc_gemv_rope(
    ordinal: int,
    n: int,
    in_features: int,
    padded_k: int,
    max_rows: int,
    split_k: int,
    warps: int,
    n_per_cta: int = 16,
    head_dim: int = 512,
    rope_dim: int = 64,
):
    max_m_tiles = (int(max_rows) + 7) // 8
    eff_warps = effective_mxfp8_tc_gemv_warps(padded_k, split_k, warps)
    n_per_cta = int(n_per_cta)
    head_dim = int(head_dim)
    rope_dim = int(rope_dim)
    kernel = MXFP8MultiRowTCGemvKernel(
        n,
        in_features,
        padded_k,
        max_m_tiles,
        split_k=split_k,
        warps=eff_warps,
        n_per_cta=n_per_cta,
        rope_head_dim=head_dim,
        rope_dim=rope_dim,
    )
    key = (
        int(n),
        int(in_features),
        int(padded_k),
        max_m_tiles,
        int(split_k),
        eff_warps,
        n_per_cta,
        head_dim,
        rope_dim,
    )
    fake = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel.launch_rope,
            *fake,
            Int32(1),
            Int64(in_features),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "gemm.blockscaled.mxfp8_tc_gemv_rope", 1, key
            ),
        )

    def run(source, values, scale_bytes, output, cos_sin, scratch, flags, stream):
        if split_k > 1 and (scratch is None or flags is None):
            raise ValueError("split_k > 1 requires scratch and flags buffers")
        sc_ptr = (
            make_ptr(
                Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            )
            if scratch is not None
            else make_ptr(
                Float32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
            )
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(
                Uint32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
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
                make_ptr(
                    Uint8, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint8,
                    scale_bytes.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    BFloat16,
                    output.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Float32,
                    cos_sin.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                sc_ptr,
                fl_ptr,
                Int32(source.shape[0]),
                Int64(source.stride(0)),
                cuda_stream_from_int_or_current(stream),
            ),
        )

    return attach_programs(run, raw)
