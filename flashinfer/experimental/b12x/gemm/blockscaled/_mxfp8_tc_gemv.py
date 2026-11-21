"""Single-launch SM120a MXFP8 A16 Tensor-Core GEMV for up to 32 BF16 rows.

Streams packed E4M3 weights + F8_128x4 UE8M0 block scales directly from global
memory, dequantizes 16 weights per thread on-the-fly with native SM120a PTX
(``cvt.rn.bf16x2.e4m3x2`` + ``cvt.rn.bf16x2.ue8m0x2`` + ``mul.bf16x2`` —
lossless in BF16 and sharing 1 scale byte across all 16 weights), and contracts
against up to 4×8 = 32 BF16 activation rows via ``bf16_mma_m16n8k16_f32``.

When ``split_k > 1``, partials are written to caller-provided workspace scratch
with L2-coherent stores/loads and reduced inside the same launch by the last
arriving CTA per 16-row N-tile, eliminating a separate SplitKReduce kernel.
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

_FLAGS_CACHE: dict[tuple[int, int, int, int], torch.Tensor] = {}


def mxfp8_tc_gemv_workspace_bytes(
    num_tokens: int, out_features: int, split_k: int
) -> int:
    if split_k <= 1:
        return 0
    n_blocks = (int(out_features) + 15) // 16
    m_tiles = (int(num_tokens) + 7) // 8
    return n_blocks * int(split_k) * m_tiles * 128 * 4


def mxfp8_tc_gemv_flags_bytes(out_features: int, split_k: int) -> int:
    if split_k <= 1:
        return 0
    return ((int(out_features) + 15) // 16) * 4


def cached_mxfp8_tc_gemv_flags(
    device: torch.device, out_features: int, padded_k: int, split_k: int
) -> torch.Tensor:
    key = (int(device.index or 0), int(out_features), int(padded_k), int(split_k))
    cached = _FLAGS_CACHE.get(key)
    if cached is not None:
        return cached
    flags = torch.zeros(((int(out_features) + 15) // 16,), dtype=torch.uint32, device=device)
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
def _st_global_wb_f32(addr: Int64, val: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(val).ir_value(loc=loc, ip=ip)],
        "st.global.wb.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_global_cg_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.cg.f32 $0, [$1];",
            "=f,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _atomic_add_gpu_u32(addr: Int64, val: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
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
def _f8_128x4_offset(
    row: Int32, g: Int32, k_blocks4: cutlass.Constexpr[int]
) -> Int64:
    tile_r = row >> Int32(7)
    tile_k = g >> Int32(2)
    r_in = row & Int32(127)
    return (
        (
            (Int64(tile_r) * Int64(k_blocks4) + Int64(tile_k)) * Int64(32)
            + Int64(r_in & Int32(31))
        )
        * Int64(16)
        + Int64((r_in >> Int32(5)) * Int32(4) + (g & Int32(3)))
    )


class MXFP8MultiRowTCGemvKernel:
    """Single-launch SM120a MXFP8 A16 Tensor-Core GEMV for M <= 32."""

    def __init__(
        self,
        n: int,
        in_features: int,
        padded_k: int,
        max_m_tiles: int,
        split_k: int = 1,
        warps: int = 8,
    ):
        self.n = int(n)
        self.in_features = int(in_features)
        self.k = int(padded_k)
        self.max_m_tiles = int(max_m_tiles)
        self.split_k = int(split_k)
        self.k_blocks4 = (self.k // 32 + 3) // 4
        if self.k % (64 * self.split_k) != 0:
            raise ValueError(
                f"MXFP8 TC-GEMV requires padded_k={self.k} divisible by 64*split_k"
            )
        if self.in_features % 16 != 0 or self.in_features > self.k:
            raise ValueError("MXFP8 TC-GEMV requires 16-aligned in_features <= padded_k")
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
        ).launch(
            grid=((self.n + 15) // 16, self.split_k, 1),
            block=(self.warps * _LANES, 1, 1),
            stream=stream,
        )

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
    ):
        tid, _, _ = cute.arch.thread_idx()
        n_tile, k_split, _ = cute.arch.block_idx()
        warp = Int32(tid) // _LANES
        lane = Int32(tid) % _LANES
        group = lane // 4
        quad = lane % 4
        first_n = Int32(n_tile) * 16
        row_a = cutlass.min(first_n + group, Int32(self.n - 1))
        row_b = cutlass.min(first_n + group + 8, Int32(self.n - 1))
        w_base = Int64(w_u8.toint())
        s_base = Int64(s_u8.toint())
        x_base = get_ptr_as_int64(x, Int64(0))

        acc = cute.make_rmem_tensor((self.max_m_tiles, 4), Float32)
        for mt in cutlass.range_constexpr(self.max_m_tiles):
            for e in cutlass.range_constexpr(4):
                acc[mt, e] = Float32(0.0)

        split_chunk_base = Int32(k_split) * self.chunks_per_split
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
                _ld_global_nc_u8(s_base + _f8_128x4_offset(row_a, g, self.k_blocks4))
            )
            sb = _ue8m0_to_bf16x2_sm120(
                _ld_global_nc_u8(s_base + _f8_128x4_offset(row_b, g, self.k_blocks4))
            )
            wa0, wa1 = _e4m3x4_scaled_to_bf16x2x2_sm120(pa0, sa)
            wa2, wa3 = _e4m3x4_scaled_to_bf16x2x2_sm120(pa1, sa)
            wa4, wa5 = _e4m3x4_scaled_to_bf16x2x2_sm120(pa2, sa)
            wa6, wa7 = _e4m3x4_scaled_to_bf16x2x2_sm120(pa3, sa)
            wb0, wb1 = _e4m3x4_scaled_to_bf16x2x2_sm120(pb0, sb)
            wb2, wb3 = _e4m3x4_scaled_to_bf16x2x2_sm120(pb1, sb)
            wb4, wb5 = _e4m3x4_scaled_to_bf16x2x2_sm120(pb2, sb)
            wb6, wb7 = _e4m3x4_scaled_to_bf16x2x2_sm120(pb3, sb)
            wa = (wa0, wa1, wa2, wa3, wa4, wa5, wa6, wa7)
            wb = (wb0, wb1, wb2, wb3, wb4, wb5, wb6, wb7)
            for mt in cutlass.range_constexpr(self.max_m_tiles):
                token = cutlass.min(Int32(mt * 8) + group, rows - 1)
                px = x_base + Int64(token) * (x_stride_elems * Int64(2)) + x_off
                x0 = Uint32(0)
                x1 = Uint32(0)
                x2 = Uint32(0)
                x3 = Uint32(0)
                x4 = Uint32(0)
                x5 = Uint32(0)
                x6 = Uint32(0)
                x7 = Uint32(0)
                if cutlass.const_expr(self.in_features == self.k) or k_col < Int32(
                    self.in_features
                ):
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
            if warp == 0:
                for mt in cutlass.range_constexpr(self.max_m_tiles):
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
                        if feature < self.n and row < rows:
                            out[Int64(row) * Int64(self.n) + Int64(feature)] = (
                                total.to(BFloat16)
                            )
        else:
            tile_elems = self.max_m_tiles * 128
            scratch_base = Int64(scratch.toint())
            if warp == 0:
                sbase = (Int64(n_tile) * self.split_k + Int64(k_split)) * tile_elems
                for mt in cutlass.range_constexpr(self.max_m_tiles):
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
                        local_row = Int32(mt * 8) + quad * 2 + (e % 2)
                        local_col = group + (8 if e >= 2 else 0)
                        elem_idx = sbase + Int64(local_row * 16 + local_col)
                        _st_global_wb_f32(scratch_base + elem_idx * Int64(4), total)
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
                elem = Int32(tid)
                tbase = Int64(n_tile) * self.split_k * tile_elems
                while elem < Int32(tile_elems):
                    total = Float32(0.0)
                    for sp in cutlass.range_constexpr(self.split_k):
                        elem_idx = tbase + Int64(sp) * tile_elems + Int64(elem)
                        total = total + _ld_global_cg_f32(
                            scratch_base + elem_idx * Int64(4)
                        )
                    local_row = elem // 16
                    local_col = elem % 16
                    feature = first_n + local_col
                    if feature < self.n and local_row < rows:
                        out[Int64(local_row) * Int64(self.n) + Int64(feature)] = (
                            total.to(BFloat16)
                        )
                    elem += Int32(self.warps * _LANES)
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
):
    max_m_tiles = (int(max_rows) + 7) // 8
    eff_warps = effective_mxfp8_tc_gemv_warps(padded_k, split_k, warps)
    kernel = MXFP8MultiRowTCGemvKernel(
        n, in_features, padded_k, max_m_tiles, split_k=split_k, warps=eff_warps
    )
    key = (int(n), int(in_features), int(padded_k), max_m_tiles, int(split_k), eff_warps)
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
                "gemm.blockscaled.mxfp8_tc_gemv", 3, key
            ),
        )

    def run(source, values, scale_bytes, output, scratch, flags, stream):
        sc_ptr = (
            make_ptr(Float32, scratch.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            if scratch is not None
            else make_ptr(Float32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        )
        fl_ptr = (
            make_ptr(Uint32, flags.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
            if flags is not None
            else make_ptr(Uint32, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        )
        run_compiled(
            raw,
            (
                make_ptr(BFloat16, source.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(Uint8, values.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(Uint8, scale_bytes.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(BFloat16, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
                sc_ptr,
                fl_ptr,
                Int32(source.shape[0]),
                Int64(source.stride(0)),
                cuda_stream_from_int_or_current(stream),
            ),
        )

    return attach_programs(run, raw)
