"""Fused paged MXFP4 tile arithmetic, quantization, and input validation.

Selection requires every query head contributing to the score on each rank.
Hierarchical block selection retains its mandatory newest scored block.
"""

from __future__ import annotations

from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from ..._lib.compiler import KernelCompileSpec, compile as b12x_compile
from ..._lib.program_cache import program_cache
from b12x._lib.intrinsics.conversion import (
    cvt_fp32x2_to_e2m1x2,
    f16x2_to_f32x2,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.packed import f32_to_raw_bits
from b12x._lib.intrinsics.memory import (
    get_ptr_as_int64,
    ld_global_nc_u32,
    ld_shared_v4_u32,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.atomics import (
    atomic_add_shared_i32,
    ld_shared_i32,
    ld_shared_u32,
    red_add_shared_i32,
    st_shared_i32,
)
from ..._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

MXFP4_INDEX_PAGE_SIZE = 64
MXFP4_INDEX_PAGE_BYTES = 64 * 68


def index_mxfp4_page_bytes(page_size: int = MXFP4_INDEX_PAGE_SIZE) -> int:
    """Page storage: ``[page_size,64]`` data, then ``[page_size,4]`` UE8M0.

    Adjacent E2M1 values occupy low/high nibbles. Every data row is 16-byte
    aligned; scales are not interleaved into a 68-byte token stride.
    """
    if page_size <= 0 or page_size % 8:
        raise ValueError("MXFP4 page_size must be a positive multiple of eight")
    return page_size * 68


def _ptr(tensor, dtype):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=1)


@cute.jit
def _flat(ptr: cute.Pointer):
    # All actual accesses are bounded by runtime extents. The large logical
    # extent avoids narrowing pointer arithmetic for high recycled pool pages.
    return cute.make_tensor(ptr, cute.make_layout((Int64(1) << Int64(40),)))


class _Quantize:
    def __init__(self, paged: bool, page_size: int):
        self.paged = paged
        self.page_size = page_size

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        q: cute.Pointer,
        scales: cute.Pointer,
        slots: cute.Pointer,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(x),
            _flat(q),
            _flat(scales),
            _flat(slots),
            rows,
            pool_pages,
            page_stride,
        ).launch(grid=((rows * 4 + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        q: cute.Tensor,
        scales: cute.Tensor,
        slots: cute.Tensor,
        rows: Int32,
        pool_pages: Int64,
        page_stride: Int64,
    ):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        group = Int32(bx) * Int32(8) + Int32(tx) // Int32(32)
        row = group // Int32(4)
        lane = Int32(tx) % Int32(32)
        if row < rows:
            v = Float32(x[Int64(group) * Int64(32) + Int64(lane)])
            amax = fabs_f32(v)
            for shift in cutlass.range_constexpr(5):
                amax = fmax_f32(
                    amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                )
            amax = fmax_f32(amax, Float32(6.0 * 2.0**-126))
            _, sf = pow2_ceil_ue8m0(amax * Float32(1.0 / 6.0))
            scaled = v * ue8m0_to_output_scale(sf)
            other = cute.arch.shuffle_sync_bfly(scaled, offset=1)
            packed = cvt_fp32x2_to_e2m1x2(scaled, other)
            data_base = Int64(row) * Int64(64)
            scale_base = Int64(row) * Int64(4)
            valid = True
            if cutlass.const_expr(self.paged):
                slot = Int64(slots[row])
                page = slot // Int64(self.page_size)
                token = slot % Int64(self.page_size)
                valid = slot >= Int64(0) and page < pool_pages
                data_base = page * page_stride + token * Int64(64)
                scale_base = (
                    page * page_stride + Int64(self.page_size * 64) + token * Int64(4)
                )
            if valid:
                if lane % Int32(2) == Int32(0):
                    q[
                        data_base
                        + Int64(group % Int32(4)) * Int64(16)
                        + Int64(lane // Int32(2))
                    ] = Uint8(packed)
                if lane == Int32(0):
                    scales[scale_base + Int64(group % Int32(4))] = Uint8(sf)


@dsl_user_op
def _mxfp4_mma(
    d0,
    d1,
    d2,
    d3,
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    sfa,
    sfb,
    *,
    scale_byte_id=0,
    loc=None,
    ip=None,
):
    """Packed E2M1 m16n8k64 with two UE8M0 group scales per operand row."""
    operands = [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
    for scale in (sfa, sfb):
        operands.extend(
            (
                Uint32(scale).ir_value(loc=loc, ip=ip),
                Uint16(scale_byte_id).ir_value(loc=loc, ip=ip),
                Uint16(0).ir_value(loc=loc, ip=ip),
            )
        )
    operands.extend(Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1, d2, d3))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        operands,
        """
        mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0
        {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$0,$1,$2,$3},
        {$10}, {$11,$12}, {$13}, {$14,$15};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@cute.jit
def _packed_word(data: cute.Tensor, offset: Int64):
    return ld_global_nc_u32(get_ptr_as_int64(data, offset))


@cute.jit
def _scale_pair(data: cute.Tensor, offset: Int64):
    return Uint32(data[offset]) | (Uint32(data[offset + Int64(1)]) << Uint32(8))


@cute.jit
def _fused_qword(data: cute.Tensor, offset: Int64, aligned_q: cutlass.Constexpr[bool]):
    if cutlass.const_expr(aligned_q):
        return _packed_word(data, offset)
    return _scale_pair(data, offset) | (
        _scale_pair(data, offset + Int64(2)) << Uint32(16)
    )


class _TensorCorePagedScore:
    """Packed MXFP4 tile arithmetic with distinct BF16 rounding boundaries."""

    def __init__(self, heads: int, candidates: bool, page_size: int):
        self.heads = heads
        self.candidates = candidates
        self.page_size = page_size

    @cute.jit
    def _load_weight_pairs(self, weights: cute.Tensor, row: Int32):
        tx, _, _ = cute.arch.thread_idx()
        lane = Int32(tx) % Int32(32)
        packed = Uint32(0)
        if lane < Int32((self.heads + 1) // 2):
            w0 = weights[row, lane * Int32(2)]
            w1 = BFloat16(0)
            if cutlass.const_expr(self.heads != 1):
                w1 = weights[row, lane * Int32(2) + Int32(1)]
            packed = _fused_bf16_pack_weights(w0, w1)
        return packed

    @cute.jit
    def _score_tile_fused(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        tile_start: Int32,
        row_index: Int32 | None = None,
        local_output: cutlass.Constexpr[bool] = False,
        output_base: Int32 = 0,
        logical_indices: cute.Tensor | None = None,
        hardware_scale_select: cutlass.Constexpr[bool] = False,
        weight_pairs: Uint32 | None = None,
        aligned_q: cutlass.Constexpr[bool] = False,
        output_width: cutlass.Constexpr[int] = 0,
        carry_query_offsets: cutlass.Constexpr[bool] = False,
        logical_query_index: cutlass.Constexpr[bool] = False,
        unsigned_candidate_pages: cutlass.Constexpr[bool] = False,
    ):
        tx, _, _ = cute.arch.thread_idx()
        _, row, _ = cute.arch.block_idx()
        if cutlass.const_expr(row_index is not None):
            row = row_index
        lane = Int32(tx) % Int32(32)
        store_width = Int32(output_width) if output_width > 0 else width
        col = Int64(tile_start) + Int64(tx // 32) * Int64(16) + Int64(lane // 4)
        if cutlass.const_expr(self.candidates):
            kb = cute.make_rmem_tensor((2,), Int64)
            valid_key = cute.make_rmem_tensor((2,), cutlass.Boolean)
            totals = cute.make_rmem_tensor((2,), Float32)
            totals.fill(0.0)
            column = col + Int64(lane % Int32(2)) * Int64(8)
            # Candidate IDs are signed 32-bit; storage offsets remain 64-bit.
            candidate_pos = Int32(-1)
            valid = column < width
            if cutlass.const_expr(self.candidates):
                valid = valid and column < candidate_lengths[row]
                if valid:
                    candidate_pos = Int32(
                        candidates[Int64(row) * Int64(width) + column]
                    )
            valid = (
                valid
                and candidate_pos >= Int32(0)
                and candidate_pos < lengths[row]
                and candidate_pos < active[0]
            )
            if cutlass.const_expr(unsigned_candidate_pages):
                candidate_page = Uint32(candidate_pos) // Uint32(self.page_size)
                valid = valid and candidate_page < Uint32(page_width)
            else:
                valid = valid and candidate_pos // Int32(self.page_size) < page_width
            page = Int64(-1)
            if valid:
                if cutlass.const_expr(unsigned_candidate_pages):
                    page = Int64(
                        pages[Int64(row) * page_row_stride + Int64(candidate_page)]
                    )
                else:
                    page = Int64(
                        pages[
                            Int64(row) * page_row_stride
                            + Int64(candidate_pos // Int32(self.page_size))
                        ]
                    )
            valid = valid and page >= Int64(0) and page < pool_pages
            if cutlass.const_expr(unsigned_candidate_pages):
                token = Int64(Uint32(candidate_pos) % Uint32(self.page_size))
            else:
                token = Int64(candidate_pos % Int32(self.page_size))
            selected_key_base = page * pool_stride + token * Int64(64)
            selected_scale_base = (
                page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
            )
            selected_scale_valid = valid
            # Selection starts after a CTA barrier, so IDs can precede their scores.
            if cutlass.const_expr(logical_indices is not None):
                if lane % Int32(4) < Int32(2) and column < width:
                    index_offset = Int64(row) * Int64(store_width) + column
                    if cutlass.const_expr(local_output):
                        index_offset = Int64(output_base) + column - Int64(tile_start)
                    logical_indices[index_offset] = (
                        Int32(candidate_pos) if valid else Int32(-1)
                    )
            # Each four-lane group shares both complete 64-bit key bases.
            for part in cutlass.range_constexpr(2):
                address_source_lane = lane // Int32(4) * Int32(4) + Int32(part)
                kb[part] = cute.arch.shuffle_sync(
                    selected_key_base, address_source_lane
                )
                valid_key[part] = cute.arch.shuffle_sync(
                    Int32(valid), address_source_lane
                ) != Int32(0)
        else:
            kb = cute.make_rmem_tensor((2,), Int64)
            sb = cute.make_rmem_tensor((2,), Int64)
            valid_key = cute.make_rmem_tensor((2,), cutlass.Boolean)
            totals = cute.make_rmem_tensor((2,), Float32)
            totals.fill(0.0)
            for part in cutlass.range_constexpr(2):
                column = col + Int32(part * 8)
                pos = column
                valid = column < width
                if cutlass.const_expr(self.candidates):
                    valid = valid and column < candidate_lengths[row]
                    if valid:
                        pos = Int64(
                            candidates[Int64(row) * Int64(width) + Int64(column)]
                        )
                valid = (
                    valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
                )
                valid = valid and pos // Int32(self.page_size) < page_width
                page = Int64(-1)
                if valid:
                    page = Int64(
                        pages[
                            Int64(row) * page_row_stride
                            + Int64(pos // Int32(self.page_size))
                        ]
                    )
                valid = valid and page >= Int64(0) and page < pool_pages
                token = Int64(pos % Int32(self.page_size))
                valid_key[part] = valid
                kb[part] = page * pool_stride + token * Int64(64)
                sb[part] = (
                    page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
                )
                # Selection starts after a CTA barrier, so IDs can precede their scores.
                if cutlass.const_expr(logical_indices is not None):
                    if lane % Int32(4) == Int32(0) and column < width:
                        index_offset = Int64(row) * Int64(store_width) + Int64(column)
                        if cutlass.const_expr(local_output):
                            index_offset = Int64(output_base) + Int64(
                                column - tile_start
                            )
                        logical_indices[index_offset] = (
                            Int32(pos) if valid else Int32(-1)
                        )
            selected_scale_base = sb[0]
            selected_scale_valid = valid_key[0]
            if lane % Int32(2) != Int32(0):
                selected_scale_base = sb[1]
                selected_scale_valid = valid_key[1]
        query_head_index = Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))
        qb = (Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))) * Int64(64)
        qsb = (Int64(row) * Int64(self.heads) + Int64(lane // Int32(4))) * Int64(4)
        for head_tile in cutlass.range((self.heads + 7) // 8, unroll=1):
            head = Int32(head_tile * 8) + lane // Int32(4)
            if cutlass.const_expr(not carry_query_offsets):
                qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
                qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
            elif cutlass.const_expr(logical_query_index):
                qb = query_head_index * Int64(64)
                qsb = query_head_index * Int64(4)
            d0, d1, d2, d3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for kstep in cutlass.range_constexpr(2):
                byte_col = Int64(kstep * 32) + Int64(lane % Int32(4)) * Int64(4)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if valid_key[0]:
                    a0 = _packed_word(pool, kb[0] + byte_col)
                    a2 = _packed_word(pool, kb[0] + byte_col + Int64(16))
                if valid_key[1]:
                    a1 = _packed_word(pool, kb[1] + byte_col)
                    a3 = _packed_word(pool, kb[1] + byte_col + Int64(16))
                b0, b1 = Uint32(0), Uint32(0)
                scale_default = 0x7F7F7F7F if hardware_scale_select else 0x7F7F
                sfa, sfb = Uint32(scale_default), Uint32(scale_default)
                if selected_scale_valid:
                    sfa = _packed_word(pool, selected_scale_base)
                    if cutlass.const_expr(not hardware_scale_select):
                        sfa = (sfa >> Uint32(kstep * 16)) & Uint32(65535)
                if head < Int32(self.heads):
                    b0 = _fused_qword(q, qb + byte_col, aligned_q)
                    b1 = _fused_qword(q, qb + byte_col + Int64(16), aligned_q)
                    sfb = _fused_qword(qs, qsb, aligned_q)
                    if cutlass.const_expr(not hardware_scale_select):
                        sfb = (sfb >> Uint32(kstep * 16)) & Uint32(65535)
                d0, d1, d2, d3 = _mxfp4_mma(
                    d0,
                    d1,
                    d2,
                    d3,
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    sfa,
                    sfb,
                    scale_byte_id=kstep * 2 if hardware_scale_select else 0,
                )
            dots = cute.make_rmem_tensor((4,), Float32)
            dots[0], dots[1], dots[2], dots[3] = d0, d1, d2, d3
            packed_products = cute.make_rmem_tensor((2,), Uint32)
            if cutlass.const_expr(weight_pairs is None):
                output_head = Int32(head_tile * 8) + (lane % Int32(4)) * Int32(2)
                w0, w1 = BFloat16(0), BFloat16(0)
                if cutlass.const_expr(self.heads >= 8):
                    w0 = weights[Int64(row) * Int64(self.heads) + Int64(output_head)]
                    w1 = weights[
                        Int64(row) * Int64(self.heads) + Int64(output_head + Int32(1))
                    ]
                else:
                    if output_head < Int32(self.heads):
                        w0 = weights[
                            Int64(row) * Int64(self.heads) + Int64(output_head)
                        ]
                    if output_head + Int32(1) < Int32(self.heads):
                        w1 = weights[
                            Int64(row) * Int64(self.heads)
                            + Int64(output_head + Int32(1))
                        ]
                for pair in cutlass.range_constexpr(2):
                    packed_products[pair] = _fused_bf16_products_packed(
                        dots[pair * 2],
                        dots[pair * 2 + 1],
                        w0,
                        w1,
                        fused_relu=not self.candidates,
                    )
            else:
                weight_word = cute.arch.shuffle_sync(
                    weight_pairs, Int32(head_tile * 4) + lane % Int32(4)
                )
                for pair in cutlass.range_constexpr(2):
                    packed_products[pair] = _fused_bf16_products_word(
                        dots[pair * 2],
                        dots[pair * 2 + 1],
                        weight_word,
                        fused_relu=not self.candidates,
                    )
            for head_pair in cutlass.range_constexpr(min(4, (self.heads + 1) // 2)):
                source_lane = lane // Int32(4) * Int32(4) + Int32(head_pair)
                for pair in cutlass.range_constexpr(2):
                    packed = cute.arch.shuffle_sync(packed_products[pair], source_lane)
                    if cutlass.const_expr(self.heads == 1):
                        totals[pair] += u32_as_f32(packed << Uint32(16))
                    else:
                        totals[pair] = _fused_bf16_pair_accumulate(totals[pair], packed)
            if cutlass.const_expr(carry_query_offsets):
                if cutlass.const_expr(not logical_query_index):
                    qb += Int64(8 * 64)
                    qsb += Int64(8 * 4)
                else:
                    query_head_index += Int64(8)
        if lane % Int32(4) == Int32(0):
            for part in cutlass.range_constexpr(2):
                column = col + Int32(part * 8)
                if column < width:
                    value = BFloat16(-float("inf"))
                    if valid_key[part]:
                        value = BFloat16(totals[part])
                    offset = Int64(row) * Int64(store_width) + Int64(column)
                    if cutlass.const_expr(local_output):
                        offset = Int64(output_base) + Int64(column - tile_start)
                    scores[offset] = value

    @cute.jit
    def _score_tiles_direct_register(
        self,
        q: cute.Tensor,
        qs: cute.Tensor,
        weights: cute.Tensor,
        pool: cute.Tensor,
        pages: cute.Tensor,
        lengths: cute.Tensor,
        active: cute.Tensor,
        candidates: cute.Tensor,
        candidate_lengths: cute.Tensor,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        tile_start: Int32,
        row_index: Int32,
        aligned_q: cutlass.Constexpr[bool] = False,
    ):
        tx, _, _ = cute.arch.thread_idx()
        row = row_index
        lane = Int32(tx) % Int32(32)
        col = Int64(tile_start) + Int64(tx // 32) * Int64(32) + Int64(lane // 4)
        kb = cute.make_rmem_tensor((4,), Int64)
        sb = cute.make_rmem_tensor((4,), Int64)
        valid_key = cute.make_rmem_tensor((4,), cutlass.Boolean)
        logical_ids = cute.make_rmem_tensor((4,), Int32)
        totals = cute.make_rmem_tensor((4,), Float32)
        totals.fill(0.0)
        for part in cutlass.range_constexpr(4):
            column = col + Int32((part % 2) * 8 + (part // 2) * 16)
            pos = column
            valid = column < width
            if cutlass.const_expr(self.candidates):
                valid = valid and column < candidate_lengths[row]
                if valid:
                    pos = Int64(candidates[Int64(row) * Int64(width) + Int64(column)])
            valid = valid and pos >= Int32(0) and pos < lengths[row] and pos < active[0]
            valid = valid and pos // Int32(self.page_size) < page_width
            page = Int64(-1)
            if valid:
                page = Int64(
                    pages[
                        Int64(row) * page_row_stride
                        + Int64(pos // Int32(self.page_size))
                    ]
                )
            valid = valid and page >= Int64(0) and page < pool_pages
            token = Int64(pos % Int32(self.page_size))
            valid_key[part] = valid
            logical_ids[part] = Int32(pos)
            kb[part] = page * pool_stride + token * Int64(64)
            sb[part] = (
                page * pool_stride + Int64(self.page_size * 64) + token * Int64(4)
            )
        for head_tile in cutlass.range((self.heads + 7) // 8):
            head = Int32(head_tile * 8) + lane // Int32(4)
            qb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(64)
            qsb = (Int64(row) * Int64(self.heads) + Int64(head)) * Int64(4)
            dots = cute.make_rmem_tensor((8,), Float32)
            dots.fill(0.0)
            for kstep in cutlass.range_constexpr(2):
                byte_col = Int64(kstep * 32) + Int64(lane % Int32(4)) * Int64(4)
                b0, b1 = Uint32(0), Uint32(0)
                sfb = Uint32(0x7F7F)
                if head < Int32(self.heads):
                    b0 = _fused_qword(q, qb + byte_col, aligned_q)
                    b1 = _fused_qword(q, qb + byte_col + Int64(16), aligned_q)
                    sfb = (
                        _fused_qword(qs, qsb, aligned_q) >> Uint32(kstep * 16)
                    ) & Uint32(65535)
                for tile in cutlass.range_constexpr(2):
                    pair = tile * 2
                    a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    if valid_key[pair]:
                        a0 = _packed_word(pool, kb[pair] + byte_col)
                        a2 = _packed_word(pool, kb[pair] + byte_col + Int64(16))
                    if valid_key[pair + 1]:
                        a1 = _packed_word(pool, kb[pair + 1] + byte_col)
                        a3 = _packed_word(pool, kb[pair + 1] + byte_col + Int64(16))
                    scale_first = lane % Int32(2) == Int32(0)
                    scale_valid = (
                        valid_key[pair] if scale_first else valid_key[pair + 1]
                    )
                    scale_offset = sb[pair] if scale_first else sb[pair + 1]
                    sfa = Uint32(0x7F7F)
                    if scale_valid:
                        sfa = (
                            _packed_word(pool, scale_offset) >> Uint32(kstep * 16)
                        ) & Uint32(65535)
                    dbase = tile * 4
                    dots[dbase], dots[dbase + 1], dots[dbase + 2], dots[dbase + 3] = (
                        _mxfp4_mma(
                            dots[dbase],
                            dots[dbase + 1],
                            dots[dbase + 2],
                            dots[dbase + 3],
                            a0,
                            a1,
                            a2,
                            a3,
                            b0,
                            b1,
                            sfa,
                            sfb,
                        )
                    )
            contributions = cute.make_rmem_tensor((8,), Float32)
            for part in cutlass.range_constexpr(4):
                output_head = (
                    Int32(head_tile * 8)
                    + (lane % Int32(4)) * Int32(2)
                    + Int32(part % 2)
                )
                weight = Float32(0)
                if output_head < Int32(self.heads):
                    weight = Float32(
                        weights[Int64(row) * Int64(self.heads) + Int64(output_head)]
                    )
                for tile in cutlass.range_constexpr(2):
                    dot = fmax_f32(Float32(BFloat16(dots[tile * 4 + part])), Float32(0))
                    contributions[tile * 4 + part] = Float32(BFloat16(dot * weight))
            packed_products = cute.make_rmem_tensor((4,), Uint32)
            for pair in cutlass.range_constexpr(4):
                packed_products[pair] = _fused_pack_bf16_pair(
                    BFloat16(contributions[pair * 2]),
                    BFloat16(contributions[pair * 2 + 1]),
                )
            for head_pair in cutlass.range_constexpr(min(4, (self.heads + 1) // 2)):
                source_lane = lane // Int32(4) * Int32(4) + Int32(head_pair)
                for pair in cutlass.range_constexpr(4):
                    packed = cute.arch.shuffle_sync(packed_products[pair], source_lane)
                    if cutlass.const_expr(self.heads == 1):
                        totals[pair] += u32_as_f32(packed << Uint32(16))
                    else:
                        totals[pair] = _fused_bf16_pair_accumulate(totals[pair], packed)
        value0, value1, value2, value3 = (
            Float32(-float("inf")),
            Float32(-float("inf")),
            Float32(-float("inf")),
            Float32(-float("inf")),
        )
        index0, index1, index2, index3 = Int32(-1), Int32(-1), Int32(-1), Int32(-1)
        if lane % Int32(4) == Int32(0):
            if valid_key[0]:
                value0, index0 = Float32(BFloat16(totals[0])), Int32(logical_ids[0])
            if valid_key[1]:
                value1, index1 = Float32(BFloat16(totals[1])), Int32(logical_ids[1])
            if valid_key[2]:
                value2, index2 = Float32(BFloat16(totals[2])), Int32(logical_ids[2])
            if valid_key[3]:
                value3, index3 = Float32(BFloat16(totals[3])), Int32(logical_ids[3])
        return value0, value1, value2, value3, index0, index1, index2, index3


@dataclass(frozen=True)
class _CompiledLauncher:
    raw: object
    dtypes: tuple

    @property
    def __b12x_dependencies__(self):
        return (self.raw,)

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys(self.raw)


@program_cache
def _compile(kind: str, recipe: tuple, device_index: int):
    if kind == "sort":
        from ._position_sort import _sort_positions

        topk, expand = recipe
        raise_if_kernel_resolution_frozen(
            "triton.compile", target=_sort_positions, cache_key=(kind, recipe)
        )
        with torch.cuda.device(device_index):
            raw = _sort_positions.warmup(
                torch.int32,
                torch.float32,
                torch.int32,
                torch.float32,
                torch.int32,
                torch.int32,
                torch.int32,
                K=topk,
                EXPAND=expand,
                num_warps=8 if expand else 4,
                grid=(1, 1, 1),
            )
        return _CompiledLauncher(raw, ())
    # Pointer-only launch ABIs: all live row/page/width/stride quantities are
    # runtime scalars and do not contribute to compile identity.
    if kind == "quantize":
        obj = _Quantize(*recipe)
        dtypes = (BFloat16, Uint8, Uint8, Int64)
        scalars = (Int32(1), Int64(1), Int64(4352))
    else:
        raise ValueError(f"unknown MXFP4 indexer kernel {kind}")
    key = (kind, recipe)
    raise_if_kernel_resolution_frozen("cute.compile", target=obj, cache_key=key)
    pointers = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=1) for t in dtypes
    )
    raw = b12x_compile(
        obj,
        *pointers,
        *scalars,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("attention.indexer.mxfp4", 9, key),
    )
    return _CompiledLauncher(raw, dtypes)


def _launch(kind, recipe, tensors, scalars, *, launcher=None):
    if tensors[0].device.type != "cuda":
        raise ValueError("native MXFP4 indexer requires CUDA tensors")
    with torch.cuda.device(tensors[0].device):
        if launcher is None:
            launcher = _compile(kind, recipe, tensors[0].device.index)
        if kind == "sort":
            launcher.raw[(scalars[0], 1, 1)](*tensors, *recipe)
            return
        launcher.raw(
            *(
                _ptr(t, dtype)
                for t, dtype in zip(tensors, launcher.dtypes, strict=True)
            ),
            *scalars,
            current_cuda_stream(),
        )


class _FusedStateReset:
    def __init__(
        self,
        groups: int,
        row_words: int,
        coarse1: int,
        arrival: int,
        contiguous: bool = False,
    ):
        self.groups = int(groups)
        self.contiguous = bool(contiguous)
        self.words = groups * row_words if self.contiguous else groups * 520
        self.row_words = row_words
        self.coarse1 = coarse1
        self.arrival = arrival

    @cute.jit
    def __call__(self, state: cute.Tensor, live_rows: Int32, stream: cuda.CUstream):
        self.kernel(state, live_rows).launch(
            grid=((self.words + 1023) // 1024, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, state: cute.Tensor, live_rows: Int32):
        cute.arch.griddepcontrol_launch_dependents()
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        word = Int32(bx) * Int32(1024) + Int32(tx) * Int32(4)
        if cutlass.const_expr(self.contiguous):
            for j in cutlass.range_constexpr(4):
                if word + Int32(j) < Int32(self.words):
                    state[Int64(word + Int32(j))] = Int32(0)
        else:
            if (word < Int32(self.words)) & (word // Int32(520) < live_rows):
                group = word // Int32(520)
                slot = word - group * Int32(520)
                physical = slot
                if slot >= Int32(512):
                    physical = slot - Int32(512) + Int32(self.arrival)
                elif slot >= Int32(256):
                    physical = slot - Int32(256) + Int32(self.coarse1)
                offset = Int64(group) * Int64(self.row_words) + Int64(physical)
                st_global_v4_u32(
                    get_ptr_as_int64(state, offset),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
            inactive = Int64(live_rows) * Int64(self.row_words) + Int64(word)
            while inactive < Int64(self.groups * self.row_words):
                st_global_v4_u32(
                    get_ptr_as_int64(state, inactive),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                    Uint32(0),
                )
                inactive += Int64(((self.words + 1023) // 1024) * 1024)


@dataclass(frozen=True)
class _PreparedFusedMXFP4:
    launcher: object
    args: tuple
    scratch: tuple
    owners: tuple
    rows: int
    reset_launcher: object
    reset_args: tuple

    @property
    def __b12x_dependencies__(self):
        return (
            (self.launcher, self.reset_launcher)
            if self.reset_launcher is not None
            else (self.launcher,)
        )

    @property
    def __b12x_programs__(self):
        from b12x._lib.compile_plan import program_keys

        return program_keys(self.__b12x_dependencies__)

    def __call__(self):
        from b12x._lib.compiler import run_compiled

        with torch.cuda.device(self.scratch[-1].device):
            stream = current_cuda_stream()
            if self.reset_launcher is not None:
                run_compiled(
                    self.reset_launcher, (self.reset_args[0], Int32(self.rows), stream)
                )
            args = (*self.args[:13], stream, self.args[14])
            run_compiled(self.launcher, args)


def prepare_fused_mxfp4(
    *,
    q_mxfp4,
    q_scales,
    query_weights,
    index_k_cache,
    page_table,
    cache_lengths,
    active_width,
    output_indices,
    output_scores=None,
    candidate_indices=None,
    candidate_lengths=None,
    page_size=64,
    ctas_per_row=None,
    _vector_output_pairs=None,
):
    """Prepare streaming top-512 selection with complete per-rank head scores.

    Queries and keys use packed E2M1 with four UE8M0 scales per head. Query
    data and scale tensors require four-byte-aligned pointers. Inputs
    remain mutable at fixed addresses. The returned callable resets its fixed
    merge arena, scores and selects without a context-sized score tensor, and
    sorts selected logical positions. Hierarchical block production is separate.
    """
    from .fused_indexer import (
        DSAFusedIndexerKernel,
        _check_mxfp4_cooperative_residency,
        KV_LAYOUT_PAGED,
        _COOP_STATE_WORDS,
        _FUSED_COARSE1,
        _FUSED_STATE_ARRIVAL,
    )
    from .kernel import _to_kernel_tensor

    raise_if_kernel_resolution_frozen(
        "fused MXFP4 preparation", target=prepare_fused_mxfp4
    )
    device = q_mxfp4.device
    if device.type != "cuda" or q_mxfp4.ndim != 3:
        raise ValueError("q_mxfp4 must be a CUDA [rows, heads, 64] tensor")
    rows, heads, packed = map(int, q_mxfp4.shape)
    if rows <= 0 or rows >= 2**31 or heads != 32 or packed != 64:
        raise ValueError(
            "fused MXFP4 requires positive rows and 32 complete index heads"
        )
    page_bytes = index_mxfp4_page_bytes(page_size)

    def checked(tensor, name, shape, dtype, *, contiguous=True, align=1):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != tuple(shape)
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            raise ValueError(f"{name} must be {dtype} {tuple(shape)} on {device}")
        if contiguous and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.data_ptr() % align:
            raise ValueError(f"{name} requires {align}-byte alignment")
        return tensor

    checked(q_mxfp4, "q_mxfp4", (rows, heads, 64), torch.uint8, align=4)
    checked(q_scales, "q_scales", (rows, heads, 4), torch.uint8, align=4)
    checked(query_weights, "query_weights", (rows, heads), torch.bfloat16, align=2)
    checked(cache_lengths, "cache_lengths", (rows,), torch.int32, align=4)
    checked(active_width, "active_width", (1,), torch.int32, align=4)
    checked(output_indices, "output_indices", (rows, 512), torch.int32, align=4)
    if output_scores is None:
        output_scores = torch.empty((rows, 512), device=device, dtype=torch.float32)
    checked(output_scores, "output_scores", (rows, 512), torch.float32, align=4)
    if (
        index_k_cache.ndim != 2
        or index_k_cache.shape[1] < page_bytes
        or index_k_cache.stride(1) != 1
        or index_k_cache.stride(0) < page_bytes
        or index_k_cache.stride(0) % 4
    ):
        raise ValueError("index_k_cache must contain aligned packed MXFP4 pages")
    checked(
        index_k_cache,
        "index_k_cache",
        index_k_cache.shape,
        torch.uint8,
        contiguous=False,
        align=4,
    )
    if (
        page_table.ndim != 2
        or page_table.shape[0] not in (1, rows)
        or page_table.stride(1) != 1
        or page_table.stride(0) < 0
    ):
        raise ValueError("page_table must have one shared row or one row per query")
    checked(
        page_table,
        "page_table",
        page_table.shape,
        torch.int32,
        contiguous=False,
        align=4,
    )
    if page_table.shape[0] == 1:
        page_table = page_table.expand(rows, -1)
    max_width = int(page_table.shape[1]) * page_size
    candidates = candidate_indices is not None
    if candidates:
        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != rows:
            raise ValueError("candidate_indices must have one row per query")
        checked(
            candidate_indices,
            "candidate_indices",
            candidate_indices.shape,
            torch.int32,
            align=4,
        )
        checked(candidate_lengths, "candidate_lengths", (rows,), torch.int32, align=4)
        width = int(candidate_indices.shape[1])
    else:
        if candidate_lengths is not None:
            raise ValueError("candidate_lengths requires candidate_indices")
        width = max_width
        candidate_indices = output_indices
        candidate_lengths = cache_lengths
    if not 0 < width < 2**31 or max_width >= 2**31:
        raise ValueError("logical positions must fit int32")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if ctas_per_row is None:
        ctas_per_row = max(1, min((width + 511) // 512, sms // rows))
    ctas_per_row = int(ctas_per_row)
    if ctas_per_row < 1:
        raise ValueError("fused CTAs per row must be positive")
    merge_threshold = width if rows * ctas_per_row > sms else 0
    raw_indices = output_indices
    raw_values = output_scores
    vector_pairs_eligible = (
        max_width <= 65536
        and raw_indices.data_ptr() % 8 == 0
        and raw_values.data_ptr() % 8 == 0
    )
    vector_output_pairs = (
        False if _vector_output_pairs is None else bool(_vector_output_pairs)
    )
    if vector_output_pairs and not vector_pairs_eligible:
        raise ValueError(
            "paired MXFP4 output stores require position capacity <=65536 "
            "and eight-byte-aligned output buffers"
        )
    kernel = DSAFusedIndexerKernel(
        num_heads_static=heads,
        topk=512,
        kv_layout=KV_LAYOUT_PAGED,
        ctas_per_group=ctas_per_row,
        num_sms=sms,
        merge_threshold=merge_threshold,
        max_seq_capacity=width,
        k_quant_page_stride=int(index_k_cache.stride(0)),
        mxfp4=True,
        mxfp4_aligned_q=True,
        candidates=candidates,
        page_size=page_size,
        score_width=width,
        ordered_output=True,
        output_position_capacity=max_width,
        state_reset_pdl=ctas_per_row > 1,
        vector_output_pairs=vector_output_pairs,
    )
    pack_elements, state_elements = kernel.merge_scratch_elements(rows)
    pack_v = torch.empty(pack_elements, dtype=torch.float32, device=device)
    pack_i = torch.empty(pack_elements, dtype=torch.int32, device=device)
    state = torch.zeros(state_elements, dtype=torch.int32, device=device)
    tensors = (
        q_mxfp4,
        query_weights,
        index_k_cache,
        q_scales,
        page_table,
        cache_lengths,
        cache_lengths,
        cache_lengths,
        raw_indices,
        raw_values,
        pack_v,
        pack_i,
        state,
    )
    dtypes = (
        Uint8,
        BFloat16,
        Uint8,
        Uint8,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Float32,
        Float32,
        Int32,
        Int32,
    )
    key = (
        "mxfp4",
        kernel.constructor_config,
        rows,
        heads,
        page_size,
        width,
        ctas_per_row,
        sms,
        candidates,
        max_width,
        vector_output_pairs,
        kernel.merge_threshold,
        kernel.mxfp4_merge_bits,
        kernel.mxfp4_histogram_count_bits,
        kernel.mxfp4_histogram_bitmap,
        kernel.mxfp4_serial_state_words,
        kernel.state_words_per_group,
        kernel.mxfp4_query_offset_mode,
        kernel.mxfp4_unsigned_candidate_pages,
        tuple(
            (tuple(t.shape), tuple(t.stride()), str(t.dtype))
            for t in (*tensors, active_width, candidate_indices)
        ),
    )
    with torch.cuda.device(device):
        args = []
        for t, d in zip(tensors, dtypes, strict=True):
            alignment = 1
            if t is raw_indices or t is raw_values:
                alignment = 8 if vector_output_pairs else 4
            elif t is pack_v or t is pack_i or t is state:
                alignment = 16
            args.append(_to_kernel_tensor(t, d, assumed_align=alignment))
        args = tuple(args)
        operands = tuple(
            _to_kernel_tensor(t, d, assumed_align=1)
            for t, d in (
                (q_scales, Uint8),
                (active_width, Int32),
                (candidate_indices, Int32),
                (candidate_lengths, Int32),
            )
        )
        operands = (*operands, Int32(width))
        args = (*args, current_cuda_stream(), operands)
        launcher = b12x_compile(
            kernel,
            *args,
            compile_spec=KernelCompileSpec.from_key(
                "attention.indexer.fused_indexer", 11, key
            ),
        )
        _check_mxfp4_cooperative_residency(kernel, launcher, num_rows=rows, num_sms=sms)
        reset_launcher = None
        reset_args = ()
        if kernel.merge_in_kernel:
            reset_args = (
                _to_kernel_tensor(state, Int32, assumed_align=16),
                Int32(rows),
            )
            reset_launcher = b12x_compile(
                _FusedStateReset(
                    rows,
                    kernel.state_words_per_group,
                    _FUSED_COARSE1,
                    _FUSED_STATE_ARRIVAL,
                    contiguous=(kernel.state_words_per_group != _COOP_STATE_WORDS),
                ),
                *reset_args,
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "attention.indexer.mxfp4_state_reset",
                    6,
                    (state.numel(), kernel.state_words_per_group),
                ),
            )
    owners = (
        *tensors,
        active_width,
        candidate_indices,
        candidate_lengths,
        output_indices,
        output_scores,
    )
    prepared = _PreparedFusedMXFP4(
        launcher,
        args,
        (raw_indices, raw_values, pack_v, pack_i, state),
        owners,
        rows,
        reset_launcher,
        reset_args,
    )
    prepared()
    return prepared


def _check(tensor, name, shape, dtype, device):
    if tensor is None or tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {shape}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on {device}")


def _quantize_q_mxfp4(
    query: torch.Tensor,
    *,
    q_mxfp4: torch.Tensor,
    q_scales: torch.Tensor,
    launcher,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the prepared query-quantizer launcher into caller-owned buffers."""
    if query.ndim < 2 or query.shape[-1] != 128:
        raise ValueError("query must have shape (...,128)")
    _check(query, "query", query.shape, torch.bfloat16, query.device)
    _check(q_mxfp4, "q_mxfp4", (*query.shape[:-1], 64), torch.uint8, query.device)
    _check(q_scales, "q_scales", (*query.shape[:-1], 4), torch.uint8, query.device)
    rows = query.numel() // 128
    if rows:
        _launch(
            "quantize",
            (False, 64),
            (query, q_mxfp4, q_scales, query),
            (rows, 0, 0),
            launcher=launcher,
        )
    return q_mxfp4, q_scales


def _quantize_write_index_k_mxfp4(
    keys: torch.Tensor,
    *,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_size: int = 64,
    launcher,
) -> torch.Tensor:
    """Run the prepared paged-key writer against caller-owned physical slots."""
    page_bytes = index_mxfp4_page_bytes(page_size)
    if keys.ndim != 2 or keys.shape[1] != 128:
        raise ValueError("keys must have shape (rows,128)")
    _check(keys, "keys", keys.shape, torch.bfloat16, keys.device)
    _check(slot_mapping, "slot_mapping", (keys.shape[0],), torch.int64, keys.device)
    _check_pool(index_k_cache, page_bytes, keys.device)
    if keys.shape[0]:
        _launch(
            "quantize",
            (True, page_size),
            (keys, index_k_cache, index_k_cache, slot_mapping),
            (keys.shape[0], index_k_cache.shape[0], index_k_cache.stride(0)),
            launcher=launcher,
        )
    return index_k_cache


def _check_pool(pool, page_bytes, device):
    if pool.ndim != 2 or pool.shape[1] != page_bytes or pool.dtype != torch.uint8:
        raise ValueError(f"index_k_cache must be uint8 (pages,{page_bytes})")
    if (
        pool.device != device
        or pool.stride(1) != 1
        or pool.stride(0) < page_bytes
        or pool.stride(0) % 16
        or pool.data_ptr() % 16
    ):
        raise ValueError(
            "index_k_cache must have aligned, nonoverlapping page storage on the plan device"
        )


def bind_mxfp4(
    plan,
    *,
    scratch,
    q_mxfp4,
    q_scales,
    query_weights,
    index_k_cache,
    page_table,
    cache_lengths,
    active_width,
    output_indices,
    output_scores,
    candidate_indices,
    candidate_lengths,
    candidate_output,
    candidate_output_lengths,
    score_width=None,
    request_ids=None,
    query_start_loc=None,
):
    caps = plan.caps
    width_capacity = caps.max_candidates or caps.max_page_table_width * caps.page_size
    if score_width is not None and (
        isinstance(score_width, bool) or not isinstance(score_width, int)
    ):
        raise TypeError("score_width must be an integer")
    score_width = width_capacity if score_width is None else score_width
    if not 0 < score_width <= width_capacity:
        raise ValueError("score_width must be positive and within planned capacity")
    if caps.max_candidates and score_width != caps.max_candidates:
        raise ValueError("candidate scoring retains its planned candidate width")
    if q_mxfp4 is None or q_mxfp4.ndim != 3:
        raise ValueError("MXFP4 recipe requires explicit q_mxfp4 and q_scales")
    rows = q_mxfp4.shape[0]
    if rows <= 0 or rows > caps.max_q_rows:
        raise ValueError("query rows must be positive and within planned capacity")
    _check(q_mxfp4, "q_mxfp4", (rows, caps.num_q_heads, 64), torch.uint8, caps.device)
    _check(q_scales, "q_scales", (rows, caps.num_q_heads, 4), torch.uint8, caps.device)
    if query_weights.ndim == 3 and query_weights.shape[-1] == 1:
        query_weights = query_weights.view(rows, caps.num_q_heads)
    _check(
        query_weights,
        "query_weights",
        (rows, caps.num_q_heads),
        torch.bfloat16,
        caps.device,
    )
    _check_pool(index_k_cache, index_mxfp4_page_bytes(caps.page_size), caps.device)
    if page_table.ndim != 2 or page_table.shape[1] > caps.max_page_table_width:
        raise ValueError("page_table width exceeds planned capacity")
    if (
        page_table.dtype != torch.int32
        or page_table.device != caps.device
        or page_table.stride(1) != 1
    ):
        raise ValueError(
            "page_table must be int32 with unit inner stride on plan device"
        )
    if request_ids is not None and query_start_loc is not None:
        raise ValueError("specify at most one of request_ids or query_start_loc")
    if request_ids is not None:
        _check(request_ids, "request_ids", (rows,), torch.int32, caps.device)
        if page_table.shape[0] < 1:
            raise ValueError("page_table must have at least one request row")
    elif query_start_loc is not None:
        if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
            raise ValueError("query_start_loc must have shape (batch + 1,)")
        _check(
            query_start_loc,
            "query_start_loc",
            query_start_loc.shape,
            torch.int32,
            caps.device,
        )
        if page_table.shape[0] < query_start_loc.shape[0] - 1:
            raise ValueError("page_table rows must cover query_start_loc batch size")
    elif page_table.shape[0] not in (1, rows):
        raise ValueError("page_table must have one shared row or one row per query")
    _check(cache_lengths, "cache_lengths", (rows,), torch.int32, caps.device)
    _check(active_width, "active_width", (1,), torch.int32, caps.device)
    _check(
        output_indices, "output_indices", (rows, caps.topk), torch.int32, caps.device
    )
    if output_scores is not None:
        _check(
            output_scores,
            "output_scores",
            (rows, caps.topk),
            torch.float32,
            caps.device,
        )
    if caps.max_candidates:
        _check(
            candidate_indices,
            "candidate_indices",
            (rows, caps.max_candidates),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_lengths, "candidate_lengths", (rows,), torch.int32, caps.device
        )
    elif candidate_indices is not None or candidate_lengths is not None:
        raise ValueError("candidate inputs require max_candidates in Caps")
    if caps.candidate_topk_blocks:
        _check(
            candidate_output,
            "candidate_output",
            (rows, caps.candidate_topk_blocks * 8),
            torch.int32,
            caps.device,
        )
        _check(
            candidate_output_lengths,
            "candidate_output_lengths",
            (rows,),
            torch.int32,
            caps.device,
        )
    elif candidate_output is not None or candidate_output_lengths is not None:
        raise ValueError("candidate output requires candidate_topk_blocks in Caps")
    runtime = plan.inner.bind(
        scratch=scratch,
        score_width=score_width,
        page_table=page_table,
        cache_lengths=cache_lengths,
        active_width=active_width,
        candidate_indices=candidate_indices,
        candidate_lengths=candidate_lengths,
        candidate_output=candidate_output,
        candidate_output_lengths=candidate_output_lengths,
    )
    return runtime


@dsl_user_op
def _fused_pack_bf16_pair(lo, hi, *, loc=None, ip=None):
    operands = [
        llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        for v in (lo, hi)
    ]
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "mov.b32 $0, {$1, $2};",
        "=r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_pair_accumulate(total, packed, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.f32(),
        [
            Float32(total).ir_value(loc=loc, ip=ip),
            Uint32(packed).ir_value(loc=loc, ip=ip),
        ],
        """{
        .reg .b16 lo, hi;
        .reg .f32 partial;
        mov.b32 {lo, hi}, $2;
        add.rn.f32.bf16 partial, lo, $1;
        add.rn.f32.bf16 $0, hi, partial;
        }""",
        "=f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def _fused_bf16_products_packed(d0, d1, w0, w1, *, fused_relu=False, loc=None, ip=None):
    operands = [Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1)]
    operands.extend(
        llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        for v in (w0, w1)
    )
    conversion = (
        "cvt.rn.relu.bf16x2.f32 dots, $2, $1;"
        if fused_relu
        else """cvt.rn.bf16x2.f32 dots, $2, $1;
        mov.b32 zero, 0;
        max.bf16x2 dots, dots, zero;"""
    )
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "{\n.reg .b32 dots, weights, zero;\n"
        + conversion
        + """

        mov.b32 weights, {$3, $4};
        mul.bf16x2 $0, dots, weights;
        }""",
        "=r,f,f,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_products_word(d0, d1, weights, *, fused_relu=False, loc=None, ip=None):
    operands = [Float32(v).ir_value(loc=loc, ip=ip) for v in (d0, d1)]
    operands.append(Uint32(weights).ir_value(loc=loc, ip=ip))
    conversion = (
        "cvt.rn.relu.bf16x2.f32 dots, $2, $1;"
        if fused_relu
        else """cvt.rn.bf16x2.f32 dots, $2, $1;
        mov.b32 zero, 0;
        max.bf16x2 dots, dots, zero;"""
    )
    result = llvm.inline_asm(
        T.i32(),
        operands,
        "{\n.reg .b32 dots, zero;\n"
        + conversion
        + """

        mul.bf16x2 $0, dots, $3;
        }""",
        "=r,f,f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@dsl_user_op
def _fused_bf16_pack_weights(w0, w1, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [
            llvm.bitcast(T.i16(), v.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
            for v in (w0, w1)
        ],
        "mov.b32 $0, {$1, $2};",
        "=r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(result)


@cute.jit
def _bf16_bits_to_ordered(bits: Uint32) -> Uint32:
    """Bijection from raw 16-bit BF16 pattern to monotonically ordered uint16."""
    neg = (bits & Uint32(0x8000)) != Uint32(0)
    return (
        ((bits ^ Uint32(0xFFFF)) & Uint32(0xFFFF)) if neg else (bits ^ Uint32(0x8000))
    )


@cute.jit
def _smem_red_i32(base: Int32, idx: Int32, val: Int32):
    red_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_xadd_i32(base: Int32, idx: Int32, val: Int32) -> Int32:
    return atomic_add_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_st_i32(base: Int32, idx: Int32, val: Int32):
    st_shared_i32(base + idx * Int32(4), val)


@cute.jit
def _smem_ld_i32(base: Int32, idx: Int32) -> Int32:
    return ld_shared_i32(base + idx * Int32(4))


@cute.jit
def _warp_suffix_pivot_256(
    hist_addr: Int32,
    tx: Int32,
    target_k: Int32,
    scalars_addr: Int32,
    record_total: cutlass.Constexpr[bool] = False,
):
    """Single-warp suffix sum over 256 int32 histogram bins to locate pivot & remainder."""
    if tx < Int32(32):
        base = hist_addr + tx * Int32(32)
        c0, c1, c2, c3 = ld_shared_v4_u32(base)
        c4, c5, c6, c7 = ld_shared_v4_u32(base + Int32(16))
        s7 = Int32(c7)
        s6 = s7 + Int32(c6)
        s5 = s6 + Int32(c5)
        s4 = s5 + Int32(c4)
        s3 = s4 + Int32(c3)
        s2 = s3 + Int32(c2)
        s1 = s2 + Int32(c1)
        s0 = s1 + Int32(c0)
        above = s0
        for distance in cutlass.range_constexpr(5):
            other = cute.arch.shuffle_sync_down(above, 1 << distance)
            if tx + Int32(1 << distance) < Int32(32):
                above += other
        above -= s0
        suffix = [
            s0 + above,
            s1 + above,
            s2 + above,
            s3 + above,
            s4 + above,
            s5 + above,
            s6 + above,
            s7 + above,
            above,
        ]
        if cutlass.const_expr(record_total):
            if tx == Int32(0):
                _smem_st_i32(scalars_addr, Int32(3), suffix[0])
        for digit in cutlass.range_constexpr(8):
            if (suffix[digit] >= target_k) & (suffix[digit + 1] < target_k):
                _smem_st_i32(scalars_addr, Int32(1), tx * Int32(8) + Int32(digit))
                _smem_st_i32(scalars_addr, Int32(2), target_k - suffix[digit + 1])


@cute.jit
def _radix_select_u16_to_slots(
    vals_u16,
    base_offset: Int64,
    n_total: Int32,
    topk: cutlass.Constexpr[int],
    tx: Int32,
    hist,
    out_slots,
    scalars,
    threads: cutlass.Constexpr[int] = 512,
    priority_index: Int32 | None = None,
    ids_for_priority=None,
    ids_base: Int64 = Int64(0),
):
    """Select exact top-k valid (> -inf) slots from BF16 raw uint16 values.

    Uses a 4-bank replicated 256-bin histogram (``hist`` needs 1025 int32s) and
    warp-aggregated emits to eliminate SMEM atomic serialization.
    Returns ``(selected_count, cutoff_ordered_u16)``.
    """
    hist_addr = shared_ptr_to_u32(hist.iterator)
    scalars_addr = shared_ptr_to_u32(scalars.iterator)
    lane = tx % Int32(32)
    bank = (tx // Int32(32)) & Int32(3)
    bank_addr = hist_addr + bank * Int32(1024)
    lt_mask = (Uint32(1) << Uint32(lane)) - Uint32(1)

    for z in cutlass.range_constexpr(1024 // threads):
        hist[tx + Int32(z * threads)] = Int32(0)
    if tx == Int32(0):
        hist[Int32(1024)] = Int32(0)
        _smem_st_i32(scalars_addr, Int32(0), Int32(0))
        _smem_st_i32(scalars_addr, Int32(3), Int32(0))
    cute.arch.sync_threads()

    # Round 1: 4-bank histogram on high byte of valid ordered keys (> 0x007F).
    base = Int32(0)
    while base < n_total:
        i = base + tx
        if i < n_total:
            key = _bf16_bits_to_ordered(Uint32(vals_u16[base_offset + Int64(i)]))
            if cutlass.const_expr(priority_index is not None):
                if cutlass.const_expr(ids_for_priority is not None):
                    if Int32(ids_for_priority[ids_base + Int64(i)]) == priority_index:
                        key = Uint32(0xFFFF)
                elif i == priority_index:
                    key = Uint32(0xFFFF)
            if key > Uint32(0x007F):
                _smem_red_i32(bank_addr, Int32(key >> Uint32(8)), Int32(1))
        base += Int32(threads)
    cute.arch.sync_threads()

    if tx < Int32(256):
        hist[tx] = (
            Int32(hist[tx])
            + Int32(hist[Int32(256) + tx])
            + Int32(hist[Int32(512) + tx])
            + Int32(hist[Int32(768) + tx])
        )
    cute.arch.sync_threads()

    _warp_suffix_pivot_256(hist_addr, tx, Int32(topk), scalars_addr, record_total=True)
    cute.arch.sync_threads()

    valid_total = _smem_ld_i32(scalars_addr, Int32(3))
    out_count = Int32(topk)
    cutoff_key = Uint32(0x007F)
    if valid_total <= Int32(topk):
        base = Int32(0)
        while base < n_total:
            i = base + tx
            acc = False
            if i < n_total:
                key = _bf16_bits_to_ordered(Uint32(vals_u16[base_offset + Int64(i)]))
                if cutlass.const_expr(priority_index is not None):
                    if cutlass.const_expr(ids_for_priority is not None):
                        if (
                            Int32(ids_for_priority[ids_base + Int64(i)])
                            == priority_index
                        ):
                            key = Uint32(0xFFFF)
                    elif i == priority_index:
                        key = Uint32(0xFFFF)
                acc = key > Uint32(0x007F)
            mask = cute.arch.vote_ballot_sync(acc)
            wn = Int32(cute.arch.popc(mask))
            wbase = Int32(0)
            if (lane == Int32(0)) & (wn > Int32(0)):
                wbase = _smem_xadd_i32(scalars_addr, Int32(0), wn)
            wbase = cute.arch.shuffle_sync(wbase, Int32(0))
            if acc:
                pos = wbase + Int32(cute.arch.popc(mask & lt_mask))
                if pos < Int32(topk):
                    out_slots[pos] = i
            base += Int32(threads)
        cute.arch.sync_threads()
        out_count = valid_total
    else:
        pivot_hi = _smem_ld_i32(scalars_addr, Int32(1))
        rem_hi = _smem_ld_i32(scalars_addr, Int32(2))
        for z in cutlass.range_constexpr(1024 // threads):
            hist[tx + Int32(z * threads)] = Int32(0)
        cute.arch.sync_threads()

        # Fused Round 1 warp-aggregated emit (hi > pivot_hi) + Round 2 4-bank hist.
        base = Int32(0)
        while base < n_total:
            i = base + tx
            acc_gt = False
            if i < n_total:
                key = _bf16_bits_to_ordered(Uint32(vals_u16[base_offset + Int64(i)]))
                if cutlass.const_expr(priority_index is not None):
                    if cutlass.const_expr(ids_for_priority is not None):
                        if (
                            Int32(ids_for_priority[ids_base + Int64(i)])
                            == priority_index
                        ):
                            key = Uint32(0xFFFF)
                    elif i == priority_index:
                        key = Uint32(0xFFFF)
                if key > Uint32(0x007F):
                    hi = Int32(key >> Uint32(8))
                    if hi > pivot_hi:
                        acc_gt = True
                    elif hi == pivot_hi:
                        _smem_red_i32(bank_addr, Int32(key & Uint32(0xFF)), Int32(1))
            mask = cute.arch.vote_ballot_sync(acc_gt)
            wn = Int32(cute.arch.popc(mask))
            wbase = Int32(0)
            if (lane == Int32(0)) & (wn > Int32(0)):
                wbase = _smem_xadd_i32(scalars_addr, Int32(0), wn)
            wbase = cute.arch.shuffle_sync(wbase, Int32(0))
            if acc_gt:
                pos = wbase + Int32(cute.arch.popc(mask & lt_mask))
                if pos < Int32(topk):
                    out_slots[pos] = i
            base += Int32(threads)
        cute.arch.sync_threads()

        if tx < Int32(256):
            hist[tx] = (
                Int32(hist[tx])
                + Int32(hist[Int32(256) + tx])
                + Int32(hist[Int32(512) + tx])
                + Int32(hist[Int32(768) + tx])
            )
        cute.arch.sync_threads()

        _warp_suffix_pivot_256(hist_addr, tx, rem_hi, scalars_addr, record_total=False)
        cute.arch.sync_threads()

        pivot_lo = _smem_ld_i32(scalars_addr, Int32(1))
        cutoff_key = (Uint32(pivot_hi) << Uint32(8)) | Uint32(pivot_lo)
        # Round 2 warp-aggregated emit (lo > pivot_lo and exact tie count).
        base = Int32(0)
        while base < n_total:
            i = base + tx
            acc_gt = False
            acc_eq = False
            if i < n_total:
                key = _bf16_bits_to_ordered(Uint32(vals_u16[base_offset + Int64(i)]))
                if cutlass.const_expr(priority_index is not None):
                    if cutlass.const_expr(ids_for_priority is not None):
                        if (
                            Int32(ids_for_priority[ids_base + Int64(i)])
                            == priority_index
                        ):
                            key = Uint32(0xFFFF)
                    elif i == priority_index:
                        key = Uint32(0xFFFF)
                if (key > Uint32(0x007F)) & (Int32(key >> Uint32(8)) == pivot_hi):
                    lo = Int32(key & Uint32(0xFF))
                    acc_gt = lo > pivot_lo
                    acc_eq = lo == pivot_lo
            m_gt = cute.arch.vote_ballot_sync(acc_gt)
            n_gt = Int32(cute.arch.popc(m_gt))
            b_gt = Int32(0)
            if (lane == Int32(0)) & (n_gt > Int32(0)):
                b_gt = _smem_xadd_i32(scalars_addr, Int32(0), n_gt)
            b_gt = cute.arch.shuffle_sync(b_gt, Int32(0))
            if acc_gt:
                pos = b_gt + Int32(cute.arch.popc(m_gt & lt_mask))
                if pos < Int32(topk):
                    out_slots[pos] = i
            m_eq = cute.arch.vote_ballot_sync(acc_eq)
            n_eq = Int32(cute.arch.popc(m_eq))
            old_rem = Int32(0)
            if (lane == Int32(0)) & (n_eq > Int32(0)):
                old_rem = _smem_xadd_i32(scalars_addr, Int32(2), -n_eq)
            old_rem = cute.arch.shuffle_sync(old_rem, Int32(0))
            if acc_eq:
                rem_slot = old_rem - Int32(cute.arch.popc(m_eq & lt_mask))
                if rem_slot > Int32(0):
                    out_slots[Int32(topk) - rem_slot] = i
            base += Int32(threads)
        cute.arch.sync_threads()
    return out_count, cutoff_key


@cute.jit
def _trim_smem_u16_512(
    carry_val,
    carry_idx,
    base: Int32,
    count: Int32,
    tx: Int32,
    hist,
    out_slots,
    scalars,
    threads: cutlass.Constexpr[int] = 512,
):
    """Trim an SMEM BF16 carry buffer in-place to top-512 and return (new_count, cutoff)."""
    selected_count, cutoff_key = _radix_select_u16_to_slots(
        carry_val,
        Int64(base),
        count,
        512,
        tx,
        hist,
        out_slots,
        scalars,
        threads=threads,
    )
    rv = Uint16(0xFF80)
    ri = Int32(-1)
    if tx < selected_count:
        src = Int32(out_slots[tx])
        rv = Uint16(carry_val[base + src])
        ri = Int32(carry_idx[base + src])
    cute.arch.sync_threads()
    if tx < selected_count:
        carry_val[base + tx] = rv
        carry_idx[base + tx] = ri
    cute.arch.sync_threads()
    return selected_count, cutoff_key


@cute.jit
def _bitonic_sort_512_u16_to_out(
    out_indices,
    out_values,
    row: Int32,
    ids,
    vals_u16,
    base: Int32,
    count: Int32,
    tx: Int32,
):
    """Sort up to 512 (int32 pos, uint16 bf16) pairs ascending by pos and write outputs."""
    if tx < Int32(512):
        pos = Int32(2147483647)
        val = Uint32(0xFF80)
        if tx < count:
            raw_pos = Int32(ids[base + tx])
            raw_val = Uint32(vals_u16[base + tx])
            if (raw_pos >= Int32(0)) & (raw_val != Uint32(0xFF80)):
                pos = raw_pos
                val = raw_val
        cute.arch.barrier(barrier_id=1, number_of_threads=512)
        for level in cutlass.range_constexpr(1, 10):
            for stage in cutlass.range_constexpr(level):
                stride = 1 << (level - 1 - stage)
                other_pos = Int32(2147483647)
                other_val = Uint32(0xFF80)
                if cutlass.const_expr(stride < 32):
                    other_pos = cute.arch.shuffle_sync_bfly(pos, offset=stride)
                    other_val = cute.arch.shuffle_sync_bfly(val, offset=stride)
                else:
                    ids[base + tx] = pos
                    vals_u16[base + tx] = Uint16(val)
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                    partner = tx ^ Int32(stride)
                    other_pos = Int32(ids[base + partner])
                    other_val = Uint32(vals_u16[base + partner])
                    cute.arch.barrier(barrier_id=1, number_of_threads=512)
                ascending = (tx & Int32(1 << level)) == Int32(0)
                lower = (tx & Int32(stride)) == Int32(0)
                take_min = ascending == lower
                if ((pos > other_pos) & take_min) | (
                    (pos < other_pos) & (not take_min)
                ):
                    pos = other_pos
                    val = other_val
        valid = pos != Int32(2147483647)
        out_off = Int64(row) * Int64(512) + Int64(tx)
        out_indices[out_off] = pos if valid else Int32(-1)
        out_values[out_off] = (
            u32_as_f32(val << Uint32(16)) if valid else Float32(-float("inf"))
        )


@cute.jit
def _resolve_request_row(
    row: Int32,
    request_meta,
    request_mode: Int32,
    num_requests: Int32,
) -> Int32:
    """Map query row to page-table row (0=identity, 1=request_ids, 2=query_start_loc)."""
    res = row
    if request_mode == Int32(1):
        res = Int32(request_meta[row])
    elif request_mode == Int32(2):
        lo = Int32(0)
        hi = num_requests
        while lo + Int32(1) < hi:
            mid = (lo + hi) // Int32(2)
            if Int32(request_meta[mid]) <= row:
                lo = mid
            else:
                hi = mid
        res = lo
    return res


def multiquery_scratch_shapes(rows: int, ctas: int) -> dict:
    """Return dtype/shape pairs for bounded multi-query dense MXFP4 scratch."""
    if rows < 1 or ctas < 1:
        raise ValueError("multiquery rows and CTA count must be positive")
    partial_shape = (rows, ctas, 512) if ctas > 1 else (1,)
    count_shape = (rows, ctas) if ctas > 1 else (1,)
    return {
        "selected_values": (torch.float32, (rows, 512)),
        "partial_indices": (torch.int32, partial_shape),
        "partial_values": (torch.int16, partial_shape),
        "partial_counts": (torch.int32, count_shape),
        "merge_state": (torch.int32, (2,)),
    }


class MultiQueryFusedMXFP4:
    """Request-grouped multi-query MXFP4 dense scorer and exact top-512 selector.

    Groups up to ``q_tile`` queries in one 512-thread CTA so queries sharing
    physical pages load each 68-byte MXFP4 key once into registers and multiply
    against all grouped queries' staged SMEM heads via ``m16n8k64``.
    """

    def __init__(
        self,
        heads: int,
        page_size: int,
        ctas: int = 1,
        q_tile: int = 8,
    ):
        if heads < 1 or heads > 64 or 64 % heads:
            raise ValueError("MXFP4 heads must divide 64")
        if page_size < 8 or page_size % 8:
            raise ValueError("MXFP4 page_size must be a positive multiple of eight")
        if not 1 <= ctas < 2**31:
            raise ValueError("invalid multiquery CTA count")
        if q_tile not in (2, 4, 8):
            raise ValueError("multiquery q_tile must be 2, 4, or 8")
        self.heads = int(heads)
        self.page_size = int(page_size)
        self.ctas = int(ctas)
        self.q_tile = int(q_tile)

    @cute.jit
    def local(
        self,
        q: cute.Pointer,
        qs: cute.Pointer,
        weights: cute.Pointer,
        pool: cute.Pointer,
        pages: cute.Pointer,
        request_meta: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        merge_state: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        request_mode: Int32,
        num_requests: Int32,
        stream: cuda.CUstream,
    ):
        num_groups = (Int64(rows) + Int64(self.q_tile - 1)) // Int64(self.q_tile)
        self.local_kernel(
            _flat(q),
            _flat(qs),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(request_meta),
            _flat(lengths),
            _flat(active),
            _flat(out_indices),
            _flat(out_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            _flat(merge_state),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
            request_mode,
            num_requests,
        ).launch(
            grid=(num_groups * Int64(self.ctas), 1, 1),
            block=(1024, 1, 1),
            stream=stream,
        )

    @cute.jit
    def merge(
        self,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.merge_kernel(
            _flat(out_indices),
            _flat(out_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            rows,
        ).launch(
            grid=(Int64(rows), 1, 1),
            block=(512, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def local_kernel(
        self,
        q,
        qs,
        weights,
        pool,
        pages,
        request_meta,
        lengths,
        active,
        out_indices,
        out_values,
        partial_indices,
        partial_values,
        partial_counts,
        merge_state,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        request_mode: Int32,
        num_requests: Int32,
    ):
        tx_u, _, _ = cute.arch.thread_idx()
        bx_u, _, _ = cute.arch.block_idx()
        tx = Int32(tx_u)
        block = Int32(bx_u)
        warp = tx // Int32(32)
        lane = tx % Int32(32)
        group = block // Int32(self.ctas)
        partition = block % Int32(self.ctas)
        row_base = group * Int32(self.q_tile)

        if (block == Int32(0)) & (tx < Int32(2)):
            merge_state[tx] = Int32(0)

        smem = cutlass.utils.SmemAllocator()
        num_ht = (self.heads + 7) // 8
        weight_pairs_per_row = (self.heads + 1) // 2
        carry_cap = 1280
        # Vectorized 128-bit Q layout: [q_tile, num_ht, 32_lanes, 4_kchunks].
        q_smem = smem.allocate_tensor(
            Uint32, cute.make_layout((self.q_tile * num_ht * 128,)), byte_alignment=16
        )
        qs_smem = smem.allocate_tensor(
            Uint32, cute.make_layout((self.q_tile * num_ht * 8,)), byte_alignment=16
        )
        w_smem = smem.allocate_tensor(
            Uint32,
            cute.make_layout((self.q_tile * weight_pairs_per_row,)),
            byte_alignment=16,
        )
        carry_val = smem.allocate_tensor(
            Uint16, cute.make_layout((self.q_tile * carry_cap,)), byte_alignment=16
        )
        carry_idx = smem.allocate_tensor(
            Int32, cute.make_layout((self.q_tile * carry_cap,)), byte_alignment=16
        )
        hist = smem.allocate_tensor(Int32, cute.make_layout((1025,)), byte_alignment=16)
        out_slots = smem.allocate_tensor(
            Int32, cute.make_layout((512,)), byte_alignment=16
        )
        scalars = smem.allocate_tensor(
            Int32, cute.make_layout((4 + 2 * self.q_tile,)), byte_alignment=16
        )
        q_extents = smem.allocate_tensor(
            Int32, cute.make_layout((self.q_tile,)), byte_alignment=16
        )
        q_page_rows = smem.allocate_tensor(
            Int32, cute.make_layout((self.q_tile,)), byte_alignment=16
        )
        scalars_addr = shared_ptr_to_u32(scalars.iterator)
        q_smem_addr = shared_ptr_to_u32(q_smem.iterator)

        active_val = cutlass.max(Int32(active[0]), Int32(0))
        if tx < Int32(self.q_tile):
            r = row_base + tx
            ext = Int32(0)
            prow = Int32(0)
            if r < rows:
                vis = cutlass.min(cutlass.max(Int32(lengths[r]), Int32(0)), active_val)
                ext = cutlass.min(vis, width)
                prow = _resolve_request_row(r, request_meta, request_mode, num_requests)
            q_extents[tx] = ext
            q_page_rows[tx] = prow
            _smem_st_i32(scalars_addr, Int32(4) + tx, Int32(0))
            _smem_st_i32(scalars_addr, Int32(4 + self.q_tile) + tx, Int32(0x007F))

        total_q_words = Int32(self.q_tile * num_ht * 128)
        idx = tx
        while idx < total_q_words:
            ql = idx // Int32(num_ht * 128)
            rem = idx - ql * Int32(num_ht * 128)
            ht = rem // Int32(128)
            rem2 = rem - ht * Int32(128)
            l_idx = rem2 // Int32(4)
            kchunk = rem2 - l_idx * Int32(4)
            h = ht * Int32(8) + (l_idx // Int32(4))
            w_in_h = kchunk * Int32(4) + (l_idx % Int32(4))
            r = row_base + ql
            word = Uint32(0)
            if (r < rows) & (h < Int32(self.heads)):
                bbase = (Int64(r) * Int64(self.heads) + Int64(h)) * Int64(64) + Int64(
                    w_in_h
                ) * Int64(4)
                word = (
                    Uint32(q[bbase])
                    | (Uint32(q[bbase + Int64(1)]) << Uint32(8))
                    | (Uint32(q[bbase + Int64(2)]) << Uint32(16))
                    | (Uint32(q[bbase + Int64(3)]) << Uint32(24))
                )
            q_smem[idx] = word
            idx += Int32(1024)

        total_qs_words = Int32(self.q_tile * num_ht * 8)
        idx = tx
        while idx < total_qs_words:
            ql = idx // Int32(num_ht * 8)
            h = idx - ql * Int32(num_ht * 8)
            r = row_base + ql
            word = Uint32(0x7F7F7F7F)
            if (r < rows) & (h < Int32(self.heads)):
                sbase = (Int64(r) * Int64(self.heads) + Int64(h)) * Int64(4)
                word = (
                    Uint32(qs[sbase])
                    | (Uint32(qs[sbase + Int64(1)]) << Uint32(8))
                    | (Uint32(qs[sbase + Int64(2)]) << Uint32(16))
                    | (Uint32(qs[sbase + Int64(3)]) << Uint32(24))
                )
            qs_smem[idx] = word
            idx += Int32(1024)

        total_w_words = Int32(self.q_tile * weight_pairs_per_row)
        idx = tx
        while idx < total_w_words:
            ql = idx // Int32(weight_pairs_per_row)
            pair = idx - ql * Int32(weight_pairs_per_row)
            r = row_base + ql
            word = Uint32(0)
            if r < rows:
                w0 = weights[Int64(r) * Int64(self.heads) + Int64(pair * Int32(2))]
                w1 = BFloat16(0)
                if cutlass.const_expr(self.heads != 1):
                    w1 = weights[
                        Int64(r) * Int64(self.heads) + Int64(pair * Int32(2) + Int32(1))
                    ]
                word = _fused_bf16_pack_weights(w0, w1)
            w_smem[idx] = word
            idx += Int32(1024)

        cute.arch.sync_threads()

        max_extent = Int32(0)
        for ql in cutlass.range_constexpr(self.q_tile):
            max_extent = cutlass.max(max_extent, Int32(q_extents[ql]))
        total_tiles16 = (max_extent + Int32(15)) // Int32(16)
        begin = Int32(
            (Int64(total_tiles16) * Int64(partition) // Int64(self.ctas)) * Int64(16)
        )
        end = Int32(
            cutlass.min(
                Int64(max_extent),
                (Int64(total_tiles16) * Int64(partition + Int32(1)) // Int64(self.ctas))
                * Int64(16),
            )
        )

        byte_off = Int64(lane & Int32(3)) * Int64(4)
        lt_mask = (Uint32(1) << Uint32(lane)) - Uint32(1)
        lane_is_out = (lane & Int32(3)) == Int32(0)
        lane_is_even = (lane & Int32(1)) == Int32(0)
        lane_row = lane >> Int32(2)
        lane_w_off = lane & Int32(3)

        tile = begin
        while tile < end:
            base_col = tile + (warp << Int32(4))
            col0 = base_col + lane_row
            col1 = col0 + Int32(8)
            in_tile0 = col0 < end
            in_tile1 = col1 < end
            pcol0 = base_col // Int32(self.page_size)
            tok0 = col0 % Int32(self.page_size)
            pcol1 = (base_col + Int32(8)) // Int32(self.page_size)
            tok1 = col1 % Int32(self.page_size)

            loaded_p0 = Int32(-2)
            loaded_p1 = Int32(-2)
            last_prow = Int32(-1)
            p0 = Int32(-1)
            p1 = Int32(-1)
            a0_0 = Uint32(0)
            a2_0 = Uint32(0)
            a0_1 = Uint32(0)
            a2_1 = Uint32(0)
            a1_0 = Uint32(0)
            a3_0 = Uint32(0)
            a1_1 = Uint32(0)
            a3_1 = Uint32(0)
            sfa = Uint32(0x7F7F7F7F)

            for ql in cutlass.range(self.q_tile, unroll=1):
                ext_q = Int32(q_extents[ql])
                if (base_col < ext_q) & (base_col < end):
                    prow_idx = Int32(q_page_rows[ql])
                    if (last_prow < Int32(0)) | (
                        (prow_idx != last_prow) & (page_row_stride != Int64(0))
                    ):
                        prow = Int64(prow_idx) * page_row_stride
                        p0 = Int32(-1)
                        if pcol0 < page_width:
                            raw_p0 = Int32(pages[prow + Int64(pcol0)])
                            if (raw_p0 >= Int32(0)) & (Int64(raw_p0) < pool_pages):
                                p0 = raw_p0
                        if cutlass.const_expr(self.page_size % 16 != 0):
                            p1 = p0
                            if cutlass.const_expr(self.page_size == 8):
                                p1 = Int32(-1)
                                if pcol1 < page_width:
                                    raw_p1 = Int32(pages[prow + Int64(pcol1)])
                                    if (raw_p1 >= Int32(0)) & (
                                        Int64(raw_p1) < pool_pages
                                    ):
                                        p1 = raw_p1
                            elif pcol1 != pcol0:
                                p1 = Int32(-1)
                                if pcol1 < page_width:
                                    raw_p1 = Int32(pages[prow + Int64(pcol1)])
                                    if (raw_p1 >= Int32(0)) & (
                                        Int64(raw_p1) < pool_pages
                                    ):
                                        p1 = raw_p1
                        last_prow = prow_idx

                    if cutlass.const_expr(self.page_size % 16 == 0):
                        p0_valid = p0 >= Int32(0)
                        p1_valid = p0_valid
                        if p0 != loaded_p0:
                            if p0_valid:
                                p0_base = Int64(p0) * pool_stride
                                kb0 = p0_base + Int64(tok0) * Int64(64) + byte_off
                                a0_0 = _packed_word(pool, kb0)
                                a2_0 = _packed_word(pool, kb0 + Int64(16))
                                a0_1 = _packed_word(pool, kb0 + Int64(32))
                                a2_1 = _packed_word(pool, kb0 + Int64(48))
                                a1_0 = _packed_word(pool, kb0 + Int64(512))
                                a3_0 = _packed_word(pool, kb0 + Int64(528))
                                a1_1 = _packed_word(pool, kb0 + Int64(544))
                                a3_1 = _packed_word(pool, kb0 + Int64(560))
                                s_tok = tok0 + ((lane & Int32(1)) << Int32(3))
                                sfa = _packed_word(
                                    pool,
                                    p0_base
                                    + Int64(self.page_size * 64)
                                    + Int64(s_tok) * Int64(4),
                                )
                            else:
                                a0_0 = Uint32(0)
                                a2_0 = Uint32(0)
                                a0_1 = Uint32(0)
                                a2_1 = Uint32(0)
                                a1_0 = Uint32(0)
                                a3_0 = Uint32(0)
                                a1_1 = Uint32(0)
                                a3_1 = Uint32(0)
                                sfa = Uint32(0x7F7F7F7F)
                            loaded_p0 = p0
                    else:
                        p0_valid = p0 >= Int32(0)
                        p1_valid = p1 >= Int32(0)
                        if p0 != loaded_p0:
                            if p0_valid:
                                p0_base = Int64(p0) * pool_stride
                                kb0 = p0_base + Int64(tok0) * Int64(64) + byte_off
                                a0_0 = _packed_word(pool, kb0)
                                a2_0 = _packed_word(pool, kb0 + Int64(16))
                                a0_1 = _packed_word(pool, kb0 + Int64(32))
                                a2_1 = _packed_word(pool, kb0 + Int64(48))
                                if lane_is_even:
                                    sfa = _packed_word(
                                        pool,
                                        p0_base
                                        + Int64(self.page_size * 64)
                                        + Int64(tok0) * Int64(4),
                                    )
                            else:
                                a0_0 = Uint32(0)
                                a2_0 = Uint32(0)
                                a0_1 = Uint32(0)
                                a2_1 = Uint32(0)
                                if lane_is_even:
                                    sfa = Uint32(0x7F7F7F7F)
                            loaded_p0 = p0
                        if p1 != loaded_p1:
                            if p1_valid:
                                p1_base = Int64(p1) * pool_stride
                                kb1 = p1_base + Int64(tok1) * Int64(64) + byte_off
                                a1_0 = _packed_word(pool, kb1)
                                a3_0 = _packed_word(pool, kb1 + Int64(16))
                                a1_1 = _packed_word(pool, kb1 + Int64(32))
                                a3_1 = _packed_word(pool, kb1 + Int64(48))
                                if not lane_is_even:
                                    sfa = _packed_word(
                                        pool,
                                        p1_base
                                        + Int64(self.page_size * 64)
                                        + Int64(tok1) * Int64(4),
                                    )
                            else:
                                a1_0 = Uint32(0)
                                a3_0 = Uint32(0)
                                a1_1 = Uint32(0)
                                a3_1 = Uint32(0)
                                if not lane_is_even:
                                    sfa = Uint32(0x7F7F7F7F)
                            loaded_p1 = p1

                    tot0 = Float32(0.0)
                    tot1 = Float32(0.0)
                    q_row_ht = ql * Int32(num_ht)
                    for head_tile in cutlass.range(num_ht, unroll=1):
                        ht_idx = q_row_ht + head_tile
                        q_addr = q_smem_addr + Uint32(
                            ((ht_idx << Int32(5)) + lane) << Int32(4)
                        )
                        b0_0, b1_0, b0_1, b1_1 = ld_shared_v4_u32(q_addr)
                        sfb = Uint32(qs_smem[(ht_idx << Int32(3)) + lane_row])
                        d0, d1, d2, d3 = _mxfp4_mma(
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            a0_0,
                            a1_0,
                            a2_0,
                            a3_0,
                            b0_0,
                            b1_0,
                            sfa,
                            sfb,
                            scale_byte_id=0,
                        )
                        d0, d1, d2, d3 = _mxfp4_mma(
                            d0,
                            d1,
                            d2,
                            d3,
                            a0_1,
                            a1_1,
                            a2_1,
                            a3_1,
                            b0_1,
                            b1_1,
                            sfa,
                            sfb,
                            scale_byte_id=2,
                        )
                        w_idx = (head_tile << Int32(2)) + lane_w_off
                        w_word = Uint32(0)
                        if cutlass.const_expr(self.heads >= 8):
                            w_word = Uint32(
                                w_smem[ql * Int32(weight_pairs_per_row) + w_idx]
                            )
                        elif w_idx < Int32(weight_pairs_per_row):
                            w_word = Uint32(
                                w_smem[ql * Int32(weight_pairs_per_row) + w_idx]
                            )
                        p_lo = _fused_bf16_products_word(
                            d0, d1, w_word, fused_relu=True
                        )
                        p_hi = _fused_bf16_products_word(
                            d2, d3, w_word, fused_relu=True
                        )
                        for hp in cutlass.range_constexpr(min(4, weight_pairs_per_row)):
                            if cutlass.const_expr(hp == 0):
                                pk0, pk1 = p_lo, p_hi
                            else:
                                pk0 = cute.arch.shuffle_sync_down(p_lo, offset=hp)
                                pk1 = cute.arch.shuffle_sync_down(p_hi, offset=hp)
                            if cutlass.const_expr(self.heads == 1):
                                tot0 += u32_as_f32(pk0 << Uint32(16))
                                tot1 += u32_as_f32(pk1 << Uint32(16))
                            else:
                                tot0 = _fused_bf16_pair_accumulate(tot0, pk0)
                                tot1 = _fused_bf16_pair_accumulate(tot1, pk1)

                    v0_ok = lane_is_out & p0_valid & (col0 < ext_q) & in_tile0
                    v1_ok = lane_is_out & p1_valid & (col1 < ext_q) & in_tile1
                    f0 = Float32(BFloat16(tot0)) if v0_ok else Float32(-float("inf"))
                    f1 = Float32(BFloat16(tot1)) if v1_ok else Float32(-float("inf"))
                    u0 = f32_to_raw_bits(f0) >> Uint32(16)
                    u1 = f32_to_raw_bits(f1) >> Uint32(16)
                    k0 = _bf16_bits_to_ordered(u0)
                    k1 = _bf16_bits_to_ordered(u1)
                    cut_q = Uint32(
                        _smem_ld_i32(scalars_addr, Int32(4 + self.q_tile) + ql)
                    )
                    acc0 = v0_ok & (k0 > cut_q)
                    acc1 = v1_ok & (k1 > cut_q)
                    m0 = cute.arch.vote_ballot_sync(acc0)
                    m1 = cute.arch.vote_ballot_sync(acc1)
                    n0 = Int32(cute.arch.popc(m0))
                    n1 = Int32(cute.arch.popc(m1))
                    wn = n0 + n1
                    if wn > Int32(0):
                        wbase = Int32(0)
                        if lane == Int32(0):
                            wbase = _smem_xadd_i32(scalars_addr, Int32(4) + ql, wn)
                        wbase = cute.arch.shuffle_sync(wbase, Int32(0))
                        cbase = ql * Int32(carry_cap)
                        if acc0:
                            slot = wbase + Int32(cute.arch.popc(m0 & lt_mask))
                            carry_val[cbase + slot] = Uint16(u0)
                            carry_idx[cbase + slot] = col0
                        if acc1:
                            slot = wbase + n0 + Int32(cute.arch.popc(m1 & lt_mask))
                            carry_val[cbase + slot] = Uint16(u1)
                            carry_idx[cbase + slot] = col1

            cute.arch.sync_threads()
            for ql in cutlass.range(self.q_tile, unroll=1):
                cnt = _smem_ld_i32(scalars_addr, Int32(4) + ql)
                if cnt > Int32(carry_cap - 512):
                    new_cnt, new_cut = _trim_smem_u16_512(
                        carry_val,
                        carry_idx,
                        ql * Int32(carry_cap),
                        cnt,
                        tx,
                        hist,
                        out_slots,
                        scalars,
                        threads=1024,
                    )
                    if tx == Int32(0):
                        _smem_st_i32(scalars_addr, Int32(4) + ql, new_cnt)
                        _smem_st_i32(
                            scalars_addr, Int32(4 + self.q_tile) + ql, Int32(new_cut)
                        )
                    cute.arch.sync_threads()
            cute.arch.sync_threads()
            tile += Int32(512)

        for ql in cutlass.range(self.q_tile, unroll=1):
            r = row_base + ql
            if r < rows:
                cnt = _smem_ld_i32(scalars_addr, Int32(4) + ql)
                if cnt > Int32(512):
                    cnt, _ = _trim_smem_u16_512(
                        carry_val,
                        carry_idx,
                        ql * Int32(carry_cap),
                        cnt,
                        tx,
                        hist,
                        out_slots,
                        scalars,
                        threads=1024,
                    )
                if cutlass.const_expr(self.ctas > 1):
                    pbase = (Int64(r) * Int64(self.ctas) + Int64(partition)) * Int64(
                        512
                    )
                    if tx < cnt:
                        partial_values[pbase + Int64(tx)] = Uint16(
                            carry_val[ql * Int32(carry_cap) + tx]
                        )
                        partial_indices[pbase + Int64(tx)] = Int32(
                            carry_idx[ql * Int32(carry_cap) + tx]
                        )
                    if tx == Int32(0):
                        partial_counts[
                            Int64(r) * Int64(self.ctas) + Int64(partition)
                        ] = cnt
                else:
                    _bitonic_sort_512_u16_to_out(
                        out_indices,
                        out_values,
                        r,
                        carry_idx,
                        carry_val,
                        ql * Int32(carry_cap),
                        cnt,
                        tx,
                    )
                    cute.arch.sync_threads()

    @cute.kernel
    def merge_kernel(
        self,
        out_indices,
        out_values,
        partial_indices,
        partial_values,
        partial_counts,
        rows: Int32,
    ):
        tx_u, _, _ = cute.arch.thread_idx()
        bx_u, _, _ = cute.arch.block_idx()
        tx = Int32(tx_u)
        row = Int32(bx_u)
        lane = tx % Int32(32)
        lt_mask = (Uint32(1) << Uint32(lane)) - Uint32(1)

        smem = cutlass.utils.SmemAllocator()
        carry_val = smem.allocate_tensor(
            Uint16, cute.make_layout((2048,)), byte_alignment=16
        )
        carry_idx = smem.allocate_tensor(
            Int32, cute.make_layout((2048,)), byte_alignment=16
        )
        hist = smem.allocate_tensor(Int32, cute.make_layout((1025,)), byte_alignment=16)
        out_slots = smem.allocate_tensor(
            Int32, cute.make_layout((512,)), byte_alignment=16
        )
        scalars = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        scalars_addr = shared_ptr_to_u32(scalars.iterator)

        if tx == Int32(0):
            _smem_st_i32(scalars_addr, Int32(4), Int32(0))
        cute.arch.sync_threads()
        cutoff = Uint32(0x007F)

        for part in cutlass.range(self.ctas):
            pbase = (Int64(row) * Int64(self.ctas) + Int64(part)) * Int64(512)
            cnt = Int32(partial_counts[Int64(row) * Int64(self.ctas) + Int64(part)])
            on = _smem_ld_i32(scalars_addr, Int32(4))
            if on + cnt > Int32(2048):
                on, cutoff = _trim_smem_u16_512(
                    carry_val,
                    carry_idx,
                    Int32(0),
                    on,
                    tx,
                    hist,
                    out_slots,
                    scalars,
                    threads=512,
                )
                if tx == Int32(0):
                    _smem_st_i32(scalars_addr, Int32(4), on)
            cute.arch.sync_threads()
            if cnt > Int32(0):
                u = Uint32(0xFF80)
                idx_v = Int32(-1)
                acc = False
                if tx < cnt:
                    u = Uint32(partial_values[pbase + Int64(tx)])
                    idx_v = Int32(partial_indices[pbase + Int64(tx)])
                    acc = _bf16_bits_to_ordered(u) > cutoff
                mask = cute.arch.vote_ballot_sync(acc)
                wn = Int32(cute.arch.popc(mask))
                wbase = Int32(0)
                if (lane == Int32(0)) & (wn > Int32(0)):
                    wbase = _smem_xadd_i32(scalars_addr, Int32(4), wn)
                wbase = cute.arch.shuffle_sync(wbase, Int32(0))
                if acc:
                    slot = wbase + Int32(cute.arch.popc(mask & lt_mask))
                    carry_val[slot] = Uint16(u)
                    carry_idx[slot] = idx_v
                cute.arch.sync_threads()

        on = _smem_ld_i32(scalars_addr, Int32(4))
        if on > Int32(512):
            on, _ = _trim_smem_u16_512(
                carry_val,
                carry_idx,
                Int32(0),
                on,
                tx,
                hist,
                out_slots,
                scalars,
                threads=512,
            )
        _bitonic_sort_512_u16_to_out(
            out_indices, out_values, row, carry_idx, carry_val, Int32(0), on, tx
        )
