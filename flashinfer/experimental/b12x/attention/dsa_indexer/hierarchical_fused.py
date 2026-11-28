"""Bounded fused token selection and hierarchical block candidate publication.

Each partition retains token top-512 and block-max top-2048 independently.
Only those bounded carries cross CTA boundaries; no context-sized scores are
materialized. Block maxima consume the scorer's final BF16-rounded values.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint16, Uint32

from b12x._lib.intrinsics.math import fmax_f32
from b12x._lib.intrinsics.memory import (
    ld_shared_v4_u32,
    shared_ptr_to_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.packed import f32_to_raw_bits
from .fused_indexer import _mxfp4_order_small_output, _mxfp4_radix_select
from .mxfp4 import (
    _bf16_bits_to_ordered,
    _bitonic_sort_512_u16_to_out,
    _flat,
    _fused_bf16_pack_weights,
    _fused_bf16_pair_accumulate,
    _fused_bf16_products_word,
    _mxfp4_mma,
    _packed_word,
    _radix_select_u16_to_slots,
    _resolve_request_row,
    _smem_ld_i32,
    _smem_st_i32,
    _smem_xadd_i32,
    _TensorCorePagedScore,
    _trim_smem_u16_512,
)


def hierarchical_scratch_shapes(rows: int, ctas: int) -> dict:
    """Return dtype/shape pairs for bounded, caller-owned producer scratch."""
    if rows < 1 or ctas < 1:
        raise ValueError("hierarchical rows and CTA count must be positive")
    if rows * ctas > 2**31 - 1:
        raise ValueError("hierarchical rows times CTA count exceeds CUDA grid x limit")
    partial = (rows, ctas, 2560) if ctas > 1 else (1,)
    return {
        "partial_indices": (torch.int32, partial),
        "partial_values": (torch.float32, partial),
        "partial_counts": (torch.int32, (rows, ctas, 2) if ctas > 1 else (1,)),
        "block_indices": (torch.int32, (rows, 2048)),
        "block_values": (torch.float32, (rows, 2048)),
        "selected_values": (torch.float32, (rows, 512)),
        "state": (torch.int32, (2,)),
    }


@cute.jit
def _trim(
    values,
    ids,
    count,
    topk: cutlass.Constexpr[int],
    capacity: cutlass.Constexpr[int],
    tx,
    hist,
    selected,
    scalars,
    priority_id: Int32 | None = None,
):
    if count > Int32(topk):
        _mxfp4_radix_select(
            values,
            ids,
            values,
            ids,
            count,
            Int32(topk),
            tx,
            hist,
            hist,
            selected,
            shared_ptr_to_u32(hist.iterator),
            shared_ptr_to_u32(scalars.iterator),
            shared_ptr_to_u32(scalars.iterator) + Int32(4),
            shared_ptr_to_u32(scalars.iterator) + Int32(8),
            max_items=capacity,
            indices_only=True,
            priority_id=priority_id,
        )
        rv = cute.make_rmem_tensor(((topk + 1023) // 1024,), Float32)
        ri = cute.make_rmem_tensor(((topk + 1023) // 1024,), Int32)
        for slot in cutlass.range_constexpr((topk + 1023) // 1024):
            i = tx + Int32(slot * 1024)
            if i < Int32(topk):
                source = Int32(selected[i])
                rv[slot] = Float32(values[source])
                ri[slot] = Int32(ids[source])
        cute.arch.sync_threads()
        for slot in cutlass.range_constexpr((topk + 1023) // 1024):
            i = tx + Int32(slot * 1024)
            if i < Int32(topk):
                values[i] = rv[slot]
                ids[i] = ri[slot]
        cute.arch.sync_threads()
        count = Int32(topk)
    return count


@cute.jit
def _row_output(ptr):
    return cute.make_tensor(
        ptr,
        cute.make_layout(
            (Int64(1) << Int64(30), Int64(512)), stride=(Int64(512), Int64(1))
        ),
    )


class HierarchicalFusedMXFP4:
    """Pointer-only local/merge programs with runtime rows, widths and strides.

    ``scorer`` may supply the production scorer's eligible byte-load policy.
    The scorer must provide the exact ``_score_tile_fused`` helper and dense
    logical indexing. Compilation and scratch ownership belong to the caller.
    """

    def __init__(self, heads: int, page_size: int, ctas: int = 1, scorer=None):
        if heads < 1 or heads > 64 or 64 % heads:
            raise ValueError("MXFP4 heads must divide 64")
        if page_size < 8 or page_size % 8 or not 1 <= ctas < 2**31:
            raise ValueError("invalid hierarchical page size or CTA count")
        self.ctas = int(ctas)
        self.heads = int(heads)
        self.page_size = int(page_size)
        self._use_fast_scorer = scorer is None
        self.scorer = (
            scorer
            if scorer is not None
            else _TensorCorePagedScore(heads, False, page_size)
        )

    @cute.jit
    def local(
        self,
        q: cute.Pointer,
        qs: cute.Pointer,
        weights: cute.Pointer,
        pool: cute.Pointer,
        pages: cute.Pointer,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        block_indices: cute.Pointer,
        block_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        state: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(q),
            _flat(qs),
            _flat(weights),
            _flat(pool),
            _flat(pages),
            _flat(lengths),
            _flat(active),
            _row_output(out_indices),
            _row_output(out_values),
            _flat(block_indices),
            _flat(block_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            _flat(state),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
            False,
        ).launch(
            grid=(Int64(rows) * Int64(self.ctas), 1, 1),
            block=(1024, 1, 1),
            stream=stream,
        )

    @cute.jit
    def merge(
        self,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        block_indices: cute.Pointer,
        block_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            None,
            None,
            None,
            None,
            None,
            _flat(lengths),
            _flat(active),
            _row_output(out_indices),
            _row_output(out_values),
            _flat(block_indices),
            _flat(block_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            None,
            rows,
            Int32(0),
            Int32(0),
            Int64(0),
            Int64(0),
            Int64(0),
            True,
        ).launch(grid=(rows, 1, 1), block=(1024, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        q,
        qs,
        weights,
        pool,
        pages,
        lengths,
        active,
        out_indices,
        out_values,
        block_indices,
        block_values,
        partial_indices,
        partial_values,
        partial_counts,
        state,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        merge: cutlass.Constexpr[bool],
    ):
        tx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        tx = Int32(tx)
        if cutlass.const_expr(merge):
            row, partition = Int32(block), Int32(0)
        else:
            row = Int32(block) // Int32(self.ctas)
            partition = Int32(block) % Int32(self.ctas)
        smem = cutlass.utils.SmemAllocator()
        ov = smem.allocate_tensor(Float32, cute.make_layout((2048,)), byte_alignment=16)
        oi = smem.allocate_tensor(Int32, cute.make_layout((2048,)), byte_alignment=16)
        bv = smem.allocate_tensor(Float32, cute.make_layout((4096,)), byte_alignment=16)
        bi = smem.allocate_tensor(Int32, cute.make_layout((4096,)), byte_alignment=16)
        hist = smem.allocate_tensor(Int32, cute.make_layout((264,)), byte_alignment=16)
        selected = smem.allocate_tensor(
            Int32, cute.make_layout((2048,)), byte_alignment=16
        )
        scalars = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        weight_pairs_per_row = (self.heads + 1) // 2
        if cutlass.const_expr(not merge and self._use_fast_scorer):
            q_smem = smem.allocate_tensor(
                Uint32, cute.make_layout((self.heads * 16,)), byte_alignment=16
            )
            qs_smem = smem.allocate_tensor(
                Uint32, cute.make_layout((self.heads,)), byte_alignment=16
            )
            w_smem = smem.allocate_tensor(
                Uint32, cute.make_layout((weight_pairs_per_row,)), byte_alignment=16
            )
        on, bn = Int32(0), Int32(0)
        visible = cutlass.min(
            cutlass.max(Int32(lengths[row]), Int32(0)),
            cutlass.max(Int32(active[0]), Int32(0)),
        )
        newest = (visible - Int32(1)) // Int32(8) if visible > Int32(0) else Int32(-1)
        if cutlass.const_expr(not merge):
            if (row == Int32(0)) & (partition == Int32(0)) & (tx < Int32(2)):
                state[tx] = Int32(0)
            visible = cutlass.min(
                cutlass.max(Int32(lengths[row]), Int32(0)),
                cutlass.max(Int32(active[0]), Int32(0)),
            )
            extent = cutlass.min(visible, width)
            blocks = (extent + Int32(7)) // Int32(8)
            begin = (Int64(blocks) * Int64(partition) // Int64(self.ctas)) * Int64(8)
            end = cutlass.min(
                Int64(extent),
                (Int64(blocks) * Int64(partition + Int32(1)) // Int64(self.ctas))
                * Int64(8),
            )
            if cutlass.const_expr(self._use_fast_scorer):
                # Stage Q, Q scales, and packed BF16 weights into SMEM once per CTA.
                total_q_words = Int32(self.heads * 16)
                idx = tx
                while idx < total_q_words:
                    bbase = Int64(row) * Int64(self.heads * 64) + Int64(idx) * Int64(4)
                    q_smem[idx] = (
                        Uint32(q[bbase])
                        | (Uint32(q[bbase + Int64(1)]) << Uint32(8))
                        | (Uint32(q[bbase + Int64(2)]) << Uint32(16))
                        | (Uint32(q[bbase + Int64(3)]) << Uint32(24))
                    )
                    idx += Int32(1024)
                if tx < Int32(self.heads):
                    sbase = Int64(row) * Int64(self.heads * 4) + Int64(tx) * Int64(4)
                    qs_smem[tx] = (
                        Uint32(qs[sbase])
                        | (Uint32(qs[sbase + Int64(1)]) << Uint32(8))
                        | (Uint32(qs[sbase + Int64(2)]) << Uint32(16))
                        | (Uint32(qs[sbase + Int64(3)]) << Uint32(24))
                    )
                if tx < Int32(weight_pairs_per_row):
                    w0 = weights[Int64(row) * Int64(self.heads) + Int64(tx * Int32(2))]
                    w1 = BFloat16(0)
                    if cutlass.const_expr(self.heads != 1):
                        w1 = weights[
                            Int64(row) * Int64(self.heads)
                            + Int64(tx * Int32(2) + Int32(1))
                        ]
                    w_smem[tx] = _fused_bf16_pack_weights(w0, w1)
                scalars_addr = shared_ptr_to_u32(scalars.iterator)
                if tx == Int32(0):
                    _smem_st_i32(scalars_addr, Int32(4), Int32(0))
                    _smem_st_i32(scalars_addr, Int32(5), Int32(0))
                cute.arch.sync_threads()

                warp = tx // Int32(32)
                lane = tx % Int32(32)
                byte_off = Int64(lane % Int32(4)) * Int64(4)
                lt_mask = (Uint32(1) << Uint32(lane)) - Uint32(1)
                lane_is_out = (lane % Int32(4)) == Int32(0)
                lane_is_even = (lane % Int32(2)) == Int32(0)
                prow = Int64(row) * page_row_stride
                tok_cutoff = Uint32(0x007F)
                blk_cutoff = Uint32(0x007F)

                tile = begin
                while tile < end:
                    if on + Int32(512) > Int32(2048):
                        on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
                        # Compute exact token cutoff from surviving 512 items.
                        min_k = Uint32(0xFFFF)
                        if tx < on:
                            min_k = _bf16_bits_to_ordered(
                                f32_to_raw_bits(Float32(ov[tx])) >> Uint32(16)
                            )
                        for delta in cutlass.range_constexpr(5):
                            min_k = cutlass.min(
                                min_k,
                                cute.arch.shuffle_sync_bfly(min_k, offset=1 << delta),
                            )
                        if lane == Int32(0):
                            hist[warp] = Int32(min_k)
                        cute.arch.sync_threads()
                        if warp == Int32(0):
                            mk = Uint32(hist[lane])
                            for delta in cutlass.range_constexpr(5):
                                mk = cutlass.min(
                                    mk,
                                    cute.arch.shuffle_sync_bfly(mk, offset=1 << delta),
                                )
                            if lane == Int32(0):
                                _smem_st_i32(scalars_addr, Int32(6), Int32(mk))
                                _smem_st_i32(scalars_addr, Int32(4), on)
                        cute.arch.sync_threads()
                        tok_cutoff = Uint32(_smem_ld_i32(scalars_addr, Int32(6)))
                    if bn + Int32(64) > Int32(4096):
                        bn = _trim(
                            bv,
                            bi,
                            bn,
                            2048,
                            4096,
                            tx,
                            hist,
                            selected,
                            scalars,
                            priority_id=newest,
                        )
                        min_bk = Uint32(0xFFFF)
                        for slot in cutlass.range_constexpr(2):
                            idx_b = tx + Int32(slot * 1024)
                            if idx_b < bn:
                                bk = _bf16_bits_to_ordered(
                                    f32_to_raw_bits(Float32(bv[idx_b])) >> Uint32(16)
                                )
                                min_bk = cutlass.min(min_bk, bk)
                        for delta in cutlass.range_constexpr(5):
                            min_bk = cutlass.min(
                                min_bk,
                                cute.arch.shuffle_sync_bfly(min_bk, offset=1 << delta),
                            )
                        if lane == Int32(0):
                            hist[warp] = Int32(min_bk)
                        cute.arch.sync_threads()
                        if warp == Int32(0):
                            mk = Uint32(hist[lane])
                            for delta in cutlass.range_constexpr(5):
                                mk = cutlass.min(
                                    mk,
                                    cute.arch.shuffle_sync_bfly(mk, offset=1 << delta),
                                )
                            if lane == Int32(0):
                                _smem_st_i32(scalars_addr, Int32(7), Int32(mk))
                                _smem_st_i32(scalars_addr, Int32(5), bn)
                        cute.arch.sync_threads()
                        blk_cutoff = Uint32(_smem_ld_i32(scalars_addr, Int32(7)))

                    base_col = tile + Int64(warp) * Int64(16)
                    if base_col < end:
                        col0 = base_col + Int64(lane // Int32(4))
                        col1 = col0 + Int64(8)
                        pcol0 = base_col // Int64(self.page_size)
                        pcol1 = (base_col + Int64(8)) // Int64(self.page_size)
                        tok0 = col0 % Int64(self.page_size)
                        tok1 = col1 % Int64(self.page_size)

                        p0 = Int64(-1)
                        if pcol0 < Int64(page_width):
                            p0 = Int64(pages[prow + pcol0])
                        p0_valid = (p0 >= Int64(0)) & (p0 < pool_pages)
                        p1 = p0
                        if cutlass.const_expr(self.page_size == 8):
                            p1 = Int64(-1)
                            if pcol1 < Int64(page_width):
                                p1 = Int64(pages[prow + pcol1])
                        elif pcol1 != pcol0:
                            p1 = Int64(-1)
                            if pcol1 < Int64(page_width):
                                p1 = Int64(pages[prow + pcol1])
                        p1_valid = (p1 >= Int64(0)) & (p1 < pool_pages)

                        a0_0 = Uint32(0)
                        a2_0 = Uint32(0)
                        a0_1 = Uint32(0)
                        a2_1 = Uint32(0)
                        sfa0 = Uint32(0x7F7F7F7F)
                        if p0_valid:
                            kb0 = p0 * pool_stride + tok0 * Int64(64)
                            sb0 = (
                                p0 * pool_stride
                                + Int64(self.page_size * 64)
                                + tok0 * Int64(4)
                            )
                            a0_0 = _packed_word(pool, kb0 + byte_off)
                            a2_0 = _packed_word(pool, kb0 + byte_off + Int64(16))
                            a0_1 = _packed_word(pool, kb0 + byte_off + Int64(32))
                            a2_1 = _packed_word(pool, kb0 + byte_off + Int64(48))
                            if lane_is_even:
                                sfa0 = _packed_word(pool, sb0)

                        a1_0 = Uint32(0)
                        a3_0 = Uint32(0)
                        a1_1 = Uint32(0)
                        a3_1 = Uint32(0)
                        sfa1 = Uint32(0x7F7F7F7F)
                        if p1_valid:
                            kb1 = p1 * pool_stride + tok1 * Int64(64)
                            sb1 = (
                                p1 * pool_stride
                                + Int64(self.page_size * 64)
                                + tok1 * Int64(4)
                            )
                            a1_0 = _packed_word(pool, kb1 + byte_off)
                            a3_0 = _packed_word(pool, kb1 + byte_off + Int64(16))
                            a1_1 = _packed_word(pool, kb1 + byte_off + Int64(32))
                            a3_1 = _packed_word(pool, kb1 + byte_off + Int64(48))
                            if not lane_is_even:
                                sfa1 = _packed_word(pool, sb1)

                        sfa_full = sfa0 if lane_is_even else sfa1
                        sfa_k0 = sfa_full & Uint32(0xFFFF)
                        sfa_k1 = (sfa_full >> Uint32(16)) & Uint32(0xFFFF)

                        tot0 = Float32(0.0)
                        tot1 = Float32(0.0)
                        for head_tile in cutlass.range_constexpr((self.heads + 7) // 8):
                            head = Int32(head_tile * 8) + lane // Int32(4)
                            b0_0 = Uint32(0)
                            b1_0 = Uint32(0)
                            b0_1 = Uint32(0)
                            b1_1 = Uint32(0)
                            sfb_full = Uint32(0x7F7F7F7F)
                            if head < Int32(self.heads):
                                qwb = head * Int32(16) + (lane % Int32(4))
                                b0_0 = Uint32(q_smem[qwb])
                                b1_0 = Uint32(q_smem[qwb + Int32(4)])
                                b0_1 = Uint32(q_smem[qwb + Int32(8)])
                                b1_1 = Uint32(q_smem[qwb + Int32(12)])
                                sfb_full = Uint32(qs_smem[head])
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
                                sfa_k0,
                                sfb_full & Uint32(0xFFFF),
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
                                sfa_k1,
                                (sfb_full >> Uint32(16)) & Uint32(0xFFFF),
                            )
                            w_idx = Int32(head_tile * 4) + (lane % Int32(4))
                            w_word = Uint32(0)
                            if w_idx < Int32(weight_pairs_per_row):
                                w_word = Uint32(w_smem[w_idx])
                            p_lo = _fused_bf16_products_word(
                                d0, d1, w_word, fused_relu=True
                            )
                            p_hi = _fused_bf16_products_word(
                                d2, d3, w_word, fused_relu=True
                            )
                            for hp in cutlass.range_constexpr(
                                min(4, weight_pairs_per_row)
                            ):
                                src_lane = (lane // Int32(4)) * Int32(4) + Int32(hp)
                                pk0 = cute.arch.shuffle_sync(p_lo, src_lane)
                                pk1 = cute.arch.shuffle_sync(p_hi, src_lane)
                                if cutlass.const_expr(self.heads == 1):
                                    tot0 += u32_as_f32(pk0 << Uint32(16))
                                    tot1 += u32_as_f32(pk1 << Uint32(16))
                                else:
                                    tot0 = _fused_bf16_pair_accumulate(tot0, pk0)
                                    tot1 = _fused_bf16_pair_accumulate(tot1, pk1)

                        v0_ok = lane_is_out & p0_valid & (col0 < end)
                        v1_ok = lane_is_out & p1_valid & (col1 < end)
                        f0 = (
                            Float32(BFloat16(tot0)) if v0_ok else Float32(-float("inf"))
                        )
                        f1 = (
                            Float32(BFloat16(tot1)) if v1_ok else Float32(-float("inf"))
                        )
                        k0 = _bf16_bits_to_ordered(f32_to_raw_bits(f0) >> Uint32(16))
                        k1 = _bf16_bits_to_ordered(f32_to_raw_bits(f1) >> Uint32(16))
                        acc0 = v0_ok & (k0 > tok_cutoff)
                        acc1 = v1_ok & (k1 > tok_cutoff)
                        m0 = cute.arch.vote_ballot_sync(acc0)
                        m1 = cute.arch.vote_ballot_sync(acc1)
                        n0 = Int32(cute.arch.popc(m0))
                        n1 = Int32(cute.arch.popc(m1))
                        wn = n0 + n1
                        wbase = Int32(0)
                        if (lane == Int32(0)) & (wn > Int32(0)):
                            wbase = _smem_xadd_i32(scalars_addr, Int32(4), wn)
                        wbase = cute.arch.shuffle_sync(wbase, Int32(0))
                        if acc0:
                            slot = wbase + Int32(cute.arch.popc(m0 & lt_mask))
                            ov[slot] = f0
                            oi[slot] = Int32(col0)
                        if acc1:
                            slot = wbase + n0 + Int32(cute.arch.popc(m1 & lt_mask))
                            ov[slot] = f1
                            oi[slot] = Int32(col1)

                        # On-the-fly in-register 8-key block-max via 3 butterfly shuffles.
                        bm0 = f0
                        bm1 = f1
                        bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=4))
                        bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=8))
                        bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=16))
                        bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=4))
                        bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=8))
                        bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=16))
                        if lane == Int32(0):
                            blk0 = Int32(base_col // Int64(8))
                            is_new0 = (visible > Int32(0)) & (blk0 == newest)
                            if is_new0:
                                bm0 = Float32(float("inf"))
                            bk0 = (
                                Uint32(0xFFFF)
                                if is_new0
                                else _bf16_bits_to_ordered(
                                    f32_to_raw_bits(bm0) >> Uint32(16)
                                )
                            )
                            if bk0 > blk_cutoff:
                                bslot = _smem_xadd_i32(scalars_addr, Int32(5), Int32(1))
                                bv[bslot] = bm0
                                bi[bslot] = blk0
                            if base_col + Int64(8) < end:
                                blk1 = blk0 + Int32(1)
                                is_new1 = (visible > Int32(0)) & (blk1 == newest)
                                if is_new1:
                                    bm1 = Float32(float("inf"))
                                bk1 = (
                                    Uint32(0xFFFF)
                                    if is_new1
                                    else _bf16_bits_to_ordered(
                                        f32_to_raw_bits(bm1) >> Uint32(16)
                                    )
                                )
                                if bk1 > blk_cutoff:
                                    bslot = _smem_xadd_i32(
                                        scalars_addr, Int32(5), Int32(1)
                                    )
                                    bv[bslot] = bm1
                                    bi[bslot] = blk1
                    cute.arch.sync_threads()
                    on = _smem_ld_i32(scalars_addr, Int32(4))
                    bn = _smem_ld_i32(scalars_addr, Int32(5))
                    cute.arch.sync_threads()
                    tile += Int64(512)
            else:
                tile = Int32(begin)
                while Int64(tile) < end:
                    n = cutlass.min(Int32(512), Int32(end - Int64(tile)))
                    nb = (n + Int32(7)) // Int32(8)
                    if on + n > Int32(2048):
                        on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
                    if bn + nb > Int32(4096):
                        bn = _trim(
                            bv,
                            bi,
                            bn,
                            2048,
                            4096,
                            tx,
                            hist,
                            selected,
                            scalars,
                            priority_id=newest,
                        )
                    if tx < n:
                        ov[on + tx] = Float32(-float("inf"))
                        oi[on + tx] = Int32(-1)
                    cute.arch.sync_threads()
                    self.scorer._score_tile_fused(
                        q,
                        qs,
                        weights,
                        pool,
                        pages,
                        lengths,
                        active,
                        partial_indices,
                        partial_counts,
                        ov,
                        rows,
                        Int32(end),
                        page_width,
                        page_row_stride,
                        pool_stride,
                        pool_pages,
                        tile,
                        row_index=row,
                        local_output=True,
                        output_base=on,
                        logical_indices=oi,
                    )
                    cute.arch.sync_threads()
                    if tx < nb:
                        value = Float32(-float("inf"))
                        for offset in cutlass.range_constexpr(8):
                            index = tx * Int32(8) + Int32(offset)
                            if index < n:
                                value = fmax_f32(value, Float32(ov[on + index]))
                        block = tile // Int32(8) + tx
                        if (visible > Int32(0)) & (
                            block == (visible - Int32(1)) // Int32(8)
                        ):
                            value = Float32(float("inf"))
                        bv[bn + tx] = value
                        bi[bn + tx] = block
                    cute.arch.sync_threads()
                    on += n
                    bn += nb
                    tile += n
        else:
            for part in cutlass.range(self.ctas):
                base = (Int64(row) * Int64(self.ctas) + Int64(part)) * Int64(2560)
                counts = (Int64(row) * Int64(self.ctas) + Int64(part)) * Int64(2)
                no = Int32(partial_counts[counts])
                if on + no > Int32(2048):
                    on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
                if tx < no:
                    ov[on + tx] = Float32(partial_values[base + Int64(tx)])
                    oi[on + tx] = Int32(partial_indices[base + Int64(tx)])
                on += no
                block_count = Int32(partial_counts[counts + Int64(1)])
                offset = Int32(0)
                while offset < block_count:
                    n = cutlass.min(Int32(512), block_count - offset)
                    if bn + n > Int32(4096):
                        bn = _trim(
                            bv,
                            bi,
                            bn,
                            2048,
                            4096,
                            tx,
                            hist,
                            selected,
                            scalars,
                            priority_id=newest,
                        )
                    if tx < n:
                        source = base + Int64(512) + Int64(offset) + Int64(tx)
                        bv[bn + tx] = Float32(partial_values[source])
                        bi[bn + tx] = Int32(partial_indices[source])
                    bn += n
                    offset += n
                    cute.arch.sync_threads()
                cute.arch.sync_threads()
        on = _trim(ov, oi, on, 512, 2048, tx, hist, selected, scalars)
        bn = _trim(
            bv, bi, bn, 2048, 4096, tx, hist, selected, scalars, priority_id=newest
        )
        if cutlass.const_expr(not merge and self.ctas > 1):
            base = (Int64(row) * Int64(self.ctas) + Int64(partition)) * Int64(2560)
            if tx < on:
                partial_values[base + Int64(tx)] = Float32(ov[tx])
                partial_indices[base + Int64(tx)] = Int32(oi[tx])
            for slot in cutlass.range_constexpr(2):
                i = tx + Int32(slot * 1024)
                if i < bn:
                    partial_values[base + Int64(512) + Int64(i)] = Float32(bv[i])
                    partial_indices[base + Int64(512) + Int64(i)] = Int32(bi[i])
            if tx == Int32(0):
                count_base = (Int64(row) * Int64(self.ctas) + Int64(partition)) * Int64(
                    2
                )
                partial_counts[count_base] = on
                partial_counts[count_base + Int64(1)] = bn
        else:
            for slot in cutlass.range_constexpr(2):
                i = tx + Int32(slot * 1024)
                valid = i < bn
                value, index = Float32(-float("inf")), Int32(-1)
                if valid:
                    value, index = Float32(bv[i]), Int32(bi[i])
                base = Int64(row) * Int64(2048) + Int64(i)
                block_values[base] = value
                block_indices[base] = index
            _mxfp4_order_small_output(out_indices, out_values, row, oi, ov, tx, on)


def multiquery_hierarchical_scratch_shapes(
    rows: int, ctas: int, max_blocks: int
) -> dict:
    """Return dtype/shape pairs for bounded multi-query hierarchical scratch."""
    if rows < 1 or ctas < 1 or max_blocks < 1:
        raise ValueError(
            "multiquery hierarchical rows, CTAs, and blocks must be positive"
        )
    partial_shape = (rows, ctas, 512) if ctas > 1 else (1,)
    count_shape = (rows, ctas) if ctas > 1 else (1,)
    return {
        "selected_values": (torch.float32, (rows, 512)),
        "partial_indices": (torch.int32, partial_shape),
        "partial_values": (torch.int16, partial_shape),
        "partial_counts": (torch.int32, count_shape),
        "block_values_u16": (torch.int16, (rows, max_blocks)),
        "state": (torch.int32, (2,)),
    }


@cute.jit
def _bitonic_sort_2048_and_expand(
    out_blocks,
    sel_count: Int32,
    visible: Int32,
    row: Int32,
    candidate_output,
    candidate_output_lengths,
    hist,
    tx: Int32,
    direct_valid: cutlass.Constexpr[bool] = False,
):
    """Sort up to 2048 selected block IDs ascending and expand 8x coalesced."""
    p0 = Int32(2147483647)
    p1 = Int32(2147483647)
    i0 = tx * Int32(2)
    i1 = i0 + Int32(1)
    if cutlass.const_expr(direct_valid):
        p0 = Int32(out_blocks[i0])
        p1 = Int32(out_blocks[i1])
    else:
        if i0 < sel_count:
            p0 = Int32(out_blocks[i0])
        if i1 < sel_count:
            p1 = Int32(out_blocks[i1])
    cute.arch.sync_threads()

    for level in cutlass.range_constexpr(1, 12):
        for stage in cutlass.range_constexpr(level):
            stride = 1 << (level - 1 - stage)
            o0 = Int32(2147483647)
            o1 = Int32(2147483647)
            if cutlass.const_expr(stride == 1):
                o0 = p1
                o1 = p0
            elif cutlass.const_expr(stride < 64):
                o0 = cute.arch.shuffle_sync_bfly(p0, offset=stride // 2)
                o1 = cute.arch.shuffle_sync_bfly(p1, offset=stride // 2)
            else:
                out_blocks[i0] = p0
                out_blocks[i1] = p1
                cute.arch.sync_threads()
                partner = tx ^ Int32(stride // 2)
                o0 = Int32(out_blocks[partner * Int32(2)])
                o1 = Int32(out_blocks[partner * Int32(2) + Int32(1)])
                cute.arch.sync_threads()
            asc0 = (i0 & Int32(1 << level)) == Int32(0)
            low0 = (i0 & Int32(stride)) == Int32(0)
            tmin0 = asc0 == low0
            if ((p0 > o0) & tmin0) | ((p0 < o0) & (not tmin0)):
                p0 = o0
            asc1 = (i1 & Int32(1 << level)) == Int32(0)
            low1 = (i1 & Int32(stride)) == Int32(0)
            tmin1 = asc1 == low1
            if ((p1 > o1) & tmin1) | ((p1 < o1) & (not tmin1)):
                p1 = o1

    # Stage sorted 2048 block IDs back to SMEM for 100% coalesced 16384-int32 writes.
    out_blocks[i0] = p0
    out_blocks[i1] = p1
    local_count = Int32(0)
    vis64 = Int64(visible)
    if p0 != Int32(2147483647):
        rem0 = vis64 - Int64(p0) * Int64(8)
        if rem0 > Int64(0):
            local_count += Int32(cutlass.min(Int64(8), rem0))
    if p1 != Int32(2147483647):
        rem1 = vis64 - Int64(p1) * Int64(8)
        if rem1 > Int64(0):
            local_count += Int32(cutlass.min(Int64(8), rem1))
    for delta in cutlass.range_constexpr(5):
        local_count += cute.arch.shuffle_sync_bfly(local_count, offset=1 << delta)
    warp = tx // Int32(32)
    lane = tx % Int32(32)
    if lane == Int32(0):
        hist[warp] = local_count
    cute.arch.sync_threads()

    if warp == Int32(0):
        total = Int32(hist[lane])
        for delta in cutlass.range_constexpr(5):
            total += cute.arch.shuffle_sync_bfly(total, offset=1 << delta)
        if lane == Int32(0):
            candidate_output_lengths[row] = total

    # Coalesced expansion: consecutive threads write consecutive int32 addresses.
    row_base = Int64(row) * Int64(16384)
    for k in cutlass.range_constexpr(16):
        s = tx + Int32(k * 1024)
        blk = Int32(out_blocks[s >> Int32(3)])
        val = Int32(-1)
        if blk != Int32(2147483647):
            tok = Int64(blk) * Int64(8) + Int64(s & Int32(7))
            if tok < vis64:
                val = Int32(tok)
        candidate_output[row_base + Int64(s)] = val


class MultiQueryHierarchicalMXFP4:
    """Request-grouped multi-query Layer-20 hierarchical MXFP4 indexer.

    Combines shared K-page loading across up to ``q_tile`` queries with
    on-the-fly in-register 8-key block-max computation, compact BF16 uint16
    block-max streaming, and fused merge + 2048-block sort + 8x expansion.
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
            raise ValueError("invalid multiquery hierarchical CTA count")
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
        block_values_u16: cute.Pointer,
        state: cute.Pointer,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        block_row_stride: Int64,
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
            _flat(block_values_u16),
            _flat(state),
            rows,
            width,
            page_width,
            page_row_stride,
            pool_stride,
            pool_pages,
            block_row_stride,
            request_mode,
            num_requests,
        ).launch(
            grid=(num_groups * Int64(self.ctas), 1, 1),
            block=(1024, 1, 1),
            stream=stream,
        )

    @cute.jit
    def merge_expand(
        self,
        lengths: cute.Pointer,
        active: cute.Pointer,
        out_indices: cute.Pointer,
        out_values: cute.Pointer,
        partial_indices: cute.Pointer,
        partial_values: cute.Pointer,
        partial_counts: cute.Pointer,
        block_values_u16: cute.Pointer,
        candidate_output: cute.Pointer,
        candidate_output_lengths: cute.Pointer,
        rows: Int32,
        width: Int32,
        block_row_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.merge_expand_kernel(
            _flat(lengths),
            _flat(active),
            _flat(out_indices),
            _flat(out_values),
            _flat(partial_indices),
            _flat(partial_values),
            _flat(partial_counts),
            _flat(block_values_u16),
            _flat(candidate_output),
            _flat(candidate_output_lengths),
            rows,
            width,
            block_row_stride,
        ).launch(
            grid=(Int64(rows), 1, 1),
            block=(1024, 1, 1),
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
        block_values_u16,
        state,
        rows: Int32,
        width: Int32,
        page_width: Int32,
        page_row_stride: Int64,
        pool_stride: Int64,
        pool_pages: Int64,
        block_row_stride: Int64,
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
            state[tx] = Int32(0)

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
        blk_smem = smem.allocate_tensor(
            Uint16, cute.make_layout((self.q_tile * 64,)), byte_alignment=16
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
        q_newest = smem.allocate_tensor(
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
            newest = Int32(-1)
            prow = Int32(0)
            if r < rows:
                vis = cutlass.min(cutlass.max(Int32(lengths[r]), Int32(0)), active_val)
                ext = cutlass.min(vis, width)
                newest = (vis - Int32(1)) // Int32(8) if vis > Int32(0) else Int32(-1)
                prow = _resolve_request_row(r, request_meta, request_mode, num_requests)
            q_extents[tx] = ext
            q_newest[tx] = newest
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

                    # On-the-fly in-register 8-key block-max staged for coalesced write.
                    bm0 = f0
                    bm1 = f1
                    bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=4))
                    bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=8))
                    bm0 = fmax_f32(bm0, cute.arch.shuffle_sync_bfly(bm0, offset=16))
                    bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=4))
                    bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=8))
                    bm1 = fmax_f32(bm1, cute.arch.shuffle_sync_bfly(bm1, offset=16))
                    if lane == Int32(0):
                        blk0 = base_col >> Int32(3)
                        newest_q = Int32(q_newest[ql])
                        if blk0 == newest_q:
                            bm0 = Float32(float("inf"))
                        blk_smem[(ql << Int32(6)) + (warp << Int32(1))] = Uint16(
                            f32_to_raw_bits(bm0) >> Uint32(16)
                        )
                        if (base_col + Int32(8) < ext_q) & (base_col + Int32(8) < end):
                            if blk0 + Int32(1) == newest_q:
                                bm1 = Float32(float("inf"))
                            blk_smem[
                                (ql << Int32(6)) + (warp << Int32(1)) + Int32(1)
                            ] = Uint16(f32_to_raw_bits(bm1) >> Uint32(16))

            cute.arch.sync_threads()
            if tx < Int32(self.q_tile * 64):
                ql_b = tx >> Int32(6)
                b_off = tx & Int32(63)
                blk_id = (tile >> Int32(3)) + b_off
                ext_b = Int32(q_extents[ql_b])
                if ((blk_id << Int32(3)) < ext_b) & ((blk_id << Int32(3)) < end):
                    r_b = row_base + ql_b
                    block_values_u16[Int64(r_b) * block_row_stride + Int64(blk_id)] = (
                        Uint16(blk_smem[tx])
                    )
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
    def merge_expand_kernel(
        self,
        lengths,
        active,
        out_indices,
        out_values,
        partial_indices,
        partial_values,
        partial_counts,
        block_values_u16,
        candidate_output,
        candidate_output_lengths,
        rows: Int32,
        width: Int32,
        block_row_stride: Int64,
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
            Int32, cute.make_layout((2048,)), byte_alignment=16
        )
        bvals_smem = smem.allocate_tensor(
            Uint16, cute.make_layout((16384,)), byte_alignment=16
        )
        scalars = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=16)
        scalars_addr = shared_ptr_to_u32(scalars.iterator)

        # Phase 1: merge cross-CTA token top-512 partials (if ctas > 1),
        # processing two partitions per step across all 1024 threads.
        if cutlass.const_expr(self.ctas > 1):
            if tx == Int32(0):
                _smem_st_i32(scalars_addr, Int32(4), Int32(0))
            cute.arch.sync_threads()
            cutoff = Uint32(0x007F)
            half = tx >> Int32(9)
            t512 = tx & Int32(511)
            for pair in cutlass.range((self.ctas + 1) // 2):
                part0 = pair * Int32(2)
                part1 = part0 + Int32(1)
                cnt0 = Int32(
                    partial_counts[Int64(row) * Int64(self.ctas) + Int64(part0)]
                )
                cnt1 = Int32(0)
                if part1 < Int32(self.ctas):
                    cnt1 = Int32(
                        partial_counts[Int64(row) * Int64(self.ctas) + Int64(part1)]
                    )
                on = _smem_ld_i32(scalars_addr, Int32(4))
                if on + cnt0 + cnt1 > Int32(2048):
                    on, cutoff = _trim_smem_u16_512(
                        carry_val,
                        carry_idx,
                        Int32(0),
                        on,
                        tx,
                        hist,
                        out_slots,
                        scalars,
                        threads=1024,
                    )
                    if tx == Int32(0):
                        _smem_st_i32(scalars_addr, Int32(4), on)
                cute.arch.sync_threads()
                if cnt0 + cnt1 > Int32(0):
                    my_part = part0 + half
                    my_cnt = cnt0 if half == Int32(0) else cnt1
                    u = Uint32(0xFF80)
                    idx_v = Int32(-1)
                    acc = False
                    if t512 < my_cnt:
                        pbase = (
                            Int64(row) * Int64(self.ctas) + Int64(my_part)
                        ) * Int64(512)
                        u = Uint32(partial_values[pbase + Int64(t512)])
                        idx_v = Int32(partial_indices[pbase + Int64(t512)])
                        acc = _bf16_bits_to_ordered(u) > cutoff
                    mask = cute.arch.vote_ballot_sync(acc)
                    wn = Int32(cute.arch.popc(mask))
                    if wn > Int32(0):
                        wbase = Int32(0)
                        if lane == Int32(0):
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
                    threads=1024,
                )
            _bitonic_sort_512_u16_to_out(
                out_indices, out_values, row, carry_idx, carry_val, Int32(0), on, tx
            )
            cute.arch.sync_threads()

        # Phase 2: select top-2048 valid blocks, bitonic sort, and expand 8x coalesced.
        visible = cutlass.min(
            cutlass.max(Int32(lengths[row]), Int32(0)),
            cutlass.max(Int32(active[0]), Int32(0)),
        )
        extent = cutlass.min(visible, width)
        num_blocks = (extent + Int32(7)) // Int32(8)
        newest = (visible - Int32(1)) // Int32(8) if visible > Int32(0) else Int32(-1)
        if num_blocks <= Int32(0):
            row_base = Int64(row) * Int64(16384)
            for slot in cutlass.range_constexpr(16):
                candidate_output[row_base + Int64(tx + Int32(slot * 1024))] = Int32(-1)
            if tx == Int32(0):
                candidate_output_lengths[row] = Int32(0)
        elif num_blocks <= Int32(2048):
            # All valid blocks fit in top-2048 and are already in ascending order:
            # compact directly via warp ballot prefix sum (bypassing radix + bitonic sort).
            brow = Int64(row) * block_row_stride
            b0 = tx * Int32(2)
            b1 = b0 + Int32(1)
            ok0 = False
            ok1 = False
            if b0 < num_blocks:
                u0 = Uint32(block_values_u16[brow + Int64(b0)])
                ok0 = (u0 != Uint32(0xFF80)) | (b0 == newest)
            if b1 < num_blocks:
                u1 = Uint32(block_values_u16[brow + Int64(b1)])
                ok1 = (u1 != Uint32(0xFF80)) | (b1 == newest)
            m0 = cute.arch.vote_ballot_sync(ok0)
            m1 = cute.arch.vote_ballot_sync(ok1)
            warp = tx >> Int32(5)
            w_tot = Int32(cute.arch.popc(m0)) + Int32(cute.arch.popc(m1))
            vis64 = Int64(visible)
            local_tok = Int32(0)
            if ok0:
                rem0 = vis64 - Int64(b0) * Int64(8)
                if rem0 > Int64(0):
                    local_tok += Int32(cutlass.min(Int64(8), rem0))
            if ok1:
                rem1 = vis64 - Int64(b1) * Int64(8)
                if rem1 > Int64(0):
                    local_tok += Int32(cutlass.min(Int64(8), rem1))
            for delta in cutlass.range_constexpr(5):
                local_tok += cute.arch.shuffle_sync_bfly(local_tok, offset=1 << delta)
            if lane == Int32(0):
                hist[warp] = w_tot
                hist[Int32(32) + warp] = local_tok
            out_slots[b0] = Int32(2147483647)
            out_slots[b1] = Int32(2147483647)
            cute.arch.sync_threads()
            if tx == Int32(0):
                acc_b = Int32(0)
                acc_t = Int32(0)
                for w in cutlass.range_constexpr(32):
                    wb = Int32(hist[w])
                    hist[w] = acc_b
                    acc_b += wb
                    acc_t += Int32(hist[Int32(32) + w])
                candidate_output_lengths[row] = acc_t
            cute.arch.sync_threads()
            dst = (
                Int32(hist[warp])
                + Int32(cute.arch.popc(m0 & lt_mask))
                + Int32(cute.arch.popc(m1 & lt_mask))
            )
            if ok0:
                out_slots[dst] = b0
                dst += Int32(1)
            if ok1:
                out_slots[dst] = b1
            cute.arch.sync_threads()
            row_base = Int64(row) * Int64(16384)
            for k in cutlass.range_constexpr(16):
                s = tx + Int32(k * 1024)
                blk = Int32(out_slots[s >> Int32(3)])
                val = Int32(-1)
                if blk != Int32(2147483647):
                    tok = Int64(blk) * Int64(8) + Int64(s & Int32(7))
                    if tok < vis64:
                        val = Int32(tok)
                candidate_output[row_base + Int64(s)] = val
        elif num_blocks <= Int32(16384):
            # Stage compact uint16 block scores once into SMEM (<= 32 KiB).
            brow = Int64(row) * block_row_stride
            b = tx
            while b < num_blocks:
                bvals_smem[b] = Uint16(block_values_u16[brow + Int64(b)])
                b += Int32(1024)
            cute.arch.sync_threads()
            sel_count, _ = _radix_select_u16_to_slots(
                bvals_smem,
                Int64(0),
                num_blocks,
                2048,
                tx,
                hist,
                out_slots,
                scalars,
                threads=1024,
                priority_index=newest,
            )
            _bitonic_sort_2048_and_expand(
                out_slots,
                sel_count,
                visible,
                row,
                candidate_output,
                candidate_output_lengths,
                hist,
                tx,
            )
        else:
            sel_count, _ = _radix_select_u16_to_slots(
                block_values_u16,
                Int64(row) * block_row_stride,
                num_blocks,
                2048,
                tx,
                hist,
                out_slots,
                scalars,
                threads=1024,
                priority_index=newest,
            )
            _bitonic_sort_2048_and_expand(
                out_slots,
                sel_count,
                visible,
                row,
                candidate_output,
                candidate_output_lengths,
                hist,
                tx,
            )
