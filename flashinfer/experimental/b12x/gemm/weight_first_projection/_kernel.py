# Copyright (c) 2025 by the b12x authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage the BF16 weights of two projections before waiting for their shared activation.

Each CTA copies one brick from the two adjacent logical output ranges. A brick
may cross the boundary between the independent weight allocations. The split-K
reduction uses FP32 partials in a fixed order. The first result stays FP32, and
the second rounds once to BF16.
"""

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int64
from cutlass.cutlass_dsl import Int32

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics.atomics import (
    atomic_add_global_acq_rel_i32,
    ld_shared_f32,
    ld_shared_i32,
    st_global_i32,
    st_shared_f32,
    st_shared_i32,
)
from b12x._lib.intrinsics.math import cvt_f32_to_bf16_bits
from b12x._lib.intrinsics.mma import bf16_mma_m16n8k16_f32
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    ld_global_cg_f32,
    ldmatrix_m8n8x2_b16,
    ldmatrix_m8n8x4_b16,
    shared_ptr_to_u32,
    st_global_f32,
    st_global_u16,
)
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr

_THREADS = 256
_WARPS = _THREADS // 32


def brick_for(n: int, k: int) -> tuple[int, int]:
    """Brick shape for a weight ``(n, k)``: the widest column count that
    divides ``k`` with enough rows that one CTA stays under the 99 KB
    shared-memory limit and enough CTAs stream the weight in the window."""
    if k % 768 == 0 and n >= 512:
        return 48, 768
    if k % 768 == 0:
        return 32, 768
    if k % 512 == 0:
        return 64, 512
    if k % 256 == 0:
        return 128, 256
    raise ValueError(f"no weight-first brick for N={n}, K={k}")


def smem_bytes(nt: int, kt: int) -> int:
    """Shared memory of one CTA: the padded brick plus a padded 16-row slice
    of the activation, followed by a 16-byte control word."""
    return (nt + 16) * (kt + 8) * 2 + 16


class MixedWeightFirstProjection:
    """Each CTA stages an output tile before synchronizing with the activation producer."""

    def __init__(self, n: int, k: int, nt: int, kt: int):
        if k % kt != 0:
            raise ValueError(f"K={k} is not a multiple of the brick width {kt}")
        self.n = int(n)
        self.k = int(k)
        self.nt = int(nt)
        self.kt = int(kt)
        self.kt8 = kt // 8  # 16-byte chunks per brick row
        self.ld8 = self.kt8 + 1  # padded row stride in 16-byte chunks
        self.ld = self.ld8 * 8  # padded row stride in elements
        self.k8 = k // 8  # 16-byte chunks per weight row
        self.ntiles = nt // 8  # 8-column MMA tiles per brick row block
        self.kw = kt // _WARPS  # columns per warp
        self.ksteps = self.kw // 16
        self.splits = k // kt
        self.grid_n = (n + nt - 1) // nt
        self.opt = (16 * nt + _THREADS - 1) // _THREADS
        self.x_offset = nt * self.ld * 2  # bytes: start of the x slice
        self.control_offset = (nt + 16) * self.ld * 2  # bytes: last-CTA flag
        self.smem_bytes = smem_bytes(nt, kt)
        assert self.kw % 16 == 0
        assert _WARPS * 16 * nt * 4 <= nt * self.ld * 2

    @cute.jit
    def store_result(
        self, value, row, col, first_rows, second_rows, first_out, second_out
    ):
        if col < first_rows:
            offset = Int64(row) * Int64(first_rows) + Int64(col)
            st_global_f32(first_out + offset * Int64(4), value)
        else:
            offset = Int64(row) * Int64(second_rows) + Int64(col - first_rows)
            st_global_u16(second_out + offset * Int64(2), cvt_f32_to_bf16_bits(value))

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        first_ptr: cute.Pointer,
        second_ptr: cute.Pointer,
        y0_ptr: cute.Pointer,
        y1_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        m_rows: Int32,
        n0: Int32,
        depth: Int32,
        use_pdl: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x_ptr,
            first_ptr,
            second_ptr,
            y0_ptr,
            y1_ptr,
            partial_ptr,
            counters_ptr,
            m_rows,
            n0,
            depth,
        ).launch(
            grid=(self.grid_n, self.splits, 1),
            block=[_THREADS, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        first_ptr: cute.Pointer,
        second_ptr: cute.Pointer,
        y0_ptr: cute.Pointer,
        y1_ptr: cute.Pointer,
        partial_ptr: cute.Pointer,
        counters_ptr: cute.Pointer,
        m_rows: Int32,
        n0: Int32,
        depth: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        n_tile = Int32(bidx)
        split = Int32(bidy)
        tile_row0 = n_tile * Int32(self.nt)
        k0_chunk = split * Int32(self.kt8)

        smem = cutlass.utils.SmemAllocator()

        words = smem.allocate_tensor(
            element_type=cutlass.Uint32,
            layout=cute.make_layout((self.smem_bytes // 4,)),
            byte_alignment=128,
        )
        sw = shared_ptr_to_u32(words.iterator)
        sx = sw + Int32(self.x_offset)
        s_control = sw + Int32(self.control_offset)

        rows = Int32(self.nt)
        if Int32(self.n) - tile_row0 < Int32(self.nt):
            rows = Int32(self.n) - tile_row0

        # 1. Stage the brick: groups of 256 copies of 16 bytes; `depth`
        #    groups may remain in flight per CTA (0: unbounded), which bounds
        #    the request pressure this kernel puts beside the allreduce's
        #    peer reads of this GPU's memory.
        first_addr = Int64(first_ptr.toint())
        second_addr = Int64(second_ptr.toint()) - Int64(n0) * Int64(self.k * 2)
        brick_chunk0 = Int64(tile_row0) * Int64(self.k8) + Int64(k0_chunk)
        total = rows * Int32(self.kt8)
        base = Int32(0)
        while base < total:
            i = base + tid
            if i < total:
                r = i // Int32(self.kt8)
                c = i - r * Int32(self.kt8)
                base_addr = first_addr if tile_row0 + r < n0 else second_addr
                gmem = base_addr + (
                    brick_chunk0 + Int64(r) * Int64(self.k8) + Int64(c)
                ) * Int64(16)
                cp_async4_shared_global(
                    sw + (r * Int32(self.ld8) + c) * Int32(16), gmem
                )
            cute.arch.cp_async_commit_group()
            if depth == Int32(1):
                cute.arch.cp_async_wait_group(1)
            elif depth == Int32(2):
                cute.arch.cp_async_wait_group(2)
            elif depth == Int32(3):
                cute.arch.cp_async_wait_group(3)
            elif depth == Int32(4):
                cute.arch.cp_async_wait_group(4)
            elif depth == Int32(6):
                cute.arch.cp_async_wait_group(6)
            elif depth == Int32(8):
                cute.arch.cp_async_wait_group(8)
            base += Int32(_THREADS)

        # 2. The activation is produced by the kernel before this one.
        cute.arch.griddepcontrol_wait()
        x_addr = Int64(x_ptr.toint())
        total_x = m_rows * Int32(self.kt8)
        i = tid
        while i < total_x:
            m = i // Int32(self.kt8)
            c = i - m * Int32(self.kt8)
            gmem = x_addr + (
                Int64(m) * Int64(self.k8) + Int64(k0_chunk) + Int64(c)
            ) * Int64(16)
            cp_async4_shared_global(sx + (m * Int32(self.ld8) + c) * Int32(16), gmem)
            i += Int32(_THREADS)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 3. Tensor-core products. ldmatrix x4 loads the 16x16 A tile of x
        #    (lanes 0-15 address rows 0-15 at column 0, lanes 16-31 the same
        #    rows at column 8); ldmatrix x2 loads the 8x16 B tile of the
        #    brick (lanes 0-7 rows at column 0, lanes 8-15 at column 8).
        kw0 = warp * Int32(self.kw)
        a_base = sx + (
            (lane & Int32(15)) * Int32(self.ld) + kw0 + (lane >> Int32(4)) * Int32(8)
        ) * Int32(2)
        b_base = sw + (
            (lane & Int32(7)) * Int32(self.ld)
            + kw0
            + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
        ) * Int32(2)
        acc = [
            [
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
                cutlass.Float32(0.0),
            ]
            for _ in range(self.ntiles)
        ]
        for ks in cutlass.range_constexpr(self.ksteps):
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(a_base + Int32(ks * 32))
            for t in cutlass.range_constexpr(self.ntiles):
                b0, b1 = ldmatrix_m8n8x2_b16(
                    b_base + Int32(t * 8 * self.ld * 2 + ks * 32)
                )
                d0, d1, d2, d3 = bf16_mma_m16n8k16_f32(
                    acc[t][0], acc[t][1], acc[t][2], acc[t][3], a0, a1, a2, a3, b0, b1
                )
                acc[t][0] = d0
                acc[t][1] = d1
                acc[t][2] = d2
                acc[t][3] = d3
        cute.arch.sync_threads()

        # 4. Warp tiles into shared memory (the brick region is free now):
        #    red[warp][16 rows][NT columns] fp32.
        row = lane >> Int32(2)
        col = (lane & Int32(3)) * Int32(2)
        red = sw
        dst = red + (warp * Int32(16 * self.nt)) * Int32(4)
        for t in cutlass.range_constexpr(self.ntiles):
            c0 = t * 8
            st_shared_f32(
                dst + (row * Int32(self.nt) + col + Int32(c0)) * Int32(4), acc[t][0]
            )
            st_shared_f32(
                dst + (row * Int32(self.nt) + col + Int32(c0 + 1)) * Int32(4), acc[t][1]
            )
            st_shared_f32(
                dst + ((row + Int32(8)) * Int32(self.nt) + col + Int32(c0)) * Int32(4),
                acc[t][2],
            )
            st_shared_f32(
                dst
                + ((row + Int32(8)) * Int32(self.nt) + col + Int32(c0 + 1)) * Int32(4),
                acc[t][3],
            )
        cute.arch.sync_threads()

        # 5. Sum the eight warp tiles in warp order; thread `tid` owns the
        #    outputs i = tid + j * 256 of the M x rows brick outputs.
        n_out = m_rows * rows
        y0_addr = Int64(y0_ptr.toint())
        y1_addr = Int64(y1_ptr.toint())
        n1 = Int32(self.n) - n0
        vals = [cutlass.Float32(0.0) for _ in range(self.opt)]
        for j in cutlass.range_constexpr(self.opt):
            i = tid + Int32(j * _THREADS)
            v = cutlass.Float32(0.0)
            if i < n_out:
                m = i // rows
                r = i - m * rows
                for wr in cutlass.range_constexpr(_WARPS):
                    v = v + ld_shared_f32(
                        red + ((Int32(wr * 16) + m) * Int32(self.nt) + r) * Int32(4)
                    )
            vals[j] = v

        if cutlass.const_expr(self.splits == 1):
            for j in cutlass.range_constexpr(self.opt):
                i = tid + Int32(j * _THREADS)
                if i < n_out:
                    m = i // rows
                    r = i - m * rows
                    n = tile_row0 + r
                    self.store_result(vals[j], m, n, n0, n1, y0_addr, y1_addr)
        else:
            partial_addr = Int64(partial_ptr.toint())
            split_stride = Int64(16 * self.n)  # fp32 elements per split
            out_partial = partial_addr + Int64(split) * split_stride * Int64(4)
            for j in cutlass.range_constexpr(self.opt):
                i = tid + Int32(j * _THREADS)
                if i < n_out:
                    m = i // rows
                    r = i - m * rows
                    st_global_f32(
                        out_partial
                        + (Int64(m) * Int64(self.n) + Int64(tile_row0) + Int64(r))
                        * Int64(4),
                        vals[j],
                    )
            # Last CTA of this n-tile (release/acquire counter; reset below
            # so graph replay needs no host fill) sums the S partials in
            # split order.
            cute.arch.sync_threads()
            counter_addr = Int64(counters_ptr.toint()) + Int64(n_tile) * Int64(4)
            if tid == Int32(0):
                old = atomic_add_global_acq_rel_i32(counter_addr, Int32(1))
                flag = Int32(0)
                if old == Int32(self.splits - 1):
                    flag = Int32(1)
                st_shared_i32(s_control, flag)
            cute.arch.sync_threads()
            is_last = ld_shared_i32(s_control)
            if is_last != Int32(0):
                cute.arch.fence_acq_rel_gpu()
                for j in cutlass.range_constexpr(self.opt):
                    i = tid + Int32(j * _THREADS)
                    if i < n_out:
                        m = i // rows
                        r = i - m * rows
                        src = partial_addr + (
                            Int64(m) * Int64(self.n) + Int64(tile_row0) + Int64(r)
                        ) * Int64(4)
                        acc4 = [
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                            cutlass.Float32(0.0),
                        ]
                        for quad in cutlass.range_constexpr(self.splits // 4):
                            for q in cutlass.range_constexpr(4):
                                acc4[q] = acc4[q] + ld_global_cg_f32(
                                    src + Int64(quad * 4 + q) * split_stride * Int64(4)
                                )
                        v = cutlass.Float32(0.0)
                        for s in cutlass.range_constexpr(
                            (self.splits // 4) * 4, self.splits
                        ):
                            v = v + ld_global_cg_f32(
                                src + Int64(s) * split_stride * Int64(4)
                            )
                        v = ((acc4[0] + acc4[1]) + (acc4[2] + acc4[3])) + v
                        n = tile_row0 + r
                        self.store_result(v, m, n, n0, n1, y0_addr, y1_addr)
                if tid == Int32(0):
                    st_global_i32(counter_addr, Int32(0))


def _dummy(dtype, alignment: int):
    return make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=alignment)


def compile_projection(first_rows, second_rows, width):
    rows = first_rows + second_rows
    nt, kt = brick_for(rows, width)
    kernel = MixedWeightFirstProjection(rows, width, nt, kt)
    types = (
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Int32,
    )
    return b12x_compile(
        kernel,
        *(_dummy(dtype, 16) for dtype in types),
        1,
        first_rows,
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.weight_first_projection", 2, (rows, width, nt, kt)
        ),
    )


def run_projection(
    compiled, source, first, second, logits, shared, partial, counters, depth, pdl
):
    tensors = (source, first, second, logits, shared, partial, counters)
    types = (
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.BFloat16,
        cutlass.Float32,
        cutlass.Int32,
    )
    pointers = [
        make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        for tensor, dtype in zip(tensors, types, strict=True)
    ]
    compiled(
        *pointers,
        source.shape[0],
        first.shape[0],
        depth,
        int(pdl),
        current_cuda_stream(),
    )
