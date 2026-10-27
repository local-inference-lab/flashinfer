# Copyright 2026 Fraser Price
# Copyright (c) 2025-2026 by the b12x authors.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.

"""Eight-CTA cluster merge over ordinary BF16 sparse-MLA split partials.

The cluster barrier and per-CTA output slicing follow Fraser BRRR's sparse
attention launch. QK, softmax, PV, and normalized BF16 partial writeback use
the shared sparse-MLA body so their existing rounding boundaries are retained.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_cg_v4_u32
from b12x._lib.intrinsics.math import fmax_f32
from .decode_math import _exp2_approx_ftz_f32, _u32_to_f32
from .kernel import UnifiedDecodeKernel


@dsl_user_op
def _cluster_sync(*, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [],
        "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _load_lse(address: Int64, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.f32(),
        [Int64(address).ir_value(loc=loc, ip=ip)],
        "ld.global.cg.f32 $0, [$1];",
        "=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


class ClusterDecodeKernel(UnifiedDecodeKernel):
    def __init__(self, *args, has_sink=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_sink = has_sink
        if self.num_splits != 8 or self.valid_hpb != 16:
            raise ValueError(
                "cluster merge requires eight splits and complete head groups"
            )

    @cute.jit
    def call_cluster(
        self,
        q_all: cute.Tensor,
        kv: cute.Tensor,
        indices: cute.Tensor,
        partials: cute.Tensor,
        lse: cute.Tensor,
        scale: Float32,
        latent: Float32,
        lengths: cute.Tensor,
        stride: Int64,
        extra_kv: cute.Tensor,
        extra_indices: cute.Tensor,
        extra_lengths: cute.Tensor,
        main_chunks: Int32,
        extra_stride: Int64,
        pages: cute.Tensor,
        table_width: Int32,
        table_stride: Int64,
        num_pages: Int64,
        output: cute.Tensor,
        sink: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.cluster_kernel(
            q_all,
            kv,
            indices,
            partials,
            lse,
            scale,
            latent,
            lengths,
            stride,
            extra_kv,
            extra_indices,
            extra_lengths,
            main_chunks,
            extra_stride,
            pages,
            table_width,
            table_stride,
            num_pages,
            output,
            sink,
        ).launch(
            grid=(rows, self.h_blocks, 8),
            cluster=(1, 1, 8),
            block=(self.block_threads, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def cluster_kernel(
        self,
        q_all,
        kv,
        indices,
        partials,
        lse,
        scale,
        latent,
        lengths,
        stride,
        extra_kv,
        extra_indices,
        extra_lengths,
        main_chunks,
        extra_stride,
        pages,
        table_width,
        table_stride,
        num_pages,
        output,
        sink,
    ):
        self._kernel_body(
            q_all,
            kv,
            indices,
            partials,
            lse,
            scale,
            latent,
            Int32(0),
            stride,
            extra_kv,
            extra_indices,
            Int32(0),
            main_chunks,
            extra_stride,
            lengths,
            extra_lengths,
            has_extra=True,
            per_token_len=True,
            mapping_page_table=pages,
            mapping_table_width=table_width,
            mapping_table_stride=table_stride,
            mapping_num_pages=num_pages,
        )
        cute.arch.sync_threads()
        _cluster_sync()
        tx, _, _ = cute.arch.thread_idx()
        row, head_block, split = cute.arch.block_idx()
        tx, row, head_block, split = (
            Int32(tx),
            Int32(row),
            Int32(head_block),
            Int32(split),
        )
        if tx < Int32(128):
            head = (head_block + Int32(self.head_block_offset)) * Int32(
                16
            ) + tx // Int32(8)
            dim = split * Int32(64) + tx % Int32(8) * Int32(8)
            maximum = Float32(-Float32.inf)
            sink_log2 = Float32(-Float32.inf)
            if cutlass.const_expr(self.has_sink):
                sink_log2 = Float32(sink[head]) * Float32(1.4426950408889634)
                maximum = sink_log2
            ls = cute.make_rmem_tensor((8,), Float32)
            for c in cutlass.range_constexpr(8):
                off = cute.crd2idx((row, head, Int32(c)), lse.layout)
                ls[c] = _load_lse(get_ptr_as_int64(lse, off))
                maximum = fmax_f32(maximum, ls[c])
            weights = cute.make_rmem_tensor((8,), Float32)
            total = Float32(0)
            if cutlass.const_expr(self.has_sink):
                if sink_log2 > Float32(-Float32.inf):
                    total = _exp2_approx_ftz_f32(sink_log2 - maximum)
            for c in cutlass.range_constexpr(8):
                weight = Float32(0)
                if ls[c] > Float32(-Float32.inf):
                    weight = _exp2_approx_ftz_f32(ls[c] - maximum)
                weights[c] = weight
                total += weight
            inv = Float32(0)
            if total > Float32(0):
                inv = Float32(1) / total
            acc = cute.make_rmem_tensor((8,), Float32)
            acc.fill(Float32(0))
            for c in cutlass.range_constexpr(8):
                if weights[c] != Float32(0):
                    off = cute.crd2idx((row, head, Int32(c), dim), partials.layout)
                    words = ld_global_cg_v4_u32(get_ptr_as_int64(partials, off))
                    for pair in cutlass.range_constexpr(4):
                        word = words[pair]
                        acc[pair * 2] += weights[c] * _u32_to_f32(word << Uint32(16))
                        acc[pair * 2 + 1] += weights[c] * _u32_to_f32(
                            word & Uint32(0xFFFF0000)
                        )
            for i in cutlass.range_constexpr(8):
                output[row, head, dim + Int32(i)] = BFloat16(acc[i] * inv)
        _cluster_sync()
