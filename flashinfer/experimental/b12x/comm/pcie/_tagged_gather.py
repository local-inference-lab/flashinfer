# Copyright 2026 Fraser Price
# Copyright (c) 2025-2026 by the b12x authors.
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.

"""Exact-word TP4 gather with payload-coupled readiness and bounded IPC storage.

The 64-bit (epoch, word) exchange follows Fraser BRRR's tagged gather. DCP owns
two dedicated receive regions, separate from its ordinary gather/LSE payloads.
Epoch wrap clears both regions before a group barrier and restarts at two,
retaining alternating parity while excluding the zero initialization tag.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._dcp_cute_common import block_pair_barrier, _membar_sys

# Every DCP barrier record uses only word zero. Graph replay uses padding
# words one and two; this counter uses the next padding word of record zero.
EPOCH_INDEX = 64 * 16 + 3


@dsl_user_op
def _load_epoch(address: Int64, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _load_word(address: Int64, *, loc=None, ip=None):
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Int64(address).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u64 $0, [$1];",
            "=l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_word(address: Int64, value: Uint64, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(address).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.sys.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class TaggedGather:
    def __init__(self, rank, threads, capacity):
        self.rank, self.threads, self.capacity = int(rank), int(threads), int(capacity)
        if (
            not 0 <= self.rank < 4
            or self.threads not in (32, 64, 128, 256, 512)
            or not 0 < self.capacity <= 4096
        ):
            raise ValueError(
                "tagged gather requires TP4, warp-aligned threads and at most 4096 words"
            )

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        p0: cute.Pointer,
        p1: cute.Pointer,
        p2: cute.Pointer,
        p3: cute.Pointer,
        s0: cute.Pointer,
        s1: cute.Pointer,
        s2: cute.Pointer,
        s3: cute.Pointer,
        stride: Int64,
        offset: Int64,
        rows: Int32,
        row_words: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            source,
            output,
            p0,
            p1,
            p2,
            p3,
            s0,
            s1,
            s2,
            s3,
            stride,
            offset,
            rows,
            row_words,
        ).launch(
            grid=(1, 1, 1), block=(self.threads, 1, 1), cluster=(1, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        source,
        output,
        p0,
        p1,
        p2,
        p3,
        s0,
        s1,
        s2,
        s3,
        stride,
        offset,
        rows,
        row_words,
    ):
        stages, signals = (p0, p1, p2, p3), (s0, s1, s2, s3)
        tid, _, _ = cute.arch.thread_idx()
        tid = Int32(tid)
        own_signal = signals[self.rank]
        last = _load_epoch((own_signal + Int64(EPOCH_INDEX)).toint())
        epoch = last + Uint32(1)
        if last == Uint32(0xFFFFFFFF):
            # Clear capacity, including words absent from the last replay. An
            # old tag must never satisfy a larger message after counter wrap.
            i = tid
            while i < Int32(2 * 4 * self.capacity):
                parity = i // Int32(4 * self.capacity)
                word = i % Int32(4 * self.capacity)
                address = (
                    stages[self.rank].toint()
                    + offset
                    + Int64(parity) * stride
                    + Int64(word) * Int64(8)
                )
                _store_word(address, Uint64(0))
                i += Int32(self.threads)
            cute.arch.sync_threads()
            _membar_sys()
            block_pair_barrier(
                signals,
                self_signal=own_signal,
                rank=self.rank,
                world_size=4,
                max_blocks=64,
            )
            epoch = Uint32(2)
        parity_offset = Int64(epoch & Uint32(1)) * stride + offset
        total = rows * row_words
        w = tid
        while w < total:
            value = Uint32(cute.arch.load(source + Int64(w), Uint32))
            packed = (Uint64(epoch) << Uint64(32)) | Uint64(value)
            for step in cutlass.range_constexpr(1, 4):
                peer = (self.rank + step) % 4
                destination = (
                    stages[peer].toint()
                    + parity_offset
                    + Int64(self.rank * self.capacity + w) * Int64(8)
                )
                _store_word(destination, packed)
            out_index = (
                w // row_words * Int32(4) + Int32(self.rank)
            ) * row_words + w % row_words
            cute.arch.store(output + Int64(out_index), value)
            w += Int32(self.threads)
        for step in cutlass.range_constexpr(1, 4):
            peer = (self.rank + step) % 4
            w = tid
            while w < total:
                address = (
                    stages[self.rank].toint()
                    + parity_offset
                    + Int64(peer * self.capacity + w) * Int64(8)
                )
                got = _load_word(address)
                while Uint32(got >> Uint64(32)) != epoch:
                    got = _load_word(address)
                out_index = (
                    w // row_words * Int32(4) + Int32(peer)
                ) * row_words + w % row_words
                cute.arch.store(output + Int64(out_index), Uint32(got))
                w += Int32(self.threads)
        cute.arch.sync_threads()
        if tid == Int32(0):
            cute.arch.store(own_signal + Int64(EPOCH_INDEX), epoch)


@program_cache
def compile_tagged(rank, threads, capacity):
    obj = TaggedGather(rank, threads, capacity)

    def ptr(dtype, value=16, align=16):
        return make_ptr(dtype, value, cute.AddressSpace.gmem, assumed_align=align)

    raw = b12x_compile(
        obj,
        ptr(Uint32, align=4),
        ptr(Uint32, align=4),
        *(ptr(Uint8) for _ in range(4)),
        *(ptr(Uint32, align=4) for _ in range(4)),
        Int64(256),
        Int64(256),
        Int32(1),
        Int32(4),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.dcp.tagged_gather", 2, (rank, threads, capacity)
        ),
    )

    def run(source, output, stages, signals, stride, offset, rows, row_words):
        if rows * row_words > capacity:
            raise ValueError("tagged gather exceeds its prepared word capacity")
        if source % 4 or output % 4:
            raise ValueError("tagged gather requires word-aligned input and output")
        raw(
            ptr(Uint32, source, 4),
            ptr(Uint32, output, 4),
            *(ptr(Uint8, value) for value in stages),
            *(ptr(Uint32, value, 4) for value in signals),
            Int64(stride),
            Int64(offset),
            Int32(rows),
            Int32(row_words),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
