"""CuTe DSL projection and top-k prototype for small BF16 router batches.

CTAs project disjoint experts with cooperating warps along the hidden dimension.
The last CTA for a row tile selects top-k and
normalizes its weights, avoiding a second launch. BF16-rounded logits remain
in caller-owned FP32 scratch. Counters start at zero and reset on completion;
concurrent calls must use distinct scratch and counters.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import torch

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, run_compiled
from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_v4_u32, u32_as_f32
from b12x._lib.intrinsics.reduction import warp_reduce
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.gemm.bf16_gemv._kernel import _dot_bf16x8, _flat
from b12x.gemm.bf16_gemv._tensor_core import TensorCoreGemvKernel
from b12x.gemm.bf16_gemv._prefill import Bf16PrefillKernel, _to_kernel_tensor


@dsl_user_op
def _arrive(address: Int64, *, loc=None, ip=None):
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.s32 $0, [$1], 1;",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _sum(a, b):
    return a + b


@dsl_user_op
def _bits(value: Float32, *, loc=None, ip=None):
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class RouterKernel:
    def __init__(
        self,
        hidden,
        experts,
        top_k,
        rows_per_tile,
        warps,
        warps_per_expert,
    ):
        self.hidden, self.experts, self.top_k = hidden, experts, top_k
        self.rows_per_tile, self.warps = rows_per_tile, warps
        self.warps_per_expert = warps_per_expert
        self.experts_per_cta = warps // warps_per_expert

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        logits: cute.Pointer,
        counters: cute.Pointer,
        selected: cute.Pointer,
        ids: cute.Pointer,
        scores: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(source),
            _flat(weight),
            _flat(logits),
            _flat(counters),
            _flat(selected),
            _flat(ids),
            _flat(scores),
            rows,
        ).launch(
            grid=(
                self.experts // self.experts_per_cta,
                (rows + self.rows_per_tile - 1) // self.rows_per_tile,
                1,
            ),
            block=(self.warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        logits: cute.Tensor,
        counters: cute.Tensor,
        selected: cute.Tensor,
        ids: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        column_tile, row_tile, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        expert = column_tile * self.experts_per_cta + warp // self.warps_per_expert
        first = row_tile * self.rows_per_tile
        acc = cute.make_rmem_tensor((self.rows_per_tile,), Float32)
        for r in cutlass.range_constexpr(self.rows_per_tile):
            acc[r] = Float32(0)
        for index in range(
            (self.hidden + 256 * self.warps_per_expert - 1)
            // (256 * self.warps_per_expert)
        ):
            k = (
                index * 32 * self.warps_per_expert
                + (warp % self.warps_per_expert) * 32
                + lane
            ) * 8
            if k < self.hidden:
                w0, w1, w2, w3 = ld_global_v4_u32(
                    get_ptr_as_int64(weight, Int64(expert * self.hidden + k))
                )
                for r in cutlass.range_constexpr(self.rows_per_tile):
                    if first + r < rows:
                        acc[r] = _dot_bf16x8(
                            source,
                            Int64((first + r) * self.hidden + k),
                            w0,
                            w1,
                            w2,
                            w3,
                            acc[r],
                        )
        allocator = cutlass.utils.SmemAllocator()
        partials = allocator.allocate_tensor(
            Float32,
            cute.make_layout((self.rows_per_tile, self.warps)),
            byte_alignment=16,
        )
        for r in cutlass.range_constexpr(self.rows_per_tile):
            total = warp_reduce(acc[r], _sum)
            if lane == 0:
                partials[r, warp] = total
        cute.arch.barrier()
        if warp % self.warps_per_expert == 0 and lane == 0:
            for r in cutlass.range_constexpr(self.rows_per_tile):
                total = Float32(0)
                for w in cutlass.range_constexpr(self.warps_per_expert):
                    total += partials[r, warp + w]
                if first + r < rows:
                    logits[(first + r) * self.experts + expert] = Float32(
                        total.to(BFloat16)
                    )
        finish_routing(
            logits,
            counters,
            selected,
            ids,
            scores,
            rows,
            row_tile,
            self.experts,
            self.top_k,
            self.rows_per_tile,
            self.warps,
            self.experts // self.experts_per_cta,
        )


@cute.jit
def finish_routing(
    logits,
    counters,
    selected,
    ids,
    scores,
    rows,
    row_tile,
    experts: cutlass.Constexpr,
    top_k: cutlass.Constexpr,
    rows_per_tile: cutlass.Constexpr,
    warps: cutlass.Constexpr,
    ctas: cutlass.Constexpr,
    split_k: cutlass.Constexpr = 1,
    round_logits: cutlass.Constexpr = False,
):
    tid, _, _ = cute.arch.thread_idx()
    lane, warp = tid % 32, tid // 32
    first = row_tile * rows_per_tile
    # Publish every warp's logits before the CTA contributes its arrival.
    cute.arch.barrier()
    smem = cutlass.utils.SmemAllocator().allocate_tensor(Int32, cute.make_layout((1,)))
    if tid == 0:
        smem[0] = _arrive(get_ptr_as_int64(counters, Int64(row_tile)))
    cute.arch.barrier()
    if smem[0] == ctas - 1:
        for row_group in cutlass.range_constexpr((rows_per_tile + warps - 1) // warps):
            row = first + row_group * warps + warp
            if row < rows and row < first + rows_per_tile:
                values = cute.make_rmem_tensor((experts // 32,), Int32)
                for e in cutlass.range_constexpr(experts // 32):
                    value = logits[row * experts + e * 32 + lane]
                    if cutlass.const_expr(split_k > 1):
                        for part in cutlass.range_constexpr(1, split_k):
                            value += logits[
                                (part * rows + row) * experts + e * 32 + lane
                            ]
                    if cutlass.const_expr(round_logits):
                        value = Float32(value.to(BFloat16))
                    bits = _bits(value) >> Uint32(16)
                    if value == Float32(0):
                        bits = Uint32(0)
                    ordered = bits ^ Uint32(0x8000)
                    if bits & Uint32(0x8000):
                        ordered = (~bits) & Uint32(0xFFFF)
                    # BF16 ordering plus the expert ID implements descending-ID ties.
                    values[e] = Int32((ordered << Uint32(10)) | Uint32(e * 32 + lane))
                chosen = cute.make_rmem_tensor(((top_k + 31) // 32,), Float32)
                for group in cutlass.range_constexpr((top_k + 31) // 32):
                    chosen[group] = Float32(0)
                chosen_ids = cute.make_rmem_tensor(((top_k + 31) // 32,), Int32)
                for group in cutlass.range_constexpr((top_k + 31) // 32):
                    chosen_ids[group] = Int32(0)
                maximum = Float32(0)
                for slot in cutlass.range_constexpr(top_k):
                    best = Int32(-1)
                    for e in cutlass.range_constexpr(experts // 32):
                        best = cutlass.max(best, values[e])
                    best = cute.arch.warp_reduction_max(best)
                    best_id = best & Int32(1023)
                    ordered = Uint32(best) >> Uint32(10)
                    bits = ordered ^ Uint32(0x8000)
                    if (ordered & Uint32(0x8000)) == Uint32(0):
                        bits = (~ordered) & Uint32(0xFFFF)
                    value = u32_as_f32(bits << Uint32(16))
                    if cutlass.const_expr(slot == 0):
                        maximum = value
                    if lane == slot % 32:
                        chosen[slot // 32] = value
                        chosen_ids[slot // 32] = best_id
                    for e in cutlass.range_constexpr(experts // 32):
                        if values[e] == best:
                            values[e] = Int32(-1)
                denominator = Float32(0)
                for group in cutlass.range_constexpr((top_k + 31) // 32):
                    exponential = Float32(0)
                    if group * 32 + lane < top_k:
                        selected[row * top_k + group * 32 + lane] = chosen[group]
                        ids[row * top_k + group * 32 + lane] = chosen_ids[group]
                        exponential = cute.math.exp(
                            chosen[group] - maximum, fastmath=True
                        )
                    chosen[group] = exponential
                    denominator += exponential
                denominator = warp_reduce(denominator, _sum)
                for group in cutlass.range_constexpr((top_k + 31) // 32):
                    if group * 32 + lane < top_k:
                        scores[row * top_k + group * 32 + lane] = (
                            chosen[group] / denominator
                        )
        cute.arch.barrier()
        if tid == 0:
            counters[row_tile] = Int32(0)


class TensorCoreRouterKernel(TensorCoreGemvKernel):
    def __init__(self, hidden, experts, top_k, rows_per_tile):
        super().__init__(experts, hidden)
        self.rows_per_tile = rows_per_tile
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        weight: cute.Pointer,
        logits: cute.Pointer,
        counters: cute.Pointer,
        selected: cute.Pointer,
        ids: cute.Pointer,
        scores: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            _flat(source),
            _flat(weight),
            _flat(logits),
            _flat(counters),
            _flat(selected),
            _flat(ids),
            _flat(scores),
            rows,
        ).launch(
            grid=(
                self.n // 16,
                (rows + self.rows_per_tile - 1) // self.rows_per_tile,
                1,
            ),
            block=(self.warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        logits: cute.Tensor,
        counters: cute.Tensor,
        selected: cute.Tensor,
        ids: cute.Tensor,
        scores: cute.Tensor,
        rows: Int32,
    ):
        _, row_tile, _ = cute.arch.block_idx()
        self.project_tile(
            source,
            weight,
            logits,
            rows,
            Int64(self.k),
            Int64(self.k),
            Int64(self.n),
            Int32(row_tile * self.rows_per_tile),
        )
        finish_routing(
            logits,
            counters,
            selected,
            ids,
            scores,
            rows,
            row_tile,
            self.n,
            self.top_k,
            self.rows_per_tile,
            self.warps,
            self.n // 16,
        )


class PipelinedRouterKernel(Bf16PrefillKernel):
    def __init__(self, hidden, experts, top_k, rows_per_tile, split_k):
        self.tile_m = rows_per_tile
        self.num_compute_warps = rows_per_tile // 16
        self.producer_warp = self.num_compute_warps
        self.num_threads = (self.num_compute_warps + 1) * 32
        super().__init__(experts, hidden, split_k)
        self.top_k = top_k

    @cute.jit
    def __call__(
        self,
        source: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        rows: Int32,
        stream: cuda.CUstream,
        counters: cute.Pointer,
        selected: cute.Pointer,
        ids: cute.Pointer,
        scores: cute.Pointer,
    ):
        Bf16PrefillKernel.__call__(
            self,
            source,
            weight,
            output,
            rows,
            stream,
            self.route,
            (counters, selected, ids, scores),
        )

    @cute.jit
    def route(self, output, rows, row_tile, counters, selected, ids, scores):
        tid, _, _ = cute.arch.thread_idx()
        _, column_tile, _ = cute.arch.block_idx()
        counter = _flat(counters)
        partials = _flat(output.iterator)
        row_tiles = (rows + self.tile_m - 1) // self.tile_m
        arrival = row_tiles + row_tile * self.n_tiles + column_tile
        flag = cutlass.utils.SmemAllocator().allocate_tensor(
            Int32, cute.make_layout((1,))
        )
        if tid == 0:
            flag[0] = _arrive(get_ptr_as_int64(counter, Int64(arrival)))
        cute.arch.barrier()
        if flag[0] == self.split_k - 1:
            for index in cutlass.range_constexpr(
                (self.tile_m * self.tile_n + self.num_threads - 1) // self.num_threads
            ):
                element = index * self.num_threads + tid
                row = row_tile * self.tile_m + element // self.tile_n
                column = column_tile * self.tile_n + element % self.tile_n
                if (
                    element < self.tile_m * self.tile_n
                    and row < rows
                    and column < self.n
                ):
                    value = Float32(0)
                    for part in cutlass.range_constexpr(self.split_k):
                        value += partials[(part * rows + row) * self.n + column]
                    partials[(self.split_k * rows + row) * self.n + column] = Float32(
                        value.to(BFloat16)
                    )
            cute.arch.barrier()
            if tid == 0:
                counter[arrival] = Int32(0)
            finish_routing(
                _flat(output.iterator + Int64(self.split_k * rows * self.n)),
                counter,
                _flat(selected),
                _flat(ids),
                _flat(scores),
                rows,
                row_tile,
                self.n,
                self.top_k,
                self.tile_m,
                self.num_threads // 32,
                self.n_tiles,
            )


def compile_router(
    *,
    device,
    hidden=2560,
    experts=512,
    top_k=10,
    rows_per_tile=4,
    warps=8,
    warps_per_expert=4,
    projection="simt",
    split_k=1,
    max_rows=128,
):
    """Prepare a single-launch router; finite inputs and contiguous storage required."""
    if any(
        type(value) is not int
        for value in (
            hidden,
            experts,
            top_k,
            rows_per_tile,
            warps,
            warps_per_expert,
            split_k,
            max_rows,
        )
    ):
        raise ValueError("router dimensions and tile sizes must be integers")
    if torch.device(device).type != "cuda":
        raise ValueError("router requires a CUDA device")
    if (
        hidden <= 0
        or hidden % 8
        or experts <= 0
        or experts % 32
        or experts > 1024
        or hidden * experts >= 2**31
    ):
        raise ValueError(
            "router requires hidden divisible by 8 and experts divisible by 32, at most 1024"
        )
    if (
        not 0 < top_k <= experts
        or rows_per_tile
        not in (
            (1, 2, 4, 8)
            if projection == "simt"
            else ((8,) if projection == "tensor_core" else (16, 32, 64))
        )
        or warps not in (4, 8)
        or warps_per_expert not in (1, 2, 4, 8)
        or warps_per_expert > warps
    ):
        raise ValueError("unsupported top-k, row tile, or warp count")
    if projection not in ("simt", "tensor_core", "tma"):
        raise ValueError("unsupported projection")
    if projection == "tensor_core" and (hidden % 64 or rows_per_tile != 8):
        raise ValueError(
            "tensor-core router requires hidden divisible by 64 and rows_per_tile=8"
        )
    if split_k < 1 or (projection != "tma" and split_k != 1):
        raise ValueError("split K is supported by the TMA projection only")
    if projection == "tma" and (hidden % (64 * split_k) or max_rows < 1):
        raise ValueError(
            "TMA projection requires hidden divisible by 64 * split_k and positive capacity"
        )
    ordinal = torch.device(device).index
    if ordinal is None:
        ordinal = torch.cuda.current_device()
    types = (BFloat16, BFloat16, Float32, Int32, Float32, Int32, Float32)

    def pointer(dtype, address):
        return make_ptr(
            dtype,
            address,
            cute.AddressSpace.gmem,
            assumed_align=4 if dtype != BFloat16 else 2,
        )

    spec = KernelCompileSpec.from_key(
        "moe.router_projection",
        2,
        (
            hidden,
            experts,
            top_k,
            rows_per_tile,
            warps,
            warps_per_expert,
            projection,
            split_k,
        ),
    )
    kernel = (
        RouterKernel(hidden, experts, top_k, rows_per_tile, warps, warps_per_expert)
        if projection == "simt"
        else TensorCoreRouterKernel(hidden, experts, top_k, rows_per_tile)
        if projection == "tensor_core"
        else PipelinedRouterKernel(hidden, experts, top_k, rows_per_tile, split_k)
    )

    def tma_args(source, weight, logits, pointers, rows):
        return (
            _to_kernel_tensor(source, BFloat16, dynamic_layout=True),
            _to_kernel_tensor(weight, BFloat16),
            _to_kernel_tensor(logits, Float32, dynamic_layout=True),
            Int32(rows),
            current_cuda_stream(),
            *pointers,
        )

    with torch.cuda.device(ordinal):
        if projection == "tma":
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                device = torch.device("cuda", ordinal)
                source = torch.empty(
                    (max_rows, hidden), device=device, dtype=torch.bfloat16
                )
                weight = torch.empty(
                    (experts, hidden), device=device, dtype=torch.bfloat16
                )
                logits = torch.empty(
                    (max_rows * (split_k + 1), experts),
                    device=device,
                    dtype=torch.float32,
                )
                args = tma_args(
                    source,
                    weight,
                    logits,
                    tuple(pointer(t, 16) for t in types[3:]),
                    max_rows,
                )
                raw = compile_kernel(kernel, *args, compile_spec=spec)
        else:
            raw = compile_kernel(
                kernel,
                *(pointer(t, 16) for t in types),
                Int32(1),
                current_cuda_stream(),
                compile_spec=spec,
            )

    def run(source, weight, logits, counters, selected, ids, scores):
        rows = source.shape[0]
        if rows * max(hidden, experts) >= 2**31:
            raise ValueError("router dimensions exceed Int32 indexing")
        shapes = (
            (rows, hidden),
            (experts, hidden),
            (rows * (split_k + 1 if projection == "tma" else 1), experts),
            (
                ((rows + rows_per_tile - 1) // rows_per_tile)
                * ((experts + 63) // 64 + 1 if projection == "tma" else 1),
            ),
            (rows, top_k),
            (rows, top_k),
            (rows, top_k),
        )
        tensors = (source, weight, logits, counters, selected, ids, scores)
        dtypes = (
            torch.bfloat16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.float32,
            torch.int32,
            torch.float32,
        )
        for tensor, shape, dtype in zip(tensors, shapes, dtypes, strict=True):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.device != torch.device("cuda", ordinal)
            ):
                raise ValueError(
                    "router operands must match prepared contiguous shapes, dtypes, and device"
                )
        if source.data_ptr() % 16 or weight.data_ptr() % 16:
            raise ValueError("router source and weight must be 16-byte aligned")
        for index, target in enumerate(tensors[2:], 2):
            start, end = (
                target.data_ptr(),
                target.data_ptr() + target.numel() * target.element_size(),
            )
            for other in tensors[:index]:
                if (
                    start < other.data_ptr() + other.numel() * other.element_size()
                    and other.data_ptr() < end
                ):
                    raise ValueError(
                        "router writable storage must not overlap inputs or other outputs"
                    )
        if not rows:
            return
        with torch.cuda.device(ordinal):
            args = (
                tma_args(
                    source,
                    weight,
                    logits,
                    tuple(
                        pointer(t, v.data_ptr()) for t, v in zip(types[3:], tensors[3:])
                    ),
                    rows,
                )
                if projection == "tma"
                else (
                    *(pointer(t, v.data_ptr()) for t, v in zip(types, tensors)),
                    Int32(rows),
                    current_cuda_stream(),
                )
            )
            run_compiled(raw, args)

    return attach_programs(run, raw)
