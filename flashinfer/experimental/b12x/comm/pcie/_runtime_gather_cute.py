"""Base launch interface and dispatch for specialized column-gather kernels."""

import cuda.bindings.driver as cuda
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._dcp_a2a_cute import _AllGatherHeadsLaunch


class SelectedGather(_AllGatherHeadsLaunch):
    def __init__(self, world, rank, row_words, threads, blocks, chunks, transport):
        if world not in (2, 4, 8, 9, 10, 12, 16) or not 0 <= rank < world:
            raise ValueError("gather requires a supported TP size and rank")
        if threads not in (32, 64, 128, 256, 512):
            raise ValueError("gather requires 32, 64, 128, 256, or 512 threads")
        if transport != "barrier":
            raise ValueError("selected gather transport must be barrier")
        if row_words <= 0 or row_words % 4 or not 1 <= blocks <= 64:
            raise ValueError("gather requires aligned rows and one to 64 CTAs")
        super().__init__(world, rank, threads, True)
        self.row_words, self.blocks, self.transport = row_words, blocks, transport

    @cute.jit
    def __call__(
        self,
        source: cute.Pointer,
        output: cute.Pointer,
        selector: cute.Pointer,
        stages: tuple,
        signals: tuple,
        rows: Int32,
        stride: Int64,
        offset: Int64,
        counter: Int64,
        stream: cuda.CUstream,
    ):
        self.selected(
            source, output, selector, stages, signals, rows, stride, offset, counter
        ).launch(grid=(self.blocks, 1, 1), block=(self._threads, 1, 1), stream=stream)

    @cute.kernel
    def selected(
        self, source, output, selector, stages, signals, rows, stride, offset, counter
    ):
        block, _, _ = cute.arch.block_idx()
        profile = Uint32(cute.arch.load(selector, Uint32))
        active = Int32(profile & Uint32(0xFFFF))
        if Int32(block) < active:
            self.gather(
                source,
                output,
                stages,
                signals,
                rows,
                Int32(1),
                Int32(self.row_words // 4),
                Int32(stride // Int64(256)),
                active,
            )


@program_cache
def compile_selected_gather(world, rank, row_words, threads, blocks, chunks, transport):
    if transport in (
        "recursive",
        "recursive_ready",
        "recursive_outer_first",
        "recursive_pix_last",
    ):
        from ._recursive_gather_cute import RecursiveGather

        operation = RecursiveGather(
            world,
            rank,
            row_words,
            threads,
            blocks,
            chunks,
            acknowledge_consumption=transport
            not in ("recursive_ready", "recursive_pix_last"),
            outer_first=transport == "recursive_outer_first",
            pix_last=transport == "recursive_pix_last",
        )
    elif transport == "ring":
        from ._ring_gather_cute import RingGather

        operation = RingGather(world, rank, row_words, threads, blocks, chunks)
    elif transport == "switch_push":
        from ._island_gather_cute import SwitchPushGather

        operation = SwitchPushGather(world, rank, row_words, threads, blocks, chunks)
    elif transport in ("phased_barrier", "phased_push"):
        from ._phased_gather_cute import PhasedGather

        operation = PhasedGather(
            world,
            rank,
            row_words,
            threads,
            blocks,
            chunks,
            phase_peer_barrier=transport == "phased_barrier",
        )
    else:
        raise ValueError(f"unsupported selected gather transport {transport!r}")

    def pointer(dtype, address=16, alignment=16):
        return make_ptr(
            dtype, address, mem_space=cute.AddressSpace.gmem, assumed_align=alignment
        )

    raw = compile_kernel(
        operation,
        pointer(Uint32),
        pointer(Uint32),
        pointer(Uint32, alignment=4),
        tuple(pointer(Uint8) for _ in range(16)),
        tuple(pointer(Uint32, alignment=4) for _ in range(16)),
        Int32(1),
        Int64(256),
        Int64(0),
        Int64(512),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.selected_gather",
            22,
            (world, rank, row_words, threads, blocks, chunks, transport),
        ),
    )

    def run(source, output, selector, stages, signals, rows, stride, offset, counter):
        if len(stages) != world or len(signals) != world:
            raise ValueError("gather requires one staging and signal address per rank")
        if rows <= 0:
            raise ValueError("gather requires at least one row")
        if source % 16 or output % 16:
            raise ValueError("gather requires 16-byte aligned input and output")
        if (
            source < output + rows * row_words * world * 4
            and output < source + rows * row_words * 4
        ):
            raise ValueError("gather requires disjoint input and output")
        raw(
            pointer(Uint32, source),
            pointer(Uint32, output),
            pointer(Uint32, selector, 4),
            tuple(pointer(Uint8, address) for address in (tuple(stages) * 16)[:16]),
            tuple(
                pointer(Uint32, address, 4) for address in (tuple(signals) * 16)[:16]
            ),
            Int32(rows),
            Int64(stride),
            Int64(offset),
            Int64(counter),
            current_cuda_stream(),
        )

    return attach_programs(run, raw)
