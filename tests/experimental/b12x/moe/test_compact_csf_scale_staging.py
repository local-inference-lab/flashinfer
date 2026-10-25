"""Compact FC1 scale staging preserves live compressed payload bytes."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import SmemAllocator
import pytest
import torch

from b12x._lib.compiler import compile as compile_kernel
from b12x._lib.intrinsics.memory import shared_ptr_to_u32
from b12x._lib.quant._csf_prepare import prepare_csf_payload
from b12x._lib.quant.csf_inline import CsfInlineReader
from b12x._lib.utils.device import current_cuda_stream
from b12x.loader._csf_source import upload_csf_planes
from b12x.moe._shared.kernels.w4a8_compact_projection import (
    W4A8CompactMicroProjectionKernel,
)
from ..conftest import require_b12x
from .test_mxfp4_csf import _csf_planes


class StageCompactScales:
    """Expose production scale staging for both projection planes and tail halves."""

    def __init__(self, tile_n):
        self.n, self.k, self.experts = 576, 5120, 3
        self.reader = W4A8CompactMicroProjectionKernel(
            self.k, self.n, 6, tile_n=tile_n, csf_inline=True
        )
        self.reader.csf_reader = CsfInlineReader(
            (
                2 * self.n,
                self.k // 32,
                1,
                9,
                self.n,
                self.k // 32,
                self.experts,
                8,
                1,
                3,
                2,
            )
        )
        self.tiles = (self.n + tile_n - 1) // tile_n

    @cute.jit
    def __call__(self, source, routes, output, stream):
        self.kernel(source, routes, output).launch(
            grid=(1, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, source, routes, output):
        tid = Int32(cute.arch.thread_idx()[0])
        slot = SmemAllocator().allocate_tensor(Uint32, cute.make_layout(128), 16)
        address = shared_ptr_to_u32(slot.iterator)
        for task in range(self.experts * 2 * self.tiles):
            expert = routes[task % self.experts]
            projection = task // Int32(self.experts * self.tiles)
            tile = task // Int32(self.experts) % Int32(self.tiles)
            physical_tile = tile // Int32(128 // self.reader.tile_n)
            section = expert * Int32(2) + projection
            bounds = self.reader.csf_reader.prefetch_column_bounds(
                source.iterator, section, physical_tile
            )
            bases = self.reader.csf_reader.prefetch_row_bases(
                source.iterator, section, physical_tile, tid
            )
            values = self.reader.csf_reader.plane.group_value_offset(source.iterator)
            for column in range(self.k // 128):
                slot[tid] = Uint32(0xDEADBEEF)
                cute.arch.sync_threads()
                self.reader._stage_n64_sfb(
                    source,
                    address,
                    expert,
                    projection,
                    tile,
                    Int32(column),
                    tid,
                    bounds,
                    values,
                    bases,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_threads()
                self.reader.csf_reader.expand_slots(
                    source.iterator, address, tid, 128, 1
                )
                cute.arch.sync_threads()
                output[task, column, tid] = slot[tid]
                cute.arch.sync_threads()


@pytest.mark.parametrize("tile_n", (64, 128))
def test_compact_prefetched_scale_bytes_follow_mutable_payload(tile_n):
    """Captured staging rereads row bases, codes, exceptions and route IDs."""
    device = require_b12x()
    stager = StageCompactScales(tile_n)
    payloads, scale_grids = [], []
    for seed in (31, 32):
        planes, scales = _csf_planes(1152, 160, experts=3, seed=seed)
        payloads.append(
            prepare_csf_payload(
                upload_csf_planes(planes.planes, device), destination=3, group_rows=576
            ).storage
        )
        scale_grids.append(scales)
    source = torch.empty(
        max(t.numel() for t in payloads), dtype=torch.uint8, device=device
    )
    source[: payloads[0].numel()].copy_(payloads[0])
    routes = torch.arange(3, dtype=torch.int32, device=device)
    output = torch.empty(
        3 * 2 * stager.tiles, 40, 128, dtype=torch.uint32, device=device
    )
    args = tuple(from_dlpack(t, assumed_align=4) for t in (source, routes, output))
    program = compile_kernel(stager, *args, current_cuda_stream())
    program(*args, current_cuda_stream())
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            program(*args, current_cuda_stream())
        for iteration in range(4):
            payload = payloads[iteration % 2]
            source.fill_(0xFF)
            source[: payload.numel()].copy_(payload)
            routes.add_(1).remainder_(3)
            expected_scales = scale_grids[iteration % 2]
            expected = (
                torch.zeros_like(output).view(torch.uint8).reshape(-1, 40, 128, 4)
            )
            for task in range(output.shape[0]):
                expert = int(routes[task % 3])
                projection = task // (3 * stager.tiles)
                tile = task // 3 % stager.tiles
                row = tile // (128 // tile_n) * 128
                count = min(128, 576 - row)
                expected[task, :, :count] = (
                    expected_scales[
                        expert, projection * 576 + row : projection * 576 + row + count
                    ]
                    .reshape(count, 40, 4)
                    .transpose(0, 1)
                )
            output.fill_(0xA5A5A5A5)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output.view(torch.uint8).reshape_as(expected), expected)
    finally:
        graph.reset()
