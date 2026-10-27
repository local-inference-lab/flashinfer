"""Exercise per-CTA rollover cleanup while another CTA still has live records."""

from __future__ import annotations

import ctypes
import os
import socket

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint32
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics.memory import st_global_v4_u32
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from b12x.comm.pcie._cute_intrinsics import (
    ld_acquire_sys_global_v4_u32,
    ld_relaxed_sys_global_v4_u32,
    st_release_sys_global_v4_u32,
)
from b12x.comm.pcie._oneshot_cute import _MAX_BLOCKS, _OneshotLaunch
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_TORTURE") != "1",
    reason="set B12X_RUN_PCIE_ONESHOT_TORTURE=1 for PCIe rollover tests",
)


class _RolloverReaderProbe(_OneshotLaunch):
    def __init__(self, rank, transport, threads, blocks):
        super().__init__("bfloat16", 2, rank, True, True, 0, transport, threads)
        self.blocks = blocks

    @cute.jit
    def __call__(
        self,
        signals: cute.Pointer,
        progress: cute.Pointer,
        observed: cute.Pointer,
        slot0: Int64,
        slot1: Int64,
        capacity: Int64,
        stream: cuda.CUstream,
    ):
        self.probe(signals, progress, observed, slot0, slot1, capacity).launch(
            grid=(self.blocks, 1, 1), block=(self._threads, 1, 1), stream=stream
        )

    @cute.kernel
    def probe(
        self,
        signals: cute.Pointer,
        progress: cute.Pointer,
        observed: cute.Pointer,
        slot0: Int64,
        slot1: Int64,
        capacity: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        progress_addr = Int64(progress.toint())
        if bidx == Int32(1):
            # Force CTA0 on both ranks to finish cleanup before CTA1 reads.
            ready = ld_acquire_sys_global_v4_u32(progress_addr)
            while ready[0] == Uint32(0):
                ready = ld_acquire_sys_global_v4_u32(progress_addr)
            index = Int64(self._threads) + Int64(tidx)
            if cutlass.const_expr(self._transport == "tp2_remote_push_stream"):
                index *= Int64(2)
            words = ld_relaxed_sys_global_v4_u32(slot0 + index * Int64(16))
            st_global_v4_u32(
                Int64(observed.toint()) + Int64(tidx) * Int64(16),
                words[0],
                words[1],
                words[2],
                words[3],
            )
            if cutlass.const_expr(self._transport == "tp2_remote_push_simple"):
                if tidx == Int32(0):
                    flag = ld_relaxed_sys_global_v4_u32(
                        slot0
                        + (capacity * Int64(2) - Int64(_MAX_BLOCKS) + Int64(1))
                        * Int64(16)
                    )
                    st_global_v4_u32(
                        Int64(observed.toint()) + Int64(self._threads) * Int64(16),
                        flag[0],
                        flag[1],
                        flag[2],
                        flag[3],
                    )
        self._clear_plain_graph_slots(signals, slot0, slot1, Int64(0), capacity)
        if bidx == Int32(0):
            if tidx == Int32(0):
                st_release_sys_global_v4_u32(
                    progress_addr, Uint32(1), Uint32(1), Uint32(1), Uint32(1)
                )


def _rollover_reader_worker(rank, port, transport, threads, blocks):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    capacity = 3 * threads
    capacity_bytes = capacity * 16
    pool = PCIeOneshotAllReducePool(
        rank=rank,
        world_size=2,
        device=device,
        exchange_group=dist.group.WORLD,
        eager_buffer_bytes=capacity_bytes,
        max_size=capacity_bytes,
        rank_data_bytes=capacity_bytes,
        single_channel=True,
    )
    try:
        stream = torch.cuda.Stream(device=device)
        with pool.capture(stream) as channel:
            state = channel._ext._state(channel._ptr)
            bases = tuple(
                channel._eager_ptrs[slot][rank]
                + state.plain_remote_push_region_packs * 16
                for slot in range(2)
            )
            progress = torch.zeros(4, dtype=torch.int32, device=device)
            observed = torch.zeros((threads + 1, 4), dtype=torch.int32, device=device)
            signals = make_ptr(
                cutlass.Uint64,
                state.signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            )
            progress_ptr = make_ptr(
                cutlass.Uint32,
                progress.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            observed_ptr = make_ptr(
                cutlass.Uint32,
                observed.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            probe = _RolloverReaderProbe(rank, transport, threads, blocks)
            run = b12x_compile(
                probe,
                signals,
                progress_ptr,
                observed_ptr,
                bases[0],
                bases[1],
                capacity,
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "test.comm.pcie.rollover_reader",
                    1,
                    (rank, transport, threads, blocks),
                ),
            )
            poison = (ctypes.c_uint32 * (capacity * 8))(
                *([0xFFFFFFFF] * (capacity * 8))
            )
            stream.wait_stream(torch.cuda.current_stream(device))
            for base in bases:
                channel._ipc.cudaMemcpyAsync(
                    base,
                    ctypes.addressof(poison),
                    ctypes.sizeof(poison),
                    int(stream.cuda_stream),
                )
            stream.synchronize()
            dist.barrier()
            with torch.cuda.stream(stream):
                run(
                    signals,
                    progress_ptr,
                    observed_ptr,
                    bases[0],
                    bases[1],
                    capacity,
                    current_cuda_stream(),
                )
            stream.synchronize()
            results = [None, None]
            checked_rows = threads + int(transport == "tp2_remote_push_simple")
            mismatches = int((observed[:checked_rows] != -1).sum().item())
            dist.all_gather_object(results, mismatches)
            assert results == [0, 0], f"rollover erased live CTA records: {results}"
            for base in bases:
                cleared = (ctypes.c_uint32 * (capacity * 8))()
                channel._ipc.cudaMemcpyAsync(
                    ctypes.addressof(cleared),
                    base,
                    ctypes.sizeof(cleared),
                    int(stream.cuda_stream),
                )
                stream.synchronize()
                assert not any(cleared), "rollover left stale capacity records"
    finally:
        pool.close()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "transport", ["tp2_remote_push", "tp2_remote_push_simple", "tp2_remote_push_stream"]
)
@pytest.mark.parametrize("threads,blocks", [(32, 2), (128, 2), (128, 3)])
def test_rollover_preserves_other_cta_live_records(transport, threads, blocks):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        _rollover_reader_worker,
        args=(port, transport, threads, blocks),
        nprocs=2,
        join=True,
    )
