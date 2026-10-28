"""Fixed-grid BF16 PCIe collectives with device-resident protocol profiles.

The selected uint32 profile contains protocol in its upper 16 bits and the
participating CTA count in its lower 16 bits. Ranks must select the same profile,
and profile changes require all ranks' previous replay to have drained.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.program_cache import program_cache
from b12x._lib.intrinsics.memory import (
    ld_global_v4_u32,
    ld_global_nc_v4_u32,
    st_global_v4_u32,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr
from ._oneshot_cute import _PackedMath, _dummy
from ._cute_intrinsics import (
    graph_epoch_arrive_serialized,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_global_v4_u32,
    st_relaxed_sys_global_v4_u32,
    pack_f32x2_to_bf16x2,
    pcie_arrive_and_wait,
)
from ._twoshot_cute import _ld_generic_v4_u32, _st_generic_v4_u32

from ._joint_profile import (
    JOINT_MAX_BLOCKS,
    JOINT_MAX_RANKS,
    JOINT_SIGNAL_BYTES,
    JOINT_MODES,
    joint_slot_packs,
)

_SELF_BYTES = JOINT_MAX_BLOCKS * JOINT_MAX_RANKS * 4
_FLAG_SLOT_BYTES = JOINT_MAX_BLOCKS * JOINT_MAX_RANKS * 32 * 4
_EPOCH_OFFSET = JOINT_SIGNAL_BYTES - 8
_PREPARED = set()

__all__ = [
    "JOINT_MAX_BLOCKS",
    "JOINT_MAX_RANKS",
    "JOINT_SIGNAL_BYTES",
    "JOINT_MODES",
    "joint_slot_packs",
    "get_joint_launcher",
    "is_joint_launcher_prepared",
]


class _JointLaunch(_PackedMath):
    def __init__(self, family, world_size, rank, threads, max_blocks):
        super().__init__("bfloat16")
        self._family = family
        self._world_size = world_size
        self._rank = rank
        self._threads = threads
        self._max_blocks = max_blocks

    @cute.jit
    def __call__(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        local_signal: cute.Pointer,
        selector: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        size: Int64,
        capacity: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            peers, signals, local_signal, selector, payload, output, size, capacity
        ).launch(
            grid=(self._max_blocks, 1, 1),
            block=(self._threads, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _barrier(self, signals: cute.Pointer):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        cute.arch.sync_threads()
        if tid < Int32(self._world_size):
            local = Int64(signals[self._rank])
            remote = Int64(signals[tid])
            counter = local + (Int64(bid) * Int64(16) + Int64(tid)) * Int64(4)
            send = (
                remote
                + Int64(_SELF_BYTES)
                + (Int64(bid) * Int64(16 * 32) + Int64(self._rank * 32)) * Int64(4)
            )
            wait = (
                local
                + Int64(_SELF_BYTES)
                + (Int64(bid) * Int64(16 * 32) + Int64(tid) * Int64(32)) * Int64(4)
            )
            pcie_arrive_and_wait(counter, send, wait, Int64(_FLAG_SLOT_BYTES))
        cute.arch.sync_threads()

    @cute.jit
    def _owner(self, index: Int64, size: Int64) -> Int32:
        part = size // Int64(self._world_size)
        owner = Int32(self._world_size - 1)
        if part > Int64(0):
            owner = Int32(cutlass.min(index // part, Int64(self._world_size - 1)))
        return owner

    @cute.jit
    def _ready(
        self,
        peers: cute.Pointer,
        payload: Int64,
        output: Int64,
        size: Int64,
        capacity: Int64,
        flat: Int64,
        stride: Int64,
        mode: Uint32,
    ):
        local = Int64(peers[self._rank]) + capacity * Int64(16)
        index = flat
        while index < size:
            words = ld_global_v4_u32(payload + index * Int64(16))
            if mode == Uint32(0):
                for target in cutlass.range_constexpr(self._world_size):
                    dest = Int64(peers[target]) + (
                        capacity + Int64(self._rank) * capacity + index
                    ) * Int64(16)
                    self._publish_bf16_words(dest, words)
            else:
                owner = self._owner(index, size)
                dest = Int64(peers[owner]) + (
                    capacity + Int64(self._rank) * capacity + index
                ) * Int64(16)
                self._publish_bf16_words(dest, words)
            index += stride
        index = flat
        while index < size:
            owner = Int32(self._rank)
            if mode == Uint32(1):
                owner = self._owner(index, size)
            if owner == Int32(self._rank):
                acc = cute.make_rmem_tensor((8,), Float32)
                self._receive_bf16_sum(local, capacity, index, acc)
                result = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    result[word] = pack_f32x2_to_bf16x2(
                        acc[word * 2], acc[word * 2 + 1]
                    )
                st_global_v4_u32(
                    output + index * Int64(16),
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                )
                if mode == Uint32(1):
                    for target in cutlass.range_constexpr(self._world_size):
                        if cutlass.const_expr(target != self._rank):
                            dest = Int64(peers[target]) + (
                                capacity + Int64(self._rank) * capacity + index
                            ) * Int64(16)
                            self._publish_bf16_words(dest, result)
            index += stride
        if mode == Uint32(1):
            index = flat
            while index < size:
                owner = self._owner(index, size)
                if owner != Int32(self._rank):
                    address = local + (Int64(owner) * capacity + index) * Int64(16)
                    gathered = cute.make_rmem_tensor((4,), Uint32)
                    waiting = cutlass.Boolean(True)
                    while waiting:
                        words = ld_relaxed_sys_global_v4_u32(address)
                        waiting = cutlass.Boolean(False)
                        for word in cutlass.range_constexpr(4):
                            gathered[word] = words[word]
                            waiting = (
                                waiting
                                | ((words[word] & Uint32(0xFFFF)) == Uint32(0))
                                | ((words[word] & Uint32(0xFFFF0000)) == Uint32(0))
                            )
                    st_global_v4_u32(
                        output + index * Int64(16),
                        gathered[0],
                        gathered[1],
                        gathered[2],
                        gathered[3],
                    )
                    st_relaxed_sys_global_v4_u32(
                        address, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    )
                index += stride

    @cute.jit
    def _pull(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        payload: Int64,
        output: Int64,
        size: Int64,
        flat: Int64,
        stride: Int64,
    ):
        index = flat
        while index < size:
            self._copy_pack(
                payload + index * Int64(16),
                Int64(peers[self._rank]) + index * Int64(16),
            )
            index += stride
        self._barrier(signals)
        index = flat
        while index < size:
            acc = cute.make_rmem_tensor((8,), Float32)
            for peer_index in cutlass.range_constexpr(self._world_size):
                peer = (self._rank + peer_index) % self._world_size
                self._load_accumulate(
                    acc, Int64(peers[peer]) + index * Int64(16), peer_index == 0
                )
            self._store_accumulator(output + index * Int64(16), acc)
            index += stride

    @cute.jit
    def _bulk(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        payload: Int64,
        output: Int64,
        size: Int64,
        capacity: Int64,
        flat: Int64,
        stride: Int64,
        push: cutlass.Constexpr[bool],
    ):
        shard = (size + Int64(self._world_size - 1)) // Int64(self._world_size)
        cap_shard = (capacity + Int64(self._world_size - 1)) // Int64(self._world_size)
        slab_offset = Int64(self._world_size + 1) * capacity
        reduced_offset = slab_offset + Int64(self._world_size) * cap_shard
        local = Int64(peers[self._rank])
        # Every stage, reduction and gather uses the same shard-relative CTA.
        index = flat
        while index < shard:
            for peer_index in cutlass.range_constexpr(1, self._world_size):
                target = (self._rank + peer_index) % self._world_size
                pack = Int64(target) * shard + index
                if pack < size:
                    words = ld_global_nc_v4_u32(payload + pack * Int64(16))
                    if cutlass.const_expr(push):
                        dest = Int64(peers[target]) + (
                            slab_offset + Int64(self._rank) * cap_shard + index
                        ) * Int64(16)
                        _st_generic_v4_u32(dest, words[0], words[1], words[2], words[3])
                    else:
                        st_global_v4_u32(
                            local + pack * Int64(16),
                            words[0],
                            words[1],
                            words[2],
                            words[3],
                        )
            index += stride
        self._barrier(signals)
        index = flat
        while index < shard:
            pack = Int64(self._rank) * shard + index
            if pack < size:
                acc = cute.make_rmem_tensor((8,), Float32)
                for lane in cutlass.range_constexpr(8):
                    acc[lane] = Float32(0.0)
                words = ld_global_nc_v4_u32(payload + pack * Int64(16))
                self._accumulate_words(
                    acc, words[0], words[1], words[2], words[3], False
                )
                for peer_index in cutlass.range_constexpr(1, self._world_size):
                    peer = (self._rank + peer_index) % self._world_size
                    if cutlass.const_expr(push):
                        address = local + (
                            slab_offset + Int64(peer) * cap_shard + index
                        ) * Int64(16)
                        words = _ld_generic_v4_u32(address)
                    else:
                        address = Int64(peers[peer]) + pack * Int64(16)
                        words = ld_global_v4_u32(address)
                    self._accumulate_words(
                        acc, words[0], words[1], words[2], words[3], False
                    )
                result = cute.make_rmem_tensor((4,), Uint32)
                for word in cutlass.range_constexpr(4):
                    result[word] = pack_f32x2_to_bf16x2(
                        acc[word * 2], acc[word * 2 + 1]
                    )
                st_global_v4_u32(
                    output + pack * Int64(16),
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                )
                if cutlass.const_expr(push):
                    for peer_index in cutlass.range_constexpr(1, self._world_size):
                        peer = (self._rank + peer_index) % self._world_size
                        address = Int64(peers[peer]) + (
                            reduced_offset + Int64(self._rank) * cap_shard + index
                        ) * Int64(16)
                        _st_generic_v4_u32(
                            address, result[0], result[1], result[2], result[3]
                        )
                else:
                    st_global_v4_u32(
                        local + (reduced_offset + index) * Int64(16),
                        result[0],
                        result[1],
                        result[2],
                        result[3],
                    )
            index += stride
        self._barrier(signals)
        for peer_index in cutlass.range_constexpr(1, self._world_size):
            peer = (self._rank + peer_index) % self._world_size
            index = flat
            while index < shard:
                pack = Int64(peer) * shard + index
                if pack < size:
                    if cutlass.const_expr(push):
                        address = local + (
                            reduced_offset + Int64(peer) * cap_shard + index
                        ) * Int64(16)
                        words = _ld_generic_v4_u32(address)
                    else:
                        address = Int64(peers[peer]) + (reduced_offset + index) * Int64(
                            16
                        )
                        words = ld_global_v4_u32(address)
                    st_global_v4_u32(
                        output + pack * Int64(16),
                        words[0],
                        words[1],
                        words[2],
                        words[3],
                    )
                index += stride

    @cute.kernel
    def kernel(
        self,
        peers: cute.Pointer,
        signals: cute.Pointer,
        local_signal: cute.Pointer,
        selector: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        size: Int64,
        capacity: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        self_signal = Int64(local_signal.toint())
        generation = ld_relaxed_gpu_u32(self_signal + Int64(_EPOCH_OFFSET))
        profile = selector[0]
        mode = profile >> Uint32(16)
        active = profile & Uint32(0xFFFF)
        if Uint32(bid) < active:
            slot_peers = peers + Int64(generation % Uint32(2)) * Int64(16)
            cute.arch.sync_threads()
            if tid == Int32(self._threads - 1):
                graph_epoch_arrive_serialized(
                    self_signal + Int64(_EPOCH_OFFSET),
                    self_signal + Int64(_EPOCH_OFFSET + 4),
                    active,
                )
            flat = Int64(bid) * Int64(self._threads) + Int64(tid)
            stride = Int64(active) * Int64(self._threads)
            src = Int64(payload.toint())
            dst = Int64(output.toint())
            if mode == Uint32(4):
                self._pull(slot_peers, signals, src, dst, size, flat, stride)
            elif mode == Uint32(1):
                self._ready(slot_peers, src, dst, size, capacity, flat, stride, mode)
            else:
                if cutlass.const_expr(self._family == "latency"):
                    self._ready(
                        slot_peers, src, dst, size, capacity, flat, stride, mode
                    )
                else:
                    if mode == Uint32(2):
                        self._bulk(
                            slot_peers,
                            signals,
                            src,
                            dst,
                            size,
                            capacity,
                            flat,
                            stride,
                            True,
                        )
                    else:
                        self._bulk(
                            slot_peers,
                            signals,
                            src,
                            dst,
                            size,
                            capacity,
                            flat,
                            stride,
                            False,
                        )


def is_joint_launcher_prepared(
    family, world_size, rank, threads, max_blocks, device_index
):
    return (family, world_size, rank, threads, max_blocks, device_index) in _PREPARED


@program_cache
def get_joint_launcher(family, world_size, rank, threads, max_blocks, device_index):
    if family not in ("latency", "bandwidth"):
        raise ValueError("joint family must be latency or bandwidth")
    if world_size not in (2, 3, 4, 6, 8, 10) or not 0 <= rank < world_size:
        raise ValueError("unsupported joint collective rank group")
    if not 1 <= max_blocks <= JOINT_MAX_BLOCKS or threads not in (
        32,
        64,
        128,
        256,
        512,
    ):
        raise ValueError("invalid joint launch geometry")
    key = (family, world_size, rank, threads, max_blocks, device_index)
    launch = _JointLaunch(family, world_size, rank, threads, max_blocks)
    raise_if_kernel_resolution_frozen("cute.compile", target=launch, cache_key=key)
    raw = b12x_compile(
        launch,
        _dummy(cutlass.Uint64, 8),
        _dummy(cutlass.Uint64, 8),
        _dummy(Uint32, 4),
        _dummy(Uint32, 4),
        _dummy(Uint32, 16),
        _dummy(Uint32, 16),
        1,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("comm.pcie.joint", 5, key[:-1]),
    )

    def run(
        peer_table_address,
        signal_table_address,
        local_signal_address,
        selector_address,
        input_address,
        output_address,
        size_packs,
        capacity_packs,
    ):
        if not 0 < size_packs <= capacity_packs:
            raise ValueError("joint payload must fit its prepared capacity")
        raw(
            make_ptr(
                cutlass.Uint64,
                peer_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                cutlass.Uint64,
                signal_table_address,
                cute.AddressSpace.gmem,
                assumed_align=8,
            ),
            make_ptr(
                Uint32, local_signal_address, cute.AddressSpace.gmem, assumed_align=4
            ),
            make_ptr(Uint32, selector_address, cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(Uint32, input_address, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Uint32, output_address, cute.AddressSpace.gmem, assumed_align=16),
            int(size_packs),
            int(capacity_packs),
            current_cuda_stream(),
        )

    _PREPARED.add(key)
    return attach_programs(run, raw)
