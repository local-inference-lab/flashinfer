# SPDX-License-Identifier: Apache-2.0
"""Caller-owned state for a fixed-geometry persistent SM120 decoder.

The channel covers four ranks, up to 128 rows, 5120 hidden features, and
576 expert features per rank. Flags, epoch, work counters and temporary
activations share one allocation so package export resets them together.
The consumer must close the channel collectively after all graph replays drain.
"""

from functools import cache

import torch
import torch.distributed as dist
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce

PEER_BYTES = 8192 + 8 * 128 * 5120 * 2
SCRATCH_COUNT = 34


@cache
def module():
    from flashinfer.jit.persistent_decoder import gen_persistent_decoder_module

    return gen_persistent_decoder_module().build_and_load()


class PersistentDecoderTP4(PCIeOneshotAllReduce):
    """A peer channel with the PCIe runtime's coordinated allocation lifecycle.

    Construct during prepared-plan materialization. ``ipc`` is the package
    storage provider for exportable VMM allocations, or the ordinary CUDA
    runtime provider for standalone correctness checks.
    """

    def __init__(self, exchange_group, *, device, ipc):
        if dist.get_world_size(exchange_group) != 4:
            raise ValueError("persistent decoder requires exactly four ranks")
        self.exchange_group, self.device = exchange_group, torch.device(device)
        self.rank = dist.get_rank(exchange_group)
        self._ipc = ipc
        self._ptr = 0
        self._closed = self._ipc_imports_closed = self._ipc_exports_freed = False
        self._coordinated_close_complete = False
        self._registered_input_ptrs = set()
        self._owned_buffers = []
        self._closed_ipc_import_indices = set()
        self.module = module()
        self.prologue_ctas = self.module.prologue_ctas(self.device.index or 0)
        self.ctas = {b: self.module.ctas(self.device.index or 0, b) for b in (1, 2)}
        offsets, cursor = [], PEER_BYTES
        for index in range(SCRATCH_COUNT):
            cursor = (cursor + 255) // 256 * 256
            nbytes = self.module.scratch_bytes(index)
            offsets.append((cursor, nbytes))
            cursor += nbytes
        self.nbytes = (cursor + 255) // 256 * 256
        shared = self._allocate_shared_buffer(
            exchange_group, self.nbytes, zero_fill=True, ipc=ipc
        )
        self._owned_buffers.append(shared)
        ordinal = self.device.index
        if ordinal is None:
            ordinal = torch.cuda.current_device()
        self.peers = [
            torch.from_dlpack(self.module.view(ptr, self.nbytes, ordinal))
            for ptr in shared.peer_ptrs
        ]
        local = self.peers[self.rank]
        self.scratch = [local[offset : offset + nbytes] for offset, nbytes in offsets]
        self.owners = (shared, local, *self.peers, *self.scratch)

    def run(self, io, weights, *, blocks_per_sm=2, shared_a16=0):
        if self._closed:
            raise RuntimeError("persistent decoder channel is closed")
        self.module.run(
            io,
            weights,
            self.scratch,
            self.peers,
            self.rank,
            self.ctas[blocks_per_sm],
            shared_a16,
        )

    def prologue(self, io, weights):
        if self._closed:
            raise RuntimeError("persistent decoder channel is closed")
        self.module.prologue(io, weights, self.scratch, self.prologue_ctas)
