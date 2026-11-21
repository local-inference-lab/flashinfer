"""Eight-batch TP16 island reduction using local SM writes and cross-root DMA."""

from functools import partial
from weakref import finalize

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.utils.device import cuda_stream_from_int_or_current, cuda_stream_to_int
from b12x._lib.program_cache import program_cache
from ._dma_kernels import _compile, _fake_ptr, _ptr
from ._island_rs_cute import (
    MAX_BLOCKS,
    _ACTIVE_EPOCH_OFFSET,
    _AG_ARRIVED,
    _RS_ARRIVED,
    _XS_ARRIVED,
    _accumulate_payload_words,
    _atomic_add_global_u32,
    _load_payload_words,
    _pack_payload_words,
    _store_payload_words,
    _store_release_sys_u32,
    _wait_for,
    pack_f32x2_to_bf16x2,
    unpack_bf16x2,
)

# Each rank has one generation word and sixteen single-writer arrival words.
_FLAG_BYTES = 256


class _Rendezvous:
    def __init__(self, peers):
        self.peers = peers

    @cute.jit
    def __call__(
        self, counter: cute.Pointer, tx: tuple, rx: tuple, stream: cuda.CUstream
    ):
        self.kernel(counter, tx, rx).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, counter: cute.Pointer, tx: tuple, rx: tuple):
        lane, _, _ = cute.arch.thread_idx()
        tag = Uint32(0)
        if lane == Int32(0):
            tag = _atomic_add_global_u32(Int64(counter.toint()), Uint32(1)) + Uint32(1)
        tag = cute.arch.shuffle_sync(tag, offset=0)
        dest, source = Int64(0), Int64(0)
        for i in cutlass.range_constexpr(self.peers):
            if lane == Int32(i):
                dest = Int64(tx[i].toint())
                source = Int64(rx[i].toint())
        if lane < Int32(self.peers):
            _store_release_sys_u32(dest, tag)
        cute.arch.barrier()
        if lane < Int32(self.peers):
            _wait_for(source, tag, 0)


class _AdvanceOwnerEpoch:
    @cute.jit
    def __call__(
        self, header: cute.Pointer, blocks: Int32, active: Int32, stream: cuda.CUstream
    ):
        self.kernel(header, blocks, active).launch(
            grid=(1, 1, 1), block=(MAX_BLOCKS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, header: cute.Pointer, blocks: Int32, active: Int32):
        block, _, _ = cute.arch.thread_idx()
        if block < blocks:
            header[block] = header[block] + Uint32(1)
        if block < active:
            slot = Int32(_ACTIVE_EPOCH_OFFSET // 4) + block
            epoch = header[slot] + Uint32(1)
            header[slot] = epoch
            # All peers have retired their preceding SM operation. The final
            # rendezvous publishes these local arrival words before its successor.
            for peer in cutlass.range_constexpr(4):
                offset = (block * Int32(4) + Int32(peer)) * Int32(128 // 4)
                header[Int32(_RS_ARRIVED // 4) + offset] = epoch
                header[Int32(_AG_ARRIVED // 4) + offset] = epoch
                header[Int32(_XS_ARRIVED // 4) + offset] = epoch * Uint32(2) + Uint32(1)


class _ReduceFour:
    def __init__(self, threads):
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        inputs: tuple,
        out: cute.Pointer,
        words: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(inputs, out, words, blocks).launch(
            grid=(blocks, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, inputs: tuple, out: cute.Pointer, words: Int64, blocks: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        thread = Int64(bid) * Int64(self.threads) + Int64(tid)
        stride = Int64(blocks) * Int64(self.threads)
        aligned = (
            Int64(out.toint())
            | Int64(inputs[0].toint())
            | Int64(inputs[1].toint())
            | Int64(inputs[2].toint())
            | Int64(inputs[3].toint())
        ) % Int64(16) == Int64(0)
        i = thread
        if aligned:
            v4 = words // Int64(4)
            while i < v4:
                base = i * Int64(4)
                totals = (Float32(0),) * 8
                for p in cutlass.range_constexpr(4):
                    w = _load_payload_words(inputs[p], base, 4)
                    totals = _accumulate_payload_words(totals, w, 4)
                packed = _pack_payload_words(totals, 4)
                _store_payload_words(out, base, packed, 4)
                i += stride
            i = v4 * Int64(4) + thread
        while i < words:
            lo, hi = Float32(0), Float32(0)
            for p in cutlass.range_constexpr(4):
                x, y = unpack_bf16x2(inputs[p][i])
                lo += x
                hi += y
            out[i] = pack_f32x2_to_bf16x2(lo, hi)
            i += stride


class _LocalTransfer:
    def __init__(self, peers, threads):
        self.peers = peers
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        sources: tuple,
        destinations: tuple,
        lengths: tuple,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(sources, destinations, lengths, blocks).launch(
            grid=(blocks, self.peers, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, sources: tuple, destinations: tuple, lengths: tuple, blocks: Int32
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, peer, _ = cute.arch.block_idx()
        thread = Int64(bid) * Int64(self.threads) + Int64(tid)
        stride = Int64(blocks) * Int64(self.threads)
        for p in cutlass.range_constexpr(self.peers):
            if peer == Int32(p):
                src = sources[p]
                dst = destinations[p]
                words = lengths[p]
                aligned = (Int64(src.toint()) | Int64(dst.toint())) % Int64(
                    16
                ) == Int64(0)
                i = thread
                if aligned:
                    while 4 * i + 3 < words:
                        values = _load_payload_words(src, 4 * i, 4)
                        _store_payload_words(dst, 4 * i, values, 4)
                        i += stride
                    i = (words // 4) * 4 + thread
                while i < words:
                    dst[i] = src[i]
                    i += stride


def copy_scratch_bytes(quarter_words):
    packet_words = quarter_words // 4
    return _FLAG_BYTES + (10 * packet_words + 4 * quarter_words) * 4


class PacketTransport:
    def __init__(self, device, ipc):
        self.ipc = ipc
        self.started = torch.cuda.Event()
        self.finished = torch.cuda.Event()
        # A pooled stream can later become the graph's capture stream.
        with torch.cuda.device(device):
            status, handle = cuda.cuStreamCreate(
                cuda.CUstream_flags.CU_STREAM_NON_BLOCKING
            )
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"creating the island packet stream failed: {status}"
                )
            try:
                self.stream = torch.cuda.ExternalStream(int(handle), device=device)
            except BaseException:
                cuda.cuStreamDestroy(handle)
                raise

        def release():
            with torch.cuda.device(device):
                (status,) = cuda.cuStreamDestroy(handle)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"destroying the island packet stream failed: {status}"
                )

        self.stream._b12x_release = finalize(self.stream, release)
        self.stream._b12x_release.atexit = False

    def close(self):
        if self.stream is not None:
            self.stream._b12x_release()
            self.stream = None


class PacketPrograms:
    def __init__(self, run):
        self.run = run

    def bind(self, transport):
        return attach_programs(partial(self.run, transport), self)


@program_cache
def get_packet_programs(rank, threads, device_index):
    del device_index  # Keep device-local program handles distinct in the process cache.
    ptr = _fake_ptr(Uint32, 4)
    rendezvous = _compile(
        "comm.pcie.island_ce.parallel_rendezvous_v2",
        (6,),
        _Rendezvous(6),
        ptr,
        (ptr,) * 6,
        (ptr,) * 6,
    )
    rendezvous3 = _compile(
        "comm.pcie.island_ce.parallel_rendezvous_v2",
        (3,),
        _Rendezvous(3),
        ptr,
        (ptr,) * 3,
        (ptr,) * 3,
    )
    reduce = _compile(
        "comm.pcie.island_ce.reduce4_v2",
        (threads,),
        _ReduceFour(threads),
        (ptr,) * 4,
        ptr,
        Int64(1),
        Int32(1),
    )
    advance_owner = _compile(
        "comm.pcie.hybrid.advance_owner_epoch",
        (),
        _AdvanceOwnerEpoch(),
        ptr,
        Int32(1),
        Int32(1),
    )
    island, lane = divmod(rank, 4)
    local = tuple(rank ^ d for d in (1, 2, 3))
    cross = tuple(rank ^ (d * 4) for d in (1, 2, 3))

    # All islands sharing a lane visit the same three XOR partners in order.
    distances = tuple(
        tuple(1 + (owner + step) % 3 for step in range(3)) for owner in range(4)
    )
    transfer = _compile(
        "comm.pcie.hybrid.local_transfer",
        (3, threads),
        _LocalTransfer(3, threads),
        (ptr,) * 3,
        (ptr,) * 3,
        (Int64(1),) * 3,
        Int32(1),
    )

    def run(
        link,
        slab_addresses,
        input_address,
        output_address,
        stage_offset,
        part_offset,
        final_offset,
        quarter_capacity,
        elements,
        blocks,
        stream=None,
        active_blocks=None,
        timing_address=0,
        timing_calls=0,
        timing_blocks=0,
        timing_slices=0,
    ):
        if len(slab_addresses) != 16:
            raise ValueError(f"expected 16 slab addresses, got {len(slab_addresses)}")
        if not 1 <= int(blocks) <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}], got {blocks}")
        active_blocks = blocks if active_blocks is None else active_blocks
        if not 1 <= int(active_blocks) <= int(blocks):
            raise ValueError("active blocks must fit the fixed physical grid")
        if timing_calls:
            raise ValueError("copy-engine collectives have no CTA timing records")
        transport = link.stream
        started = link.started
        finished = link.finished
        transport_stream = cuda_stream_from_int_or_current(transport.cuda_stream)
        main = (
            torch.cuda.current_stream()
            if stream is None
            else stream
            if isinstance(stream, torch.cuda.Stream)
            else torch.cuda.ExternalStream(
                cuda_stream_to_int(stream), device=transport.device
            )
        )
        if main.cuda_stream == transport.cuda_stream:
            raise ValueError("island packet compute and copy streams must be distinct")
        stream = cuda_stream_from_int_or_current(main.cuda_stream)
        quarter_bytes = quarter_capacity * 4
        # The SM protocol keeps both of its banks through the next SM invocation.
        base_offset = (final_offset + 8 * quarter_bytes + 255) // 256 * 256
        bases = tuple(ptr + base_offset for ptr in slab_addresses)
        packet_bytes = quarter_bytes // 4
        stage = _FLAG_BYTES
        inbox = stage + 6 * packet_bytes
        assembled = inbox + 4 * packet_bytes
        local_base = bases[rank]

        def ptr32(address):
            return _ptr(Uint32, address, 4)

        peers = local + cross
        arrivals = tuple(ptr32(local_base + 4 + 4 * p) for p in peers)
        publications = tuple(ptr32(bases[p] + 4 + 4 * rank) for p in peers)
        counter = ptr32(local_base)
        # Bytes [128..195] inside _FLAG_BYTES=256 hold disjoint 3-peer local rendezvous slots.
        arrivals3 = tuple(ptr32(local_base + 132 + 4 * p) for p in local)
        publications3 = tuple(ptr32(bases[p] + 132 + 4 * rank) for p in local)
        counter3 = ptr32(local_base + 128)

        def barrier():
            rendezvous(counter, publications, arrivals, stream)

        def barrier_local():
            rendezvous3(counter3, publications3, arrivals3, stream)

        def reduce_four(inputs, dst, words):
            reduce(
                tuple(ptr32(p) for p in inputs),
                ptr32(dst),
                words,
                active_blocks,
                stream,
            )

        pairs = elements // 2
        quarter = (pairs + 3) // 4
        stripe = (quarter + 3) // 4

        def shard(owner, root):
            begin = min(quarter * owner + stripe * root, (owner + 1) * quarter, pairs)
            end = min(begin + stripe, (owner + 1) * quarter, pairs)
            return 4 * begin, end - begin

        def local_slot(target, sender, generation):
            peer_index = (sender % 4 ^ target % 4) - 1
            return (
                bases[target]
                + stage
                + (3 * (generation % 2) + peer_index) * packet_bytes
            )

        self_partial = local_base + inbox + island * packet_bytes

        def local_sum(root, generation):
            offset, count = shard(lane, root)
            inputs = tuple(
                input_address + offset
                if other == lane
                else local_slot(rank, 4 * island + other, generation)
                for other in range(4)
            )
            reduce_four(inputs, self_partial, count)

        for batch in range(8):
            if batch == 4:
                local_sum(island, 3)
                offset, count = shard(lane, island)
                inputs = tuple(local_base + inbox + j * packet_bytes for j in range(4))
                reduce_four(inputs, local_base + assembled + offset, count)

            if 1 <= batch <= 6:
                reducing = batch < 4
                distance = distances[lane][batch - 1 if reducing else batch - 4]
                peer_root = island ^ distance
                target = 4 * peer_root + lane
                if reducing:
                    local_sum(peer_root, batch - 1)
                    _, count = shard(lane, peer_root)
                    src = self_partial
                    dst = bases[target] + inbox + island * packet_bytes
                else:
                    offset, count = shard(lane, island)
                    src = local_base + assembled + offset
                    dst = bases[target] + assembled + offset
                started.record(main)
                transport.wait_event(started)
                if count:
                    link.ipc.cudaMemcpyAsync(dst, src, count * 4, int(transport_stream))
                finished.record(transport)

            copies = []
            for target in local:
                if batch < 4:
                    owner = target % 4
                    root = island if batch == 3 else island ^ distances[owner][batch]
                    offset, count = shard(owner, root)
                    src = input_address + offset
                    dst = local_slot(target, rank, batch)
                else:
                    root = island if batch == 4 else island ^ distances[lane][batch - 5]
                    offset, count = shard(lane, root)
                    src = local_base + assembled + offset
                    dst = bases[target] + assembled + offset
                copies.append((ptr32(src), ptr32(dst), count))
            sources, destinations, lengths = zip(*copies)
            transfer(sources, destinations, lengths, active_blocks, stream)
            if batch == 7:
                # Publish SM owner-epoch updates before the final 6-peer rendezvous so
                # subsequent SM calls observe updated counters without a 9th barrier.
                advance_owner(ptr32(slab_addresses[rank]), blocks, active_blocks, stream)
            if 1 <= batch <= 6:
                main.wait_event(finished)
                barrier()
            elif batch == 0:
                barrier_local()
            else:
                barrier()

        link.ipc.cudaMemcpyAsync(
            output_address, local_base + assembled, pairs * 4, int(stream)
        )

    return attach_programs(
        PacketPrograms(run), reduce, advance_owner, transfer, rendezvous, rendezvous3
    )
