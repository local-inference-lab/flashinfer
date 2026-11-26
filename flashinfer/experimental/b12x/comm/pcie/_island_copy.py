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


class _ReduceFourAndCross:
    """Bit-exact fused local four-way BF16 reduction and four-island cross reduction."""

    def __init__(self, threads, self_island):
        self.threads = threads
        self.self_island = int(self_island)

    @cute.jit
    def __call__(
        self,
        local_inputs: tuple,
        cross_inputs: tuple,
        out: cute.Pointer,
        words: Int64,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(local_inputs, cross_inputs, out, words, blocks).launch(
            grid=(blocks, 1, 1), block=(self.threads, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        local_inputs: tuple,
        cross_inputs: tuple,
        out: cute.Pointer,
        words: Int64,
        blocks: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        thread = Int64(bid) * Int64(self.threads) + Int64(tid)
        stride = Int64(blocks) * Int64(self.threads)
        mask = (
            Int64(out.toint())
            | Int64(local_inputs[0].toint())
            | Int64(local_inputs[1].toint())
            | Int64(local_inputs[2].toint())
            | Int64(local_inputs[3].toint())
        )
        for j in cutlass.range_constexpr(4):
            if j != self.self_island:
                mask = mask | Int64(cross_inputs[j].toint())
        aligned = mask % Int64(16) == Int64(0)
        i = thread
        if aligned:
            v4 = words // Int64(4)
            while i < v4:
                base = i * Int64(4)
                local_totals = (Float32(0),) * 8
                for p in cutlass.range_constexpr(4):
                    w = _load_payload_words(local_inputs[p], base, 4)
                    local_totals = _accumulate_payload_words(local_totals, w, 4)
                local_packed = _pack_payload_words(local_totals, 4)
                totals = (Float32(0),) * 8
                for j in cutlass.range_constexpr(4):
                    if j == self.self_island:
                        totals = _accumulate_payload_words(totals, local_packed, 4)
                    else:
                        w = _load_payload_words(cross_inputs[j], base, 4)
                        totals = _accumulate_payload_words(totals, w, 4)
                packed = _pack_payload_words(totals, 4)
                _store_payload_words(out, base, packed, 4)
                i += stride
            i = v4 * Int64(4) + thread
        while i < words:
            lo, hi = Float32(0), Float32(0)
            for p in cutlass.range_constexpr(4):
                x, y = unpack_bf16x2(local_inputs[p][i])
                lo += x
                hi += y
            local_word = pack_f32x2_to_bf16x2(lo, hi)
            tlo, thi = Float32(0), Float32(0)
            for j in cutlass.range_constexpr(4):
                w = local_word if j == self.self_island else cross_inputs[j][i]
                x, y = unpack_bf16x2(w)
                tlo += x
                thi += y
            out[i] = pack_f32x2_to_bf16x2(tlo, thi)
            i += stride


def _coalesced_shard_spans(shard_indices):
    ordered = sorted(shard_indices)
    if not ordered:
        return ()
    spans = []
    begin = prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            spans.append((begin, prev - begin + 1))
            begin = prev = idx
    spans.append((begin, prev - begin + 1))
    return tuple(spans)


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


class _A2APackLocalPush:
    """Pack rank-owned rows into per-destination-island blocks on all four local lanes."""

    def __init__(self, threads):
        self.threads = threads

    @cute.jit
    def __call__(
        self,
        src: cute.Pointer,
        destinations: tuple,
        st: Int64,
        c_words: Int64,
        lane: Int32,
        blocks: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(src, destinations, st, c_words, lane, blocks).launch(
            grid=(blocks, 4, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Pointer,
        destinations: tuple,
        st: Int64,
        c_words: Int64,
        lane: Int32,
        blocks: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, l_dst, _ = cute.arch.block_idx()
        thread = Int64(bid) * Int64(self.threads) + Int64(tid)
        stride = Int64(blocks) * Int64(self.threads)
        for p in cutlass.range_constexpr(4):
            if l_dst == Int32(p):
                dst = destinations[p]
                aligned = (Int64(src.toint()) | Int64(dst.toint())) % Int64(
                    16
                ) == Int64(0) and c_words % Int64(4) == Int64(0)
                if aligned:
                    c_v4 = c_words // Int64(4)
                    total_v4 = Int64(4) * st * c_v4
                    idx = thread
                    while idx < total_v4:
                        col_v4 = idx % c_v4
                        tmp = idx // c_v4
                        row = tmp % st
                        i_dst = tmp // st
                        r_dst = i_dst * Int64(4) + Int64(p)
                        src_base = ((row * Int64(16) + r_dst) * c_v4 + col_v4) * Int64(
                            4
                        )
                        dst_base = (
                            ((i_dst * Int64(4) + Int64(lane)) * st + row) * c_v4
                            + col_v4
                        ) * Int64(4)
                        values = _load_payload_words(src, src_base, 4)
                        _store_payload_words(dst, dst_base, values, 4)
                        idx += stride
                else:
                    total = Int64(4) * st * c_words
                    idx = thread
                    while idx < total:
                        col = idx % c_words
                        tmp = idx // c_words
                        row = tmp % st
                        i_dst = tmp // st
                        r_dst = i_dst * Int64(4) + Int64(p)
                        src_idx = (row * Int64(16) + r_dst) * c_words + col
                        dst_idx = (
                            (i_dst * Int64(4) + Int64(lane)) * st + row
                        ) * c_words + col
                        dst[dst_idx] = src[src_idx]
                        idx += stride


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
            prio_status, _lo, hi = cuda.cuCtxGetStreamPriorityRange()
            if prio_status == cuda.CUresult.CUDA_SUCCESS:
                status, handle = cuda.cuStreamCreateWithPriority(
                    cuda.CUstream_flags.CU_STREAM_NON_BLOCKING, hi
                )
            else:
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

    reduce_final = _compile(
        "comm.pcie.island_ce.reduce4_cross_v1",
        (threads, island),
        _ReduceFourAndCross(threads, island),
        (ptr,) * 4,
        (ptr,) * 4,
        ptr,
        Int64(1),
        Int32(1),
    )

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

        def reduce_four(inputs, dst, words, target_stream=stream):
            reduce(
                tuple(ptr32(p) for p in inputs),
                ptr32(dst),
                words,
                active_blocks,
                target_stream,
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

        def local_inputs_for(root, generation):
            offset, count = shard(lane, root)
            inputs = tuple(
                input_address + offset
                if other == lane
                else local_slot(rank, 4 * island + other, generation)
                for other in range(4)
            )
            return inputs, offset, count

        def local_sum(root, generation, target_stream=stream):
            inputs, _, count = local_inputs_for(root, generation)
            reduce_four(inputs, self_partial, count, target_stream)

        for batch in range(8):
            if batch == 4:
                local_inputs, offset, count = local_inputs_for(island, 3)
                cross_inputs = tuple(
                    local_base + inbox + j * packet_bytes for j in range(4)
                )
                reduce_final(
                    tuple(ptr32(p) for p in local_inputs),
                    tuple(ptr32(p) for p in cross_inputs),
                    ptr32(local_base + assembled + offset),
                    count,
                    active_blocks,
                    stream,
                )

            if 1 <= batch <= 6:
                reducing = batch < 4
                distance = distances[lane][batch - 1 if reducing else batch - 4]
                peer_root = island ^ distance
                target = 4 * peer_root + lane
                started.record(main)
                transport.wait_event(started)
                if reducing:
                    local_sum(peer_root, batch - 1, transport_stream)
                    _, count = shard(lane, peer_root)
                    src = self_partial
                    dst = bases[target] + inbox + island * packet_bytes
                else:
                    offset, count = shard(lane, island)
                    src = local_base + assembled + offset
                    dst = bases[target] + assembled + offset
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
                advance_owner(
                    ptr32(slab_addresses[rank]), blocks, active_blocks, stream
                )
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
        PacketPrograms(run),
        reduce,
        reduce_final,
        advance_owner,
        transfer,
        rendezvous,
        rendezvous3,
    )


def _setup_packet_context(
    rank,
    link,
    slab_addresses,
    final_offset,
    quarter_capacity,
    blocks,
    active_blocks,
    stream,
    timing_calls,
    rendezvous,
    rendezvous3,
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
    cu_stream = cuda_stream_from_int_or_current(main.cuda_stream)
    quarter_bytes = quarter_capacity * 4
    base_offset = (final_offset + 8 * quarter_bytes + 255) // 256 * 256
    bases = tuple(ptr + base_offset for ptr in slab_addresses)
    packet_bytes = quarter_bytes // 4
    stage = _FLAG_BYTES
    inbox = stage + 6 * packet_bytes
    assembled = inbox + 4 * packet_bytes
    local_base = bases[rank]

    def ptr32(address):
        return _ptr(Uint32, address, 4)

    local = tuple(rank ^ d for d in (1, 2, 3))
    cross = tuple(rank ^ (d * 4) for d in (1, 2, 3))
    peers = local + cross
    arrivals = tuple(ptr32(local_base + 4 + 4 * p) for p in peers)
    publications = tuple(ptr32(bases[p] + 4 + 4 * rank) for p in peers)
    counter = ptr32(local_base)
    arrivals3 = tuple(ptr32(local_base + 132 + 4 * p) for p in local)
    publications3 = tuple(ptr32(bases[p] + 132 + 4 * rank) for p in local)
    counter3 = ptr32(local_base + 128)

    def barrier():
        rendezvous(counter, publications, arrivals, cu_stream)

    def barrier_local():
        rendezvous3(counter3, publications3, arrivals3, cu_stream)

    return (
        main,
        cu_stream,
        transport,
        transport_stream,
        started,
        finished,
        active_blocks,
        quarter_bytes,
        packet_bytes,
        stage,
        inbox,
        assembled,
        bases,
        local_base,
        ptr32,
        barrier,
        barrier_local,
    )


@program_cache
def get_reduce_scatter_programs(rank, threads, device_index, transport="sm_local"):
    del device_index
    if transport not in ("sm_local", "ce_local"):
        raise ValueError(f"unsupported island reduce_scatter transport {transport!r}")
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
    cross_threads = 512 if transport == "ce_local" else threads
    reduce = _compile(
        "comm.pcie.island_ce.reduce4_v2",
        (cross_threads,),
        _ReduceFour(cross_threads),
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
    transfer = (
        None
        if transport == "ce_local"
        else _compile(
            "comm.pcie.hybrid.local_transfer",
            (3, threads),
            _LocalTransfer(3, threads),
            (ptr,) * 3,
            (ptr,) * 3,
            (Int64(1),) * 3,
            Int32(1),
        )
    )
    island, lane = divmod(rank, 4)
    local = tuple(rank ^ d for d in (1, 2, 3))
    distances = tuple(
        tuple(1 + (owner + step) % 3 for step in range(3)) for owner in range(4)
    )
    reduce_final = _compile(
        "comm.pcie.island_ce.reduce4_cross_v1",
        (threads, island),
        _ReduceFourAndCross(threads, island),
        (ptr,) * 4,
        (ptr,) * 4,
        ptr,
        Int64(1),
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
        del stage_offset, part_offset, timing_address, timing_blocks, timing_slices
        (
            main,
            cu_stream,
            transport_s,
            transport_stream,
            started,
            finished,
            active_blocks,
            _quarter_bytes,
            packet_bytes,
            stage,
            inbox,
            _assembled,
            bases,
            local_base,
            ptr32,
            barrier,
            barrier_local,
        ) = _setup_packet_context(
            rank,
            link,
            slab_addresses,
            final_offset,
            quarter_capacity,
            blocks,
            active_blocks,
            stream,
            timing_calls,
            rendezvous,
            rendezvous3,
        )
        pairs = elements // 2
        if elements % 32 != 0:
            raise ValueError(
                f"island reduce_scatter requires elements divisible by 32, got {elements}"
            )
        shard_pairs = pairs // 16
        if shard_pairs * 4 > packet_bytes:
            raise ValueError("island reduce_scatter shard exceeds packet buffer")

        def rank_shard(owner, root):
            begin = (4 * root + owner) * shard_pairs
            return 4 * begin, shard_pairs

        def local_slot(target, sender, generation):
            peer_index = (sender % 4 ^ target % 4) - 1
            return (
                bases[target]
                + stage
                + (3 * (generation % 2) + peer_index) * packet_bytes
            )

        def reduce_four(
            inputs, dst, words, target_stream=cu_stream, grid_blocks=active_blocks
        ):
            reduce(
                tuple(ptr32(p) for p in inputs),
                ptr32(dst),
                words,
                grid_blocks,
                target_stream,
            )

        self_partial = local_base + inbox + island * packet_bytes

        def local_inputs_for(root, generation):
            offset, count = rank_shard(lane, root)
            inputs = tuple(
                input_address + offset
                if other == lane
                else local_slot(rank, 4 * island + other, generation)
                for other in range(4)
            )
            return inputs, count

        def local_sum(root, generation, target_stream=cu_stream):
            inputs, count = local_inputs_for(root, generation)
            reduce_four(inputs, self_partial, count, target_stream)

        if transport == "ce_local":
            # Flipped engine assignment for compute-overlapped collectives:
            # Intra-4 (55 GB/s, 3 peers) runs on the copy engine (0 SMs), while
            # cross-4 (14 GB/s, 1 peer) fuses reduce4 + remote store onto 1 SM CTA.
            for batch in range(4):
                started.record(main)
                transport_s.wait_event(started)
                if shard_pairs:
                    for target in local:
                        owner = target % 4
                        root = (
                            island if batch == 3 else island ^ distances[owner][batch]
                        )
                        offset, count = rank_shard(owner, root)
                        link.ipc.cudaMemcpyAsync(
                            local_slot(target, rank, batch),
                            input_address + offset,
                            count * 4,
                            int(transport_stream),
                        )
                finished.record(transport_s)

                if 1 <= batch <= 3:
                    distance = distances[lane][batch - 1]
                    peer_root = island ^ distance
                    target = 4 * peer_root + lane
                    dst = bases[target] + inbox + island * packet_bytes
                    inputs, count = local_inputs_for(peer_root, batch - 1)
                    reduce_four(inputs, dst, count, cu_stream, grid_blocks=1)

                main.wait_event(finished)
                if batch == 0:
                    barrier_local()
                else:
                    barrier()

            local_inputs, _ = local_inputs_for(island, 3)
            cross_inputs = tuple(
                local_base + inbox + j * packet_bytes for j in range(4)
            )
            reduce_final(
                tuple(ptr32(p) for p in local_inputs),
                tuple(ptr32(p) for p in cross_inputs),
                ptr32(output_address),
                shard_pairs,
                min(active_blocks, 4),
                cu_stream,
            )
            advance_owner(ptr32(slab_addresses[rank]), blocks, active_blocks, cu_stream)
            barrier()
            return

        for batch in range(4):
            if 1 <= batch <= 3:
                distance = distances[lane][batch - 1]
                peer_root = island ^ distance
                target = 4 * peer_root + lane
                dst = bases[target] + inbox + island * packet_bytes
                started.record(main)
                transport_s.wait_event(started)
                local_sum(peer_root, batch - 1, transport_stream)
                if shard_pairs:
                    link.ipc.cudaMemcpyAsync(
                        dst, self_partial, shard_pairs * 4, int(transport_stream)
                    )
                finished.record(transport_s)

            copies = []
            for target in local:
                owner = target % 4
                root = island if batch == 3 else island ^ distances[owner][batch]
                offset, count = rank_shard(owner, root)
                copies.append(
                    (
                        ptr32(input_address + offset),
                        ptr32(local_slot(target, rank, batch)),
                        count,
                    )
                )
            sources, destinations, lengths = zip(*copies)
            transfer(sources, destinations, lengths, active_blocks, cu_stream)
            if batch == 0:
                barrier_local()
            else:
                main.wait_event(finished)
                barrier()

        local_inputs, _ = local_inputs_for(island, 3)
        cross_inputs = tuple(local_base + inbox + j * packet_bytes for j in range(4))
        reduce_final(
            tuple(ptr32(p) for p in local_inputs),
            tuple(ptr32(p) for p in cross_inputs),
            ptr32(output_address),
            shard_pairs,
            active_blocks,
            cu_stream,
        )
        # Publish SM owner-epoch updates and guarantee every peer finished
        # consuming inbox/stage before any successor writes into them.
        advance_owner(ptr32(slab_addresses[rank]), blocks, active_blocks, cu_stream)
        barrier()

    attached = (
        (reduce, reduce_final, advance_owner, rendezvous, rendezvous3)
        if transfer is None
        else (reduce, reduce_final, advance_owner, transfer, rendezvous, rendezvous3)
    )
    return attach_programs(PacketPrograms(run), *attached)


@program_cache
def get_all_gather_programs(rank, threads, device_index, transport="sm_local"):
    del device_index
    if transport not in ("sm_local", "ce_local"):
        raise ValueError(f"unsupported island all_gather transport {transport!r}")
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
    advance_owner = _compile(
        "comm.pcie.hybrid.advance_owner_epoch",
        (),
        _AdvanceOwnerEpoch(),
        ptr,
        Int32(1),
        Int32(1),
    )
    if transport == "ce_local":
        transfer = _compile(
            "comm.pcie.hybrid.local_transfer",
            (1, 512),
            _LocalTransfer(1, 512),
            (ptr,),
            (ptr,),
            (Int64(1),),
            Int32(1),
        )
    else:
        transfer = _compile(
            "comm.pcie.hybrid.local_transfer",
            (3, threads),
            _LocalTransfer(3, threads),
            (ptr,) * 3,
            (ptr,) * 3,
            (Int64(1),) * 3,
            Int32(1),
        )
    island, lane = divmod(rank, 4)
    local = tuple(rank ^ d for d in (1, 2, 3))
    other_lanes = tuple(other for other in range(4) if other != lane)
    distances = tuple(
        tuple(1 + (owner + step) % 3 for step in range(3)) for owner in range(4)
    )
    # Shards settled in local assembled after each step's barrier, coalesced into
    # contiguous spans so D2D copies overlap with the next step's LocalTransfer.
    step_copy_spans = (
        (),
        _coalesced_shard_spans(
            [4 * island + other for other in other_lanes]
            + [4 * (island ^ distances[lane][0]) + lane]
        ),
        _coalesced_shard_spans(
            [4 * (island ^ distances[other][0]) + other for other in other_lanes]
            + [4 * (island ^ distances[lane][1]) + lane]
        ),
        _coalesced_shard_spans(
            [4 * (island ^ distances[other][1]) + other for other in other_lanes]
            + [4 * (island ^ distances[lane][2]) + lane]
        ),
    )
    tail_copy_spans = _coalesced_shard_spans(
        [4 * (island ^ distances[other][2]) + other for other in other_lanes]
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
        del stage_offset, part_offset, timing_address, timing_blocks, timing_slices
        (
            main,
            cu_stream,
            transport_s,
            transport_stream,
            started,
            finished,
            active_blocks,
            _quarter_bytes,
            packet_bytes,
            _stage,
            _inbox,
            assembled,
            bases,
            local_base,
            ptr32,
            barrier,
            _barrier_local,
        ) = _setup_packet_context(
            rank,
            link,
            slab_addresses,
            final_offset,
            quarter_capacity,
            blocks,
            active_blocks,
            stream,
            timing_calls,
            rendezvous,
            rendezvous3,
        )
        if elements % 2 != 0:
            raise ValueError(
                f"island all_gather requires even shard elements, got {elements}"
            )
        shard_pairs = elements // 2
        shard_bytes = shard_pairs * 4
        if shard_bytes > packet_bytes:
            raise ValueError("island all_gather shard exceeds packet buffer")

        def rank_shard(owner, root):
            begin = (4 * root + owner) * shard_pairs
            return 4 * begin, shard_pairs

        def copy_spans(spans, target_stream_int):
            if not shard_bytes:
                return
            for begin_shard, count_shards in spans:
                offset = begin_shard * shard_bytes
                link.ipc.cudaMemcpyAsync(
                    output_address + offset,
                    local_base + assembled + offset,
                    count_shards * shard_bytes,
                    target_stream_int,
                )

        own_offset, _ = rank_shard(lane, island)
        # Ensure any preceding collective's trailing copy out of assembled has retired.
        barrier()

        if transport == "ce_local":
            for step in range(4):
                started.record(main)
                transport_s.wait_event(started)
                if shard_bytes:
                    if step == 0:
                        src_addr = input_address
                        dst_off = own_offset
                    else:
                        root = island ^ distances[lane][step - 1]
                        dst_off, _ = rank_shard(lane, root)
                        src_addr = local_base + assembled + dst_off
                    for target in local:
                        link.ipc.cudaMemcpyAsync(
                            bases[target] + assembled + dst_off,
                            src_addr,
                            shard_bytes,
                            int(transport_stream),
                        )
                finished.record(transport_s)

                if step < 3 and shard_pairs:
                    distance = distances[lane][step]
                    peer_root = island ^ distance
                    target = 4 * peer_root + lane
                    transfer(
                        (ptr32(input_address),),
                        (ptr32(bases[target] + assembled + own_offset),),
                        (shard_pairs,),
                        1,
                        cu_stream,
                    )

                main.wait_event(finished)
                if step == 3:
                    advance_owner(
                        ptr32(slab_addresses[rank]), blocks, active_blocks, cu_stream
                    )
                barrier()

            if shard_bytes:
                link.ipc.cudaMemcpyAsync(
                    output_address,
                    local_base + assembled,
                    16 * shard_bytes,
                    int(cu_stream),
                )
                link.ipc.cudaMemcpyAsync(
                    output_address + own_offset,
                    input_address,
                    shard_bytes,
                    int(cu_stream),
                )
            return

        for step in range(4):
            started.record(main)
            transport_s.wait_event(started)
            if step < 3:
                distance = distances[lane][step]
                peer_root = island ^ distance
                target = 4 * peer_root + lane
                dst = bases[target] + assembled + own_offset
                if shard_bytes:
                    link.ipc.cudaMemcpyAsync(
                        dst, input_address, shard_bytes, int(transport_stream)
                    )
            if step == 0:
                if shard_bytes:
                    link.ipc.cudaMemcpyAsync(
                        output_address + own_offset,
                        input_address,
                        shard_bytes,
                        int(transport_stream),
                    )
            else:
                copy_spans(step_copy_spans[step], int(transport_stream))
            finished.record(transport_s)

            if step == 0:
                copies = [
                    (
                        ptr32(input_address),
                        ptr32(bases[target] + assembled + own_offset),
                        shard_pairs,
                    )
                    for target in local
                ]
            else:
                root = island ^ distances[lane][step - 1]
                offset, count = rank_shard(lane, root)
                src = local_base + assembled + offset
                copies = [
                    (ptr32(src), ptr32(bases[target] + assembled + offset), count)
                    for target in local
                ]
            sources, destinations, lengths = zip(*copies)
            transfer(sources, destinations, lengths, active_blocks, cu_stream)
            main.wait_event(finished)
            if step == 3:
                advance_owner(
                    ptr32(slab_addresses[rank]), blocks, active_blocks, cu_stream
                )
            barrier()

        copy_spans(tail_copy_spans, int(cu_stream))

    return attach_programs(
        PacketPrograms(run), advance_owner, transfer, rendezvous, rendezvous3
    )


@program_cache
def get_row_to_col_all_to_all_programs(rank, threads, device_index):
    del device_index
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
    advance_owner = _compile(
        "comm.pcie.hybrid.advance_owner_epoch",
        (),
        _AdvanceOwnerEpoch(),
        ptr,
        Int32(1),
        Int32(1),
    )
    pack_local = _compile(
        "comm.pcie.hybrid.a2a_pack_local_push",
        (threads,),
        _A2APackLocalPush(threads),
        ptr,
        (ptr,) * 4,
        Int64(1),
        Int64(1),
        Int32(0),
        Int32(1),
    )
    island, lane = divmod(rank, 4)
    distances = tuple(
        tuple(1 + (owner + step) % 3 for step in range(3)) for owner in range(4)
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
        st,
        c_elems,
        blocks,
        stream=None,
        active_blocks=None,
        timing_address=0,
        timing_calls=0,
        timing_blocks=0,
        timing_slices=0,
    ):
        del stage_offset, part_offset, timing_address, timing_blocks, timing_slices
        (
            main,
            cu_stream,
            transport,
            transport_stream,
            started,
            finished,
            active_blocks,
            quarter_bytes,
            _packet_bytes,
            _stage,
            _inbox,
            assembled,
            bases,
            local_base,
            ptr32,
            barrier,
            barrier_local,
        ) = _setup_packet_context(
            rank,
            link,
            slab_addresses,
            final_offset,
            quarter_capacity,
            blocks,
            active_blocks,
            stream,
            timing_calls,
            rendezvous,
            rendezvous3,
        )
        if st <= 0 or c_elems <= 0 or c_elems % 2 != 0:
            raise ValueError(
                f"island row_to_col_all_to_all requires positive st and even c, got ({st}, {c_elems})"
            )
        c_words = c_elems // 2
        total_bytes = st * 16 * c_words * 4
        hop2_delta = 2 * quarter_bytes
        if total_bytes > hop2_delta:
            raise ValueError(
                "island row_to_col_all_to_all payload exceeds half of assembled slab"
            )
        chunk_bytes = total_bytes // 4

        # Ensure any preceding collective's trailing copy out of assembled has retired.
        barrier()

        # Hop 1: pack and push columns to all four lanes in our own island.
        local_dsts = tuple(
            ptr32(bases[4 * island + l_dst] + assembled) for l_dst in range(4)
        )
        pack_local(
            ptr32(input_address),
            local_dsts,
            Int64(st),
            Int64(c_words),
            Int32(lane),
            Int32(active_blocks),
            cu_stream,
        )
        barrier_local()

        # Hop 2: cross-island CE DMA of each pre-packed island chunk into hop2.
        started.record(main)
        transport.wait_event(started)
        for step in range(3):
            peer_island = island ^ distances[lane][step]
            target = 4 * peer_island + lane
            src = local_base + assembled + peer_island * chunk_bytes
            dst = bases[target] + assembled + hop2_delta + island * chunk_bytes
            link.ipc.cudaMemcpyAsync(dst, src, chunk_bytes, int(transport_stream))
        finished.record(transport)

        # Copy own island's chunk directly to output on main while cross-island DMA runs.
        link.ipc.cudaMemcpyAsync(
            output_address + island * chunk_bytes,
            local_base + assembled + island * chunk_bytes,
            chunk_bytes,
            int(cu_stream),
        )
        main.wait_event(finished)
        advance_owner(ptr32(slab_addresses[rank]), blocks, active_blocks, cu_stream)
        barrier()

        for begin_island, count_islands in _coalesced_shard_spans(
            [j for j in range(4) if j != island]
        ):
            offset = begin_island * chunk_bytes
            link.ipc.cudaMemcpyAsync(
                output_address + offset,
                local_base + assembled + hop2_delta + offset,
                count_islands * chunk_bytes,
                int(cu_stream),
            )

    return attach_programs(
        PacketPrograms(run), advance_owner, pack_local, rendezvous, rendezvous3
    )
