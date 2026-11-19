# SPDX-License-Identifier: Apache-2.0
"""Gather four Kimi latent and router pairs and select 16 experts in each row CTA."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import current_cuda_stream
from ._cute_intrinsics import ld_relaxed_gpu_u32
from ._dcp_a2a_cute import (
    _DCPA2ABase,
    _GRAPH_ARRIVED_INDEX,
    _GRAPH_EPOCH_INDEX,
    _MAX_BLOCKS,
    _SLOT_ALIGNMENT,
    _a2a_graph_epoch_arrive,
    _copy_16b,
    _u32_ptr,
    _u8_ptr,
    _kimi_key_expert,
    _kimi_pack_key,
    _kimi_warp_max_key,
    _f32_ptr,
    _i32_ptr,
)
from ._dcp_cute_common import block_pair_barrier


class GatherRouterTopK(_DCPA2ABase):
    @cute.jit
    def __call__(
        self,
        first: cute.Pointer,
        second: cute.Pointer,
        out_first: cute.Pointer,
        out_second: cute.Pointer,
        bias: cute.Pointer,
        weights: cute.Pointer,
        ids: cute.Pointer,
        stages: tuple,
        signals: tuple,
        stride: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            first,
            second,
            out_first,
            out_second,
            bias,
            weights,
            ids,
            stages,
            signals,
            stride,
        ).launch(grid=(4, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(
        self,
        first,
        second,
        out_first,
        out_second,
        bias,
        weights,
        ids,
        staging,
        signals,
        stride,
    ):
        tid, _, _ = cute.arch.thread_idx()
        first_packs, second_packs = Int32(28), Int32(14)
        slot_offset = Int64(0)
        if cutlass.const_expr(self._device_slot_selection):
            epoch = ld_relaxed_gpu_u32(
                (signals[self._rank] + Int64(_GRAPH_EPOCH_INDEX)).toint()
            )
            slot_offset = (
                Int64(epoch % Uint32(2)) * Int64(stride) * Int64(_SLOT_ALIGNMENT)
            )
        row, _, _ = cute.arch.block_idx()
        combined = first_packs + second_packs
        base = Int64(row) * Int64(combined) * 4
        local = self._staging_words(staging[self._rank] + slot_offset) + base
        if tid < first_packs:
            _copy_16b(
                first + Int64(row * first_packs + tid) * 4, local + Int64(tid) * 4
            )
        if tid < second_packs:
            _copy_16b(
                second + Int64(row * second_packs + tid) * 4,
                local + Int64(first_packs + tid) * 4,
            )
        block_pair_barrier(
            signals,
            self_signal=signals[self._rank],
            rank=self._rank,
            world_size=self._world_size,
            max_blocks=_MAX_BLOCKS,
        )
        # One warp reads both fields of one peer. Each field fits in 32 aligned packs.
        lane, warp = tid % 32, tid // 32
        for peer in cutlass.range_constexpr(self._world_size):
            if warp == Int32(peer):
                source = self._staging_words(staging[peer] + slot_offset) + base
                out_row = Int64(row) * self._world_size + peer
                if lane < first_packs:
                    _copy_16b(
                        source + Int64(lane) * 4,
                        out_first + (out_row * first_packs + lane) * 4,
                    )
                if lane < second_packs:
                    _copy_16b(
                        source + Int64(first_packs + lane) * 4,
                        out_second + (out_row * second_packs + lane) * 4,
                    )
        cute.arch.sync_threads()
        if cutlass.const_expr(self._device_slot_selection):
            if tid == 0:
                flags = signals[self._rank]
                _a2a_graph_epoch_arrive(
                    (flags + Int64(_GRAPH_EPOCH_INDEX)).toint(),
                    (flags + Int64(_GRAPH_ARRIVED_INDEX)).toint(),
                    Uint32(4),
                )

        self.select(cute.recast_ptr(out_second, dtype=Float32), bias, weights, ids)

    @cute.jit
    def select(self, router_logits, correction_bias, output_weights, output_ids):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = Int32(tidx) % Int32(32)
        warp = Int32(tidx) // Int32(32)
        smem_alloc = cutlass.utils.SmemAllocator()

        @cute.struct
        class SharedStorage:
            unbiased_scores: cute.struct.Align[cute.struct.MemRange[Float32, 896], 16]
            warp_keys: cute.struct.Align[cute.struct.MemRange[Uint64, 128], 16]

        storage = smem_alloc.allocate(SharedStorage)
        unbiased_scores = storage.unbiased_scores.get_tensor(
            cute.make_layout((896,), stride=(1,))
        )
        warp_keys = storage.warp_keys.get_tensor(cute.make_layout((128,), stride=(1,)))
        if tidx < Int32(256):
            keys = cute.make_rmem_tensor((4,), Uint64)
            row_offset = Int64(bidx) * Int64(896)
            for item in cutlass.range_constexpr(4):
                expert = Int32(tidx) + Int32(item * 256)
                key = Uint64(0)
                if expert < Int32(896):
                    router_value = cute.arch.load(
                        router_logits + row_offset + Int64(expert), Float32
                    )
                    bias = cute.arch.load(correction_bias + Int64(expert), Float32)
                    unbiased = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-router_value, fastmath=True)
                    )
                    if not cute.math.isfinite(unbiased):
                        unbiased = Float32(0.0)
                    selection = unbiased + bias
                    if not cute.math.isfinite(selection):
                        if selection > Float32(0.0):
                            selection = Float32(float("inf"))
                        else:
                            selection = Float32(float("-inf"))
                    if selection == Float32(0.0):
                        selection = Float32(0.0)
                    key = _kimi_pack_key(selection, expert)
                    unbiased_scores[expert] = unbiased
                keys[item] = key

            # Packed keys are unique because they include the expert ID. Each warp
            # therefore removes exactly one candidate per selection without a CTA
            # barrier or an active-expert table in shared memory.
            for selected in cutlass.range_constexpr(16):
                # Score keys occupy the unsigned high half; load register elements
                # explicitly in that domain before reducing them.
                thread_key = Uint64(keys[0])
                for item in cutlass.range_constexpr(1, 4):
                    thread_key = cutlass.max(thread_key, Uint64(keys[item]))
                winner = _kimi_warp_max_key(thread_key)
                if lane == Int32(0):
                    warp_keys[warp * Int32(16) + Int32(selected)] = winner
                for item in cutlass.range_constexpr(4):
                    if keys[item] == winner:
                        keys[item] = Uint64(0)
        cute.arch.sync_threads()

        if warp == Int32(0):
            cursor = Int32(0)
            head = Uint64(0)
            if lane < Int32(8):
                head = warp_keys[lane * Int32(16)]
            selected_ids = cute.make_rmem_tensor((16,), Int32)
            selected_weights = cute.make_rmem_tensor((16,), Float32)
            for selected in cutlass.range_constexpr(16):
                winner = _kimi_warp_max_key(head)
                expert = _kimi_key_expert(winner)
                if lane == Int32(0):
                    selected_ids[selected] = expert
                    selected_weights[selected] = unbiased_scores[expert]
                owner_warp = (expert % Int32(256)) // Int32(32)
                if lane == owner_warp:
                    cursor += Int32(1)
                    head = Uint64(0)
                    if cursor < Int32(16):
                        head = warp_keys[lane * Int32(16) + cursor]
            if lane == Int32(0):
                weight_sum = Float32(0.0)
                for selected in cutlass.range_constexpr(16):
                    weight_sum += selected_weights[selected]
                denominator = Float32(1.0)
                if weight_sum > Float32(0.0):
                    denominator = weight_sum
                scale = Float32(1.0) / denominator
                output_offset = Int64(bidx) * Int64(16)
                for selected in cutlass.range_constexpr(16):
                    cute.arch.store(
                        output_weights + output_offset + Int64(selected),
                        selected_weights[selected] * scale,
                    )
                    cute.arch.store(
                        output_ids + output_offset + Int64(selected),
                        selected_ids[selected],
                    )


@program_cache
def compile_row_route(rank, device_slot_selection):
    launch = GatherRouterTopK(16, rank, 512, device_slot_selection)
    words = _u32_ptr(16, align=4)
    raw = compile_kernel(
        launch,
        words,
        words,
        words,
        words,
        _f32_ptr(16),
        _f32_ptr(16),
        _i32_ptr(16),
        (_u8_ptr(16),) * 16,
        (words,) * 16,
        1,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "comm.pcie.row_gather_kimi_topk",
            1,
            (rank, device_slot_selection),
            labels=("rank", "device_slot_selection"),
        ),
    )

    def run(down, router, bias, latent, logits, weights, ids, stages, signals, delta):
        raw(
            _u32_ptr(down, align=4),
            _u32_ptr(router, align=4),
            _u32_ptr(latent, align=4),
            _u32_ptr(logits, align=4),
            _f32_ptr(bias),
            _f32_ptr(weights),
            _i32_ptr(ids),
            tuple(_u8_ptr(v) for v in stages),
            tuple(_u32_ptr(v, align=4) for v in signals),
            delta,
            current_cuda_stream(),
        )

    return attach_programs(run, raw)


def compile_four_row_route(rank, device_slot_selection, fused, threads):
    """Compile the complete four-row exchange, including selection and its visible logits."""
    if fused:
        if threads != 512:
            raise ValueError("one warp per remote rank requires a 512-thread CTA")
        return compile_row_route(rank, device_slot_selection)
    from ._dcp_a2a_cute import _get_compiled_all_gather_pair, _get_compiled_kimi_topk16

    gather = _get_compiled_all_gather_pair(
        16, rank, threads, device_slot_selection, False
    )
    select = _get_compiled_kimi_topk16(256)

    def run(down, router, bias, latent, logits, weights, ids, stages, signals, delta):
        gather(
            down,
            router,
            router,
            latent,
            logits,
            logits,
            stages,
            signals,
            4,
            28,
            14,
            delta,
        )
        select(logits, bias, weights, ids, 4)

    return attach_programs(run, gather, select)
