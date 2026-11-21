"""Bounded expert-route metadata produced beside CSF scale expansion."""

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64


class CsfRoutePack:
    def __init__(self, geometry):
        self.capacity, self.block_m, self.slots, self.blocks = geometry
        self.rounded = 1 << (self.capacity - 1).bit_length()
        self.levels = (self.rounded - 1).bit_length()
        exchanges = sum(max(0, level - 5) for level in range(1, self.levels + 1))
        self.key_offset = (exchanges % 2) * self.rounded
        self.scan_offset = (1 - exchanges % 2) * self.rounded

    @cute.jit
    def run(
        self,
        ids,
        routes,
        experts,
        count,
        live: Int32,
        num_experts: Int32,
        tid: Int32,
        threads: cutlass.Constexpr,
        shared=None,
    ):
        if cutlass.const_expr(shared is None):
            smem = cutlass.utils.SmemAllocator()
            shared = smem.allocate_tensor(
                Int32, cute.make_layout(2 * self.rounded), byte_alignment=16
            )
        else:
            shared = cute.recast_tensor(shared, Int32)
        slot = tid
        while slot < Int32(self.slots):
            routes[slot] = live
            slot += Int32(threads)
        slot = tid
        while slot < Int32(self.blocks):
            experts[slot] = Int32(-1)
            slot += Int32(threads)
        if tid == Int32(0):
            count[Int32(0)] = Int32(0)

        if cutlass.const_expr(self.rounded > threads):
            # Two-elements-per-thread shared bitonic sort & Hillis-Steele scans
            # support route capacities up to 2 * threads (e.g. 512 with 256 threads).
            i = tid
            while i < Int32(self.rounded):
                exp = num_experts
                if i < live:
                    raw = ids[i].to(Int64)
                    if raw >= Int64(0) and raw < num_experts.to(Int64):
                        exp = raw.to(Int32)
                shared[i] = exp * Int32(self.rounded) + i
                i += Int32(threads)
            cute.arch.sync_threads()

            half = self.rounded // 2
            for level in cutlass.range_constexpr(1, self.levels + 1):
                for step in cutlass.range_constexpr(level):
                    distance = 1 << (level - step - 1)
                    if tid < Int32(half):
                        lo = (
                            (tid >> Int32(level - step - 1)) << Int32(level - step)
                        ) | (tid & Int32(distance - 1))
                        hi = lo | Int32(distance)
                        k_lo = shared[lo]
                        k_hi = shared[hi]
                        asc = (lo & Int32(1 << level)) == Int32(0)
                        if (asc and k_lo > k_hi) or ((not asc) and k_lo < k_hi):
                            shared[lo] = k_hi
                            shared[hi] = k_lo
                    cute.arch.sync_threads()

            i = tid
            while i < Int32(self.rounded):
                exp = shared[i] // Int32(self.rounded)
                is_first = i == Int32(0)
                if i > Int32(0):
                    is_first = exp != shared[i - Int32(1)] // Int32(self.rounded)
                shared[Int32(self.rounded) + i] = i if is_first else Int32(0)
                i += Int32(threads)
            cute.arch.sync_threads()

            for d_shift in cutlass.range_constexpr(self.levels):
                d = 1 << d_shift
                v0 = Int32(0)
                v1 = Int32(0)
                if tid < Int32(self.rounded):
                    v0 = shared[Int32(self.rounded) + tid]
                    if tid >= Int32(d):
                        p0 = shared[Int32(self.rounded) + tid - Int32(d)]
                        if p0 > v0:
                            v0 = p0
                i1 = tid + Int32(threads)
                if i1 < Int32(self.rounded):
                    v1 = shared[Int32(self.rounded) + i1]
                    if i1 >= Int32(d):
                        p1 = shared[Int32(self.rounded) + i1 - Int32(d)]
                        if p1 > v1:
                            v1 = p1
                cute.arch.sync_threads()
                if tid < Int32(self.rounded):
                    shared[Int32(self.rounded) + tid] = v0
                if i1 < Int32(self.rounded):
                    shared[Int32(self.rounded) + i1] = v1
                cute.arch.sync_threads()

            s0 = (
                shared[Int32(self.rounded) + tid]
                if tid < Int32(self.rounded)
                else Int32(0)
            )
            p0 = Int32(0)
            if tid > Int32(0) and tid < Int32(self.rounded):
                prev_s = shared[Int32(self.rounded) + tid - Int32(1)]
                if s0 == tid:
                    p0 = (
                        Int32(self.block_m)
                        - (tid - prev_s) % Int32(self.block_m)
                    ) % Int32(self.block_m)
            s1 = Int32(0)
            p1 = Int32(0)
            i1 = tid + Int32(threads)
            if i1 < Int32(self.rounded):
                s1 = shared[Int32(self.rounded) + i1]
                prev_s1 = shared[Int32(self.rounded) + i1 - Int32(1)]
                if s1 == i1:
                    p1 = (
                        Int32(self.block_m)
                        - (i1 - prev_s1) % Int32(self.block_m)
                    ) % Int32(self.block_m)
            cute.arch.sync_threads()
            if tid < Int32(self.rounded):
                shared[Int32(self.rounded) + tid] = p0
            if i1 < Int32(self.rounded):
                shared[Int32(self.rounded) + i1] = p1
            cute.arch.sync_threads()

            for d_shift in cutlass.range_constexpr(self.levels):
                d = 1 << d_shift
                v0 = Int32(0)
                v1 = Int32(0)
                if tid < Int32(self.rounded):
                    v0 = shared[Int32(self.rounded) + tid]
                    if tid >= Int32(d):
                        v0 += shared[Int32(self.rounded) + tid - Int32(d)]
                if i1 < Int32(self.rounded):
                    v1 = shared[Int32(self.rounded) + i1]
                    if i1 >= Int32(d):
                        v1 += shared[Int32(self.rounded) + i1 - Int32(d)]
                cute.arch.sync_threads()
                if tid < Int32(self.rounded):
                    shared[Int32(self.rounded) + tid] = v0
                if i1 < Int32(self.rounded):
                    shared[Int32(self.rounded) + i1] = v1
                cute.arch.sync_threads()

            for pass_id in cutlass.range_constexpr(2):
                idx = tid + Int32(pass_id * threads)
                st = s0 if pass_id == 0 else s1
                if idx < Int32(self.rounded):
                    key = shared[idx]
                    exp = key // Int32(self.rounded)
                    pad = shared[Int32(self.rounded) + idx]
                    out_slot = idx + pad
                    if exp < num_experts:
                        routes[out_slot] = key % Int32(self.rounded)
                        if (idx - st) % Int32(self.block_m) == Int32(0):
                            experts[out_slot // Int32(self.block_m)] = exp
                        is_last = idx == Int32(self.rounded - 1)
                        if idx < Int32(self.rounded - 1):
                            is_last = (
                                shared[idx + Int32(1)] // Int32(self.rounded)
                                == num_experts
                            )
                        if is_last:
                            count[Int32(0)] = (
                                out_slot // Int32(self.block_m) + Int32(1)
                            ) * Int32(self.block_m)
        else:
            expert = num_experts
            if tid < live:
                raw = ids[tid].to(Int64)
                if raw >= Int64(0) and raw < num_experts.to(Int64):
                    expert = raw.to(Int32)
            key = expert * Int32(self.rounded) + tid
            for level in cutlass.range_constexpr(1, self.levels + 1):
                for step in cutlass.range_constexpr(level):
                    distance = 1 << (level - step - 1)
                    other = key
                    if cutlass.const_expr(distance < 32):
                        other = cute.arch.shuffle_sync_bfly(key, offset=distance)
                    else:
                        # Alternate halves so writes preserve the preceding exchange.
                        exchange = (level - 6) * (level - 5) // 2 + step
                        offset = (exchange % 2) * self.rounded
                        if tid < Int32(self.rounded):
                            shared[Int32(offset) + tid] = key
                        cute.arch.sync_threads()
                        if tid < Int32(self.rounded):
                            other = shared[Int32(offset) + (tid ^ Int32(distance))]
                    take_min = ((tid & Int32(1 << level)) == Int32(0)) == (
                        (tid & Int32(distance)) == Int32(0)
                    )
                    if take_min:
                        if other < key:
                            key = other
                    else:
                        if other > key:
                            key = other
            if tid < Int32(self.rounded):
                shared[Int32(self.key_offset) + tid] = key
            cute.arch.sync_threads()
            expert = key // Int32(self.rounded)
            first = tid == Int32(0)
            if tid > Int32(0) and tid < Int32(self.rounded):
                first = expert != shared[
                    Int32(self.key_offset) + tid - Int32(1)
                ] // Int32(self.rounded)
            start = Int32(0)
            if first:
                start = tid
            lane = tid % Int32(32)
            warp = tid // Int32(32)
            for step in cutlass.range_constexpr(5):
                prior = cute.arch.shuffle_sync_up(start, offset=1 << step)
                if lane >= Int32(1 << step) and prior > start:
                    start = prior
            if tid < Int32(self.rounded):
                shared[Int32(self.scan_offset) + tid] = start
            cute.arch.sync_threads()
            carry = Int32(0)
            for preceding in cutlass.range_constexpr(self.rounded // 32):
                if warp > Int32(preceding):
                    prior = shared[Int32(self.scan_offset + (preceding + 1) * 32 - 1)]
                    if prior > carry:
                        carry = prior
            if carry > start:
                start = carry
            previous_start = cute.arch.shuffle_sync_up(start, offset=1)
            if lane == Int32(0):
                previous_start = carry
            padding = Int32(0)
            if first and tid > Int32(0) and tid < Int32(self.rounded):
                padding = (
                    Int32(self.block_m) - (tid - previous_start) % Int32(self.block_m)
                ) % Int32(self.block_m)
            for step in cutlass.range_constexpr(5):
                prior = cute.arch.shuffle_sync_up(padding, offset=1 << step)
                if lane >= Int32(1 << step):
                    padding += prior
            cute.arch.sync_threads()
            if tid < Int32(self.rounded):
                shared[Int32(self.scan_offset) + tid] = padding
            cute.arch.sync_threads()
            for preceding in cutlass.range_constexpr(self.rounded // 32):
                if warp > Int32(preceding):
                    padding += shared[
                        Int32(self.scan_offset + (preceding + 1) * 32 - 1)
                    ]
            slot = tid + padding
            if tid < Int32(self.rounded) and expert < num_experts:
                routes[slot] = key % Int32(self.rounded)
                if (tid - start) % Int32(self.block_m) == Int32(0):
                    experts[slot // Int32(self.block_m)] = expert
                last = tid == Int32(self.rounded - 1)
                if tid < Int32(self.rounded - 1):
                    last = (
                        shared[Int32(self.key_offset) + tid + Int32(1)]
                        // Int32(self.rounded)
                        == num_experts
                    )
                if last:
                    count[Int32(0)] = (slot // Int32(self.block_m) + Int32(1)) * Int32(
                        self.block_m
                    )
