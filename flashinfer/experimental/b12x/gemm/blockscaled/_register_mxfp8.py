# Copyright 2026 Fraser Price
# Copyright (c) 2025-2026 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Register-resident MXFP8 projection for at most eight activation rows.

Adapted from the register staging and alternating MMA chains in Fraser BRRR's
``skinny.cu`` and ``fp8_rows.cuh``. The operands retain b12x's ordinary MXFP8
payload and F8_128x4 scale storage; activation quantization remains a separate
prepared operator with its existing numerical contract.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_v4_u32
from b12x._lib.intrinsics.mma import mxfp8_mma_m16n8k32_f32_e4m3
from b12x._lib.utils.device import current_cuda_stream
from b12x._lib.utils.pointers import make_ptr


@cute.jit
def _flat(ptr):
    return cute.make_tensor(ptr, cute.make_layout(Int64(1) << Int64(40)))


@cute.jit
def _scale_offset(row, group, groups: cutlass.Constexpr):
    return (
        (row // Int32(128) * Int32((groups + 3) // 4) + group // Int32(4)) * Int32(512)
        + row % Int32(32) * Int32(16)
        + row // Int32(32) % Int32(4) * Int32(4)
        + group % Int32(4)
    )


class RegisterMXFP8:
    def __init__(self, k: int, n: int):
        if k not in (1280, 5120) or n % 32:
            raise ValueError("register MXFP8 requires K1280/K5120 and N32")
        self.k, self.n = k, n
        self.tiles = 1 if k == 5120 else 4 if n >= 8192 else 2
        self.stride = 8 // self.tiles
        self.units = (k // 64) // self.stride
        if self.units * self.stride != k // 64:
            raise ValueError("register MXFP8 K partitions must be exact")

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        xs: cute.Pointer,
        w: cute.Pointer,
        ws: cute.Pointer,
        y: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(_flat(x), _flat(xs), _flat(w), _flat(ws), _flat(y), m).launch(
            grid=(self.n // (8 * self.tiles), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, xs, w, ws, y, m):
        tx, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        tx, bx = Int32(tx), Int32(bx)
        warp, lane = tx // Int32(32), tx % Int32(32)
        g, t = lane // Int32(4), lane % Int32(4)
        tile = bx * Int32(self.tiles) + warp % Int32(self.tiles)
        kp0 = warp // Int32(self.tiles)
        smem = utils.SmemAllocator()
        # Pad row pitch by sixteen bytes so neighboring activation rows rotate
        # shared banks while every lane keeps aligned eight-byte MMA fragments.
        pitch = self.k + 16
        xq = smem.allocate_tensor(Uint32, cute.make_layout(8 * pitch // 4), 16)
        xscale = smem.allocate_tensor(Uint8, cute.make_layout(8 * self.k // 32), 16)
        red = smem.allocate_tensor(Float32, cute.make_layout(8 * 8 * 8), 16)
        weights = cute.make_rmem_tensor((self.units, 4), Uint32)
        scales = cute.make_rmem_tensor((self.units,), Uint32)
        for j in cutlass.range_constexpr(self.units):
            kp = kp0 + Int32(j * self.stride)
            off = (Int64(tile) * Int64(8) + Int64(g)) * Int64(self.k)
            off += Int64(kp) * Int64(64) + Int64(t) * Int64(16)
            v0, v1, v2, v3 = ld_global_v4_u32(get_ptr_as_int64(w, off))
            weights[j, 0], weights[j, 1] = v0, v1
            weights[j, 2], weights[j, 3] = v2, v3
            r = tile * Int32(8) + g
            group = kp * Int32(2)
            s0 = Uint32(ws[_scale_offset(r, group, self.k // 32)])
            s2 = Uint32(ws[_scale_offset(r, group + Int32(1), self.k // 32)])
            scales[j] = s0 | (s2 << Uint32(16))
        for step in cutlass.range_constexpr((8 * self.k // 16 + 255) // 256):
            v = tx + Int32(step * 256)
            if v < Int32(8 * self.k // 16):
                row, col = v // Int32(self.k // 16), v % Int32(self.k // 16)
                a0, a1, a2, a3 = Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                if row < m:
                    a0, a1, a2, a3 = ld_global_v4_u32(
                        get_ptr_as_int64(
                            x, Int64(row) * Int64(self.k) + Int64(col * Int32(16))
                        )
                    )
                base = row * Int32(pitch // 4) + col * Int32(4)
                xq[base], xq[base + Int32(1)] = a0, a1
                xq[base + Int32(2)], xq[base + Int32(3)] = a2, a3
        for step in cutlass.range_constexpr((8 * self.k // 32 + 255) // 256):
            i = tx + Int32(step * 256)
            if i < Int32(8 * self.k // 32):
                row, group = i // Int32(self.k // 32), i % Int32(self.k // 32)
                value = Uint8(127)
                if row < m:
                    value = Uint8(xs[_scale_offset(row, group, self.k // 32)])
                xscale[i] = value
        cute.arch.sync_threads()
        low = t < Int32(2)
        for j in cutlass.range_constexpr(self.units):
            v0, v1, v2, v3 = (
                weights[j, 0],
                weights[j, 1],
                weights[j, 2],
                weights[j, 3],
            )
            r0 = cute.arch.shuffle_sync_bfly((v2 if low else v0), offset=2)
            r1 = cute.arch.shuffle_sync_bfly((v3 if low else v1), offset=2)
            weights[j, 0] = v0 if low else r0
            weights[j, 1] = v1 if low else r1
            weights[j, 2] = r0 if low else v2
            weights[j, 3] = r1 if low else v3
        acc = [Float32(0), Float32(0), Float32(0), Float32(0)]
        odd = [Float32(0), Float32(0), Float32(0), Float32(0)]
        piece = Int32(2) * (t % Int32(2)) + t // Int32(2)
        for j in cutlass.range_constexpr(self.units):
            kb = (kp0 + Int32(j * self.stride)) * Int32(2)
            base = g * Int32(pitch // 4) + piece * Int32(2) + kb * Int32(8)
            a0, a2 = Uint32(xq[base]), Uint32(xq[base + Int32(1)])
            a1, a3 = Uint32(0), Uint32(0)
            sa = Uint32(xscale[g * Int32(self.k // 32) + kb])
            acc[0], acc[1], acc[2], acc[3] = mxfp8_mma_m16n8k32_f32_e4m3(
                *acc,
                a0,
                a1,
                a2,
                a3,
                weights[j, 0],
                weights[j, 1],
                sa,
                scales[j],
            )
            a0, a2 = Uint32(xq[base + Int32(8)]), Uint32(xq[base + Int32(9)])
            sa = Uint32(xscale[g * Int32(self.k // 32) + kb + Int32(1)])
            odd[0], odd[1], odd[2], odd[3] = mxfp8_mma_m16n8k32_f32_e4m3(
                *odd,
                a0,
                a1,
                a2,
                a3,
                weights[j, 2],
                weights[j, 3],
                sa,
                scales[j],
                bid_b=2,
            )
        base = (warp * Int32(8) + g) * Int32(8) + t * Int32(2)
        red[base], red[base + Int32(1)] = acc[0] + odd[0], acc[1] + odd[1]
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr((self.tiles * 64 + 255) // 256):
            idx = tx + Int32(step * 256)
            if idx < Int32(self.tiles * 64):
                i, row, n = idx // Int32(64), idx // Int32(8) % Int32(8), idx % Int32(8)
                total = Float32(0)
                for part in cutlass.range_constexpr(8 // self.tiles):
                    src_warp = i + Int32(part * self.tiles)
                    total += red[(src_warp * Int32(8) + row) * Int32(8) + n]
                if row < m:
                    y[
                        Int64(row) * Int64(self.n)
                        + Int64((bx * Int32(self.tiles) + i) * Int32(8) + n)
                    ] = BFloat16(total)


def compile_register(k, n):
    obj = RegisterMXFP8(k, n)
    types = (Uint8, Uint8, Uint8, Uint8, BFloat16)
    ptrs = tuple(
        make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=16) for t in types
    )
    return b12x_compile(
        obj,
        *ptrs,
        Int32(1),
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "gemm.blockscaled.register_mxfp8", 1, (k, n)
        ),
    )
