"""SM120 cluster-parallel split-lagged mHC kernels for decode, verify, and draft (M <= 8).

In lagged hyperconnections, the branch input ``y = RMSNorm(contract(pre_mix, residual_out))``
depends only on ``residual_out`` and ``pre_mix`` — not on ``fn @ residual_out`` or the
Sinkhorn projection (``pre_out, post_out, comb_out``), which are not consumed until after
the sublayer's tensor-parallel AllReduce.

Splitting mHC into two stages separates the critical path from the coefficient projection:
1. ``ClusterMhcPostYNorm`` (main stream): SM120 cluster-parallel post-update, lagged
   contraction, and distributed-shared-memory-reduced RMSNorm, with optional bit-exact
   MXFP8 quantization matching ``quantize_mxfp8_rows_cute``.
2. ``ReadOnlyFnPartialM8`` + ``_SplitFinalize`` (auxiliary stream): reads the finalized
   residual stream and computes ``fn @ residual_out`` plus Sinkhorn normalization
   concurrently with the sublayer's tensor-parallel AllReduce.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile, run_compiled
from b12x._lib.intrinsics.constants import FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_f32x4_to_e4m3x4,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import fabs_f32, fmax_f32
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    ld_global_v4_u32,
    shared_ptr_to_u32,
    st_global_v4_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.mma import tf32_mma_m16n8k8_f32
from b12x._lib.intrinsics.packed import f32_to_raw_bits, f32_to_tf32_bits
from b12x._lib.program_cache import program_cache
from b12x._lib.utils.device import (
    cuda_stream_from_int_or_current,
    current_cuda_stream,
)
from b12x._lib.utils.pointers import make_ptr

from ._fused_tf32 import _pack_bf16_pair
from ._kernels import (
    _contract_four,
    _tf32_residual_bits,
    _warp_allreduce_sum,
    _warp_column4_allreduce_sum,
    _warp_quad_allreduce_max,
    _warp_quad_allreduce_sum,
)

SPLIT_LAGGED_MAX_TOKENS = 8
SPLIT_LAGGED_HIDDEN_SIZES = (4096, 5120, 7168)
_PARTIAL_STRIDE = 32


def supports_split_lagged(
    device: torch.device,
    max_tokens: int,
    hidden_size: int,
    norm_dtype: torch.dtype,
) -> bool:
    """Return True when SM120 cluster split-lagged mHC is supported for this geometry."""
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device)
    if major < 12:
        return False
    return (
        1 <= int(max_tokens) <= SPLIT_LAGGED_MAX_TOKENS
        and int(hidden_size) in SPLIT_LAGGED_HIDDEN_SIZES
        and norm_dtype in (torch.bfloat16, torch.float32)
    )


def split_lagged_partials_shape(tokens: int, hidden_size: int) -> tuple[int, int, int]:
    return (int(tokens), int(hidden_size) // 32, _PARTIAL_STRIDE)


@dsl_user_op
def _ld_dsm_f32(smem_u32: Uint32, cta_rank: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(smem_u32).ir_value(loc=loc, ip=ip),
                Int32(cta_rank).ir_value(loc=loc, ip=ip),
            ],
            """{
                .reg .u32 remote_addr;
                mapa.shared::cluster.u32 remote_addr, $1, $2;
                ld.shared::cluster.f32 $0, [remote_addr];
            }""",
            "=f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class ClusterMhcPostYNorm:
    """SM120 cluster-parallel post-update + lagged contraction + RMSNorm [+ bit-exact MXFP8].

    Grid is ``(cluster_size, tokens, 1)`` with hardware cluster ``(cluster_size, 1, 1)``.
    For hidden sizes in (4096, 5120, 7168) with ``cluster_size = 8``, every CTA owns
    ``(hidden // 8) // 8`` 128-bit packs (64, 80, or 112 — always a multiple of 4 so
    every 4 consecutive threads hold 32 consecutive BF16 values = 1 exact MXFP8 block).
    All warp-level shuffles execute uniformly across all 32 lanes (outside ``if active``).
    """

    def __init__(
        self,
        hidden: int,
        pre_only: bool,
        norm_eps: float,
        norm_is_bf16: bool,
        cluster_size: int = 8,
        emit_mxfp8: bool = False,
    ):
        self.hidden = int(hidden)
        self.pre_only = bool(pre_only)
        self.norm_eps = float(norm_eps)
        self.norm_is_bf16 = bool(norm_is_bf16)
        self.packs = self.hidden // 8
        self.cluster_size = int(cluster_size)
        if self.packs % self.cluster_size != 0:
            raise ValueError(
                f"hidden packs ({self.packs}) must be divisible by cluster_size ({self.cluster_size})"
            )
        self.packs_per_cta = self.packs // self.cluster_size
        if self.packs_per_cta % 4 != 0:
            raise ValueError(
                f"packs_per_cta ({self.packs_per_cta}) must be a multiple of 4 for MXFP8 block alignment"
            )
        self.threads = ((self.packs_per_cta + 31) // 32) * 32
        self.num_warps = self.threads // 32
        self.emit_mxfp8 = bool(emit_mxfp8)

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        prev_post_ptr: cute.Pointer,
        prev_comb_ptr: cute.Pointer,
        incoming_ptr: cute.Pointer,
        norm_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        a_values_u32_ptr: cute.Pointer,
        a_scales_ptr: cute.Pointer,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x_ptr,
            res_ptr,
            cute.make_tensor(prev_post_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            cute.make_tensor(prev_comb_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            cute.make_tensor(incoming_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            norm_ptr,
            cute.make_tensor(norm_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            out_ptr,
            y_ptr,
            cute.make_tensor(
                a_values_u32_ptr, cute.make_layout((Int64(1) << Int64(30),))
            ),
            cute.make_tensor(a_scales_ptr, cute.make_layout((Int64(1) << Int64(30),))),
        ).launch(
            grid=(self.cluster_size, tokens, 1),
            block=(self.threads, 1, 1),
            cluster=(self.cluster_size, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        prev_post: cute.Tensor,
        prev_comb: cute.Tensor,
        incoming: cute.Tensor,
        norm_ptr: cute.Pointer,
        norm_t: cute.Tensor,
        out_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        a_values_u32: cute.Tensor,
        a_scales: cute.Tensor,
    ):
        c_rank, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32

        smem = utils.SmemAllocator()
        warp_sq = smem.allocate_tensor(Float32, cute.make_layout((16,)), 16)
        cta_sq = smem.allocate_tensor(Float32, cute.make_layout((1,)), 16)
        cta_sq_u32 = shared_ptr_to_u32(cta_sq.iterator)

        row4 = Int64(row) * Int64(4)
        in0 = Float32(incoming[row4 + Int64(0)])
        in1 = Float32(incoming[row4 + Int64(1)])
        in2 = Float32(incoming[row4 + Int64(2)])
        in3 = Float32(incoming[row4 + Int64(3)])
        c00 = Float32(0)
        c01 = Float32(0)
        c02 = Float32(0)
        c03 = Float32(0)
        c10 = Float32(0)
        c11 = Float32(0)
        c12 = Float32(0)
        c13 = Float32(0)
        c20 = Float32(0)
        c21 = Float32(0)
        c22 = Float32(0)
        c23 = Float32(0)
        c30 = Float32(0)
        c31 = Float32(0)
        c32 = Float32(0)
        c33 = Float32(0)
        p0 = Float32(0)
        p1 = Float32(0)
        p2 = Float32(0)
        p3 = Float32(0)
        if const_expr(not self.pre_only):
            cv = (
                Float32(prev_comb[Int64(row) * Int64(16) + Int64(lane)])
                if lane < 16
                else Float32(0)
            )
            pv = Float32(prev_post[row4 + Int64(lane)]) if lane < 4 else Float32(0)
            c00 = Float32(cute.arch.shuffle_sync(cv, Int32(0)))
            c01 = Float32(cute.arch.shuffle_sync(cv, Int32(4)))
            c02 = Float32(cute.arch.shuffle_sync(cv, Int32(8)))
            c03 = Float32(cute.arch.shuffle_sync(cv, Int32(12)))
            c10 = Float32(cute.arch.shuffle_sync(cv, Int32(1)))
            c11 = Float32(cute.arch.shuffle_sync(cv, Int32(5)))
            c12 = Float32(cute.arch.shuffle_sync(cv, Int32(9)))
            c13 = Float32(cute.arch.shuffle_sync(cv, Int32(13)))
            c20 = Float32(cute.arch.shuffle_sync(cv, Int32(2)))
            c21 = Float32(cute.arch.shuffle_sync(cv, Int32(6)))
            c22 = Float32(cute.arch.shuffle_sync(cv, Int32(10)))
            c23 = Float32(cute.arch.shuffle_sync(cv, Int32(14)))
            c30 = Float32(cute.arch.shuffle_sync(cv, Int32(3)))
            c31 = Float32(cute.arch.shuffle_sync(cv, Int32(7)))
            c32 = Float32(cute.arch.shuffle_sync(cv, Int32(11)))
            c33 = Float32(cute.arch.shuffle_sync(cv, Int32(15)))
            p0 = Float32(cute.arch.shuffle_sync(pv, Int32(0)))
            p1 = Float32(cute.arch.shuffle_sync(pv, Int32(1)))
            p2 = Float32(cute.arch.shuffle_sync(pv, Int32(2)))
            p3 = Float32(cute.arch.shuffle_sync(pv, Int32(3)))

        row_bytes = Int64(self.hidden * 2)
        res_base = Int64(res_ptr.toint()) + Int64(row) * Int64(4) * row_bytes
        out_base = Int64(out_ptr.toint()) + Int64(row) * Int64(4) * row_bytes
        x_base = Int64(x_ptr.toint()) + Int64(row) * row_bytes
        y_base = Int64(y_ptr.toint()) + Int64(row) * row_bytes
        norm_base = Int64(norm_ptr.toint())

        y_reg = cute.make_rmem_tensor((4,), Uint32)
        sq_y = Float32(0.0)
        pack = c_rank * self.packs_per_cta + tid
        active = tid < self.packs_per_cta
        if active:
            boff = Int64(pack) * Int64(16)
            rw0 = ld_global_v4_u32(res_base + Int64(0) * row_bytes + boff)
            rw1 = ld_global_v4_u32(res_base + Int64(1) * row_bytes + boff)
            rw2 = ld_global_v4_u32(res_base + Int64(2) * row_bytes + boff)
            rw3 = ld_global_v4_u32(res_base + Int64(3) * row_bytes + boff)
            xw = (
                ld_global_v4_u32(x_base + boff)
                if const_expr(not self.pre_only)
                else rw0
            )
            o0 = cute.make_rmem_tensor((4,), Uint32)
            o1 = cute.make_rmem_tensor((4,), Uint32)
            o2 = cute.make_rmem_tensor((4,), Uint32)
            o3 = cute.make_rmem_tensor((4,), Uint32)
            for wd in cutlass.range_constexpr(4):
                r0_lo = u32_as_f32(rw0[wd] << Uint32(16))
                r0_hi = u32_as_f32(rw0[wd] & Uint32(0xFFFF0000))
                r1_lo = u32_as_f32(rw1[wd] << Uint32(16))
                r1_hi = u32_as_f32(rw1[wd] & Uint32(0xFFFF0000))
                r2_lo = u32_as_f32(rw2[wd] << Uint32(16))
                r2_hi = u32_as_f32(rw2[wd] & Uint32(0xFFFF0000))
                r3_lo = u32_as_f32(rw3[wd] << Uint32(16))
                r3_hi = u32_as_f32(rw3[wd] & Uint32(0xFFFF0000))
                if const_expr(not self.pre_only):
                    xv_lo = u32_as_f32(xw[wd] << Uint32(16))
                    xv_hi = u32_as_f32(xw[wd] & Uint32(0xFFFF0000))
                    b0_lo = (
                        p0 * xv_lo
                        + c00 * r0_lo
                        + c01 * r1_lo
                        + c02 * r2_lo
                        + c03 * r3_lo
                    ).to(BFloat16)
                    b0_hi = (
                        p0 * xv_hi
                        + c00 * r0_hi
                        + c01 * r1_hi
                        + c02 * r2_hi
                        + c03 * r3_hi
                    ).to(BFloat16)
                    b1_lo = (
                        p1 * xv_lo
                        + c10 * r0_lo
                        + c11 * r1_lo
                        + c12 * r2_lo
                        + c13 * r3_lo
                    ).to(BFloat16)
                    b1_hi = (
                        p1 * xv_hi
                        + c10 * r0_hi
                        + c11 * r1_hi
                        + c12 * r2_hi
                        + c13 * r3_hi
                    ).to(BFloat16)
                    b2_lo = (
                        p2 * xv_lo
                        + c20 * r0_lo
                        + c21 * r1_lo
                        + c22 * r2_lo
                        + c23 * r3_lo
                    ).to(BFloat16)
                    b2_hi = (
                        p2 * xv_hi
                        + c20 * r0_hi
                        + c21 * r1_hi
                        + c22 * r2_hi
                        + c23 * r3_hi
                    ).to(BFloat16)
                    b3_lo = (
                        p3 * xv_lo
                        + c30 * r0_lo
                        + c31 * r1_lo
                        + c32 * r2_lo
                        + c33 * r3_lo
                    ).to(BFloat16)
                    b3_hi = (
                        p3 * xv_hi
                        + c30 * r0_hi
                        + c31 * r1_hi
                        + c32 * r2_hi
                        + c33 * r3_hi
                    ).to(BFloat16)
                    r0_lo, r0_hi = Float32(b0_lo), Float32(b0_hi)
                    r1_lo, r1_hi = Float32(b1_lo), Float32(b1_hi)
                    r2_lo, r2_hi = Float32(b2_lo), Float32(b2_hi)
                    r3_lo, r3_hi = Float32(b3_lo), Float32(b3_hi)
                o0[wd] = _pack_bf16_pair(r0_lo, r0_hi)
                o1[wd] = _pack_bf16_pair(r1_lo, r1_hi)
                o2[wd] = _pack_bf16_pair(r2_lo, r2_hi)
                o3[wd] = _pack_bf16_pair(r3_lo, r3_hi)
                yb_lo = _contract_four(
                    in0, in1, in2, in3, r0_lo, r1_lo, r2_lo, r3_lo
                ).to(BFloat16)
                yb_hi = _contract_four(
                    in0, in1, in2, in3, r0_hi, r1_hi, r2_hi, r3_hi
                ).to(BFloat16)
                yl, yh = Float32(yb_lo), Float32(yb_hi)
                sq_y += yl * yl + yh * yh
                y_reg[wd] = _pack_bf16_pair(yl, yh)
            st_global_v4_u32(
                out_base + Int64(0) * row_bytes + boff, o0[0], o0[1], o0[2], o0[3]
            )
            st_global_v4_u32(
                out_base + Int64(1) * row_bytes + boff, o1[0], o1[1], o1[2], o1[3]
            )
            st_global_v4_u32(
                out_base + Int64(2) * row_bytes + boff, o2[0], o2[1], o2[2], o2[3]
            )
            st_global_v4_u32(
                out_base + Int64(3) * row_bytes + boff, o3[0], o3[1], o3[2], o3[3]
            )

        # CTA-local reduction (all 32 lanes participate) -> cta_sq[0]
        wsum = _warp_allreduce_sum(sq_y)
        if lane == 0:
            warp_sq[warp] = wsum
        cute.arch.sync_threads()
        if warp == 0:
            ct = warp_sq[lane] if lane < self.num_warps else Float32(0.0)
            ct = _warp_allreduce_sum(ct)
            if lane == 0:
                cta_sq[0] = ct
        # Cluster-wide DSM reduction (~25ns)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        tot = Float32(0.0)
        for cr in cutlass.range_constexpr(self.cluster_size):
            tot += _ld_dsm_f32(cta_sq_u32, Int32(cr))
        inv = cute.math.rsqrt(
            tot / Float32(self.hidden) + Float32(self.norm_eps), fastmath=True
        )

        vals_f = cute.make_rmem_tensor((8,), Float32)
        max_abs = Float32(0.0)
        if active:
            boff = Int64(pack) * Int64(16)
            outw = cute.make_rmem_tensor((4,), Uint32)
            norm_aligned = (norm_base & Int64(15)) == Int64(0)
            if norm_aligned:
                if const_expr(self.norm_is_bf16):
                    ww = ld_global_v4_u32(norm_base + boff)
                    for wd in cutlass.range_constexpr(4):
                        yw = y_reg[wd]
                        v0 = u32_as_f32(yw << Uint32(16))
                        v1 = u32_as_f32(yw & Uint32(0xFFFF0000))
                        w0 = u32_as_f32(ww[wd] << Uint32(16))
                        w1 = u32_as_f32(ww[wd] & Uint32(0xFFFF0000))
                        p_u32 = _pack_bf16_pair(v0 * inv * w0, v1 * inv * w1)
                        outw[wd] = p_u32
                        if const_expr(self.emit_mxfp8):
                            vals_f[wd * 2] = u32_as_f32(p_u32 << Uint32(16))
                            vals_f[wd * 2 + 1] = u32_as_f32(p_u32 & Uint32(0xFFFF0000))
                else:
                    wlo = ld_global_v4_u32(norm_base + Int64(pack) * Int64(32))
                    whi = ld_global_v4_u32(
                        norm_base + Int64(pack) * Int64(32) + Int64(16)
                    )
                    ws = (
                        u32_as_f32(wlo[0]),
                        u32_as_f32(wlo[1]),
                        u32_as_f32(wlo[2]),
                        u32_as_f32(wlo[3]),
                        u32_as_f32(whi[0]),
                        u32_as_f32(whi[1]),
                        u32_as_f32(whi[2]),
                        u32_as_f32(whi[3]),
                    )
                    for wd in cutlass.range_constexpr(4):
                        yw = y_reg[wd]
                        v0 = u32_as_f32(yw << Uint32(16))
                        v1 = u32_as_f32(yw & Uint32(0xFFFF0000))
                        p_u32 = _pack_bf16_pair(
                            v0 * inv * ws[wd * 2], v1 * inv * ws[wd * 2 + 1]
                        )
                        outw[wd] = p_u32
                        if const_expr(self.emit_mxfp8):
                            vals_f[wd * 2] = u32_as_f32(p_u32 << Uint32(16))
                            vals_f[wd * 2 + 1] = u32_as_f32(p_u32 & Uint32(0xFFFF0000))
            else:
                col_base = Int64(pack) * Int64(8)
                for wd in cutlass.range_constexpr(4):
                    yw = y_reg[wd]
                    v0 = u32_as_f32(yw << Uint32(16))
                    v1 = u32_as_f32(yw & Uint32(0xFFFF0000))
                    w0 = Float32(norm_t[col_base + Int64(wd * 2)])
                    w1 = Float32(norm_t[col_base + Int64(wd * 2 + 1)])
                    p_u32 = _pack_bf16_pair(v0 * inv * w0, v1 * inv * w1)
                    outw[wd] = p_u32
                    if const_expr(self.emit_mxfp8):
                        vals_f[wd * 2] = u32_as_f32(p_u32 << Uint32(16))
                        vals_f[wd * 2 + 1] = u32_as_f32(p_u32 & Uint32(0xFFFF0000))
            st_global_v4_u32(y_base + boff, outw[0], outw[1], outw[2], outw[3])
            if const_expr(self.emit_mxfp8):
                max_abs = fabs_f32(vals_f[0])
                for e in cutlass.range_constexpr(1, 8):
                    max_abs = fmax_f32(max_abs, fabs_f32(vals_f[e]))

        if const_expr(self.emit_mxfp8):
            # Uniform across all 32 lanes in every warp (zero warp divergence!)
            max_abs = fmax_f32(max_abs, cute.arch.shuffle_sync_bfly(max_abs, offset=1))
            max_abs = fmax_f32(max_abs, cute.arch.shuffle_sync_bfly(max_abs, offset=2))
            if active:
                _, scale_byte = pow2_ceil_ue8m0(
                    max_abs * Float32(1.0 / FLOAT8_E4M3_MAX)
                )
                if max_abs == Float32(0.0):
                    scale_byte = Uint32(127)
                inv_sc = ue8m0_to_output_scale(scale_byte)
                word_base = Int64(row) * Int64(self.hidden // 4) + Int64(pack * 2)
                a_values_u32[word_base] = cvt_f32x4_to_e4m3x4(
                    vals_f[0] * inv_sc,
                    vals_f[1] * inv_sc,
                    vals_f[2] * inv_sc,
                    vals_f[3] * inv_sc,
                )
                a_values_u32[word_base + Int64(1)] = cvt_f32x4_to_e4m3x4(
                    vals_f[4] * inv_sc,
                    vals_f[5] * inv_sc,
                    vals_f[6] * inv_sc,
                    vals_f[7] * inv_sc,
                )
                if (lane & Int32(3)) == Int32(0):
                    a_scales[
                        Int64(row) * Int64(self.hidden // 32) + Int64(pack >> Int32(2))
                    ] = scale_byte.to(Uint8)

        # Hold shared memory until every CTA in the cluster has read cta_sq
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


class ReadOnlyFnPartialM8:
    """Read-only ``residual_out @ fn.T`` partial projection for M <= 8 (6 compensated TF32 warps)."""

    def __init__(self, hidden: int):
        self.hidden = int(hidden)
        self.splits = self.hidden // 32

    @cute.jit
    def __call__(
        self,
        res_ptr: cute.Pointer,
        fn_ptr: cute.Pointer,
        partials_ptr: cute.Pointer,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.make_tensor(res_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            fn_ptr,
            cute.make_tensor(partials_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            tokens,
        ).launch(
            grid=(self.splits, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        residual_out: cute.Tensor,
        fn_ptr: cute.Pointer,
        partials: cute.Tensor,
        tokens: Int32,
    ):
        split, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = utils.SmemAllocator()
        a = smem.allocate_tensor(
            BFloat16, cute.make_layout((8, 128), stride=(132, 1)), 16
        )
        b = smem.allocate_tensor(
            Float32, cute.make_layout((24, 128), stride=(132, 1)), 16
        )
        half_sum = smem.allocate_tensor(
            Float32, cute.make_layout((8, 24), stride=(24, 1)), 16
        )
        bbase = shared_ptr_to_u32(b.iterator)
        fn_base = Int64(fn_ptr.toint())
        hbase = split * 32
        for i in cutlass.range_constexpr(3):
            linear = i * 1024 + tid * 4
            n, k = linear // 128, linear % 128
            src = (
                Int64(n) * Int64(4 * self.hidden)
                + Int64(k // 32) * Int64(self.hidden)
                + Int64(hbase + k % 32)
            )
            dst = bbase + (n * 132 + k) * 4
            address = fn_base + src * Int64(4)
            if (address & Int64(15)) == Int64(0):
                cp_async4_shared_global(dst, address)
            else:
                for part in cutlass.range_constexpr(4):
                    cp_async_u32_shared_global(
                        dst + part * 4, address + Int64(part * 4)
                    )
        row = warp
        h = hbase + lane
        r0, r1, r2, r3 = Float32(0), Float32(0), Float32(0), Float32(0)
        sq_r_acc = Float32(0.0)
        if row < tokens:
            rbase = Int64(row) * Int64(4 * self.hidden) + Int64(h)
            r0 = Float32(residual_out[rbase + Int64(0 * self.hidden)])
            r1 = Float32(residual_out[rbase + Int64(1 * self.hidden)])
            r2 = Float32(residual_out[rbase + Int64(2 * self.hidden)])
            r3 = Float32(residual_out[rbase + Int64(3 * self.hidden)])
            sq_r_acc = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
        a[row, 0 + lane] = r0.to(BFloat16)
        a[row, 32 + lane] = r1.to(BFloat16)
        a[row, 64 + lane] = r2.to(BFloat16)
        a[row, 96 + lane] = r3.to(BFloat16)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        group, sub = lane // 4, lane % 4
        k_half = warp // 3
        n_grp = warp % 3
        mix = n_grp * 8 + group
        total0, total1 = Float32(0.0), Float32(0.0)
        if warp < 6:
            hi0, hi1, hi2, hi3 = Float32(0), Float32(0), Float32(0), Float32(0)
            lo0, lo1, lo2, lo3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for kk in cutlass.range_constexpr(8):
                k0 = (k_half * 8 + kk) * 8 + sub
                k1 = k0 + 4
                a0 = f32_to_raw_bits(Float32(a[group, k0]))
                a2 = f32_to_raw_bits(Float32(a[group, k1]))
                b0 = Float32(b[mix, k0])
                b1 = Float32(b[mix, k1])
                h0, h1 = f32_to_tf32_bits(b0), f32_to_tf32_bits(b1)
                hi0, hi1, hi2, hi3 = tf32_mma_m16n8k8_f32(
                    hi0, hi1, hi2, hi3, a0, Uint32(0), a2, Uint32(0), h0, h1
                )
                lo0, lo1, lo2, lo3 = tf32_mma_m16n8k8_f32(
                    lo0,
                    lo1,
                    lo2,
                    lo3,
                    a0,
                    Uint32(0),
                    a2,
                    Uint32(0),
                    _tf32_residual_bits(b0, h0),
                    _tf32_residual_bits(b1, h1),
                )
                if const_expr(kk % 4 == 3):
                    total0 += hi0 + lo0
                    total1 += hi1 + lo1
                    hi0, hi1, hi2, hi3 = Float32(0), Float32(0), Float32(0), Float32(0)
                    lo0, lo1, lo2, lo3 = Float32(0), Float32(0), Float32(0), Float32(0)

        if warp >= 3 and warp < 6:
            n0 = n_grp * 8 + sub * 2
            half_sum[group, n0] = total0
            half_sum[group, n0 + 1] = total1
        sq_r = _warp_allreduce_sum(sq_r_acc)
        cute.arch.sync_threads()
        if warp < 3 and group < tokens:
            n0 = n_grp * 8 + sub * 2
            pbase = (Int64(group) * Int64(self.splits) + Int64(split)) * Int64(
                _PARTIAL_STRIDE
            )
            partials[pbase + Int64(n0 + 1)] = total0 + half_sum[group, n0]
            partials[pbase + Int64(n0 + 2)] = total1 + half_sum[group, n0 + 1]
        if lane == 0 and row < tokens:
            pbase = (Int64(row) * Int64(self.splits) + Int64(split)) * Int64(
                _PARTIAL_STRIDE
            )
            partials[pbase] = sq_r


class _SplitFinalize:
    """Reduce split-K partials and run Sinkhorn to produce (pre_out, post_out, comb_out)."""

    def __init__(self, hidden: int, rms_eps: float, hc_eps: float, sinkhorn_iters: int):
        self.hidden = int(hidden)
        self.rms_eps = float(rms_eps)
        self.hc_eps = float(hc_eps)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.splits = self.hidden // 32
        self.num_warps = 16 if (self.splits % 16 == 0) else 8
        self.splits_per_warp = self.splits // self.num_warps

    @cute.jit
    def __call__(
        self,
        partials_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        bias_ptr: cute.Pointer,
        pre_ptr: cute.Pointer,
        post_ptr: cute.Pointer,
        comb_ptr: cute.Pointer,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.make_tensor(partials_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            cute.make_tensor(scale_ptr, cute.make_layout((Int64(1) << Int64(16),))),
            cute.make_tensor(bias_ptr, cute.make_layout((Int64(1) << Int64(16),))),
            cute.make_tensor(pre_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            cute.make_tensor(post_ptr, cute.make_layout((Int64(1) << Int64(30),))),
            cute.make_tensor(comb_ptr, cute.make_layout((Int64(1) << Int64(30),))),
        ).launch(
            grid=(tokens, 1, 1),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        pre: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        warp = tid // 32

        smem = utils.SmemAllocator()
        warp_partials = smem.allocate_tensor(
            Float32, cute.make_layout((self.num_warps, 32), stride=(32, 1)), 16
        )

        if lane < 25:
            acc = Float32(0)
            row_base = Int64(row) * Int64(self.splits)
            split_base = warp * self.splits_per_warp
            for s in cutlass.range_constexpr(self.splits_per_warp):
                idx = (row_base + Int64(split_base + s)) * Int64(
                    _PARTIAL_STRIDE
                ) + Int64(lane)
                acc += Float32(partials[idx])
            warp_partials[warp, lane] = acc
        cute.arch.sync_threads()

        if warp == 0:
            value = Float32(0)
            if lane < 25:
                for w in cutlass.range_constexpr(self.num_warps):
                    value += warp_partials[w, lane]
            square = Float32(cute.arch.shuffle_sync(value, Int32(0)))
            inv = cute.math.rsqrt(
                square / Float32(4 * self.hidden) + Float32(self.rms_eps), fastmath=True
            )
            prevalue = Float32(cute.arch.shuffle_sync(value, lane + 1))
            postvalue = Float32(cute.arch.shuffle_sync(value, lane + 5))
            cvalue = Float32(cute.arch.shuffle_sync(value, lane + 9))
            if lane < 4:
                pre[Int64(row) * Int64(4) + Int64(lane)] = Float32(1) / (
                    Float32(1)
                    + cute.math.exp(
                        -(prevalue * inv * Float32(scale[0]) + Float32(bias[lane])),
                        fastmath=True,
                    )
                ) + Float32(self.hc_eps)
                post[Int64(row) * Int64(4) + Int64(lane)] = Float32(2) / (
                    Float32(1)
                    + cute.math.exp(
                        -(
                            postvalue * inv * Float32(scale[1])
                            + Float32(bias[lane + 4])
                        ),
                        fastmath=True,
                    )
                )
            if lane < 16:
                cvalue = cvalue * inv * Float32(scale[2]) + Float32(bias[lane + 8])
            rowmax = _warp_quad_allreduce_max(cvalue)
            cvalue = cute.math.exp(cvalue - rowmax, fastmath=True)
            rowsum = _warp_quad_allreduce_sum(cvalue)
            cvalue = cvalue * cute.arch.rcp_approx(rowsum) + Float32(self.hc_eps)
            colsum = _warp_column4_allreduce_sum(cvalue) + Float32(self.hc_eps)
            cvalue *= cute.arch.rcp_approx(colsum)
            for _ in cutlass.range_constexpr(self.sinkhorn_iters - 1):
                rowsum = _warp_quad_allreduce_sum(cvalue) + Float32(self.hc_eps)
                cvalue *= cute.arch.rcp_approx(rowsum)
                colsum = _warp_column4_allreduce_sum(cvalue) + Float32(self.hc_eps)
                cvalue *= cute.arch.rcp_approx(colsum)
            if lane < 16:
                comb[Int64(row) * Int64(16) + Int64(lane)] = cvalue


@program_cache
def compile_cluster_post_y_norm(
    ordinal: int,
    hidden: int,
    pre_only: bool,
    norm_eps: float,
    norm_dtype: torch.dtype,
    *,
    cluster_size: int = 8,
    emit_mxfp8: bool = False,
):
    hidden = int(hidden)
    pre_only = bool(pre_only)
    norm_eps = float(norm_eps)
    norm_is_bf16 = norm_dtype == torch.bfloat16
    cluster_size = int(cluster_size)
    emit_mxfp8 = bool(emit_mxfp8)
    norm_cute_dt = BFloat16 if norm_is_bf16 else Float32

    kernel = ClusterMhcPostYNorm(
        hidden,
        pre_only,
        norm_eps,
        norm_is_bf16,
        cluster_size=cluster_size,
        emit_mxfp8=emit_mxfp8,
    )
    key = (
        hidden,
        pre_only,
        norm_eps,
        "bf16" if norm_is_bf16 else "fp32",
        cluster_size,
        emit_mxfp8,
    )
    fake = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(norm_cute_dt, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Uint8, 16, cute.AddressSpace.gmem, assumed_align=16),
    )
    with torch.cuda.device(ordinal):
        raw = b12x_compile(
            kernel,
            *fake,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "norm.mhc.cluster_post_y_norm", 1, key
            ),
        )

    def run(
        x,
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out,
        y,
        a_values=None,
        a_scales=None,
        stream=None,
    ):
        tokens = int(residual.shape[0])
        if tokens == 0:
            return
        x_t = residual if pre_only else x
        pp_t = incoming if pre_only else prev_post
        pc_t = incoming if pre_only else prev_comb
        av_t = a_values if (emit_mxfp8 and a_values is not None) else y
        as_t = a_scales if (emit_mxfp8 and a_scales is not None) else y
        stream_int = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
        run_compiled(
            raw,
            (
                make_ptr(
                    BFloat16, x_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16,
                    residual.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Float32, pp_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    Float32, pc_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    Float32,
                    incoming.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    norm_cute_dt,
                    norm_weight.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint32, av_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    Uint8, as_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                Int32(tokens),
                cuda_stream_from_int_or_current(stream_int),
            ),
        )

    return attach_programs(run, raw)


@program_cache
def compile_readonly_fn_sinkhorn(
    ordinal: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
):
    hidden = int(hidden)
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)

    k_partial = ReadOnlyFnPartialM8(hidden)
    k_finalize = _SplitFinalize(hidden, rms_eps, hc_eps, sinkhorn_iters)

    fake_p = (
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
    )
    fake_f = (
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
        make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
    )
    with torch.cuda.device(ordinal):
        raw_partial = b12x_compile(
            k_partial,
            *fake_p,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "norm.mhc.readonly_fn_partial_m8", 1, (hidden,)
            ),
        )
        raw_finalize = b12x_compile(
            k_finalize,
            *fake_f,
            Int32(1),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_key(
                "norm.mhc.split_finalize",
                1,
                (hidden, rms_eps, hc_eps, sinkhorn_iters),
            ),
        )

    def run(
        residual_out,
        fn,
        scale,
        bias,
        partials,
        pre_out,
        post_out,
        comb_out,
        stream=None,
    ):
        tokens = int(residual_out.shape[0])
        if tokens == 0:
            return
        stream_int = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
        cs = cuda_stream_from_int_or_current(stream_int)
        p_ptr = make_ptr(
            Float32, partials.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
        )
        run_compiled(
            raw_partial,
            (
                make_ptr(
                    BFloat16,
                    residual_out.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Float32, fn.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                p_ptr,
                Int32(tokens),
                cs,
            ),
        )
        run_compiled(
            raw_finalize,
            (
                p_ptr,
                make_ptr(
                    Float32, scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    Float32, bias.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    Float32, pre_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                make_ptr(
                    Float32,
                    post_out.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    Float32,
                    comb_out.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                Int32(tokens),
                cs,
            ),
        )

    return attach_programs(run, raw_partial, raw_finalize)


__all__ = [
    "SPLIT_LAGGED_HIDDEN_SIZES",
    "SPLIT_LAGGED_MAX_TOKENS",
    "compile_cluster_post_y_norm",
    "compile_readonly_fn_sinkhorn",
    "split_lagged_partials_shape",
    "supports_split_lagged",
]
