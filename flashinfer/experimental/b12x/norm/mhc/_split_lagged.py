"""SM120 cluster-parallel split-lagged mHC kernels for decode, verify, and draft (M <= 128).

In lagged hyperconnections, the branch input ``y = RMSNorm(contract(pre_mix, residual_out))``
depends only on ``residual_out`` and ``pre_mix`` — not on ``fn @ residual_out`` or the
Sinkhorn projection (``pre_out, post_out, comb_out``), which are not consumed until after
the sublayer's tensor-parallel AllReduce.

Splitting mHC into two stages separates the critical path from the coefficient projection:
1. ``ClusterMhcPostYNorm`` (main stream): SM120 cluster-parallel (or single-CTA for wide
   batches) post-update, lagged contraction, and RMSNorm, with optional bit-exact MXFP8
   quantization matching ``quantize_mxfp8_rows_cute``.
2. ``ReadOnlyFnPartialM8`` (M <= 8) or ``ReadOnlyFnPartial`` (9 <= M <= 128) followed by
   ``_SplitFinalize`` (auxiliary stream): reads the finalized residual stream and computes
   ``fn @ residual_out`` plus Sinkhorn normalization concurrently with the sublayer's
   tensor-parallel AllReduce.
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
from b12x.comm.pcie._cute_intrinsics import (
    add_bf16x2,
    ld_relaxed_gpu_u32,
    ld_relaxed_sys_global_v4_u32,
    pack_f32x2_to_bf16x2,
    plain_graph_epoch_arrive,
    st_relaxed_sys_global_v4_u32,
    sync_threads_after_u32,
    unpack_bf16x2,
)
from b12x.comm.pcie._oneshot_cute import (
    _MAX_RANKS,
    _PLAIN_GRAPH_ARRIVED_OFFSET,
    _PLAIN_GRAPH_EPOCH_OFFSET,
)

from ._fused_tf32 import _pack_bf16_pair
from ._kernels import (
    _contract_four,
    _tf32_residual_bits,
    _warp_allreduce_sum,
    _warp_column4_allreduce_sum,
    _warp_quad_allreduce_max,
    _warp_quad_allreduce_sum,
)

SPLIT_LAGGED_MAX_TOKENS = 128
SPLIT_LAGGED_HIDDEN_SIZES = (4096, 5120, 7168)
CLUSTER_AR_MHC_MAX_TOKENS = 128
CLUSTER_AR_MHC_ONESHOT_MAX_TOKENS = 4
CLUSTER_AR_MHC_WORLD_SIZES = (2, 3, 4, 6, 8)
_CLUSTER_AR_MHC_TRANSPORTS = ("oneshot", "rsag")
_PARTIAL_STRIDE = 32


def cluster_allreduce_transport(tokens: int, world_size: int) -> str:
    """Return the Lamport PCIe transport for ``ClusterMhcAllReducePostYNorm``.

    For TP2 (where 1-hop and 2-hop move identical byte volume) or tiny batches
    (``tokens <= 4``), 1-hop ``"oneshot"`` minimizes round-trip latency. For
    wider batches (``5 <= tokens <= 128``) with ``world_size >= 3``, 2-hop
    Lamport reduce-scatter + all-gather (``"rsag"``) halves per-rank PCIe TX
    traffic on TP4 (1.5x vs 3.0x payload) while keeping the entire AllReduce +
    mHC + RMSNorm + MXFP8 sequence fused in a single kernel launch.
    """
    if int(world_size) <= 2 or int(tokens) <= CLUSTER_AR_MHC_ONESHOT_MAX_TOKENS:
        return "oneshot"
    return "rsag"


def split_lagged_cluster_size(tokens: int) -> int:
    """Return the SM120 CTA-per-row cluster size for ``ClusterMhcPostYNorm``.

    Smaller batches split each hidden row across an 8-CTA or 4-CTA DSM cluster to
    minimize per-row latency; wider batches (M > 48) already saturate SM120 SMs
    with one CTA per row and avoid hardware cluster-slot serialization.
    """
    tokens = int(tokens)
    if tokens <= 16:
        return 8
    if tokens <= 48:
        return 4
    return 1


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
        if self.threads > 1024 or self.num_warps > 32:
            raise ValueError(
                f"ClusterMhcPostYNorm requires threads <= 1024, got {self.threads}"
            )
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
        launch_call = self.kernel(
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
        )
        if const_expr(self.cluster_size > 1):
            launch_call.launch(
                grid=(self.cluster_size, tokens, 1),
                block=(self.threads, 1, 1),
                cluster=(self.cluster_size, 1, 1),
                stream=stream,
            )
        else:
            launch_call.launch(
                grid=(1, tokens, 1),
                block=(self.threads, 1, 1),
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
        warp_sq = smem.allocate_tensor(Float32, cute.make_layout((32,)), 16)
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
        if const_expr(self.cluster_size > 1):
            # Cluster-wide DSM reduction (~25ns)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tot = Float32(0.0)
            for cr in cutlass.range_constexpr(self.cluster_size):
                tot += _ld_dsm_f32(cta_sq_u32, Int32(cr))
        else:
            cute.arch.sync_threads()
            tot = cta_sq[0]
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

        if const_expr(self.cluster_size > 1):
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


class ReadOnlyFnPartial:
    """Read-only ``residual_out @ fn.T`` partial projection for M <= 128 (tile_m=32, 6 TF32 warps).

    Uses a 2D ``(ceil(tokens / 32), hidden // 32, 1)`` grid with 20.6 KiB SMEM per CTA
    (``a[32, 128]`` BF16 + ``b[24, 128]`` FP32, stride 132) and compensated TF32
    ``m16n8k8`` MMAs matching ``_FusedPartial`` bit-for-bit.
    """

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
            grid=((tokens + Int32(31)) // Int32(32), self.splits, 1),
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
        tile, split, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = utils.SmemAllocator()
        a = smem.allocate_tensor(
            BFloat16, cute.make_layout((32, 128), stride=(132, 1)), 16
        )
        b = smem.allocate_tensor(
            Float32, cute.make_layout((24, 128), stride=(132, 1)), 16
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
        cute.arch.cp_async_commit_group()

        m_base = tile * Int32(32)
        h = hbase + lane
        for mi in cutlass.range_constexpr(4):
            localrow = warp + mi * 8
            row = m_base + localrow
            r0, r1, r2, r3 = Float32(0), Float32(0), Float32(0), Float32(0)
            sq_r_acc = Float32(0.0)
            if row < tokens:
                rbase = Int64(row) * Int64(4 * self.hidden) + Int64(h)
                r0 = Float32(residual_out[rbase + Int64(0 * self.hidden)])
                r1 = Float32(residual_out[rbase + Int64(1 * self.hidden)])
                r2 = Float32(residual_out[rbase + Int64(2 * self.hidden)])
                r3 = Float32(residual_out[rbase + Int64(3 * self.hidden)])
                sq_r_acc = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
            a[localrow, 0 + lane] = r0.to(BFloat16)
            a[localrow, 32 + lane] = r1.to(BFloat16)
            a[localrow, 64 + lane] = r2.to(BFloat16)
            a[localrow, 96 + lane] = r3.to(BFloat16)
            sq_r = _warp_allreduce_sum(sq_r_acc)
            if lane == 0 and row < tokens:
                pbase = (Int64(row) * Int64(self.splits) + Int64(split)) * Int64(
                    _PARTIAL_STRIDE
                )
                partials[pbase] = sq_r
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        group, sub = lane // 4, lane % 4
        row0 = (warp // 3) * 16 + group
        row1 = row0 + 8
        mix = (warp % 3) * 8 + group
        n0 = (warp % 3) * 8 + sub * 2
        if warp < 6 and m_base + (warp // 3) * 16 < tokens:
            total0, total1, total2, total3 = (
                Float32(0),
                Float32(0),
                Float32(0),
                Float32(0),
            )
            hi0, hi1, hi2, hi3 = Float32(0), Float32(0), Float32(0), Float32(0)
            lo0, lo1, lo2, lo3 = Float32(0), Float32(0), Float32(0), Float32(0)
            for kk in cutlass.range_constexpr(16):
                k0 = kk * 8 + sub
                k1 = k0 + 4
                a0 = f32_to_raw_bits(Float32(a[row0, k0]))
                a1 = f32_to_raw_bits(Float32(a[row1, k0]))
                a2 = f32_to_raw_bits(Float32(a[row0, k1]))
                a3 = f32_to_raw_bits(Float32(a[row1, k1]))
                b0 = Float32(b[mix, k0])
                b1 = Float32(b[mix, k1])
                h0, h1 = f32_to_tf32_bits(b0), f32_to_tf32_bits(b1)
                hi0, hi1, hi2, hi3 = tf32_mma_m16n8k8_f32(
                    hi0, hi1, hi2, hi3, a0, a1, a2, a3, h0, h1
                )
                lo0, lo1, lo2, lo3 = tf32_mma_m16n8k8_f32(
                    lo0,
                    lo1,
                    lo2,
                    lo3,
                    a0,
                    a1,
                    a2,
                    a3,
                    _tf32_residual_bits(b0, h0),
                    _tf32_residual_bits(b1, h1),
                )
                if const_expr(kk % 4 == 3):
                    total0 += hi0 + lo0
                    total1 += hi1 + lo1
                    total2 += hi2 + lo2
                    total3 += hi3 + lo3
                    hi0, hi1, hi2, hi3 = (
                        Float32(0),
                        Float32(0),
                        Float32(0),
                        Float32(0),
                    )
                    lo0, lo1, lo2, lo3 = (
                        Float32(0),
                        Float32(0),
                        Float32(0),
                        Float32(0),
                    )
            if m_base + row0 < tokens:
                p0 = (Int64(m_base + row0) * Int64(self.splits) + Int64(split)) * Int64(
                    _PARTIAL_STRIDE
                )
                partials[p0 + Int64(n0 + 1)] = total0
                partials[p0 + Int64(n0 + 2)] = total1
            if m_base + row1 < tokens:
                p1 = (Int64(m_base + row1) * Int64(self.splits) + Int64(split)) * Int64(
                    _PARTIAL_STRIDE
                )
                partials[p1 + Int64(n0 + 1)] = total2
                partials[p1 + Int64(n0 + 2)] = total3


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
    cluster_size: int | None = None,
    emit_mxfp8: bool = False,
):
    hidden = int(hidden)
    pre_only = bool(pre_only)
    norm_eps = float(norm_eps)
    norm_is_bf16 = norm_dtype == torch.bfloat16
    emit_mxfp8 = bool(emit_mxfp8)
    norm_cute_dt = BFloat16 if norm_is_bf16 else Float32
    cluster_sizes = (8, 4, 1) if cluster_size is None else (int(cluster_size),)

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
    compiled_by_cluster: dict[int, object] = {}
    with torch.cuda.device(ordinal):
        for cs in cluster_sizes:
            kernel = ClusterMhcPostYNorm(
                hidden,
                pre_only,
                norm_eps,
                norm_is_bf16,
                cluster_size=cs,
                emit_mxfp8=emit_mxfp8,
            )
            key = (
                hidden,
                pre_only,
                norm_eps,
                "bf16" if norm_is_bf16 else "fp32",
                cs,
                emit_mxfp8,
            )
            compiled_by_cluster[cs] = b12x_compile(
                kernel,
                *fake,
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "norm.mhc.cluster_post_y_norm", 2, key
                ),
            )
    fixed_raw = (
        compiled_by_cluster[cluster_sizes[0]] if len(cluster_sizes) == 1 else None
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
        raw = (
            fixed_raw
            if fixed_raw is not None
            else compiled_by_cluster[split_lagged_cluster_size(tokens)]
        )
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

    return attach_programs(run, *compiled_by_cluster.values())


@program_cache
def compile_readonly_fn_sinkhorn(
    ordinal: int,
    hidden: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    *,
    max_tokens: int | None = None,
):
    hidden = int(hidden)
    rms_eps = float(rms_eps)
    hc_eps = float(hc_eps)
    sinkhorn_iters = int(sinkhorn_iters)
    if max_tokens is not None:
        max_tokens = int(max_tokens)
        if not 1 <= max_tokens <= SPLIT_LAGGED_MAX_TOKENS:
            raise ValueError(
                f"max_tokens must be in [1, {SPLIT_LAGGED_MAX_TOKENS}], got {max_tokens}"
            )
        need_m8 = max_tokens <= 8
        need_wide = max_tokens > 8
    else:
        need_m8 = True
        need_wide = True

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
    raw_partial_m8 = None
    raw_partial_wide = None
    programs = []
    with torch.cuda.device(ordinal):
        if need_m8:
            raw_partial_m8 = b12x_compile(
                ReadOnlyFnPartialM8(hidden),
                *fake_p,
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "norm.mhc.readonly_fn_partial_m8", 1, (hidden,)
                ),
            )
            programs.append(raw_partial_m8)
        if need_wide:
            raw_partial_wide = b12x_compile(
                ReadOnlyFnPartial(hidden),
                *fake_p,
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "norm.mhc.readonly_fn_partial", 1, (hidden,)
                ),
            )
            programs.append(raw_partial_wide)
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
        programs.append(raw_finalize)

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
        if tokens > SPLIT_LAGGED_MAX_TOKENS:
            raise ValueError(
                f"split-lagged mHC supports at most {SPLIT_LAGGED_MAX_TOKENS} tokens, got {tokens}"
            )
        if tokens <= 8 and raw_partial_m8 is not None:
            raw_partial = raw_partial_m8
        elif raw_partial_wide is not None:
            raw_partial = raw_partial_wide
        else:
            raise ValueError(
                f"split-lagged mHC runner was compiled for tokens <= 8, got {tokens}"
            )
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

    return attach_programs(run, *programs)


def supports_cluster_allreduce_post_y_norm(
    device: torch.device,
    max_tokens: int,
    hidden_size: int,
    norm_dtype: torch.dtype,
    world_size: int,
) -> bool:
    """Return True when fused Lamport-sentinel AllReduce + ClusterMhcPostYNorm is supported.

    Verifies SM120+ architecture, geometry, world size, and full-grid SM residency
    so peer-waiting Lamport CTAs can never deadlock from partial grid scheduling.
    """
    max_tokens_i = int(max_tokens)
    if not (
        supports_split_lagged(device, max_tokens_i, hidden_size, norm_dtype)
        and max_tokens_i <= CLUSTER_AR_MHC_MAX_TOKENS
        and int(world_size) in CLUSTER_AR_MHC_WORLD_SIZES
    ):
        return False
    sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
    # ClusterMhcAllReducePostYNorm launches 1 CTA/row (cs=1, M>48, 1 CTA/SM),
    # 4 CTAs/row (cs=4, 17<=M<=48, >=4 CTAs/SM), or 8 CTAs/row (cs=8, M<=16, >=8 CTAs/SM).
    # Requiring sm_count >= max(36, max_tokens) guarantees full grid residency.
    return sm_count >= max(36, max_tokens_i)


def oneshot_channel_sentinel_params(
    channel,
    *,
    capturing: bool | None = None,
) -> tuple[int, int, int, int, bool, int]:
    """Extract Lamport-sentinel launch parameters from a live ``PCIeOneshotAllReduce`` channel.

    Returns ``(peer_table_address, signal_table_address, shard_packs,
    payload_ready_region_packs, device_slot_selection, slot_bias)``.
    """
    from b12x.comm.pcie.pcie_oneshot import (
        _enable_device_slot_selection,
        _is_current_stream_capturing,
    )

    state = channel._ext._state(channel._ptr)
    if state.eager_tables is None or state.eager_buffer_bytes is None:
        raise ValueError(
            "fused sentinel ClusterMhcAllReducePostYNorm requires staged PCIe eager buffers"
        )
    if state.plain_payload_ready_region_packs <= 0:
        raise ValueError(
            "PCIe oneshot channel does not have payload-ready Lamport sentinel storage"
        )
    is_capturing = (
        _is_current_stream_capturing(channel.device)
        if capturing is None
        else bool(capturing)
    )
    _enable_device_slot_selection(state, capturing=is_capturing)
    if state.device_slot_selection:
        peer_table_address = int(state.eager_tables[0])
    else:
        slot = state.eager_slot % 2
        state.eager_slot += 1
        peer_table_address = int(state.eager_tables[slot])
    return (
        peer_table_address,
        int(state.signal_table_address),
        int(state.eager_buffer_bytes) // 16,
        int(state.plain_payload_ready_region_packs),
        bool(state.device_slot_selection),
        int(state.slot_bias) & 1 if state.device_slot_selection else 0,
    )


@cute.jit
def _encode_bf16_sentinel_word(value: Uint32) -> Uint32:
    """Encode +0.0 BF16 lanes (0x0000) as -0.0 (0x8000) so 0x0000 stays the empty sentinel."""
    return (
        value
        | (Uint32((value & Uint32(0xFFFF)) == Uint32(0)) << Uint32(15))
        | (Uint32((value & Uint32(0xFFFF0000)) == Uint32(0)) << Uint32(31))
    )


class ClusterMhcAllReducePostYNorm:
    """Fused 1-hop Lamport-sentinel PCIe AllReduce + ``ClusterMhcPostYNorm``.

    Eliminates the standalone AllReduce kernel launch and intermediate ``[M, H]`` BF16
    HBM write/read by fusing:
    1. Rotated 1-hop Lamport-sentinel peer push (``st.relaxed.sys.global.v4.u32``) of
       local ``branch_output`` (optionally ``+ input_addend`` rounded to BF16 via
       ``add.rn.bf16x2``, matching ``OneshotAllReduce``).
    2. Overlapped ``[M, 4, H]`` residual preload while peer TLPs are in flight.
    3. Lamport-sentinel poll (``ld.relaxed.sys.global.v4.u32``), immediate slot reset
       to zero, canonical rank-order (``0 -> 1 -> ... -> world_size - 1``) FP32 sum,
       and exact BF16 rounding seam (``pack_f32x2_to_bf16x2``).
    4. In-register ``ClusterMhcPostYNorm`` residual stream update, lagged contraction,
       DSMEM cluster RMSNorm, and optional bit-exact MXFP8 quantization + SM120 PDL
       ``griddepcontrol.launch_dependents``.
    """

    def __init__(
        self,
        hidden: int,
        world_size: int,
        rank: int,
        norm_eps: float,
        norm_is_bf16: bool,
        cluster_size: int = 8,
        emit_mxfp8: bool = False,
        has_addend: bool = False,
        device_slot_selection: bool = True,
        slot_bias: int = 0,
        release_dependents: bool = False,
        transport: str = "oneshot",
    ):
        self.hidden = int(hidden)
        self.world_size = int(world_size)
        self.rank = int(rank)
        if self.world_size not in CLUSTER_AR_MHC_WORLD_SIZES:
            raise ValueError(
                f"world_size must be in {CLUSTER_AR_MHC_WORLD_SIZES}, got {self.world_size}"
            )
        if not 0 <= self.rank < self.world_size:
            raise ValueError(f"rank ({self.rank}) must be in [0, {self.world_size})")
        if transport not in _CLUSTER_AR_MHC_TRANSPORTS:
            raise ValueError(
                f"transport must be in {_CLUSTER_AR_MHC_TRANSPORTS}, got {transport!r}"
            )
        self.transport = str(transport)
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
        if self.threads > 1024 or self.num_warps > 32:
            raise ValueError(
                f"ClusterMhcAllReducePostYNorm requires threads <= 1024, got {self.threads}"
            )
        self.emit_mxfp8 = bool(emit_mxfp8)
        self.has_addend = bool(has_addend)
        self.device_slot_selection = bool(device_slot_selection)
        self.slot_bias = int(slot_bias) & 1 if self.device_slot_selection else 0
        self.release_dependents = bool(release_dependents)

    @cute.jit
    def __call__(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        x_ptr: cute.Pointer,
        addend_ptr: cute.Pointer,
        res_ptr: cute.Pointer,
        prev_post_ptr: cute.Pointer,
        prev_comb_ptr: cute.Pointer,
        incoming_ptr: cute.Pointer,
        norm_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        a_values_u32_ptr: cute.Pointer,
        a_scales_ptr: cute.Pointer,
        shard_packs: Int64,
        payload_ready_region_packs: Int64,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        launch_call = self.kernel(
            peer_ptrs,
            signal_ptrs,
            x_ptr,
            addend_ptr,
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
            shard_packs,
            payload_ready_region_packs,
        )
        if const_expr(self.cluster_size > 1):
            launch_call.launch(
                grid=(self.cluster_size, tokens, 1),
                block=(self.threads, 1, 1),
                cluster=(self.cluster_size, 1, 1),
                stream=stream,
            )
        else:
            launch_call.launch(
                grid=(1, tokens, 1),
                block=(self.threads, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def kernel(
        self,
        peer_ptrs: cute.Pointer,
        signal_ptrs: cute.Pointer,
        x_ptr: cute.Pointer,
        addend_ptr: cute.Pointer,
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
        shard_packs: Int64,
        payload_ready_region_packs: Int64,
    ):
        c_rank, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        gdim_x, gdim_y, _ = cute.arch.grid_dim()
        lane, warp = tid % 32, tid // 32

        smem = utils.SmemAllocator()
        warp_sq = smem.allocate_tensor(Float32, cute.make_layout((32,)), 16)
        cta_sq = smem.allocate_tensor(Float32, cute.make_layout((1,)), 16)
        cta_sq_u32 = shared_ptr_to_u32(cta_sq.iterator)

        slot_table = peer_ptrs
        if const_expr(self.device_slot_selection):
            self_signal = Int64(signal_ptrs[self.rank])
            generation = ld_relaxed_gpu_u32(
                self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET)
            )
            # Every thread's epoch load must complete before the CTA barrier so
            # the final CTA cannot advance the epoch ahead of a pending load.
            generation = sync_threads_after_u32(generation)
            slot = (generation + Uint32(self.slot_bias)) % Uint32(2)
            slot_table = peer_ptrs + Int64(slot) * Int64(_MAX_RANKS)
            if tid == Int32(0):
                plain_graph_epoch_arrive(
                    self_signal + Int64(_PLAIN_GRAPH_EPOCH_OFFSET),
                    self_signal + Int64(_PLAIN_GRAPH_ARRIVED_OFFSET),
                    Uint32(gdim_x) * Uint32(gdim_y),
                )

        row_bytes = Int64(self.hidden * 2)
        res_base = Int64(res_ptr.toint()) + Int64(row) * Int64(4) * row_bytes
        out_base = Int64(out_ptr.toint()) + Int64(row) * Int64(4) * row_bytes
        x_base = Int64(x_ptr.toint()) + Int64(row) * row_bytes
        addend_base = Int64(addend_ptr.toint()) + Int64(row) * row_bytes
        y_base = Int64(y_ptr.toint()) + Int64(row) * row_bytes
        norm_base = Int64(norm_ptr.toint())
        region_bytes = payload_ready_region_packs * Int64(16)

        pack = c_rank * self.packs_per_cta + tid
        active = tid < self.packs_per_cta
        pack_idx = Int64(row) * Int64(self.packs) + Int64(pack)
        boff = Int64(pack) * Int64(16)

        xw = cute.make_rmem_tensor((4,), Uint32)
        rw_buf = cute.make_rmem_tensor((4, 4), Uint32)

        if active:
            # Step A: load local branch output (+ optional addend), sentinel-encode, and
            # push to all remote peers in rotated order to avoid RX incast.
            local_words = ld_global_v4_u32(x_base + boff)
            if const_expr(self.has_addend):
                extra_words = ld_global_v4_u32(addend_base + boff)
                self_enc0 = _encode_bf16_sentinel_word(
                    add_bf16x2(local_words[0], extra_words[0])
                )
                self_enc1 = _encode_bf16_sentinel_word(
                    add_bf16x2(local_words[1], extra_words[1])
                )
                self_enc2 = _encode_bf16_sentinel_word(
                    add_bf16x2(local_words[2], extra_words[2])
                )
                self_enc3 = _encode_bf16_sentinel_word(
                    add_bf16x2(local_words[3], extra_words[3])
                )
            else:
                self_enc0 = _encode_bf16_sentinel_word(local_words[0])
                self_enc1 = _encode_bf16_sentinel_word(local_words[1])
                self_enc2 = _encode_bf16_sentinel_word(local_words[2])
                self_enc3 = _encode_bf16_sentinel_word(local_words[3])

            self_shard_off = (Int64(self.rank) * shard_packs + pack_idx) * Int64(16)
            if const_expr(self.transport == "oneshot"):
                for peer_index in cutlass.range_constexpr(1, self.world_size):
                    target = (self.rank + peer_index) % self.world_size
                    dst_addr = Int64(slot_table[target]) + region_bytes + self_shard_off
                    st_relaxed_sys_global_v4_u32(
                        dst_addr, self_enc0, self_enc1, self_enc2, self_enc3
                    )

                # Overlap [M, 4, H] residual stream preload while peer PCIe TLPs are in flight.
                rw0 = ld_global_v4_u32(res_base + Int64(0) * row_bytes + boff)
                rw1 = ld_global_v4_u32(res_base + Int64(1) * row_bytes + boff)
                rw2 = ld_global_v4_u32(res_base + Int64(2) * row_bytes + boff)
                rw3 = ld_global_v4_u32(res_base + Int64(3) * row_bytes + boff)
                for wd in cutlass.range_constexpr(4):
                    rw_buf[0, wd] = rw0[wd]
                    rw_buf[1, wd] = rw1[wd]
                    rw_buf[2, wd] = rw2[wd]
                    rw_buf[3, wd] = rw3[wd]

                # Step B: spin on local Lamport slots until every remote peer arrives,
                # clear consumed slots to zero, and sum in canonical rank order 0..W-1.
                local_base = Int64(slot_table[self.rank]) + region_bytes
                incoming_words = cute.make_rmem_tensor((self.world_size, 4), Uint32)
                incoming_words[self.rank, 0] = self_enc0
                incoming_words[self.rank, 1] = self_enc1
                incoming_words[self.rank, 2] = self_enc2
                incoming_words[self.rank, 3] = self_enc3

                waiting = cutlass.Boolean(True)
                while waiting:
                    waiting = cutlass.Boolean(False)
                    for peer_index in cutlass.range_constexpr(1, self.world_size):
                        peer = (self.rank + peer_index) % self.world_size
                        addr = local_base + (
                            Int64(peer) * shard_packs + pack_idx
                        ) * Int64(16)
                        words = ld_relaxed_sys_global_v4_u32(addr)
                        for wd in cutlass.range_constexpr(4):
                            incoming_words[peer, wd] = words[wd]
                            waiting = (
                                waiting
                                | ((words[wd] & Uint32(0xFFFF)) == Uint32(0))
                                | ((words[wd] & Uint32(0xFFFF0000)) == Uint32(0))
                            )

                for peer_index in cutlass.range_constexpr(1, self.world_size):
                    peer = (self.rank + peer_index) % self.world_size
                    addr = local_base + (Int64(peer) * shard_packs + pack_idx) * Int64(
                        16
                    )
                    st_relaxed_sys_global_v4_u32(
                        addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    )

                for wd in cutlass.range_constexpr(4):
                    acc_lo = Float32(0.0)
                    acc_hi = Float32(0.0)
                    for peer in cutlass.range_constexpr(self.world_size):
                        lo, hi = unpack_bf16x2(incoming_words[peer, wd])
                        if const_expr(peer == 0):
                            acc_lo = lo
                            acc_hi = hi
                        else:
                            acc_lo = acc_lo + lo
                            acc_hi = acc_hi + hi
                    xw[wd] = pack_f32x2_to_bf16x2(acc_lo, acc_hi)
            else:
                # 2-hop Lamport reduce-scatter + all-gather with warp-uniform rotated owners.
                # Every 32-thread warp owns one uniform peer rank, ensuring coalesced 512B
                # PCIe TLP bursts and zero warp divergence between owner and non-owner warps.
                owner = (
                    Int32(row) + Int32(c_rank) * Int32(self.num_warps) + Int32(warp)
                ) % Int32(self.world_size)
                local_base = Int64(slot_table[self.rank]) + region_bytes

                if owner != Int32(self.rank):
                    dst_addr = Int64(slot_table[owner]) + region_bytes + self_shard_off
                    st_relaxed_sys_global_v4_u32(
                        dst_addr, self_enc0, self_enc1, self_enc2, self_enc3
                    )

                # Overlap [M, 4, H] residual stream preload while Hop-1 PCIe TLPs are in flight.
                rw0 = ld_global_v4_u32(res_base + Int64(0) * row_bytes + boff)
                rw1 = ld_global_v4_u32(res_base + Int64(1) * row_bytes + boff)
                rw2 = ld_global_v4_u32(res_base + Int64(2) * row_bytes + boff)
                rw3 = ld_global_v4_u32(res_base + Int64(3) * row_bytes + boff)
                for wd in cutlass.range_constexpr(4):
                    rw_buf[0, wd] = rw0[wd]
                    rw_buf[1, wd] = rw1[wd]
                    rw_buf[2, wd] = rw2[wd]
                    rw_buf[3, wd] = rw3[wd]

                if owner == Int32(self.rank):
                    incoming_words = cute.make_rmem_tensor((self.world_size, 4), Uint32)
                    incoming_words[self.rank, 0] = self_enc0
                    incoming_words[self.rank, 1] = self_enc1
                    incoming_words[self.rank, 2] = self_enc2
                    incoming_words[self.rank, 3] = self_enc3

                    waiting = cutlass.Boolean(True)
                    while waiting:
                        waiting = cutlass.Boolean(False)
                        for peer_index in cutlass.range_constexpr(1, self.world_size):
                            peer = (self.rank + peer_index) % self.world_size
                            addr = local_base + (
                                Int64(peer) * shard_packs + pack_idx
                            ) * Int64(16)
                            words = ld_relaxed_sys_global_v4_u32(addr)
                            for wd in cutlass.range_constexpr(4):
                                incoming_words[peer, wd] = words[wd]
                                waiting = (
                                    waiting
                                    | ((words[wd] & Uint32(0xFFFF)) == Uint32(0))
                                    | ((words[wd] & Uint32(0xFFFF0000)) == Uint32(0))
                                )

                    for peer_index in cutlass.range_constexpr(1, self.world_size):
                        peer = (self.rank + peer_index) % self.world_size
                        addr = local_base + (
                            Int64(peer) * shard_packs + pack_idx
                        ) * Int64(16)
                        st_relaxed_sys_global_v4_u32(
                            addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                        )

                    for wd in cutlass.range_constexpr(4):
                        acc_lo = Float32(0.0)
                        acc_hi = Float32(0.0)
                        for peer in cutlass.range_constexpr(self.world_size):
                            lo, hi = unpack_bf16x2(incoming_words[peer, wd])
                            if const_expr(peer == 0):
                                acc_lo = lo
                                acc_hi = hi
                            else:
                                acc_lo = acc_lo + lo
                                acc_hi = acc_hi + hi
                        xw[wd] = _encode_bf16_sentinel_word(
                            pack_f32x2_to_bf16x2(acc_lo, acc_hi)
                        )

                    # Hop 2 (All-Gather): broadcast reduced sentinel words to every peer.
                    for peer_index in cutlass.range_constexpr(1, self.world_size):
                        target = (self.rank + peer_index) % self.world_size
                        dst_addr = (
                            Int64(slot_table[target]) + region_bytes + self_shard_off
                        )
                        st_relaxed_sys_global_v4_u32(
                            dst_addr, xw[0], xw[1], xw[2], xw[3]
                        )
                else:
                    # Hop 2 (All-Gather): wait on owner's reduced pack and clear slot.
                    addr = local_base + (Int64(owner) * shard_packs + pack_idx) * Int64(
                        16
                    )
                    waiting = cutlass.Boolean(True)
                    while waiting:
                        words = ld_relaxed_sys_global_v4_u32(addr)
                        waiting = cutlass.Boolean(False)
                        for wd in cutlass.range_constexpr(4):
                            xw[wd] = words[wd]
                            waiting = (
                                waiting
                                | ((words[wd] & Uint32(0xFFFF)) == Uint32(0))
                                | ((words[wd] & Uint32(0xFFFF0000)) == Uint32(0))
                            )
                    st_relaxed_sys_global_v4_u32(
                        addr, Uint32(0), Uint32(0), Uint32(0), Uint32(0)
                    )

        # Step C: in-register ClusterMhcPostYNorm (load coefficients after PCIe spin to
        # keep register pressure low and preserve Sinkhorn overlap).
        row4 = Int64(row) * Int64(4)
        in0 = Float32(incoming[row4 + Int64(0)])
        in1 = Float32(incoming[row4 + Int64(1)])
        in2 = Float32(incoming[row4 + Int64(2)])
        in3 = Float32(incoming[row4 + Int64(3)])
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

        y_reg = cute.make_rmem_tensor((4,), Uint32)
        sq_y = Float32(0.0)
        if active:
            o0 = cute.make_rmem_tensor((4,), Uint32)
            o1 = cute.make_rmem_tensor((4,), Uint32)
            o2 = cute.make_rmem_tensor((4,), Uint32)
            o3 = cute.make_rmem_tensor((4,), Uint32)
            for wd in cutlass.range_constexpr(4):
                r0_lo = u32_as_f32(rw_buf[0, wd] << Uint32(16))
                r0_hi = u32_as_f32(rw_buf[0, wd] & Uint32(0xFFFF0000))
                r1_lo = u32_as_f32(rw_buf[1, wd] << Uint32(16))
                r1_hi = u32_as_f32(rw_buf[1, wd] & Uint32(0xFFFF0000))
                r2_lo = u32_as_f32(rw_buf[2, wd] << Uint32(16))
                r2_hi = u32_as_f32(rw_buf[2, wd] & Uint32(0xFFFF0000))
                r3_lo = u32_as_f32(rw_buf[3, wd] << Uint32(16))
                r3_hi = u32_as_f32(rw_buf[3, wd] & Uint32(0xFFFF0000))
                xv_lo = u32_as_f32(xw[wd] << Uint32(16))
                xv_hi = u32_as_f32(xw[wd] & Uint32(0xFFFF0000))
                b0_lo = (
                    p0 * xv_lo + c00 * r0_lo + c01 * r1_lo + c02 * r2_lo + c03 * r3_lo
                ).to(BFloat16)
                b0_hi = (
                    p0 * xv_hi + c00 * r0_hi + c01 * r1_hi + c02 * r2_hi + c03 * r3_hi
                ).to(BFloat16)
                b1_lo = (
                    p1 * xv_lo + c10 * r0_lo + c11 * r1_lo + c12 * r2_lo + c13 * r3_lo
                ).to(BFloat16)
                b1_hi = (
                    p1 * xv_hi + c10 * r0_hi + c11 * r1_hi + c12 * r2_hi + c13 * r3_hi
                ).to(BFloat16)
                b2_lo = (
                    p2 * xv_lo + c20 * r0_lo + c21 * r1_lo + c22 * r2_lo + c23 * r3_lo
                ).to(BFloat16)
                b2_hi = (
                    p2 * xv_hi + c20 * r0_hi + c21 * r1_hi + c22 * r2_hi + c23 * r3_hi
                ).to(BFloat16)
                b3_lo = (
                    p3 * xv_lo + c30 * r0_lo + c31 * r1_lo + c32 * r2_lo + c33 * r3_lo
                ).to(BFloat16)
                b3_hi = (
                    p3 * xv_hi + c30 * r0_hi + c31 * r1_hi + c32 * r2_hi + c33 * r3_hi
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
        if const_expr(self.cluster_size > 1):
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            tot = Float32(0.0)
            for cr in cutlass.range_constexpr(self.cluster_size):
                tot += _ld_dsm_f32(cta_sq_u32, Int32(cr))
        else:
            cute.arch.sync_threads()
            tot = cta_sq[0]
        inv = cute.math.rsqrt(
            tot / Float32(self.hidden) + Float32(self.norm_eps), fastmath=True
        )

        vals_f = cute.make_rmem_tensor((8,), Float32)
        max_abs = Float32(0.0)
        if active:
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

        if const_expr(self.release_dependents):
            cute.arch.griddepcontrol_launch_dependents()

        if const_expr(self.cluster_size > 1):
            # Hold shared memory until every CTA in the cluster has read cta_sq.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()


@program_cache
def compile_cluster_allreduce_post_y_norm(
    ordinal: int,
    hidden: int,
    world_size: int,
    rank: int,
    norm_eps: float,
    norm_dtype: torch.dtype,
    *,
    cluster_size: int | None = None,
    emit_mxfp8: bool = False,
    has_addend: bool = False,
    device_slot_selection: bool = True,
    slot_bias: int = 0,
    release_dependents: bool = False,
    transport: str = "auto",
):
    hidden = int(hidden)
    world_size = int(world_size)
    rank = int(rank)
    norm_eps = float(norm_eps)
    norm_is_bf16 = norm_dtype == torch.bfloat16
    emit_mxfp8 = bool(emit_mxfp8)
    has_addend = bool(has_addend)
    device_slot_selection = bool(device_slot_selection)
    slot_bias = int(slot_bias) & 1 if device_slot_selection else 0
    release_dependents = bool(release_dependents)
    if transport not in ("auto", *_CLUSTER_AR_MHC_TRANSPORTS):
        raise ValueError(
            f"transport must be 'auto' or in {_CLUSTER_AR_MHC_TRANSPORTS}, got {transport!r}"
        )
    norm_cute_dt = BFloat16 if norm_is_bf16 else Float32
    cluster_sizes = (8, 4, 1) if cluster_size is None else (int(cluster_size),)
    specs: list[tuple[int, str]] = []
    for cs in cluster_sizes:
        if transport != "auto":
            specs.append((cs, transport))
        elif world_size <= 2:
            specs.append((cs, "oneshot"))
        elif cs == 8:
            specs.append((8, "oneshot"))
            specs.append((8, "rsag"))
        else:
            specs.append((cs, "rsag"))

    fake = (
        make_ptr(cutlass.Uint64, 16, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(cutlass.Uint64, 16, cute.AddressSpace.gmem, assumed_align=8),
        make_ptr(BFloat16, 16, cute.AddressSpace.gmem, assumed_align=16),
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
    compiled_by_spec: dict[tuple[int, str], object] = {}
    with torch.cuda.device(ordinal):
        for cs, resolved_t in specs:
            kernel = ClusterMhcAllReducePostYNorm(
                hidden,
                world_size,
                rank,
                norm_eps,
                norm_is_bf16,
                cluster_size=cs,
                emit_mxfp8=emit_mxfp8,
                has_addend=has_addend,
                device_slot_selection=device_slot_selection,
                slot_bias=slot_bias,
                release_dependents=release_dependents,
                transport=resolved_t,
            )
            key = (
                hidden,
                world_size,
                rank,
                norm_eps,
                "bf16" if norm_is_bf16 else "fp32",
                cs,
                emit_mxfp8,
                has_addend,
                device_slot_selection,
                slot_bias,
                release_dependents,
                resolved_t,
            )
            compiled_by_spec[(cs, resolved_t)] = b12x_compile(
                kernel,
                *fake,
                Int64(1),
                Int64(0),
                Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key(
                    "norm.mhc.cluster_allreduce_post_y_norm", 3, key
                ),
            )
    fixed_raw = compiled_by_spec[specs[0]] if len(specs) == 1 else None

    def run(
        peer_table_address: int,
        signal_table_address: int,
        x,
        residual,
        prev_post,
        prev_comb,
        incoming,
        norm_weight,
        out,
        y,
        *,
        shard_packs: int,
        payload_ready_region_packs: int = 0,
        input_addend=None,
        a_values=None,
        a_scales=None,
        stream=None,
    ):
        tokens = int(residual.shape[0])
        if tokens == 0:
            return
        if tokens > CLUSTER_AR_MHC_MAX_TOKENS:
            raise ValueError(
                f"ClusterMhcAllReducePostYNorm supports at most {CLUSTER_AR_MHC_MAX_TOKENS} "
                f"tokens, got {tokens}"
            )
        if has_addend:
            if input_addend is None:
                raise ValueError(
                    "ClusterMhcAllReducePostYNorm compiled with has_addend=True requires input_addend"
                )
        elif input_addend is not None:
            raise ValueError(
                "ClusterMhcAllReducePostYNorm compiled with has_addend=False does not accept input_addend"
            )
        if int(shard_packs) < tokens * (hidden // 8):
            raise ValueError(
                f"shard_packs ({shard_packs}) must cover payload packs ({tokens * (hidden // 8)})"
            )
        if fixed_raw is not None:
            raw = fixed_raw
        else:
            cs = (
                int(cluster_size)
                if cluster_size is not None
                else split_lagged_cluster_size(tokens)
            )
            resolved_t = (
                cluster_allreduce_transport(tokens, world_size)
                if transport == "auto"
                else transport
            )
            raw = compiled_by_spec[(cs, resolved_t)]
        add_t = input_addend if has_addend else x
        av_t = a_values if (emit_mxfp8 and a_values is not None) else y
        as_t = a_scales if (emit_mxfp8 and a_scales is not None) else y
        sig_addr = (
            int(signal_table_address)
            if device_slot_selection
            else int(peer_table_address)
        )
        stream_int = stream.cuda_stream if hasattr(stream, "cuda_stream") else stream
        run_compiled(
            raw,
            (
                make_ptr(
                    cutlass.Uint64,
                    int(peer_table_address),
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                ),
                make_ptr(
                    cutlass.Uint64,
                    sig_addr,
                    cute.AddressSpace.gmem,
                    assumed_align=8,
                ),
                make_ptr(
                    BFloat16, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16, add_t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
                ),
                make_ptr(
                    BFloat16,
                    residual.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    Float32,
                    prev_post.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    Float32,
                    prev_comb.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
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
                Int64(int(shard_packs)),
                Int64(int(payload_ready_region_packs)),
                Int32(tokens),
                cuda_stream_from_int_or_current(stream_int),
            ),
        )

    return attach_programs(run, *compiled_by_spec.values())


__all__ = [
    "CLUSTER_AR_MHC_MAX_TOKENS",
    "CLUSTER_AR_MHC_ONESHOT_MAX_TOKENS",
    "CLUSTER_AR_MHC_WORLD_SIZES",
    "ClusterMhcAllReducePostYNorm",
    "SPLIT_LAGGED_HIDDEN_SIZES",
    "SPLIT_LAGGED_MAX_TOKENS",
    "cluster_allreduce_transport",
    "compile_cluster_allreduce_post_y_norm",
    "compile_cluster_post_y_norm",
    "compile_readonly_fn_sinkhorn",
    "oneshot_channel_sentinel_params",
    "split_lagged_cluster_size",
    "split_lagged_partials_shape",
    "supports_cluster_allreduce_post_y_norm",
    "supports_split_lagged",
]
