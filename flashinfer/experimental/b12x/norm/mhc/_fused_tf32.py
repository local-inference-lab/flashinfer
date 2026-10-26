"""Row-tile mHC projection with full-range BF16 operands.

The row-tile, split-K scheduling follows DeepSeek TileKernels' mHC projection;
SM120 uses compensated mma.sync TF32 with the lagged four-stream contract.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.compile_plan import compile_only_launches_enabled
from b12x._lib.compiler import DimKey, KernelCompileSpec, run_compiled, tensor_key
from b12x._lib.intrinsics.memory import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    shared_ptr_to_u32,
    u32_as_f32,
)
from b12x._lib.intrinsics.mma import tf32_mma_m16n8k8_f32
from b12x._lib.intrinsics.packed import f32_to_raw_bits, f32_to_tf32_bits
from b12x._lib.utils.device import current_cuda_stream
from ._kernels import (
    _MHC_PDL,
    _compile_mhc_entry,
    _contract_four,
    _tf32_residual_bits,
    _warp_allreduce_sum,
    _warp_quad_allreduce_max,
    _warp_quad_allreduce_sum,
    _warp_column4_allreduce_sum,
)


@dsl_user_op
def _pack_bf16_pair(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _FusedPartial:
    def __init__(self, hidden, pre_only, tile_m, fuse_norm):
        self.hidden = hidden
        self.pre_only = pre_only
        self.tile_m = tile_m
        self.fuse_norm = fuse_norm
        self.splits = hidden // 32

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        incoming: cute.Tensor,
        out: cute.Tensor,
        y: cute.Tensor,
        partials: cute.Tensor,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            x, residual, post, comb, fn, incoming, out, y, partials, tokens
        ).launch(
            grid=((tokens + self.tile_m - 1) // self.tile_m, self.splits, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=_MHC_PDL,
        )

    @cute.jit
    def load(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        incoming: cute.Tensor,
        out: cute.Tensor,
        y: cute.Tensor,
        a: cute.Tensor,
        b: cute.Tensor,
        stats: cute.Tensor,
        tile: Int32,
        split: Int32,
        tokens: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        hbase = split * 32
        bbase = shared_ptr_to_u32(b.iterator)
        incoming_values = cute.make_rmem_tensor(
            cute.make_layout((self.tile_m // 8, 4), stride=(4, 1)), Float32
        )
        incoming_values.fill(0.0)
        for mi in cutlass.range_constexpr(self.tile_m // 8):
            row = tile * self.tile_m + warp + mi * 8
            if row < tokens:
                for stream in cutlass.range_constexpr(4):
                    incoming_values[mi, stream] = Float32(incoming[row, stream])
        for i in cutlass.range_constexpr(3):
            linear = i * 1024 + tid * 4
            n, k = linear // 128, linear % 128
            src = (
                Int64(n) * Int64(4 * self.hidden)
                + Int64(k // 32) * Int64(self.hidden)
                + Int64(hbase + k % 32)
            )
            dst = bbase + (n * 132 + k) * 4
            address = get_ptr_as_int64(fn, src)
            if address & Int64(15) == Int64(0):
                cp_async4_shared_global(dst, address)
            else:
                for part in cutlass.range_constexpr(4):
                    cp_async_u32_shared_global(
                        dst + part * 4, address + Int64(part * 4)
                    )
        for mi in cutlass.range_constexpr(self.tile_m // 8):
            localrow = warp + mi * 8
            row = tile * self.tile_m + localrow
            for half in cutlass.range_constexpr(1):
                h = hbase + half * 32 + lane
                r0, r1, r2, r3 = Float32(0), Float32(0), Float32(0), Float32(0)
                if row < tokens:
                    r0 = Float32(residual[row, 0, h])
                    r1 = Float32(residual[row, 1, h])
                    r2 = Float32(residual[row, 2, h])
                    r3 = Float32(residual[row, 3, h])
                    if const_expr(not self.pre_only):
                        cv, pv = Float32(0), Float32(0)
                        if lane < 16:
                            cv = Float32(comb[row, lane // 4, lane % 4])
                        if lane < 4:
                            pv = Float32(post[row, lane])
                        xv = Float32(x[row, h])
                        b0 = (
                            Float32(cute.arch.shuffle_sync(pv, Int32(0))) * xv
                            + Float32(cute.arch.shuffle_sync(cv, Int32(0))) * r0
                            + Float32(cute.arch.shuffle_sync(cv, Int32(4))) * r1
                            + Float32(cute.arch.shuffle_sync(cv, Int32(8))) * r2
                            + Float32(cute.arch.shuffle_sync(cv, Int32(12))) * r3
                        ).to(cutlass.BFloat16)
                        b1 = (
                            Float32(cute.arch.shuffle_sync(pv, Int32(1))) * xv
                            + Float32(cute.arch.shuffle_sync(cv, Int32(1))) * r0
                            + Float32(cute.arch.shuffle_sync(cv, Int32(5))) * r1
                            + Float32(cute.arch.shuffle_sync(cv, Int32(9))) * r2
                            + Float32(cute.arch.shuffle_sync(cv, Int32(13))) * r3
                        ).to(cutlass.BFloat16)
                        b2 = (
                            Float32(cute.arch.shuffle_sync(pv, Int32(2))) * xv
                            + Float32(cute.arch.shuffle_sync(cv, Int32(2))) * r0
                            + Float32(cute.arch.shuffle_sync(cv, Int32(6))) * r1
                            + Float32(cute.arch.shuffle_sync(cv, Int32(10))) * r2
                            + Float32(cute.arch.shuffle_sync(cv, Int32(14))) * r3
                        ).to(cutlass.BFloat16)
                        b3 = (
                            Float32(cute.arch.shuffle_sync(pv, Int32(3))) * xv
                            + Float32(cute.arch.shuffle_sync(cv, Int32(3))) * r0
                            + Float32(cute.arch.shuffle_sync(cv, Int32(7))) * r1
                            + Float32(cute.arch.shuffle_sync(cv, Int32(11))) * r2
                            + Float32(cute.arch.shuffle_sync(cv, Int32(15))) * r3
                        ).to(cutlass.BFloat16)
                        r0, r1, r2, r3 = (
                            Float32(b0),
                            Float32(b1),
                            Float32(b2),
                            Float32(b3),
                        )
                    out[row, 0, h] = r0.to(cutlass.BFloat16)
                    out[row, 1, h] = r1.to(cutlass.BFloat16)
                    out[row, 2, h] = r2.to(cutlass.BFloat16)
                    out[row, 3, h] = r3.to(cutlass.BFloat16)
                a[localrow, 0 + half * 32 + lane] = r0.to(cutlass.BFloat16)
                a[localrow, 32 + half * 32 + lane] = r1.to(cutlass.BFloat16)
                a[localrow, 64 + half * 32 + lane] = r2.to(cutlass.BFloat16)
                a[localrow, 96 + half * 32 + lane] = r3.to(cutlass.BFloat16)
                yv = Float32(0)
                if row < tokens:
                    yb = _contract_four(
                        incoming_values[mi, 0],
                        incoming_values[mi, 1],
                        incoming_values[mi, 2],
                        incoming_values[mi, 3],
                        r0,
                        r1,
                        r2,
                        r3,
                    ).to(cutlass.BFloat16)
                    y[row, h] = yb
                    yv = Float32(yb)
                stats[mi, 0] += _warp_allreduce_sum(
                    r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3
                )
                if const_expr(self.fuse_norm):
                    stats[mi, 1] += _warp_allreduce_sum(yv * yv)
        cute.arch.cp_async_commit_group()

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        residual: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        fn: cute.Tensor,
        incoming: cute.Tensor,
        out: cute.Tensor,
        y: cute.Tensor,
        partials: cute.Tensor,
        tokens: Int32,
    ):
        if const_expr(_MHC_PDL):
            cute.arch.griddepcontrol_wait()
        tile, split, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = tid % 32, tid // 32
        smem = utils.SmemAllocator()
        a = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((self.tile_m, 128), stride=(132, 1)),
            16,
        )
        b = smem.allocate_tensor(
            Float32, cute.make_layout((24, 128), stride=(132, 1)), 16
        )
        stats = cute.make_rmem_tensor(
            cute.make_layout((self.tile_m // 8, 2), stride=(2, 1)), Float32
        )
        stats.fill(0.0)
        total = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        hi = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        lo = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        total.fill(0.0)
        hi.fill(0.0)
        lo.fill(0.0)
        group, sub = lane // 4, lane % 4
        row0, row1 = (warp // 3) * 16 + group, (warp // 3) * 16 + group + 8
        mix = (warp % 3) * 8 + group
        self.load(
            x,
            residual,
            post,
            comb,
            fn,
            incoming,
            out,
            y,
            a,
            b,
            stats,
            tile,
            split,
            tokens,
        )
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if warp < self.tile_m // 16 * 3:
            for kk in cutlass.range_constexpr(16):
                k0, k1 = kk * 8 + sub, kk * 8 + sub + 4
                a0, a1 = (
                    f32_to_raw_bits(Float32(a[row0, k0])),
                    f32_to_raw_bits(Float32(a[row1, k0])),
                )
                a2, a3 = (
                    f32_to_raw_bits(Float32(a[row0, k1])),
                    f32_to_raw_bits(Float32(a[row1, k1])),
                )
                b0, b1 = (
                    Float32(b[mix, k0]),
                    Float32(b[mix, k1]),
                )
                h0, h1 = f32_to_tf32_bits(b0), f32_to_tf32_bits(b1)
                d0, d1, d2, d3 = tf32_mma_m16n8k8_f32(
                    hi[0], hi[1], hi[2], hi[3], a0, a1, a2, a3, h0, h1
                )
                l0, l1, l2, l3 = tf32_mma_m16n8k8_f32(
                    lo[0],
                    lo[1],
                    lo[2],
                    lo[3],
                    a0,
                    a1,
                    a2,
                    a3,
                    _tf32_residual_bits(b0, h0),
                    _tf32_residual_bits(b1, h1),
                )
                hi[0], hi[1], hi[2], hi[3] = d0, d1, d2, d3
                lo[0], lo[1], lo[2], lo[3] = l0, l1, l2, l3
                if const_expr(kk % 4 == 3):
                    for j in cutlass.range_constexpr(4):
                        total[j] += hi[j] + lo[j]
                        hi[j], lo[j] = Float32(0), Float32(0)
        cute.arch.sync_threads()
        if warp < self.tile_m // 16 * 3:
            n0 = (warp % 3) * 8 + sub * 2
            if tile * self.tile_m + row0 < tokens:
                partials[tile * self.tile_m + row0, split, n0 + 1] = total[0]
                partials[tile * self.tile_m + row0, split, n0 + 2] = total[1]
            if tile * self.tile_m + row1 < tokens:
                partials[tile * self.tile_m + row1, split, n0 + 1] = total[2]
                partials[tile * self.tile_m + row1, split, n0 + 2] = total[3]
        if lane == 0:
            for mi in cutlass.range_constexpr(self.tile_m // 8):
                row = tile * self.tile_m + warp + mi * 8
                if row < tokens:
                    partials[row, split, 0] = stats[mi, 0]
                    if const_expr(self.fuse_norm):
                        partials[row, split, 25] = stats[mi, 1]
        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


class _FusedFinalize:
    def __init__(self, hidden, rms_eps, hc_eps, norm_eps, sinkhorn_iters, fuse_norm):
        self.hidden, self.rms_eps, self.hc_eps = hidden, rms_eps, hc_eps
        self.norm_eps, self.sinkhorn_iters = norm_eps, sinkhorn_iters
        self.fuse_norm = fuse_norm
        self.splits = hidden // 32

    @cute.jit
    def __call__(
        self,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        norm: cute.Tensor,
        y: cute.Tensor,
        pre: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(partials, scale, bias, norm, y, pre, post, comb).launch(
            grid=(tokens, 1, 1),
            block=(160 if self.fuse_norm else 32, 1, 1),
            stream=stream,
            use_pdl=_MHC_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        partials: cute.Tensor,
        scale: cute.Tensor,
        bias: cute.Tensor,
        norm: cute.Tensor,
        y: cute.Tensor,
        pre: cute.Tensor,
        post: cute.Tensor,
        comb: cute.Tensor,
    ):
        if const_expr(_MHC_PDL):
            cute.arch.griddepcontrol_wait()
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        if tid < 32:
            value = Float32(0)
            if lane < 25:
                for split in cutlass.range_constexpr(self.splits):
                    value += Float32(partials[row, split, lane])
            square = Float32(cute.arch.shuffle_sync(value, Int32(0)))
            inv = cute.math.rsqrt(
                square / Float32(4 * self.hidden) + Float32(self.rms_eps), fastmath=True
            )
            prevalue = Float32(cute.arch.shuffle_sync(value, lane + 1))
            postvalue = Float32(cute.arch.shuffle_sync(value, lane + 5))
            cvalue = Float32(cute.arch.shuffle_sync(value, lane + 9))
            if lane < 4:
                pre[row, lane] = Float32(1) / (
                    Float32(1)
                    + cute.math.exp(
                        -(prevalue * inv * Float32(scale[0]) + Float32(bias[lane])),
                        fastmath=True,
                    )
                ) + Float32(self.hc_eps)
                post[row, lane] = Float32(2) / (
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
                comb[row, lane // 4, lane % 4] = cvalue
        elif const_expr(self.fuse_norm):
            square = Float32(0)
            for group in cutlass.range_constexpr((self.splits + 31) // 32):
                split = lane + group * 32
                if split < self.splits:
                    square += Float32(partials[row, split, 25])
            square = _warp_allreduce_sum(square)
            inv = cute.math.rsqrt(
                square / Float32(self.hidden) + Float32(self.norm_eps), fastmath=True
            )
            values = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
            weights = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
            next_values = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
            next_weights = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
            pairs = cute.recast_tensor(y, Uint32)
            if const_expr(norm.element_type == cutlass.BFloat16):
                norm_pairs = cute.recast_tensor(norm, Uint32)
            for j in cutlass.range_constexpr(2):
                h = tid - 32 + j * 128
                packed = pairs[row, h]
                values[j * 2] = u32_as_f32(packed << Uint32(16))
                values[j * 2 + 1] = u32_as_f32(packed & Uint32(0xFFFF0000))
                if const_expr(norm.element_type == cutlass.BFloat16):
                    packed_weight = norm_pairs[h]
                    weights[j * 2] = u32_as_f32(packed_weight << Uint32(16))
                    weights[j * 2 + 1] = u32_as_f32(packed_weight & Uint32(0xFFFF0000))
                else:
                    weights[j * 2], weights[j * 2 + 1] = norm[h * 2], norm[h * 2 + 1]
            for chunk in range(self.hidden // 512):
                if chunk + 1 < self.hidden // 512:
                    for j in cutlass.range_constexpr(2):
                        h = (chunk + 1) * 256 + tid - 32 + j * 128
                        packed = pairs[row, h]
                        next_values[j * 2] = u32_as_f32(packed << Uint32(16))
                        next_values[j * 2 + 1] = u32_as_f32(packed & Uint32(0xFFFF0000))
                        if const_expr(norm.element_type == cutlass.BFloat16):
                            packed_weight = norm_pairs[h]
                            next_weights[j * 2] = u32_as_f32(
                                packed_weight << Uint32(16)
                            )
                            next_weights[j * 2 + 1] = u32_as_f32(
                                packed_weight & Uint32(0xFFFF0000)
                            )
                        else:
                            next_weights[j * 2], next_weights[j * 2 + 1] = (
                                norm[h * 2],
                                norm[h * 2 + 1],
                            )
                for j in cutlass.range_constexpr(2):
                    h = chunk * 256 + tid - 32 + j * 128
                    pairs[row, h] = _pack_bf16_pair(
                        values[j * 2] * inv * weights[j * 2],
                        values[j * 2 + 1] * inv * weights[j * 2 + 1],
                    )
                for j in cutlass.range_constexpr(4):
                    values[j], weights[j] = next_values[j], next_weights[j]
        if const_expr(_MHC_PDL):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()


@cache
def _partial(hidden, pre_only, tile_m, fuse_norm):
    return _FusedPartial(hidden, pre_only, tile_m, fuse_norm)


@cache
def _finalize(hidden, rms_eps, hc_eps, norm_eps, sinkhorn_iters, fuse_norm):
    return _FusedFinalize(hidden, rms_eps, hc_eps, norm_eps, sinkhorn_iters, fuse_norm)


def _compact_tensor(value):
    dtype = cutlass.BFloat16 if value.element_size() == 2 else cutlass.Float32
    if compile_only_launches_enabled() and hasattr(value, "fake_mode"):
        from cutlass.cute.runtime import make_fake_tensor

        return make_fake_tensor(
            dtype,
            (cute.sym_int(32), *value.shape[1:]),
            tuple(value.stride()),
            assumed_align=4,
        )
    from cutlass.cute.runtime import from_dlpack

    tensor = from_dlpack(value.detach(), assumed_align=4)
    tensor = tensor.mark_compact_shape_dynamic(
        mode=0, stride_order=tuple(range(value.ndim))
    )
    tensor.element_type = dtype
    return tensor


def _launch(kernel, family, operands, *, _prepared, constants):
    args = tuple(_compact_tensor(value) for value in operands.values())
    args += (Int32(next(iter(operands.values())).shape[0]), current_cuda_stream())
    if _prepared is not None:
        return run_compiled(_prepared, args)
    key = tuple(
        tensor_key(
            name,
            value,
            dims=(DimKey.dynamic(),) + tuple(DimKey.exact(d) for d in value.shape[1:]),
        )
        for name, value in operands.items()
    ) + tuple(constants)
    return _compile_mhc_entry(
        kernel,
        compile_spec=KernelCompileSpec.from_key(family, 4, key),
        compile_args=args,
        runtime_args=args,
    )


def partial(
    *,
    residual,
    fn,
    partials,
    out,
    pre_mix,
    y,
    pre_only=False,
    tile_m=16,
    fuse_norm=True,
    x=None,
    prev_post=None,
    prev_comb=None,
    fn_bf16=None,
    _prepared=None,
):
    operands = dict(
        x=residual if pre_only else x,
        residual=residual,
        post=pre_mix if pre_only else prev_post,
        comb=pre_mix if pre_only else prev_comb,
        fn=fn,
        incoming=pre_mix,
        out=out,
        y=y,
        partials=partials,
    )
    return _launch(
        _partial(residual.shape[-1], pre_only, tile_m, fuse_norm),
        "norm.mhc.fused_tf32_partial",
        operands,
        _prepared=_prepared,
        constants=(
            ("pre_only", pre_only),
            ("tile_m", tile_m),
            ("fuse_norm", fuse_norm),
        ),
    )


def finalize(
    *,
    residual,
    partials,
    scale,
    bias,
    y,
    post,
    comb,
    norm_weight,
    pre_mix,
    pre_out,
    rms_eps,
    hc_eps,
    norm_eps,
    sinkhorn_iters,
    fuse_norm=True,
    _prepared=None,
):
    operands = dict(
        partials=partials,
        scale=scale,
        bias=bias,
        norm=norm_weight if fuse_norm else bias,
        y=y,
        pre=pre_out,
        post=post,
        comb=comb,
    )
    return _launch(
        _finalize(
            residual.shape[-1], rms_eps, hc_eps, norm_eps, sinkhorn_iters, fuse_norm
        ),
        "norm.mhc.fused_tf32_finalize",
        operands,
        _prepared=_prepared,
        constants=(
            ("rms_eps", rms_eps),
            ("hc_eps", hc_eps),
            ("norm_eps", norm_eps),
            ("sinkhorn_iters", sinkhorn_iters),
            ("fuse_norm", fuse_norm),
        ),
    )
