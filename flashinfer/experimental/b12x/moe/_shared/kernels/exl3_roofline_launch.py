"""Roofline SM120 EXL3 2-bit MoE planner, compiler, binder, and executor.

Shares a SINGLE canonical ``trellis_t256`` weight representation in VRAM across:
  - M=1 (int32 routes):   Exl3InputBoundary(fp16) -> Exl3W2A16DecodeFC1(16W, split_k, direct_m1)
                          -> Exl3ActHadF16 -> Exl3W2A16DecodeFC2(16W, N512, direct_m1) -> Exl3OutputBoundary
  - M=1..32 (packed):     Exl3InputBoundary(fp16) -> route_pack(bs=8)
                          -> Exl3W2A16DecodeFC1(16W, fused) -> Exl3W2A16DecodeFC2(16W, N512) -> Exl3OutputBoundary
  - M=33..512 (and >512 when prefill_mode=="w2a16"):
                          Exl3InputBoundary(fp16) -> route_pack(bs=64)
                          -> Exl3W2A16PrefillFC1(16W) -> Exl3W2A16PrefillFC2(16W) -> Exl3OutputBoundary
  - M>512 (when prefill_mode=="mxfp8"):
                          Exl3InputBoundary(mxfp8) -> route_pack(bs=64)
                          -> Exl3MxFC1(16W) -> Exl3MxFC2(16W) -> Exl3OutputBoundary

All execution is 100% allocation-free, host-sync-free, and CUDA-graph capturable.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import cutlass
import cutlass.cute as cute
import torch

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.quant.lut_e4m3 import (
    LUT_E4M3_DIRECT_TABLE_ENTRIES,
    lut_e4m3_direct_table,
)
from b12x._lib.utils.device import current_cuda_stream, get_num_sm
from b12x.moe._shared.kernels.exl3_boundary import (
    Exl3InputBoundaryKernel,
    Exl3OutputBoundaryKernel,
)
from b12x.moe._shared.kernels.exl3_mxfp8_moe import (
    Exl3MxFC1Kernel16W,
    Exl3MxFC2Kernel16W,
)
from b12x.moe._shared.kernels.exl3_w2a16_moe import (
    Exl3ActHadF16Kernel,
    Exl3W2A16DecodeFC1Kernel16W,
    Exl3W2A16DecodeFC2Kernel16W,
    Exl3W2A16PrefillFC1Kernel16W,
    Exl3W2A16PrefillFC2Kernel16W,
)
from b12x.moe._shared.kernels.w4a16.host import route_pack_capacity
from b12x.moe._shared.kernels.w4a16.route_pack import (
    W4A16RoutePackLaunches,
    compile_w4a16_route_pack_launches,
    pack_topk_routes_by_expert,
)

_DECODE_MAX_M = 32
_MXFP8_PREFILL_MIN_M = 513


def exl3_roofline_enabled() -> bool:
    """Whether the SM120 EXL3 2-bit roofline engine is enabled."""
    raw = os.environ.get("B12X_EXL3_ROOFLINE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def exl3_roofline_prefill_mode() -> str:
    """Prefill precision mode for M > 512: ``'w2a16'`` (default, lossless) or ``'mxfp8'``."""
    raw = os.environ.get("B12X_EXL3_PREFILL_MODE", "w2a16").strip().lower()
    if raw not in ("w2a16", "mxfp8"):
        raise ValueError(
            f"B12X_EXL3_PREFILL_MODE must be 'w2a16' or 'mxfp8', got {raw!r}"
        )
    return raw


def exl3_roofline_eligible(
    *,
    enabled: bool | None = None,
    trellis_codebook: str | None,
    trellis_bits: int,
    intermediate_hadamard: bool,
    activation: str,
    swiglu_limit: float | None,
    swiglu_clamp_after_silu: bool,
    dtype: torch.dtype,
    hidden_size: int,
    intermediate_size: int,
    weight_E: int,
    route_E: int,
    apply_router_weight_on_input: bool,
    collect_activation_amax: bool,
    raw_routes: bool,
    fast_math: bool = True,
    deterministic_output: bool = False,
    device: torch.device | None = None,
) -> bool:
    """Return whether a full-rotation Trellis plan can run on the SM12x EXL3 roofline suite."""
    if enabled is None:
        enabled = exl3_roofline_enabled()
    if not enabled or deterministic_output:
        return False
    if device is not None:
        if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] != 12:
            return False
    return (
        trellis_codebook == "lut_e4m3"
        and int(trellis_bits) == 2
        and bool(intermediate_hadamard)
        and activation == "situ"
        and swiglu_limit is None
        and not swiglu_clamp_after_silu
        and dtype == torch.bfloat16
        and int(hidden_size) > 0
        and int(hidden_size) % 512 == 0
        and int(intermediate_size) > 0
        and int(intermediate_size) % 128 == 0
        and int(route_E) == int(weight_E)
        and not apply_router_weight_on_input
        and not collect_activation_amax
        and not raw_routes
        and bool(fast_math)
    )


def exl3_roofline_scratch_geometry(
    *,
    capacity_tokens: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    prefill_mode: str,
) -> tuple[int, int, int, int]:
    """Compute (max_slots, max_blks, max_slots_64, inter_u32_words) for one capacity."""
    cap_m = max(int(capacity_tokens), 1)
    top_k = max(int(top_k), 1)
    I = int(intermediate_size)
    E = int(num_experts)

    dec_m = min(cap_m, _DECODE_MAX_M)
    _, max_slots_8, max_blks_8 = route_pack_capacity(
        dec_m * top_k, 8, E, topk=top_k, bucket_tokens=False
    )
    if cap_m > _DECODE_MAX_M:
        _, max_slots_64, max_blks_64 = route_pack_capacity(
            cap_m * top_k, 64, E, topk=top_k, bucket_tokens=False
        )
    else:
        max_slots_64, max_blks_64 = 0, 0

    max_slots = max(int(max_slots_8), int(max_slots_64), 1)
    max_blks = max(int(max_blks_8), int(max_blks_64), 1)

    # Decode intermediate_f16: [dec_m * top_k, I] fp16 -> dec_m * top_k * (I // 2) u32
    inter_words = dec_m * top_k * (I // 2)
    if cap_m > _DECODE_MAX_M:
        if prefill_mode == "w2a16":
            inter_words = max(inter_words, int(max_slots_64) * (I // 2))
        else:
            # 33..512 uses W2A16 prefill; >512 uses MXFP8 prefill.
            w2a16_pre_m = min(cap_m, _MXFP8_PREFILL_MIN_M - 1)
            _, w2a16_slots_64, _ = route_pack_capacity(
                w2a16_pre_m * top_k, 64, E, topk=top_k, bucket_tokens=False
            )
            w2a16_words = int(w2a16_slots_64) * (I // 2)
            mx_words = int(max_slots_64) * (I // 4) + (I // 128) * int(max_slots_64)
            inter_words = max(inter_words, w2a16_words, mx_words)

    return max_slots, max_blks, int(max_slots_64), int(inter_words)


@dataclass(frozen=True)
class Exl3RooflineLaunches:
    """Precompiled CuTeDSL and Triton carriers for one EXL3 roofline capacity."""

    capacity_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    trellis_bits: int
    prefill_mode: str
    sms: int
    c_in_fp16: object
    c_out_bf16: object
    c_out_f32: object
    c_out_f32_bf16: object
    c_out_f32_f32: object
    c_dec_fc1_m1: object
    c_dec_act: object
    c_dec_fc2_m1: object
    c_dec_fc1_fused: object
    c_dec_fc2_pk: object
    rp_launches_8: tuple[tuple[int, W4A16RoutePackLaunches], ...]
    c_pre_fc1: object | None = None
    c_pre_fc2: object | None = None
    rp_launches_64: tuple[tuple[int, W4A16RoutePackLaunches], ...] = ()
    c_in_mxfp8: object | None = None
    c_mx_fc1: object | None = None
    c_mx_fc2: object | None = None
    direct_lut: torch.Tensor | None = None

    def carriers(self) -> tuple[object, ...]:
        items: list[object] = [
            self.c_in_fp16,
            self.c_out_bf16,
            self.c_out_f32,
            self.c_out_f32_bf16,
            self.c_out_f32_f32,
            self.c_dec_fc1_m1,
            self.c_dec_act,
            self.c_dec_fc2_m1,
            self.c_dec_fc1_fused,
            self.c_dec_fc2_pk,
        ]
        for opt in (
            self.c_pre_fc1,
            self.c_pre_fc2,
            self.c_in_mxfp8,
            self.c_mx_fc1,
            self.c_mx_fc2,
        ):
            if opt is not None:
                items.append(opt)
        for _, rpl in self.rp_launches_8:
            items.extend(rpl.carriers())
        for _, rpl in self.rp_launches_64:
            items.extend(rpl.carriers())
        return tuple(items)

    def select_route_pack(self, tokens: int, block_size: int) -> W4A16RoutePackLaunches:
        table = self.rp_launches_8 if block_size == 8 else self.rp_launches_64
        for cap, rpl in table:
            if tokens <= cap:
                return rpl
        raise RuntimeError(
            f"EXL3 roofline route-pack capacity missing for tokens={tokens}, block_size={block_size}"
        )


@dataclass(frozen=True)
class Exl3RooflineBuffers:
    """Caller-owned arena views for one bound EXL3 roofline invocation."""

    x_rot_f16: torch.Tensor
    a_scales: torch.Tensor
    intermediate_u32: torch.Tensor
    fc1_accum_f32: torch.Tensor
    y_rot_bf16: torch.Tensor
    packed_route_indices: torch.Tensor
    block_expert_ids: torch.Tensor
    packed_route_count: torch.Tensor
    expert_offsets: torch.Tensor
    expert_counts: torch.Tensor
    direct_lut: torch.Tensor
    w13_u32: torch.Tensor
    w2_u32: torch.Tensor
    rot_f16: torch.Tensor
    rot_u16: torch.Tensor
    packed_rot: int
    sign_patterns: torch.Tensor
    sign_table: torch.Tensor
    gate_suh: torch.Tensor
    down_svh: torch.Tensor


def _fake(dtype, shape):
    stride_order = tuple(range(len(shape) - 1, -1, -1))
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=stride_order, assumed_align=16
    )


def compile_exl3_roofline_launches(
    *,
    capacity_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    trellis_bits: int,
    prefill_mode: str,
    device: torch.device,
) -> Exl3RooflineLaunches:
    """Precompile every CuTeDSL and Triton kernel reachable for this capacity."""
    if device.type != "cuda":
        raise RuntimeError("EXL3 roofline launch planning requires a CUDA device")
    if torch.cuda.get_device_capability(device)[0] != 12:
        raise RuntimeError("EXL3 roofline launch planning requires an SM12x Blackwell device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("EXL3 roofline launch planning cannot run during capture")

    cap_m = max(int(capacity_tokens), 1)
    H = int(hidden_size)
    I = int(intermediate_size)
    E = int(num_experts)
    K = int(top_k)
    bits = int(trellis_bits)
    ordinal = device.index if device.index is not None else torch.cuda.current_device()

    w13_u32_numel = 2 * E * (H // 16) * (I // 16) * (8 * bits)
    w2_u32_numel = E * (I // 16) * (H // 16) * (8 * bits)
    rot_f16_numel = E * 3 * I
    sign_table_numel = 8 * 3 * I
    lut_numel = LUT_E4M3_DIRECT_TABLE_ENTRIES

    with torch.cuda.device(ordinal):
        sms = int(get_num_sm(device))
        st = current_cuda_stream()

        sym_m = cute.sym_int()
        sym_a = cute.sym_int()
        sym_s = cute.sym_int()
        sym_int = cute.sym_int()
        sym_acc = cute.sym_int()
        sym_r = cute.sym_int()
        sym_b = cute.sym_int()
        sym_rw = cute.sym_int()
        sym_rot = cute.sym_int()

        in_fp16 = Exl3InputBoundaryKernel(hidden_size=H, mode="fp16")
        out_bf16 = Exl3OutputBoundaryKernel(hidden_size=H, out_dtype="bf16")
        out_f32 = Exl3OutputBoundaryKernel(hidden_size=H, out_dtype="f32")

        c_in_fp16 = b12x_compile(
            in_fp16,
            _fake(cutlass.BFloat16, (sym_m, H)),
            _fake(cutlass.Float16, (H,)),
            _fake(cutlass.Uint32, (sym_a,)),
            _fake(cutlass.Uint8, (sym_m, H // 32)),
            1,
            st,
            compile_spec=KernelCompileSpec.from_fields("exl3.in_fp16", 1, ("H", H)),
        )
        c_out_bf16 = b12x_compile(
            out_bf16,
            _fake(cutlass.BFloat16, (sym_m, H)),
            _fake(cutlass.Float16, (H,)),
            _fake(cutlass.Uint32, (sym_a,)),
            1,
            st,
            compile_spec=KernelCompileSpec.from_fields("exl3.out_bf16", 2, ("H", H)),
        )
        c_out_f32 = b12x_compile(
            out_f32,
            _fake(cutlass.BFloat16, (sym_m, H)),
            _fake(cutlass.Float16, (H,)),
            _fake(cutlass.Float32, (sym_m, H)),
            1,
            st,
            compile_spec=KernelCompileSpec.from_fields("exl3.out_f32", 2, ("H", H)),
        )
        c_out_f32_bf16 = b12x_compile(
            out_bf16,
            _fake(cutlass.Float32, (sym_m, H)),
            _fake(cutlass.Float16, (H,)),
            _fake(cutlass.Uint32, (sym_a,)),
            1,
            st,
            compile_spec=KernelCompileSpec.from_fields("exl3.out_f32_bf16", 2, ("H", H)),
        )
        c_out_f32_f32 = b12x_compile(
            out_f32,
            _fake(cutlass.Float32, (sym_m, H)),
            _fake(cutlass.Float16, (H,)),
            _fake(cutlass.Float32, (sym_m, H)),
            1,
            st,
            compile_spec=KernelCompileSpec.from_fields("exl3.out_f32_f32", 2, ("H", H)),
        )

        dec_fc1_m1 = Exl3W2A16DecodeFC1Kernel16W(
            top_k=K, trellis_bits=bits, split_k_mode=True, direct_m1=True
        )
        dec_fc1_fused = Exl3W2A16DecodeFC1Kernel16W(
            top_k=K, trellis_bits=bits, split_k_mode=False, direct_m1=False
        )
        dec_act = Exl3ActHadF16Kernel()
        dec_fc2_m1 = Exl3W2A16DecodeFC2Kernel16W(
            top_k=K, trellis_bits=bits, tile_n=512, direct_m1=True, use_bf16_scatter=False
        )
        dec_fc2_pk = Exl3W2A16DecodeFC2Kernel16W(
            top_k=K, trellis_bits=bits, tile_n=512, direct_m1=False, use_bf16_scatter=True
        )

        def _comp_dec_fc1(kern, tag: str):
            return b12x_compile(
                kern,
                _fake(cutlass.Uint32, (sym_a,)),
                _fake(cutlass.Uint32, (w13_u32_numel,)),
                _fake(cutlass.Uint32, (sym_int,)),
                _fake(cutlass.Float32, (sym_acc,)),
                _fake(cutlass.Int32, (sym_r,)),
                _fake(cutlass.Int32, (sym_b,)),
                _fake(cutlass.Int32, (1,)),
                _fake(cutlass.Uint8, (lut_numel,)),
                _fake(cutlass.Float16, (sym_rot,)),
                _fake(cutlass.Uint16, (sym_rot,)),
                _fake(cutlass.Uint8, (E,)),
                _fake(cutlass.Float16, (sign_table_numel,)),
                1,
                H // 128,
                I // 128,
                1,
                0,
                sms,
                st,
                compile_spec=KernelCompileSpec.from_fields(
                    tag, 5, ("H", H), ("I", I), ("E", E), ("topk", K), ("bits", bits)
                ),
            )

        def _comp_dec_fc2(kern, tag: str, scatter_dtype):
            return b12x_compile(
                kern,
                _fake(cutlass.Uint32, (sym_int,)),
                _fake(cutlass.Uint32, (w2_u32_numel,)),
                _fake(scatter_dtype, (sym_m, H)),
                _fake(cutlass.Int32, (sym_r,)),
                _fake(cutlass.Int32, (sym_b,)),
                _fake(cutlass.Int32, (1,)),
                _fake(cutlass.Float32, (sym_rw,)),
                _fake(cutlass.Uint8, (lut_numel,)),
                1,
                I // 128,
                H // 512,
                sms,
                st,
                compile_spec=KernelCompileSpec.from_fields(
                    tag, 5, ("H", H), ("I", I), ("E", E), ("topk", K), ("bits", bits)
                ),
            )

        c_dec_fc1_m1 = _comp_dec_fc1(dec_fc1_m1, "exl3.w2a16_dec16w_fc1_m1")
        c_dec_fc1_fused = _comp_dec_fc1(dec_fc1_fused, "exl3.w2a16_dec16w_fc1_fused")
        c_dec_act = b12x_compile(
            dec_act,
            _fake(cutlass.Float32, (sym_acc,)),
            _fake(cutlass.Int32, (sym_rw,)),
            _fake(cutlass.Float16, (sym_rot,)),
            _fake(cutlass.Uint16, (sym_rot,)),
            _fake(cutlass.Uint8, (E,)),
            _fake(cutlass.Float16, (sign_table_numel,)),
            _fake(cutlass.Uint32, (sym_int,)),
            1,
            I // 128,
            0,
            st,
            compile_spec=KernelCompileSpec.from_fields(
                "exl3.w2a16_act_had", 3, ("I", I), ("E", E)
            ),
        )
        c_dec_fc2_m1 = _comp_dec_fc2(
            dec_fc2_m1, "exl3.w2a16_dec16w_fc2_m1_n512_f32", cutlass.Float32
        )
        c_dec_fc2_pk = _comp_dec_fc2(
            dec_fc2_pk, "exl3.w2a16_dec16w_fc2_pk_n512_bf16", cutlass.BFloat16
        )

        dec_cap = min(cap_m, _DECODE_MAX_M)
        rp8_buckets = sorted({t for t in (4, 8, 16, 32) if t < dec_cap} | {dec_cap})
        rp_launches_8 = tuple(
            (
                t,
                compile_w4a16_route_pack_launches(
                    tokens=t,
                    topk=K,
                    block_size=8,
                    num_experts=E,
                    ordinal=ordinal,
                    bucket_tokens=False,
                ),
            )
            for t in rp8_buckets
        )

        c_pre_fc1 = None
        c_pre_fc2 = None
        rp_launches_64: tuple[tuple[int, W4A16RoutePackLaunches], ...] = ()
        c_in_mxfp8 = None
        c_mx_fc1 = None
        c_mx_fc2 = None

        if cap_m > _DECODE_MAX_M:
            pre_fc1 = Exl3W2A16PrefillFC1Kernel16W(top_k=K, trellis_bits=bits, stages=3)
            pre_fc2 = Exl3W2A16PrefillFC2Kernel16W(top_k=K, trellis_bits=bits, stages=4)
            c_pre_fc1 = b12x_compile(
                pre_fc1,
                _fake(cutlass.Uint32, (sym_a,)),
                _fake(cutlass.Uint32, (w13_u32_numel,)),
                _fake(cutlass.Uint32, (sym_int,)),
                _fake(cutlass.Int32, (sym_r,)),
                _fake(cutlass.Int32, (sym_b,)),
                _fake(cutlass.Int32, (1,)),
                _fake(cutlass.Uint8, (lut_numel,)),
                _fake(cutlass.Float16, (sym_rot,)),
                _fake(cutlass.Uint16, (sym_rot,)),
                _fake(cutlass.Uint8, (E,)),
                _fake(cutlass.Float16, (sign_table_numel,)),
                1,
                H // 64,
                I // 128,
                0,
                sms,
                st,
                compile_spec=KernelCompileSpec.from_fields(
                    "exl3.w2a16_pre16w_fc1_v3",
                    5,
                    ("H", H),
                    ("I", I),
                    ("E", E),
                    ("topk", K),
                    ("bits", bits),
                ),
            )
            c_pre_fc2 = b12x_compile(
                pre_fc2,
                _fake(cutlass.Uint32, (sym_int,)),
                _fake(cutlass.Uint32, (w2_u32_numel,)),
                _fake(cutlass.BFloat16, (sym_m, H)),
                _fake(cutlass.Int32, (sym_r,)),
                _fake(cutlass.Int32, (sym_b,)),
                _fake(cutlass.Int32, (1,)),
                _fake(cutlass.Float32, (sym_rw,)),
                _fake(cutlass.Uint8, (lut_numel,)),
                1,
                I // 64,
                H // 256,
                sms,
                st,
                compile_spec=KernelCompileSpec.from_fields(
                    "exl3.w2a16_pre16w_fc2_v3",
                    3,
                    ("H", H),
                    ("I", I),
                    ("E", E),
                    ("topk", K),
                    ("bits", bits),
                ),
            )
            rp64_buckets = sorted(
                {t for t in (64, 256, 512, 1024) if t < cap_m} | {cap_m}
            )
            rp_launches_64 = tuple(
                (
                    t,
                    compile_w4a16_route_pack_launches(
                        tokens=t,
                        topk=K,
                        block_size=64,
                        num_experts=E,
                        ordinal=ordinal,
                        bucket_tokens=False,
                    ),
                )
                for t in rp64_buckets
            )

            if cap_m >= _MXFP8_PREFILL_MIN_M and prefill_mode == "mxfp8":
                in_mxfp8 = Exl3InputBoundaryKernel(hidden_size=H, mode="mxfp8")
                mx_fc1 = Exl3MxFC1Kernel16W(top_k=K, trellis_bits=bits, stages=3)
                mx_fc2 = Exl3MxFC2Kernel16W(
                    top_k=K, trellis_bits=bits, stages=3, direct_f32_atomic=False
                )
                c_in_mxfp8 = b12x_compile(
                    in_mxfp8,
                    _fake(cutlass.BFloat16, (sym_m, H)),
                    _fake(cutlass.Float16, (H,)),
                    _fake(cutlass.Uint32, (sym_a,)),
                    _fake(cutlass.Uint8, (sym_m, H // 32)),
                    1,
                    st,
                    compile_spec=KernelCompileSpec.from_fields(
                        "exl3.in_mxfp8", 1, ("H", H)
                    ),
                )
                c_mx_fc1 = b12x_compile(
                    mx_fc1,
                    _fake(cutlass.Uint32, (sym_a,)),
                    _fake(cutlass.Uint32, (sym_s,)),
                    _fake(cutlass.Uint32, (w13_u32_numel,)),
                    _fake(cutlass.Uint32, (sym_int,)),
                    _fake(cutlass.Int32, (sym_r,)),
                    _fake(cutlass.Int32, (sym_b,)),
                    _fake(cutlass.Int32, (1,)),
                    _fake(cutlass.Uint8, (lut_numel,)),
                    _fake(cutlass.Float16, (sym_rot,)),
                    _fake(cutlass.Uint16, (sym_rot,)),
                    _fake(cutlass.Uint8, (E,)),
                    _fake(cutlass.Float16, (sign_table_numel,)),
                    1,
                    1,
                    H // 128,
                    I // 128,
                    0,
                    sms,
                    st,
                    compile_spec=KernelCompileSpec.from_fields(
                        "exl3.mx_fc1_16w_v5",
                        7,
                        ("H", H),
                        ("I", I),
                        ("E", E),
                        ("topk", K),
                        ("bits", bits),
                    ),
                )
                c_mx_fc2 = b12x_compile(
                    mx_fc2,
                    _fake(cutlass.Uint32, (sym_int,)),
                    _fake(cutlass.Uint32, (w2_u32_numel,)),
                    _fake(cutlass.BFloat16, (sym_m, H)),
                    _fake(cutlass.Int32, (sym_r,)),
                    _fake(cutlass.Int32, (sym_b,)),
                    _fake(cutlass.Int32, (1,)),
                    _fake(cutlass.Float32, (sym_rw,)),
                    _fake(cutlass.Uint8, (lut_numel,)),
                    1,
                    1,
                    I // 128,
                    H // 256,
                    sms,
                    st,
                    compile_spec=KernelCompileSpec.from_fields(
                        "exl3.mx_fc2_16w_v5",
                        5,
                        ("H", H),
                        ("I", I),
                        ("E", E),
                        ("topk", K),
                        ("bits", bits),
                    ),
                )

    return Exl3RooflineLaunches(
        capacity_tokens=cap_m,
        hidden_size=H,
        intermediate_size=I,
        num_experts=E,
        top_k=K,
        trellis_bits=bits,
        prefill_mode=prefill_mode,
        sms=sms,
        c_in_fp16=c_in_fp16,
        c_out_bf16=c_out_bf16,
        c_out_f32=c_out_f32,
        c_out_f32_bf16=c_out_f32_bf16,
        c_out_f32_f32=c_out_f32_f32,
        c_dec_fc1_m1=c_dec_fc1_m1,
        c_dec_act=c_dec_act,
        c_dec_fc2_m1=c_dec_fc2_m1,
        c_dec_fc1_fused=c_dec_fc1_fused,
        c_dec_fc2_pk=c_dec_fc2_pk,
        rp_launches_8=rp_launches_8,
        c_pre_fc1=c_pre_fc1,
        c_pre_fc2=c_pre_fc2,
        rp_launches_64=rp_launches_64,
        c_in_mxfp8=c_in_mxfp8,
        c_mx_fc1=c_mx_fc1,
        c_mx_fc2=c_mx_fc2,
        direct_lut=lut_e4m3_direct_table(device),
    )


def run_exl3_roofline_moe(
    *,
    a: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    launches: Exl3RooflineLaunches,
    buffers: Exl3RooflineBuffers,
    route_expert_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute one bound EXL3 2-bit MoE invocation with zero allocations."""
    m = int(a.shape[0])
    H = launches.hidden_size
    I = launches.intermediate_size
    E = launches.num_experts
    K = launches.top_k
    sms = launches.sms
    live_routes = m * K

    stream = current_cuda_stream()
    y_rot = buffers.y_rot_bf16[:m]
    flat_wts = topk_weights.reshape(-1)

    if output.dtype == torch.bfloat16:
        out_kernel = launches.c_out_bf16
        out_arg = output[:m].view(torch.uint32).reshape(-1)
    elif output.dtype == torch.float32:
        out_kernel = launches.c_out_f32
        out_arg = output[:m]
    else:
        raise TypeError(f"EXL3 roofline output must be bf16 or f32, got {output.dtype}")

    if m == 1 and topk_ids.dtype == torch.int32 and route_expert_map is None:
        a_u32 = buffers.x_rot_f16[:1].view(torch.uint32).reshape(-1)
        flat_ids = topk_ids.reshape(-1)
        launches.c_in_fp16(
            a, buffers.gate_suh, a_u32, buffers.a_scales[:1], 1, stream
        )
        buffers.fc1_accum_f32[: live_routes * 2 * I].zero_()
        k128_tiles = H // 128
        i128_tiles = I // 128
        base_ctas = K * i128_tiles
        split_k = max(1, min(k128_tiles, sms // max(base_ctas, 1)))
        launches.c_dec_fc1_m1(
            a_u32,
            buffers.w13_u32,
            buffers.intermediate_u32,
            buffers.fc1_accum_f32,
            flat_ids,
            buffers.block_expert_ids[:1],
            buffers.packed_route_count,
            buffers.direct_lut,
            buffers.rot_f16,
            buffers.rot_u16,
            buffers.sign_patterns,
            buffers.sign_table,
            live_routes,
            k128_tiles,
            i128_tiles,
            split_k,
            buffers.packed_rot,
            sms,
            stream,
        )
        launches.c_dec_act(
            buffers.fc1_accum_f32,
            flat_ids,
            buffers.rot_f16,
            buffers.rot_u16,
            buffers.sign_patterns,
            buffers.sign_table,
            buffers.intermediate_u32,
            live_routes,
            i128_tiles,
            buffers.packed_rot,
            stream,
        )
        # Reuse dead fc1_accum_f32[:H] as FP32 scatter accumulator for M=1:
        # 3x higher accuracy (rel_l2=0.00177) and bit-repeatable BF16 output at zero latency cost!
        y_rot_f32 = buffers.fc1_accum_f32[:H].view(1, H)
        y_rot_f32.zero_()
        launches.c_dec_fc2_m1(
            buffers.intermediate_u32,
            buffers.w2_u32,
            y_rot_f32,
            flat_ids,
            buffers.block_expert_ids[:1],
            buffers.packed_route_count,
            flat_wts,
            buffers.direct_lut,
            live_routes,
            i128_tiles,
            H // 512,
            sms,
            stream,
        )
        out_f32_kernel = (
            launches.c_out_f32_bf16
            if output.dtype == torch.bfloat16
            else launches.c_out_f32_f32
        )
        out_f32_kernel(y_rot_f32, buffers.down_svh, out_arg, 1, stream)
        return output[:1]

    if m <= _DECODE_MAX_M:
        a_u32 = buffers.x_rot_f16[:m].view(torch.uint32).reshape(-1)
        rpl = launches.select_route_pack(m, 8)
        pri = buffers.packed_route_indices[: rpl.max_packed_routes]
        bei = buffers.block_expert_ids[: rpl.max_route_blocks]
        launches.c_in_fp16(
            a, buffers.gate_suh, a_u32, buffers.a_scales[:m], m, stream
        )
        pack_topk_routes_by_expert(
            topk_ids,
            8,
            E,
            expert_map=route_expert_map,
            packed_route_indices=pri,
            block_expert_ids=bei,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            launches=rpl,
        )
        y_rot.zero_()
        launches.c_dec_fc1_fused(
            a_u32,
            buffers.w13_u32,
            buffers.intermediate_u32,
            buffers.fc1_accum_f32,
            pri,
            bei,
            buffers.packed_route_count,
            buffers.direct_lut,
            buffers.rot_f16,
            buffers.rot_u16,
            buffers.sign_patterns,
            buffers.sign_table,
            live_routes,
            H // 128,
            I // 128,
            1,
            buffers.packed_rot,
            sms,
            stream,
        )
        launches.c_dec_fc2_pk(
            buffers.intermediate_u32,
            buffers.w2_u32,
            y_rot,
            pri,
            bei,
            buffers.packed_route_count,
            flat_wts,
            buffers.direct_lut,
            live_routes,
            I // 128,
            H // 512,
            sms,
            stream,
        )
        out_kernel(y_rot, buffers.down_svh, out_arg, m, stream)
        return output[:m]

    # Prefill (M > 32): W2A8-MXFP8 for M >= 513 when prefill_mode == "mxfp8",
    # lossless W2A16 Prefill16W otherwise.
    rpl = launches.select_route_pack(m, 64)
    pri = buffers.packed_route_indices[: rpl.max_packed_routes]
    bei = buffers.block_expert_ids[: rpl.max_route_blocks]

    if m >= _MXFP8_PREFILL_MIN_M and launches.prefill_mode == "mxfp8":
        assert launches.c_in_mxfp8 is not None
        assert launches.c_mx_fc1 is not None
        assert launches.c_mx_fc2 is not None
        # Alias packed_a (uint8 [m, H]) inside x_rot_f16 (float16 [cap_m, H]).
        pa_u32 = buffers.x_rot_f16.view(torch.uint32).reshape(-1)[: m * (H // 4)]
        asc = buffers.a_scales[:m]
        asc_u32 = asc.view(torch.uint32).reshape(-1)
        launches.c_in_mxfp8(a, buffers.gate_suh, pa_u32, asc, m, stream)
        pack_topk_routes_by_expert(
            topk_ids,
            64,
            E,
            expert_map=route_expert_map,
            packed_route_indices=pri,
            block_expert_ids=bei,
            packed_route_count=buffers.packed_route_count,
            expert_offsets=buffers.expert_offsets,
            expert_counts=buffers.expert_counts,
            launches=rpl,
        )
        y_rot.zero_()
        launches.c_mx_fc1(
            pa_u32,
            asc_u32,
            buffers.w13_u32,
            buffers.intermediate_u32,
            pri,
            bei,
            buffers.packed_route_count,
            buffers.direct_lut,
            buffers.rot_f16,
            buffers.rot_u16,
            buffers.sign_patterns,
            buffers.sign_table,
            live_routes,
            rpl.max_packed_routes,
            H // 128,
            I // 128,
            buffers.packed_rot,
            sms,
            stream,
        )
        launches.c_mx_fc2(
            buffers.intermediate_u32,
            buffers.w2_u32,
            y_rot,
            pri,
            bei,
            buffers.packed_route_count,
            flat_wts,
            buffers.direct_lut,
            live_routes,
            rpl.max_packed_routes,
            I // 128,
            H // 256,
            sms,
            stream,
        )
        out_kernel(y_rot, buffers.down_svh, out_arg, m, stream)
        return output[:m]

    assert launches.c_pre_fc1 is not None
    assert launches.c_pre_fc2 is not None
    a_u32 = buffers.x_rot_f16[:m].view(torch.uint32).reshape(-1)
    launches.c_in_fp16(a, buffers.gate_suh, a_u32, buffers.a_scales[:m], m, stream)
    pack_topk_routes_by_expert(
        topk_ids,
        64,
        E,
        expert_map=route_expert_map,
        packed_route_indices=pri,
        block_expert_ids=bei,
        packed_route_count=buffers.packed_route_count,
        expert_offsets=buffers.expert_offsets,
        expert_counts=buffers.expert_counts,
        launches=rpl,
    )
    y_rot.zero_()
    launches.c_pre_fc1(
        a_u32,
        buffers.w13_u32,
        buffers.intermediate_u32,
        pri,
        bei,
        buffers.packed_route_count,
        buffers.direct_lut,
        buffers.rot_f16,
        buffers.rot_u16,
        buffers.sign_patterns,
        buffers.sign_table,
        live_routes,
        H // 64,
        I // 128,
        buffers.packed_rot,
        sms,
        stream,
    )
    launches.c_pre_fc2(
        buffers.intermediate_u32,
        buffers.w2_u32,
        y_rot,
        pri,
        bei,
        buffers.packed_route_count,
        flat_wts,
        buffers.direct_lut,
        live_routes,
        I // 64,
        H // 256,
        sms,
        stream,
    )
    out_kernel(y_rot, buffers.down_svh, out_arg, m, stream)
    return output[:m]


__all__ = [
    "Exl3RooflineBuffers",
    "Exl3RooflineLaunches",
    "compile_exl3_roofline_launches",
    "exl3_roofline_eligible",
    "exl3_roofline_enabled",
    "exl3_roofline_prefill_mode",
    "exl3_roofline_scratch_geometry",
    "lut_e4m3_direct_table",
    "run_exl3_roofline_moe",
]
