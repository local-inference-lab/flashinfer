# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JIT compilers for packed-contiguous SM120 FP8 FMHA kernels.

Ragged and paged compilers use symbolic token, batch, page-pool, and
block-table dimensions. Sequence lengths are runtime metadata and never part
of the compiler cache key.

Requires a compatible CuTe DSL package providing ``cutlass.experimental``.
"""

import functools
from pathlib import Path

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32

_CUTLASS_DTYPE = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


def _validate_balanced_scheduler(is_causal: bool, balanced_scheduler: bool) -> None:
    if balanced_scheduler and not is_causal:
        raise ValueError("balanced_scheduler=True requires is_causal=True")


def _compile_sm120_fmha_fp8_ragged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
    enable_skip_softmax: bool = False,
):
    """Compile one sequence-length-independent packed ragged kernel."""

    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from .fmha_prefill_fp8_tma import SM120FusedMultiHeadAttentionFP8ForwardTMA

    in_ct = _CUTLASS_DTYPE[in_dtype]
    out_ct = _CUTLASS_DTYPE[out_dtype]

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=in_ct,
        out_dtype=out_ct,
        is_causal=is_causal,
        head_tile=head_dim,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=False,
        balanced_scheduler=balanced_scheduler,
    )

    sym_total_q = cute.sym_int()
    sym_total_k = cute.sym_int()
    sym_cu = cute.sym_int()
    sym_b = cute.sym_int()
    fake_q = make_fake_compact_tensor(
        in_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_o = make_fake_compact_tensor(
        out_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_lse = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_total_q, num_qo_heads),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )
    fake_k = make_fake_compact_tensor(
        in_ct,
        (sym_total_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_v = make_fake_compact_tensor(
        in_ct,
        (sym_total_k, num_kv_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_cu_seqlens_q = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)
    fake_cu_seqlens_k = make_fake_compact_tensor(Int32, (sym_cu,), assumed_align=4)
    fake_skip_softmax_threshold = (
        make_fake_compact_tensor(cutlass.Float32, (sym_b,), assumed_align=4)
        if enable_skip_softmax
        else None
    )
    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        cutlass.Float32(1.0),  # softmax_scale_log2 placeholder
        cutlass.Float32(1.0),  # output_scale placeholder
        fake_skip_softmax_threshold,
        stream_fake,
        None,  # seqlens_kv
        fake_cu_seqlens_q,
        None,  # block_tables (non-paged)
        fake_cu_seqlens_k,
        cutlass.Int32(1),  # runtime max_seqlen_q placeholder
        True,  # use_pdl placeholder (runtime-dynamic)
        options="--enable-tvm-ffi",
    )


def _compile_sm120_fmha_fp8_paged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    num_tokens_per_page: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
    enable_skip_softmax: bool = False,
    with_kv_tile_ranges: bool = False,
    gqa_pack_size: int = 1,
    kv_layout: str = "HND",
    with_output_gate: bool = False,
    with_relative_bias: bool = False,
    with_row_sink: bool = False,
    window_left: int = -1,
    with_factored_relative_bias: bool = False,
):
    """Compile one sequence-length-independent packed-Q paged kernel."""
    _validate_balanced_scheduler(is_causal, balanced_scheduler)
    if kv_layout not in ("HND", "NHD"):
        raise ValueError("kv_layout must be HND or NHD")

    from .fmha_prefill_fp8_tma import SM120FusedMultiHeadAttentionFP8ForwardTMA

    in_ct = _CUTLASS_DTYPE[in_dtype]
    out_ct = _CUTLASS_DTYPE[out_dtype]

    fmha = SM120FusedMultiHeadAttentionFP8ForwardTMA(
        in_dtype=in_ct,
        out_dtype=out_ct,
        is_causal=is_causal,
        head_tile=head_dim,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=True,
        num_tokens_per_page=num_tokens_per_page,
        balanced_scheduler=balanced_scheduler,
        gqa_pack_size=gqa_pack_size,
        paged_kv_layout=kv_layout,
        window_left=window_left,
    )

    sym_b = cute.sym_int()
    sym_m = cute.sym_int()
    sym_num_pages = cute.sym_int()
    sym_total_q = cute.sym_int()
    sym_seqlens = cute.sym_int()
    sym_cu_q = cute.sym_int()
    # Split KV writes num_chunks partial O/LSE rows per Q token.
    sym_total_o = cute.sym_int() if with_kv_tile_ranges else sym_total_q
    fake_q = make_fake_compact_tensor(
        in_ct,
        (sym_total_q, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_o = make_fake_compact_tensor(
        out_ct,
        (sym_total_o, num_qo_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    fake_output_gate = (
        cute.runtime.make_fake_tensor(
            out_ct,
            (sym_total_q, num_qo_heads, head_dim),
            stride=(cute.sym_int(), cute.sym_int(), 1),
            assumed_align=16,
        )
        if with_output_gate
        else None
    )
    fake_relative_bias = (
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), num_qo_heads, cute.sym_int()),
            stride=(cute.sym_int(), cute.sym_int(), 1),
            assumed_align=2,
        )
        if with_relative_bias
        else None
    )
    fake_relative_factor = (
        cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), num_qo_heads, 16),
            stride=(cute.sym_int(), cute.sym_int(), 1),
            assumed_align=4,
        )
        if with_factored_relative_bias
        else None
    )
    fake_relative_projection = (
        make_fake_compact_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), 16),
            stride_order=(1, 0),
            assumed_align=4,
        )
        if with_factored_relative_bias
        else None
    )
    fake_row_sink = (
        cute.runtime.make_fake_tensor(
            cutlass.Float32,
            (cute.sym_int(), num_qo_heads),
            stride=(cute.sym_int(), 1),
            assumed_align=4,
        )
        if with_row_sink
        else None
    )
    fake_lse = (
        make_fake_compact_tensor(
            cutlass.Float32,
            (sym_total_o, num_qo_heads),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if with_lse
        else None
    )

    # Logical HND paged KV pool: (num_pages, Hkv, num_tokens_per_page, D). Keep the
    # outer page stride symbolic so a plane view from a combined
    # [num_pages, 2, Hkv, page_size, D] allocation has the same compiled ABI
    # as a standalone compact HND pool.
    fake_k = cute.runtime.make_fake_tensor(
        in_ct,
        (sym_num_pages, num_kv_heads, num_tokens_per_page, head_dim),
        stride=(
            cute.sym_int(),
            num_tokens_per_page * head_dim if kv_layout == "HND" else head_dim,
            head_dim if kv_layout == "HND" else num_kv_heads * head_dim,
            1,
        ),
        assumed_align=16,
    )
    fake_v = cute.runtime.make_fake_tensor(
        in_ct,
        (sym_num_pages, num_kv_heads, num_tokens_per_page, head_dim),
        stride=(
            cute.sym_int(),
            num_tokens_per_page * head_dim if kv_layout == "HND" else head_dim,
            head_dim if kv_layout == "HND" else num_kv_heads * head_dim,
            1,
        ),
        assumed_align=16,
    )

    fake_seqlens_kv = make_fake_compact_tensor(Int32, (sym_seqlens,), assumed_align=4)
    fake_cu_seqlens_q = make_fake_compact_tensor(Int32, (sym_cu_q,), assumed_align=4)
    fake_skip_softmax_threshold = (
        make_fake_compact_tensor(cutlass.Float32, (sym_b,), assumed_align=4)
        if enable_skip_softmax
        else None
    )
    # Both dimensions are dynamic: M is a runtime row stride, not a
    # sequence-capacity specialization.
    fake_block_tables = make_fake_compact_tensor(
        Int32,
        (sym_b, sym_m),
        stride_order=(1, 0),
        assumed_align=4,
    )

    fake_kv_tile_ranges = (
        make_fake_compact_tensor(
            Int32, (cute.sym_int(), 2), stride_order=(1, 0), assumed_align=4
        )
        if with_kv_tile_ranges
        else None
    )

    stream_fake = make_fake_stream(use_tvm_ffi_env_stream=True)

    if with_relative_bias or with_row_sink or with_factored_relative_bias:
        # The bias and sink follow every other optional operand of ``_launch``.
        return cute.compile(
            fmha._launch_extended,
            fake_q,
            fake_k,
            fake_v,
            fake_o,
            fake_lse,
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            fake_skip_softmax_threshold,
            stream_fake,
            fake_seqlens_kv,
            fake_cu_seqlens_q,
            fake_block_tables,
            None,
            cutlass.Int32(1),
            True,
            fake_output_gate,
            fake_kv_tile_ranges,
            fake_relative_bias,
            fake_row_sink,
            fake_relative_factor,
            fake_relative_projection,
            options="--enable-tvm-ffi",
        )
    return cute.compile(
        fmha._launch if with_output_gate else fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        fake_lse,
        cutlass.Float32(1.0),  # softmax_scale_log2 placeholder
        cutlass.Float32(1.0),  # output_scale placeholder
        fake_skip_softmax_threshold,
        stream_fake,
        fake_seqlens_kv,
        fake_cu_seqlens_q,
        fake_block_tables,
        None,  # cu_seqlens_k
        cutlass.Int32(1),  # runtime max_seqlen_q placeholder
        True,  # use_pdl placeholder (runtime-dynamic)
        *((fake_output_gate,) if with_output_gate else ()),
        *(
            (fake_kv_tile_ranges,)
            if not with_output_gate or with_kv_tile_ranges
            else ()
        ),
        options="--enable-tvm-ffi",
    )


def _cache_key_files():
    root = Path(__file__).resolve().parent
    return (
        str(Path(__file__).resolve()),
        str(root / "fmha_prefill_fp8_tma.py"),
    )


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.").replace(".", "_")


@functools.cache
def compile_sm120_fmha_fp8_ragged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
    enable_skip_softmax: bool = False,
):
    _validate_balanced_scheduler(is_causal, balanced_scheduler)

    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    device = torch.device(device)
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("SM120 PRIMS FMHA compilation requires SM120")
    kernel_name = (
        f"ragged_{_dtype_name(in_dtype)}_{_dtype_name(out_dtype)}"
        f"_hq{num_qo_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_causal{int(is_causal)}_kt{kv_tile}_qt{q_tile}"
        f"_lse{int(with_lse)}_balanced{int(balanced_scheduler)}"
        f"_skip{int(enable_skip_softmax)}"
    )
    return build_and_load_cute_dsl_kernel(
        "sm120_prims_fmha_fp8",
        kernel_name,
        lambda: _compile_sm120_fmha_fp8_ragged_kernel(
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            is_causal,
            kv_tile,
            q_tile,
            device,
            with_lse,
            balanced_scheduler,
            enable_skip_softmax,
        ),
        extra_key_files=_cache_key_files(),
    )


@functools.cache
def compile_sm120_fmha_fp8_paged_kernel(
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    is_causal: bool,
    kv_tile: int,
    q_tile: int,
    num_tokens_per_page: int,
    device: torch.device,
    with_lse: bool = False,
    balanced_scheduler: bool = False,
    enable_skip_softmax: bool = False,
    with_kv_tile_ranges: bool = False,
    gqa_pack_size: int = 1,
    kv_layout: str = "HND",
    with_output_gate: bool = False,
    with_relative_bias: bool = False,
    with_row_sink: bool = False,
    window_left: int = -1,
    with_factored_relative_bias: bool = False,
):
    """Compile the paged FP8 prefill program for one layer geometry.

    ``with_relative_bias``, ``with_row_sink``, ``with_factored_relative_bias``
    and ``with_output_gate`` add operands to the launch, in the order
    ``output_gate, kv_tile_ranges, relative_bias, row_sink, relative_factor,
    relative_projection`` after ``use_pdl``; a program with a bias or sink takes
    all six, with ``None`` for the ones it was compiled without.
    ``with_factored_relative_bias`` evaluates the bias
    ``round_bf16(relative_factor[row, head, :] @ relative_projection[d, :])``
    in the kernel in place of reading ``relative_bias``.  The
    compiled program does not inspect its operands: call
    :func:`validate_sm120_fmha_fp8_paged_operands` on the tensors before the
    first launch or graph capture.
    """
    _validate_balanced_scheduler(is_causal, balanced_scheduler)
    if type(window_left) is not int or window_left < -1:
        raise ValueError(
            f"window_left must be -1 (no window) or a nonnegative int, got {window_left!r}"
        )
    if with_factored_relative_bias and with_relative_bias:
        raise ValueError(
            "with_factored_relative_bias replaces with_relative_bias; pass one"
        )
    if with_factored_relative_bias and (gqa_pack_size, q_tile, kv_tile) != (
        1,
        128,
        128,
    ):
        raise NotImplementedError(
            "the factored relative bias requires gqa_pack_size 1 and 128-row "
            "Q and K/V tiles"
        )
    with_relative_bias_operand = with_relative_bias
    with_relative_bias = with_relative_bias or with_factored_relative_bias
    extras = with_relative_bias or with_row_sink or window_left >= 0
    if window_left >= 0 and not is_causal:
        raise ValueError("window_left requires is_causal=True")
    if extras and (head_dim, q_tile, kv_tile) == (256, 128, 128):
        raise NotImplementedError(
            "relative bias, row sink and window are not implemented for "
            "head_dim 256 with 128-row Q and K/V tiles"
        )
    if extras and (enable_skip_softmax or with_kv_tile_ranges):
        raise NotImplementedError(
            "relative bias, row sink and window exclude skip-softmax and "
            "K/V tile ranges"
        )
    if (with_relative_bias or with_row_sink) and with_lse:
        raise NotImplementedError("relative bias and row sink exclude the LSE output")

    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    device = torch.device(device)
    if torch.cuda.get_device_capability(device) not in ((12, 0), (12, 1)):
        raise RuntimeError("Paged FP8 TMA FMHA compilation requires SM120 or SM121")
    kernel_name = (
        f"paged_{kv_layout.lower()}_{_dtype_name(in_dtype)}_{_dtype_name(out_dtype)}"
        f"_hq{num_qo_heads}_hkv{num_kv_heads}_d{head_dim}"
        f"_causal{int(is_causal)}_kt{kv_tile}_qt{q_tile}"
        f"_page{num_tokens_per_page}_lse{int(with_lse)}"
        f"_balanced{int(balanced_scheduler)}"
        f"_skip{int(enable_skip_softmax)}_gate{int(with_output_gate)}"
        + ("_kvsplit" if with_kv_tile_ranges else "")
        + (f"_gqa{gqa_pack_size}" if gqa_pack_size > 1 else "")
        + ("_relbias" if with_relative_bias_operand else "")
        + ("_rowsink" if with_row_sink else "")
        + (f"_win{window_left}" if window_left >= 0 else "")
        + ("_relrank16" if with_factored_relative_bias else "")
    )
    return build_and_load_cute_dsl_kernel(
        "sm120_prims_fmha_fp8",
        kernel_name,
        lambda: _compile_sm120_fmha_fp8_paged_kernel(
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            is_causal,
            kv_tile,
            q_tile,
            num_tokens_per_page,
            device,
            with_lse,
            balanced_scheduler,
            enable_skip_softmax,
            with_kv_tile_ranges,
            gqa_pack_size,
            kv_layout,
            with_output_gate,
            with_relative_bias_operand,
            with_row_sink,
            window_left,
            with_factored_relative_bias,
        ),
        extra_key_files=_cache_key_files(),
    )


def validate_sm120_fmha_fp8_paged_operands(
    q: torch.Tensor,
    *,
    relative_bias: torch.Tensor | None = None,
    row_sink: torch.Tensor | None = None,
    output_gate: torch.Tensor | None = None,
    relative_factor: torch.Tensor | None = None,
    relative_projection: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Check the optional operands of a paged FP8 prefill launch against ``q``.

    ``q`` is the packed query ``[total_q, Hq, D]``.  The kernel indexes each
    optional operand by the query row and head without bounds checks, so each
    must cover every row of ``q`` on its device:

    - ``relative_bias``: BF16 ``[rows >= total_q, Hq, extent >= 1]``, unit
      inner stride.
    - ``row_sink``: float32 ``[rows >= total_q, Hq]``, unit inner stride.
    - ``output_gate``: ``out_dtype`` ``[rows >= total_q, Hq, D]``, unit inner
      stride, 16-byte aligned data and outer strides.
    - ``relative_factor``: BF16 ``[rows >= total_q, Hq, 16]``, unit inner
      stride, 4-byte aligned data and outer strides; given together with
    - ``relative_projection``: contiguous BF16 ``[extent >= 1, 16]``, 4-byte
      aligned.  The pair replaces ``relative_bias``.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [total_q, Hq, D], got {tuple(q.shape)}")
    rows, heads, dim = (int(n) for n in q.shape)

    def check(name, tensor, dtype, trailing, layout):
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on {q.device}, got {tensor.device}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must be {dtype}, got {tensor.dtype}")
        if (
            tensor.ndim != 1 + len(trailing)
            or int(tensor.shape[0]) < rows
            or any(
                want is not None and int(got) != want
                for got, want in zip(tensor.shape[1:], trailing, strict=False)
            )
            or any(int(n) < 1 for n in tensor.shape[1:])
        ):
            raise ValueError(
                f"{name} must be {layout} with at least {rows} rows, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.stride(-1) != 1 or any(step <= 0 for step in tensor.stride()):
            raise ValueError(
                f"{name} must have a unit inner stride and positive outer "
                f"strides, got {tuple(tensor.stride())}"
            )

    if (relative_factor is None) != (relative_projection is None):
        raise ValueError("relative_factor and relative_projection come together")
    if relative_factor is not None:
        if relative_bias is not None:
            raise ValueError(
                "relative_factor and relative_projection replace relative_bias"
            )
        check(
            "relative_factor",
            relative_factor,
            torch.bfloat16,
            (heads, 16),
            f"[rows, {heads}, 16]",
        )
        if relative_factor.data_ptr() % 4 or any(
            step % 2 for step in relative_factor.stride()[:-1]
        ):
            raise ValueError(
                "relative_factor data and outer strides must be 4-byte aligned"
            )
        if (
            relative_projection.device != q.device
            or relative_projection.dtype != torch.bfloat16
            or relative_projection.ndim != 2
            or relative_projection.shape[0] < 1
            or relative_projection.shape[1] != 16
            or not relative_projection.is_contiguous()
            or relative_projection.data_ptr() % 4
        ):
            raise ValueError(
                "relative_projection must be contiguous, 4-byte aligned "
                f"torch.bfloat16 [extent, 16] on {q.device}, got "
                f"{relative_projection.dtype} {tuple(relative_projection.shape)}"
            )
    if relative_bias is not None:
        check(
            "relative_bias",
            relative_bias,
            torch.bfloat16,
            (heads, None),
            f"[rows, {heads}, extent]",
        )
    if row_sink is not None:
        check("row_sink", row_sink, torch.float32, (heads,), f"[rows, {heads}]")
    if output_gate is not None:
        check(
            "output_gate",
            output_gate,
            out_dtype,
            (heads, dim),
            f"[rows, {heads}, {dim}]",
        )
        item = output_gate.element_size()
        if output_gate.data_ptr() % 16 or any(
            step * item % 16 for step in output_gate.stride()[:-1]
        ):
            raise ValueError(
                "output_gate data and outer strides must be 16-byte aligned"
            )
