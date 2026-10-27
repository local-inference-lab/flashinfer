"""Public surface for quantization.mxfp8 (docs in the op ``__init__``)."""

from __future__ import annotations

import torch

from ..._lib.gating import default_is_supported
from ...gemm._shared.wo_mxfp8 import MXFP8Rows, mxfp8_rows_from_bases
from . import META
from ._preparation import (
    Mxfp8Config,
    Mxfp8Query,
    plan,
    query_from_call,
    quantize_dense_rows,
    quantize_rows,
)


def dense_rows_storage_shapes(m: int, k: int) -> tuple[tuple[int, ...], ...]:
    """Shapes of caller-owned values, compact scales, and physical MMA scales.

    Values use E4M3 storage; both scale bases use uint8 storage. The returned
    shapes cover one dense-GEMM batch and include MMA scale padding.
    """
    if type(m) is not int or m <= 0 or type(k) is not int or k <= 0 or k % 128:
        raise ValueError("dense MXFP8 rows require positive M and K divisible by 128")
    return ((m, k), (1, m, k // 32), (1, (m + 127) // 128, k // 128, 32, 4, 4))


def dense_rows_from_bases(
    values_base: torch.Tensor,
    scale_rows_base: torch.Tensor,
    scale_mma_base: torch.Tensor,
) -> MXFP8Rows:
    """Build canonical dense-GEMM views over caller-owned contiguous buffers."""
    if values_base.ndim != 2:
        raise ValueError("dense MXFP8 values must have shape [M,K]")
    m, k = (int(size) for size in values_base.shape)
    expected = dense_rows_storage_shapes(m, k)
    for name, tensor, shape, dtype in (
        ("values", values_base, expected[0], torch.float8_e4m3fn),
        ("scale_rows", scale_rows_base, expected[1], torch.uint8),
        ("scale_mma", scale_mma_base, expected[2], torch.uint8),
    ):
        if tensor.shape != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"dense MXFP8 {name} base differs from its storage contract"
            )
        if tensor.device != values_base.device or tensor.device.type != "cuda":
            raise ValueError("dense MXFP8 bases must share one CUDA device")
        if tensor.data_ptr() % 16:
            raise ValueError(f"dense MXFP8 {name} base must be 16-byte aligned")
    return mxfp8_rows_from_bases(
        values_base, scale_rows_base, scale_mma_base, m, k, num_groups=1
    )


def dense_scale_storage(scale_mma: torch.Tensor) -> torch.Tensor:
    """Return the contiguous physical uint8 base of a canonical MMA-scale view.

    The returned view shares storage with ``scale_mma`` and can be passed
    directly to ``dense_rows_from_bases`` or an opaque prepared launch.
    """
    if scale_mma.ndim != 6:
        raise ValueError("dense MXFP8 MMA scales require six logical dimensions")
    row_tiles, k_tiles = int(scale_mma.shape[2]), int(scale_mma.shape[4])
    logical_shape = (32, 4, row_tiles, 4, k_tiles, 1)
    logical_stride = (16, 4, k_tiles * 512, 1, 512, row_tiles * k_tiles * 512)
    if (
        row_tiles <= 0
        or k_tiles <= 0
        or tuple(scale_mma.shape) != logical_shape
        or scale_mma.stride() != logical_stride
        or scale_mma.dtype not in (torch.uint8, torch.float8_e8m0fnu)
        or scale_mma.device.type != "cuda"
        or scale_mma.data_ptr() % 16
    ):
        raise ValueError("dense MXFP8 MMA scales differ from the canonical layout")
    physical_shape = (1, row_tiles, k_tiles, 32, 4, 4)
    physical_stride = (row_tiles * k_tiles * 512, k_tiles * 512, 512, 16, 4, 1)
    return scale_mma.view(torch.uint8).as_strided(physical_shape, physical_stride)


def dense_scale_view_from_storage(
    scale_mma_base: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    """Build the canonical UE8M0 MMA view over a contiguous physical base."""
    expected = dense_rows_storage_shapes(rows, columns)[2]
    if (
        scale_mma_base.shape != expected
        or scale_mma_base.dtype != torch.uint8
        or scale_mma_base.device.type != "cuda"
        or not scale_mma_base.is_contiguous()
        or scale_mma_base.data_ptr() % 16
    ):
        raise ValueError("dense MXFP8 MMA scale base differs from its storage contract")
    return scale_mma_base.view(torch.float8_e8m0fnu).permute(3, 4, 1, 5, 2, 0)


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)


__all__ = [
    "Mxfp8Config",
    "Mxfp8Query",
    "MXFP8Rows",
    "dense_rows_storage_shapes",
    "dense_rows_from_bases",
    "dense_scale_storage",
    "dense_scale_view_from_storage",
    "plan",
    "query_from_call",
    "quantize_rows",
    "quantize_dense_rows",
    "is_supported",
]
