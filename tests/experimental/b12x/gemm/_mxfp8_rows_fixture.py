"""Test storage for the dense MXFP8 layout available in public FlashInfer."""

from b12x.gemm._shared.wo_mxfp8 import (
    empty_mxfp8_rows_bases,
    mxfp8_rows_from_bases,
)


def rows_and_bases(rows, columns, *, device):
    bases = empty_mxfp8_rows_bases(rows, columns, num_groups=1, device=device)
    return mxfp8_rows_from_bases(*bases, rows, columns, num_groups=1), bases
