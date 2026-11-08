"""Prepared BF16 matmul that retains an enclosing matrix's reduction schedule."""

import torch


MAX_WORKSPACE_BYTES = 32 << 20


class ReferenceMatmul:
    """Resolve one cuBLASLt algorithm before capture and use caller-owned storage.

    The logical dimensions select the same tile and split-K schedule before and
    after slicing output columns or independent input rows. Exact row comparisons
    also require preserving the original row's placement within the MMA tile;
    callers retaining arbitrary suffixes must keep alignment padding. Algorithms are resolved
    for the installed CUDA library and are never persisted across library versions.
    """

    def __init__(self, *, rows, columns, inner, logical_rows, logical_columns, device):
        from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

        if not (
            0 < rows <= logical_rows and 0 < columns <= logical_columns and inner > 0
        ):
            raise ValueError(
                "physical BF16 matmul dimensions must fit the logical geometry"
            )
        self.shape = rows, columns, inner
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.module = gen_mm_bf16_cublaslt_module().build_and_load()
            self.handle = torch.cuda.current_blas_handle()
            self.algorithm = torch.empty(64, dtype=torch.uint8, device="cpu")
            count = self.module.mm_bf16_cublaslt_get_algos_geometry(
                logical_rows,
                logical_columns,
                inner,
                MAX_WORKSPACE_BYTES,
                self.handle,
                self.algorithm,
            )
            if count != 1:
                raise RuntimeError(
                    "cuBLASLt returned no algorithm for logical BF16 matmul geometry"
                )
            self.workspace_bytes = self.module.mm_bf16_cublaslt_algo_workspace(
                rows, columns, inner, self.handle, self.algorithm
            )
            if self.workspace_bytes > MAX_WORKSPACE_BYTES:
                raise ValueError(
                    "physical BF16 matmul exceeds its prepared workspace envelope"
                )

    def __call__(self, source, weight, output, workspace):
        rows, columns, inner = self.shape
        if (
            tuple(source.shape) != (rows, inner)
            or tuple(weight.shape) != (columns, inner)
            or tuple(output.shape) != (rows, columns)
            or any(
                t.dtype != torch.bfloat16
                or t.device != self.device
                or not t.is_contiguous()
                for t in (source, weight, output)
            )
        ):
            raise ValueError(
                "logical-geometry BF16 matmul operands differ from preparation"
            )
        if (
            workspace is None
            or workspace.device != self.device
            or workspace.dtype != torch.uint8
            or workspace.ndim != 1
            or not workspace.is_contiguous()
            or workspace.numel() < self.workspace_bytes
        ):
            raise ValueError(
                "logical-geometry BF16 matmul requires sufficient caller-owned byte scratch"
            )
        self.module.mm_bf16_cublaslt_run_with_algo(
            source, weight, None, output, workspace, self.handle, self.algorithm, 0
        )
        return output
