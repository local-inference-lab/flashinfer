"""Split Q/KV RMSNorm preserves the BF16 seam and canonical MXFP8 rows."""

from dataclasses import replace

import pytest

from b12x.quantization.mla_norm_mxfp8._tuning import Config, Query, TUNING


def query(
    rows=8, *, affine=False, release=False, q_dtype="bfloat16", kv_dtype="bfloat16"
):
    return Query(
        rows=rows,
        query_weight_dtype=q_dtype,
        kv_weight_dtype=kv_dtype,
        affine_prefetch_ready=affine,
        release_dependents=release,
    )


def test_incoming_prefetch_and_outgoing_release_are_independent():
    outgoing = query(release=True)
    TUNING.validate_query(outgoing, None)
    TUNING.validate_config(outgoing, Config(False), None)
    with pytest.raises(ValueError, match="readiness"):
        TUNING.validate_config(outgoing, Config(True), None)
    assert [
        dict(value) for value in TUNING.parameter_space(outgoing, None).configurations()
    ] == [{"programmatic_serialization": False}]
    incoming = replace(outgoing, affine_prefetch_ready=True)
    for pdl in (False, True):
        TUNING.validate_config(incoming, Config(pdl), None)


@pytest.mark.parametrize(
    "change",
    (
        {"rows": 0},
        {"query_features": 1024},
        {"kv_features": 256},
        {"epsilon": 1e-6},
        {"min_amax": 0.0},
        {"query_weight_dtype": "float16"},
        {"kv_weight_dtype": "float16"},
    ),
)
def test_unsupported_arithmetic_is_rejected(change):
    with pytest.raises(ValueError):
        TUNING.validate_query(replace(query(), **change), None)


@pytest.mark.parametrize(
    "change", ({"affine_prefetch_ready": 1}, {"release_dependents": 1})
)
def test_dependency_promises_are_boolean(change):
    with pytest.raises(TypeError):
        TUNING.validate_query(replace(query(), **change), None)


def test_launch_policy_is_boolean():
    with pytest.raises(TypeError):
        TUNING.validate_config(query(affine=True), Config(1), None)


def reference_quantize(source):
    import torch

    blocks = source.float().view(source.shape[0], -1, 32)
    ratio = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    scale = (exponent << 23).view(torch.float32)
    values = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.view_as(source), exponent.to(torch.uint8)


def check_outputs(
    source, qw, kw, query_bf16, kv_bf16, values_base, scale_rows_base, scale_mma_base
):
    import torch

    for begin, end, weight, actual in (
        (0, 1280, qw, query_bf16),
        (1280, 1792, kw, kv_bf16),
    ):
        x = source[:, begin:end].double()
        expected = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-20) * weight.double()
        ).bfloat16()
        # The independent FP64 reduction can cross one BF16 rounding boundary
        # relative to the declared FP32 reduction.
        torch.testing.assert_close(actual, expected, rtol=2**-7, atol=0)
    expected_values, expected_scales = reference_quantize(query_bf16)
    assert torch.equal(values_base.view(torch.uint8), expected_values.view(torch.uint8))
    assert torch.equal(scale_rows_base[0], expected_scales)
    rows = source.shape[0]
    row = torch.arange(rows, device=source.device)[:, None]
    group = torch.arange(40, device=source.device)[None, :]
    offsets = (
        (row // 128 * 10 + group // 4) * 512
        + row % 32 * 16
        + row // 32 % 4 * 4
        + group % 4
    )
    assert torch.equal(scale_mma_base.view(-1)[offsets], expected_scales)


@pytest.mark.parametrize(
    "rows,q_dtype,kv_dtype,incoming,release",
    (
        (1, "bfloat16", "bfloat16", False, True),
        (7, "bfloat16", "bfloat16", True, True),
        (8, "float32", "bfloat16", False, True),
        (32, "bfloat16", "float32", True, False),
        (4096, "bfloat16", "bfloat16", False, False),
    ),
)
def test_compiled_graph_replay_preserves_normalization_and_encoding(
    rows, q_dtype, kv_dtype, incoming, release
):
    import torch
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.quantization import mla_norm_mxfp8, mxfp8
    from b12x._lib.runtime_control import kernel_resolution_guard
    from ..conftest import require_b12x

    device = require_b12x()
    source = (torch.randn((rows, 1792), device=device) * 0.125).bfloat16()
    qw = torch.linspace(0.5, 2.625, 1280, device=device).to(getattr(torch, q_dtype))
    kw = torch.linspace(0.25, 1.5, 512, device=device).to(getattr(torch, kv_dtype))
    query_bf16 = torch.empty((rows, 1280), dtype=torch.bfloat16, device=device)
    kv_bf16 = torch.empty((rows, 512), dtype=torch.bfloat16, device=device)
    value_shape, row_shape, mma_shape = mxfp8.dense_rows_storage_shapes(rows, 1280)
    values_base = torch.empty(value_shape, dtype=torch.float8_e4m3fn, device=device)
    scale_rows_base = torch.empty(row_shape, dtype=torch.uint8, device=device)
    scale_mma_base = torch.empty(mma_shape, dtype=torch.uint8, device=device)
    encoded = mxfp8.dense_rows_from_bases(values_base, scale_rows_base, scale_mma_base)
    assert encoded.values.data_ptr() == values_base.data_ptr()
    assert encoded.scale_mma.data_ptr() == scale_mma_base.data_ptr()
    operands = (
        source,
        qw,
        kw,
        query_bf16,
        kv_bf16,
        values_base,
        scale_rows_base,
        scale_mma_base,
    )
    declaration = mla_norm_mxfp8.plan(
        query(
            rows, affine=incoming, release=release, q_dtype=q_dtype, kv_dtype=kv_dtype
        ),
        override=Config(incoming),
    )
    request = declaration.request(
        name="split-normalization",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(*operands), owners=(encoded,)
        ),
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))

        def forward(*values):
            mla_norm_mxfp8.run(*values, plan=declaration)

        compiled = torch.compile(forward, fullgraph=True)
        compiled(*operands)
        torch.cuda.synchronize()
        check_outputs(*operands)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled(*operands)
        session.freeze()
        try:
            for iteration in range(3):
                source.normal_(std=0.125 * (iteration + 1))
                source[0].zero_()
                qw.mul_(1.0 + iteration * 0.125)
                forward(*operands)
                expected = tuple(
                    tensor.clone()
                    for tensor in (
                        query_bf16,
                        kv_bf16,
                        values_base.view(torch.uint8),
                        scale_rows_base,
                        scale_mma_base,
                    )
                )
                query_bf16.fill_(float("nan"))
                kv_bf16.fill_(float("nan"))
                values_base.view(torch.uint8).fill_(255)
                scale_rows_base.fill_(255)
                scale_mma_base.fill_(255)
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                with kernel_resolution_guard("compiled split-normalization replay"):
                    graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key]
                for actual, reference in zip(
                    (
                        query_bf16,
                        kv_bf16,
                        values_base.view(torch.uint8),
                        scale_rows_base,
                    ),
                    expected[:4],
                    strict=True,
                ):
                    assert torch.equal(actual, reference)
                check_outputs(*operands)
                assert torch.count_nonzero(query_bf16[0]) == 0
                assert torch.count_nonzero(kv_bf16[0]) == 0
        finally:
            graph.reset()


def test_released_query_encoding_feeds_public_prepared_gemm_bases():
    import torch
    from b12x import gemm
    from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.quantization import mla_norm_mxfp8, mxfp8
    from b12x._lib.runtime_control import kernel_resolution_guard
    from ..conftest import require_b12x

    device = require_b12x()
    rows, width, columns = 8, 1280, 64
    source = (torch.randn((rows, width + 512), device=device) * 0.125).bfloat16()
    qw = torch.linspace(0.5, 1.5, width, device=device, dtype=torch.bfloat16)
    kw = torch.linspace(0.25, 1.25, 512, device=device, dtype=torch.bfloat16)
    query_bf16 = torch.empty((rows, width), dtype=torch.bfloat16, device=device)
    kv_bf16 = torch.empty((rows, 512), dtype=torch.bfloat16, device=device)
    lhs_bases = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(
            mxfp8.dense_rows_storage_shapes(rows, width),
            (torch.float8_e4m3fn, torch.uint8, torch.uint8),
            strict=True,
        )
    )
    rhs_bases = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(
            mxfp8.dense_rows_storage_shapes(columns, width),
            (torch.float8_e4m3fn, torch.uint8, torch.uint8),
            strict=True,
        )
    )
    lhs = mxfp8.dense_rows_from_bases(*lhs_bases)
    rhs = mxfp8.dense_rows_from_bases(*rhs_bases)
    rhs_bases[0].copy_(
        (torch.randn((columns, width), device=device) * 0.125).to(torch.float8_e4m3fn)
    )
    rhs_bases[1].fill_(127)
    rhs_bases[2].fill_(127)
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
    workspace = torch.empty((0,), dtype=torch.uint8, device=device)
    norm = mla_norm_mxfp8.plan(
        query(rows, affine=True, release=True), override=Config(True)
    )
    dense_query = gemm.DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=rows,
        in_features=width,
        out_features=columns,
        output_mode="provided",
        alpha_mode="unit",
        workspace_form="provided",
        workspace_nbytes=0,
    )
    dense_config = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=16,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=1,
        large_m_unroll=True,
        target_occupancy=None,
    )
    dense = gemm.plan(dense_query, override=dense_config)
    norm_args = (source, qw, kw, query_bf16, kv_bf16, *lhs_bases)

    def norm_request(state):
        return PreparedCall(run=lambda: state.run(*norm_args), owners=(lhs, *lhs_bases))

    def dense_request(state):
        return PreparedCall(
            run=lambda: state.run(
                lhs.values.view(rows, width, 1),
                lhs.scale_mma,
                rhs.values.view(columns, width, 1),
                rhs.scale_mma,
                None,
                output.view(rows, columns, 1),
                workspace.view(torch.float32),
                torch.bfloat16,
                None,
            ),
            owners=(lhs, rhs, output, workspace),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (
                norm.request(name="query-producer", prepare_call=norm_request),
                dense.request(name="query-consumer", prepare_call=dense_request),
            )
        )

        def forward():
            mla_norm_mxfp8.run(*norm_args, plan=norm)
            return gemm.mxfp8_mm_bases(
                lhs_bases[0],
                lhs_bases[2],
                rhs_bases[0],
                rhs_bases[2],
                output,
                workspace,
                plan=dense,
            )

        compiled = torch.compile(forward, fullgraph=True)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for iteration in range(4):
                source.normal_(std=0.125 * (iteration + 1))
                source[0].zero_()
                query_bf16.fill_(float("nan"))
                kv_bf16.fill_(float("nan"))
                lhs_bases[0].view(torch.uint8).fill_(255)
                lhs_bases[1].fill_(255)
                lhs_bases[2].fill_(255)
                output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                with kernel_resolution_guard("released query graph replay"):
                    graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key]
                check_outputs(*norm_args)
                a = dequantize_mxfp8_rows_torch(lhs.values, lhs.scale_rows).double()
                b = dequantize_mxfp8_rows_torch(rhs.values, rhs.scale_rows).double()
                exact = a @ b.T
                u = (width + 1) * 2.0**-24
                error = (u / (1.0 - u)) * (a.abs() @ b.abs().T)
                below = torch.nextafter(
                    output, torch.full_like(output, -float("inf"))
                ).double()
                above = torch.nextafter(
                    output, torch.full_like(output, float("inf"))
                ).double()
                error += 0.5 * torch.maximum(
                    output.double() - below, above - output.double()
                )
                assert torch.isfinite(output).all()
                assert bool(((output.double() - exact).abs() <= error).all())
                assert torch.count_nonzero(output[0]) == 0
        finally:
            graph.reset()
