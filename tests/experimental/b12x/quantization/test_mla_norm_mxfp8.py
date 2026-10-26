"""Split RMS normalization declares its arithmetic and valid launch alternatives."""

from dataclasses import replace
import pytest
from b12x.quantization.mla_norm_mxfp8._tuning import Query, Config, TUNING


@pytest.mark.parametrize("pdl", [False, True])
def test_launch_alternatives_are_normal_candidates(pdl):
    query = Query(rows=8, norm_weight_dtype="bfloat16", affine_prefetch_ready=True)
    config = Config(pdl)
    TUNING.validate_query(query, None)
    TUNING.validate_config(query, config, None)
    TUNING.parameter_space(query, None).validate(TUNING.encode_config(config))


@pytest.mark.parametrize(
    "change",
    [
        {"rows": 0},
        {"query_features": 1024},
        {"kv_features": 256},
        {"epsilon": 1e-6},
        {"min_amax": 0.0},
        {"norm_weight_dtype": "float16"},
    ],
)
def test_unsupported_arithmetic_is_rejected(change):
    with pytest.raises(ValueError):
        TUNING.validate_query(
            replace(Query(rows=8, norm_weight_dtype="float32"), **change), None
        )


def test_programmatic_launch_setting_has_a_boolean_abi():
    with pytest.raises(TypeError):
        TUNING.validate_config(
            Query(rows=8, norm_weight_dtype="bfloat16"), Config(1), None
        )


@pytest.mark.parametrize(
    "rows,dtype,pdl",
    [
        (8, "bfloat16", False),
        (8, "bfloat16", True),
        (7, "bfloat16", True),
        (1, "bfloat16", False),
        (32, "bfloat16", True),
        (64, "bfloat16", True),
        (4096, "bfloat16", True),
        (8, "float32", True),
    ],
)
def test_prepared_norm_preserves_bf16_encoding_seam_and_replays(rows, dtype, pdl):
    import torch
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.quantization import mla_norm_mxfp8
    from b12x._lib.runtime_control import kernel_resolution_guard
    from ..conftest import require_b12x

    device = require_b12x()
    source = torch.randn((rows, 1792), device=device).bfloat16()
    qw = torch.linspace(0.5, 2.625, 1280, device=device).to(getattr(torch, dtype))
    kw = torch.linspace(0.25, 1.5, 512, device=device).to(getattr(torch, dtype))
    query = torch.empty((rows, 1280), dtype=torch.bfloat16, device=device)
    kv = torch.empty((rows, 512), dtype=torch.bfloat16, device=device)
    values = torch.empty_like(query, dtype=torch.float8_e4m3fn)
    scales = torch.full(
        (((rows + 127) // 128) * 5120,), 127, dtype=torch.uint8, device=device
    )
    scale_rows = torch.empty((rows, 40), dtype=torch.uint8, device=device)
    declaration = mla_norm_mxfp8.plan(
        Query(rows=rows, norm_weight_dtype=dtype, affine_prefetch_ready=True),
        override=Config(pdl),
    )
    request = declaration.request(
        name="split-normalization",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, qw, kw, query, kv, values, scales, scale_rows)
        ),
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))

        def forward(source, qw, kw, query, kv, values, scales, scale_rows):
            mla_norm_mxfp8.run(
                source, qw, kw, query, kv, values, scales, scale_rows, plan=declaration
            )

        compiled = torch.compile(forward, fullgraph=True)
        compiled(source, qw, kw, query, kv, values, scales, scale_rows)
        torch.cuda.synchronize()
        for part, weight, output in (
            (source[:, :1280], qw, query),
            (source[:, 1280:], kw, kv),
        ):
            reference = part.double() * torch.rsqrt(
                part.double().square().mean(-1, keepdim=True) + 1e-20
            )
            reference = (reference * weight.double()).bfloat16()
            # One BF16 spacing permits a rounding-boundary crossing caused by
            # FP32 sum/rsqrt error; the reference statistics use FP64.
            torch.testing.assert_close(output, reference, rtol=2**-7, atol=0)
        blocks = query.float().view(rows, 40, 32)
        ratio = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
        bits = ratio.view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
        factors = (exponent << 23).view(torch.float32)
        encoded = (
            (blocks / factors[:, :, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .view(rows, 1280)
        )
        assert torch.equal(values.view(torch.uint8), encoded.view(torch.uint8))
        assert torch.equal(scale_rows, exponent.to(torch.uint8))
        del reference, blocks, ratio, bits, exponent, factors, encoded
        expected = tuple(
            x.clone() for x in (query, kv, values.view(torch.uint8), scales, scale_rows)
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(source, qw, kw, query, kv, values, scales, scale_rows)
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        with kernel_resolution_guard("allocation-stable normalization replay"):
            for _ in range(16):
                graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before
        for iteration in range(4):
            source.normal_(std=0.125)
            source[0].zero_()
            # Eager and graph calls share exact selected arithmetic and operands.
            forward(source, qw, kw, query, kv, values, scales, scale_rows)
            for target, value in zip(
                expected, (query, kv, values.view(torch.uint8), scales, scale_rows)
            ):
                target.copy_(value)
            query.fill_(float("nan"))
            kv.fill_(float("nan"))
            values.view(torch.uint8).fill_(255)
            scale_rows.fill_(255)
            with kernel_resolution_guard("prepared split-normalization replay"):
                graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(
                (query, kv, values.view(torch.uint8), scales, scale_rows), expected
            ):
                assert torch.equal(actual, reference)
            assert (
                torch.count_nonzero(query[0]) == 0 and torch.count_nonzero(kv[0]) == 0
            )


def test_dynamic_affine_inputs_cannot_prefetch_before_their_producer():
    q = Query(rows=8, norm_weight_dtype="bfloat16")
    TUNING.validate_config(q, Config(False), None)
    with pytest.raises(ValueError, match="readiness"):
        TUNING.validate_config(q, Config(True), None)
    assert [
        dict(value) for value in TUNING.parameter_space(q, None).configurations()
    ] == [{"programmatic_serialization": False}]
