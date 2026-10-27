"""Prepared MXFP8 quantization preserves values and both scale storage forms."""

from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics.reference import (
    as_grouped_scale_view_mx,
    quantize_grouped_mxfp8_torch,
    swizzle_block_scale,
)
from b12x.preparation import PreparedCall, PreparationSession
from b12x.quantization import mxfp8
from ..conftest import require_b12x


def test_dense_rows_public_storage_and_scale_floor_contract():
    from dataclasses import replace
    from b12x.quantization.mxfp8._tuning import TUNING

    assert mxfp8.dense_rows_storage_shapes(8, 1280) == (
        (8, 1280),
        (1, 8, 40),
        (1, 1, 10, 32, 4, 4),
    )
    with pytest.raises(ValueError, match="divisible by 128"):
        mxfp8.dense_rows_storage_shapes(8, 96)
    base = mxfp8.Mxfp8Query(
        rows=8,
        columns=1280,
        dtype="bfloat16",
        value_order="linear",
        scale_rows_layout="grouped",
        scale_rows_storage="float8_e8m0fnu",
        scale_mma_layout="dense_gemm_swizzled",
        scale_mma_storage="float8_e8m0fnu",
        min_amax=1e-4,
    )
    TUNING.validate_query(base, None)
    TUNING.validate_query(replace(base, release_dependents=True), None)
    with pytest.raises(ValueError, match="min_amax"):
        TUNING.validate_query(replace(base, min_amax=1e-5), None)
    with pytest.raises(TypeError, match="dependent release"):
        TUNING.validate_query(replace(base, release_dependents=1), None)


@pytest.mark.parametrize("rows", (1, 7, 8))
def test_prepared_dense_rows_keep_reference_floor(rows):
    device = require_b12x()
    k = 1280
    source = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
    source[:, :32] = torch.tensor(2e-5, dtype=torch.bfloat16, device=device)
    source[:, 64:96] = torch.linspace(-0.125, 0.125, 32, device=device).bfloat16()
    shapes = mxfp8.dense_rows_storage_shapes(rows, k)
    bases = (
        torch.empty(shapes[0], dtype=torch.float8_e4m3fn, device=device),
        torch.empty(shapes[1], dtype=torch.uint8, device=device),
        torch.empty(shapes[2], dtype=torch.uint8, device=device),
    )
    output = mxfp8.dense_rows_from_bases(*bases)
    physical = mxfp8.dense_scale_storage(output.scale_mma)
    assert physical.shape == bases[2].shape
    assert physical.is_contiguous() and physical.data_ptr() == bases[2].data_ptr()
    rebuilt = mxfp8.dense_rows_from_bases(bases[0], bases[1], physical)
    assert rebuilt.scale_mma.data_ptr() == output.scale_mma.data_ptr()
    with pytest.raises(ValueError, match="canonical layout"):
        mxfp8.dense_scale_storage(output.scale_mma.transpose(0, 1))
    query = mxfp8.query_from_call(
        source,
        output.values,
        output.scale_rows,
        output.scale_mma,
        min_amax=1e-4,
    )
    declaration = mxfp8.plan(query)
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                declaration.request(
                    name="floored-rows",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            source, output.values, output.scale_rows, output.scale_mma
                        ),
                        owners=bases,
                    ),
                ),
            )
        )
        mxfp8.quantize_rows(
            source,
            output.values,
            output.scale_rows,
            output.scale_mma,
            plan=declaration,
        )
    blocks = source.float().view(rows, k // 32, 32)
    ratio = blocks.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = ratio.view(torch.int32)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (exponent << 23).view(torch.float32)
    expected = (blocks / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        output.values.view(torch.uint8),
        expected.view(rows, k).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        output.scale_rows.view(torch.uint8)[0], exponent.to(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("scale_storage", (torch.float8_e8m0fnu, torch.uint8))
def test_quantize_rows_matches_oracle_and_dequantizes(dtype, scale_storage):
    require_b12x()
    torch.manual_seed(20260715)
    rows, columns = 5, 256
    source = (torch.randn((rows, columns), device="cuda", dtype=dtype) / 4).contiguous()
    expected_values, expected_mma = quantize_grouped_mxfp8_torch(
        source.unsqueeze(0),
        torch.tensor([rows], dtype=torch.int32, device=source.device),
    )
    padded_rows, padded_groups = 128, columns // 32
    expected_rows = (
        expected_mma.view(torch.uint8)
        .permute(5, 2, 1, 0, 4, 3)
        .contiguous()
        .view(
            1,
            padded_rows,
            padded_groups,
        )[0, :rows]
        .contiguous()
    )
    padded = torch.full(
        (1, padded_rows, padded_groups), 127, dtype=torch.uint8, device=source.device
    )
    padded[:, :rows].copy_(expected_rows)
    expected_mma = as_grouped_scale_view_mx(swizzle_block_scale(padded), rows, columns)
    values = torch.empty(
        (rows, columns), device=source.device, dtype=torch.float8_e4m3fn
    )
    scale_rows = torch.empty(
        (rows, columns // 32), device=source.device, dtype=scale_storage
    )
    storage = torch.full(
        (1, padded_rows, padded_groups), 127, device=source.device, dtype=torch.uint8
    )
    scale_mma = as_grouped_scale_view_mx(storage, rows, columns)
    if scale_storage == torch.uint8:
        scale_mma = scale_mma.view(torch.uint8)

    query = mxfp8.query_from_call(source, values, scale_rows, scale_mma)
    declaration = mxfp8.plan(query)
    request = declaration.request(
        name="rows",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(source, values, scale_rows, scale_mma)
        ),
    )
    with PreparationSession(device=source.device) as session:
        session.prepare((request,))
        mxfp8.quantize_rows(source, values, scale_rows, scale_mma, plan=declaration)
        torch.testing.assert_close(
            values.view(torch.uint8), expected_values[:, :, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale_rows.view(torch.uint8), expected_rows, rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale_mma.view(torch.uint8), expected_mma.view(torch.uint8), rtol=0, atol=0
        )
        dequantized = values.float() * torch.exp2(
            scale_rows.view(torch.uint8).float() - 127
        ).repeat_interleave(32, dim=1)
        assert (dequantized - source.float()).abs().max().item() < 0.05
