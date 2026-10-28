# SPDX-License-Identifier: Apache-2.0
"""Correctness fixtures for persistent decoder quantization and TP4 phases."""

import torch


def test_input_quantization():
    from flashinfer.experimental.persistent_decoder import module

    mod = module()
    generator = torch.Generator(device="cuda").manual_seed(937)
    for rows in (1, 2, 8, 17, 32, 128):
        source = torch.randn(
            (rows, 8192), device="cuda", dtype=torch.bfloat16, generator=generator
        )
        source[:, :32] = 0
        source[:, 32:64] = 1e-8
        values = torch.empty_like(source, dtype=torch.uint8)
        scales = torch.empty((rows, 256), dtype=torch.uint8, device="cuda")
        mod.quantize(source, values, scales, False)
        grouped = source.float().view(rows, 256, 32)
        scale = torch.exp2(
            torch.ceil(
                torch.log2(
                    torch.where(
                        grouped.abs().amax(-1) > 0, grouped.abs().amax(-1) / 448, 1.0
                    )
                )
            )
        )
        expected = (grouped / scale[..., None]).to(torch.float8_e4m3fn).view(rows, 8192)
        torch.testing.assert_close(values, expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(
            scales, (torch.log2(scale) + 127).to(torch.uint8), rtol=0, atol=0
        )


def test_routed_activation_quantization_preserves_small_blocks():
    from b12x._lib.intrinsics.reference import pow2_ceil_ue8m0_torch
    from flashinfer.experimental.persistent_decoder import module

    source = torch.zeros((1, 8192), device="cuda", dtype=torch.bfloat16)
    source[0, :32] = torch.linspace(0, 1e-9, 32, device="cuda").bfloat16()
    values = torch.empty_like(source, dtype=torch.uint8)
    scales = torch.empty((1, 256), dtype=torch.uint8, device="cuda")
    module().quantize(source, values, scales, True)
    grouped = source.float().view(1, 256, 32)
    rounded, expected_scales = pow2_ceil_ue8m0_torch(
        grouped.abs().amax(-1) * (1.0 / 448.0)
    )
    inverse = torch.where(
        expected_scales == 0, 0.0, torch.exp2(127.0 - expected_scales.float())
    )
    expected = (
        (grouped * inverse[..., None])
        .to(torch.float8_e4m3fn)
        .view(1, 8192)
        .view(torch.uint8)
    )
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    assert int(torch.count_nonzero(values[0, :32])) > 0
