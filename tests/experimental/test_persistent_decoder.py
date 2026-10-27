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
        mod.quantize(source, values, scales)
        grouped = source.float().view(rows, 256, 32)
        scale = torch.exp2(
            torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448))
        )
        expected = (grouped / scale[..., None]).to(torch.float8_e4m3fn).view(rows, 8192)
        torch.testing.assert_close(values, expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(
            scales, (torch.log2(scale) + 127).to(torch.uint8), rtol=0, atol=0
        )
