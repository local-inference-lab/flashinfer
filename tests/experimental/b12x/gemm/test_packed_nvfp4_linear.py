"""Prepared packed NVFP4 projections preserve weights and mutable graph inputs."""

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled._a16 import _weight_parts
from b12x.preparation import PreparationSession, PreparedCall


def reference(source, codes, scales, gain):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=source.device,
        dtype=torch.float32,
    )
    indices = (
        torch.stack((codes & 15, codes >> 4), -1).reshape(codes.shape[0], -1).long()
    )
    weight = (table[indices] * scales.float().repeat_interleave(16, 1)).bfloat16()
    return (source.float() @ weight.float().T * gain).bfloat16()


@pytest.mark.parametrize(
    "capacity,n,k",
    [
        (8, 512, 512),
        (16, 512, 512),
        (32, 512, 512),
        (64, 512, 512),
        (2048, 512, 512),
        (64, 34816, 5120),
        (64, 5120, 17408),
    ],
)
def test_packed_projection_mutable_capture(capacity, n, k):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("requires SM120/SM121")
    torch.manual_seed(17)
    codes = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // 16), 0.5, device="cuda", dtype=torch.float8_e4m3fn)
    gain = torch.tensor([0.125], device="cuda")
    weight = blockscaled.pack_nvfp4_a16_linear_weight(codes, scales, gain)
    source = torch.randn(capacity, k, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(capacity, n, device="cuda", dtype=torch.bfloat16)
    query = blockscaled.query_from_call(
        source, weight, out=output, activation_mode="a16"
    )
    plan = blockscaled.plan(query)
    values, packed_scales, packed_gain, _ = _weight_parts(weight)
    request = plan.request(
        name="packed_nvfp4",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(
                source, values, packed_scales, packed_gain, out=output
            ),
            owners=(weight, source, output),
        ),
    )
    with PreparationSession(
        device=source.device, autotune=False, compile_workers=2
    ) as session:
        session.prepare((request,))
        for rows in (0, 1, 8, 16, 32, 64, capacity):
            x, y = source[:rows], output[:rows]

            def run():
                blockscaled.mm(x, weight, out=y, plan=plan)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.cuda.synchronize()
            for turn in range(3):
                source.normal_()
                source[::3].zero_()
                output.fill_(float("nan"))
                allocated = torch.cuda.memory_allocated()
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                if rows:
                    expected = reference(x, codes, scales, gain)
                    actual = y.float().flatten()
                    target = expected.float().flatten()
                    relative = torch.linalg.vector_norm(
                        actual - target
                    ) / torch.linalg.vector_norm(target).clamp_min(1e-20)
                    cosine = torch.nn.functional.cosine_similarity(
                        actual, target, dim=0
                    )
                    assert float(relative) <= 0.005
                    assert float(cosine) >= 0.9999 or not bool(target.any())
                    assert torch.equal(y[::3], torch.zeros_like(y[::3]))
                    del expected, actual, target, relative, cosine
            graph.reset()


def test_tuning_excludes_unqualified_register_geometry():
    from b12x.gemm.blockscaled._packed_nvfp4 import CONTRACT, PackedNVFP4Config
    from b12x.gemm.blockscaled._tuning import BlockscaledQuery
    from b12x.preparation import DeviceIdentity

    device = DeviceIdentity(
        vendor="nvidia", compute_capability=(12, 1), sm_count=48, product_name="GB10"
    )
    query = BlockscaledQuery(
        recipe="nvfp4_packed",
        num_tokens=64,
        in_features=5120,
        padded_in_features=5120,
        out_features=34816,
        activation_mode="a16",
        global_scale_kind="multiplier",
    )
    unsupported = PackedNVFP4Config(mode="a16", tile_m=64, tile_n=512, tile_k=32)
    with pytest.raises(ValueError, match="register count"):
        CONTRACT.validate_config(query, unsupported, device)
    candidates = CONTRACT.eligible_plan(query, device).candidates
    assert len(candidates) > 1
    assert all(config != unsupported for _, config in candidates)
    for rows in (8, 16, 32):
        from dataclasses import replace

        small = replace(query, num_tokens=rows)
        configs = [
            config for _, config in CONTRACT.eligible_plan(small, device).candidates
        ]
        assert any(
            c.tile_n == 512 and c.tile_k == 32 and c.pipeline_stages == 4
            for c in configs
        )
