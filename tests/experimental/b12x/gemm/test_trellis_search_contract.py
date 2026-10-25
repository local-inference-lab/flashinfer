"""Host launch legality and heuristic membership for dense Trellis."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.gemm.trellis_linear._tuning import (
    TUNING,
    TrellisConfig,
    TrellisQuery,
    _shared_memory_bytes,
)
from b12x.moe._shared.kernels.w4a16.kernel.gemm import W4A16GemmKernel
from b12x.preparation import DeviceIdentity


DEVICE = DeviceIdentity("nvidia", (12, 0), 120, "SM120")


def query(layout="native", **changes):
    paired = layout != "native"
    axis = layout[-1] if paired else None
    return replace(
        TrellisQuery(
            max_rows=128,
            in_features=256 if axis == "k" else 4096,
            out_features=256 if axis == "n" else 4096,
            input_dtype="float16",
            compute_dtype="float16",
            output_dtype="float16",
            codebook="mcg",
            bits=3,
            weight_layout=layout,
            pair_kind=layout[:3].upper() if paired else None,
            rate_axis=axis,
        ),
        **changes,
    )


def kernel(q, config):
    return W4A16GemmKernel(
        size_m=q.max_rows,
        size_n=q.out_features,
        size_k=q.in_features,
        num_experts=1,
        top_k=1,
        mul_topk_weights=False,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        moe_block_size=config.block_rows,
        max_m_blocks=(q.max_rows + config.block_rows - 1) // config.block_rows,
        element_dtype="fp16",
        weight_layout="trellis_t256",
        scale_format="e4m3_k32",
        w13_layout="packed",
        trellis_bits=q.bits,
        trellis_codebook=q.codebook,
        trellis_pair_kind=q.pair_kind,
        trellis_rate_axis=q.rate_axis,
        dense_route_fast_path=True,
        schedule_whole_tiles=True,
    )


def test_three_bit_large_tile_matches_kernel_allocation():
    q = query(autotune_effort="audit")
    config = TrellisConfig(backend="cutedsl", block_rows=64, tile_k=128, tile_n=128)
    assert _shared_memory_bytes(q, config) == 93_184
    assert kernel(q, config).shared_words * 4 == 93_184
    assert config in {config for _, config in TUNING.choices(q, DEVICE)}


@pytest.mark.parametrize("layout", ("native", "p24_k", "p33_k", "p24_n", "p33_n"))
def test_search_memory_accounting_matches_kernel(layout):
    for bits in (2, 3, 4, 5, 6) if layout == "native" else (3,):
        q = query(layout, bits=bits)
        for _, config in TUNING.choices(q, DEVICE):
            assert _shared_memory_bytes(q, config) == kernel(q, config).shared_words * 4


@pytest.mark.parametrize("layout", ("native", "p24_k", "p33_k", "p24_n", "p33_n"))
@pytest.mark.parametrize("rows", (1, 128, 7681, 15361))
def test_defaults_belong_to_every_effort(layout, rows):
    q = query(layout, max_rows=rows)
    default = TUNING.default_config(q, DEVICE)
    for effort in ("off", "default", "audit"):
        active = replace(q, autotune_effort=effort)
        configured = TUNING.configure(active, device=DEVICE)
        configured.space.validate(TUNING.config_payload(default))
        choices = {config for _, config in TUNING.choices(active, DEVICE)}
        assert default in choices
        if effort == "off":
            assert choices == {default}
    if layout.endswith("_n"):
        assert default.tile_n == 256


def test_effort_is_captured_at_query_construction(monkeypatch):
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "off")
    q = query()
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "max")
    assert q.autotune_effort == "off"
    assert TUNING.parameter_space(q, DEVICE).autotune_effort == "off"


def test_effort_does_not_change_plan_compile_jobs():
    import torch

    from b12x.gemm.trellis_linear._preparation import plan
    from b12x.moe._shared.kernels.w4a16.prepare import PreparedTrellis256DenseWeight

    low = query(autotune_effort="off")
    maximum = replace(low, autotune_effort="audit")
    empty = torch.empty(0)
    weight = PreparedTrellis256DenseWeight(
        trellis=empty,
        suh=empty,
        svh=empty,
        scale=empty,
        global_scale=empty,
        workspace=empty,
        in_features=low.in_features,
        out_features=low.out_features,
        params_dtype=torch.float16,
        trellis_bits=low.bits,
        trellis_codebook=low.codebook,
    )
    low_plan = plan(low, weight=weight)
    maximum_plan = plan(maximum, weight=weight)
    config = TUNING.default_config(low, DEVICE)
    device = SimpleNamespace(ordinal=0, identity=DEVICE)
    assert TUNING.encode_query(low) != TUNING.encode_query(maximum)
    jobs = low_plan._compile_jobs(config, device)
    assert jobs == maximum_plan._compile_jobs(config, device)
    assert jobs[0].args[0]["autotune_effort"] == "default"
