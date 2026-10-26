"""The CUDA candidate is described by the encoded dense GEMM contract."""

from dataclasses import replace

import pytest

from b12x.gemm._tuning import (
    DenseGemmQuery,
    DenseGemmConfig,
    TUNING,
    streaming_eligible,
)
from b12x.gemm.mxfp8_streaming._preparation import source_spec
from b12x.preparation import DeviceIdentity
from b12x._lib.compile_pool import describe_compilation, CompileJob


def query():
    return DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=8,
        in_features=5120,
        out_features=1792,
        output_mode="provided",
        alpha_mode="unit",
        scale_domain="bounded_fp32_products",
        rhs_prefetch_ready=True,
    )


def config(pdl=True, evict=True):
    return DenseGemmConfig(
        backend="cuda_streaming",
        tile_m=16,
        tile_n=8,
        tile_k=128,
        load_path="streaming",
        swap_ab=False,
        split_k_slices=1,
        large_m_unroll=False,
        target_occupancy=None,
        programmatic_serialization=pdl,
        evict_first=evict,
    )


@pytest.mark.parametrize("pdl,evict", [(False, False), (True, False), (True, True)])
def test_supported_launch_alternatives_are_normal_candidates(pdl, evict):
    q = query()
    c = config(pdl, evict)
    device = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
    TUNING.validate_config(q, c, device)
    TUNING.parameter_space(q, device).validate(TUNING.encode_config(c))


@pytest.mark.parametrize(
    "changes",
    [
        {"max_rows": 9},
        {"batch": 2},
        {"alpha_mode": "tensor"},
        {"output_dtype": "float32"},
        {"in_features": 5136},
        {"recipe": "mxfp4"},
        {"output_mode": "functional"},
        {"sfb_k_replicated": True},
        {"scale_domain": "ue8m0"},
        {"in_features": 2**31},
        {"out_features": 2**31},
        {"out_features": 2**27, "in_features": 4096},
    ],
)
def test_streaming_does_not_expand_the_mathematical_or_layout_domain(changes):
    assert not streaming_eligible(replace(query(), **changes))
    with pytest.raises(ValueError, match="contract"):
        TUNING.validate_config(replace(query(), **changes), config(), None)


def test_cache_hint_requires_its_prefetch_launch():
    with pytest.raises(ValueError, match="prefetch"):
        TUNING.validate_config(query(), config(False, True), None)


def test_sm121_does_not_inherit_sm120_kernel_qualification():
    device = DeviceIdentity("nvidia", (12, 1), 48, "SM121")
    with pytest.raises(ValueError, match="contract"):
        TUNING.validate_config(query(), config(), device)


def test_actual_source_describes_a_flat_cuda_program_without_compiling(
    tmp_path, monkeypatch
):
    from b12x._lib import cuda_program

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/nvcc").write_bytes(b"nvcc")
    (tmp_path / "include").mkdir()
    (tmp_path / "include/cuda.h").write_bytes(b"header")
    spec = source_spec(config(), cuda_home=tmp_path)
    monkeypatch.setattr(
        cuda_program.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("metadata compiled CUDA"),
    )
    description = describe_compilation(
        CompileJob.create(
            "b12x.gemm.mxfp8_streaming._preparation:compile_streaming", spec
        )
    )
    assert description.programs == (spec.program,)
    assert spec.kernel.argument_types == ("ptr",) * 5 + ("i32",) * 3
    assert spec.kernel.dynamic_smem_bytes == 17408
    assert spec.kernel.programmatic_serialization
    source = __import__("pathlib").Path(spec.source).read_text()
    assert "torch/extension" not in source and "getenv" not in source
    assert "SPDX-License-Identifier: Apache-2.0" in source and "5d25cff77495" in source


@pytest.mark.parametrize(
    "rows,columns,inner", [(1, 1, 128), (7, 9, 256), (8, 1792, 5120)]
)
def test_physical_layout_matches_logical_tails(rows, columns, inner):
    import torch
    from types import SimpleNamespace
    from b12x.gemm.mxfp8_streaming._preparation import PreparedStreamingDense

    device = torch.device("cuda", 0)
    calls = []
    q = replace(query(), max_rows=rows, out_features=columns, in_features=inner)

    def value(shape, dtype, *, address=4096, stride=None):
        return SimpleNamespace(
            device=device,
            shape=shape,
            dtype=dtype,
            is_contiguous=lambda: True,
            data_ptr=lambda: address,
            stride=lambda: stride,
        )

    a = value((rows, inner, 1), torch.float8_e4m3fn)
    b = value((columns, inner, 1), torch.float8_e4m3fn)
    out = value((rows, columns, 1), torch.bfloat16)

    def scales(count):
        rb, kb = (count + 127) // 128, inner // 128
        return value(
            (32, 4, rb, 4, kb, 1),
            torch.uint8,
            stride=(16, 4, kb * 512, 1, 512, rb * kb * 512),
        )

    sa, sb = scales(rows), scales(columns)
    prepared = PreparedStreamingDense(q, device, lambda *a, **kw: calls.append((a, kw)))
    assert prepared.run(a, sa, b, sb, None, out, None, torch.bfloat16, 17) is out
    assert calls[0][1] == {"grid": ((columns + 7) // 8, 1, 1), "stream": 17}
    assert calls[0][0][-3:] == (rows, columns, inner)
    b.shape = (columns + 128, inner, 1)
    with pytest.raises(ValueError, match="layout"):
        prepared.run(a, sa, b, sb, None, out, None, torch.bfloat16, 17)
    b.shape = (columns, inner, 1)
    a.data_ptr = lambda: 4097
    with pytest.raises(ValueError, match="layout"):
        prepared.run(a, sa, b, sb, None, out, None, torch.bfloat16, 17)
    assert len(calls) == 1


def test_mutable_rhs_keeps_stream_dependencies_even_with_bounded_scales():
    q = replace(query(), rhs_prefetch_ready=False)
    TUNING.validate_config(q, config(False, False), None)
    with pytest.raises(ValueError, match="RHS readiness"):
        TUNING.validate_config(q, config(True, False), None)
    with pytest.raises(ValueError, match="RHS readiness"):
        TUNING.validate_config(q, config(True, True), None)
    candidates = [
        choice
        for choice in TUNING.parameter_space(q, None).configurations()
        if choice["backend"] == "cuda_streaming"
    ]
    assert len(candidates) == 1
    assert candidates[0]["programmatic_serialization"] is False
    assert candidates[0]["evict_first"] is False
