"""MXFP8 FP32 reduction rounds once after all K contributions."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x import gemm
from b12x.preparation import PreparationSession, PreparedCall
from b12x.quantization import mxfp8
from flashinfer.experimental.b12x.gemm._tuning import TUNING
from ..conftest import require_b12x
from ._mxfp8_rows_fixture import rows_and_bases


def test_fp32_split_k_pdl_is_a_legal_prepared_mxfp8_candidate():
    device = SimpleNamespace(sm_count=188)
    query = gemm.DenseGemmQuery(
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
        workspace_form="provided",
        workspace_nbytes=4 * 8 * 1792 * 4,
        rhs_prefetch_ready=True,
        allow_bf16_atomic_reduction=False,
    )
    configs = (
        config
        for _, config in TUNING.eligible_plan(query, device).candidates
        if config.programmatic_serialization
    )
    assert {config.split_k_slices for config in configs} == {1, 2, 4}
    for changed, allowed in (
        (replace(query, allow_bf16_atomic_reduction=True), {1}),
        (replace(query, rhs_prefetch_ready=False), set()),
    ):
        configs = (
            config
            for _, config in TUNING.eligible_plan(changed, device).candidates
            if config.programmatic_serialization
        )
        assert {config.split_k_slices for config in configs} == allowed


def test_mxfp8_cancellation_and_mutable_native_replay():
    device = require_b12x()
    m, n, k = 1, 128, 5120
    positions = (0, 1280, 2560, 3840)
    terms = (1.0, 1 / 512, 1 / 512, 1 / 512)
    exact = torch.tensor(sum(terms), dtype=torch.float32)
    assert exact.bfloat16().item() == 1.0078125
    atomic_order = torch.tensor(0.0, dtype=torch.bfloat16)
    for term in terms:
        atomic_order = (atomic_order.float() + term).bfloat16()
    assert atomic_order.item() == 1.0

    a, (a_values, a_rows, a_scales) = rows_and_bases(m, k, device=device)
    a_values.zero_()
    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device)
    weight_scales = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device)
    for index, position in enumerate(positions):
        a_values[:, position] = 1
        weight[:, position] = 1
        if index:
            weight_scales[:, position // 32] = 118
    packed = gemm.blockscaled.pack_weight(weight, weight_scales)
    b_values = packed.weight.values
    b_scales = packed.weight.scale_mma.view(torch.uint8)
    outputs = tuple(
        torch.empty((m, n), dtype=torch.bfloat16, device=device) for _ in range(3)
    )
    workspace_nbytes = 4 * m * n * 4
    workspace = torch.empty(
        (workspace_nbytes // 4,), dtype=torch.float32, device=device
    )
    query = gemm.DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=m,
        in_features=k,
        out_features=n,
        output_mode="provided",
        alpha_mode="unit",
        workspace_form="provided",
        workspace_nbytes=workspace_nbytes,
        allow_bf16_atomic_reduction=False,
    )
    base = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=16,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=1,
        large_m_unroll=False,
        target_occupancy=None,
    )
    plans = tuple(
        gemm.plan(query, override=replace(base, split_k_slices=slices))
        for slices in (1, 2, 4)
    )

    def request(plan, output, name):
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    a.values.view(m, k, 1),
                    a.scale_mma,
                    b_values.view(n, k, 1),
                    packed.weight.scale_mma,
                    None,
                    output.view(m, n, 1),
                    workspace,
                    torch.bfloat16,
                    None,
                ),
                owners=(a_values, a_scales, packed, output, workspace),
            ),
        )

    def oracle():
        result = torch.zeros((n,), dtype=torch.float64, device=device)
        for index, position in enumerate(positions):
            result += (
                a_values[0, position].float().double()
                * b_values[:, position].float().double()
                * terms[index]
            )
        return result.bfloat16().view(m, n)

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            tuple(
                request(plan, output, f"fp32-{i}")
                for i, (plan, output) in enumerate(zip(plans, outputs))
            )
        )
        assert plans[0].memory_requirements().scratch_nbytes == 0
        assert plans[1].memory_requirements().scratch_nbytes == 2 * m * n * 4
        assert plans[2].memory_requirements().scratch_nbytes == workspace_nbytes

        def run(plan, output):
            gemm.mm(
                (a.values.view(m, k, 1), a.scale_mma),
                (b_values.view(n, k, 1), packed.weight.scale_mma),
                out=output.view(m, n, 1),
                workspace=workspace,
                plan=plan,
            )
            return output

        for plan, output in zip(plans, outputs):
            torch.testing.assert_close(run(plan, output), oracle(), rtol=0, atol=0)

        eager_output = torch.empty((m, n, 1), dtype=torch.bfloat16, device=device)
        gemm.blockscaled.mm(
            (a.values.view(m, k, 1), a.scale_mma),
            (b_values.view(n, k, 1), packed.weight.scale_mma),
            out=eager_output,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            mma_tiler_mn=(16, 64),
            load_path="tma",
            swap_ab=False,
            _tile_k_override=128,
            _split_k_slices_override=4,
            _split_k_workspace=workspace,
            allow_bf16_atomic_reduction=False,
        )
        torch.testing.assert_close(eager_output.view(m, n), oracle(), rtol=0, atol=0)

        compiled = torch.compile(lambda: run(plans[2], outputs[2]), fullgraph=True)
        torch.testing.assert_close(compiled(), oracle(), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture():
            with torch.cuda.graph(graph):
                compiled()
        session.freeze()
        for lhs_first, rhs_first in ((1, 1), (2, 1), (1, 2), (0, 1)):
            a_values[0, positions[0]] = lhs_first
            b_values[:, positions[0]] = rhs_first
            expected = oracle()
            outputs[2].fill_(float("nan"))
            workspace.fill_(255)
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
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
            torch.testing.assert_close(outputs[2], expected, rtol=0, atol=0)
        a_values.zero_()
        b_values.zero_()
        a_scales.fill_(254)
        b_scales.fill_(254)
        outputs[2].fill_(float("nan"))
        workspace.fill_(255)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(outputs[2]) == 0
        assert torch.isfinite(outputs[2]).all()
        for plan, output in zip(plans, outputs):
            result = run(plan, output)
            assert torch.count_nonzero(result) == 0
            assert torch.isfinite(result).all()
        graph.reset()


@pytest.mark.parametrize("rows", (1, 7, 8))
def test_qkv_geometry_fp32_s4_oracle_and_native_replay(rows):
    device = require_b12x()
    k, n = 5120, 1792
    torch.manual_seed(51201792 + rows)
    source = (torch.randn((rows, k), device=device) * 0.25).bfloat16()
    a, (a_values, a_rows, a_scales) = rows_and_bases(rows, k, device=device)
    a_values.zero_()
    weight_values = (torch.randn((n, k), device=device) * 64).clamp(-448, 448)
    weight_values = weight_values.to(torch.float8_e4m3fn)
    weight_scales = torch.randint(
        114, 121, (n, k // 32), dtype=torch.uint8, device=device
    )
    packed = gemm.blockscaled.pack_weight(weight_values, weight_scales)
    b_values = packed.weight.values
    workspace_nbytes = 4 * rows * n * 4
    workspace = torch.empty(
        (workspace_nbytes // 4,), dtype=torch.float32, device=device
    )
    outputs = tuple(
        torch.empty((rows, n), dtype=torch.bfloat16, device=device) for _ in range(2)
    )
    quant = mxfp8.plan(
        mxfp8.query_from_call(
            source,
            a.values,
            a.scale_rows,
            a.scale_mma,
        )
    )
    query = gemm.DenseGemmQuery(
        recipe="mxfp8",
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=rows,
        in_features=k,
        out_features=n,
        output_mode="provided",
        alpha_mode="unit",
        workspace_form="provided",
        workspace_nbytes=workspace_nbytes,
        rhs_prefetch_ready=True,
        allow_bf16_atomic_reduction=False,
    )
    config = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=16,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=1,
        large_m_unroll=False,
        target_occupancy=None,
    )
    plans = (
        gemm.plan(query, override=config),
        gemm.plan(query, override=replace(config, split_k_slices=4)),
    )

    def dense_call(state, output):
        return PreparedCall(
            run=lambda: state.run(
                a.values.view(rows, k, 1),
                a.scale_mma,
                b_values.view(n, k, 1),
                packed.weight.scale_mma,
                None,
                output.view(rows, n, 1),
                workspace,
                torch.bfloat16,
                None,
            ),
            owners=(a_values, a_scales, packed, output, workspace),
        )

    def oracle():
        sa = torch.exp2(a.scale_rows.view(torch.uint8)[0].double() - 127)
        sb = torch.exp2(packed.weight.scale_rows.view(torch.uint8)[0].double() - 127)
        av = a_values.float().double() * sa.repeat_interleave(32, dim=1)
        bv = b_values.float().double() * sb.repeat_interleave(32, dim=1)
        return (av @ bv.T).bfloat16()

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (
                quant.request(
                    name="qkv-quant",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            source, a.values, a.scale_rows, a.scale_mma
                        ),
                        owners=(a_values, a_rows, a_scales),
                    ),
                ),
                plans[0].request(
                    name="qkv-s1",
                    prepare_call=lambda state: dense_call(state, outputs[0]),
                ),
                plans[1].request(
                    name="qkv-s4",
                    prepare_call=lambda state: dense_call(state, outputs[1]),
                ),
            )
        )

        def forward(plan, output):
            mxfp8.quantize_rows(source, a.values, a.scale_rows, a.scale_mma, plan=quant)
            return run_dense(plan, output)

        def run_dense(plan, output):
            gemm.mm(
                (a.values.view(rows, k, 1), a.scale_mma),
                (b_values.view(n, k, 1), packed.weight.scale_mma),
                out=output.view(rows, n, 1),
                workspace=workspace,
                plan=plan,
            )
            return output

        reference = forward(plans[0], outputs[0]).clone()
        candidate = forward(plans[1], outputs[1]).clone()
        exact = oracle()
        reference_error = float((reference.float() - exact.float()).abs().max())
        candidate_error = float((candidate.float() - exact.float()).abs().max())
        assert candidate_error <= reference_error, (
            rows,
            reference_error,
            candidate_error,
        )
        assert torch.isfinite(candidate).all()
        compiled = torch.compile(
            lambda: run_dense(plans[1], outputs[1]), fullgraph=True
        )
        torch.testing.assert_close(compiled(), candidate, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with session.capture():
            with torch.cuda.graph(graph):
                mxfp8.quantize_rows(
                    source, a.values, a.scale_rows, a.scale_mma, plan=quant
                )
                compiled()
        session.freeze()
        for factor in (0.5, 1.5):
            source.mul_(factor)
            b_values[:, 0].fill_(factor)
            expected = forward(plans[1], outputs[1]).clone()
            a_values.view(torch.uint8).fill_(255)
            a_scales.fill_(255)
            workspace.fill_(255)
            outputs[1].fill_(float("nan"))
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
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
            torch.testing.assert_close(outputs[1], expected, rtol=0, atol=0)
        graph.reset()
