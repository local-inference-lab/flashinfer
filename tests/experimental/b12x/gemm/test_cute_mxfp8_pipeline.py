"""Prepared CuTe MXFP8 launch policy preserves the complete UE8M0 domain."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl

from b12x import gemm
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from ._mxfp8_rows_fixture import rows_and_bases as _rows


@triton.jit
def _produce_activation(
    values,
    scales,
    source_values,
    source_scales,
    value_count: tl.constexpr,
    scale_count: tl.constexpr,
):
    # The successor may launch while these writes are still in progress.
    tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(source_values + offsets, offsets < value_count, other=0)
    scale = tl.load(source_scales + offsets, offsets < scale_count, other=127)
    tl.store(values + offsets, value, offsets < value_count)
    tl.store(scales + offsets, scale, offsets < scale_count)


@pytest.mark.parametrize("rows", (1, 7, 8))
@pytest.mark.parametrize("pdl,evict", ((False, True), (True, False), (True, True)))
def test_prepared_mxfp8_rhs_policy_matches_cute_and_full_domain(rows, pdl, evict):
    device = require_b12x()
    k, n = 1280, 8192
    a, a_bases = _rows(rows, k, device=device)
    b, b_bases = _rows(n, k, device=device)
    torch.manual_seed(8152026)
    a.values.copy_((torch.randn((rows, k), device=device) * 0.25).to(a.values.dtype))
    b.values.copy_((torch.randn((n, k), device=device) * 0.25).to(b.values.dtype))
    out_control = torch.empty((rows, n, 1), dtype=torch.bfloat16, device=device)
    out_policy = torch.empty_like(out_control)
    workspace = torch.empty((0,), dtype=torch.float32, device=device)
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
        workspace_nbytes=0,
        rhs_prefetch_ready=True,
    )
    control = gemm.DenseGemmConfig(
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
    policy = replace(control, programmatic_serialization=pdl, evict_first=evict)
    plans = (gemm.plan(query, override=control), gemm.plan(query, override=policy))

    def request(plan, out, name):
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    a.values.view(rows, k, 1),
                    a.scale_mma,
                    b.values.view(n, k, 1),
                    b.scale_mma,
                    None,
                    out,
                    workspace,
                    torch.bfloat16,
                    None,
                ),
                owners=(*a_bases, *b_bases, out, workspace),
            ),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            (
                request(plans[0], out_control, "control"),
                request(plans[1], out_policy, "policy"),
            )
        )

        def run():
            gemm.mm(
                (a.values.view(rows, k, 1), a.scale_mma),
                (b.values.view(n, k, 1), b.scale_mma),
                out=out_control,
                workspace=workspace,
                plan=plans[0],
            )
            gemm.mm(
                (a.values.view(rows, k, 1), a.scale_mma),
                (b.values.view(n, k, 1), b.scale_mma),
                out=out_policy,
                workspace=workspace,
                plan=plans[1],
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(out_policy, out_control, rtol=0, atol=0)

        run()
        a.values.fill_(1)
        b.values.fill_(1)
        for lhs_scale, rhs_scale in ((64, 190), (190, 64)):
            a_bases[2].fill_(lhs_scale)
            b_bases[2].fill_(rhs_scale)
            run()
            assert torch.all(
                out_policy == torch.tensor(k, dtype=torch.bfloat16, device=device)
            )
        # UE8M0 254*254 overflows FP32 in software rescaling, even for zero
        # products. Native MXFP8 MMA must keep the finite zero result.
        a.values.zero_()
        b.values.zero_()
        a_bases[2].fill_(254)
        b_bases[2].fill_(254)
        run()
        assert torch.count_nonzero(out_policy) == 0
        assert torch.isfinite(out_policy).all()

        if rows == 8 and pdl:
            source_values = (
                (torch.randn((rows, k), device=device) * 0.125)
                .to(torch.float8_e4m3fn)
                .view(torch.uint8)
                .contiguous()
            )
            source_scales = torch.full_like(a_bases[2], 127)
            graph = torch.cuda.CUDAGraph()
            with session.capture():
                with torch.cuda.graph(graph):
                    _produce_activation[(triton.cdiv(rows * k, 256),)](
                        a.values.view(torch.uint8),
                        a_bases[2].view(-1),
                        source_values.view(-1),
                        source_scales.view(-1),
                        rows * k,
                        source_scales.numel(),
                        launch_pdl=True,
                    )
                    gemm.mm(
                        (a.values.view(rows, k, 1), a.scale_mma),
                        (b.values.view(n, k, 1), b.scale_mma),
                        out=out_policy,
                        workspace=workspace,
                        plan=plans[1],
                    )
            session.freeze()
            for rhs_scale in (127, 128, 254):
                # The RHS is stable during one replay, but may change between
                # replays. Poison A and SFA to prove the producer is observed.
                if rhs_scale == 254:
                    b.values.zero_()
                    source_values.zero_()
                else:
                    b.values.copy_(
                        (torch.randn((n, k), device=device) * 0.125).to(
                            torch.float8_e4m3fn
                        )
                    )
                b_bases[2].fill_(rhs_scale)
                source_scales.fill_(rhs_scale)
                a.values.view(torch.uint8).fill_(255)
                a_bases[2].fill_(255)
                out_policy.fill_(float("nan"))
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
                gemm.mm(
                    (a.values.view(rows, k, 1), a.scale_mma),
                    (b.values.view(n, k, 1), b.scale_mma),
                    out=out_control,
                    workspace=workspace,
                    plan=plans[0],
                )
                torch.testing.assert_close(out_policy, out_control, rtol=0, atol=0)
            graph.reset()


@pytest.mark.parametrize("rows", (1, 7, 8))
def test_evict_first_preserves_split_k_mxfp8(rows):
    device = require_b12x()
    k, n = 5120, 1792
    a, a_bases = _rows(rows, k, device=device)
    b, b_bases = _rows(n, k, device=device)
    torch.manual_seed(4102026)
    a.values.copy_((torch.randn((rows, k), device=device) * 0.125).to(a.values.dtype))
    b.values.copy_((torch.randn((n, k), device=device) * 0.125).to(b.values.dtype))
    out = tuple(
        torch.empty((rows, n, 1), dtype=torch.bfloat16, device=device) for _ in range(2)
    )
    workspace = torch.empty((0,), dtype=torch.float32, device=device)
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
        workspace_nbytes=0,
        rhs_prefetch_ready=True,
    )
    config = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=16,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=4,
        large_m_unroll=False,
        target_occupancy=None,
    )
    plans = (
        gemm.plan(query, override=config),
        gemm.plan(query, override=replace(config, evict_first=True)),
    )

    def execute(plan, output):
        return gemm.mm(
            (a.values.view(rows, k, 1), a.scale_mma),
            (b.values.view(n, k, 1), b.scale_mma),
            out=output,
            workspace=workspace,
            plan=plan,
        )

    def request(plan, output, name):
        return plan.request(
            name=name,
            prepare_call=lambda state: PreparedCall(
                run=lambda: state.run(
                    a.values.view(rows, k, 1),
                    a.scale_mma,
                    b.values.view(n, k, 1),
                    b.scale_mma,
                    None,
                    output,
                    workspace,
                    torch.bfloat16,
                    None,
                ),
                owners=(*a_bases, *b_bases, output, workspace),
            ),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        session.prepare(
            tuple(
                request(plan, output, f"split-k-{i}")
                for i, (plan, output) in enumerate(zip(plans, out))
            )
        )
        controls = [execute(plans[0], out[0]).clone() for _ in range(8)]
        candidates = [execute(plans[1], out[1]).clone() for _ in range(8)]
        oracle = (
            (a.values.float().double() @ b.values.float().double().T)
            .bfloat16()
            .unsqueeze(-1)
        )

        def deviation(x, y):
            return float((x.float() - y.float()).abs().max())

        control_span = max(deviation(x, controls[0]) for x in controls[1:])
        control_oracle = max(deviation(x, oracle) for x in controls)
        candidate_span = max(deviation(x, candidates[0]) for x in candidates[1:])
        candidate_oracle = max(deviation(x, oracle) for x in candidates)
        cross = max(deviation(x, y) for x in controls for y in candidates)
        # Atomic BF16 split-K has measurable variation even with unchanged
        # operands. The cache hint must stay within that control envelope and
        # the independently dequantized FP64 product's observed control error.
        bound = max(control_span, control_oracle)
        assert all(
            value <= bound for value in (candidate_span, candidate_oracle, cross)
        ), (rows, control_span, control_oracle, candidate_span, candidate_oracle, cross)
        graph = torch.cuda.CUDAGraph()
        with session.capture():
            with torch.cuda.graph(graph):
                execute(plans[1], out[1])
        session.freeze()
        for _ in range(3):
            out[1].fill_(float("nan"))
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
            assert deviation(out[1], oracle) <= bound
        a.values.zero_()
        b.values.zero_()
        a_bases[2].fill_(254)
        b_bases[2].fill_(254)
        for plan, output in zip(plans, out):
            result = execute(plan, output)
            assert torch.count_nonzero(result) == 0
            assert torch.isfinite(result).all()
        out[1].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(out[1]) == 0
        assert torch.isfinite(out[1]).all()
        graph.reset()
