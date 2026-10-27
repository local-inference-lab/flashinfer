"""Dependent FP32 split-K replay waits for mutable activation partials."""

from dataclasses import replace

import pytest
import torch
import triton
import triton.language as tl
from cuda.bindings import runtime as cudart

from b12x import gemm
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from ._mxfp8_rows_fixture import rows_and_bases


@triton.jit
def _timer():
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        constraints="=l",
        args=[],
        dtype=tl.int64,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _delayed_activation(
    source_values,
    source_rows,
    source_mma,
    values,
    rows,
    mma,
    VALUE_COUNT: tl.constexpr,
    ROW_COUNT: tl.constexpr,
    MMA_COUNT: tl.constexpr,
    DELAY_NS: tl.constexpr,
):
    tl.extra.cuda.gdc_launch_dependents()
    started = _timer()
    now = started
    while now - started < DELAY_NS:
        now = _timer()
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    tl.store(
        values + offsets,
        tl.load(source_values + offsets, offsets < VALUE_COUNT, 0),
        offsets < VALUE_COUNT,
    )
    tl.store(
        rows + offsets,
        tl.load(source_rows + offsets, offsets < ROW_COUNT, 127),
        offsets < ROW_COUNT,
    )
    tl.store(
        mma + offsets,
        tl.load(source_mma + offsets, offsets < MMA_COUNT, 127),
        offsets < MMA_COUNT,
    )


@triton.jit
def _dependent_output(out, probe, start_time, COUNT: tl.constexpr):
    started = _timer()
    tl.extra.cuda.gdc_wait()
    offsets = tl.program_id(0) * 256 + tl.arange(0, 256)
    values = tl.load(out + offsets, offsets < COUNT, 0)
    tl.store(probe + offsets, values, offsets < COUNT)
    if tl.program_id(0) == 0:
        tl.store(start_time, started)


def _programmatic_edges(graph):
    status, _, _, _, count = cudart.cudaGraphGetEdges(graph.raw_cuda_graph())
    assert status == cudart.cudaError_t.cudaSuccess
    status, _, _, data, returned = cudart.cudaGraphGetEdges(
        graph.raw_cuda_graph(), count
    )
    assert status == cudart.cudaError_t.cudaSuccess and returned == count
    return sum(
        int(edge.from_port) == 1 and int(edge.type) == 1 for edge in data[:returned]
    )


@pytest.mark.parametrize(
    "n,k,slices", ((1792, 5120, 2), (1792, 5120, 4), (8192, 1280, 2))
)
def test_fp32_split_k_pdl_delayed_mutable_replay(n, k, slices):
    device = require_b12x()
    m = 8
    torch.manual_seed(4172027 + slices)
    source, (source_values, source_rows, source_mma) = rows_and_bases(
        m, k, device=device
    )
    activation, (values, rows, mma) = rows_and_bases(m, k, device=device)
    rhs, (b_values, b_rows, b_mma) = rows_and_bases(n, k, device=device)
    workspace_nbytes = slices * m * n * 4
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
        rhs_prefetch_ready=True,
        allow_bf16_atomic_reduction=False,
    )
    base = gemm.DenseGemmConfig(
        backend="cutedsl",
        tile_m=16,
        tile_n=64,
        tile_k=128,
        load_path="tma",
        swap_ab=False,
        split_k_slices=slices,
        large_m_unroll=False,
        target_occupancy=None,
    )
    plans = (
        gemm.plan(query, override=base),
        gemm.plan(query, override=replace(base, programmatic_serialization=True)),
    )
    outputs = tuple(
        torch.empty((m, n), dtype=torch.bfloat16, device=device) for _ in plans
    )
    probes = tuple(torch.empty_like(output) for output in outputs)
    start_times = tuple(
        torch.empty((), dtype=torch.int64, device=device) for _ in plans
    )
    workspaces = tuple(
        torch.empty(workspace_nbytes // 4, dtype=torch.float32, device=device)
        for _ in plans
    )

    def prepared(state, index):
        return PreparedCall(
            run=lambda: state.run(
                activation.values.view(m, k, 1),
                activation.scale_mma,
                rhs.values.view(n, k, 1),
                rhs.scale_mma,
                None,
                outputs[index].view(m, n, 1),
                workspaces[index],
                torch.bfloat16,
                None,
            ),
            owners=(activation, rhs, outputs[index], workspaces[index]),
        )

    def consumer(index):
        gemm.mm(
            (activation.values.view(m, k, 1), activation.scale_mma),
            (rhs.values.view(n, k, 1), rhs.scale_mma),
            out=outputs[index].view(m, n, 1),
            workspace=workspaces[index],
            plan=plans[index],
        )

    compiled_candidate = torch.compile(lambda: consumer(1), fullgraph=True)

    def produce(index):
        _delayed_activation[(triton.cdiv(m * k, 256),)](
            source_values.view(torch.uint8),
            source_rows,
            source_mma,
            values.view(torch.uint8),
            rows,
            mma,
            source_values.numel(),
            source_rows.numel(),
            source_mma.numel(),
            200_000,
            launch_pdl=True,
        )
        if index == 1:
            compiled_candidate()
        else:
            consumer(index)
        _dependent_output[(triton.cdiv(m * n, 256),)](
            outputs[index],
            probes[index],
            start_times[index],
            m * n,
            launch_pdl=True,
        )

    def oracle():
        left = dequantize_mxfp8_rows_torch(source.values, source.scale_rows).double()
        right = dequantize_mxfp8_rows_torch(rhs.values, rhs.scale_rows).double()
        return left @ right.T, left.abs() @ right.abs().T

    with PreparationSession(
        device=device, autotune=False, compile_workers=2
    ) as session:
        source_values.copy_(
            (torch.randn((m, k), device=device) * 0.125).to(source_values.dtype)
        )
        b_values.copy_((torch.randn((n, k), device=device) * 0.125).to(b_values.dtype))
        values.copy_(source_values)
        rows.copy_(source_rows)
        mma.copy_(source_mma)
        session.prepare(
            tuple(
                plan.request(
                    name=f"split-{slices}-pdl-{index}",
                    prepare_call=lambda state, index=index: prepared(state, index),
                )
                for index, plan in enumerate(plans)
            )
        )
        assert all(
            plan.memory_requirements().scratch_nbytes == workspace_nbytes
            for plan in plans
        )
        produce(0)
        produce(1)
        torch.cuda.synchronize(device)
        graphs = []
        for index in range(2):
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                produce(index)
            graph.instantiate()
            graphs.append(graph)
        assert _programmatic_edges(graphs[1]) >= 3
        session.freeze()
        for step, (left_scale, right_scale) in enumerate(
            ((127, 127), (128, 126), (254, 254))
        ):
            if step == 2:
                source_values.zero_()
                b_values.zero_()
            else:
                source_values.copy_(
                    (torch.randn((m, k), device=device) * 0.125).to(source_values.dtype)
                )
                b_values.copy_(
                    (torch.randn((n, k), device=device) * 0.125).to(b_values.dtype)
                )
            source_rows.fill_(left_scale)
            source_mma.fill_(left_scale)
            b_rows.fill_(right_scale)
            b_mma.fill_(right_scale)
            exact, magnitude = oracle()
            previous = None
            for index, graph in enumerate(graphs):
                values.view(torch.uint8).fill_(255)
                rows.fill_(255)
                mma.fill_(255)
                outputs[index].fill_(float("nan"))
                probes[index].fill_(float("nan"))
                workspaces[index].fill_(255)
                torch.cuda.synchronize(device)
                before = torch.cuda.memory_stats(device)
                graph.replay()
                torch.cuda.synchronize(device)
                after = torch.cuda.memory_stats(device)
                for key in (
                    "allocation.all.allocated",
                    "allocation.all.freed",
                    "segment.all.allocated",
                    "segment.all.freed",
                ):
                    assert before[key] == after[key]
                assert torch.equal(values, source_values)
                assert torch.equal(rows, source_rows)
                assert torch.equal(mma, source_mma)
                assert torch.equal(probes[index], outputs[index])
                actual = outputs[index].double()
                below = torch.nextafter(
                    outputs[index],
                    torch.full_like(outputs[index], -float("inf")),
                ).double()
                above = torch.nextafter(
                    outputs[index],
                    torch.full_like(outputs[index], float("inf")),
                ).double()
                u = (k + slices) * 2.0**-24
                error = u / (1.0 - u) * magnitude
                error += 0.5 * torch.maximum(actual - below, above - actual)
                assert torch.isfinite(outputs[index]).all()
                assert bool(((actual - exact).abs() <= error).all())
                if previous is not None:
                    assert torch.equal(outputs[index], previous)
                previous = outputs[index].clone()
            if step == 2:
                assert torch.count_nonzero(outputs[1]) == 0
        for graph in graphs:
            graph.reset()
