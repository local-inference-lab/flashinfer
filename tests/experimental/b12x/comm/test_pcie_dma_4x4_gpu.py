"""16-GPU correctness, 200-cycle graph stress, and GEMM-overlap benchmark for TP16 4x4 PCIe DMA ring."""

from __future__ import annotations

import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie._dma_preparation import (
    plan as dma_plan,
    prepared_call,
    query_from_metadata,
)
from b12x.comm.pcie.pcie_dma import PCIeDmaAllReduce, _create_high_priority_stream
from b12x.preparation import PreparationSession

WORLD_SIZE = 16

_skip_unless_pcie_dma_gpu = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DMA_TEST") != "1",
    reason="set B12X_RUN_PCIE_DMA_TEST=1 to run 16-GPU PCIe DMA 4x4 tests",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ring_bf16_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() + b.float()).to(torch.bfloat16)


def _ref_4x4_rs_bf16(all_inputs: list[torch.Tensor], rank: int) -> torch.Tensor:
    """Exact bit-for-bit reference matching the 4x4 ring BF16 reduction order."""
    island, lane = divmod(rank, 4)
    # Phase 1: within each island i, reduce shard (island, lane) along lane ring ending at lane.
    intra_sums = []
    for i in range(4):
        # Hop order arriving at `lane`: starts from (lane + 1) % 4 -> (lane + 2) -> (lane + 3) -> lane.
        order = [(lane + 1 + step) % 4 for step in range(4)]
        acc = all_inputs[4 * i + order[0]][rank]
        for l_next in order[1:]:
            acc = _ring_bf16_add(all_inputs[4 * i + l_next][rank], acc)
        intra_sums.append(acc)
    # Phase 2: across islands along island ring ending at `island`.
    i_order = [(island + 1 + step) % 4 for step in range(4)]
    acc = intra_sums[i_order[0]]
    for i_next in i_order[1:]:
        acc = _ring_bf16_add(intra_sums[i_next], acc)
    return acc


def _worker(rank: int, world_size: int, port: int, run_bench: bool) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    dma = None
    sess = None
    aux = None
    try:
        max_bytes = 2048 * 7168 * 2  # 28 MiB
        dma = PCIeDmaAllReduce(
            exchange_group=dist.group.WORLD, max_bytes=max_bytes, device=device
        )
        assert dma._slab_nbytes <= (29 << 20), (
            f"unexpected slab size {dma._slab_nbytes}"
        )

        probe_in = torch.zeros((2048, 7168), dtype=torch.bfloat16, device=device)
        probe_out = torch.empty_like(probe_in)
        plan = dma_plan(
            query_from_metadata(dma, shape=(2048, 7168), dtype=torch.bfloat16),
            runtime=dma,
        )
        sess = PreparationSession(device=device, compile_workers=0, autotune=False)
        dist.barrier()
        sess.prepare(
            (
                plan.request(
                    name="dma_4x4",
                    prepare_call=lambda st: prepared_call(
                        st, inp=probe_in, out=probe_out
                    ),
                ),
            )
        )
        dist.barrier()

        # Deterministic rank-dependent BF16 inputs for all operations.
        g = torch.Generator(device=device).manual_seed(1000 + rank)
        rs28_in = (
            torch.randn((2048, 7168), dtype=torch.bfloat16, device=device, generator=g)
            * 0.25
        )
        rs28_out = torch.empty((128, 7168), dtype=torch.bfloat16, device=device)
        ag28_in = (
            torch.randn((128, 7168), dtype=torch.bfloat16, device=device, generator=g)
            * 0.5
        )
        ag28_out = torch.empty((16, 128, 7168), dtype=torch.bfloat16, device=device)
        route_in = (
            torch.randn((2048, 336), dtype=torch.bfloat16, device=device, generator=g)
            * 0.5
        )
        route_out = torch.empty((16, 2048, 336), dtype=torch.bfloat16, device=device)
        exp_in = (
            torch.randn((2048, 3584), dtype=torch.bfloat16, device=device, generator=g)
            * 0.25
        )
        exp_out = torch.empty((128, 3584), dtype=torch.bfloat16, device=device)
        sm_rs_in = (
            torch.randn((128, 1024), dtype=torch.bfloat16, device=device, generator=g)
            * 0.25
        )
        sm_rs_out = torch.empty((8, 1024), dtype=torch.bfloat16, device=device)
        sm_ag_in = (
            torch.randn((8, 1024), dtype=torch.bfloat16, device=device, generator=g)
            * 0.5
        )
        sm_ag_out = torch.empty((16, 8, 1024), dtype=torch.bfloat16, device=device)
        ar28_out = torch.empty_like(rs28_in)

        # Build exact CPU references via Gloo gather.
        def gather_cpu(t: torch.Tensor) -> list[torch.Tensor]:
            src = t.detach().cpu()
            buf = [torch.empty_like(src) for _ in range(world_size)]
            dist.all_gather(buf, src)
            return buf

        rs28_all = [x.view(16, 128, 7168) for x in gather_cpu(rs28_in)]
        ref_rs28 = _ref_4x4_rs_bf16(rs28_all, rank).to(device)
        ref_ag28 = torch.stack(gather_cpu(ag28_in), dim=0).to(device)
        ref_route = torch.stack(gather_cpu(route_in), dim=0).to(device)
        exp_all = [x.view(16, 128, 3584) for x in gather_cpu(exp_in)]
        ref_exp = _ref_4x4_rs_bf16(exp_all, rank).to(device)
        sm_all = [x.view(16, 8, 1024) for x in gather_cpu(sm_rs_in)]
        ref_sm_rs = _ref_4x4_rs_bf16(sm_all, rank).to(device)
        ref_sm_ag = torch.stack(gather_cpu(sm_ag_in), dim=0).to(device)
        ref_ar28 = torch.cat(
            [_ref_4x4_rs_bf16(rs28_all, r) for r in range(world_size)], dim=0
        ).to(device)

        # 1. Eager correctness across all ops (mixed pieces=1 and pieces=2).
        dma.reduce_scatter(rs28_in, plan=plan, out=rs28_out)
        dma.all_gather(ag28_in, plan=plan, out=ag28_out)
        dma.all_gather(route_in, plan=plan, out=route_out)
        dma.reduce_scatter(exp_in, plan=plan, out=exp_out)
        dma.reduce_scatter(sm_rs_in, plan=plan, out=sm_rs_out)
        dma.all_gather(sm_ag_in, plan=plan, out=sm_ag_out)
        dma.all_reduce(rs28_in, plan=plan, out=ar28_out)
        torch.cuda.synchronize(device)

        assert torch.equal(rs28_out, ref_rs28), f"rank {rank} eager RS28 mismatch"
        assert torch.equal(ag28_out, ref_ag28), f"rank {rank} eager AG28 mismatch"
        assert torch.equal(route_out, ref_route), f"rank {rank} eager RouteAG mismatch"
        assert torch.equal(exp_out, ref_exp), f"rank {rank} eager ExpRS mismatch"
        assert torch.equal(sm_rs_out, ref_sm_rs), f"rank {rank} eager SmallRS mismatch"
        assert torch.equal(sm_ag_out, ref_sm_ag), f"rank {rank} eager SmallAG mismatch"
        assert torch.equal(ar28_out, ref_ar28), f"rank {rank} eager AR28 mismatch"

        # 2. CUDA graph capture + 200 alternating stress cycles (1,400 ops!).
        for buf in (
            rs28_out,
            ag28_out,
            route_out,
            exp_out,
            sm_rs_out,
            sm_ag_out,
            ar28_out,
        ):
            buf.zero_()
        torch.cuda.synchronize(device)
        dist.barrier()

        graph_stream = torch.cuda.Stream(device=device)
        stress_graph = torch.cuda.CUDAGraph()
        with dma.capture(graph_stream):
            with torch.cuda.graph(stress_graph, stream=graph_stream):
                dma.reduce_scatter(rs28_in, plan=plan, out=rs28_out)
                dma.all_gather(ag28_in, plan=plan, out=ag28_out)
                dma.all_gather(route_in, plan=plan, out=route_out)
                dma.reduce_scatter(exp_in, plan=plan, out=exp_out)
                dma.reduce_scatter(sm_rs_in, plan=plan, out=sm_rs_out)
                dma.all_gather(sm_ag_in, plan=plan, out=sm_ag_out)
                dma.all_reduce(rs28_in, plan=plan, out=ar28_out)
        torch.cuda.synchronize(device)
        dist.barrier()

        for cycle in range(200):
            with torch.cuda.stream(graph_stream):
                stress_graph.replay()
        graph_stream.synchronize()
        dist.barrier()

        assert torch.equal(rs28_out, ref_rs28), f"rank {rank} graph RS28 mismatch"
        assert torch.equal(ag28_out, ref_ag28), f"rank {rank} graph AG28 mismatch"
        assert torch.equal(route_out, ref_route), f"rank {rank} graph RouteAG mismatch"
        assert torch.equal(exp_out, ref_exp), f"rank {rank} graph ExpRS mismatch"
        assert torch.equal(sm_rs_out, ref_sm_rs), f"rank {rank} graph SmallRS mismatch"
        assert torch.equal(sm_ag_out, ref_sm_ag), f"rank {rank} graph SmallAG mismatch"
        assert torch.equal(ar28_out, ref_ar28), f"rank {rank} graph AR28 mismatch"

        if run_bench:
            aux = _create_high_priority_stream(device)
            ev_fork = torch.cuda.Event()
            ev_join = torch.cuda.Event()
            # ~450 us 188-SM BF16 GEMM on RTX PRO 6000 Blackwell
            gemm_a = torch.randn((2048, 7168), dtype=torch.bfloat16, device=device)
            gemm_b = torch.randn((7168, 7168), dtype=torch.bfloat16, device=device)
            gemm_c = torch.empty((2048, 7168), dtype=torch.bfloat16, device=device)

            def measure_graph(fn, reps=30):
                # Warmup
                with torch.cuda.stream(graph_stream):
                    fn()
                graph_stream.synchronize()
                dist.barrier()
                g_rec = torch.cuda.CUDAGraph()
                with dma.capture(graph_stream):
                    with torch.cuda.graph(g_rec, stream=graph_stream):
                        fn()
                graph_stream.synchronize()
                dist.barrier()
                for _ in range(5):
                    with torch.cuda.stream(graph_stream):
                        g_rec.replay()
                graph_stream.synchronize()
                dist.barrier()
                evs = [
                    (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    for _ in range(reps)
                ]
                with torch.cuda.stream(graph_stream):
                    for s, e in evs:
                        s.record(graph_stream)
                        g_rec.replay()
                        e.record(graph_stream)
                graph_stream.synchronize()
                times = sorted(s.elapsed_time(e) * 1000.0 for s, e in evs)
                return times[len(times) // 2]

            gemm_us = measure_graph(lambda: torch.mm(gemm_a, gemm_b, out=gemm_c))
            cases = [
                (
                    "RS 28 MiB (2048x7168)",
                    lambda: dma.reduce_scatter(rs28_in, plan=plan, out=rs28_out),
                ),
                (
                    "AG 28 MiB (128x7168)",
                    lambda: dma.all_gather(ag28_in, plan=plan, out=ag28_out),
                ),
                (
                    "Route AG 21 MiB (2048x336)",
                    lambda: dma.all_gather(route_in, plan=plan, out=route_out),
                ),
                (
                    "Expert RS 14 MiB (2048x3584)",
                    lambda: dma.reduce_scatter(exp_in, plan=plan, out=exp_out),
                ),
            ]
            if rank == 0:
                print(
                    f"\n=== TP16 4x4 CE Ring Microbenchmark (slab={dma._slab_nbytes / 1048576:.2f} MiB, "
                    f"fused_wait={dma._fused_wait_add_4x4}, grid_cap={dma._add_grid_cap_4x4}, "
                    f"GEMM={gemm_us:.1f} us) ===",
                    flush=True,
                )
            for label, op_fn in cases:
                solo_us = measure_graph(op_fn)

                def overlap_fn(_op=op_fn):
                    ev_fork.record(graph_stream)
                    aux.wait_event(ev_fork)
                    with torch.cuda.stream(aux):
                        _op()
                        ev_join.record(aux)
                    torch.mm(gemm_a, gemm_b, out=gemm_c)
                    graph_stream.wait_event(ev_join)

                ov_us = measure_graph(overlap_fn)
                # Fraction of the smaller of (solo, gemm) hidden by overlap:
                hidden_us = (solo_us + gemm_us) - ov_us
                hidden_pct = 100.0 * hidden_us / min(solo_us, gemm_us)
                if rank == 0:
                    print(
                        f"  {label:30s}: solo={solo_us:6.1f} us | +GEMM={ov_us:6.1f} us "
                        f"| hidden={hidden_us:5.1f} us ({hidden_pct:5.1f}%)",
                        flush=True,
                    )
    finally:
        if aux is not None:
            aux._b12x_release()
        if dma is not None:
            dma.close()
        dist.destroy_process_group()


@_skip_unless_pcie_dma_gpu
def test_tp16_pcie_dma_4x4_eager_graph_and_stress() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert torch.cuda.device_count() == WORLD_SIZE
    run_bench = os.getenv("B12X_BENCH_PCIE_DMA_4X4", "0") == "1"
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _free_port(), run_bench),
        nprocs=WORLD_SIZE,
        join=True,
    )
