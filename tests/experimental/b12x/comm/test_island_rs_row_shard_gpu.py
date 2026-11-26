"""TP16 GPU correctness, graph replay, and interleaving tests for island row-shard ops."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie._collective_tuning import CollectiveConfig
from b12x.comm.pcie._owner_preparation import prepared_call
from b12x.comm.pcie.pcie_allreduce import PCIeAllReduce
from b12x.preparation import PreparationSession, autotune_effort


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ISLAND_RS_TEST") != "1",
    reason="set B12X_RUN_PCIE_ISLAND_RS_TEST=1 to run the TP16 GPU test",
)

WORLD_SIZE = 16


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _allocator_replay_state(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    keys = (
        "allocation.all.allocated",
        "allocated_bytes.all.allocated",
        "reserved_bytes.all.allocated",
        "segment.all.allocated",
    )
    return {key: int(stats[key]) for key in keys}


def _ordered_island_sum(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Match island_hybrid_rsag's two-level ordered FP32->BF16 reduction."""
    islands = []
    for island in range(4):
        acc = sum(tensors[4 * island + lane].float() for lane in range(4))
        islands.append(acc.to(torch.bfloat16))
    return sum(t.float() for t in islands).to(torch.bfloat16)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["B12X_PCIE_ALLREDUCE_ALGORITHM"] = "island_rs"
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    max_size = 1024 * 7168 * 2
    runtime = PCIeAllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=max_size,
    )
    try:
        assert runtime._island_rs is not None

        def make_tensor(shape, seed):
            g = torch.Generator(device=device).manual_seed(seed + rank * 97)
            return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=g)

        # Cover aligned Kimi shapes plus small unaligned edge cases.
        rs_inp_large = make_tensor((1024, 7168), 101)
        rs_out_large = torch.empty((64, 7168), dtype=torch.bfloat16, device=device)
        rs_inp_moe = make_tensor((2048, 3584), 102)
        rs_out_moe = torch.empty((128, 3584), dtype=torch.bfloat16, device=device)
        rs_inp_small = make_tensor((48, 6), 103)
        rs_out_small = torch.empty((3, 6), dtype=torch.bfloat16, device=device)

        ag_inp_large = make_tensor((64, 7168), 201)
        ag_out_large = torch.empty((16, 64, 7168), dtype=torch.bfloat16, device=device)
        ag_inp_small = make_tensor((3, 6), 202)
        ag_out_small = torch.empty((16, 3, 6), dtype=torch.bfloat16, device=device)

        a2a_inp_moe = make_tensor((128, 3584), 301)
        a2a_out_moe = torch.empty((2048, 224), dtype=torch.bfloat16, device=device)
        a2a_inp_small = make_tensor((5, 96), 302)
        a2a_out_small = torch.empty((80, 6), dtype=torch.bfloat16, device=device)

        ar_inp = make_tensor((1024, 7168), 401)
        ar_out_hybrid = torch.empty_like(ar_inp)
        ar_out_sm = torch.empty_like(ar_inp)

        with autotune_effort("off"), PreparationSession(device=device) as session:
            p_rs_large = runtime.plan(
                rs_inp_large, operation="reduce_scatter", out=rs_out_large
            )
            p_rs_moe = runtime.plan(
                rs_inp_moe, operation="reduce_scatter", out=rs_out_moe
            )
            p_rs_small = runtime.plan(
                rs_inp_small, operation="reduce_scatter", out=rs_out_small
            )
            p_ag_large = runtime.plan(
                ag_inp_large, operation="all_gather", out=ag_out_large
            )
            p_ag_small = runtime.plan(
                ag_inp_small, operation="all_gather", out=ag_out_small
            )
            p_a2a_moe = runtime.plan(
                a2a_inp_moe, operation="row_to_col_all_to_all", out=a2a_out_moe
            )
            p_a2a_small = runtime.plan(
                a2a_inp_small, operation="row_to_col_all_to_all", out=a2a_out_small
            )
            p_ar_hybrid = runtime.plan(
                ar_inp,
                operation="all_reduce",
                out=ar_out_hybrid,
                override=CollectiveConfig(
                    implementation="island_hybrid_rsag",
                    threads=1024,
                    blocks=runtime._island_rs.blocks,
                    active_blocks=8,
                ),
            )
            p_ar_sm = runtime.plan(
                ar_inp,
                operation="all_reduce",
                out=ar_out_sm,
                override=CollectiveConfig(
                    implementation="island_push_rsag",
                    threads=512,
                    blocks=runtime._island_rs.blocks,
                    active_blocks=16,
                ),
            )
            requests = (
                p_rs_large.request(
                    name="rs_large",
                    prepare_call=lambda s: prepared_call(
                        s, inp=rs_inp_large, out=rs_out_large
                    ),
                ),
                p_rs_moe.request(
                    name="rs_moe",
                    prepare_call=lambda s: prepared_call(
                        s, inp=rs_inp_moe, out=rs_out_moe
                    ),
                ),
                p_rs_small.request(
                    name="rs_small",
                    prepare_call=lambda s: prepared_call(
                        s, inp=rs_inp_small, out=rs_out_small
                    ),
                ),
                p_ag_large.request(
                    name="ag_large",
                    prepare_call=lambda s: prepared_call(
                        s, inp=ag_inp_large, out=ag_out_large
                    ),
                ),
                p_ag_small.request(
                    name="ag_small",
                    prepare_call=lambda s: prepared_call(
                        s, inp=ag_inp_small, out=ag_out_small
                    ),
                ),
                p_a2a_moe.request(
                    name="a2a_moe",
                    prepare_call=lambda s: prepared_call(
                        s, inp=a2a_inp_moe, out=a2a_out_moe
                    ),
                ),
                p_a2a_small.request(
                    name="a2a_small",
                    prepare_call=lambda s: prepared_call(
                        s, inp=a2a_inp_small, out=a2a_out_small
                    ),
                ),
                p_ar_hybrid.request(
                    name="ar_hybrid",
                    prepare_call=lambda s: prepared_call(
                        s, inp=ar_inp, out=ar_out_hybrid
                    ),
                ),
                p_ar_sm.request(
                    name="ar_sm",
                    prepare_call=lambda s: prepared_call(s, inp=ar_inp, out=ar_out_sm),
                ),
            )
            result = session.prepare(requests, autotune=False)
            try:

                def gather_all(t):
                    out = [torch.empty_like(t) for _ in range(world_size)]
                    dist.all_gather(out, t)
                    return out

                def check_all_outputs():
                    # RS: exact match to two-level island FP32->BF16 reduction.
                    for inp, out in (
                        (rs_inp_large, rs_out_large),
                        (rs_inp_moe, rs_out_moe),
                        (rs_inp_small, rs_out_small),
                    ):
                        full = _ordered_island_sum(gather_all(inp))
                        st = inp.shape[0] // world_size
                        expected = full[rank * st : (rank + 1) * st]
                        assert torch.equal(out, expected), (
                            f"RS mismatch on rank {rank} shape={tuple(inp.shape)}"
                        )

                    # AG: bit-exact rank-major gather.
                    for inp, out in (
                        (ag_inp_large, ag_out_large),
                        (ag_inp_small, ag_out_small),
                    ):
                        expected = torch.stack(gather_all(inp), dim=0)
                        assert torch.equal(out, expected), (
                            f"AG mismatch on rank {rank} shape={tuple(inp.shape)}"
                        )

                    # A2A: bit-exact row-to-col transpose across ranks.
                    for inp, out in (
                        (a2a_inp_moe, a2a_out_moe),
                        (a2a_inp_small, a2a_out_small),
                    ):
                        all_in = gather_all(inp)
                        c = inp.shape[1] // world_size
                        expected = torch.cat(
                            [t[:, rank * c : (rank + 1) * c] for t in all_in], dim=0
                        )
                        assert torch.equal(out, expected), (
                            f"A2A mismatch on rank {rank} shape={tuple(inp.shape)}"
                        )

                    # Interleaved hybrid & SM all_reduce remain exact.
                    ar_ref = _ordered_island_sum(gather_all(ar_inp))
                    assert torch.equal(ar_out_hybrid, ar_ref)
                    torch.testing.assert_close(
                        ar_out_sm.float(), ar_ref.float(), rtol=0.02, atol=0.125
                    )

                def run_interleaved_sequence(stream=None):
                    # Exercises every transition pair across {RS, AG, A2A, AR_hybrid, AR_sm}.
                    runtime.reduce_scatter(
                        rs_inp_large, plan=p_rs_large, out=rs_out_large, stream=stream
                    )
                    runtime.reduce_scatter(
                        rs_inp_moe, plan=p_rs_moe, out=rs_out_moe, stream=stream
                    )
                    runtime.all_gather(
                        ag_inp_large, plan=p_ag_large, out=ag_out_large, stream=stream
                    )
                    runtime.row_to_col_all_to_all(
                        a2a_inp_moe, plan=p_a2a_moe, out=a2a_out_moe, stream=stream
                    )
                    runtime.all_reduce(
                        ar_inp, plan=p_ar_hybrid, out=ar_out_hybrid, stream=stream
                    )
                    runtime.row_to_col_all_to_all(
                        a2a_inp_small,
                        plan=p_a2a_small,
                        out=a2a_out_small,
                        stream=stream,
                    )
                    runtime.all_gather(
                        ag_inp_small, plan=p_ag_small, out=ag_out_small, stream=stream
                    )
                    runtime.reduce_scatter(
                        rs_inp_small, plan=p_rs_small, out=rs_out_small, stream=stream
                    )
                    runtime.all_reduce(
                        ar_inp, plan=p_ar_sm, out=ar_out_sm, stream=stream
                    )
                    # Follow SM op back into RS/AG/A2A.
                    runtime.all_gather(
                        ag_inp_large, plan=p_ag_large, out=ag_out_large, stream=stream
                    )
                    runtime.reduce_scatter(
                        rs_inp_large, plan=p_rs_large, out=rs_out_large, stream=stream
                    )
                    runtime.row_to_col_all_to_all(
                        a2a_inp_moe, plan=p_a2a_moe, out=a2a_out_moe, stream=stream
                    )

                # Eager interleaved back-to-back stress without host syncs.
                for _ in range(8):
                    run_interleaved_sequence()
                torch.cuda.synchronize(device)
                check_all_outputs()

                # CUDA graph capture + repeated replay with changed inputs and poisoned outputs.
                stream = torch.cuda.Stream(device=device)
                graph = torch.cuda.CUDAGraph()
                with (
                    runtime.capture(stream=stream),
                    torch.cuda.graph(graph, stream=stream),
                ):
                    run_interleaved_sequence(stream=stream)
                torch.cuda.synchronize(device)

                outputs = (
                    rs_out_large,
                    rs_out_moe,
                    rs_out_small,
                    ag_out_large,
                    ag_out_small,
                    a2a_out_moe,
                    a2a_out_small,
                    ar_out_hybrid,
                    ar_out_sm,
                )
                for step in range(4):
                    for out in outputs:
                        out.fill_(float("nan"))
                    rs_inp_large.copy_(make_tensor((1024, 7168), 1000 + step))
                    rs_inp_moe.copy_(make_tensor((2048, 3584), 2000 + step))
                    rs_inp_small.copy_(make_tensor((48, 6), 3000 + step))
                    ag_inp_large.copy_(make_tensor((64, 7168), 4000 + step))
                    ag_inp_small.copy_(make_tensor((3, 6), 5000 + step))
                    a2a_inp_moe.copy_(make_tensor((128, 3584), 6000 + step))
                    a2a_inp_small.copy_(make_tensor((5, 96), 7000 + step))
                    ar_inp.copy_(make_tensor((1024, 7168), 8000 + step))
                    torch.cuda.synchronize(device)
                    dist.barrier()
                    alloc_before = _allocator_replay_state(device)
                    for _ in range(10):
                        graph.replay()
                    torch.cuda.synchronize(device)
                    alloc_after = _allocator_replay_state(device)
                    assert alloc_after == alloc_before
                    check_all_outputs()
            finally:
                result.close()
    finally:
        runtime.close()
        dist.destroy_process_group()


def test_tp16_island_row_shard_eager_and_graph_correctness() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    assert torch.cuda.device_count() == WORLD_SIZE
    mp.spawn(_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)
