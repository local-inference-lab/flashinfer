"""Check that both prepared four-row routes publish identical TP16 tensors during graph replay."""

from datetime import timedelta
from pathlib import Path

import pytest


def _worker(rank, directory):
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie._collective_tuning import CollectiveConfig
    from b12x.preparation import CollectiveRequirement, PreparationSession, PreparedCall

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=16,
        timeout=timedelta(minutes=5),
    )
    owner = pcie.DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=8,
        total_heads=16,
        head_dim=8,
        query_head_dim=672,
        stream_affine=False,
    )
    session = PreparationSession(
        device=device, autotune=False, compile_workers=4, prime=False
    )
    graphs = []
    try:
        local_down = torch.empty((4, 224), dtype=torch.bfloat16, device=device)
        local_router = torch.empty((4, 56), device=device)
        bias = torch.empty(896, device=device)
        requests, roles = [], []
        for impl, blocks in (("dcp_route_separate", 1), ("dcp_route_row", 4)):
            call = dict(
                local_down=local_down,
                local_router=local_router,
                correction_bias=bias,
                out_down=torch.empty((4, 3584), dtype=local_down.dtype, device=device),
                out_logits=torch.empty((4, 896), device=device),
                topk_weights=torch.empty((4, 16), device=device),
                topk_ids=torch.empty((4, 16), dtype=torch.int32, device=device),
            )
            q = pcie.query_from_runtime(
                owner, surface="DcpAllToAll.all_gather_pair_kimi_topk", call=call
            )
            plan = pcie.plan(
                q,
                runtime=owner,
                override=CollectiveConfig(
                    implementation=impl, threads=512, blocks=blocks
                ),
            )
            requests.append(
                plan.request(
                    name=impl,
                    prepare_call=lambda state, c=call: PreparedCall(
                        run=lambda: state.run(**c),
                        output=c["out_down"],
                        owners=(owner,),
                    ),
                    collective=CollectiveRequirement(impl, tuple(range(16))),
                )
            )
            roles.append((plan, call))
        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        for plan, call in roles:
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                owner.all_gather_pair_kimi_topk(**call, plan=plan)
            graphs.append(graph)
        ids = torch.arange(896, device=device)
        columns = torch.arange(4 * 3584, device=device).reshape(4, 3584)
        for step in range(12):
            source = (columns * 197 + step).to(torch.int16).view(torch.bfloat16)
            logits = torch.sin(
                torch.arange(4 * 896, device=device).reshape(4, 896)
                * (0.1 + step * 0.003)
            )
            bias.copy_((ids % 19).float() / 32)
            if step % 3 == 0:
                logits.zero_()
                bias.zero_()
            if step % 3 == 1:
                logits[:, ::11] = float("nan")
                logits[:, ::19] = float("inf")
                bias[::23] = -float("inf")
            local_down.copy_(source[:, rank * 224 : (rank + 1) * 224])
            local_router.copy_(logits[:, rank * 56 : (rank + 1) * 56])
            for i in (step % 2, (step + 1) % 2):
                graphs[i].replay()
            torch.cuda.synchronize()
            for _, call in roles:
                assert torch.equal(
                    call["out_down"].view(torch.int16), source.view(torch.int16)
                )
                assert torch.equal(
                    call["out_logits"].view(torch.int32), logits.view(torch.int32)
                )
            for key in ("topk_weights", "topk_ids"):
                assert torch.equal(roles[0][1][key], roles[1][1][key])
    finally:
        torch.cuda.synchronize()
        for g in graphs:
            g.reset()
        session.close()
        owner.close()
        dist.destroy_process_group()


def test_four_row_gather_selection_matches_unfused_and_rank_major_bytes(tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 16 or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires sixteen assigned SM120 GPUs")
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=16, join=True)
