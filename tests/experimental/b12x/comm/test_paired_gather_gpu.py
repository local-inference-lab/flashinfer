"""Exact multi-CTA paired gathers across queued graph and epoch rollover."""

import ctypes
from datetime import timedelta
import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="requires explicitly assigned peer-access GPUs",
)

_FIRST_WIDTH = 24
_SECOND_WIDTH = 20
_MAX_ROWS = 17


def _worker(rank, world, origin, directory):
    import torch
    import torch.distributed as dist
    from b12x.comm import pcie
    from b12x.comm.pcie import _dcp_preparation
    from b12x.comm.pcie._dcp_a2a_cute import (
        _GRAPH_ARRIVED_INDEX,
        _GRAPH_EPOCH_INDEX,
    )
    from b12x.comm.pcie.pcie_dcp_a2a import _SIGNAL_BYTES
    from b12x.preparation import CollectiveRequirement, PreparationSession

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=(Path(directory) / "group").as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=5),
    )
    owner = pcie.DcpAllToAll.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=_MAX_ROWS,
        total_heads=world,
        head_dim=8,
        query_head_dim=2 * _FIRST_WIDTH + 4 * _SECOND_WIDTH,
        stream_affine=False,
    )
    session = PreparationSession(
        device=device, autotune=False, compile_workers=1, prime=False
    )
    cases = {}
    graphs = {}
    requests = []
    try:
        for rows, row_blocks in (
            (1, 1),
            (1, 4),
            (3, 2),
            (4, 4),
            (8, 2),
            (8, 8),
            (17, 4),
            (17, 8),
        ):
            first = torch.empty(
                (rows, _FIRST_WIDTH), dtype=torch.bfloat16, device=device
            )
            second = torch.empty(
                (rows, _SECOND_WIDTH), dtype=torch.float32, device=device
            )
            first_guard = torch.full(
                (rows * world * _FIRST_WIDTH * 2 + 32,),
                0xA5,
                dtype=torch.uint8,
                device=device,
            )
            second_guard = torch.full(
                (rows * world * _SECOND_WIDTH * 4 + 32,),
                0xA5,
                dtype=torch.uint8,
                device=device,
            )
            out_first = (
                first_guard[16:-16]
                .view(torch.bfloat16)
                .view(rows, world * _FIRST_WIDTH)
            )
            out_second = (
                second_guard[16:-16]
                .view(torch.float32)
                .view(rows, world * _SECOND_WIDTH)
            )
            call = dict(
                local_first=first,
                local_second=second,
                out_first=out_first,
                out_second=out_second,
                threads=512,
                row_blocks=row_blocks,
            )
            query = _dcp_preparation.query_from_runtime(
                owner, surface="DcpAllToAll.all_gather_pair", call=call
            )
            assert query.call["row_blocks"] == row_blocks
            assert query.call["disjoint_output"]
            plan = _dcp_preparation.plan(query, runtime=owner)
            name = f"paired-rows{rows}-blocks{row_blocks}"
            requests.append(
                plan.request(
                    name=name,
                    prepare_call=lambda state, call=call: _dcp_preparation.prepare_call(
                        state, **call
                    ),
                    collective=CollectiveRequirement(name, tuple(range(world))),
                )
            )

            variations = []
            for variation in range(3):
                generator = torch.Generator().manual_seed(
                    100000 * rows + 1000 * row_blocks + variation
                )
                first_bits = torch.randint(
                    0,
                    256,
                    (world, rows, _FIRST_WIDTH, 2),
                    dtype=torch.uint8,
                    generator=generator,
                )
                second_bits = torch.randint(
                    0,
                    256,
                    (world, rows, _SECOND_WIDTH, 4),
                    dtype=torch.uint8,
                    generator=generator,
                )
                first_values = first_bits.view(torch.bfloat16).squeeze(-1).to(device)
                second_values = second_bits.view(torch.float32).squeeze(-1).to(device)
                expected_first = (
                    torch.cat([first_bits[peer] for peer in range(world)], dim=1)
                    .flatten(1)
                    .to(device)
                )
                expected_second = (
                    torch.cat([second_bits[peer] for peer in range(world)], dim=1)
                    .flatten(1)
                    .to(device)
                )
                variations.append(
                    (
                        first_values[rank],
                        second_values[rank],
                        expected_first,
                        expected_second,
                    )
                )
            key = rows, row_blocks
            cases[key] = (
                first,
                second,
                out_first,
                out_second,
                first_guard,
                second_guard,
                variations,
                plan,
            )

        session.prepare(
            tuple(requests),
            coordinator=lambda progress: (
                progress.ready_collectives[0].key
                if progress.ready_collectives
                else None
            ),
        )
        first_case = cases[1, 1]
        for _ in range(1 if origin else 2):
            owner.all_gather_pair(
                first_case[0],
                first_case[1],
                first_case[2],
                first_case[3],
                plan=first_case[-1],
            )
        torch.cuda.synchronize()
        dist.barrier()
        for key, case in cases.items():
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                owner.all_gather_pair(case[0], case[1], case[2], case[3], plan=case[-1])
            graphs[key] = graph
        assert owner._device_slot_selection
        assert owner._graph_base_slot == origin
        session.freeze()

        stream = torch.cuda.current_stream(device)

        def signal_word(offset, value=None):
            host = (ctypes.c_uint32 * 1)(0 if value is None else value)
            address = owner._signal_ptrs[rank] + offset * 4
            owner._ipc.cudaMemcpyAsync(
                ctypes.addressof(host) if value is None else address,
                address if value is None else ctypes.addressof(host),
                4,
                stream.cuda_stream,
            )
            stream.synchronize()
            return host[0]

        assert signal_word(_GRAPH_ARRIVED_INDEX) == 0
        state_words = (ctypes.c_uint32 * (_SIGNAL_BYTES // 4))()
        for index in range(64 * 16):
            state_words[index] = 0xFFFFFFFD
        for index in range(64 * 16, len(state_words), 32):
            state_words[index] = 0xFFFFFFFD
        state_words[_GRAPH_EPOCH_INDEX] = 0xFFFFFFFD
        state_words[_GRAPH_ARRIVED_INDEX] = 0
        owner._ipc.cudaMemcpyAsync(
            owner._signal_ptrs[rank],
            ctypes.addressof(state_words),
            _SIGNAL_BYTES,
            stream.cuda_stream,
        )
        stream.synchronize()
        dist.barrier()

        schedule = [
            (key, variation)
            for variation in range(3)
            for key in (list(cases) if variation % 2 == 0 else list(reversed(cases)))
        ]
        snapshots = [
            (torch.empty_like(cases[key][2]), torch.empty_like(cases[key][3]))
            for key, _ in schedule
        ]
        for (key, variation), snapshot in zip(schedule, snapshots, strict=True):
            case = cases[key]
            operands = case[6][variation]
            case[0].copy_(operands[0])
            case[1].copy_(operands[1])
            case[2].view(torch.uint8).fill_(0xA5)
            case[3].view(torch.uint8).fill_(0xA5)
            graphs[key].replay()
            snapshot[0].copy_(case[2])
            snapshot[1].copy_(case[3])
        torch.cuda.synchronize()

        for (key, variation), snapshot in zip(schedule, snapshots, strict=True):
            expected = cases[key][6][variation]
            assert torch.equal(snapshot[0].view(torch.uint8), expected[2]), key
            assert torch.equal(snapshot[1].view(torch.uint8), expected[3]), key
        for case in cases.values():
            assert torch.all(case[4][:16] == 0xA5)
            assert torch.all(case[4][-16:] == 0xA5)
            assert torch.all(case[5][:16] == 0xA5)
            assert torch.all(case[5][-16:] == 0xA5)
            assert torch.equal(
                case[0].view(torch.uint8), case[6][-1][0].view(torch.uint8)
            )
            assert torch.equal(
                case[1].view(torch.uint8), case[6][-1][1].view(torch.uint8)
            )
        assert (
            signal_word(_GRAPH_EPOCH_INDEX) == (0xFFFFFFFD + len(schedule)) & 0xFFFFFFFF
        )
        assert signal_word(_GRAPH_ARRIVED_INDEX) == 0
        dist.barrier()
    finally:
        torch.cuda.synchronize()
        for graph in graphs.values():
            graph.reset()
        session.close()
        owner.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("origin", (0, 1))
def test_paired_gather_multi_cta_graph_rollover(origin, tmp_path):
    import torch
    import torch.multiprocessing as mp

    world = int(os.getenv("B12X_PCIE_DCP_A2A_WORLD_SIZE", "2"))
    if world not in (2, 4, 8, 16) or torch.cuda.device_count() < world:
        pytest.skip(f"requires {world} assigned peer-access GPUs")
    mp.spawn(_worker, args=(world, origin, str(tmp_path)), nprocs=world, join=True)
