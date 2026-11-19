# SPDX-License-Identifier: Apache-2.0
"""Check captured TP16 slot selection against independent CPU stable radix ordering."""

from datetime import timedelta
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie import interleaved_topk
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
from b12x.preparation import PreparationSession, PreparedCall


WORLD, K, ROWS = 16, 2048, 4
PAGES = 313
CONTEXTS = (
    (-1, 0, 16, 2047),
    (2048, 65535, 2047, -1),
    (262143, 262144, 1_048_575, 1_048_606),
    ((1 << 63) - 1, 1 << 31, 32767, 32768),
)


def fixture(positions, seed, rank, table_rows):
    """Generate rank-major candidates, including exact ties around invalid page entries."""
    random = np.random.default_rng(seed)
    ids = np.broadcast_to(np.arange(K, dtype=np.int32), (ROWS, WORLD, K)).copy()
    words = random.choice(
        np.array(
            [
                0xFFFFFFFF,
                0xFF800001,
                0xFF800000,
                0x80000000,
                0,
                0x3F800000,
                0x40000000,
                0x7F800000,
                0x7FC00003,
            ],
            dtype=np.uint32,
        ),
        size=ids.shape,
    )
    pages = (
        np.arange(table_rows * 521, dtype=np.int32).reshape(table_rows, 521) * 17
    ) % PAGES
    pages[:, 5::59] = -1
    pages[:, 13::61] = PAGES
    tables = np.broadcast_to(pages, (ROWS, pages.shape[1]))
    expected_slots = np.full((ROWS, K), -1, dtype=np.int32)
    counts = np.zeros(ROWS, dtype=np.int32)
    for row, pos in enumerate(positions):
        for peer in range(WORLD):
            visible = max(0, (int(pos) + WORLD - peer) // WORLD)
            length = min(K, visible)
            rng = ids[row, peer, :length].astype(np.int64)
            if visible > K:
                rng = (rng * 31 + peer * 67) % min(visible, 66580)
            ids[row, peer, :length] = rng
            ids[row, peer, length:] = -1
            ids[row, peer, 17:length:509] = -1
            if 0 < length < K:
                # The source selector allows gaps before the last valid column.
                ids[row, peer, -1] = ids[row, peer, 0]
                words[row, peer, -1] = words[row, peer, 0]
                ids[row, peer, 0] = -1
                if pos == 0 and peer == 0:
                    words[row, peer, -1] = 0x3F800000
            if length == K:
                # Future or out-of-table IDs still participate in score selection.
                ids[row, peer, 31] = (1 << 30) + 1
            words[row, peer, ids[row, peer] < 0] = 0xFF800000
        score = words[row].reshape(-1)
        ordered = np.where(score >> 31, ~score, score ^ np.uint32(1 << 31))
        selected = np.zeros(WORLD * K, dtype=np.bool_)
        selected[np.argsort(~ordered, kind="stable")[:K]] = True
        local = ids[row, rank][selected.reshape(WORLD, K)[rank]].astype(np.int64)
        local = local[
            (local >= 0)
            & (local * WORLD + rank <= pos)
            & (local // 128 < tables.shape[1])
        ]
        owner_page = tables[row, local // 128]
        valid = (owner_page >= 0) & (owner_page < PAGES)
        survivors = (owner_page * 128 + local % 128)[valid]
        counts[row] = len(survivors)
        expected_slots[row, : len(survivors)] = survivors
    return (
        ids[:, rank].copy(),
        words[:, rank].copy().view(np.float32),
        pages,
        expected_slots,
        counts,
    )


def worker(rank, rendezvous, cache):
    torch.cuda.set_device(rank)
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=WORLD,
        init_method=rendezvous,
        timeout=timedelta(minutes=3),
    )
    device = torch.device("cuda", rank)
    runtime = PCIeDCPA2A.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_batch_size=4,
        total_heads=16,
        head_dim=1024,
        stream_affine=False,
    )
    try:
        ids = torch.zeros((ROWS, K), dtype=torch.int32, device=device)
        scores = torch.zeros_like(ids, dtype=torch.float32)
        positions = torch.full((ROWS,), -1, dtype=torch.int64, device=device)
        tables = torch.zeros((ROWS, 521), dtype=torch.int32, device=device)
        slots, owners = torch.empty_like(ids), torch.empty_like(ids)
        counts = torch.empty(ROWS, dtype=torch.int32, device=device)

        def call(state, mapping):
            return state.run(ids, scores, positions, mapping, owners, slots, counts)

        session = PreparationSession(
            device=device,
            cache_dir=Path(cache) / f"rank-{rank}",
            autotune=False,
            compile_workers=0,
        )
        plan = interleaved_topk.plan(runtime, cache_pages=PAGES)
        session.prepare(
            (
                plan.request(
                    name="interleaved-top-2048",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: call(state, tables), output=slots
                    ),
                ),
            )
        )
        state = plan.prepared.state
        graphs = {}
        for rows in (1, ROWS):
            graph = torch.cuda.CUDAGraph()
            mapping = tables if rows == ROWS else tables[:1].expand(ROWS, -1)
            dist.barrier()
            with session.capture(), torch.cuda.graph(graph):
                call(state, mapping)
            graphs[rows] = graph
        session.freeze()

        schedule = [(0, 2)] * 66 + [(1, 7), (2, 8), (0, 16), (3, 21), (2, 4)]
        cases = {}
        for step, (kind, seed) in enumerate(schedule):
            rows = 1 if step % 2 == 0 else ROWS
            key = kind, seed, rows
            if key not in cases:
                cases[key] = fixture(CONTEXTS[kind], seed, rank, rows)
            local, values, table, expected, sizes = cases[key]
            ids.copy_(torch.from_numpy(local))
            scores.copy_(torch.from_numpy(values))
            positions.copy_(torch.tensor(CONTEXTS[kind], dtype=torch.int64))
            tables[:rows].copy_(torch.from_numpy(table))
            graphs[rows].replay()
            np.testing.assert_array_equal(slots.cpu().numpy(), expected)
            np.testing.assert_array_equal(
                owners.cpu().numpy(), np.where(expected >= 0, rank, -1)
            )
            np.testing.assert_array_equal(counts.cpu().numpy(), sizes)
        # With a delayed reader, fast ranks must retire the short-row summary
        # ring before overwriting an earlier generation.
        local, values, table, expected, sizes = fixture(CONTEXTS[0], 11, rank, 1)
        ids.copy_(torch.from_numpy(local))
        scores.copy_(torch.from_numpy(values))
        positions.copy_(torch.tensor(CONTEXTS[0], dtype=torch.int64))
        tables[:1].copy_(torch.from_numpy(table))
        dist.barrier()
        if rank == WORLD - 1:
            torch.cuda._sleep(1_000_000)
        for _ in range(129):
            graphs[1].replay()
        np.testing.assert_array_equal(slots.cpu().numpy(), expected)
        np.testing.assert_array_equal(counts.cpu().numpy(), sizes)
        dist.barrier()
        session.close()
    finally:
        runtime.close()
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_A2A_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_A2A_TEST=1 for the TP16 GPU test",
)
def test_tp16_stable_slots_and_ring_replay(tmp_path):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires 16 CUDA IPC peers")
    mp.spawn(
        worker, nprocs=WORLD, args=((tmp_path / "rendezvous").as_uri(), str(tmp_path))
    )
