from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_ONESHOT_RANK_IDENTITY") != "1",
    reason=(
        "set B12X_RUN_PCIE_ONESHOT_RANK_IDENTITY=1 to run the four-rank "
        "PCIe oneshot rank-identity GPU test"
    ),
)

WORLD_SIZE = 4
ITERS = 16
SHAPES = ((1, 4096), (8, 4096), (32, 4096), (6, 8192))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int, dtype_name: str) -> None:
    dtype = getattr(torch, dtype_name)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    numel = max(rows * cols for rows, cols in SHAPES)
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=numel * dtype.itemsize,
        max_size=numel * dtype.itemsize,
        max_concurrent_channels=1,
    )
    stream = torch.cuda.Stream(device=device)
    try:
        pool.prepare_channels(("identity",))
        channel = pool.for_stream(stream, channel_id="identity")
        dist.barrier()
        for shape in SHAPES:
            for iteration in range(ITERS):
                generator = torch.Generator(device=device)
                generator.manual_seed(1000 * iteration + 17 * rank + shape[0])
                # Magnitudes spanning six decades make the FP32 sum order visible.
                magnitude = torch.pow(
                    10.0, torch.rand(shape, generator=generator, device=device) * 6 - 3
                )
                values = torch.randn(shape, generator=generator, device=device)
                inp = (values * magnitude).to(dtype)
                out = torch.empty_like(inp)
                with torch.cuda.stream(stream):
                    channel.all_reduce(inp, out=out)
                torch.cuda.synchronize(device)
                root = out.clone()
                dist.broadcast(root, src=0)
                assert torch.equal(out, root), (
                    f"rank {rank} differs from rank 0 for {dtype_name} {shape}"
                )
        dist.barrier()
    finally:
        try:
            pool.close()
        finally:
            dist.destroy_process_group()


@pytest.mark.parametrize("dtype_name", ["bfloat16", "float32"])
def test_pcie_oneshot_outputs_are_identical_on_every_rank(dtype_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(
            f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, _free_port(), dtype_name),
        nprocs=WORLD_SIZE,
        join=True,
    )
