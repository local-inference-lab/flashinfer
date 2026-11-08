"""Output-sharded BF16 execution inherits a declared full-matrix reduction schedule."""

import pytest
import torch

from b12x.gemm import bf16_gemv
from b12x.gemm.bf16_gemv import partition
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an assigned CUDA GPU"
)
@pytest.mark.parametrize(
    "rows,backend", [(1, "simt"), (4, "torch"), (8, "tc"), (16, "torch"), (32, "torch")]
)
@torch.no_grad()
def test_partition_inherits_reference_and_replays_changing_operands(rows, backend):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 required")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(913)
    x = torch.randn(
        (rows, 1536), device=device, dtype=torch.bfloat16, generator=generator
    )
    full = torch.randn(
        (4096, 1536), device=device, dtype=torch.bfloat16, generator=generator
    )
    shard = full[1536:2048].clone()
    expected = torch.empty((rows, 4096), device=device, dtype=torch.bfloat16)
    output = torch.empty((rows, 512), device=device, dtype=torch.bfloat16)
    scratch = torch.empty(32 << 20, device=device, dtype=torch.uint8)
    parent = bf16_gemv.plan(
        bf16_gemv.query_from_call(x, full, out=expected),
        override=bf16_gemv.GemvConfig(backend=backend),
    )
    child = partition.plan(parent, 512, dependency="reference")
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                parent.request(
                    name="reference",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(x, full, out=expected)
                    ),
                ),
                child.request(
                    name="partition",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(x, shard, out=output, workspace=scratch)
                    ),
                ),
            )
        )
        run = lambda: partition.mm(x, shard, out=output, workspace=scratch, plan=child)
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for value in (1.0, -0.5, 2.0, -1.0):
            x.mul_(value)
            shard.copy_(full[1536:2048])
            bf16_gemv.mm(x, full, out=expected, plan=parent)
            scratch.fill_(0x7F)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.equal(
                output.view(torch.int16), expected[:, 1536:2048].view(torch.int16)
            )
        graph.reset()
