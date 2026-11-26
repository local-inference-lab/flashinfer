import pytest
import torch

from b12x.comm.pcie.pcie_dma import _create_high_priority_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dedicated_dma_streams_do_not_alias_pooled_streams():
    device = torch.device("cuda:0")
    dedicated = [_create_high_priority_stream(device) for _ in range(4)]
    try:
        dedicated_handles = {s.cuda_stream for s in dedicated}
        assert len(dedicated_handles) == 4
        # Cycle well past PyTorch's 32-stream per-priority round-robin pool.
        pooled = [torch.cuda.Stream(device=device) for _ in range(64)]
        pooled_handles = {s.cuda_stream for s in pooled}
        assert dedicated_handles.isdisjoint(pooled_handles)
        # Verify stream sync / event interop with a pooled stream.
        ev = torch.cuda.Event()
        with torch.cuda.stream(pooled[0]):
            x = torch.ones(16, device=device)
            ev.record(pooled[0])
        with torch.cuda.stream(dedicated[0]):
            dedicated[0].wait_event(ev)
            y = x + 1
        pooled[0].wait_stream(dedicated[0])
        pooled[0].synchronize()
        assert torch.all(y == 2)
    finally:
        for s in dedicated:
            s._b12x_release()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dedicated_dma_streams_use_highest_priority():
    from cuda.bindings import driver as cuda

    device = torch.device("cuda:0")
    with torch.cuda.device(device):
        torch.cuda.current_stream(device)
        pstatus, _least, greatest = cuda.cuCtxGetStreamPriorityRange()
    assert pstatus == cuda.CUresult.CUDA_SUCCESS
    stream = _create_high_priority_stream(device)
    try:
        sstatus, prio = cuda.cuStreamGetPriority(stream.cuda_stream)
        assert sstatus == cuda.CUresult.CUDA_SUCCESS
        if greatest < 0:
            assert prio == greatest
    finally:
        stream._b12x_release()
