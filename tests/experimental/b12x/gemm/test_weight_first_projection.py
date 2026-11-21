# SPDX-License-Identifier: Apache-2.0

"""The split-K projection retains FP32 router bits across independent weight allocations."""

import pytest
import torch

from b12x.gemm import weight_first_projection as projection
from b12x.preparation import PreparationSession, PreparedCall


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows,second_rows", [(1, 512), (7, 501), (16, 512)])
def test_shared_tile_boundary_and_replay(rows, second_rows):
    device = torch.device("cuda", torch.cuda.current_device())
    query = projection.Query(rows, 6144, 256, second_rows, staging_depth=1, pdl=True)
    x = torch.zeros((rows, query.width), device=device, dtype=torch.bfloat16)
    a, b = (
        torch.zeros((count, query.width), device=device, dtype=x.dtype)
        for count in (query.first_rows, query.second_rows)
    )
    for matrix, sign in ((a, 1), (b, -1)):
        matrix[:, 0] = sign
        matrix[:, 769] = torch.arange(matrix.shape[0], device=device) % 13 + 1
    logit = torch.empty((rows, query.first_rows), dtype=torch.float32, device=device)
    shared = torch.empty((rows, query.second_rows), dtype=x.dtype, device=device)
    scratch = torch.empty(query.partial_shape, dtype=torch.float32, device=device)
    counters = torch.zeros(query.counter_shape, dtype=torch.int32, device=device)
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        plan = projection.plan(query)
        session.prepare(
            (
                plan.request(
                    name="two-projections",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            x, a, b, logit, shared, scratch, counters
                        ),
                        output=(logit, shared),
                    ),
                ),
            )
        )
        args = (x, a, b, logit, shared, scratch, counters, plan.handle)
        projection.project(*args)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            projection.project(*args)
        session.freeze()
        for sign in (1, -1, 1):
            x[:, 0] = torch.arange(1, rows + 1, device=device) * sign
            x[:, 769] = 2.0**-12
            if rows > 1:
                x[-1].zero_()
            logit.fill_(float("nan"))
            shared.fill_(float("nan"))
            graph.replay()
            expected_logits = (
                x[:, :1].float() + x[:, 769:770].float() * a[:, 769].float()
            )
            expected_shared = (
                -x[:, :1].float() + x[:, 769:770].float() * b[:, 769].float()
            )
            torch.testing.assert_close(logit, expected_logits, rtol=0, atol=0)
            torch.testing.assert_close(
                shared, expected_shared.bfloat16(), rtol=0, atol=0
            )
            assert bool((expected_logits != expected_logits.bfloat16().float()).any())
            assert int(counters.count_nonzero()) == 0


@pytest.mark.parametrize("rows", [0, 17])
def test_row_capacity(rows):
    with pytest.raises(ValueError, match="1–16 input rows"):
        projection.Query(rows, 6144, 256, 512, 1, True)
