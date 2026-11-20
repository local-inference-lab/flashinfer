"""NVFP4 gated FC1 with an intermediate width that is not a multiple of 128.

The gate half of the joint FC1 block-scale plane then starts inside a 128-row
scale atom. For 32-aligned widths the swapped FC1 reads it from the joint plane,
so a launch builds no padded per-half planes. The result must equal the same
experts stored at the next multiple of 128 with zero rows and columns appended,
which run the unswapped FC1 on atom-aligned halves. One route per token keeps
the comparison free of the route sum's accumulation order; the order of FC2's
intermediate slices remains, so the comparison is bounded by the difference
between two launches of one owner.
"""

from __future__ import annotations

import pytest
import torch

from b12x._lib.intrinsics.reference import swizzle_block_scale
from b12x.moe import fused_moe as moe
from b12x.moe.fused_moe import _impl
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x

EXPERTS, HIDDEN, TOKENS = 12, 512, 96


def _prepare(intermediate, w13, s13, w2, s2):
    plan = moe.plan_weights(
        source=moe.PackedSource(
            format=moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=moe.W13Layout("w13"),
        ),
        activation=moe.ActivationSpec(
            mode=moe.ActivationMode.A4, nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(
            num_experts=EXPERTS, hidden_size=HIDDEN, intermediate_size=intermediate
        ),
    )
    return moe.prepare_weights(
        plan=plan,
        weights=moe.PackedWeights(
            w13=w13,
            w2=w2,
            w13_block_scales=swizzle_block_scale(s13),
            w2_block_scales=swizzle_block_scale(s2),
            w13_global_scales=torch.linspace(0.03, 0.05, EXPERTS, device="cuda"),
            w2_global_scales=torch.linspace(0.04, 0.07, EXPERTS, device="cuda"),
            input_scale=torch.ones(EXPERTS, device="cuda"),
            intermediate_scale=torch.ones(EXPERTS, device="cuda"),
            immutable_input_scales=True,
        ),
    )


def _run(prepared, x, ids, weights):
    output = torch.empty((TOKENS, HIDDEN), device="cuda", dtype=torch.bfloat16)
    declaration = moe.plan_execution(
        experts=prepared,
        capacity=moe.ExecutionCapacity(
            max_tokens=TOKENS, top_k=1, output_layout="token"
        ),
        invocation=FrozenMapping({}),
    )
    retained = []

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=s.device)
            for s in state.scratch.scratch_specs()
        )
        binding = state.bind(
            scratch=scratch,
            a=x,
            topk_ids=ids,
            topk_weights=weights,
            output=output,
            input_scales_static=True,
        )
        retained[:] = [binding, scratch]
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    with PreparationSession(
        device="cuda", autotune=False, compile_workers=0
    ) as session:
        session.prepare((declaration.request(name="experts", prepare_call=prepare),))
        launches = []
        for _ in range(2):
            retained[0].run()
            torch.cuda.synchronize()
            launches.append(output.clone())
    return launches


@pytest.mark.parametrize("intermediate", [96, 160, 320, 576])
def test_gate_scales_read_from_the_joint_plane_match_atom_aligned_halves(
    intermediate, monkeypatch
):
    require_b12x()
    torch.manual_seed(31 + intermediate)
    padded = -(-intermediate // 128) * 128
    w13 = torch.randint(
        0,
        256,
        (EXPERTS, 2 * intermediate, HIDDEN // 2),
        device="cuda",
        dtype=torch.uint8,
    )
    w2 = torch.randint(
        0, 256, (EXPERTS, HIDDEN, intermediate // 2), device="cuda", dtype=torch.uint8
    )

    def scales(*shape):
        # Distinct scales per row and block expose a read of the wrong half or atom.
        return (0.0625 + 0.375 * torch.rand(shape, device="cuda")).to(
            torch.float8_e4m3fn
        )

    s13 = scales(EXPERTS, 2 * intermediate, HIDDEN // 16)
    s2 = scales(EXPERTS, HIDDEN, intermediate // 16)
    wide13 = torch.zeros(
        (EXPERTS, 2 * padded, HIDDEN // 2), device="cuda", dtype=torch.uint8
    )
    wide_s13 = torch.ones((EXPERTS, 2 * padded, HIDDEN // 16), device="cuda").to(
        torch.float8_e4m3fn
    )
    for half in range(2):
        source = slice(half * intermediate, (half + 1) * intermediate)
        target = slice(half * padded, half * padded + intermediate)
        wide13[:, target] = w13[:, source]
        wide_s13[:, target] = s13[:, source]
    wide2 = torch.zeros(
        (EXPERTS, HIDDEN, padded // 2), device="cuda", dtype=torch.uint8
    )
    wide2[:, :, : intermediate // 2] = w2
    wide_s2 = torch.ones((EXPERTS, HIDDEN, padded // 16), device="cuda").to(
        torch.float8_e4m3fn
    )
    wide_s2[:, :, : intermediate // 16] = s2

    x = (torch.randn(TOKENS, HIDDEN, device="cuda") * 0.25).bfloat16()
    ids = torch.randint(0, EXPERTS, (TOKENS, 1), device="cuda", dtype=torch.int32)
    weights = torch.rand(TOKENS, 1, device="cuda") + 0.5

    split_calls = []
    split = _impl._split_gated_nvfp4_scale_views
    monkeypatch.setattr(
        _impl,
        "_split_gated_nvfp4_scale_views",
        lambda *args, **kwargs: split_calls.append(args[1]) or split(*args, **kwargs),
    )
    narrow, narrow_again = _run(
        _prepare(intermediate, w13, s13, w2, s2), x, ids, weights
    )
    assert not split_calls, "a 32-aligned width must not rebuild per-half scale planes"
    wide, wide_again = _run(
        _prepare(padded, wide13, wide_s13, wide2, wide_s2), x, ids, weights
    )
    assert torch.isfinite(narrow).all() and narrow.float().abs().max() > 0

    def distance(left, right):
        return float(
            torch.linalg.vector_norm(left.float() - right.float())
            / torch.linalg.vector_norm(right.float())
        )

    # FC2 adds one partial per 128-column intermediate slice in launch order, so two
    # launches of one owner differ; the two owners may differ by no more than that.
    repeat = max(distance(narrow, narrow_again), distance(wide, wide_again))
    assert distance(narrow, wide) <= 3 * repeat, (distance(narrow, wide), repeat)
