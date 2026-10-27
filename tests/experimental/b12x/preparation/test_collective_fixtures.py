"""Collective fixtures share immutable references and validate each candidate."""

from types import SimpleNamespace
import weakref

import pytest
import torch

from b12x.comm.pcie._benchmark import _DcpInputs, _ReductionInputs, _assert_output


def reduction(rank=0, *, world=2, dtype=torch.bfloat16, inplace=False, **fused):
    shape = (7, 24) if not fused else (3, 2, 24)
    call = {"dtype": str(dtype).removeprefix("torch.")}
    if fused:
        call.update(
            residual_dtype=call["dtype"],
            output_dtype=call["dtype"],
            weight_dtype=call["dtype"],
            normalize=True,
            input_addend=False,
            group_size=2,
            group_ids_dtype="int32",
            group_id_limit=1,
            reduction_dtype=None,
            zero_centered=False,
        )
        call.update(fused)
    query = SimpleNamespace(
        surface="OneshotAllReduce.all_reduce"
        if not fused
        else "OneshotAllReduce.all_reduce_fused_add_rms_norm",
        rank=rank,
        world_size=world,
        call=call,
        setup={"shape": shape, "logical_shape": (3, 24)},
    )
    return _ReductionInputs(
        SimpleNamespace(
            query=query, runtime=SimpleNamespace(device=torch.device("cpu"))
        ),
        inplace=inplace,
    )


def test_references_are_shared_but_writable_buffers_are_private_and_close_releases_them():
    first, second = reduction(), reduction(inplace=True)
    reference = weakref.ref(first.reference)
    assert first.reference is second.reference
    assert first.source.data_ptr() != second.source.data_ptr()
    seed = first.reference.seeds[0].clone()
    first.produce(0)
    first.source.zero_()
    torch.testing.assert_close(first.reference.seeds[0], seed, rtol=0, atol=0)
    first.close()
    assert reference() is second.reference
    second.close()
    assert reference() is None
    assert first.producers == second.producers == ()


@pytest.mark.parametrize("world", (2, 4))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("inplace", (False, True))
def test_plain_references_validate_all_producers_and_reject_corruption(
    world, dtype, inplace
):
    inputs = [
        reduction(rank, world=world, dtype=dtype, inplace=inplace)
        for rank in range(world)
    ]
    try:
        for step in range(3):
            for item in inputs:
                item.produce(step)
            expected = (
                torch.stack([item.source.float() for item in inputs]).sum(0).to(dtype)
            )
            for item in inputs:
                item.out.copy_(expected)
                item.validate()
                item.out[0, 0] = float("nan")
                with pytest.raises(AssertionError):
                    item.validate()
        assert inputs[0].reference is not inputs[1].reference
    finally:
        for item in inputs:
            item.close()


@pytest.mark.parametrize(
    "normalize,addend,limit,centered,reduction_dtype",
    (
        (False, False, 1, False, None),
        (True, True, 1, True, "bfloat16"),
        (True, False, 0, False, "float32"),
    ),
)
def test_fused_references_preserve_group_reduction_and_normalization(
    normalize, addend, limit, centered, reduction_dtype
):
    inputs = [
        reduction(
            rank,
            normalize=normalize,
            input_addend=addend,
            group_id_limit=limit,
            zero_centered=centered,
            reduction_dtype=reduction_dtype,
        )
        for rank in range(2)
    ]
    try:
        for step in range(3):
            for item in inputs:
                item.produce(step)
            partials = []
            for item in inputs:
                partial = (item.source.float() * 0.5).sum(1)
                if limit == 0:
                    partial.zero_()
                partial = partial.bfloat16().float()
                if addend:
                    partial = (partial + 0.125).bfloat16().float()
                partials.append(partial)
            total = torch.stack(partials).sum(0)
            if reduction_dtype is not None:
                total = total.to(getattr(torch, reduction_dtype)).float()
            residual = total + 0.25
            output = residual * torch.rsqrt(
                residual.square().mean(-1, keepdim=True) + 1e-6
            )
            if centered:
                output *= 2
            for item in inputs:
                item.residual.copy_(residual)
                if normalize:
                    item.out.copy_(output)
                item.validate()
                item.residual[0, 0] += 1
                with pytest.raises(AssertionError):
                    item.validate()
    finally:
        for item in inputs:
            item.close()


def test_fused_semantics_do_not_alias_reference_sets():
    first = reduction(normalize=True, zero_centered=False)
    second = reduction(normalize=True, zero_centered=True)
    try:
        assert first.reference is not second.reference
        torch.testing.assert_close(
            second.reference.expected[0], first.reference.expected[0] * 2
        )
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("lse_reduce", (False, True))
@pytest.mark.parametrize("natural_log", (False, True))
def test_dcp_reuses_reference_and_validates_strided_outputs(lse_reduce, natural_log):
    inputs = []
    try:
        for rank in range(2):
            call = dict(
                shape=(2, 4, 8),
                stride=(48, 8, 1),
                dtype="torch.bfloat16",
                out_shape=(2, 2 if lse_reduce else 8, 8),
                out_stride=(32 if lse_reduce else 80, 8, 1),
                lse_shape=(2, 4),
                lse_stride=(4, 1),
                natural_log=natural_log,
            )
            query = SimpleNamespace(
                surface="DcpAllToAll."
                + ("lse_reduce_scatter" if lse_reduce else "all_gather_heads"),
                world_size=2,
                rank=rank,
                call=call,
            )
            state = SimpleNamespace(
                query=query, runtime=SimpleNamespace(device=torch.device("cpu"))
            )
            inputs.append(_DcpInputs(state))
            other = _DcpInputs(state)
            assert other.reference is inputs[-1].reference
            assert other.source.data_ptr() != inputs[-1].source.data_ptr()
            other.close()
        for step in range(2):
            for item in inputs:
                item.produce(step)
            for rank, item in enumerate(inputs):
                if lse_reduce:
                    from b12x.comm.pcie.pcie_dcp_a2a import lse_reduce_scatter_reference

                    expected = lse_reduce_scatter_reference(
                        torch.stack([x.source for x in inputs]),
                        torch.stack([x.lse for x in inputs]),
                        rank,
                        is_lse_base_on_e=natural_log,
                    )
                else:
                    expected = torch.cat([x.source for x in inputs], dim=1)
                item.out.copy_(expected)
                item.validate()
                item.out[0, 0, 0] = float("nan")
                with pytest.raises(AssertionError):
                    item.validate()
        refs = [weakref.ref(item.reference) for item in inputs]
    finally:
        for item in inputs:
            item.close()
    assert all(ref() is None for ref in refs)


def test_exact_comparison_keeps_dtype_validation():
    with pytest.raises(AssertionError):
        _assert_output(torch.ones(4), torch.ones(4, dtype=torch.bfloat16))


def test_binding_failure_releases_shared_reference_without_garbage_collection():
    from b12x.comm.pcie._benchmark import _REFERENCES, benchmark_call

    def fail(source):
        raise ValueError("invalid binding")

    state = SimpleNamespace(
        query=SimpleNamespace(
            surface="OneshotAllReduce.all_reduce",
            world_size=2,
            rank=0,
            setup={"shape": (19, 24)},
            call={"dtype": "bfloat16"},
        ),
        runtime=SimpleNamespace(device=torch.device("cpu")),
        bind_input=fail,
    )
    before = set(_REFERENCES)
    with pytest.raises(ValueError, match="invalid binding"):
        benchmark_call(state)
    assert set(_REFERENCES) == before
