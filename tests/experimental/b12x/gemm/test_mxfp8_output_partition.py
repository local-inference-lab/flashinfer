"""Logical output extents preserve full-N BF16 arithmetic with local MXFP8 weights."""

from contextlib import contextmanager
from dataclasses import replace
import json

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _preparation, _tuning
from b12x.gemm.blockscaled._a16 import _weight_parts
from b12x.preparation import (
    DeviceIdentity,
    FrozenMapping,
    PreparationSession,
    PreparedCall,
)
from b12x.preparation.session import _declaration_key


DEVICE = DeviceIdentity(
    vendor="nvidia", compute_capability=(12, 0), sm_count=188, product_name="gb202"
)


def _query(**changes):
    return replace(
        _tuning.BlockscaledQuery(
            recipe="mxfp8",
            num_tokens=256,
            in_features=512,
            padded_in_features=512,
            out_features=128,
            activation_mode="a16",
            output_mode="provided",
            workspace_form="provided",
            expected_m=256,
            output_partition=(1024, 128, 256),
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "partition",
    [
        (1024, 128),
        (1024, 128, True),
        (1024, "128", 256),
        (1024, -128, 0),
        (1024, 1024, 1152),
        (1024, 128, 255),
        (1023, 128, 256),
    ],
)
def test_output_partition_rejects_invalid_bounds(partition):
    with pytest.raises(ValueError, match="output partition"):
        _tuning.TUNING.validate_query(_query(output_partition=partition), DEVICE)


@pytest.mark.parametrize(
    "changes",
    [
        {"recipe": "nvfp4"},
        {"activation_mode": "auto"},
        {"num_tokens": 64, "expected_m": 64},
        {"expected_m": None},
        {"expected_m": 128},
        {"source_contiguous": False},
        {"source_aligned": False},
        {"output_mode": "functional"},
        {"workspace_form": "owned"},
        {"padded_in_features": 544},
    ],
)
def test_output_partition_rejects_unsupported_arithmetic_or_storage(changes):
    with pytest.raises(ValueError):
        _tuning.TUNING.validate_query(_query(**changes), DEVICE)


def test_output_partition_preserves_default_query_identity_and_freezes_geometry():
    plain = _query(output_partition=None)
    expected = {
        name: getattr(plain, name)
        for name in plain.__dataclass_fields__
        if name != "output_partition"
    }
    assert _tuning.TUNING.encode_query(plain) == expected
    assert (
        _preparation._decode_query(_preparation._compile_query_payload(plain)) == plain
    )
    declarations, compiled = [], []
    for partition in (None, (1024, 0, 128), (1024, 128, 256), (2048, 128, 256)):
        query = _query(output_partition=partition)
        _tuning.TUNING.validate_query(query, DEVICE)
        payload = _preparation._compile_query_payload(query)
        assert _preparation._decode_query(payload) == query
        restored = json.loads(json.dumps(FrozenMapping(payload).to_dict()))
        assert _preparation._decode_query(restored) == query
        declarations.append(_declaration_key(blockscaled.plan(query)))
        compiled.append(tuple(payload.items()))
    assert len(set(declarations)) == len(set(compiled)) == 4


def test_output_partition_freezes_deserialized_row_lists():
    rows = [1024, 128, 256]
    query = _query(output_partition=rows)
    rows[1] = 0
    assert query.output_partition == (1024, 128, 256)
    _tuning.TUNING.validate_query(query, DEVICE)


def test_output_partition_requires_expansion_and_accounts_for_both_buffers():
    query = _query()
    config = _tuning.TUNING.default_config(query, DEVICE)
    expected = 2 * 1024 * (512 + 256)
    assert _preparation._workspace_bytes(query, config, DEVICE) == expected
    _tuning.TUNING.validate_config(
        replace(query, workspace_nbytes=expected), config, DEVICE
    )
    with pytest.raises(ValueError, match="logical output extent"):
        _tuning.TUNING.validate_config(
            replace(query, workspace_nbytes=expected - 1), config, DEVICE
        )
    with pytest.raises(ValueError, match="expanded BF16"):
        _tuning.TUNING.validate_config(
            query, replace(config, a16_backend="packed"), DEVICE
        )
    with pytest.raises(ValueError, match="expanded BF16"):
        _tuning.TUNING.validate_config(
            replace(query, output_partition=(1 << 20, 128, 256)), config, DEVICE
        )
    choices = [value for _, value in _tuning.TUNING.choices(query, DEVICE)]
    assert choices and all(
        value.mode == "a16" and value.a16_backend == "auto" for value in choices
    )


@pytest.mark.parametrize("rows", [0, 64, 128, 512])
def test_partition_run_rejects_any_changed_m_before_cuda(rows):
    query = _query()
    state = _preparation._PackedExecutionState(
        query=query,
        config=_tuning.TUNING.default_config(query, DEVICE),
        device=torch.device("cpu"),
        programs={},
        dense=None,
        alpha_one=None,
        offsets=None,
        required_workspace=0,
    )
    with pytest.raises(ValueError, match="exact prepared M"):
        state.run(torch.empty((rows, 512), dtype=torch.bfloat16), None, None, None)


@contextmanager
def _prepared(source, packed, output, partition):
    query = blockscaled.query_from_call(
        source,
        packed,
        out=output,
        activation_mode="a16",
        expected_m=source.shape[0],
        output_partition=partition,
    )
    query = replace(query, workspace_form="provided")
    plan = blockscaled.plan(query)
    config = _tuning.TUNING.default_config(query, DEVICE)
    scratch = torch.empty(
        _preparation._workspace_bytes(query, config, DEVICE),
        dtype=torch.uint8,
        device=source.device,
    )
    values, scales, gain, _ = _weight_parts(packed)
    with PreparationSession(
        device=source.device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="partition",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            source, values, scales, gain, out=output, workspace=scratch
                        )
                    ),
                ),
            )
        )
        yield plan, scratch, session


def _weights(logical, columns, first, end, device):
    generator = torch.Generator(device=device).manual_seed(7739)
    values = torch.randn((logical, columns), device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(
        120,
        133,
        (logical, columns // 32),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    )
    decoded = (
        values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
    ).bfloat16()
    packed = blockscaled.pack_weight(
        values[first:end].clone(), scales[first:end].clone()
    )
    return packed, decoded


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an assigned CUDA GPU"
)
@pytest.mark.parametrize(
    "rows,logical,columns,first,end",
    [
        (128, 1024, 1536, 0, 128),
        (256, 1024, 4096, 384, 512),
        (1024, 1024, 4096, 896, 1024),
        (256, 7168, 4096, 2688, 3584),
        (1024, 7168, 4096, 6272, 7168),
        (257, 80, 128, 3, 11),
    ],
)
@torch.no_grad()
def test_partition_matches_full_n_and_replays_dirty_scratch(
    rows, logical, columns, first, end
):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    device = torch.device("cuda:0")
    packed, decoded = _weights(logical, columns, first, end, device)
    source = torch.randn((rows, columns), dtype=torch.bfloat16, device=device)
    output = torch.empty((rows, end - first), dtype=torch.bfloat16, device=device)
    with _prepared(source, packed, output, (logical, first, end)) as (
        plan,
        scratch,
        session,
    ):
        run = lambda: blockscaled.mm(
            source, packed, out=output, workspace=scratch, plan=plan
        )
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for scale in (1, -0.5, 2, -1):
            source.mul_(scale)
            reference = torch.mm(source, decoded.T)[:, first:end].contiguous()
            scratch.fill_(0x7F)
            output.fill_(torch.nan)
            allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
            assert torch.equal(output, reference)
            expanded = (
                scratch[: logical * columns * 2]
                .view(torch.bfloat16)
                .view(logical, columns)
            )
            assert not expanded[:first].count_nonzero()
            assert not expanded[end:].count_nonzero()
            assert torch.equal(expanded[first:end], decoded[first:end])
        graph.reset()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an assigned CUDA GPU"
)
@torch.no_grad()
def test_partition_rejects_live_aliases_and_layout_changes_before_writes():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 execution required")
    device = torch.device("cuda:0")
    packed, _ = _weights(1024, 512, 128, 256, device)
    source = torch.randn((256, 512), dtype=torch.bfloat16, device=device)
    output = torch.empty((256, 128), dtype=torch.bfloat16, device=device)
    with _prepared(source, packed, output, (1024, 128, 256)) as (plan, scratch, _):
        for selected_source, selected_output, selected_scratch in (
            (source, output, scratch[:-1]),
            (source, output, scratch.view(2, -1)),
            (source.T.contiguous().T, output, scratch),
            (source, output.T.contiguous().T, scratch),
            (
                source,
                scratch[: output.nbytes].view(torch.bfloat16).view_as(output),
                scratch,
            ),
            (
                scratch[: source.nbytes].view(torch.bfloat16).view_as(source),
                output,
                scratch,
            ),
            (source, source.view(-1)[: output.numel()].view_as(output), scratch),
        ):
            scratch.fill_(0xA5)
            with pytest.raises(ValueError):
                blockscaled.mm(
                    selected_source,
                    packed,
                    out=selected_output,
                    workspace=selected_scratch,
                    plan=plan,
                )
            assert torch.all(scratch == 0xA5)
