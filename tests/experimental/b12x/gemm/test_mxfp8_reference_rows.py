"""Enclosing row geometry is explicit and does not alter ordinary declarations."""

from dataclasses import replace
import json

import pytest
import torch

from b12x.gemm import blockscaled
from b12x.gemm.blockscaled import _preparation, _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.session import _declaration_key
from b12x.preparation import PreparationSession, PreparedCall
from b12x.gemm.blockscaled._a16 import _weight_parts

DEVICE = DeviceIdentity(
    vendor="nvidia", compute_capability=(12, 0), sm_count=188, product_name="gb202"
)


def query(**changes):
    return replace(
        _tuning.BlockscaledQuery(
            recipe="mxfp8",
            num_tokens=1024,
            in_features=7168,
            padded_in_features=7168,
            out_features=360,
            activation_mode="a16",
            output_mode="provided",
            workspace_form="provided",
            expected_m=1024,
            reference_rows=3072,
        ),
        **changes,
    )


@pytest.mark.parametrize(
    "changes",
    [
        dict(reference_rows=True),
        dict(reference_rows=1023),
        dict(reference_rows=-1),
        dict(reference_rows=1.5),
        dict(recipe="mxfp4"),
        dict(activation_mode="auto"),
        dict(expected_m=None),
        dict(expected_m=512),
        dict(num_tokens=64, expected_m=64),
        dict(workspace_form="owned"),
        dict(output_mode="functional"),
        dict(source_contiguous=False),
        dict(source_aligned=False),
        dict(padded_in_features=7200),
        dict(output_partition=(720, 0, 360)),
    ],
)
def test_invalid_reference_row_contract(changes):
    with pytest.raises(ValueError):
        _tuning.TUNING.validate_query(query(**changes), DEVICE)


def test_reference_rows_roundtrip_and_distinct_identity():
    keys = []
    for rows in (None, 1024, 2048, 3072, 4096):
        value = query(reference_rows=rows)
        _tuning.TUNING.validate_query(value, DEVICE)
        payload = _preparation._compile_query_payload(value)
        assert (
            _preparation._decode_query(
                json.loads(json.dumps(FrozenMapping(payload).to_dict()))
            )
            == value
        )
        assert ("reference_rows" in _tuning.TUNING.encode_query(value)) == (
            rows is not None
        )
        keys.append(_declaration_key(blockscaled.plan(value)))
    assert len(set(keys)) == len(keys)


def test_reference_rows_reserve_disjoint_gemm_workspace():
    value = query()
    config = _tuning.TUNING.default_config(value, DEVICE)
    baseline = _preparation._workspace_bytes(
        replace(value, reference_rows=None), config, DEVICE
    )
    needed = _preparation._workspace_bytes(value, config, DEVICE)
    assert needed == (baseline + 255) // 256 * 256 + (32 << 20)
    _tuning.TUNING.validate_config(
        replace(value, workspace_nbytes=needed), config, DEVICE
    )
    with pytest.raises(ValueError):
        _tuning.TUNING.validate_config(
            replace(value, workspace_nbytes=needed - 1), config, DEVICE
        )
    with pytest.raises(ValueError):
        _tuning.TUNING.validate_config(
            value, replace(config, a16_backend="packed"), DEVICE
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an assigned CUDA GPU"
)
@pytest.mark.parametrize(
    "logical_rows,rows,offset", [(1536, 1024, 0), (2048, 1024, 17), (3072, 1024, 2047)]
)
@torch.no_grad()
def test_reference_rows_preserve_cropped_outputs_under_graph_replay(
    logical_rows, rows, offset
):
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("SM120/SM121 required")
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(371)
    values = torch.randn((128, 512), device=device, generator=generator).to(
        torch.float8_e4m3fn
    )
    scales = torch.randint(
        120, 133, (128, 16), device=device, dtype=torch.uint8, generator=generator
    )
    decoded = (
        values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, 1)
    ).bfloat16()
    packed = blockscaled.pack_weight(values, scales)
    source = torch.randn(
        (logical_rows, 512), device=device, dtype=torch.bfloat16, generator=generator
    )
    # Keep the requested suffix's row placement instead of moving its first
    # arbitrary row to MMA row zero. Offset 17 exercises a cancellation case
    # that differs even with the same cuBLASLt algorithm when padding is omitted.
    origin = offset // 256 * 256
    capacity = rows + 256
    compact = torch.zeros((capacity, 512), device=device, dtype=torch.bfloat16)
    available = min(capacity, logical_rows - origin)
    compact[:available].copy_(source[origin : origin + available])
    output = torch.empty((capacity, 128), device=device, dtype=torch.bfloat16)
    query = blockscaled.query_from_call(
        compact,
        packed,
        out=output,
        activation_mode="a16",
        expected_m=capacity,
        reference_rows=logical_rows,
    )
    query = replace(query, workspace_form="provided", workspace_nbytes=None)
    plan = blockscaled.plan(query)
    config = _tuning.TUNING.default_config(query, DEVICE)
    scratch = torch.empty(
        _preparation._workspace_bytes(query, config, DEVICE),
        device=device,
        dtype=torch.uint8,
    )
    weight, scale, gain, _ = _weight_parts(packed)
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="cropped",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            compact, weight, scale, gain, out=output, workspace=scratch
                        )
                    ),
                ),
            )
        )
        run = lambda: blockscaled.mm(
            compact, packed, out=output, workspace=scratch, plan=plan
        )
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            run()
        session.freeze()
        for multiplier in (1, -0.5, 2, -1):
            source.mul_(multiplier)
            compact.zero_()
            compact[:available].copy_(source[origin : origin + available])
            expected = (source @ decoded.T)[offset : offset + rows]
            scratch.fill_(0x7F)
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            assert torch.equal(
                output[offset - origin : offset - origin + rows].view(torch.int16),
                expected.view(torch.int16),
            )
        graph.reset()
