"""Tensor binding validation for prepared BF16 and FP8 two-shot collectives."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import _twoshot_preparation as preparation

_SURFACES = (
    ("PCIeTwoShotBF16.all_reduce", torch.bfloat16),
    ("PCIeTwoShotBF16.reduce_scatter", torch.bfloat16),
    ("PCIeTwoShotBF16.all_gather", torch.bfloat16),
    ("TwoShotReduceScatter.reduce_scatter_fp8", torch.float8_e4m3fn),
    ("TwoShotReduceScatter.all_gather_fp8", torch.float8_e4m3fn),
    ("TwoShotReduceScatter.reduce_scatter_fp8", torch.uint8),
    ("TwoShotReduceScatter.all_gather_fp8", torch.uint8),
)


def _bindings(surface, dtype, *, declare_output=True):
    payload = torch.empty((4, 16), dtype=dtype)
    output_rows = (
        2 if "reduce_scatter" in surface else 8 if "all_gather" in surface else 4
    )
    output = torch.empty((output_rows, 16), dtype=torch.bfloat16)
    scale = torch.empty(4, dtype=torch.float32) if surface.endswith("_fp8") else None
    launches = []
    runtime = SimpleNamespace(
        rank=0,
        world_size=2,
        row_elems=16,
        max_rows=8,
        device=torch.device("cpu"),
        _pack_stride=16,
        _scale_stride=16,
        _slot_bytes=2 << 20,
        _signal_ptrs=(100,) * 16,
        _staging_ptrs=((100 + (2 << 20),) * 16,) * 2,
        _launch_prepared=lambda *args, **kwargs: launches.append(args),
    )
    call = {"inp" if surface.endswith("all_reduce") else "payload": payload}
    if declare_output:
        call["out"] = output
    if scale is not None:
        call["scale"] = scale
    query = preparation.query_from_runtime(runtime, surface=surface, call=call)
    state = preparation._TwoShotExecutionState(query, runtime, {})
    return state, payload, scale, output, launches


@pytest.mark.parametrize("surface,dtype", _SURFACES)
@pytest.mark.parametrize("declare_output", (False, True))
def test_valid_prepared_bindings(surface, dtype, declare_output):
    state, payload, scale, output, launches = _bindings(
        surface, dtype, declare_output=declare_output
    )
    preparation.prepared_call(state, payload=payload, scale=scale, out=output).run()
    assert len(launches) == 1
    assert launches[0][0] is payload and launches[0][2] is output


@pytest.mark.parametrize("surface,dtype", _SURFACES)
@pytest.mark.parametrize("declare_output", (False, True))
@pytest.mark.parametrize("operand", ("payload", "output", "scale"))
@pytest.mark.parametrize("mismatch", ("dtype", "device"))
def test_prepared_rebinding_rejects_dtype_and_device(
    surface, dtype, declare_output, operand, mismatch
):
    state, payload, scale, output, launches = _bindings(
        surface, dtype, declare_output=declare_output
    )
    if operand == "scale" and scale is None:
        pytest.skip("BF16 transport has no scale binding")
    operands = {"payload": payload, "scale": scale, "output": output}
    original = operands[operand]
    wrong_dtype = torch.uint8 if original.dtype != torch.uint8 else torch.bfloat16
    operands[operand] = torch.empty(
        original.shape,
        dtype=wrong_dtype if mismatch == "dtype" else original.dtype,
        device="meta" if mismatch == "device" else original.device,
    )
    bound = preparation.prepared_call(
        state,
        payload=operands["payload"],
        scale=operands["scale"],
        out=operands["output"],
    )
    with pytest.raises(
        TypeError if mismatch == "dtype" else ValueError,
        match=f"{operand} {mismatch} differs",
    ):
        bound.run()
    assert not launches


@pytest.mark.parametrize("surface,dtype", _SURFACES)
@pytest.mark.parametrize("mismatch", ("dtype", "device"))
def test_declared_output_rejects_wrong_dtype_or_device(surface, dtype, mismatch):
    state, payload, scale, output, launches = _bindings(surface, dtype)
    bad_output = torch.empty(
        output.shape,
        dtype=torch.uint8 if mismatch == "dtype" else output.dtype,
        device="meta" if mismatch == "device" else output.device,
    )
    call = {
        "inp" if surface.endswith("all_reduce") else "payload": payload,
        "out": bad_output,
    }
    if scale is not None:
        call["scale"] = scale
    with pytest.raises(
        TypeError if mismatch == "dtype" else ValueError, match="output"
    ):
        preparation.query_from_runtime(state.runtime, surface=surface, call=call)
    assert not launches


@pytest.mark.parametrize("surface,dtype", _SURFACES)
@pytest.mark.parametrize("declare_output", (False, True))
@pytest.mark.parametrize("defect", ("short", "reshaped", "strided", "unaligned"))
def test_prepared_output_geometry(surface, dtype, declare_output, defect):
    state, payload, scale, output, launches = _bindings(
        surface, dtype, declare_output=declare_output
    )
    if defect == "short":
        bad = torch.empty(output.numel() - 8, dtype=output.dtype)
    elif defect == "reshaped":
        bad = torch.empty(output.numel(), dtype=output.dtype)
    elif defect == "strided":
        bad = torch.empty((*output.shape, 2), dtype=output.dtype)[..., 0]
    else:
        bad = torch.empty(output.numel() + 1, dtype=output.dtype)[1:].view_as(output)
    bound = preparation.prepared_call(state, payload=payload, scale=scale, out=bad)
    with pytest.raises(ValueError, match="output"):
        bound.run()
    assert not launches
    call = {"inp" if surface.endswith("all_reduce") else "payload": payload, "out": bad}
    if scale is not None:
        call["scale"] = scale
    with pytest.raises(ValueError, match="output"):
        preparation.query_from_runtime(state.runtime, surface=surface, call=call)


@pytest.mark.parametrize("surface,dtype", _SURFACES)
@pytest.mark.parametrize(
    "defect",
    (
        "payload_capacity",
        "pack_capacity",
        "payload_stride",
        "payload_alignment",
        "native_shape",
    ),
)
def test_declared_payload_geometry(surface, dtype, defect):
    state, payload, scale, output, launches = _bindings(surface, dtype)
    runtime = state.runtime
    if defect == "payload_capacity":
        runtime.max_rows = 1
    elif defect == "pack_capacity":
        runtime._pack_stride = 0
    elif defect == "payload_stride":
        payload = torch.empty((*payload.shape, 2), dtype=dtype)[..., 0]
    elif defect == "payload_alignment":
        payload = torch.empty(payload.numel() + 1, dtype=dtype)[1:].view_as(payload)
    else:
        if surface.endswith("all_reduce"):
            pytest.skip("BF16 allreduce accepts arbitrary contiguous tensor shapes")
        payload = payload.view(-1)
    call = {"inp" if surface.endswith("all_reduce") else "payload": payload}
    if scale is not None:
        call["scale"] = scale
    with pytest.raises(ValueError, match="payload|capacity"):
        preparation.query_from_runtime(runtime, surface=surface, call=call)
    assert not launches


@pytest.mark.parametrize("surface,dtype", _SURFACES[3:])
@pytest.mark.parametrize("defect", ("short", "strided", "capacity"))
def test_fp8_scale_geometry(surface, dtype, defect):
    state, payload, scale, output, launches = _bindings(surface, dtype)
    if defect == "short":
        scale = scale[:-1]
    elif defect == "strided":
        scale = torch.empty(8, dtype=torch.float32)[::2]
    else:
        state.runtime._scale_stride = 0
    with pytest.raises(ValueError, match="scale"):
        preparation.prepared_call(state, payload=payload, scale=scale, out=output).run()
    with pytest.raises(ValueError, match="scale"):
        preparation.query_from_runtime(
            state.runtime, surface=surface, call={"payload": payload, "scale": scale}
        )
    assert not launches


@pytest.mark.parametrize("surface,dtype", _SURFACES[3:])
def test_contiguous_fp8_scale_shape_remains_supported(surface, dtype):
    state, payload, scale, output, launches = _bindings(surface, dtype)
    scale = scale.view(2, 2)
    query = preparation.query_from_runtime(
        state.runtime, surface=surface, call={"payload": payload, "scale": scale}
    )
    state = preparation._TwoShotExecutionState(query, state.runtime, {})
    preparation.prepared_call(state, payload=payload, scale=scale, out=output).run()
    assert len(launches) == 1


@pytest.mark.parametrize("surface,dtype", _SURFACES[1:])
@pytest.mark.parametrize("declare_output", (False, True))
def test_empty_standalone_collective_bindings(surface, dtype, declare_output):
    state, payload, scale, output, launches = _bindings(surface, dtype)
    payload = payload[:0]
    output = output[:0]
    if scale is not None:
        scale = scale[:0]
    call = {"payload": payload}
    if declare_output:
        call["out"] = output
    if scale is not None:
        call["scale"] = scale
    query = preparation.query_from_runtime(state.runtime, surface=surface, call=call)
    state = preparation._TwoShotExecutionState(query, state.runtime, {})
    preparation.prepared_call(state, payload=payload, scale=scale, out=output).run()
    assert len(launches) == 1
