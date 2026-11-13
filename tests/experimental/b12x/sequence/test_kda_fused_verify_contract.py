# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Host-side contracts for prepared KDA convolution and recovery launches."""

import gc
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch

from b12x.sequence.gdn_decode import _fused_verify as fused
from b12x.sequence.gdn_decode._preparation import _kda_strides
from b12x.sequence.gdn_decode._tuning import GdnQuery
from b12x.sequence.short_conv._tuning import ShortConvQuery


@pytest.fixture
def invocation(monkeypatch):
    # Compilation and pointer conversion are the CUDA boundary; validation runs
    # against real tensor metadata without launching a device program.
    tokens, heads, slots, window = 4, 1, 3, 4
    bf16 = lambda *shape: torch.empty(shape, dtype=torch.bfloat16)
    binding = SimpleNamespace(
        plan=object(),
        mixed_qkv=bf16(tokens, 384),
        raw_g=bf16(tokens, heads, 128),
        raw_beta=bf16(tokens, heads),
        z=bf16(tokens, heads, 128),
        A_log=torch.zeros(heads),
        dt_bias=torch.zeros(heads, 128),
        norm_weight=bf16(128),
        recurrent_state=torch.zeros(slots, heads, 128, 128),
        query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        num_accepted_tokens=torch.ones(1, dtype=torch.int32),
        state_indices=torch.ones((1, window), dtype=torch.int32),
        num_seqs=torch.ones(1, dtype=torch.int32),
        num_tokens=torch.tensor([tokens], dtype=torch.int32),
        output=bf16(tokens, heads, 128),
        correction_cache=torch.empty(slots, heads, window, 128),
        kg_cache=bf16(slots, heads, window, 256),
    )
    conv = SimpleNamespace(
        plan=object(),
        x=bf16(tokens, 384),
        conv_weight=bf16(384, 4),
        query_start_loc=binding.query_start_loc,
        state_slot_ids=torch.ones(1, dtype=torch.int64),
        state_is_fresh=torch.zeros(1, dtype=torch.bool),
        num_accepted_tokens=binding.num_accepted_tokens,
        num_seqs=binding.num_seqs,
        num_tokens=binding.num_tokens,
        conv_state=bf16(slots, 384, 6),
        out=binding.mixed_qkv,
        conv_bias=None,
    )
    query = GdnQuery(
        gate_activation="sigmoid",
        qk_l2norm=True,
        state_dtype="float32",
        key_heads=heads,
        value_heads=heads,
        max_seqs=1,
        max_tokens=tokens,
        max_state_slots=slots,
        state_index_columns=window,
        null_state_index=0,
        recover_speculative_state=True,
        kda_strides=_kda_strides(
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.dt_bias,
            binding.recurrent_state,
            binding.output,
        ),
    )
    cq = ShortConvQuery(
        mode="decode",
        channels=384,
        kernel_size=4,
        dilation=1,
        max_tokens=tokens,
        max_seqs=1,
        max_state_slots=slots,
        max_speculative_tokens=3,
    )
    state = SimpleNamespace(query=query, layout=SimpleNamespace(recurrent_block_v=32))
    conv_state = SimpleNamespace(query=cq)
    from b12x.preparation import types

    monkeypatch.setattr(
        types,
        "require_prepared",
        lambda plan, *args: state if plan is binding.plan else conv_state,
    )
    launches = []
    monkeypatch.setattr(
        fused,
        "b12x_compile",
        lambda *args, **kwargs: lambda *args: launches.append(args),
    )
    monkeypatch.setattr(fused, "_fake_pointer", lambda dtype: None)
    monkeypatch.setattr(fused, "_pointer", lambda tensor, dtype: tensor.data_ptr())
    monkeypatch.setattr(fused, "current_cuda_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return binding, conv, launches


def test_public_run_without_retained_examples_requires_dynamic_operands(invocation):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    assert prepared.binding is None and prepared.conv is None
    with pytest.raises(ValueError, match="run_dynamic"):
        fused.run_kda_fused_verify(prepared)
    prepared.run_dynamic(binding, conv, scale=128**-0.5, lower_bound=-5.0, eps=1e-6)
    assert len(launches) == 1


def test_public_dynamic_run_returns_caller_output(invocation):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    result = fused.run_kda_fused_verify(prepared, binding=binding, conv=conv)
    assert result is binding.output
    assert len(launches) == 1


@pytest.mark.parametrize("operand", ["binding", "conv"])
def test_public_dynamic_run_requires_both_operands(invocation, operand):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    with pytest.raises(ValueError, match="both"):
        fused.run_kda_fused_verify(
            prepared, **{operand: binding if operand == "binding" else conv}
        )
    assert not launches


def test_dynamic_preparation_releases_example_tensors(invocation):
    binding, conv, _ = invocation
    references = [weakref.ref(tensor) for tensor in fused._operands(binding, conv)]
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    # The fixture retains its namespaces; clearing their fields leaves compiled
    # metadata and closures as the only possible owners of example tensors.
    binding.__dict__.clear()
    conv.__dict__.clear()
    gc.collect()
    assert prepared.binding is None and prepared.conv is None
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "option,value",
    [
        ("scale", float("inf")),
        ("scale", float("nan")),
        ("scale", 0.0),
        ("eps", float("inf")),
        ("eps", 0.0),
        ("lower_bound", float("-inf")),
        ("lower_bound", 0.0),
    ],
)
def test_launch_rejects_invalid_numerical_options(invocation, dynamic, option, value):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(
        binding, conv, retain_examples=not dynamic
    )
    options = dict(scale=128**-0.5, lower_bound=-5.0, eps=1e-6)
    options[option] = value
    with pytest.raises(ValueError):
        if dynamic:
            prepared.run_dynamic(binding, conv, **options)
        else:
            fused.run_kda_fused_verify(prepared, **options)
    assert not launches


@pytest.mark.parametrize("operand", ["raw_g", "output", "conv_weight", "x"])
def test_dynamic_operands_preserve_compiled_dtype(invocation, operand):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    target = conv if operand in ("conv_weight", "x") else binding
    setattr(target, operand, getattr(target, operand).float())
    with pytest.raises((TypeError, ValueError)):
        prepared.run_dynamic(binding, conv, scale=128**-0.5, lower_bound=-5.0, eps=1e-6)
    assert not launches


@pytest.mark.parametrize("operand", ["raw_g", "output", "x", "conv_state"])
def test_dynamic_operands_preserve_compiled_strides(invocation, operand):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    target = conv if operand in ("x", "conv_state") else binding
    tensor = getattr(target, operand)
    storage = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype
    )
    setattr(target, operand, storage[..., ::2])
    with pytest.raises((TypeError, ValueError)):
        prepared.run_dynamic(binding, conv, scale=128**-0.5, lower_bound=-5.0, eps=1e-6)
    assert not launches


def test_dynamic_operands_accept_runtime_strides(invocation):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    binding.raw_beta = torch.empty((4, 2), dtype=torch.bfloat16)[:, ::2]
    binding.z = torch.empty((4, 2, 128), dtype=torch.bfloat16)[:, :1]
    for target, name in (
        (binding, "recurrent_state"),
        (binding, "correction_cache"),
        (binding, "kg_cache"),
        (binding, "state_indices"),
        (conv, "conv_state"),
    ):
        tensor = getattr(target, name)
        storage = torch.empty(
            (tensor.shape[0] * 2, *tensor.shape[1:]), dtype=tensor.dtype
        )
        setattr(target, name, storage[::2])
    prepared.run_dynamic(binding, conv, scale=128**-0.5, lower_bound=-5.0, eps=1e-6)
    assert len(launches) == 1


def test_dynamic_output_cannot_alias_a_later_padded_convolution_slot(invocation):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    storage = torch.empty(3 * 4608, dtype=torch.bfloat16)
    conv.conv_state = storage.as_strided((3, 384, 6), (4608, 6, 1))
    binding.output = storage[9216:9728].view(4, 1, 128)
    with pytest.raises(ValueError, match="overlap"):
        fused.run_kda_fused_verify(prepared, binding=binding, conv=conv)
    assert not launches


@pytest.mark.parametrize("convolution_slots", [3, 5])
def test_disjoint_convolution_and_recurrent_fields_share_paged_storage(
    invocation, convolution_slots
):
    binding, conv, launches = invocation
    row_bytes = 4608 + 65536
    storage = torch.empty((convolution_slots, row_bytes), dtype=torch.uint8)
    conv.conv_state = (
        storage[:, :4608].view(torch.bfloat16).view(convolution_slots, 384, 6)
    )
    binding.recurrent_state = (
        storage[:3, 4608:].view(torch.float32).view(3, 1, 128, 128)
    )
    from b12x.preparation.types import require_prepared

    prepared_conv = require_prepared(conv.plan, "sequence.short_conv")
    prepared_conv.query = replace(
        prepared_conv.query, max_state_slots=convolution_slots
    )
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    fused.run_kda_fused_verify(prepared, binding=binding, conv=conv)
    assert len(launches) == 1


@pytest.mark.parametrize("operand", ["conv_state", "correction_cache", "kg_cache"])
def test_dynamic_mutable_state_slots_must_not_overlap(invocation, operand):
    binding, conv, launches = invocation
    prepared = fused.prepare_kda_fused_verify(binding, conv, retain_examples=False)
    target = conv if operand == "conv_state" else binding
    tensor = getattr(target, operand)
    setattr(target, operand, tensor.as_strided(tensor.shape, (0, *tensor.stride()[1:])))
    with pytest.raises(ValueError):
        fused.run_kda_fused_verify(prepared, binding=binding, conv=conv)
    assert not launches


@pytest.mark.parametrize("operand", ["query_start_loc", "num_seqs", "num_tokens"])
def test_prepare_requires_shared_convolution_and_recovery_metadata(invocation, operand):
    binding, conv, launches = invocation
    setattr(conv, operand, getattr(conv, operand).clone())
    with pytest.raises(ValueError):
        fused.prepare_kda_fused_verify(binding, conv)
    assert not launches


@pytest.mark.parametrize("operand", ["raw_g", "output", "x"])
def test_prepare_rejects_strides_differing_from_the_selected_plans(
    invocation, monkeypatch, operand
):
    binding, conv, launches = invocation
    if operand == "x":
        conv.x = torch.empty((4, 768), dtype=torch.bfloat16)[:, :384]
    else:
        setattr(binding, operand, torch.empty((4, 2, 128), dtype=torch.bfloat16)[:, :1])

    def unexpected_compile(*args, **kwargs):
        pytest.fail("invalid preparation operands must be rejected before compilation")

    monkeypatch.setattr(fused, "b12x_compile", unexpected_compile)
    with pytest.raises(ValueError):
        fused.prepare_kda_fused_verify(binding, conv)
    assert not launches


@pytest.mark.parametrize(
    "operand,dtype",
    [
        ("A_log", torch.bfloat16),
        ("dt_bias", torch.bfloat16),
        ("state_indices", torch.int64),
    ],
)
def test_prepare_rejects_dtypes_differing_from_the_compile_key(
    invocation, monkeypatch, operand, dtype
):
    binding, conv, launches = invocation
    setattr(binding, operand, getattr(binding, operand).to(dtype))

    def unexpected_compile(*args, **kwargs):
        pytest.fail("pointer types must match the compile key before compilation")

    monkeypatch.setattr(fused, "b12x_compile", unexpected_compile)
    with pytest.raises((TypeError, ValueError)):
        fused.prepare_kda_fused_verify(binding, conv)
    assert not launches
