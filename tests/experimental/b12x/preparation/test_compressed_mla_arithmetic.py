"""Arithmetic constraints select legal MLA candidates without pinning kernel plans."""

from dataclasses import asdict, replace
from contextlib import nullcontext
import inspect
from types import SimpleNamespace

import pytest
import torch

from b12x.attention import compressed_sparse_mla as mla
from b12x.attention.compressed_sparse_mla._preparation import _caps
from b12x.attention.compressed_sparse_mla._tuning import SparseMlaQuery, TUNING
from b12x.preparation import DeviceIdentity, FrozenMapping
from b12x.preparation.session import PreparationJob, _declaration_key


DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


def declaration(*, heads=16, mode="decode", indexed=False, arithmetic=None):
    q = torch.empty((3, heads, 512), dtype=torch.bfloat16)
    cache = torch.empty((2, 64 * 528), dtype=torch.uint8)
    options = {} if arithmetic is None else {"compute_mode": arithmetic}
    caps = mla.Caps(
        device="cpu",
        num_q_heads=heads,
        max_q_rows=3,
        max_width=256 if indexed else 128,
        swa_width=128,
        indexed_width=128 if indexed else 0,
        cache_format="deepseek_v41",
        mode=mode,
        **options,
    )
    return mla.plan(
        caps,
        invocation=mla.invocation_from_tensors(
            q=q,
            swa_k_cache=cache,
            indexed_k_cache=cache if indexed else None,
            out=torch.empty_like(q),
        ),
    )


@pytest.mark.parametrize("heads", [1, 8, 16, 24, 32])
@pytest.mark.parametrize("mode", ["decode", "extend", "verify", "draft_extend"])
@pytest.mark.parametrize("indexed", [False, True])
def test_bf16_restricts_arithmetic_and_retains_other_tuning_domains(
    heads, mode, indexed
):
    auto = declaration(heads=heads, mode=mode, indexed=indexed)
    constrained = declaration(
        heads=heads, mode=mode, indexed=indexed, arithmetic="bf16"
    )
    assert auto.override is None and constrained.override is None
    auto_configuration = auto.contract.configure(auto.query, device=DEVICE)
    configuration = constrained.contract.configure(constrained.query, device=DEVICE)
    auto_choices = auto.contract.eligible_plan(auto.query, DEVICE).candidates
    choices = constrained.contract.eligible_plan(constrained.query, DEVICE).candidates
    assert {config for _, config in choices} == {
        config for _, config in auto_choices if config.v41_compute_mode == "bf16"
    }
    assert choices and configuration.default in {config for _, config in choices}
    assert configuration.default.v41_compute_mode == "bf16"
    assert configuration.default.v41_heads_per_block == 16
    auto_domains = {knob.name: knob.values for knob in auto_configuration.space.knobs}
    domains = {knob.name: knob.values for knob in configuration.space.knobs}
    assert domains.pop("v41_compute_mode") == ("bf16",)
    assert auto_domains.pop("v41_compute_mode") == ("fp8", "bf16")
    assert domains == auto_domains
    encoded = constrained.contract.encode_query(constrained.query)
    assert constrained.contract.query_schema_version == 26
    assert "compute_mode" in constrained.contract.query_fields
    assert "v41_compute_mode" not in encoded
    assert encoded["compute_mode"] == "bf16"
    assert SparseMlaQuery(**encoded) == constrained.query
    assert _declaration_key(auto) != _declaration_key(constrained)
    restored = _caps(constrained.query, configuration.default, 0)
    assert restored.compute_mode == "bf16"
    assert restored.kv_dtype == torch.uint8
    for field in (
        "kv_dtype",
        "swa_page_size",
        "indexed_page_size",
        "swa_cache_shape",
        "indexed_cache_shape",
    ):
        assert getattr(constrained.query, field) == getattr(auto.query, field)


@pytest.mark.parametrize("heads", [1, 8, 16, 24, 32])
@pytest.mark.parametrize("mode", ["decode", "extend", "verify", "draft_extend"])
def test_auto_retains_default_candidates_and_query_cache_identity(heads, mode):
    implicit = declaration(heads=heads, mode=mode)
    explicit = declaration(heads=heads, mode=mode, arithmetic="auto")
    assert implicit.contract is TUNING and explicit.contract is TUNING
    assert TUNING.query_schema_version == 25
    legacy_payload = asdict(implicit.query)
    del legacy_payload["compute_mode"]
    assert implicit.contract.encode_query(implicit.query) == legacy_payload
    legacy = replace(implicit, query=SparseMlaQuery(**legacy_payload))
    assert _declaration_key(implicit) == _declaration_key(legacy)
    assert implicit.contract.encode_query(
        implicit.query
    ) == explicit.contract.encode_query(explicit.query)
    defaults = [
        plan.contract.configure(plan.query, device=DEVICE).default
        for plan in (implicit, explicit)
    ]
    assert defaults[0] == defaults[1]
    assert defaults[0].v41_compute_mode == "fp8"
    assert defaults[0].v41_heads_per_block == (
        8 if mode == "decode" and heads % 16 else 16
    )
    configs = [
        config for _, config in TUNING.eligible_plan(implicit.query, DEVICE).candidates
    ]
    expected = (
        {("bf16", 16), ("fp8", 16)} if mode != "decode" else {("bf16", 16), ("fp8", 8)}
    )
    if mode == "decode" and heads % 16 == 0:
        expected.add(("fp8", 16))
    assert {
        (config.v41_compute_mode, config.v41_heads_per_block) for config in configs
    } == expected


@pytest.mark.parametrize("mode", ["decode", "extend", "verify", "draft_extend"])
def test_bf16_rejects_direct_or_cached_fp8_configuration(mode):
    auto = declaration(mode=mode)
    constrained = declaration(mode=mode, arithmetic="bf16")
    fp8 = auto.contract.configure(auto.query, device=DEVICE).default
    assignment = FrozenMapping(asdict(fp8))
    contract = constrained.contract
    configuration = contract.configure(constrained.query, device=DEVICE)
    with pytest.raises(ValueError, match="arithmetic constraint"):
        contract.configure(constrained.query, device=DEVICE, override=fp8)
    with pytest.raises(ValueError, match="arithmetic constraint"):
        contract.validate_config(constrained.query, fp8, DEVICE)
    with pytest.raises(ValueError):
        configuration.space.validate(assignment)
    with pytest.raises(ValueError):
        contract.lower(constrained.query, DEVICE, assignment)
    # Cached assignments also pass through lowering after parameter-space validation.
    with pytest.raises(ValueError, match="arithmetic constraint"):
        contract._lower(constrained.query, DEVICE, assignment)


def test_compile_payload_rejects_fp8_before_lowering(monkeypatch):
    from b12x.attention.compressed_sparse_mla import _preparation

    auto = declaration()
    constrained = declaration(arithmetic="bf16")
    fp8 = auto.contract.configure(auto.query, device=DEVICE).default
    monkeypatch.setattr(
        _preparation,
        "_lower_launch",
        lambda *_: pytest.fail("invalid config reached lowering"),
    )
    compile_payload = inspect.unwrap(_preparation.compile_compressed_sparse_mla)
    with pytest.raises(ValueError, match="arithmetic constraint"):
        compile_payload(constrained.query.to_dict(), asdict(fp8), 0)


@pytest.mark.parametrize("arithmetic", ["fp8", "float32", None, 1])
def test_caps_rejects_unsupported_arithmetic_constraints(arithmetic):
    with pytest.raises(ValueError, match="'auto' or 'bf16'"):
        mla.Caps(
            device="cpu",
            num_q_heads=16,
            max_q_rows=3,
            max_width=128,
            cache_format="deepseek_v41",
            compute_mode=arithmetic,
        )


def test_bf16_constraint_requires_mixed_v41_cache_format():
    with pytest.raises(ValueError, match="requires cache_format='deepseek_v41'"):
        mla.Caps(
            device="cpu",
            num_q_heads=16,
            max_q_rows=3,
            max_width=128,
            cache_format="deepseek_v4",
            compute_mode="bf16",
        )


def test_bf16_qk_accumulator_candidates_preserve_arithmetic_constraint():
    declaration_plan = declaration(arithmetic="bf16", indexed=True)
    configs = [
        config
        for _, config in declaration_plan.contract.eligible_plan(
            declaration_plan.query, DEVICE
        ).candidates
    ]
    assert {config.v41_qk_accumulators for config in configs} == {1, 2}
    assert all(config.v41_compute_mode == "bf16" for config in configs)
    default = declaration_plan.contract.configure(
        declaration_plan.query, device=DEVICE
    ).default
    assert default.v41_qk_accumulators == 1
    assert (
        declaration_plan.contract.decode_config(FrozenMapping(asdict(default)))
        == default
    )
    fp8_plan = declaration()
    fp8_default = fp8_plan.contract.configure(fp8_plan.query, device=DEVICE).default
    with pytest.raises(ValueError, match="parallel QK accumulation"):
        fp8_plan.contract.validate_config(
            fp8_plan.query, replace(fp8_default, v41_qk_accumulators=2), DEVICE
        )


def test_cluster_candidates_preserve_bf16_and_output_constraints():
    base = declaration(arithmetic="bf16", indexed=True)
    q = replace(base.query, indexed_width=512)
    configuration = base.contract.configure(q, device=DEVICE)
    assert configuration.default.v41_merge == "separate"
    configs = [
        config for _, config in base.contract.eligible_plan(q, DEVICE).candidates
    ]
    clustered = [config for config in configs if config.v41_merge == "cluster8"]
    assert {config.v41_qk_accumulators for config in clustered} == {1, 2}
    assert all(config.v41_compute_mode == "bf16" for config in clustered)
    for unsupported in (
        replace(q, return_lse=True),
        replace(q, num_q_heads=24),
        replace(q, indexed_cache_present=False),
        replace(q, mode="extend"),
    ):
        with pytest.raises(ValueError, match="cluster merge|config path"):
            base.contract.validate_config(unsupported, clustered[0], DEVICE)
