"""HyperConnection search effort preserves launch legality and native geometry."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.norm.hyperconnection import _preparation, _tuning
from b12x.preparation import FrozenMapping


def query(operation="grouped_rmsnorm", hidden=5120, effort="default", **kwargs):
    return _tuning.HyperConnectionQuery(
        dtype="bfloat16",
        max_tokens=128,
        hidden_size=hidden,
        streams=kwargs.pop("streams", 4),
        lowrank=320,
        operation=operation,
        autotune_effort=effort,
        **kwargs,
    )


def assignment(config):
    return {key: value for key, value in config.to_dict().items() if key != "backend"}


@pytest.mark.parametrize("effort,floor", [("default", 32), ("audit", 8)])
def test_gate_mean_efficiency_floor(effort, floor):
    q = query("gate_mean", hidden=2560, effort=effort)
    space = _tuning.TUNING.parameter_space(q, None)
    blocks = {choice["pointwise_block"] for choice in space.configurations()}
    assert blocks == {1 << bit for bit in range(floor.bit_length() - 1, 13)}
    for knob in space.knobs:
        if knob.name == "pointwise_block":
            assert knob.values == tuple(1 << bit for bit in range(13))


@pytest.mark.parametrize("effort", ["off", "default", "audit"])
@pytest.mark.parametrize("hidden", [1, 8, 128, 512, 1280, 2560, 5120])
@pytest.mark.parametrize(
    "operation,options",
    [
        ("grouped_rmsnorm", {}),
        ("grouped_rmsnorm", {"zero_centered": False}),
        ("grouped_rmsnorm", {"zero_centered": False, "add_residual": True}),
        ("grouped_rmsnorm", {"zero_centered": False, "paired": True, "streams": 2}),
        ("grouped_rmsnorm", {"zero_centered": False, "output_dtype": "float32"}),
        ("gate_mean", {}),
        ("engram_mix", {}),
        ("swiglu", {}),
    ],
)
def test_default_naturally_belongs_to_space(effort, hidden, operation, options):
    q = query(operation, hidden, effort, **options)
    default = _tuning.TUNING.default_config(q, None)
    space = _tuning.TUNING.parameter_space(q, None)
    space.validate(assignment(default))
    assert assignment(default) in [dict(choice) for choice in space.configurations()]


@pytest.mark.parametrize(
    "options",
    [
        {"zero_centered": False, "add_residual": True},
        {"zero_centered": False, "paired": True, "streams": 2},
        {"zero_centered": False, "output_dtype": "float32"},
    ],
)
def test_noncentered_triton_reductions_expose_geometry(options):
    q = query(hidden=512, **options)
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert {choice["reduction_block_h"] for choice in configurations} == {512, 1024}
    assert {choice["reduction_num_warps"] for choice in configurations} == {1, 2, 4, 8}


def test_fp32_input_reduction_exposes_geometry():
    q = replace(query(hidden=512, zero_centered=False), dtype="float32")
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert len(configurations) == 8


@pytest.mark.parametrize("operation", ["grouped_rmsnorm", "engram_mix", "add"])
def test_native_geometry_is_canonical(operation):
    q = (
        query(operation, zero_centered=False)
        if operation == "grouped_rmsnorm"
        else query(operation)
    )
    configurations = [
        dict(choice)
        for choice in _tuning.TUNING.parameter_space(q, None).configurations()
    ]
    assert configurations == [assignment(_tuning.TUNING.default_config(q, None))]


@pytest.mark.parametrize("effort", ["default", "audit"])
def test_effort_keeps_launch_legality(effort):
    q = query(hidden=512, effort=effort)
    default = _tuning.TUNING.default_config(q, None)
    for changes in (
        {"reduction_block_h": 256},
        {"reduction_num_warps": 3},
        {"pointwise_block": 3},
    ):
        with pytest.raises(ValueError):
            _tuning.TUNING.validate_config(q, replace(default, **changes), None)


def test_effort_captured_but_excluded_from_program_key(monkeypatch):
    from b12x.norm import hyperconnection

    caps = hyperconnection.Caps(device="cpu", max_tokens=128, hidden_size=5120)
    plans = []
    for effort in ("default", "audit"):
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", effort)
        plans.append(
            hyperconnection.plan(
                caps, invocation=FrozenMapping({"operation": "gate_mean"})
            )
        )
    assert [plan.query.autotune_effort for plan in plans] == ["default", "audit"]
    payloads = [_tuning.TUNING.encode_query(plan.query) for plan in plans]
    assert payloads[0] != payloads[1]
    jobs = [
        plan._compile_jobs(
            _tuning.TUNING.default_config(plan.query, None), SimpleNamespace(ordinal=0)
        )
        for plan in plans
    ]
    assert jobs[0] == jobs[1]
    assert jobs[0][0].factory == (
        "b12x.norm.hyperconnection._preparation:_compile_hyperconnection"
    )
    assert "autotune_effort" not in jobs[0][0].args[0]
    monkeypatch.setattr(_preparation, "_compile_hyperconnection", lambda *args: args)
    keys = [
        _preparation.compile_hyperconnection(payload, {}, 0) for payload in payloads
    ]
    assert keys[0] == keys[1]
    assert "autotune_effort" not in keys[0][0]


def test_off_iterates_only_default():
    configuration = _tuning.TUNING.configure(
        query("gate_mean", effort="off"), device=None
    )
    assert [config for _, config in _tuning.TUNING.iterate(configuration)] == [
        configuration.default
    ]


def test_explicit_effort_overrides_environment(monkeypatch):
    from b12x.norm import hyperconnection
    from b12x.preparation import autotune_effort

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    caps = hyperconnection.Caps(device="cpu", max_tokens=128, hidden_size=5120)
    with autotune_effort("default"):
        plan = hyperconnection.plan(caps, invocation={"operation": "gate_mean"})
    assert plan.query.autotune_effort == "default"
    plan = hyperconnection.plan(caps, invocation={"operation": "gate_mean"})
    assert plan.query.autotune_effort == "audit"


@pytest.mark.parametrize(
    "rows,hidden,warps",
    [(128, 2560, {8, 10, 16}), (129, 2560, {8}), (128, 2559, {8}), (128, 2561, {8})],
)
def test_combine_norm_cta_candidates_require_packed_width(rows, hidden, warps):
    q = replace(query("combine_norm", hidden=hidden), max_tokens=rows)
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert {choice["reduction_num_warps"] for choice in configurations} == warps
    default = _tuning.TUNING.default_config(q, None)
    if hidden % 8:
        for larger in (10, 16):
            with pytest.raises(ValueError, match="reduction_num_warps"):
                _tuning.TUNING.validate_config(
                    q, replace(default, reduction_num_warps=larger), None
                )


@pytest.mark.parametrize("effort", ["default", "audit"])
@pytest.mark.parametrize("hidden", [8, 768, 12288, 24576])
def test_swiglu_races_scalar_against_eight_element_vectors(effort, hidden):
    q = query("swiglu", hidden=hidden, effort=effort)
    configurations = [
        dict(choice)
        for choice in _tuning.TUNING.parameter_space(q, None).configurations()
    ]
    assert {choice["pointwise_vector"] for choice in configurations} == {1, 8}
    assert len(configurations) == 2
    default = _tuning.TUNING.default_config(q, None)
    assert default.pointwise_vector == 1
    _tuning.TUNING.validate_config(q, replace(default, pointwise_vector=8), None)


@pytest.mark.parametrize("hidden", [1, 4, 12, 2052, 12289])
def test_swiglu_width_not_divisible_by_eight_is_scalar_only(hidden):
    q = query("swiglu", hidden=hidden)
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert {choice["pointwise_vector"] for choice in configurations} == {1}
    default = _tuning.TUNING.default_config(q, None)
    assert default.pointwise_vector == 1
    with pytest.raises(ValueError, match="pointwise_vector"):
        _tuning.TUNING.validate_config(q, replace(default, pointwise_vector=8), None)


@pytest.mark.parametrize("operation", ["relu2", "add", "sigmoid", "head_gate"])
def test_other_pointwise_operations_have_no_vector_candidates(operation):
    options = {"gate_stride": 4, "streams": 4} if operation == "head_gate" else {}
    q = query(operation, hidden=2048, **options)
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert {choice["pointwise_vector"] for choice in configurations} == {1}
    default = _tuning.TUNING.default_config(q, None)
    with pytest.raises(ValueError, match="pointwise_vector"):
        _tuning.TUNING.validate_config(q, replace(default, pointwise_vector=8), None)


def test_swiglu_vector_requires_bfloat16_operands():
    q = query("swiglu", hidden=2048)
    q = replace(q, output_dtype="float32")
    # The query validator rejects non-BF16 SwiGLU before a vector config exists.
    with pytest.raises(ValueError):
        _tuning.TUNING.validate_query(q, None)
    assert not _tuning.supports_vector_swiglu(q)


def test_off_effort_selects_the_scalar_default():
    q = query("swiglu", hidden=2048, effort="off")
    configuration = _tuning.TUNING.configure(q, device=None)
    assert [
        config.pointwise_vector for _, config in _tuning.TUNING.iterate(configuration)
    ] == [1]


def test_config_payload_carries_the_vector_width():
    q = query("swiglu", hidden=2048)
    config = _tuning.TUNING.default_config(q, None)
    payload = _tuning.TUNING.config_payload(config)
    assert payload["pointwise_vector"] == 1
    vector = replace(config, pointwise_vector=8)
    assert _tuning.TUNING.decode_config(_tuning.TUNING.config_payload(vector)) == vector
    assert _tuning.TUNING.decode_config(payload) == config
    without_vector = FrozenMapping(
        {key: value for key, value in payload.items() if key != "pointwise_vector"}
    )
    with pytest.raises(ValueError, match="config fields"):
        _tuning.TUNING.decode_config(without_vector)


def test_vector_swiglu_candidates_do_not_depend_on_the_device():
    from b12x.preparation import DeviceIdentity

    q = query("swiglu", hidden=2048)
    devices = (
        None,
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(10, 3),
            sm_count=152,
            product_name="GB300",
        ),
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 0),
            sm_count=188,
            product_name="RTX PRO 6000",
        ),
        DeviceIdentity(
            vendor="nvidia",
            compute_capability=(12, 1),
            sm_count=48,
            product_name="GB10",
        ),
    )
    for device in devices:
        space = _tuning.TUNING.parameter_space(q, device)
        assert {c["pointwise_vector"] for c in space.configurations()} == {1, 8}
        assert _tuning.TUNING.default_config(q, device).pointwise_vector == 1


@pytest.mark.parametrize("operand", ["left_alignment", "output_alignment"])
@pytest.mark.parametrize("alignment", [1, 2, 4, 8])
def test_misaligned_swiglu_operand_removes_the_vector_candidate(operand, alignment):
    q = query("swiglu", hidden=2048, **{operand: alignment})
    configurations = list(_tuning.TUNING.parameter_space(q, None).configurations())
    assert {choice["pointwise_vector"] for choice in configurations} == {1}
    default = _tuning.TUNING.default_config(q, None)
    assert default.pointwise_vector == 1
    with pytest.raises(ValueError, match="pointwise_vector"):
        _tuning.TUNING.validate_config(q, replace(default, pointwise_vector=8), None)
    assert not _tuning.supports_vector_swiglu(q)


def test_aligned_swiglu_operands_keep_the_vector_candidate():
    q = query("swiglu", hidden=2048, left_alignment=16, output_alignment=16)
    assert _tuning.supports_vector_swiglu(q)
    assert _tuning.supports_vector_swiglu(query("swiglu", hidden=2048))


@pytest.mark.parametrize("alignment", [0, 3, 32, True])
def test_operand_alignment_must_be_a_supported_byte_count(alignment):
    q = query("swiglu", hidden=2048, left_alignment=alignment)
    with pytest.raises(ValueError, match="alignments"):
        _tuning.TUNING.validate_query(q, None)


def test_alignments_select_candidates_but_not_the_compiled_program():
    aligned = query("swiglu", hidden=2048)
    misaligned = query("swiglu", hidden=2048, left_alignment=2)
    assert _tuning.TUNING.encode_query(aligned) != _tuning.TUNING.encode_query(
        misaligned
    )
    assert _preparation._program_query_payload(
        _tuning.TUNING.encode_query(aligned)
    ) == _preparation._program_query_payload(_tuning.TUNING.encode_query(misaligned))
