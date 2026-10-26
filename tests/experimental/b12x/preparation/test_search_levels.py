"""Host-only effort tiers preserve legality and widen candidate sets monotonically."""

from dataclasses import replace

import pytest

from b12x.preparation import DeviceIdentity
from b12x.preparation import _efficiency as efficiency, tuning
from b12x.gemm._tuning import DenseGemmQuery, DenseGemmConfig, TUNING

DEVICE = DeviceIdentity(
    vendor="nvidia",
    compute_capability=(12, 0),
    sm_count=188,
    product_name="RTX PRO 6000 Blackwell",
)


def test_environment_precedence_and_validation(monkeypatch):
    monkeypatch.delenv("B12X_AUTOTUNE_EFFORT", raising=False)
    monkeypatch.setenv("B12X_AUTOTUNE_EXHAUSTIVE", "1")
    assert efficiency.capture_autotune_effort() == "default"
    for level in ("off", "default", "audit"):
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", level)
        assert efficiency.capture_autotune_effort() == level
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "invalid")
    with pytest.raises(ValueError, match="B12X_AUTOTUNE_EFFORT"):
        efficiency.capture_autotune_effort()


def test_audit_lifts_efficiency_predicates_and_keeps_mandatory_ones():
    space = tuning.ParameterSpace.create(
        (tuning.Knob(name="tile", values=(8, 16, 32, 64)),),
        predicates=(lambda p: p["tile"] != 8, lambda p: p["tile"] <= 32),
        efficiency_predicates=(lambda p: p["tile"] <= 16,),
    )
    assert space.autotune_effort == "default"
    for level, expected in (("default", [16]), ("audit", [16, 32])):
        active = replace(space, autotune_effort=level)
        assert [p["tile"] for p in active.configurations()] == expected
        for tile in expected:
            active.validate({"tile": tile})
        for tile in (8, 64):
            with pytest.raises(ValueError, match="correctness"):
                active.validate({"tile": tile})
    with pytest.raises(ValueError, match="efficiency"):
        space.validate({"tile": 32})
    with pytest.raises(ValueError):
        replace(space, autotune_effort="invalid")


def dense_query(recipe, m, n, k, effort):
    return DenseGemmQuery(
        recipe=recipe,
        entry_point="gemm.mm",
        weight_storage="native",
        output_dtype="bfloat16",
        batch=1,
        max_rows=m,
        in_features=k,
        out_features=n,
        output_mode="provided",
        expected_m=m,
        autotune_effort=effort,
    )


def dense_space(recipe, m, n, k, level):
    return TUNING.parameter_space(dense_query(recipe, m, n, k, level), DEVICE)


@pytest.mark.parametrize(
    "m,n,k,changes",
    (
        (8, 32768, 1280, dict(tile_m=64, tile_n=16, swap_ab=True)),
        (8, 1792, 5120, dict(tile_m=128)),
        (256, 32768, 1280, dict(tile_m=16)),
    ),
)
def test_audit_admits_mxfp8_geometry(m, n, k, changes):
    choice = (
        dict(
            backend="cutedsl",
            programmatic_serialization=False,
            evict_first=False,
            tile_m=64,
            tile_n=64,
            tile_k=128,
            load_path="tma",
            swap_ab=False,
            split_k_slices=1,
            large_m_unroll=False,
            target_occupancy=None,
        )
        | changes
    )
    with pytest.raises(ValueError, match="efficiency"):
        dense_space("mxfp8", m, n, k, "default").validate(choice)
    dense_space("mxfp8", m, n, k, "audit").validate(choice)
    assert TUNING.lower(
        dense_query("mxfp8", m, n, k, "audit"), DEVICE, choice
    ) == DenseGemmConfig(**choice)


@pytest.mark.parametrize(
    "m,changes",
    (
        (64, dict(load_path="cpasync")),
        (512, dict(tile_k=512)),
    ),
)
def test_nvfp4_dominated_variants_are_never_raced_but_can_be_pinned(m, changes):
    choice = (
        dict(
            backend="cutedsl",
            programmatic_serialization=False,
            evict_first=False,
            tile_m=64,
            tile_n=64,
            tile_k=128,
            load_path="tma",
            swap_ab=False,
            split_k_slices=None,
            large_m_unroll=None,
            target_occupancy=1,
        )
        | changes
    )
    for level in ("default", "audit"):
        with pytest.raises(ValueError, match="correctness"):
            dense_space("nvfp4", m, 8192, 5120, level).validate(choice)
    query = dense_query("nvfp4", m, 8192, 5120, "audit")
    with pytest.raises(ValueError, match="correctness"):
        TUNING.lower(query, DEVICE, choice)
    config = DenseGemmConfig(**choice)
    assert TUNING.configure(query, device=DEVICE, override=config).pinned == config


def test_off_uses_heuristic_default():
    query = dense_query("mxfp8", 8, 32768, 1280, "off")
    cfg = TUNING.configure(query, device=DEVICE)
    choices = list(TUNING.iterate(cfg))
    assert choices == [(TUNING.config_payload(cfg.default), cfg.default)]


def test_block32_propagates_effort():
    from b12x.gemm.block_fp8_linear._tuning import (
        BlockFp8LinearQuery,
        dense_query as lower,
    )

    query = BlockFp8LinearQuery(
        max_tokens=8,
        in_features=1280,
        out_features=32768,
        source_dtype="bfloat16",
        output_dtype="bfloat16",
        output_mode="provided",
        weight_block_size=32,
        autotune_effort="audit",
    )
    assert lower(query).autotune_effort == "audit"
    assert not lower(query).sfb_k_replicated


def test_dense_candidate_sets_expand_at_audit():
    candidates = []
    for effort in ("default", "audit"):
        query = dense_query("mxfp8", 8, 32768, 1280, effort)
        candidates.append({config for _, config in TUNING.choices(query, DEVICE)})
    assert candidates[0] < candidates[1]


@pytest.mark.parametrize("sm_count", (48, 188))
@pytest.mark.parametrize(
    "recipe", ("nvfp4", "mxfp4", "mxfp8", "tensor_fp8", "block_fp8")
)
def test_dense_filters_preserve_heuristic_defaults(sm_count, recipe):
    device = replace(DEVICE, sm_count=sm_count)
    for m in (1, 6, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1536, 2048, 4096):
        for n, k in (
            (1792, 5120),
            (32768, 1280),
            (8192, 1280),
            (5120, 8192),
            (1024, 4096),
            (256, 5120),
            (4096, 4096),
        ):
            for effort in ("default", "audit"):
                query = dense_query(recipe, m, n, k, effort)
                config = TUNING.default_config(query, device)
                TUNING.validate_config(query, config, device)
                assignment = TUNING.config_payload(config)
                TUNING.parameter_space(query, device).validate(assignment)
                assert TUNING.lower(query, device, assignment) == config


@pytest.mark.parametrize("m", (1, 48, 64))
@pytest.mark.parametrize("n", (256, 4096))
def test_mxfp4_heuristic_keeps_unswapped_scale_layout(m, n):
    query = dense_query("mxfp4", m, n, 5120, "default")
    device = replace(DEVICE, sm_count=48)
    config = TUNING.default_config(query, device)
    assert not config.swap_ab
    assert config.tile_n >= 64
    TUNING.validate_config(query, config, device)
    TUNING.parameter_space(query, device).validate(TUNING.config_payload(config))


@pytest.mark.parametrize("index", (2, 3, 4, 5, 7))
def test_captured_family_efforts_preserve_default_and_expand(index):
    from tests.experimental.b12x.preparation.test_component_efficiency import (
        _declarations,
    )

    module, query_type, values = _declarations()[index]
    candidates = []
    encodings = []
    for effort in ("default", "audit"):
        query = query_type(**values, autotune_effort=effort)
        config = module.TUNING.default_config(query, DEVICE)
        module.TUNING.validate_config(query, config, DEVICE)
        space = module.TUNING.parameter_space(query, DEVICE)
        payload = module.TUNING.config_payload(config)
        assignment = {knob.name: payload[knob.name] for knob in space.knobs}
        space.validate(assignment)
        assert module.TUNING.lower(query, DEVICE, assignment) == config
        candidates.append(
            {
                module.TUNING.config_payload(c)
                for _, c in module.TUNING.choices(query, DEVICE)
            }
        )
        encodings.append(module.TUNING.encode_query(query)["autotune_effort"])
    assert candidates[0] <= candidates[1]
    assert encodings == ["default", "audit"]


def test_paged_residency_domain_expands_without_device_queries():
    from tests.experimental.b12x.preparation.test_component_efficiency import (
        _declarations,
    )

    module, query_type, values = _declarations()[6]
    choices = []
    for effort in ("default", "audit"):
        query = query_type(**values, autotune_effort=effort)
        choices.append(
            set(module.TUNING.parameter_space(query, DEVICE).configurations())
        )
        assert module.TUNING.encode_query(query)["autotune_effort"] == effort
    assert choices[0] < choices[1]


@pytest.mark.parametrize(
    "family", ("dsa_indexer", "dense_mla", "gdn_prefill", "kda_prefill")
)
def test_effort_does_not_change_compile_jobs(family, monkeypatch):
    from types import SimpleNamespace
    import torch
    from b12x.attention import dsa_indexer, dense_mla
    from b12x.sequence import gdn_prefill, kda_prefill

    if family == "dsa_indexer":
        api = dsa_indexer
        caps = api.Caps(
            device="cpu",
            num_q_heads=32,
            max_q_rows=96,
            max_page_table_width=64,
            topk=512,
            mode="prefill",
            page_size=256,
            cache_format="mxfp4",
        )
    elif family == "dense_mla":
        api = dense_mla
        caps = api.Caps(
            device="cpu",
            mode="decode",
            kv_dtype=torch.bfloat16,
            num_q_heads=16,
            page_size=64,
            max_total_q=1,
            max_batch=1,
            max_cache_tokens=1024,
            max_page_table_width=16,
            num_cache_pages=16,
        )
    elif family == "gdn_prefill":
        api = gdn_prefill
        caps = api.Caps(
            device="cuda:0",
            max_tokens=512,
            max_seqs=1,
            max_state_slots=1,
            key_heads=16,
            value_heads=48,
        )
    else:
        api = kda_prefill
        caps = api.Caps(
            device="cuda:0", max_tokens=512, max_seqs=1, max_state_slots=1, heads=16
        )
    plans = []
    for effort in ("default", "audit"):
        monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", effort)
        plans.append(api.plan(caps))
    config = plans[0].contract.default_config(plans[0].query, DEVICE)
    jobs = [plan._compile_jobs(config, SimpleNamespace(ordinal=0)) for plan in plans]
    assert all(job == jobs[0] for job in jobs)
    assert [plan.query.autotune_effort for plan in plans] == ["default", "audit"]


def test_explicit_effort_scope_restores_after_nested_exception(monkeypatch):
    from b12x.preparation import AutotuneEffort, autotune_effort

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    assert efficiency.capture_autotune_effort() is AutotuneEffort.AUDIT
    with autotune_effort("default"):
        assert efficiency.capture_autotune_effort() is AutotuneEffort.DEFAULT
        with pytest.raises(RuntimeError, match="nested"):
            with autotune_effort(AutotuneEffort.OFF):
                assert efficiency.capture_autotune_effort() is AutotuneEffort.OFF
                raise RuntimeError("nested")
        assert efficiency.capture_autotune_effort() is AutotuneEffort.DEFAULT
    assert efficiency.capture_autotune_effort() is AutotuneEffort.AUDIT
    with pytest.raises(ValueError):
        with autotune_effort("invalid"):
            pytest.fail("invalid effort entered scope")
    assert efficiency.capture_autotune_effort() is AutotuneEffort.AUDIT


def test_explicit_effort_overrides_environment_and_query_retains_capture(
    monkeypatch,
):
    from b12x.preparation import autotune_effort

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "invalid")
    with autotune_effort("audit"):
        query = DenseGemmQuery(
            recipe="mxfp8",
            entry_point="gemm.mm",
            weight_storage="native",
            output_dtype="bfloat16",
            batch=1,
            max_rows=8,
            in_features=1280,
            out_features=32768,
        )
        assert efficiency.capture_autotune_effort() == "audit"
    assert query.autotune_effort == "audit"
    assert TUNING.parameter_space(query, DEVICE).autotune_effort == "audit"
    with pytest.raises(ValueError, match="B12X_AUTOTUNE_EFFORT"):
        efficiency.capture_autotune_effort()


def test_block_fp8_effort_does_not_change_compile_jobs():
    from types import SimpleNamespace
    from b12x.gemm import block_fp8_linear
    from b12x.preparation import autotune_effort

    caps = block_fp8_linear.Caps(
        device="cuda:0",
        max_tokens=8,
        in_features=1280,
        out_features=32768,
        block_size=(32, 32),
    )
    plans = []
    for effort in ("off", "default", "audit"):
        with autotune_effort(effort):
            plans.append(block_fp8_linear.plan(caps))
    config = plans[0].contract.default_config(plans[0].query, DEVICE)
    device = SimpleNamespace(ordinal=0, identity=DEVICE)
    jobs = [plan._compile_jobs(config, device) for plan in plans]
    assert all(job == jobs[0] for job in jobs)
    assert [plan.query.autotune_effort for plan in plans] == [
        "off",
        "default",
        "audit",
    ]


def test_off_does_not_insert_a_default_excluded_by_predicates():
    query = dense_query("mxfp8", 8, 32768, 1280, "off")
    configuration = TUNING.configure(query, device=DEVICE)
    configuration = replace(
        configuration,
        space=replace(
            configuration.space, efficiency_predicates=(lambda choice: False,)
        ),
    )
    with pytest.raises(ValueError, match="heuristic default is excluded"):
        list(TUNING.iterate(configuration))
