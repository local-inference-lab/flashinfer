"""Host checks for QSA direct KV warp search and declaration identity."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from b12x.attention.qsa._tuning import QsaConfig, QsaQuery, TUNING
from b12x.preparation import DeviceIdentity
from b12x.preparation._efficiency import AutotuneEffort
from b12x._lib.compile_pool import CompileJob


DEVICE = DeviceIdentity("nvidia", (12, 0), 148, "SM120")


def query():
    ranks = {
        "request_ids": ("int32", 1),
        "rope_positions": ("int64", 2),
        "index_query": ("bfloat16", 3),
        "raw_index_key": ("bfloat16", 2),
        "main_k_cache": ("bfloat16", 4),
        "main_v_cache": ("bfloat16", 4),
        "main_block_table": ("int32", 2),
        "compressed_k_cache": ("bfloat16", 3),
        "compressed_block_table": ("int32", 2),
        "raw_k_ring": ("bfloat16", 3),
        "raw_logical_positions": ("int64", 2),
        "raw_rope_positions": ("int64", 3),
        "raw_interval_start_positions": ("int64", 1),
        "raw_state_slot_ids": ("int32", 1),
        "index_q_norm_weight": ("bfloat16", 1),
        "index_k_norm_weight": ("bfloat16", 1),
        "rope_cos": ("bfloat16", 2),
        "rope_sin": ("bfloat16", 2),
    }
    return QsaQuery(
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        q_heads=24,
        kv_heads=2,
        head_dim=256,
        index_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        index_rotary_dim=64,
        main_page_size=3008,
        max_batch=1,
        max_q_rows=65,
        max_seq_len=3008,
        max_speculative_tokens=0,
        compress_ratio=4,
        budget=2048,
        position_axes=1,
        mrope_interleaved=False,
        max_raw_state_slots=1,
        num_main_cache_pages=4,
        num_compressed_cache_pages=4,
        compressed_page_size=752,
        mrope_sections=None,
        rms_norm_eps=1e-6,
        abi={
            "operands": {
                name: {"dtype": dtype, "strides": (1,) * rank}
                for name, (dtype, rank) in ranks.items()
            }
        },
    )


@pytest.mark.parametrize(
    "effort,warps",
    [("off", {2}), ("default", {1, 2, 4}), ("audit", {1, 2, 4})],
)
def test_legal_warps_and_natural_heuristic_membership(effort, warps):
    candidate_query = replace(query(), autotune_effort=effort)
    configs = {config for _, config in TUNING.choices(candidate_query, DEVICE)}
    assert {config.sparse_gqa_direct_kv_warps for config in configs} == warps
    assert TUNING.default_config(candidate_query, DEVICE) in configs
    for config in configs:
        TUNING.validate_config(candidate_query, config, DEVICE)


def test_pins_preserve_legal_warps():
    candidate_query = replace(query(), autotune_effort="off")
    for warps in (1, 4):
        config = QsaConfig(backend="cutedsl", sparse_gqa_direct_kv_warps=warps)
        assert (
            TUNING.configure(candidate_query, device=DEVICE, override=config).pinned
            == config
        )
    with pytest.raises(ValueError, match="must be 1, 2, or 4"):
        TUNING.validate_config(
            candidate_query, replace(config, sparse_gqa_direct_kv_warps=3), DEVICE
        )


def test_effort_capture_serialization_and_program_key_neutrality(monkeypatch):
    from b12x.attention.qsa import _contract
    from b12x.preparation.session import _declaration_key

    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "default")
    low = query()
    monkeypatch.setenv("B12X_AUTOTUNE_EFFORT", "audit")
    maximum = query()
    assert low.autotune_effort is AutotuneEffort.DEFAULT
    assert maximum.autotune_effort is AutotuneEffort.AUDIT
    assert QsaQuery(**low.to_dict()) == low
    assert TUNING.encode_query(low) != TUNING.encode_query(maximum)
    assert "autotune_effort" not in low.program_payload()
    assert low.program_payload() == maximum.program_payload()
    caps = _contract._caps_from_query(low, ordinal=0)
    monkeypatch.setattr(_contract, "_query_from_caps", lambda *_: low)
    low_plan = _contract.plan(caps)
    monkeypatch.setattr(_contract, "_query_from_caps", lambda *_: maximum)
    max_plan = _contract.plan(caps)
    assert _declaration_key(low_plan) != _declaration_key(max_plan)
    default = TUNING.default_config(low, DEVICE)
    compile_device = SimpleNamespace(ordinal=0)
    assert low_plan._compile_jobs(default, compile_device) == max_plan._compile_jobs(
        default, compile_device
    )
    config = TUNING.encode_config(TUNING.default_config(low, DEVICE))
    assert CompileJob.create(
        "b12x.attention.qsa._contract:compile_qsa", low.program_payload(), config, 0
    ) == CompileJob.create(
        "b12x.attention.qsa._contract:compile_qsa", maximum.program_payload(), config, 0
    )


def test_unset_environment_captures_the_default_tier(monkeypatch):
    monkeypatch.delenv("B12X_AUTOTUNE_EFFORT", raising=False)
    assert query().autotune_effort is AutotuneEffort.DEFAULT
