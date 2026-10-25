"""Host checks for paged-attention split policy search and workspace limits."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.attention.paged import _tuning
from b12x.preparation import DeviceIdentity, FrozenMapping


DEVICE = DeviceIdentity("nvidia", (12, 1), 48, "Synthetic SM121")


@pytest.fixture(autouse=True)
def host_device(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=48),
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))


def query(**changes):
    return replace(
        _tuning.GqaQuery(
            device="cuda:0",
            mode="decode",
            q_dtype="bfloat16",
            kv_dtype="bfloat16",
            q_heads=8,
            kv_heads=1,
            head_dim_qk=256,
            head_dim_vo=256,
            page_size=64,
            kv_cache_layout="separate",
            batch_size=1,
            query_len=1,
            cache_tokens=16384,
            window_left=-1,
            requested_graph_ctas_per_sm=None,
            requested_max_work_items=None,
            requested_max_partial_rows=None,
            force_split_kv=None,
            abi=FrozenMapping(),
            controls=FrozenMapping(),
            autotune_effort="default",
        ),
        **changes,
    )


def lower(q, *, residency=1, policy=False, split=True):
    return _tuning.TUNING.lower(
        q,
        DEVICE,
        dict(
            graph_ctas_per_sm=residency,
            force_split_kv=split,
            apply_chunk_efficiency_policy=policy,
        ),
    )


def test_audit_retains_configs_and_natural_default_membership():
    q = query()
    previous = set()
    for effort in ("default", "audit"):
        active = replace(q, autotune_effort=effort)
        space = _tuning.TUNING.parameter_space(active, DEVICE)
        configs = {
            _tuning.TUNING.lower(active, DEVICE, p) for p in space.configurations()
        }
        assert previous <= configs
        assert _tuning._heuristic(active, DEVICE) in configs
        previous = configs
    assert lower(replace(q, autotune_effort="audit")).max_chunks_per_request == 48
    assert lower(q, policy=True).max_chunks_per_request == 19
    with pytest.raises(ValueError, match="efficiency"):
        lower(q)


@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
def test_audit_lifts_measured_split_knees_and_preserves_residency(dtype):
    q = query(kv_dtype=dtype, autotune_effort="audit")
    assert lower(q, residency=1).max_chunks_per_request == 48
    assert lower(q, residency=2).max_chunks_per_request == 96
    assert lower(q, policy=True).max_chunks_per_request == (
        19 if dtype == "bfloat16" else 22
    )


def test_audit_lifts_nominal_and_bf16_minimax_split_limits():
    q = query(
        q_heads=6,
        head_dim_qk=128,
        head_dim_vo=128,
        cache_tokens=32768,
        autotune_effort="audit",
    )
    assert lower(q, residency=2, policy=True).max_chunks_per_request == 20
    assert lower(q, residency=2).max_chunks_per_request == 96


def test_audit_lifts_one_wave_limit(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    q = query(
        q_heads=36,
        kv_heads=4,
        head_dim_qk=128,
        head_dim_vo=128,
        page_size=128,
        cache_tokens=32768,
        kv_dtype="float8_e4m3fn",
        autotune_effort="audit",
    )
    assert lower(q, residency=2, policy=True).max_chunks_per_request == 12
    assert lower(q, residency=2).max_chunks_per_request == 24


def test_broader_policy_preserves_storage_and_direct_constraints():
    q = query(
        autotune_effort="audit",
        requested_max_work_items=7,
        requested_max_partial_rows=5,
    )
    cfg = lower(q, residency=2)
    assert cfg.max_chunks_per_request == 5
    assert cfg.max_work_items <= 7
    assert cfg.max_partial_rows <= 5
    _tuning._validate(q, cfg, DEVICE)
    direct = lower(replace(q, force_split_kv=False), split=False)
    assert direct.max_chunks_per_request == 1
    assert direct.max_partial_rows == 0
    with pytest.raises(ValueError, match="forced decode graph split"):
        lower(replace(q, requested_max_partial_rows=0))


@pytest.mark.parametrize(
    "name",
    [
        "B12X_PAGED_DECODE_GRAPH_CHUNK_PAGES",
        "B12X_PAGED_DECODE_GRAPH_MIN_CHUNK_PAGES",
    ],
)
def test_broader_policy_keeps_explicit_chunk_controls(name):
    q = query(autotune_effort="audit", controls=FrozenMapping({name: "16"}))
    cfg = lower(q)
    assert cfg.chunk_pages_lut()[-1] == 16
    assert cfg.max_chunks_per_request == 16
