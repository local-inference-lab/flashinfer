"""Tagged gather eligibility, storage separation and arithmetic-free declarations."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from b12x.comm import pcie
from b12x.comm.pcie.pcie_dcp_a2a import _staging_layout
from b12x.comm.pcie._collective_tuning import CollectiveConfig
from b12x.preparation import DeviceIdentity, FrozenMapping

DEVICE = DeviceIdentity("nvidia", (12, 0), 188, "SM120")


def declaration(rows=8, alias=False, aligned=True):
    layout = _staging_layout(
        signal_bytes=266240,
        world_size=4,
        max_batch_size=128,
        total_heads=4,
        head_dim=64,
    )
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        rank=0,
        world_size=4,
        max_batch_size=128,
        total_heads=4,
        head_dim=64,
        query_head_dim=64,
        _slot_bytes=layout.slot_bytes,
        _signal_ptrs=(1, 2, 3, 4),
        _threads_override=0,
        _block_limit_override=0,
        _tagged_offset=layout.tagged_offset,
        _tagged_words=layout.tagged_words,
        _resolve_launch_config=lambda threads, block_limit, **_: (threads, block_limit),
    )
    output = torch.empty((rows, 4, 64), dtype=torch.bfloat16)
    source = (
        output.view(-1)[: rows * 64].view(rows, 1, 64)
        if alias
        else torch.empty((rows, 1, 64), dtype=torch.bfloat16)
    )
    if not aligned:
        source = torch.empty(rows * 64 + 1, dtype=torch.bfloat16)[1:].view(rows, 1, 64)
    query = pcie.query_from_runtime(
        runtime,
        surface="DcpAllToAll.all_gather_heads",
        call=dict(local_input=source, out=output),
    )
    return pcie.plan(query, runtime=runtime), layout


def test_tagged_transport_is_eligible_without_changing_default():
    plan, layout = declaration()
    configs = [
        config
        for _, config in plan.contract.eligible_plan(plan.query, DEVICE).candidates
    ]
    assert plan.contract.default_config(plan.query, DEVICE).implementation == "dcp"
    tagged = [config for config in configs if config.implementation == "dcp_tagged"]
    assert {config.threads for config in tagged} == {32, 64, 128, 256, 512}
    assert {config.blocks for config in tagged} == {1}
    assert layout.tagged_offset >= layout.lse_offset + layout.lse_capacity * 4
    assert layout.slot_bytes >= layout.tagged_offset + 4 * layout.tagged_words * 8
    assert layout.tagged_words == 4096
    for config in tagged:
        assert (
            plan.contract.decode_config(
                FrozenMapping(plan.contract.encode_config(config))
            )
            == config
        )


def test_aliasing_does_not_enter_tagged_candidate_catalog():
    plan, _ = declaration(alias=True)
    configs = [
        config
        for _, config in plan.contract.eligible_plan(plan.query, DEVICE).candidates
    ]
    assert all(config.implementation == "dcp" for config in configs)
    with pytest.raises(ValueError):
        plan.contract.validate_config(
            plan.query,
            CollectiveConfig(implementation="dcp_tagged", threads=128, blocks=1),
            DEVICE,
        )


def test_tagged_query_rejects_insufficient_capacity_and_aliasing():
    from b12x.comm.pcie._tuning import PcieQuery, TUNING

    plan, _ = declaration()
    native = PcieQuery(**plan.query.alternatives["dcp_tagged"])
    for query in (
        replace(native, world_size=2),
        replace(native, setup=FrozenMapping({**native.setup, "tagged_words": 1})),
        replace(native, call=FrozenMapping({**native.call, "output_alias": True})),
    ):
        with pytest.raises(ValueError, match="tagged DCP gather"):
            TUNING.validate_query(query, DEVICE)


def test_misaligned_words_do_not_enter_tagged_catalog():
    plan, _ = declaration(aligned=False)
    assert all(
        config.implementation == "dcp"
        for _, config in plan.contract.eligible_plan(plan.query, DEVICE).candidates
    )


def test_oversized_rows_do_not_reserve_unusable_tagged_storage():
    layout = _staging_layout(
        signal_bytes=266240,
        world_size=4,
        max_batch_size=128,
        total_heads=4 * 512,
        head_dim=64,
    )
    assert layout.tagged_words == 0
    assert layout.slot_bytes == layout.tagged_offset


def test_cached_tagged_layout_must_match_the_bound_owner():
    from b12x.comm.pcie._dcp_preparation import _DcpExecutionState
    from b12x.comm.pcie._tuning import PcieQuery

    plan, _ = declaration()
    native = PcieQuery(**plan.query.alternatives["dcp_tagged"])
    owner = SimpleNamespace(
        rank=0,
        world_size=4,
        **{
            "_" + field: native.setup[field]
            for field in ("tagged_words", "tagged_offset", "slot_bytes")
        },
    )
    _DcpExecutionState(native, owner, {}).require_runtime(owner)
    for field in ("tagged_words", "tagged_offset", "slot_bytes"):
        wrong = replace(
            native,
            setup=FrozenMapping({**native.setup, field: native.setup[field] + 256}),
        )
        with pytest.raises(ValueError, match="owned IPC layout"):
            _DcpExecutionState(wrong, owner, {}).require_runtime(owner)
