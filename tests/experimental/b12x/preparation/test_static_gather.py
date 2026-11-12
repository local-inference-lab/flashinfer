"""Static DCP gather races preserve byte semantics, launch pins, and ownership."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import _preparation
from b12x.comm.pcie._benchmark import _DcpInputs, _dcp_reference
from b12x.comm.pcie._collective_tuning import CollectiveConfig, TUNING
from b12x.comm.pcie._tuning import PcieConfig, TUNING as NATIVE
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A
from b12x.preparation import DeviceIdentity, FrozenMapping, autotune_effort
from b12x.preparation.device import DetectedDevice
from b12x.preparation.types import CandidateResourceIneligibleError


SM120 = DeviceIdentity("nvidia", (12, 0), 188, "test device")


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("static gather contract tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse)


def native_query(*, world=16, rank=0, rows=3, dtype=torch.bfloat16, pins=(0, 0)):
    runtime = SimpleNamespace(
        world_size=world,
        rank=rank,
        device=torch.device("cpu"),
        max_batch_size=128,
        total_heads=2 * world,
        heads_per_rank=2,
        head_dim=8,
        query_head_dim=8,
        _slot_bytes=1 << 20,
        _exchange_chunks=64,
        _exchange_offset=16384,
        _exchange_counter_offset=2 << 20,
        _signal_ptrs=tuple(range(world)),
        _threads_override=pins[0],
        _block_limit_override=pins[1],
        _owned_buffers=(SimpleNamespace(nbytes=2 << 20),),
    )
    runtime._resolve_launch_config = MethodType(
        PCIeDCPA2A._resolve_launch_config, runtime
    )
    source = torch.empty((rows, 2, 8), dtype=dtype)
    output = torch.empty((rows, world * 2, 8), dtype=dtype)
    query = _preparation.query_from_runtime(
        runtime,
        surface="DcpAllToAll.all_gather_heads",
        call={"local_input": source, "out": output},
    )
    return query, runtime


def configured(plan, device=SM120, *, effort="default"):
    with autotune_effort(effort):
        return TUNING.configure(plan.query, device=device, override=plan.override)


def configs(plan, device=SM120, *, effort="default"):
    configuration = configured(plan, device, effort=effort)
    return [value for _, value in TUNING.iterate(configuration)]


def test_static_tp16_gather_races_native_geometry_and_complete_phase_clusters():
    query, runtime = native_query()
    plan = _preparation.plan(query, runtime=runtime)
    assert plan.contract is TUNING
    assert set(plan.query.alternatives) == {
        "dcp",
        "dcp_phased",
        "dcp_ring",
        "dcp_recursive",
        "dcp_recursive_ready",
    }
    assert plan.query.default == CollectiveConfig(
        implementation="dcp", threads=256, blocks=16
    )
    values = configs(plan)
    phased = [value for value in values if value.implementation == "dcp_phased"]
    assert {value.threads for value in phased} == {128, 256, 512}
    assert all(
        (value.blocks, value.active_blocks, value.submode) == (8, 8, "barrier")
        for value in phased
    )
    assert any(value.implementation == "dcp" for value in values)
    assert "selected_phased_push" not in repr(plan.query.alternatives)


def test_dcp8_races_phased_push_recursive_and_ring():
    query, runtime = native_query(world=8)
    selected = _preparation._selected_gather_query(
        query, "selected_phased_push", (128,)
    )
    assert (
        selected is not None
        and selected.call["gather_transport"] == "selected_phased_push"
    )
    assert (
        _preparation._selected_gather_query(query, "selected_recursive", (128,))
        is not None
    )
    assert (
        _preparation._selected_gather_query(query, "selected_ring", (128,)) is not None
    )
    plan = _preparation.plan(query, runtime=runtime)
    assert "dcp_phased_push" in plan.query.alternatives
    assert "dcp_recursive" in plan.query.alternatives
    assert "dcp_ring" in plan.query.alternatives
    push = [
        value for value in configs(plan) if value.implementation == "dcp_phased_push"
    ]
    assert push == [
        CollectiveConfig(
            implementation="dcp_phased_push",
            threads=128,
            blocks=8,
            submode="barrier",
            active_blocks=8,
        )
    ]


@pytest.mark.parametrize("pins", [(512, 8), (128, 8), (256, 0), (0, 8)])
def test_environment_geometry_overrides_constrain_both_implementations(pins):
    query, runtime = native_query(pins=pins)
    plan = _preparation.plan(query, runtime=runtime)
    values = configs(plan)
    expected = {"dcp", "dcp_phased"}
    if pins[0] in (0, 128):
        expected.update(("dcp_ring", "dcp_recursive", "dcp_recursive_ready"))
    assert {value.implementation for value in values} == expected
    if pins[0]:
        assert {value.threads for value in values} == {pins[0]}
    if pins[1]:
        assert {value.blocks for value in values} == {pins[1]}
    if pins == (512, 8):
        assert len(values) == 2


@pytest.mark.parametrize("pins", [(512, 16), (128, 4), (64, 8), (96, 8)])
def test_incompatible_environment_pins_leave_the_native_candidate(pins):
    query, runtime = native_query(pins=pins)
    plan = _preparation.plan(query, runtime=runtime)
    assert set(plan.query.alternatives) == {"dcp"}
    assert configs(plan) == [
        CollectiveConfig(implementation="dcp", threads=pins[0], blocks=pins[1])
    ]


def test_native_pin_retains_the_fixed_plan_and_explicit_profile_is_authoritative():
    query, runtime = native_query(pins=(512, 8))
    pin = PcieConfig(backend="native")
    fixed = _preparation.plan(query, runtime=runtime, override=pin)
    assert fixed.contract is NATIVE and fixed.query == query and fixed.override is pin
    phased = CollectiveConfig(
        implementation="dcp_phased",
        threads=512,
        blocks=8,
        submode="barrier",
        active_blocks=8,
    )
    selected = _preparation.plan(query, runtime=runtime, override=phased)
    configuration = configured(selected)
    assert configuration.pinned == configuration.default == phased
    with pytest.raises(ValueError, match="parameter space"):
        configured(
            _preparation.plan(
                query, runtime=runtime, override=replace(phased, threads=128)
            )
        )


@pytest.mark.parametrize(
    "device",
    [
        None,
        DeviceIdentity("nvidia", (12, 1), 188, "test device"),
        DeviceIdentity("nvidia", (10, 0), 148, "test device"),
        DeviceIdentity("nvidia", (12, 0), 63, "test device"),
        DeviceIdentity("other", (12, 0), 188, "test device"),
    ],
)
@pytest.mark.parametrize("effort", ["default", "off"])
@pytest.mark.parametrize("world", [8, 16])
def test_unqualified_hardware_does_not_admit_phased_candidates(device, effort, world):
    query, runtime = native_query(world=world)
    plan = _preparation.plan(query, runtime=runtime)
    assert {value.implementation for value in configs(plan, device, effort=effort)} == {
        "dcp"
    }
    phased = CollectiveConfig(
        implementation="dcp_phased",
        threads=256,
        blocks=8,
        submode="barrier",
        active_blocks=8,
    )
    with pytest.raises(ValueError, match="parameter space"):
        TUNING.validate_config(plan.query, phased, device)


@pytest.mark.parametrize("owner", ["DcpAllToAll", "DcpAllToAllPool"])
@pytest.mark.parametrize("normalize_call", [False, True])
def test_fused_pair_expert_selection_rejects_batched_loads(owner, normalize_call):
    query, runtime = native_query(world=8, rows=1)
    surface = f"{owner}.all_gather_pair_kimi_topk"
    call = {"threads": 512, "load_batch": 2}
    with pytest.raises(ValueError, match="batched paired loads|load[ _]batch"):
        if normalize_call:
            query = _preparation.query_from_runtime(runtime, surface=surface, call=call)
        else:
            query = replace(query, surface=surface, call=FrozenMapping(call))
        NATIVE.validate_query(query, SM120)


@pytest.mark.parametrize("owner", ["DcpAllToAll", "DcpAllToAllPool"])
@pytest.mark.parametrize("normalize_call", [False, True])
def test_fused_pair_expert_selection_requires_single_cta(owner, normalize_call):
    query, runtime = native_query(world=8, rows=1)
    surface = f"{owner}.all_gather_pair_kimi_topk"
    call = {"threads": 512, "row_blocks": 2}
    with pytest.raises(ValueError, match="row_blocks"):
        if normalize_call:
            query = _preparation.query_from_runtime(runtime, surface=surface, call=call)
        else:
            query = replace(query, surface=surface, call=FrozenMapping(call))
        NATIVE.validate_query(query, SM120)


@pytest.mark.parametrize("owner", ["DcpAllToAll", "DcpAllToAllPool"])
@pytest.mark.parametrize("metadata", [{}, {"row_blocks": 1, "load_batch": 1}])
def test_fused_pair_expert_selection_accepts_default_geometry(owner, metadata):
    query, runtime = native_query(world=8, rows=1)
    surface = f"{owner}.all_gather_pair_kimi_topk"
    call = {"threads": 512, **metadata}
    raw = replace(query, surface=surface, call=FrozenMapping(call))
    normalized = _preparation.query_from_runtime(runtime, surface=surface, call=call)
    NATIVE.validate_query(raw, SM120)
    NATIVE.validate_query(normalized, SM120)
    assert normalized.call["row_blocks"] == normalized.call["load_batch"] == 1


@pytest.mark.parametrize("world", [8, 16])
@pytest.mark.parametrize(
    "module,launcher",
    [
        ("_ring_gather_cute", "RingGather"),
        ("_recursive_gather_cute", "RecursiveGather"),
    ],
)
def test_selected_gather_flags_preserve_native_signal_layout(
    monkeypatch, world, module, launcher
):
    from importlib import import_module

    from b12x.comm.pcie.pcie_dcp_a2a import _FLAG_STRIDE, _MAX_BLOCKS, _MAX_RANKS

    kernels = import_module(f"b12x.comm.pcie.{module}")
    monkeypatch.setattr(kernels, "Int64", int)
    monkeypatch.setattr(kernels, "Uint32", int)
    flag = getattr(kernels, launcher)._flag.__wrapped__
    owner = SimpleNamespace(_world_size=world)
    for generation in (1, 2, 3, 0xFFFFFFFF):
        for block in range(8):
            for sender in range(world):
                native = (
                    _MAX_BLOCKS * _MAX_RANKS
                    + (
                        (generation % 2 * _MAX_BLOCKS + block)
                        * _MAX_RANKS
                        * _FLAG_STRIDE
                    )
                    + sender * _FLAG_STRIDE
                )
                assert flag(owner, 0, sender, block, generation) == native


@pytest.mark.parametrize("world", [2, 4, 8, 9, 10, 12])
def test_other_rank_counts_catalog_with_dcp8_specializations(world):
    query, runtime = native_query(world=world)
    plan = _preparation.plan(query, runtime=runtime)
    if world < 8:
        assert "dcp_phased" not in plan.query.alternatives
    elif world == 8:
        assert "dcp_phased" in plan.query.alternatives
        assert "dcp_phased_push" in plan.query.alternatives
        assert "dcp_recursive" in plan.query.alternatives
        assert "dcp_ring" in plan.query.alternatives
    else:
        assert plan.contract is NATIVE and plan.query == query


@pytest.mark.parametrize(
    "change",
    [
        dict(disjoint_output=False),
        dict(pack_aligned=False),
        dict(stride=(17, 8, 1)),
        dict(out_stride=(257, 8, 1)),
    ],
)
def test_phased_candidate_requires_its_actual_layout_contract(change):
    query, runtime = native_query()
    query = replace(query, call=FrozenMapping(dict(query.call, **change)))
    plan = _preparation.plan(query, runtime=runtime)
    assert set(plan.query.alternatives) == {"dcp"}


def test_tp16_lse_and_explicit_selected_queries_remain_native():
    query, runtime = native_query()
    lse = replace(query, surface="DcpAllToAll.lse_reduce_scatter")
    assert _preparation.plan(lse, runtime=runtime).contract is NATIVE
    explicit = _preparation._selected_gather_query(
        query, "selected_phased_barrier", (128, 256, 512)
    )
    assert _preparation.plan(explicit, runtime=runtime).contract is NATIVE


def test_shape_rank_and_byte_width_participate_in_collective_identity():
    signatures = []
    for options in ({}, {"rank": 15}, {"rows": 1}, {"dtype": torch.float8_e4m3fn}):
        query, runtime = native_query(**options)
        plan = _preparation.plan(query, runtime=runtime)
        signatures.append(TUNING.collective_signature(plan.query))
    assert signatures[0] == signatures[1]
    assert signatures[0] != signatures[2]
    assert signatures[0] != signatures[3]
    assert TUNING.candidate_contract_version == 6


@pytest.mark.parametrize(
    "implementation,threads,transport",
    [
        ("dcp_phased", 512, "phased_barrier"),
        ("dcp_ring", 128, "ring"),
        ("dcp_recursive", 128, "recursive"),
        ("dcp_recursive_ready", 128, "recursive_ready"),
    ],
)
def test_selected_compile_and_materialization_use_the_winning_profile(
    monkeypatch, implementation, threads, transport
):
    from b12x.comm.pcie import (
        _gather_preparation,
        _runtime_gather_cute,
        _joint_cluster_cute,
    )

    compiled, occupancy = [], []
    launcher = lambda *args: None

    def compile_selected(*args):
        compiled.append(args)
        return launcher

    monkeypatch.setattr(
        _runtime_gather_cute, "compile_selected_gather", compile_selected
    )
    monkeypatch.setattr(_gather_preparation, "load_programs", lambda value: value)
    monkeypatch.setattr(
        _joint_cluster_cute,
        "validate_cluster_residency",
        lambda *args: occupancy.append(args),
    )
    query, runtime = native_query(pins=(0, 8))
    plan = _preparation.plan(query, runtime=runtime)
    winner = CollectiveConfig(
        implementation=implementation,
        threads=threads,
        blocks=8,
        submode="barrier",
        active_blocks=8,
    )
    device = DetectedDevice(ordinal=None, identity=SM120)
    (job,) = plan._compile_jobs(winner, device)
    assert job.factory.endswith("_runtime_gather_cute:compile_selected_gather")
    assert job.args == (16, 0, 8, threads, 8, 64, transport)
    state = plan._materialize(SimpleNamespace(config=winner), device)
    assert compiled == [job.args]
    assert occupancy == (
        [(launcher, 8, threads, 8)] if implementation == "dcp_phased" else []
    )
    assert state.runtime is runtime and state.selector.item() == 8
    assert (state.query.call["submode"], state.query.call["active_blocks"]) == (
        "barrier",
        8,
    )
    assert (state.query.call["threads"], state.query.call["block_limit"]) == (
        threads,
        8,
    )

    def reject(*args):
        raise CandidateResourceIneligibleError("compiled cluster does not fit")

    monkeypatch.setattr(_joint_cluster_cute, "validate_cluster_residency", reject)
    if implementation == "dcp_phased":
        with pytest.raises(CandidateResourceIneligibleError, match="does not fit"):
            plan._materialize(SimpleNamespace(config=winner), device)


@pytest.mark.parametrize(
    "changes",
    [
        dict(threads=64),
        dict(threads=512),
        dict(blocks=16),
        dict(active_blocks=4),
        dict(submode="word_exchange"),
    ],
)
@pytest.mark.parametrize(
    "implementation", ["dcp_ring", "dcp_recursive", "dcp_recursive_ready"]
)
def test_gather_candidate_rejects_profiles_outside_its_qualified_geometry(
    changes, implementation
):
    query, runtime = native_query(pins=(0, 8))
    plan = _preparation.plan(query, runtime=runtime)
    ring = next(
        value for value in configs(plan) if value.implementation == implementation
    )
    assert (ring.threads, ring.blocks, ring.active_blocks, ring.submode) == (
        128,
        8,
        8,
        "barrier",
    )
    with pytest.raises(ValueError, match="parameter space"):
        TUNING.validate_config(plan.query, replace(ring, **changes), SM120)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rows", [1, 3, 31, 32])
def test_gather_reference_reorders_mutable_bytes_without_numeric_conversion(
    dtype, rows
):
    references = []
    for rank in range(16):
        query, runtime = native_query(rank=rank, rows=rows, dtype=dtype)
        references.append(
            _dcp_reference(query, (rows, 2, 8), dtype, runtime.device, False)
        )
    assert all(len(reference.seeds) == 3 for reference in references)
    for seed in range(3):
        rank_bytes = torch.stack(
            [reference.seeds[seed].view(torch.uint8) for reference in references]
        )
        expected = rank_bytes.permute(1, 0, 2, 3).reshape(rows, 32, -1)
        for reference in references:
            assert torch.equal(reference.expected[seed].view(torch.uint8), expected)
    assert not torch.equal(
        references[0].seeds[0].view(torch.uint8),
        references[0].seeds[2].view(torch.uint8),
    )


def test_byte_validation_detects_signed_zero_corruption_and_input_mutation():
    query, runtime = native_query()
    inputs = _DcpInputs(SimpleNamespace(query=query, runtime=runtime))
    assert len(inputs.producers) == 3
    inputs.produce(1)
    assert torch.all(inputs.out.view(torch.uint8) == 0xA5)
    with pytest.raises(AssertionError):
        inputs.validate()
    inputs.out.copy_(inputs.reference.expected[1])
    inputs.validate()
    inputs.out.view(torch.uint8).reshape(-1)[1] = 0x80
    assert torch.equal(inputs.out, inputs.reference.expected[1])
    with pytest.raises(AssertionError):
        inputs.validate()
    inputs.out.copy_(inputs.reference.expected[1])
    inputs.source.view(torch.uint8).reshape(-1)[0] = 1
    with pytest.raises(AssertionError):
        inputs.validate()


def test_byte_validation_accepts_exact_nan_payloads():
    query, runtime = native_query()
    inputs = _DcpInputs(SimpleNamespace(query=query, runtime=runtime))
    inputs.produce(0)
    inputs.out.copy_(inputs.reference.expected[0])
    assert inputs.out.isnan().any()
    inputs.validate()


def test_parallel_paired_gather_rejects_overlapping_outputs_before_preparation():
    from b12x.comm.pcie.pcie_dcp_a2a import _pair_outputs_disjoint

    _, runtime = native_query(rows=4)
    runtime.query_head_dim = 672
    first = torch.empty((4, 224), dtype=torch.bfloat16)
    second = torch.empty((4, 56), dtype=torch.float32)
    out_first = torch.empty((4, 3584), dtype=torch.bfloat16)
    out_second = torch.empty((4, 896), dtype=torch.float32)
    assert _pair_outputs_disjoint(first, second, out_first, out_second)
    call = dict(
        local_first=first,
        local_second=second,
        out_first=out_first,
        out_second=out_second,
        row_blocks=4,
        load_batch=2,
    )
    query = _preparation.query_from_runtime(
        runtime, surface="DcpAllToAll.all_gather_pair", call=call
    )
    NATIVE.validate_query(query, SM120)
    assert query.call["row_blocks"] == 4 and query.call["disjoint_output"]
    assert query.call["load_batch"] == 2
    shared = torch.empty((4 * 3584,), dtype=torch.bfloat16)
    call.update(
        local_first=shared[: 4 * 224].view(4, 224), out_first=shared.view(4, 3584)
    )
    query = _preparation.query_from_runtime(
        runtime, surface="DcpAllToAll.all_gather_pair", call=call
    )
    with pytest.raises(ValueError, match="disjoint"):
        NATIVE.validate_query(query, SM120)
    # The one-CTA declaration preserves its existing alias contract.
    query = replace(
        query, call=FrozenMapping(dict(query.call, row_blocks=1, load_batch=1))
    )
    NATIVE.validate_query(query, SM120)
    assert not _pair_outputs_disjoint(
        first, second, out_first, out_first.view(torch.float32)
    )


@pytest.mark.parametrize("row_blocks", [True, 2.5, "2"])
def test_paired_gather_rejects_non_integer_row_blocks(row_blocks):
    from b12x.comm import pcie

    _, runtime = native_query(rows=4)
    runtime.query_head_dim = 672
    call = dict(
        local_first=torch.empty((4, 224), dtype=torch.bfloat16),
        local_second=torch.empty((4, 56), dtype=torch.float32),
        row_blocks=row_blocks,
    )
    with pytest.raises(ValueError, match="row blocks"):
        query = pcie.query_from_runtime(
            runtime, surface="DcpAllToAll.all_gather_pair", call=call
        )
        pcie.plan(query, runtime=runtime)
