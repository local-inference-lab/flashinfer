"""Output compatibility and coordinated setup for the PCIe dispatcher."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_allreduce as dispatch
from b12x.comm.pcie import pcie_oneshot as oneshot
from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16, _make_layout


@pytest.mark.parametrize(
    "nbytes, expected",
    (
        (16, "remote_push_full"),
        (24576, "remote_push_full"),
        (24592, "remote_push_rsag"),
        (655360, "remote_push_rsag"),
        (2097168, "remote_push_rsag"),
    ),
)
def test_payload_ready_graph_plan_requires_owned_incoming_storage(nbytes, expected):
    native = SimpleNamespace(
        world_size=4,
        eager_tables=(256, 512),
        eager_buffer_bytes=4 << 20,
        plain_payload_ready_region_packs=4 * (4 << 20) // 16,
        transport_policy=(False, True, True, True, True),
    )
    inp = torch.empty(nbytes // 2, dtype=torch.bfloat16)
    config = oneshot._CuTeOneshotBackend._plain_launch_config
    assert config(native, inp, graph_channel=True) == (expected, 256, 16)
    assert config(native, inp, graph_channel=False)[0] == "pull"
    assert config(native, inp.to(torch.float16), graph_channel=True)[0] == "pull"
    native.plain_payload_ready_region_packs = 0
    assert config(native, inp, graph_channel=True)[0] == "pull"


def _runtime():
    runtime = object.__new__(PCIeTwoShotBF16)
    runtime.device = torch.device("cpu")
    runtime.world_size = 4
    runtime.row_elems = 8
    runtime.max_rows = (2 << 20) // 16
    runtime._closed = False
    return runtime


@pytest.mark.parametrize(
    "nbytes",
    (
        65536 - 64,
        65536,
        65536 + 64,
        640 << 10,
        (640 << 10) + 64,
        2 << 20,
        (2 << 20) + 64,
    ),
)
@pytest.mark.parametrize("payload_ready", (False, True))
def test_output_aware_twoshot_size_band(nbytes, payload_ready):
    primary = SimpleNamespace(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        max_size=(2 << 20) + 64,
        _sentinel_supported=payload_ready,
        _transport_policy=(False, True, True, True, True),
    )
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=_runtime(), algorithm_override="auto"
    )
    inp = torch.empty(nbytes // 2, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    expected = (
        nbytes > 640 << 10 if payload_ready else nbytes >= 65536
    ) and nbytes <= 2 << 20
    assert manager._use_bf16_twoshot(inp, out) == expected
    assert manager._use_bf16_twoshot(inp, None) == expected
    assert not manager._use_bf16_twoshot(inp, inp)


def _prepared_twoshot_runtime(capacity=2 << 20):
    """A host two-shot runtime with the dispatcher's one-pack native rows."""
    max_rows = (capacity + 63) // 64 * 4
    layout = _make_layout(max_rows, 8, 4)
    base = 1 << 32
    return PCIeTwoShotBF16._from_prepared_factory(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        signal_ptrs=(base,) * 8,
        staging_ptrs=tuple(
            (base + layout.signal_bytes + slot * layout.slot_bytes,) * 8
            for slot in (0, 1)
        ),
        owned_buffers=(),
        ipc=None,
        exchange_group=None,
        max_rows=max_rows,
        row_elems=8,
        pack_stride=layout.pack_stride,
        reduced_offset=layout.reduced_offset,
        slot_bytes=layout.slot_bytes,
    )


def _oneshot_query(nbytes, transport, *, payload_ready):
    """The staged one-shot declaration a TP4 single-channel runtime reports."""
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.preparation import FrozenMapping

    return PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            {
                "dtype": "bfloat16",
                "transport": transport,
                "threads": 256,
                "blocks": 16,
                "eager_transport": "pull",
                "eager_threads": 512,
                "eager_blocks": 36,
                "size_packs": nbytes // 16,
                "stage_input": True,
                "device_index": 0,
                "remote_push_region_packs": (4 << 20) // 16 if payload_ready else 0,
            }
        ),
        setup=FrozenMapping({"shape": (nbytes // 2,), "dtype": "bfloat16"}),
    )


@pytest.mark.parametrize("nbytes", (24 << 10, 65600, 73728, 640 << 10, 2 << 20))
@pytest.mark.parametrize("payload_ready", (False, True))
def test_tp4_plan_declares_push_and_pull_twoshot_candidates(
    monkeypatch, nbytes, payload_ready
):
    from b12x.comm.pcie import _preparation, _twoshot_preparation
    from b12x.comm.pcie._collective_tuning import (
        TUNING,
        CollectiveConfig,
        CollectiveQuery,
    )
    from b12x.preparation import autotune_effort
    from b12x.preparation.types import _plan_scope

    twoshot = _prepared_twoshot_runtime()
    channel = SimpleNamespace(device=torch.device("cpu"))
    primary = SimpleNamespace(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        max_size=(2 << 20) + 64,
        single_channel=True,
        _sentinel_supported=payload_ready,
        _transport_policy=(False, True, True, True, True),
        _prepared_channel_for_stream=lambda stream, channel_id: channel,
    )
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=twoshot, algorithm_override="auto"
    )
    oneshot_transport = (
        dispatch.tp4_bf16_transport(nbytes, payload_ready=True, twoshot_available=False)
        if payload_ready
        else "pull"
    )

    def query_from_runtime(runtime, *, surface, call):
        if surface.startswith("OneshotAllReduce."):
            return _oneshot_query(
                nbytes, oneshot_transport, payload_ready=payload_ready
            )
        return _twoshot_preparation.query_from_runtime(
            runtime, surface=surface, call=call
        )

    monkeypatch.setattr(_preparation, "query_from_runtime", query_from_runtime)
    inp = torch.empty(nbytes // 2, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    plan = manager.plan(inp, out=out)
    query = plan.query
    assert isinstance(query, CollectiveQuery)
    candidates = query.alternatives
    assert candidates["twoshot"]["call"]["transport"] == "push"
    assert candidates["twoshot_pull"]["call"]["transport"] == "pull"
    assert candidates["twoshot_pull"]["setup"]["push_shard_nbytes"] == 0
    # Without tuning the static crossover policy applies; it never picks pull.
    static = "twoshot" if manager._use_bf16_twoshot(inp, out) else oneshot_transport
    assert query.default.implementation == static
    with autotune_effort("off"):
        off = [
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        ]
    with autotune_effort("default"):
        default = [
            config for _, config in TUNING.iterate(TUNING.configure(query, device=None))
        ]
    pull = CollectiveConfig(implementation="twoshot_pull", threads=512, blocks=64)
    # Tuning off prepares the static choice alone; the default search races pull.
    assert off == [query.default]
    assert pull in default
    assert {
        (config.threads, config.blocks)
        for config in default
        if config.implementation == "twoshot_pull"
    } == {(512, 32), (512, 64)}
    narrow = CollectiveConfig(implementation="twoshot_pull", threads=512, blocks=32)
    assert TUNING.configure(query, device=None, override=narrow).pinned == narrow
    with pytest.raises(ValueError, match="outside the native collective"):
        TUNING.configure(
            query,
            device=None,
            override=CollectiveConfig(
                implementation="twoshot_pull", threads=512, blocks=48
            ),
        )
    # A selected pull candidate lowers to the pull declaration and its geometry.
    (job,) = plan._compile_jobs(narrow, None)
    lowered = job.args[0]
    assert lowered["call"]["transport"] == "pull"
    assert lowered["call"]["block_limit"] == 32
    with _plan_scope(plan):
        pull_memory = plan._memory_requirements(narrow, None)
        push_memory = plan._memory_requirements(
            CollectiveConfig(implementation="twoshot", threads=512, blocks=64), None
        )
    slab = candidates["twoshot"]["setup"]["slab_nbytes"]
    assert [entry.required_nbytes for entry in pull_memory.persistent] == [slab]
    assert sorted(entry.required_nbytes for entry in push_memory.persistent) == sorted(
        (slab, (2 << 20) // 4)
    )
    # Aliased output keeps the one-shot candidates only.
    aliased = manager.plan(inp, out=inp)
    assert not {"twoshot", "twoshot_pull"} & set(aliased.query.alternatives)


def test_twoshot_rejects_partially_overlapping_output():
    primary = SimpleNamespace(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        max_size=2 << 20,
        _sentinel_supported=False,
        _transport_policy=(False, True, True, True, True),
    )
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=_runtime(), algorithm_override="auto"
    )
    storage = torch.empty(32768 + 8, dtype=torch.bfloat16)
    assert not manager._use_bf16_twoshot(storage[:-8], storage[8:])
    assert not manager._use_bf16_twoshot(
        storage[:-8], torch.empty(128, 256, dtype=torch.bfloat16)
    )
    assert not manager._use_bf16_twoshot(
        storage[:-8], torch.empty(32768, dtype=torch.float16)
    )


@pytest.mark.parametrize("failure", ("local", "peer", "construction"))
def test_setup_failure_closes_existing_runtime(monkeypatch, failure):
    closed = []
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"), close=lambda: closed.append(True)
    )
    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(
        dispatch.PCIeOneshotAllReducePool,
        "from_exchange_group",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: None
    )

    def properties(device):
        if failure == "local":
            raise RuntimeError("injected capability failure")
        return SimpleNamespace(multi_processor_count=64)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)

    def exchange(local_error, **kwargs):
        if failure == "local":
            assert str(local_error) == "injected capability failure"
            return ((str(local_error),), (), (), ())
        assert local_error is None
        return (
            ((), ("injected peer capability failure",), (), ())
            if failure == "peer"
            else ((),) * 4
        )

    monkeypatch.setattr(oneshot, "_exchange_setup_failures", exchange)

    def gather(count, group):
        assert failure == "construction", (
            "SM exchange must follow successful capability setup"
        )
        return [count] * 4

    monkeypatch.setattr(dispatch, "_broadcast_gather_object", gather)

    def construct(**kwargs):
        raise RuntimeError("injected construction failure")

    monkeypatch.setattr(dispatch.PCIeTwoShotBF16, "from_exchange_group", construct)
    with pytest.raises(RuntimeError, match="injected"):
        dispatch.PCIeAllReduce.from_exchange_group(
            exchange_group=object(), device="cuda:0", single_channel=True
        )
    assert closed == [True]


@pytest.mark.parametrize(
    "counts, expected", (([64] * 4, True), ([64, 63, 64, 64], False))
)
def test_capability_requires_resident_grid_on_every_rank(monkeypatch, counts, expected):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(multi_processor_count=64),
    )
    monkeypatch.setattr(
        oneshot, "_exchange_setup_failures", lambda *args, **kwargs: ((),) * 4
    )
    monkeypatch.setattr(dispatch, "_broadcast_gather_object", lambda *args: counts)
    assert (
        dispatch._bf16_twoshot_supported(
            exchange_group=object(), device="cuda:0", world_size=4
        )
        == expected
    )


@pytest.mark.parametrize("world", (2, 3, 4, 6, 8, 10))
@pytest.mark.parametrize("capacity", (16, 65552, (2 << 20) + 16))
def test_direct_manager_provisions_twoshot_for_full_capacity(
    monkeypatch, world, capacity
):
    runtime = SimpleNamespace(rank=0, world_size=world, device=torch.device("cpu"))
    provisioned = []
    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: world)
    monkeypatch.setattr(
        dispatch.PCIeOneshotAllReducePool,
        "from_exchange_group",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: None
    )
    monkeypatch.setattr(dispatch, "_bf16_twoshot_supported", lambda **kwargs: True)

    def construct(**kwargs):
        provisioned.append(kwargs)
        return object()

    monkeypatch.setattr(dispatch.PCIeTwoShotBF16, "from_exchange_group", construct)
    manager = dispatch.PCIeAllReduce.from_exchange_group(
        exchange_group=object(),
        device="cpu",
        single_channel=True,
        max_size=capacity,
        eager_buffer_bytes=capacity,
    )
    assert manager.algorithm == "oneshot"
    assert provisioned[0]["max_rows"] == (capacity + 15) // 16
    assert provisioned[0]["row_elems"] == 8


@pytest.mark.parametrize("world", (2, 3, 6, 8))
def test_provisioning_preserves_disabled_tuning_default(world):
    primary = SimpleNamespace(rank=0, world_size=world, device=torch.device("cpu"))
    manager = dispatch.PCIeAllReduce(
        primary, "oneshot", bf16_twoshot=object(), algorithm_override="auto"
    )
    assert not manager._use_bf16_twoshot(torch.empty(1, dtype=torch.bfloat16), None)


@pytest.mark.parametrize("failure", (None, "mapping", "unrelated", "cleanup"))
def test_tp10_auto_uses_actual_mapping_verdict(monkeypatch, failure):
    calls = []
    direct = SimpleNamespace(rank=0, world_size=10, device=torch.device("cpu"))
    hierarchical = SimpleNamespace(rank=0, world_size=10, device=torch.device("cpu"))
    error = (
        dispatch.DirectMappingUnavailable("peer mappings unavailable")
        if failure == "mapping"
        else RuntimeError("retained cleanup" if failure == "cleanup" else "unrelated")
    )
    if failure == "cleanup":
        error.retryable_setup = object()

    def construct(**kwargs):
        calls.append("direct")
        if failure:
            raise error
        return direct

    def hierarchy(**kwargs):
        calls.append("hierarchical")
        return hierarchical

    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 10)
    monkeypatch.setattr(dispatch, "_algorithm_override", lambda: "auto")
    monkeypatch.setattr(
        dispatch.PCIeOneshotAllReducePool, "from_exchange_group", construct
    )
    monkeypatch.setattr(dispatch, "PCIeHierarchicalAllReduce", hierarchy)
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: None
    )
    monkeypatch.setattr(dispatch, "_bf16_twoshot_supported", lambda **kwargs: False)
    if failure in ("unrelated", "cleanup"):
        with pytest.raises(RuntimeError) as caught:
            dispatch.PCIeAllReduce.from_exchange_group(
                exchange_group=object(), device="cpu", single_channel=True
            )
        assert caught.value is error
        assert calls == ["direct"]
    else:
        manager = dispatch.PCIeAllReduce.from_exchange_group(
            exchange_group=object(), device="cpu", single_channel=True
        )
        assert manager._runtime is (hierarchical if failure else direct)
        assert manager.algorithm == ("hierarchical" if failure else "oneshot")
        assert calls == (["direct", "hierarchical"] if failure else ["direct"])


@pytest.mark.parametrize(
    "override, single_channel",
    (("hierarchical", True), ("island_rs", True), ("auto", False)),
)
def test_tp10_unqualified_declarations_preserve_hierarchy(
    monkeypatch, override, single_channel
):
    runtime = SimpleNamespace(rank=0, world_size=10, device=torch.device("cpu"))
    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 10)
    monkeypatch.setattr(dispatch, "_algorithm_override", lambda: override)
    monkeypatch.setattr(dispatch, "PCIeHierarchicalAllReduce", lambda **kwargs: runtime)
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: None
    )

    def unexpected(**kwargs):
        pytest.fail("direct provisioning must not run")

    monkeypatch.setattr(
        dispatch.PCIeOneshotAllReducePool, "from_exchange_group", unexpected
    )
    manager = dispatch.PCIeAllReduce.from_exchange_group(
        exchange_group=object(), device="cpu", single_channel=single_channel
    )
    assert manager._runtime is runtime
    assert manager.algorithm == "hierarchical"
