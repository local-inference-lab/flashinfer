"""Output compatibility and coordinated setup for the PCIe dispatcher."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_allreduce as dispatch
from b12x.comm.pcie import pcie_oneshot as oneshot
from b12x.comm.pcie.pcie_twoshot_bf16 import PCIeTwoShotBF16


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


def _oneshot_plan_runtime(world):
    native = SimpleNamespace(
        eager_buffer_bytes=16384, transport_policy=(False, False, True, True, False)
    )
    runtime = SimpleNamespace(
        rank=0,
        world_size=world,
        device=torch.device("cpu"),
        single_channel=True,
        rank_data=torch.empty(128, dtype=torch.uint8),
        _ptr=1,
        _ext=SimpleNamespace(_state=lambda ptr: native, meta_size=lambda: 256),
        _sentinel_supported=True,
        _owned_buffers=[object()],
    )
    runtime._prepared_channel_for_stream = lambda stream, channel: runtime
    return runtime


@pytest.mark.parametrize("world", (2, 4))
def test_public_plan_exposes_static_transports(monkeypatch, world):
    from b12x.comm.pcie import _oneshot_preparation, _preparation
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.preparation import FrozenMapping, autotune_effort

    runtime = _oneshot_plan_runtime(world)
    rank_bytes, slab_bytes, slabs = _oneshot_preparation._owned_resident_layout(
        runtime, _oneshot_preparation._state(runtime)
    )
    manager = dispatch.PCIeAllReduce(runtime, "oneshot")
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=world,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                transport="pull",
                dtype="bfloat16",
                threads=256,
                blocks=4,
                eager_threads=256,
                eager_blocks=4,
                size_packs=32,
                stage_input=True,
                remote_push_region_packs=1024,
                payload_ready_region_packs=(1 + 2 * (world - 1)) * 1024,
            )
        ),
        setup=FrozenMapping(
            dict(
                eager_buffer_bytes=16384,
                eager_storage_shards=1 + 2 * (world - 1) + world,
                variants=((False, 0), (True, 0), (True, 1)),
                rank_data_nbytes=rank_bytes,
                owned_slab_nbytes=slab_bytes,
                owned_slab_count=slabs,
            )
        ),
    )
    monkeypatch.setattr(_preparation, "query_from_runtime", lambda *a, **k: query)
    plan = manager.plan(torch.empty(256, dtype=torch.bfloat16))
    expected = {
        "pull",
        "remote_push_ll",
        "remote_push_simple",
        "remote_push_full",
        "remote_push_rsag",
    }
    if world == 2:
        expected |= {"tp2_remote_push", "tp2_remote_push_simple"}
    assert set(plan.query.alternatives) == expected
    assert plan.query.default.implementation == "pull"
    with autotune_effort("default"):
        configs = tuple(
            config
            for _, config in plan.contract.iterate(
                plan.contract.configure(plan.query, device=None)
            )
        )
    assert {config.implementation for config in configs} == expected
    assert len(configs) > len(expected)


@pytest.mark.parametrize("cc,sms", [((12, 0), 148), ((12, 1), 148), ((12, 0), 32)])
@pytest.mark.parametrize("transport", ["pull", "push"])
@pytest.mark.parametrize("prefer_twoshot", [False, True])
def test_tp4_plan_only_offers_pairwise_for_push(
    monkeypatch, transport, prefer_twoshot, cc, sms
):
    from b12x.comm.pcie import _preparation, _collective_tuning
    from b12x.comm.pcie._tuning import PcieQuery
    from b12x.preparation import DeviceIdentity, FrozenMapping, autotune_effort

    device = DeviceIdentity("nvidia", cc, sms, "synthetic device")
    monkeypatch.setattr(
        dispatch, "detect_device", lambda d: SimpleNamespace(identity=device)
    )
    runtime = _oneshot_plan_runtime(4)
    twoshot = SimpleNamespace(
        accepts=lambda inp: True, rank=0, world_size=4, device=runtime.device
    )
    manager = dispatch.PCIeAllReduce(runtime, "oneshot", bf16_twoshot=twoshot)
    monkeypatch.setattr(manager, "_use_bf16_twoshot", lambda *args: prefer_twoshot)

    def query(owner, *, surface, call):
        shape = tuple(call["inp"].shape)
        if owner is twoshot:
            setup = dict(slab_nbytes=16384, push_shard_nbytes=512)
        else:
            setup = dict(shape=shape)
        return PcieQuery(
            surface=surface,
            world_size=4,
            rank=0,
            topology="pcie_ipc",
            call=FrozenMapping(
                dict(
                    transport=transport if owner is twoshot else "pull",
                    stage_input=True,
                    threads=128,
                    blocks=8,
                    block_limit=8,
                    dtype="bfloat16",
                    shape=shape,
                    row_elems=4096,
                    size_packs=call["inp"].numel() // 8,
                )
            ),
            setup=FrozenMapping(setup),
        )

    monkeypatch.setattr(_preparation, "query_from_runtime", query)
    monkeypatch.setattr(
        _collective_tuning, "oneshot_alternatives", lambda q, r: {"pull": (q, r)}
    )
    plan = manager.plan(torch.empty(256, dtype=torch.bfloat16))
    admitted = transport == "push" and cc == (12, 0) and sms >= 64
    assert ("twoshot_pairwise" in plan.query.alternatives) == admitted
    assert "twoshot" in plan.query.alternatives
    assert "pull" in plan.query.alternatives
    bucket = plan.contract.collective_bucket(plan.query)
    assert bucket.size_bytes == 512
    assert bucket.grid.upper_bytes == 512
    assert set(bucket.identity["alternatives"]) == set(plan.query.alternatives)
    larger = manager.plan(torch.empty(512, dtype=torch.bfloat16))
    larger_bucket = larger.contract.collective_bucket(larger.query)
    assert larger_bucket.size_bytes == 1024
    assert larger_bucket.identity == bucket.identity
    expected = "twoshot" if prefer_twoshot else "pull"
    assert plan.query.default.implementation == expected
    with autotune_effort("default"):
        configs = [
            c
            for _, c in plan.contract.iterate(
                plan.contract.configure(plan.query, device=device)
            )
        ]
    assert {c.implementation for c in configs} == set(plan.query.alternatives)
    with autotune_effort("off"):
        configured = plan.contract.configure(plan.query, device=device)
    assert configured.default.implementation == expected


@pytest.mark.parametrize("attach_island", (True, False))
def test_tp16_island_rs_caps_hierarchical_slab(monkeypatch, attach_island):
    seen = {}
    island_closed = []
    island = (
        SimpleNamespace(
            rank=0,
            world_size=16,
            device=torch.device("cpu"),
            close=lambda: island_closed.append(True),
        )
        if attach_island
        else None
    )

    def hierarchy(**kwargs):
        seen["max_elements"] = kwargs["max_elements"]
        return SimpleNamespace(rank=0, world_size=16, device=torch.device("cpu"))

    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(dispatch, "_algorithm_override", lambda: "island_rs")
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: island
    )
    monkeypatch.setattr(dispatch, "PCIeHierarchicalAllReduce", hierarchy)
    max_size = 28 << 20
    manager = dispatch.PCIeAllReduce.from_exchange_group(
        exchange_group=object(), device="cpu", max_size=max_size
    )
    expected = dispatch.ISLAND_RS_MAX_BYTES // 2 if attach_island else max_size // 2
    assert seen["max_elements"] == expected
    assert manager._island_rs is island
    assert island_closed == []


def test_tp16_hierarchical_failure_closes_attached_island(monkeypatch):
    island_closed = []
    island = SimpleNamespace(
        rank=0,
        world_size=16,
        device=torch.device("cpu"),
        close=lambda: island_closed.append(True),
    )

    def fail_hierarchy(**kwargs):
        raise RuntimeError("injected hierarchical failure")

    monkeypatch.setattr(dispatch.dist, "get_world_size", lambda group: 16)
    monkeypatch.setattr(dispatch, "_algorithm_override", lambda: "island_rs")
    monkeypatch.setattr(
        dispatch.PCIeAllReduce, "_maybe_island_rs", lambda **kwargs: island
    )
    monkeypatch.setattr(dispatch, "PCIeHierarchicalAllReduce", fail_hierarchy)
    with pytest.raises(RuntimeError, match="injected hierarchical failure"):
        dispatch.PCIeAllReduce.from_exchange_group(
            exchange_group=object(), device="cpu", max_size=28 << 20
        )
    assert island_closed == [True]
