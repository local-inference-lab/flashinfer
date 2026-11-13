"""Island calls preserve physical grids, coordinated setup, and storage contracts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import _owner_preparation, _preparation, pcie_island_rs
from b12x.comm.pcie import pcie_oneshot
from b12x.comm.pcie._tuning import PcieConfig, PcieQuery
from b12x.preparation import FrozenMapping


@pytest.fixture(autouse=True)
def no_cuda_initialization(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("owner contract checks must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", refuse)


def _owner_and_query():
    owner = object.__new__(pcie_island_rs.PCIeIslandRSAllReduce)
    owner.rank, owner.world_size = 0, 16
    owner.device, owner.blocks = torch.device("cpu"), 64
    query = PcieQuery(
        surface="PCIeIslandRSAllReduce.all_reduce",
        world_size=16,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(dict(threads=512, blocks=64, active_blocks=16)),
        setup=FrozenMapping(),
    )
    return owner, query


def _bind(owner, query, direct):
    if direct:
        return _owner_preparation.plan(query, runtime=owner)
    return _preparation.plan(
        query, runtime=owner, override=PcieConfig(backend="native")
    )


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("blocks", [16, 32])
def test_replaced_query_cannot_change_the_owner_physical_grid(direct, blocks):
    owner, query = _owner_and_query()
    replaced = replace(query, call=FrozenMapping(dict(query.call, blocks=blocks)))
    with pytest.raises(ValueError, match="fixed physical grid"):
        _bind(owner, replaced, direct)


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("active", [1, 3, 16, 64])
def test_matching_grid_binds_independent_active_participation(direct, active):
    owner, query = _owner_and_query()
    query = replace(query, call=FrozenMapping(dict(query.call, active_blocks=active)))
    assert _bind(owner, query, direct).query == query


@pytest.mark.parametrize(
    "input_start,output_start,reject",
    [(0, 64, True), (64, 0, True), (0, 0, False), (0, 512, False)],
    ids=["forward-overlap", "backward-overlap", "exact-alias", "disjoint-views"],
)
@pytest.mark.parametrize("entry_point", ["all_reduce", "prepared_call"])
def test_output_storage_contract(
    monkeypatch, input_start, output_start, reject, entry_point
):
    storage = torch.empty(1024, dtype=torch.bfloat16)
    source = storage[input_start : input_start + 512]
    output = storage[output_start : output_start + 512]
    owner, query = _owner_and_query()
    owner._closed, owner.max_elements = False, source.numel()
    query = replace(
        query,
        call=FrozenMapping(dict(query.call, inp={"shape": tuple(source.shape)})),
    )
    launcher, plan = object(), object()
    state = _owner_preparation._OwnerState(query, owner, launcher)
    launches = []

    def prepared(actual_plan, component, device):
        assert actual_plan is plan
        assert component == "comm.pcie" and device == owner.device
        return state

    def run(actual_launcher, inp, out, blocks, stream, *, active_blocks):
        assert actual_launcher is launcher
        assert inp is source and out is output
        assert blocks == 64 and active_blocks == 16
        launches.append(out)
        return out

    monkeypatch.setattr(pcie_island_rs, "require_prepared", prepared)
    monkeypatch.setattr(owner, "_run_prepared", run)

    def invoke():
        if entry_point == "all_reduce":
            return owner.all_reduce(source, out=output, plan=plan)
        call = _owner_preparation.prepared_call(state, inp=source, out=output)
        assert call.output is output and call.capture_safe
        return call.run()

    if reject:
        with pytest.raises(ValueError, match="overlap|alias"):
            invoke()
        assert not launches
    else:
        assert invoke() is output
        assert len(launches) == 1


@pytest.mark.parametrize("local_failure", [False, True])
def test_sm_query_failures_are_exchanged_before_geometry_or_ipc(
    monkeypatch, local_failure
):
    group, exchanged = object(), []
    monkeypatch.setattr(pcie_island_rs.dist, "get_rank", lambda **kwargs: 0)
    monkeypatch.setattr(pcie_island_rs.dist, "get_world_size", lambda **kwargs: 16)

    def properties(device):
        if local_failure:
            raise RuntimeError("local SM query failed")
        return SimpleNamespace(multi_processor_count=188)

    def exchange(value, exchange_group):
        assert exchange_group is group
        exchanged.append(value)
        statuses = [()] * 16
        statuses[0 if local_failure else 7] = (
            value if local_failure else ("RuntimeError: peer SM query failed",)
        )
        return statuses

    def premature(*args, **kwargs):
        pytest.fail(
            "failed SM queries must stop before geometry exchange or IPC allocation"
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(pcie_oneshot, "_broadcast_gather_object", exchange)
    monkeypatch.setattr(pcie_island_rs.dist, "all_gather_object", premature)
    monkeypatch.setattr(pcie_island_rs, "CudaRTLibrary", premature)
    with pytest.raises(RuntimeError, match="capability query pre-allocation setup"):
        pcie_island_rs.PCIeIslandRSAllReduce(
            exchange_group=group, device=torch.device("cuda", 0), max_elements=1024
        )
    assert len(exchanged) == 1
    assert bool(exchanged[0]) is local_failure
