"""Collective candidate metadata stays bounded without changing prepared owners."""

from types import SimpleNamespace
import weakref

import pytest
import torch

from b12x.comm.pcie import _collective_tuning, _joint_preparation
from b12x.comm.pcie._joint_profile import JOINT_FAMILIES
from b12x.comm.pcie._tuning import PcieQuery
from b12x.preparation import DetectedDevice, FrozenMapping
from b12x.preparation.types import _plan_scope


def declaration(family, runtime=None):
    if runtime is None:
        runtime = SimpleNamespace(device=torch.device("cpu"), _joint_workspace=object())
    query = PcieQuery(
        surface="OneshotAllReduce.all_reduce",
        world_size=4,
        rank=0,
        topology="pcie_ipc",
        call=FrozenMapping(
            dict(
                transport=family,
                threads=256,
                blocks=64,
                active_blocks=16,
                submode=JOINT_FAMILIES[family][0],
                dtype="bfloat16",
                stage_input=True,
                size_packs=128,
            )
        ),
        setup=FrozenMapping(
            dict(
                joint_max_blocks=64,
                rank_data_nbytes=128,
                owned_slab_nbytes=4096,
                owned_slab_count=1,
                joint_workspace_nbytes=16384,
                variants=((True, 0), (True, 1)),
            )
        ),
    )
    return _collective_tuning.plan({family: (query, runtime)}, default=family), runtime


@pytest.mark.parametrize("family", JOINT_FAMILIES)
def test_candidate_metadata_is_bounded_and_revisit_preserves_compilation_and_memory(
    monkeypatch, family
):
    native = _joint_preparation.plan
    references = []

    def observe(*args, **kwargs):
        value = native(*args, **kwargs)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(_joint_preparation, "plan", observe)
    logical, runtime = declaration(family)
    device = DetectedDevice(None, None)
    configs = tuple(_collective_tuning._configs(logical.query))
    with _plan_scope(logical):
        jobs = logical._compile_jobs(configs[0], device)
        memory = logical._memory_requirements(configs[0], device)
        assert len(references) == 1
        for config in configs[1:]:
            logical._memory_requirements(config, device)
        assert sum(reference() is not None for reference in references) == 1
        assert logical._compile_jobs(configs[0], device) == jobs
        assert logical._memory_requirements(configs[0], device) == memory
        assert len(references) == len(configs) + 1
        assert sum(reference() is not None for reference in references) == 1

    other, _ = declaration(family, runtime)
    with _plan_scope(other):
        other_memory = other._memory_requirements(configs[0], device)
    assert memory.persistent[0] == other_memory.persistent[0]
    assert memory.persistent[1].key != other_memory.persistent[1].key


@pytest.mark.parametrize("family", JOINT_FAMILIES)
def test_materialized_selectors_and_programs_survive_candidate_metadata_eviction(
    monkeypatch, family
):
    from b12x._lib.compile_plan import (
        CompiledCuTeProgram,
        ProgramKey,
        retain_compiled_programs,
    )

    references = []

    def executable(*args):
        program = CompiledCuTeProgram(lambda: None, ProgramKey("cute", "a" * 64))
        references.append(weakref.ref(program))
        return program

    monkeypatch.setattr(_joint_preparation, "compile_joint", executable)
    first, runtime = declaration(family)
    second, _ = declaration(family, runtime)
    config = first.query.default
    selection = SimpleNamespace(config=config)
    device = DetectedDevice(None, None)
    with retain_compiled_programs() as retained:
        with _plan_scope(first):
            a = first._materialize(selection, device)
            for candidate in _collective_tuning._configs(first.query):
                first._memory_requirements(candidate, device)
        with _plan_scope(second):
            b = second._materialize(selection, device)
    assert a.runtime is b.runtime is runtime
    assert a.selector.data_ptr() != b.selector.data_ptr()
    assert a.selector.item() == b.selector.item()
    assert a.query.call["active_blocks"] == config.active_blocks
    assert a.query.call["submode"] == config.submode
    del first, second
    assert all(reference() is not None for reference in references)
    assert {id(a.launcher), id(b.launcher)} <= retained.owners.keys()
