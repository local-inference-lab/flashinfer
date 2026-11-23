"""Collective preparation calibrates payload intervals and validates their winners."""

from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import pytest

from b12x.preparation import (
    CollectiveBucket,
    CollectiveGrid,
    CollectiveRequirement,
    FrozenMapping,
    Knob,
    ParameterSpace,
    PreparedCall,
    autotune_effort,
)
from b12x.preparation import _collective
from .test_defaults import Query, contract
from .test_session import _deterministic_timer, request, session


def coordinate(progress):
    if progress.ready_collective_tuning is not None:
        value = progress.ready_collective_tuning
        return (value,) * len(value.ranks)
    if progress.ready_collectives:
        return progress.ready_collectives[0].key


@pytest.fixture
def host_races(monkeypatch):
    _deterministic_timer(monkeypatch)
    monkeypatch.setattr(_collective, "capture_benchmark_call", lambda call, **kw: call)


@dataclass(frozen=True)
class LaunchConfig:
    width: int
    blocks: int


@pytest.mark.parametrize(
    "changes",
    [
        {"identity": {}},
        {"size_bytes": 0},
        {"size_bytes": True},
        {"size_bytes": 9},
        {"limit": 0},
        {"limit": 1.5},
    ],
)
def test_bucket_rejects_invalid_identity_size_and_limit(changes):
    payload = dict(
        identity={"operation": "arithmetic"},
        size_bytes=5,
        limit=2,
        grid=CollectiveGrid(upper_bytes=8, band=1),
    )
    with pytest.raises(ValueError):
        CollectiveBucket(**(payload | changes))


def test_invalid_bucket_callback_fails_before_racing(tmp_path, host_races):
    validated = []
    (item,) = grid_declarations((5,), validated)
    item = replace(
        item,
        plan=replace(
            item.plan,
            contract=replace(item.plan.contract, collective_bucket=lambda query: ()),
        ),
    )
    with session(tmp_path) as engine:
        with pytest.raises(TypeError, match="CollectiveBucket"):
            engine.prepare((item,), coordinator=coordinate)
    assert validated == []


@pytest.mark.parametrize("mode", ("custom_benchmark", "dependency", "pin"))
def test_custom_benchmark_dependencies_and_pins_bypass_calibration(
    tmp_path, host_races, mode
):
    items = []
    for item in grid_declarations((17, 19), []):
        if mode == "pin":
            item = replace(item, plan=replace(item.plan, override=LaunchConfig(4, 1)))
        elif mode == "dependency":
            item = replace(item, dependencies=("input",))
        else:
            item = replace(
                item, benchmark_call=lambda state, call=item.benchmark_call: call(state)
            )
        items.append(item)
    if mode == "dependency":
        items.insert(
            0,
            request(
                name="input",
                tuning=replace(contract(values=(2,)), component_id="test.input"),
            ),
        )
    with session(tmp_path) as engine:
        result = engine.prepare(items, coordinator=coordinate)
    if mode == "pin":
        assert result.benchmarked_candidates == 0
        assert all(
            choice.config == LaunchConfig(4, 1) for choice in result.selections.values()
        )
    else:
        assert result.benchmarked_candidates == 12


@pytest.mark.parametrize("limit", (6, 8))
def test_small_competing_catalog_reuses_complete_winner(tmp_path, host_races, limit):
    validated = []
    with session(tmp_path) as engine:
        result = engine.prepare(
            grid_declarations((17, 19), validated, limit=limit), coordinator=coordinate
        )
    assert result.benchmarked_candidates == 6
    assert result.coverage["rows-17"]["measured_count"] == 0
    assert (
        result.selections["rows-17"].config
        == result.selections["rows-19"].config
        == LaunchConfig(2, 2)
    )
    assert [(pair, producer) for row, pair, producer in validated if row == 17] == [
        ((2, 2), 1),
        ((2, 2), 2),
    ]


def test_grid_separates_shapes_with_different_eligible_assignments(
    tmp_path, host_races
):
    first, second = grid_declarations((17, 19), [])
    tuning = first.plan.contract
    original_parameters = tuning.parameters

    def parameters(query, device):
        space = original_parameters(query, device)
        return replace(
            space, predicates=(lambda assignment: assignment["blocks"] == 1,)
        )

    first = replace(
        first, plan=replace(first.plan, contract=replace(tuning, parameters=parameters))
    )
    with session(tmp_path) as engine:
        result = engine.prepare((first, second), coordinator=coordinate)
    assert result.coverage["rows-17"]["measured_count"] == 3
    assert result.coverage["rows-19"]["measured_count"] == 6
    assert result.selections["rows-17"].config == LaunchConfig(1, 1)
    assert result.selections["rows-19"].config == LaunchConfig(2, 2)


@pytest.mark.parametrize(
    "choices,match",
    [
        (None, "nonempty"),
        ([], "nonempty"),
        ([{"width": width, "blocks": 1} for width in (1, 2, 4)], "within the limit"),
        ([{"width": 1, "blocks": 1}, {"width": 1, "blocks": 1}], "unique"),
        (
            [{"width": 1, "blocks": 1}, {"width": 1, "blocks": 4}],
            "eligible assignments",
        ),
        ([{"width": 1}], "eligible assignments"),
        ([{"width": 2, "blocks": 1}], "differs from its first choice"),
        (
            [{"width": 2, "blocks": 1}, {"width": 1, "blocks": 1}],
            "differs from its first choice",
        ),
    ],
)
def test_invalid_cached_grid_choices_fail_before_binding(
    tmp_path, host_races, choices, match
):
    import json

    with session(tmp_path) as engine:
        engine.prepare(grid_declarations((16,), []), coordinator=coordinate)
        path = engine._cache.path
    payload = json.loads(path.read_text())
    record = next(iter(payload["records"].values()))
    record["coverage"]["bucket_choices"] = choices
    path.write_text(json.dumps(payload))
    validated = []
    (item,) = grid_declarations((16,), validated)
    with session(tmp_path) as engine:
        with pytest.raises(ValueError, match=match):
            engine.prepare((item,), coordinator=coordinate)
        assert item.plan.prepared is None
    assert validated == []


def test_plain_buckets_follow_bytes_and_preserve_collective_semantics():
    from b12x.comm.pcie._collective_tuning import _bucket

    def query(shape, **changes):
        payload = dict(
            surface="OneshotAllReduce.all_reduce",
            topology="pcie_ipc",
            call=dict(dtype="bfloat16", stage_input=True),
            setup=dict(shape=shape),
        )
        return SimpleNamespace(
            operation="all_reduce",
            world_size=changes.get("world_size", 4),
            inplace=changes.get("inplace", False),
            alternatives={"pull": payload},
        )

    first = _bucket(query((32, 5120)))
    assert first.identity == _bucket(query((36, 5120))).identity
    assert first.identity == _bucket(query((16, 10240))).identity
    assert first.identity == _bucket(query((40, 5120))).identity
    assert first.grid != _bucket(query((40, 5120))).grid
    assert first.grid == _bucket(query((16, 10240))).grid
    assert first.identity != _bucket(query((32, 5120), inplace=True)).identity
    assert first.identity != _bucket(query((32, 5120), world_size=2)).identity
    assert first.grid == CollectiveGrid(upper_bytes=320 * 1024, band=9)
    assert _bucket(query((160, 5120))).grid == CollectiveGrid(
        upper_bytes=1600 * 1024, band=10
    )

    def twoshot(transport):
        native = dict(
            surface="PCIeTwoShotBF16.all_reduce",
            topology="pcie_ipc",
            setup={},
            call=dict(
                dtype="bfloat16", shape=(4096,), row_elems=4096, transport=transport
            ),
        )
        return _bucket(
            SimpleNamespace(
                operation="all_reduce",
                world_size=4,
                inplace=False,
                alternatives={"twoshot": native},
            )
        )

    assert twoshot("push").grid == twoshot("pull").grid
    assert twoshot("push").identity != twoshot("pull").identity

    plain_addend = SimpleNamespace(
        operation="all_reduce",
        world_size=4,
        inplace=False,
        alternatives={
            "remote_push_rsag": dict(
                surface="OneshotAllReduce.all_reduce",
                topology="pcie_ipc",
                call=dict(dtype="bfloat16", stage_input=True, has_addend=True),
                setup=dict(shape=(32, 5120), payload_ready_max_blocks=64),
            )
        },
    )
    plain_no_addend = SimpleNamespace(
        operation="all_reduce",
        world_size=4,
        inplace=False,
        alternatives={
            "remote_push_rsag": dict(
                surface="OneshotAllReduce.all_reduce",
                topology="pcie_ipc",
                call=dict(dtype="bfloat16", stage_input=True),
                setup=dict(shape=(32, 5120), payload_ready_max_blocks=64),
            )
        },
    )
    assert _bucket(plain_addend).grid == _bucket(plain_no_addend).grid
    assert _bucket(plain_addend).identity != _bucket(plain_no_addend).identity


@pytest.mark.parametrize("pairwise", (False, True), ids=("oneshot", "twoshot_pairwise"))
def test_tp4_allreduce_reuses_complete_interval_winner(tmp_path, host_races, pairwise):
    from b12x.comm.pcie._collective_tuning import (
        CollectiveConfig,
        CollectiveQuery,
        TUNING,
    )
    from b12x.preparation import (
        DetectedDevice,
        DeviceIdentity,
        MemoryRequirements,
        Plan,
    )

    validated = []
    items = []
    ranks = tuple(range(4))
    sizes = (4160, 4168) if pairwise else (264, 280, 288, 304)
    preferred = CollectiveConfig(
        implementation="twoshot_pairwise" if pairwise else "remote_push_full",
        threads=128 if pairwise else 64,
        blocks=8 if pairwise else 2,
    )
    for elements in sizes:
        native = dict(world_size=4, rank=0, topology="pcie_ipc")
        if pairwise:
            native.update(surface="PCIeTwoShotBF16.all_reduce", setup={})
            call = dict(
                dtype="bfloat16",
                shape=(elements,),
                threads=128,
                block_limit=8,
                row_elems=4096,
                transport="push",
            )
            alternatives = {
                "twoshot": dict(native, call=call),
                "twoshot_pairwise": dict(
                    native, call=dict(call, payload_schedule="pairwise")
                ),
            }
            default = "twoshot"
        else:
            native.update(
                surface="OneshotAllReduce.all_reduce", setup={"shape": (elements,)}
            )
            alternatives = {
                name: dict(
                    native,
                    call=dict(
                        transport=name,
                        dtype="bfloat16",
                        stage_input=True,
                        size_packs=elements // 8,
                        threads=128,
                        blocks=8,
                    ),
                )
                for name in ("pull", "remote_push_full")
            }
            default = "pull"
        query = CollectiveQuery(
            operation="all_reduce",
            world_size=4,
            rank=0,
            alternatives=FrozenMapping(alternatives),
            default=CollectiveConfig(implementation=default, threads=128, blocks=8),
        )

        def benchmark(state, elements=elements):
            return PreparedCall(
                run=lambda: None,
                output=6 if state == preferred else 7,
                produce=lambda: None,
                validate=lambda: validated.append((elements, state)),
            )

        plan = Plan(
            contract=TUNING,
            query=query,
            _compile_jobs=lambda config, device: (),
            _memory_requirements=lambda config, device: MemoryRequirements(),
            _materialize=lambda selection, device: selection.config,
            _benchmark_call=benchmark,
        )
        name = str(elements)
        items.append(
            plan.request(
                name=name,
                collective=CollectiveRequirement(name, ranks),
                prepare_call=lambda state: PreparedCall(run=lambda: None),
            )
        )

    with session(tmp_path, prime=False) as engine:
        engine.device = DetectedDevice(
            None, DeviceIdentity("nvidia", (12, 0), 148, "test SM120")
        )
        result = engine.prepare(items, coordinator=coordinate)
        follower = 4160 if pairwise else 280
        representative = 4168 if pairwise else 288
        assert result.selections[str(representative)].config == preferred
        assert result.coverage[str(representative)]["measured_count"] > 0
        assert result.coverage[str(follower)]["measured_count"] == 0
        assert result.selections[str(follower)].config == preferred
        assert [config for size, config in validated if size == follower] == [preferred]
        if not pairwise:
            lower = result.coverage["264"]["measured_count"]
            upper = result.coverage["304"]["measured_count"]
            assert 0 < result.coverage["288"]["measured_count"] < lower == upper


def test_finalist_policy_preserves_full_families_and_cached_candidate_indices(
    tmp_path, host_races
):
    from b12x.preparation import FinalistPolicy

    items = tuple(
        replace(item, finalist_policy=FinalistPolicy(family_fields=("width", "blocks")))
        for item in grid_declarations((17, 19), [])
    )
    original = None
    for cached in (False, True):
        with session(tmp_path) as engine:
            result = engine.prepare(items, coordinator=coordinate)
            assert result.benchmarked_candidates == (0 if cached else 12)
            assert result.cache_hits == (2 if cached else 0)
            for name, finalists in result.finalists.items():
                assert {item.selection.config.width for item in finalists} == {1, 2, 4}
                assert {item.candidate_index for item in finalists} == set(range(6))
                assert "bucket_choices" not in result.coverage[name]
            if cached:
                assert result.finalists == {
                    name: tuple(
                        replace(
                            item, selection=replace(item.selection, source="cached")
                        )
                        for item in finalists
                    )
                    for name, finalists in original.items()
                }
            else:
                original = result.finalists


def grid_declarations(rows, validated, *, failing=None, limit=2):
    from b12x.preparation._efficiency import capture_autotune_effort

    knobs = (Knob(name="width", values=(1, 2, 4)), Knob(name="blocks", values=(1, 2)))
    tuning = replace(
        contract(),
        config_fields=frozenset(("width", "blocks")),
        encode_config=asdict,
        decode_config=lambda payload: LaunchConfig(**payload),
        validate_config=lambda query, config, device: None,
        default_config=lambda query, device: LaunchConfig(2, 1),
        knobs=knobs,
        parameters=lambda query, device: ParameterSpace(
            knobs=knobs, autotune_effort=capture_autotune_effort()
        ),
        materialize=lambda query, device, assignment: LaunchConfig(**assignment),
        collective_signature=lambda query: query.rows,
        collective_bucket=lambda query: CollectiveBucket(
            identity={"operation": "arithmetic"},
            size_bytes=query.rows,
            limit=limit,
            grid=CollectiveGrid(
                upper_bytes=-(-query.rows // 4) * 4,
                band=(query.rows.bit_length() - 1) // 2,
            ),
        ),
    )

    def materialize(selection, device):
        row = selection.query["rows"]
        pair = (selection.config.width, selection.config.blocks)
        preferred = ((1, 1), (2, 1)) if row <= 16 else ((4, 2), (2, 2))
        if 16 < row <= 20:
            preferred = ((2, 2), (1, 1))
        value = 6 + (preferred.index(pair) if pair in preferred else 10)
        return SimpleNamespace(value=value, rows=row, pair=pair, produced=None)

    def benchmark(state):
        def produce(value):
            state.produced = value

        def validate():
            validated.append((state.rows, state.pair, state.produced))
            if state.rows == failing:
                raise ValueError("selected configuration failed exact-query validation")

        return PreparedCall(
            run=lambda: state.value,
            output=state.value,
            produce=lambda: produce(1),
            benchmark_producers=(lambda: produce(1), lambda: produce(2)),
            validate=validate,
        )

    items = []
    for row in rows:
        item = request(
            name=f"rows-{row}",
            tuning=tuning,
            benchmark=benchmark,
            collective=CollectiveRequirement(f"group/{row}", (0, 1)),
        )
        items.append(
            replace(
                item,
                plan=replace(
                    item.plan,
                    query=Query(row),
                    _materialize=materialize,
                    _benchmark_call=benchmark,
                ),
            )
        )
    return tuple(items)


@pytest.mark.parametrize("reverse", (False, True))
def test_grid_calibrates_anchors_and_reuses_validated_interval_winners(
    tmp_path, host_races, reverse
):
    rows = (16, 17, 19, 21, 22, 24)
    if reverse:
        rows = rows[::-1]
    for cached in (False, True):
        validated = []
        catalogs = []

        def record(progress):
            value = progress.ready_collective_tuning
            if value is not None and value.step == "catalog":
                catalogs.append(
                    (value.key, value.payload["grid"], len(value.payload["candidates"]))
                )
            return coordinate(progress)

        with session(tmp_path, prime=False) as engine:
            result = engine.prepare(
                grid_declarations(rows, validated), coordinator=record
            )
            assert result.benchmarked_candidates == (0 if cached else 16)
            assert result.cache_hits == (6 if cached else 0)
            assert [key for key, _, _ in catalogs] == [
                "group/16",
                "group/24",
                "group/19",
                "group/17",
                "group/21",
                "group/22",
            ]
            assert all(count == 6 for _, _, count in catalogs)
            assert [grid["role"] for _, grid, _ in catalogs] == [
                "full",
                "full",
                "refined",
                "reused",
                "reused",
                "reused",
            ]
            assert catalogs[2][1]["parents"] == ("group/16", "group/24")
            assert result.selections["rows-16"].config == LaunchConfig(1, 1)
            assert result.selections["rows-19"].config == LaunchConfig(2, 2)
            assert result.selections["rows-17"].config == LaunchConfig(2, 2)
            assert result.selections["rows-21"].config == LaunchConfig(4, 2)
            assert result.selections["rows-22"].config == LaunchConfig(4, 2)
            for row in (17, 21, 22):
                assert result.coverage[f"rows-{row}"]["measured_count"] == 0
            if cached:
                assert validated == []
            else:
                assert [
                    (pair, producer) for row, pair, producer in validated if row == 17
                ] == [((2, 2), 1), ((2, 2), 2)]
                assert len(validated) == (16 + 3) * 2


@pytest.mark.parametrize("changes", ({"upper_bytes": 0}, {"band": -1}, {"band": True}))
def test_grid_rejects_invalid_intervals(changes):
    with pytest.raises(ValueError):
        CollectiveGrid(**({"upper_bytes": 32, "band": 2} | changes))


def test_grid_validation_failure_does_not_cache_the_follower(tmp_path, host_races):
    validated = []
    with session(tmp_path, prime=False) as engine:
        with pytest.raises(ValueError, match="exact-query validation"):
            engine.prepare(
                grid_declarations((16, 17, 19, 24), validated, failing=17),
                coordinator=coordinate,
            )
    validated.clear()
    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(
            grid_declarations((16, 17, 19, 24), validated), coordinator=coordinate
        )
        assert result.selections["rows-17"].source == "tuned"
        assert [(pair, producer) for row, pair, producer in validated if row == 17] == [
            ((2, 2), 1),
            ((2, 2), 2),
        ]


def test_grid_agrees_on_calibration_role_before_binding(tmp_path, host_races):
    validated = []

    def disagree(progress):
        value = progress.ready_collective_tuning
        if value is not None and value.step == "catalog":
            peer = replace(
                value,
                payload=FrozenMapping(
                    {**value.payload, "grid": {"role": "reused", "parents": ("other",)}}
                ),
            )
            return value, peer
        return coordinate(progress)

    with session(tmp_path, prime=False) as engine:
        with pytest.raises(ValueError, match="disagree on grid"):
            engine.prepare(
                grid_declarations((16, 17, 19, 24), validated), coordinator=disagree
            )
    assert validated == []


def test_grid_parent_program_change_invalidates_only_dependent_intervals(
    tmp_path, host_races, monkeypatch
):
    import importlib
    from b12x._lib.compile_plan import ProgramKey, record_program
    from b12x._lib.compile_pool import CompilationPlan, CompileJob

    module = importlib.import_module("b12x.preparation.session")
    available = set()
    changed = False

    def program(row, width, blocks):
        version = int(changed and (row, width, blocks) == (16, 4, 1))
        return ProgramKey("cute", f"grid-{row}-{width}-{blocks}-{version}")

    def build(plans):
        for plan in plans:
            available.update(plan.programs)

    monkeypatch.setattr(
        module,
        "describe_compilation",
        lambda job: CompilationPlan(job, (program(*job.args),)),
    )
    monkeypatch.setattr(
        module, "compiled_program_available", lambda key: key in available
    )
    monkeypatch.setattr(module, "compile_in_process", build)

    def prepare():
        items = []
        for item in grid_declarations((16, 17, 19, 21, 22, 24), []):
            row = item.plan.query.rows
            factory = item.benchmark_call
            original = item.plan._materialize

            def materialize(selection, device, original=original, row=row):
                state = original(selection, device)
                state.program = program(row, *state.pair)
                return state

            def benchmark(state, factory=factory):
                call = factory(state)

                def run():
                    assert state.program in available
                    record_program(state.program)
                    return state.value

                return replace(call, run=run)

            items.append(
                replace(
                    item,
                    benchmark_call=benchmark,
                    plan=replace(
                        item.plan,
                        _materialize=materialize,
                        _benchmark_call=benchmark,
                        _compile_jobs=lambda config, device, row=row: (
                            CompileJob.create(
                                "test.compiler:compile",
                                row,
                                config.width,
                                config.blocks,
                            ),
                        ),
                    ),
                )
            )
        with session(tmp_path, prime=False, compile_workers=0) as engine:
            return engine.prepare(items, coordinator=coordinate)

    assert prepare().benchmarked_candidates == 16
    assert prepare().cache_hits == 6
    changed = True
    result = prepare()
    assert result.benchmarked_candidates == 10
    assert result.cache_hits == 3
    assert {
        name
        for name, selected in result.selections.items()
        if selected.source == "tuned"
    } == {"rows-16", "rows-19", "rows-17"}


def test_grid_parent_shortlist_change_invalidates_dependent_choices(
    tmp_path, host_races
):
    import json

    rows = (16, 17, 19, 21, 22, 24)
    with session(tmp_path, prime=False) as engine:
        engine.prepare(grid_declarations(rows, []), coordinator=coordinate)
        path = engine._cache.path
    payload = json.loads(path.read_text())
    anchor = next(
        record
        for record in payload["records"].values()
        if record["assignment"] == {"width": 1, "blocks": 1}
    )
    anchor["coverage"]["bucket_choices"][1] = {"width": 4, "blocks": 1}
    path.write_text(json.dumps(payload))
    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(grid_declarations(rows, []), coordinator=coordinate)
        assert result.benchmarked_candidates == 4
        assert result.cache_hits == 4
        assert result.selections["rows-19"].source == "tuned"
        assert result.selections["rows-17"].source == "tuned"


def test_grid_cancellation_does_not_publish_a_reused_choice(tmp_path, host_races):
    rows = (16, 17, 19, 24)
    cancelled = False

    def cancel(progress):
        nonlocal cancelled
        value = progress.ready_collective_tuning
        if (
            value is not None
            and value.key == "group/17"
            and value.step.startswith("validated/")
        ):
            cancelled = True
            peer = replace(
                value, payload=FrozenMapping({**value.payload, "cancel": True})
            )
            return value, peer
        return coordinate(progress)

    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(grid_declarations(rows, []), coordinator=cancel)
        assert cancelled
        assert result.selections["rows-17"].source == "default"
    validated = []
    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(
            grid_declarations(rows, validated), coordinator=coordinate
        )
        assert result.benchmarked_candidates == 0
        assert result.cache_hits == 3
        assert result.selections["rows-17"].source == "tuned"
        assert [producer for row, _, producer in validated if row == 17] == [1, 2]


def test_grid_rank_membership_separates_calibration_families(tmp_path, host_races):
    items = grid_declarations((16, 17, 19, 24), [])
    items = tuple(
        replace(item, collective=replace(item.collective, ranks=(1, 2)))
        if item.plan.query.rows == 17
        else item
        for item in items
    )
    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(items, coordinator=coordinate)
        assert result.coverage["rows-17"]["measured_count"] == 6
        assert result.benchmarked_candidates == 22


def test_grid_uses_global_minimum_and_band_maxima_as_full_anchors(tmp_path, host_races):
    catalogs = []

    def record(progress):
        value = progress.ready_collective_tuning
        if value is not None and value.step == "catalog":
            catalogs.append((value.key, value.payload["grid"]))
        return coordinate(progress)

    with session(tmp_path, prime=False) as engine:
        result = engine.prepare(
            grid_declarations((68, 16, 64, 24, 20, 72), []), coordinator=record
        )
    assert [key for key, plan in catalogs if plan["role"] == "full"] == [
        "group/16",
        "group/24",
        "group/72",
    ]
    assert [plan["parents"] for _, plan in catalogs if plan["role"] == "refined"] == [
        ("group/16", "group/24"),
        ("group/24", "group/72"),
        ("group/24", "group/72"),
    ]
    assert result.benchmarked_candidates == 26


@pytest.mark.parametrize("effort", ("audit", "off"))
def test_grid_preserves_full_audit_and_explicit_untuned_preparation(
    tmp_path, host_races, effort
):
    with autotune_effort(effort), session(tmp_path, prime=False) as engine:
        result = engine.prepare(
            grid_declarations((16, 17, 19, 24), []), coordinator=coordinate
        )
    assert result.benchmarked_candidates == (24 if effort == "audit" else 0)
    assert all("bucket_choices" not in value for value in result.coverage.values())


def test_grid_rejects_divergent_parent_records_before_refinement(tmp_path, host_races):
    validated = []

    def disagree(progress):
        value = progress.ready_collective_tuning
        if value is not None and value.key == "group/19" and value.step == "cache":
            parents = value.payload["grid_parents"]
            peer = replace(
                value,
                payload=FrozenMapping(
                    {
                        **value.payload,
                        "grid_parents": (("other", parents[0][1]), *parents[1:]),
                    }
                ),
            )
            return value, peer
        return coordinate(progress)

    with session(tmp_path, prime=False) as engine:
        with pytest.raises(ValueError, match="disagree on grid_parents"):
            engine.prepare(
                grid_declarations((16, 17, 19, 24), validated), coordinator=disagree
            )
    assert {row for row, _, _ in validated} == {16, 24}


def test_grid_cache_only_reuses_equivalent_renamed_bindings(tmp_path, host_races):
    rows = (16, 17, 19, 21, 22, 24)
    with session(tmp_path, prime=False) as engine:
        original = engine.prepare(grid_declarations(rows, []), coordinator=coordinate)
    validated = []
    renamed = tuple(
        replace(
            item,
            name="alias/" + item.name,
            collective=replace(item.collective, key="alias/" + item.collective.key),
        )
        for item in grid_declarations(rows, validated)
    )
    parents = []

    def record(progress):
        value = progress.ready_collective_tuning
        if value is not None and value.step == "catalog":
            parents.extend(value.payload["grid"]["parents"])
        return coordinate(progress)

    with session(tmp_path, prime=False, cache_only=True) as engine:
        result = engine.prepare(renamed, coordinator=record)
    assert result.cache_hits == len(rows)
    assert result.benchmarked_candidates == 0
    assert validated == []
    assert parents and all(name.startswith("alias/") for name in parents)
    assert {
        name.removeprefix("alias/"): value.config
        for name, value in result.selections.items()
    } == {name: value.config for name, value in original.selections.items()}


@pytest.mark.parametrize(
    "seed_cohort,cache_only", ((True, True), (True, False), (False, False))
)
def test_grid_retained_anchor_preserves_same_cohort_and_bound_payload(
    tmp_path, host_races, seed_cohort, cache_only
):
    rows = (16, 17, 19, 21, 22, 24)
    if seed_cohort:
        with session(tmp_path, prime=False) as engine:
            engine.prepare(grid_declarations(rows, []), coordinator=coordinate)
    validated = []
    items = grid_declarations(rows, validated)
    with session(tmp_path, prime=False, cache_only=cache_only) as engine:
        engine.prepare((items[0],), coordinator=coordinate)
        retained = items[0].plan.prepared
        validated.clear()
        result = engine.prepare(items, coordinator=coordinate)
        assert items[0].plan.prepared is retained
        assert result.benchmarked_candidates == (0 if seed_cohort else 10)
        assert result.cache_hits == (5 if seed_cohort else 0)
        assert result.selections["rows-17"].config == LaunchConfig(2, 2)
        assert result.selections["rows-21"].config == LaunchConfig(4, 2)
        assert all(row != 16 for row, _, _ in validated)


@pytest.mark.parametrize("retained_row", (17, 19))
def test_grid_does_not_promote_retained_partial_calibration_to_full_anchor(
    tmp_path, host_races, retained_row
):
    items = grid_declarations((16, 17, 19, 24), [])
    with session(tmp_path, prime=False) as engine:
        engine.prepare(items, coordinator=coordinate)
        retained = next(item for item in items if item.plan.query.rows == retained_row)
        payload = retained.plan.prepared
        other = grid_declarations((24,), [])[0]
        with pytest.raises(
            ValueError, match="cannot supply the declared full calibration"
        ):
            engine.prepare((retained, other), coordinator=coordinate)
        assert retained.plan.prepared is payload


def test_grid_preserves_ready_custom_benchmark_binding_without_calibration_record(
    tmp_path, host_races
):
    validated = []
    items = grid_declarations((16, 17, 19, 24), validated)
    custom = replace(
        items[0], benchmark_call=lambda state: items[0].benchmark_call(state)
    )
    with session(tmp_path, prime=False) as engine:
        engine.prepare((custom,), coordinator=coordinate)
        payload = items[0].plan.prepared
        assert payload.collective_tuning is None
        validated.clear()
        result = engine.prepare(items, coordinator=coordinate)
        assert items[0].plan.prepared is payload
        assert result.benchmarked_candidates == 12
        assert all(row != 16 for row, _, _ in validated)


@pytest.mark.parametrize("retained_row", (17, 19))
def test_grid_cancellation_preserves_retained_partial_binding(
    tmp_path, host_races, retained_row
):
    rows = (16, 17, 19, 24)
    with session(tmp_path, prime=False) as engine:
        items = grid_declarations(rows, [])
        engine.prepare(items, coordinator=coordinate)
        retained = next(item for item in items if item.plan.query.rows == retained_row)
        payload = retained.plan.prepared
        records = dict(engine._selection_cache().records)
        fresh = grid_declarations(rows, [])
        mixed = tuple(
            retained if item.plan.query.rows == retained_row else item for item in fresh
        )

        def cancel(progress):
            value = progress.ready_collective_tuning
            if (
                value is not None
                and value.key == "group/16"
                and value.step == "catalog"
            ):
                engine.cancel_tuning()
            return coordinate(progress)

        result = engine.prepare(mixed, coordinator=cancel)
        assert retained.plan.prepared is payload
        assert result.selections[retained.name] == payload.selection
        assert result.selections["rows-16"].source == "cached"
        assert all(
            selection.source == "default"
            for name, selection in result.selections.items()
            if name not in (retained.name, "rows-16")
        )
        assert result.benchmarked_candidates == 0
        assert engine._selection_cache().records == records


@pytest.mark.parametrize("frozen", (False, True))
@pytest.mark.parametrize("retained_row", (16, 19))
def test_grid_retained_binding_rejects_different_rank_membership(
    tmp_path, host_races, frozen, retained_row
):
    with session(tmp_path, prime=False) as engine:
        items = grid_declarations((16, 17, 19, 24), [])
        engine.prepare(items, coordinator=coordinate)
        retained = next(item for item in items if item.plan.query.rows == retained_row)
        payload = retained.plan.prepared
        if frozen:
            engine.freeze()
        result = engine.prepare(items, coordinator=coordinate)
        assert result.benchmarked_candidates == 0
        changed = replace(
            retained, collective=replace(retained.collective, ranks=(0, 2))
        )
        with pytest.raises(ValueError, match="prepared ranks are \\(0, 1\\)"):
            engine.prepare((changed,), coordinator=coordinate)
        assert retained.plan.prepared is payload


def test_grid_same_plan_rejects_different_rank_groups_before_preparation(
    tmp_path, host_races
):
    validated = []
    (item,) = grid_declarations((16,), validated)
    alias = replace(
        item,
        name="different-ranks",
        collective=replace(item.collective, key="different-ranks", ranks=(0, 2)),
    )
    with session(tmp_path, prime=False) as engine:
        with pytest.raises(
            ValueError, match="same Plan must agree on collective ranks"
        ):
            engine.prepare((item, alias), coordinator=coordinate)
        assert item.plan.prepared is None
        assert validated == []
        assert not engine._selection_cache().records
        result = engine.prepare((item,), coordinator=coordinate)
        assert result.selections[item.name].source == "tuned"


@pytest.mark.parametrize("grid", (False, True))
@pytest.mark.parametrize("failure", ("validation", "cancel", "success"))
def test_each_producer_agrees_on_validation_and_cancellation_before_invoking_again(
    tmp_path, host_races, monkeypatch, grid, failure
):
    from contextlib import ExitStack

    from triton import knobs

    from b12x._lib import compiler

    original_compile = compiler._call_cute_compile
    original_listener = knobs.compilation.listener
    monkeypatch.setattr(compiler, "_call_cute_compile", original_compile)
    monkeypatch.setattr(knobs.compilation, "listener", original_listener)
    entered = [[], []]
    with ExitStack() as stack:
        jobs = []
        for rank in range(2):
            engine = stack.enter_context(
                session(tmp_path / str(rank), prime=False, compile_workers=0)
            )
            items = grid_declarations((16, 17, 19, 24), [])
            if not grid:
                items = tuple(
                    replace(
                        item,
                        plan=replace(
                            item.plan,
                            contract=replace(
                                item.plan.contract, collective_bucket=None
                            ),
                        ),
                    )
                    for item in items
                )
            target = items[1]

            def benchmark(
                state, factory=target.benchmark_call, rank=rank, engine=engine
            ):
                call = factory(state)

                def run():
                    entered[rank].append(state.produced)
                    return call.run()

                def validate():
                    call.validate()
                    if rank == 0 and state.produced == 1:
                        if failure == "validation":
                            raise ValueError("rank zero rejected the first producer")
                        if failure == "cancel":
                            engine.cancel_tuning()

                return replace(call, run=run, validate=validate)

            items = (
                items[0],
                replace(
                    target,
                    benchmark_call=benchmark,
                    plan=replace(target.plan, _benchmark_call=benchmark),
                ),
                *items[2:],
            )
            jobs.append(engine.begin(items))
        allowed = {}
        states = [None, None]
        errors = [None, None]
        try:
            for _ in range(10000):
                for rank, job in enumerate(jobs):
                    try:
                        if states[rank] is None or not states[rank].done:
                            states[rank] = job.advance(**allowed)
                    except Exception as error:
                        errors[rank] = error
                    assert compiler._call_cute_compile is original_compile
                    assert knobs.compilation.listener is original_listener
                if any(errors) or all(state.done for state in states):
                    break
                allowed = {}
                if all(state.ready_collective_tuning is not None for state in states):
                    allowed["collective_tuning"] = tuple(
                        state.ready_collective_tuning for state in states
                    )
                elif all(state.ready_collectives for state in states):
                    pending = tuple(state.ready_collectives[0] for state in states)
                    assert pending[0] == pending[1]
                    allowed["collective_key"] = pending[0].key
            else:
                pytest.fail("collective preparation did not terminate")
            expected = [1, 2] * (1 if grid else 6) if failure == "success" else [1]
            assert entered == [expected, expected]
            if failure == "validation":
                assert isinstance(errors[0], ValueError)
                assert str(errors[0]) == "rank zero rejected the first producer"
                assert isinstance(errors[1], RuntimeError)
                assert (
                    "rank 0: ValueError: rank zero rejected the first producer"
                    in str(errors[1])
                )
            else:
                assert errors == [None, None]
                source = "tuned" if failure == "success" else "default"
                assert all(
                    job.result().selections["rows-17"].source == source for job in jobs
                )
        finally:
            for job in jobs:
                job.close()
            assert compiler._call_cute_compile is original_compile
            assert knobs.compilation.listener is original_listener
