"""CPU coverage, deadline and report contracts for collective ladder searches."""

import json

import pytest

from b12x.tools._collective_ladder_sampling import (
    MAX_ROUNDS,
    SAMPLES_PER_ROUND,
    measurement_rounds,
)
from b12x.tools._collective_ladder_search import (
    RawReport,
    Search,
    Trial,
    ladder_sizes,
    read_report,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def config(implementation="joint", threads=32, blocks=36, submode="push", active=1):
    return dict(
        implementation=implementation,
        threads=threads,
        blocks=blocks,
        submode=submode,
        active_blocks=active,
    )


def records(trial, latency=lambda size, c: 1.0):
    return [
        dict(
            bytes=trial.size_bytes,
            config=c,
            samples_us=[latency(trial.size_bytes, c)]
            * (trial.rounds * SAMPLES_PER_ROUND),
            correctness="passed",
            phase=trial.phase,
        )
        for c in trial.configs
    ]


def test_default_ladder_has_every_logarithmic_aligned_point():
    sizes = ladder_sizes()
    assert len(sizes) == len(set(sizes)) == 209
    assert sizes[0] == 1024 and sizes[-1] == 8 << 20
    assert all(size % 16 == 0 for size in sizes)
    assert all(sizes[index * 16] == 1024 * 2**index for index in range(14))
    assert all(a < b for a, b in zip(sizes, sizes[1:]))


def test_every_legal_configuration_is_measured_and_winners_are_per_payload():
    clock = Clock()
    a, b = config("a"), config("b")
    search = Search({1024: [a, b], 1040: [a, b]}, clock=clock, deadline=300)
    observed = []
    while (trial := search.propose()) is not None:
        observed.extend(
            (trial.size_bytes, c["implementation"], trial.phase) for c in trial.configs
        )
        search.observe(
            trial,
            records(
                trial,
                lambda size, c: (
                    {"a": 1.0, "b": 1.2} if size == 1024 else {"a": 10.0, "b": 1.0}
                )[c["implementation"]],
            ),
        )
    summary = search.summary()
    assert summary["status"] == "complete"
    assert [w["config"]["implementation"] for w in summary["winners"]] == ["a", "b"]
    assert {
        (size, name)
        for size, name, phase in observed
        if phase not in ("confirmation", "final")
    } == {(1024, "a"), (1024, "b"), (1040, "a"), (1040, "b")}
    assert summary["confirmed_sizes"] == 2


def test_group_coverage_precedes_runtime_refinement_at_fine_points():
    configs = [
        config(threads=t, submode=m, active=a)
        for t in (32, 64)
        for m in ("push", "pull")
        for a in (1, 2, 4)
    ]
    clock = Clock()
    search = Search(
        {1024: configs, 1040: configs, 1088: configs}, deadline=300, clock=clock
    )
    first = search.propose()
    assert first.size_bytes == 1024 and len(first.configs) == 8
    assert len({(c["threads"], c["submode"]) for c in first.configs}) == 4
    search.observe(first, records(first))
    clock.now = 46
    second = search.propose()
    assert second.size_bytes == 1040 and len(second.configs) == 8
    search.observe(second, records(second))
    third = search.propose()
    assert third.size_bytes == 1088 and len(third.configs) == 8
    search.observe(third, records(third))
    assert all(
        row["measured_groups"] == 4 for row in search.summary()["search_coverage"]
    )
    assert search.summary()["unexplored_configurations"] == 12


def test_deadline_reserves_confirmation_and_never_invents_missing_points():
    clock = Clock()
    search = Search({1024: [config()], 1040: [config()]}, deadline=300, clock=clock)
    trial = search.propose()
    search.observe(trial, records(trial))
    clock.now = 231
    confirmation = search.propose()
    assert confirmation.phase == "confirmation" and confirmation.rounds >= 2
    search.observe(confirmation, records(confirmation))
    clock.now = 291
    assert search.propose() is None
    summary = search.summary()
    assert summary["status"] == "partial"
    assert summary["measured_sizes"] == 1
    assert [entry["size_bytes"] for entry in summary["missing"]] == [1040]
    assert [row["size_bytes"] for row in summary["winners"]] == [1024]


def test_batch_cost_admission_and_compilation_deadline_adjustment():
    clock = Clock()
    search = Search(
        {1024: [config(threads=t) for t in (32, 64, 128)]}, deadline=300, clock=clock
    )
    trial = search.propose()
    clock.now += 110
    search.deadline += 100
    search.observe(trial, records(trial))
    assert search._seconds_per_config_round == pytest.approx(
        10 / (len(trial.configs) * trial.rounds) * 1.1
    )
    clock.now = search.deadline - 10.01
    assert search.propose() is None
    assert search.summary()["reason"] == "confirmation_budget_exhausted"


def test_exhaustive_mode_attempts_all_finite_profiles():
    configs = [config(active=n) for n in (1, 2, 4)]
    search = Search(
        {1024: configs, 1040: configs},
        deadline=300,
        clock=Clock(),
        exhaustive=True,
        batch_size=2,
        rounds=4,
    )
    attempts = []
    while (trial := search.propose()) is not None:
        assert trial.rounds == measurement_rounds(
            trial.size_bytes, trial.phase, estimate_us=1.0, minimum_rounds=4
        )
        attempts.extend(
            (trial.size_bytes, c["active_blocks"])
            for c in trial.configs
            if trial.phase not in ("confirmation", "final")
        )
        search.observe(trial, records(trial))
    assert len(attempts) == len(set(attempts)) == 6
    assert search.summary()["status"] == "complete"


def test_invalid_observation_is_atomic_and_failed_config_is_not_recommended():
    search = Search({1024: [config("a"), config("b")]}, deadline=300, clock=Clock())
    trial = search.propose()
    with pytest.raises(ValueError, match="each proposed"):
        search.observe(trial, records(trial)[:1])
    assert search.pending == trial and search.records == []
    result = records(trial)
    result[0]["correctness"] = "failed"
    search.observe(trial, result)
    assert search.summary()["failed_configurations"] == 1
    assert search.summary()["winners"][0]["config"]["implementation"] == "b"
    assert search.summary()["status"] == "partial"


def test_resume_uses_actual_samples_and_rejects_outside_catalog():
    trial = Trial(1024, (config(),), 3)
    previous = records(trial)
    search = Search({1024: [config()]}, deadline=300, clock=Clock(), records=previous)
    assert search.propose().phase == "confirmation"
    with pytest.raises(ValueError, match="outside"):
        Search({1040: [config()]}, deadline=300, clock=Clock(), records=previous)


def test_raw_report_append_resume_preserves_samples_and_extra_fields(tmp_path):
    path = tmp_path / "timings.jsonl"
    metadata = dict(hardware=["gpu0", "gpu1"], catalogs=(config(),), cache_policy="hot")
    report = RawReport(path, metadata)
    row = dict(records(Trial(1024, (config(),)))[0], replay_sources=["reference"])
    report.append(row)
    report.append_summary(
        dict(winners=[dict(size_bytes=1024, config=config(), median_us=1.0)])
    )
    report.close()
    with pytest.raises(FileExistsError):
        RawReport(path, metadata)
    report = RawReport(path, json.loads(json.dumps(metadata)), resume=True)
    assert report.records[0]["samples_us"] == [1.0] * (3 * SAMPLES_PER_ROUND)
    assert report.records[0]["replay_sources"] == ["reference"]
    report.close()
    loaded, measurements, summaries = read_report(path)
    assert loaded["hardware"] == ["gpu0", "gpu1"]
    assert len(measurements) == 1 and summaries[-1]["winners"][0]["size_bytes"] == 1024
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incompatible"):
        RawReport(path, dict(metadata, hardware=["gpu1", "gpu0"]), resume=True)
    assert path.read_bytes() == before


def test_report_rejects_invalid_timings_and_truncated_records(tmp_path):
    path = tmp_path / "timings.jsonl"
    report = RawReport(path, {})
    with pytest.raises(ValueError):
        report.append(dict(bytes=1024, config=config(), samples_us=[float("nan")]))
    report.close()
    with path.open("a") as stream:
        stream.write("{")
    with pytest.raises(ValueError, match=":2:"):
        read_report(path)


def test_fine_coverage_seeds_each_group_from_its_own_measured_profile():
    configs = [config(name, active=active) for name in ("a", "b") for active in (1, 4)]
    trial = Trial(1024, tuple(configs))
    previous = records(
        trial,
        lambda size, c: (
            1.0
            if (c["implementation"], c["active_blocks"]) in [("a", 1), ("b", 4)]
            else 10.0
        ),
    )
    search = Search(
        {1024: configs, 1040: configs}, deadline=300, clock=Clock(), records=previous
    )
    proposed = search.propose()
    assert proposed.size_bytes == 1040
    assert {(c["implementation"], c["active_blocks"]) for c in proposed.configs} == {
        ("a", 1),
        ("a", 4),
        ("b", 4),
    }


def test_closed_confirmation_uses_fresh_batch_without_promoting_unconfirmed_candidates():
    configs = [config(name) for name in ("a", "b", "c", "d")]
    search = Search({1024: configs}, deadline=300, clock=Clock())
    trial = search.propose()
    search.observe(
        trial,
        records(
            trial,
            lambda size, c: {"a": 1.0, "b": 1.02, "c": 1.04, "d": 4.0}[
                c["implementation"]
            ],
        ),
    )
    confirmation = search.propose()
    assert [c["implementation"] for c in confirmation.configs] == ["a", "b", "c"]
    search.observe(
        confirmation,
        records(
            confirmation,
            lambda size, c: {"a": 10.0, "b": 9.0, "c": 12.0}[c["implementation"]],
        ),
    )
    assert search.summary()["confirmed_sizes"] == 0
    final = search.propose()
    assert final.phase == "final" and final.rounds == measurement_rounds(
        1024, "final", estimate_us=1.0
    )
    assert [c["implementation"] for c in final.configs] == ["b", "a"]
    search.observe(
        final, records(final, lambda size, c: {"a": 1.0, "b": 2.0}[c["implementation"]])
    )
    assert search.propose() is None
    (winner,) = search.summary()["winners"]
    assert winner["config"]["implementation"] == "a" and winner["median_us"] == 1.0
    assert winner["batch"] == final.batch_id and winner["rounds"] == final.rounds
    assert search.summary()["status"] == "complete"


def test_anchor_runtime_probes_cover_extremes_before_nearest_neighbors():
    configs = [config(active=active) for active in (1, 2, 4, 8, 16, 32, 36)]
    search = Search(
        {1024: configs, 1040: configs}, deadline=300, clock=Clock(), batch_size=3
    )
    first = search.propose()
    assert [c["active_blocks"] for c in first.configs] == [1, 36]
    search.observe(first, records(first))
    second = search.propose()
    assert second.phase == "anchor"
    assert [c["active_blocks"] for c in second.configs] == [8, 32]


def test_partial_confirmation_resume_preserves_raw_samples_without_publishing_winner():
    a, b = config("a"), config("b")
    screening = records(Trial(1024, (a, b)))
    partial = dict(
        bytes=1024,
        config=a,
        samples_us=[1.01] * (7 * SAMPLES_PER_ROUND),
        correctness="passed",
        phase="confirmation",
        batch=17,
        batch_size=2,
        rounds=7,
    )
    search = Search(
        {1024: [a, b]}, deadline=300, clock=Clock(), records=[*screening, partial]
    )
    assert len(search.records) == 3
    assert search.summary()["confirmed_sizes"] == 0
    trial = search.propose()
    assert (
        trial.phase == "confirmation"
        and len(trial.configs) == 2
        and trial.batch_id > 17
    )
    search.observe(trial, records(trial))
    resumed = Search(
        {1024: [a, b]}, deadline=300, clock=Clock(), records=search.records
    )
    assert resumed.summary()["status"] == "partial"
    finish_final(resumed)
    assert resumed.summary()["status"] == "complete"


def test_explicit_resume_repairs_only_incomplete_final_json(tmp_path):
    path = tmp_path / "timings.jsonl"
    report = RawReport(path, {"dtype": "bfloat16"})
    report.append(records(Trial(1024, (config(),)))[0])
    report.close()
    good = path.read_bytes()
    path.write_bytes(good + b'{"kind":"measurement","bytes":')
    with pytest.raises(ValueError):
        read_report(path)
    report = RawReport(path, {"dtype": "bfloat16"}, resume=True)
    assert len(report.records) == 1 and path.read_bytes() == good
    report.close()
    path.write_bytes(good + b"{bad}\n" + b"{")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        RawReport(path, {"dtype": "bfloat16"}, resume=True)
    assert path.read_bytes() == before


@pytest.mark.parametrize("exhaustive", [False, True])
def test_complete_requires_full_catalog_only_for_exhaustive_search(exhaustive):
    clock = Clock()
    configs = [config(active=active) for active in (1, 2, 4)]
    search = Search(
        {1024: configs}, deadline=300, clock=clock, exhaustive=exhaustive, batch_size=1
    )
    first = search.propose()
    search.observe(first, records(first))
    second = search.propose()
    search.observe(second, records(second))
    clock.now = 231
    confirmation = search.propose()
    assert confirmation.phase == "confirmation"
    search.observe(confirmation, records(confirmation))
    finish_final(search)
    assert search.propose() is None
    result = search.summary()
    assert result["unexplored_configurations"] == 1
    assert result["failed_configurations"] == 0
    assert result["status"] == ("partial" if exhaustive else "complete")


def test_measurement_without_explicit_correctness_is_not_a_valid_winner():
    search = Search({1024: [config()]}, deadline=300, clock=Clock())
    trial = search.propose()
    measurements = records(trial)
    del measurements[0]["correctness"]
    search.observe(trial, measurements)
    summary = search.summary()
    assert summary["winners"] == []
    assert summary["measured_configurations"] == 0
    assert summary["failed_configurations"] == 1
    assert summary["status"] == "partial"


def save_measurements(path, measurements):
    report = RawReport(path, {"operation": "all_reduce", "dtype": "bfloat16"})
    for measurement in measurements:
        report.append(measurement)
    report.close()
    return read_report(path)[1]


def finish_final(search, latency=lambda size, c: 1.0):
    trial = search.propose()
    assert trial.phase == "final" and trial.rounds >= 4
    search.observe(trial, records(trial, latency))
    return trial


def completed_adaptive_search(catalogs):
    clock = Clock()
    search = Search(catalogs, deadline=300, clock=clock, batch_size=1)
    while any(search.coarse_remaining.values()):
        trial = search.propose()
        assert trial.phase != "confirmation"
        search.observe(trial, records(trial))
    clock.now = 231
    confirmation = search.propose()
    search.observe(confirmation, records(confirmation))
    finish_final(search)
    assert search.summary()["status"] == "complete"
    return search


def test_completed_adaptive_report_resume_proposes_no_measurements(tmp_path):
    catalogs = {1024: [config(active=n) for n in (1, 2, 4)]}
    complete = completed_adaptive_search(catalogs)
    stored = save_measurements(tmp_path / "complete.jsonl", complete.records)
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=stored)
    assert resumed.summary()["unexplored_configurations"] == 1
    assert resumed.propose() is None
    assert resumed.summary()["reason"] == "cached_complete"
    assert resumed.summary()["status"] == "complete"
    assert len(resumed.records) == len(stored)


def test_partial_report_resume_only_measures_unfinished_payloads(tmp_path):
    configs = [config(active=n) for n in (1, 2, 4)]
    catalogs = {1024: configs, 1040: configs}
    clock = Clock()
    partial = Search(catalogs, deadline=300, clock=clock, batch_size=1)
    first = partial.propose()
    partial.observe(first, records(first))
    clock.now = 46
    while any(partial.coarse_remaining.values()):
        trial = partial.propose()
        partial.observe(trial, records(trial))
    clock.now = 231
    confirmation = partial.propose()
    assert confirmation.size_bytes == 1024
    partial.observe(confirmation, records(confirmation))
    finish_final(partial)
    assert partial.summary()["confirmed_sizes"] == 1
    stored = save_measurements(tmp_path / "partial.jsonl", partial.records)
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=stored)
    attempts = []
    while (trial := resumed.propose()) is not None:
        attempts.append(trial.size_bytes)
        resumed.observe(trial, records(trial))
    assert attempts and set(attempts) == {1040}
    assert resumed.summary()["status"] == "complete"
    assert resumed.summary()["unexplored_configurations"] == 1


def test_exhaustive_resume_invalidates_final_and_reconfirms_new_screening(tmp_path):
    catalogs = {1024: [config(active=n) for n in (1, 2, 4)]}
    complete = completed_adaptive_search(catalogs)
    stored = save_measurements(tmp_path / "adaptive.jsonl", complete.records)
    resumed = Search(
        catalogs, deadline=300, clock=Clock(), records=stored, exhaustive=True
    )
    trial = resumed.propose()
    assert trial.phase != "confirmation" and trial.configs[0]["active_blocks"] == 2
    resumed.observe(trial, records(trial, lambda size, c: 0.5))
    assert resumed.summary()["confirmed_sizes"] == 0
    assert resumed.summary()["status"] == "partial"
    after_screen = save_measurements(tmp_path / "screened.jsonl", resumed.records)
    restarted = Search(catalogs, deadline=300, clock=Clock(), records=after_screen)
    assert restarted.summary()["confirmed_sizes"] == 0
    confirmation = restarted.propose()
    assert confirmation.phase == "confirmation"
    assert confirmation.configs[0]["active_blocks"] == 2
    restarted.observe(confirmation, records(confirmation, lambda size, c: 0.5))
    finish_final(restarted, lambda size, c: 0.5)
    assert restarted.propose() is None
    assert restarted.summary()["status"] == "complete"
    (winner,) = restarted.summary()["winners"]
    assert winner["config"]["active_blocks"] == 2 and winner["median_us"] == 0.5


def test_fine_point_requires_upper_and_payload_specific_numeric_profile():
    base = [config(active=n) for n in (1, 2, 4, 8, 16, 32, 64)]
    fine = [*base, config(active=27)]
    previous = [
        *records(
            Trial(1024, tuple(base)),
            lambda size, c: 1.0 if c["active_blocks"] == 8 else 2.0,
        ),
        *records(
            Trial(2048, tuple(base)),
            lambda size, c: 1.0 if c["active_blocks"] == 8 else 2.0,
        ),
    ]
    search = Search(
        {1024: base, 1536: fine, 2048: base},
        deadline=300,
        clock=Clock(),
        records=previous,
    )
    trial = search.propose()
    assert trial.size_bytes == 1536
    assert {c["active_blocks"] for c in trial.configs} == {8, 27, 64}
    search.observe(
        trial, records(trial, lambda size, c: 1.0 if c["active_blocks"] == 64 else 2.0)
    )
    row = next(
        row for row in search.summary()["search_coverage"] if row["size_bytes"] == 1536
    )
    assert row["missing_profiles"] == 0 and row["required_profiles"] == 3


def test_refinement_balances_payloads_instead_of_exhausting_small_sizes():
    configs = [config(active=n) for n in (1, 2, 4, 8, 16, 32, 64)]
    catalogs = {size: configs for size in (1024, 1536, 2048)}
    initial = [
        row
        for size in catalogs
        for row in records(Trial(size, (configs[0], configs[-1])))
    ]
    clock = Clock()
    search = Search(catalogs, deadline=300, clock=clock, records=initial, batch_size=1)
    clock.now = 46
    first = search.propose()
    search.observe(first, records(first))
    second = search.propose()
    assert first.size_bytes != second.size_bytes


def test_resume_does_not_recompute_completed_warmstart_from_later_anchor(tmp_path):
    configs = [config(active=n) for n in (1, 2, 4)]
    catalogs = {size: configs for size in (1024, 1536, 2048)}
    observed = []

    def append(trial, latency):
        for row in records(trial, latency):
            observed.append(
                dict(
                    row,
                    batch=trial.batch_id,
                    batch_size=len(trial.configs),
                    rounds=trial.rounds,
                )
            )

    for batch, size in enumerate(catalogs, 1):
        append(
            Trial(size, (configs[0], configs[-1]), phase="coverage", batch_id=batch),
            lambda size, c: 1.0 if c["active_blocks"] == 1 else 2.0,
        )
    for batch, size in enumerate((1024, 2048), 4):
        append(
            Trial(size, (configs[1],), phase="refinement", batch_id=batch),
            lambda size, c: 0.5,
        )
    for batch, size in enumerate(catalogs, 6):
        winner = configs[0] if size == 1536 else configs[1]
        append(
            Trial(size, (winner,), rounds=7, phase="confirmation", batch_id=batch),
            lambda size, c: 1.0 if size == 1536 else 0.5,
        )
    for batch, size in enumerate(catalogs, 9):
        winner = configs[0] if size == 1536 else configs[1]
        append(
            Trial(size, (winner,), rounds=15, phase="final", batch_id=batch),
            lambda size, c: 1.0 if size == 1536 else 0.5,
        )
    stored = save_measurements(tmp_path / "changed-anchor.jsonl", observed)
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=stored)
    assert resumed.summary()["status"] == "complete"
    assert resumed.propose() is None
    assert resumed.summary()["reason"] == "cached_complete"


@pytest.mark.parametrize("row", [[], 7, "measurement", None])
@pytest.mark.parametrize("line_number", [1, 2])
def test_report_rejects_non_object_rows_with_path_and_line(tmp_path, row, line_number):
    path = tmp_path / "invalid-row.jsonl"
    lines = (
        []
        if line_number == 1
        else [json.dumps(dict(kind="metadata", schema_version=1, metadata={}))]
    )
    path.write_text("\n".join([*lines, json.dumps(row)]) + "\n")
    before = path.read_bytes()
    with pytest.raises(
        ValueError, match=f"{path}:{line_number}: report record must be a JSON object"
    ):
        read_report(path)
    assert path.read_bytes() == before


def test_rejected_short_confirmation_preserves_raw_partial_batch_for_resume(tmp_path):
    catalogs = {1024: [config("a"), config("b")]}
    search = Search(catalogs, deadline=300, clock=Clock())
    screening = search.propose()
    search.observe(screening, records(screening))
    confirmation = search.propose()
    returned = records(confirmation)[:1]
    writer = RawReport(tmp_path / "short-confirmation.jsonl", {})
    for row in search.records:
        writer.append(row)
    for row in returned:
        writer.append(
            dict(
                row,
                batch=confirmation.batch_id,
                batch_size=len(confirmation.configs),
                rounds=confirmation.rounds,
            )
        )
    writer.close()
    with pytest.raises(ValueError, match="each proposed"):
        search.observe(confirmation, returned)
    loaded = read_report(tmp_path / "short-confirmation.jsonl")[1]
    assert len(loaded) == len(search.records) + 1
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=loaded)
    assert resumed.summary()["confirmed_sizes"] == 0
    assert resumed.summary()["status"] == "partial"
    retry = resumed.propose()
    assert retry.phase == "confirmation" and len(retry.configs) == 2
    resumed.observe(retry, records(retry))
    finish_final(resumed)
    assert resumed.summary()["status"] == "complete"


def test_partial_final_resume_retries_whole_pair_without_repeating_initial_cohort(
    tmp_path,
):
    configs = [config("a"), config("b"), config("c")]
    catalogs = {1024: configs}
    search = Search(catalogs, deadline=300, clock=Clock())
    screening = search.propose()
    search.observe(screening, records(screening))
    first = search.propose()
    search.observe(first, records(first))
    final = search.propose()
    assert final.phase == "final" and len(final.configs) == 2
    partial = dict(
        records(final)[0], batch=final.batch_id, batch_size=2, rounds=final.rounds
    )
    stored = save_measurements(
        tmp_path / "partial-final.jsonl", [*search.records, partial]
    )
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=stored)
    assert resumed.summary()["initial_confirmed_sizes"] == 1
    assert resumed.summary()["confirmed_sizes"] == 0
    retry = resumed.propose()
    assert (
        retry.phase == "final"
        and retry.rounds == final.rounds
        and retry.configs == final.configs
    )
    resumed.observe(retry, records(retry))
    assert resumed.summary()["status"] == "complete"
    complete = Search(catalogs, deadline=300, clock=Clock(), records=resumed.records)
    assert complete.propose() is None and complete.summary()["status"] == "complete"


def test_report_after_initial_confirmation_seeds_final_pair(
    tmp_path,
):
    configs = [config("a"), config("b")]
    catalogs = {1024: configs}
    search = Search(catalogs, deadline=300, clock=Clock())
    screening = search.propose()
    search.observe(screening, records(screening))
    confirmation = search.propose()
    search.observe(confirmation, records(confirmation))
    cohort_records = save_measurements(
        tmp_path / "confirmation-only.jsonl", search.records
    )
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=cohort_records)
    assert (
        resumed.summary()["status"] == "partial"
        and resumed.summary()["confirmed_sizes"] == 0
    )
    trial = resumed.propose()
    assert (
        trial.phase == "final"
        and trial.rounds == measurement_rounds(1024, "final", estimate_us=1.0)
        and len(trial.configs) == 2
    )
    resumed.observe(trial, records(trial))
    assert resumed.summary()["status"] == "complete"


def test_final_batch_is_not_admitted_if_whole_pair_exceeds_remaining_budget():
    clock = Clock()
    search = Search({1024: [config("a"), config("b")]}, deadline=300, clock=clock)
    screening = search.propose()
    search.observe(screening, records(screening))
    confirmation = search.propose()
    search.observe(confirmation, records(confirmation))
    clock.now = 289.99
    assert search.propose() is None
    assert search.summary()["reason"] == "final_budget_exhausted"
    assert search.summary()["confirmed_sizes"] == 0
    assert search.summary()["status"] == "partial"


def test_confirmation_and_final_records_cannot_close_a_mixed_phase_cohort():
    configs = [config("a"), config("b")]
    screening = records(Trial(1024, tuple(configs)))
    mixed = [
        dict(
            bytes=1024,
            config=c,
            samples_us=[1.0] * (15 * SAMPLES_PER_ROUND),
            correctness="passed",
            phase=phase,
            batch=5,
            batch_size=2,
            rounds=15,
        )
        for c, phase in zip(configs, ("confirmation", "final"), strict=True)
    ]
    with pytest.raises(ValueError, match="inconsistent confirmation batch"):
        Search(
            {1024: configs}, deadline=300, clock=Clock(), records=[*screening, *mixed]
        )


def test_dynamic_sampling_scales_small_and_large_payload_timing_counts():
    assert SAMPLES_PER_ROUND == 6 and MAX_ROUNDS == 512
    assert measurement_rounds(4096, "coverage") == 8
    assert measurement_rounds(1 << 20, "coverage") == 1
    small = measurement_rounds(4096, "final", estimate_us=6.0)
    large = measurement_rounds(1 << 20, "final", estimate_us=100.0)
    assert (small, small * SAMPLES_PER_ROUND) == (112, 672)
    assert (large, large * SAMPLES_PER_ROUND) == (7, 42)
    assert measurement_rounds(4096, "replay", estimate_us=6.0) == small
    assert measurement_rounds(4096, "confirmation", estimate_us=6.0) == 28


def test_dynamic_sampling_honors_phase_floors_minimum_and_cap():
    assert measurement_rounds(1024, "coverage", minimum_rounds=16) == 16
    assert measurement_rounds(4096, "confirmation", estimate_us=1000.0) == 2
    assert measurement_rounds(4096, "final", estimate_us=1000.0) == 4
    assert measurement_rounds(4096, "final", estimate_us=1e-320) == MAX_ROUNDS
    assert (
        measurement_rounds(4096, "final", estimate_us=100.0, minimum_rounds=100) == 100
    )


@pytest.mark.parametrize("estimate", [0, -1, float("nan"), float("inf"), True])
def test_dynamic_sampling_rejects_invalid_latency_estimates(estimate):
    with pytest.raises(ValueError, match="latency estimate"):
        measurement_rounds(4096, "final", estimate_us=estimate)


@pytest.mark.parametrize("minimum", [0, -1, 513, True])
def test_dynamic_sampling_rejects_invalid_minimum_rounds(minimum):
    with pytest.raises(ValueError, match="minimum rounds"):
        measurement_rounds(4096, "final", minimum_rounds=minimum)


def test_search_uses_one_payload_estimate_for_balanced_dynamic_cohorts():
    configs = [config("a"), config("b")]
    search = Search(
        {4096: configs, 1 << 20: configs}, deadline=300, clock=Clock(), rounds=1
    )
    latencies = {4096: 6.0, 1 << 20: 100.0}
    while (trial := search.propose()) is not None:
        assert trial.rounds == measurement_rounds(
            trial.size_bytes,
            trial.phase,
            estimate_us=latencies[trial.size_bytes],
            minimum_rounds=1,
        )
        measurements = records(trial, lambda size, c: latencies[size])
        assert len({len(row["samples_us"]) for row in measurements}) == 1
        search.observe(trial, measurements)
    assert search.summary()["status"] == "complete"
    assert {
        row["size_bytes"]: row["rounds"] for row in search.summary()["winners"]
    } == {4096: 112, 1 << 20: 7}


def test_restored_cohort_requires_raw_sample_count_declared_by_requested_rounds():
    configs = [config("a"), config("b")]
    screening = records(Trial(4096, tuple(configs)))
    incomplete = [
        dict(
            bytes=4096,
            config=c,
            samples_us=[6.0] * 3,
            correctness="passed",
            phase="confirmation",
            batch=10,
            batch_size=2,
            rounds=1,
        )
        for c in configs
    ]
    restored = Search(
        {4096: configs}, deadline=300, clock=Clock(), records=[*screening, *incomplete]
    )
    assert restored.summary()["initial_confirmed_sizes"] == 0
    assert restored.summary()["confirmed_sizes"] == 0
    assert restored.summary()["status"] == "partial"


def test_bracketing_anchor_profiles_resume_only_missing_fine_coverage(tmp_path):
    configs = [config(active=n) for n in (1, 8, 16, 32, 64)]
    catalogs = {size: configs for size in (4096, 4864, 6144, 8192)}
    observed = []

    def append(size, chosen, phase, latency):
        trial = Trial(
            size, tuple(chosen), rounds=4, phase=phase, batch_id=len(observed) + 1
        )
        observed.extend(
            dict(row, batch=trial.batch_id, batch_size=len(chosen), rounds=trial.rounds)
            for row in records(trial, latency)
        )

    for size in (4096, 8192):
        append(
            size,
            [c for c in configs if c["active_blocks"] != 16],
            "anchor",
            lambda size, c: (
                1.0 if c["active_blocks"] == (8 if size == 4096 else 1) else 2.0
            ),
        )
    append(4864, [configs[1], configs[-1]], "coverage", lambda size, c: 1.0)
    append(6144, [configs[0], configs[1], configs[-1]], "coverage", lambda size, c: 1.0)
    # Later refinement cannot change the profiles exported by anchor calibration.
    for size in (4096, 8192):
        append(size, [configs[3]], "refinement", lambda size, c: 0.5)
    for phase in ("confirmation", "final"):
        for size in catalogs:
            append(
                size,
                [configs[1] if size not in (4096, 8192) else configs[3]],
                phase,
                lambda size, c: 0.5,
            )
    stored = save_measurements(tmp_path / "brackets.jsonl", observed)
    search = Search(catalogs, deadline=300, clock=Clock(), records=stored)
    assert search.summary()["status"] == "partial"
    trial = search.propose()
    assert trial.size_bytes == 4864 and trial.phase == "coverage"
    assert trial.configs == (configs[0],)
    search.observe(trial, records(trial))
    interrupted = Search(catalogs, deadline=300, clock=Clock(), records=search.records)
    assert interrupted.propose().phase == "confirmation"
    while (trial := search.propose()) is not None:
        assert trial.size_bytes == 4864
        assert trial.phase in ("confirmation", "final")
        search.observe(trial, records(trial))
    assert search.summary()["status"] == "complete"
    resumed = Search(catalogs, deadline=300, clock=Clock(), records=search.records)
    assert resumed.propose() is None
    assert resumed.summary()["reason"] == "cached_complete"


def test_fine_coverage_measures_both_anchor_winners():
    configs = [config(active=n) for n in (1, 8, 64)]
    observed = []
    for size in (4096, 8192):
        observed.extend(
            records(
                Trial(size, tuple(configs), phase="anchor"),
                lambda size, c: (
                    1.0 if c["active_blocks"] == (8 if size == 4096 else 1) else 2.0
                ),
            )
        )
    # A recorded coverage phase prevents resumed calibration from consuming time.
    observed.extend(records(Trial(4096, (configs[-1],), phase="coverage")))
    search = Search(
        {size: configs for size in (4096, 4864, 8192)},
        deadline=300,
        clock=Clock(),
        records=observed,
    )
    trial = search.propose()
    assert trial.size_bytes == 4864 and trial.phase == "coverage"
    assert {c["active_blocks"] for c in trial.configs} == {1, 8, 64}
