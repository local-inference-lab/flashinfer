"""Collective ladder arguments and replay of measured recommendations."""

from copy import deepcopy
from pathlib import Path

import pytest

from b12x.tools import collective_ladder as cli
from b12x.tools._collective_ladder_search import RawReport, Trial, read_report


CONFIG = dict(
    implementation="joint", threads=128, blocks=16, submode="bulk_push", active_blocks=4
)
OTHER = dict(CONFIG, active_blocks=8)


@pytest.fixture
def metadata():
    return dict(
        versions=dict(torch="test", cuda="13.0", flashinfer="test", python="3.12"),
        timing_protocol=dict(
            version=3,
            preconditioning_invocations_per_fixture=1,
            samples_per_fixture=2,
            buffer_slots="balanced_per_fixture",
            input_cases=3,
            invocations_per_sample=1,
            samples="gpu_max_over_ranks_us",
            order="rotated_and_reversed",
        ),
        dtype="bfloat16",
        cache_policy="cold",
        alias=False,
        sizes=[1024, 2048],
        hardware=dict(
            world_size=4, devices=[dict(rank=i, uuid=f"GPU-{i}") for i in range(4)]
        ),
        catalogs=[dict(bytes=size, configs=[CONFIG, OTHER]) for size in (1024, 2048)],
    )


def report(path, metadata, winners=None, *, measurements=None, summary=True):
    if winners is None:
        winners = [
            dict(size_bytes=size, config=CONFIG, median_us=2.0)
            for size in metadata["sizes"]
        ]
    if measurements is None:
        measurements = [
            dict(
                bytes=row["size_bytes"],
                config=row["config"],
                samples_us=[1.0, 2.0, 3.0],
                correctness="passed",
            )
            for row in winners
        ]
    writer = RawReport(path, metadata)
    try:
        for row in measurements:
            writer.append(row)
        if summary:
            writer.append_summary(dict(status="partial", winners=winners))
    finally:
        writer.close()
    return path


def replay(paths, metadata):
    trials, sources, selections = cli._replay_trials(
        paths, metadata, read_report, Trial
    )
    return list(trials), sources, selections


def test_default_budget_and_full_logarithmic_ladder():
    args = cli.arguments(["--output", "short.jsonl"])
    assert args.budget_seconds == 600
    assert args.rounds == 1 and args.compile_workers is None
    assert args.cache_policy == "cold" and not args.alias and not args.inplace
    assert not args.exhaustive and args.resume is None and args.replay is None
    assert not args.plot
    assert cli.arguments(["--output", "plotted.jsonl", "--plot"]).plot
    sizes = cli.range_sizes(
        args.minimum_bytes, args.maximum_bytes, args.points_per_doubling
    )
    assert len(sizes) == len(set(sizes)) == 209
    assert sizes[0] == 1024 and sizes[-1] == 8 << 20
    assert all(size % 16 == 0 for size in sizes)
    assert (
        cli.arguments(["--output", "long.jsonl", "--exhaustive"]).budget_seconds == 3600
    )


def test_explicit_arguments_and_resume_path():
    args = cli.arguments(
        [
            "--resume",
            "history.jsonl",
            "--cache-policy",
            "warm",
            "--alias",
            "--rounds",
            "15",
            "--budget-seconds",
            "42",
            "--compile-workers",
            "2",
            "--sizes",
            "2KiB",
            "1024",
            "2KiB",
        ]
    )
    assert args.output == args.resume == Path("history.jsonl")
    assert args.sizes == (1024, 2048)
    assert args.inplace and args.cache_policy == "warm"
    assert (args.rounds, args.budget_seconds, args.compile_workers) == (15, 42.0, 2)
    args = cli.arguments(
        ["--output", "wide.jsonl", "--min-bytes", "16", "--max-bytes", "16MiB"]
    )
    assert (args.minimum_bytes, args.maximum_bytes) == (16, 16 << 20)


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--resume", "a", "--output", "b"],
        ["--resume", "a", "--fresh"],
        ["--output", "a", "--replay", "b", "--exhaustive"],
        ["--output", "a", "--replay", "b", "--plot"],
        ["--resume", "a", "--replay", "b"],
        ["--output", "a", "--rounds", "0"],
        ["--output", "a", "--rounds", "513"],
        ["--output", "a", "--compile-workers", "0"],
        ["--output", "a", "--budget-seconds", "nan"],
        ["--output", "a", "--budget-seconds", "inf"],
        ["--output", "a", "--budget-seconds", "-1"],
        ["--output", "a", "--points-per-doubling", "0"],
        ["--output", "a", "--min-bytes", "2KiB", "--max-bytes", "1KiB"],
        ["--output", "a", "--sizes", "1025"],
        ["--output", "a", "--sizes", "1KiB", "--min-bytes", "2KiB"],
    ],
)
def test_invalid_arguments_fail_before_distributed_setup(flags):
    with pytest.raises(SystemExit) as error:
        cli.arguments(flags)
    assert error.value.code == 2


def test_replay_unions_measured_winners_per_size_without_duplicates(tmp_path, metadata):
    first = report(tmp_path / "first.jsonl", metadata)
    winners = [
        dict(size_bytes=1024, config=OTHER, median_us=1.0),
        dict(size_bytes=2048, config=CONFIG, median_us=2.0),
    ]
    second = report(tmp_path / "second.jsonl", metadata, winners)
    trials, sources, selections = replay([first, second], metadata)
    assert [trial.size_bytes for trial in trials] == [1024, 2048]
    assert [len(trial.configs) for trial in trials] == [2, 1]
    assert all(trial.rounds > 15 and trial.phase == "replay" for trial in trials)
    assert trials[0].configs == (CONFIG, OTHER)
    assert sources[2048][cli._key(CONFIG)] == [str(first), str(second)]
    assert selections[1024] == {cli._key(CONFIG): CONFIG, cli._key(OTHER): OTHER}


def test_replay_requires_recommendation_at_every_requested_size(tmp_path, metadata):
    path = report(
        tmp_path / "partial.jsonl",
        metadata,
        [dict(size_bytes=1024, config=CONFIG, median_us=2.0)],
    )
    with pytest.raises(ValueError, match="no recommendation.*2048"):
        replay([path], metadata)


def test_replay_requires_a_summary(tmp_path, metadata):
    path = report(tmp_path / "no-summary.jsonl", metadata, summary=False)
    with pytest.raises(ValueError, match="no recommendation summary"):
        replay([path], metadata)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_policy", "warm"),
        ("alias", True),
        ("dtype", "float32"),
        ("hardware", dict(world_size=2)),
    ],
)
def test_replay_rejects_incompatible_measurement_protocol(
    tmp_path, metadata, field, value
):
    different = deepcopy(metadata)
    different[field] = value
    path = report(tmp_path / "different.jsonl", different)
    with pytest.raises(
        ValueError, match="incompatible hardware or measurement semantics"
    ):
        replay([path], metadata)


@pytest.mark.parametrize(
    "failure",
    [
        dict(samples_us=[], correctness="failed", error="fixture mismatch"),
        dict(samples_us=[1.0, 2.0, 3.0], correctness="failed"),
    ],
)
def test_replay_rejects_a_recommendation_with_failed_measurements(
    tmp_path, metadata, failure
):
    records = [
        dict(
            bytes=size, config=CONFIG, samples_us=[1.0, 2.0, 3.0], correctness="passed"
        )
        for size in metadata["sizes"]
    ]
    records.append(dict(bytes=1024, config=CONFIG, **failure))
    path = report(tmp_path / "failed.jsonl", metadata, measurements=records)
    with pytest.raises(ValueError, match="failed|invalid|successful"):
        replay([path], metadata)


def test_replay_rejects_an_unmeasured_recommendation(tmp_path, metadata):
    path = report(tmp_path / "unmeasured.jsonl", metadata, measurements=[])
    with pytest.raises(ValueError, match="measured|successful"):
        replay([path], metadata)


def test_replay_collects_fresh_timings_after_sampling_protocol_changes(
    tmp_path, metadata
):
    previous = deepcopy(metadata)
    previous["timing_protocol"] = dict(version=2, input_cases=3, samples_per_fixture=1)
    path = report(tmp_path / "previous-sampling.jsonl", previous)
    trials, sources, _ = replay([path], metadata)
    assert len(trials) == 2 and all(trial.phase == "replay" for trial in trials)
    assert all(trial.rounds > 15 for trial in trials)
    assert all(
        sources[size][cli._key(CONFIG)] == [str(path)] for size in metadata["sizes"]
    )


def test_replay_scales_sampling_from_reported_candidate_latency(tmp_path, metadata):
    metadata = deepcopy(metadata)
    metadata["sizes"] = [4096, 1048576]
    metadata["catalogs"] = [
        dict(bytes=size, configs=[CONFIG]) for size in metadata["sizes"]
    ]
    winners = [
        dict(size_bytes=size, config=CONFIG, median_us=latency)
        for size, latency in ((4096, 6.0), (1048576, 100.0))
    ]
    path = report(tmp_path / "sizes.jsonl", metadata, winners)
    trials, _, _ = replay([path], metadata)
    assert [trial.rounds * 6 for trial in trials] == [672, 42]
