"""CPU rendering of per-report collective recommendations."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from b12x.tools._collective_ladder_plot import (
    _bucket_figure,
    _choice_figure,
    _choices,
    _device_label,
    _heading,
    plot_report,
)
from b12x.tools._collective_ladder_search import RawReport


@pytest.fixture
def report(tmp_path):
    metadata = dict(
        sizes=[16, 48, 80],
        dtype="bfloat16",
        cache_policy="warm",
        alias=True,
        hardware=dict(
            devices=[dict(rank=0, host="host", uuid="GPU-abc")],
            topologies=[dict(host="host", gpu_map=[dict(index=7, uuid="abc")])],
        ),
    )
    winner = dict(
        size_bytes=48,
        confirmed=False,
        config=dict(
            implementation="joint_bandwidth",
            submode="bulk_pull",
            threads=128,
            blocks=32,
            active_blocks=3,
        ),
    )
    summary = dict(status="incomplete", winners=[winner])
    path = tmp_path / "topology.jsonl"
    writer = RawReport(path, metadata)
    writer.append_summary(dict(status="incomplete", winners=[]))
    writer.append_summary(summary)
    writer.close()
    return path, metadata, summary


def test_exact_choices_preserve_missing_and_unconfirmed_points(report):
    path, metadata, summary = report
    sizes, rows = _choices(metadata, summary)
    assert sizes == [16, 48, 80]
    assert rows == [None, summary["winners"][0], None]
    heading, details = _heading(path, metadata, summary, rows)
    assert "host/GPU 7" in heading
    assert "warm cache" in details and "aliased" in details
    assert "0/3 confirmed" in details and "incomplete" in details
    metadata["hardware"].pop("topologies")
    assert _device_label(metadata) == "host/rank 0"


def test_figures_show_actual_config_and_payloads(report):
    pytest.importorskip("matplotlib")
    path, metadata, summary = report
    sizes, rows = _choices(metadata, summary)
    heading, details = _heading(path, metadata, summary, rows)
    figure = _choice_figure(sizes, rows, heading, details)
    try:
        assert len(figure.axes) == 5
        for axis, value in zip(figure.axes[2:], (128, 32, 3)):
            assert axis.lines[0].get_ydata()[1] == value
        assert [label.get_text() for label in figure.axes[-1].get_xticklabels()] == [
            "16 B",
            "48 B",
            "80 B",
        ]
    finally:
        figure.clear()
    figure = _bucket_figure(sizes, rows, heading, details)
    try:
        labels = [label.get_text() for label in figure.axes[0].texts]
        assert labels == [
            "#000  16 B\nNo recommendation",
            "#001  48 B\nB/bulk_pull\nT128 G32 A3 ?",
            "#002  80 B\nNo recommendation",
        ]
    finally:
        figure.clear()


def test_plot_cli_renders_latest_summary_without_importing_torch(report):
    pytest.importorskip("matplotlib")
    path, _, _ = report
    before = path.read_bytes()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from b12x.tools.collective_ladder import main; "
            "import sys; main(sys.argv[1:]); assert 'torch' not in sys.modules",
            "plot",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(result.stdout)
    assert output["report"] == str(path)
    assert output["plots"] == [
        str(path.with_suffix(f".{kind}.png")) for kind in ("choices", "buckets")
    ]
    from matplotlib.image import imread

    for filename in output["plots"]:
        pixels = imread(filename)
        assert pixels.shape[0] > 300 and pixels.shape[1] > 1000
        assert pixels.min() < pixels.max()
        assert Path(filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert path.read_bytes() == before


def test_no_recommendations_rejected_before_rendering(report):
    _, metadata, summary = report
    summary["winners"] = []
    with pytest.raises(ValueError, match="no selected recommendations"):
        _choices(metadata, summary)


def test_single_bucket_render(report):
    pytest.importorskip("matplotlib")
    path, metadata, summary = report
    metadata["sizes"] = [48]
    writer = RawReport(path.with_name("single.jsonl"), metadata)
    writer.append_summary(summary)
    writer.close()
    assert all(path.is_file() for path in plot_report(path.with_name("single.jsonl")))
