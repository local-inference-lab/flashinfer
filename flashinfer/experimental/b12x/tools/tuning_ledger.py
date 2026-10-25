"""Summarize b12x tuning ledger records.

Preparation sessions append one record per completed race to the ``ledger_v1``
table of each race-progress database beside the selection cache (see
``b12x.preparation._ledger``). This tool reads those records and reports, per
component: candidates raced, default regret (how much slower the heuristic
default is than the winner), pruning regret per autotune effort (how much
slower the best candidate admitted at a lower effort is than the winner),
runner-up margin, round-to-round noise, and wall time. Shard records that
tensor-parallel ranks wrote for one race are merged into one logical run
first. ``--replications`` compares repeated runs of the same key: winner
agreement, per-candidate latency spread, and the regret of one run's winner
under another run's measurements.

    python -m b12x.tools.tuning_ledger [--root DIR] [--component TEXT]
        [--records] [--replications] [--json PATH]

The default root is the ``preparation`` directory under the CuTe compile cache
(``B12X_COMPILE_CACHE_DIR``, else ``$XDG_CACHE_HOME/b12x/compile``, else
``~/.cache/b12x/compile``).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

EFFORT_RANK = {"low": 0, "medium": 1, "high": 2, "max": 3}
MARGINS_PERCENT = (1, 2, 5, 10)


def default_root() -> Path:
    root = os.environ.get("B12X_COMPILE_CACHE_DIR")
    if root:
        return Path(root) / "preparation"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "b12x" / "compile" / "preparation"
    return Path.home() / ".cache" / "b12x" / "compile" / "preparation"


def load_records(root):
    """Load every ledger record under ``root`` with its cache identity and selection."""
    records = []
    for path in sorted(Path(root).glob("*.progress.sqlite3")):
        identity, selections = {}, {}
        sibling = path.with_name(path.name[: -len(".progress.sqlite3")] + ".json")
        if sibling.exists():
            try:
                payload = json.loads(sibling.read_text())
                identity = payload.get("identity", {})
                selections = payload.get("records", {})
            except (OSError, ValueError):
                identity, selections = {}, {}
        connection = sqlite3.connect(path)
        try:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if "ledger_v1" not in tables:
                continue
            rows = connection.execute(
                "SELECT id, key, shard_index, shard_count, session, wall_time, payload "
                "FROM ledger_v1 ORDER BY id"
            ).fetchall()
        finally:
            connection.close()
        for row in rows:
            record = json.loads(row[6])
            record["_ledger"] = {
                "database": str(path),
                "id": row[0],
                "shard_index": row[2],
                "shard_count": row[3],
                "session": row[4],
                "wall_time": row[5],
                "namespace": identity.get("namespace", {}),
                "selected": selections.get(row[1]),
            }
            records.append(record)
    return records


def _merge_shards(shards):
    """Combine one race's shard records into a single logical run."""
    shards = sorted(shards, key=lambda r: r["_ledger"]["shard_index"])
    run = json.loads(json.dumps({k: v for k, v in shards[0].items() if k != "_ledger"}))
    run["_ledger"] = dict(shards[0]["_ledger"])
    by_index = {c["index"]: c for c in run["candidates"]}
    rejected, batches = [], []
    phase_seconds = defaultdict(float)
    seconds = 0.0
    for shard in shards:
        index = shard["_ledger"]["shard_index"]
        for candidate in shard["candidates"]:
            if candidate.get("latency_us") is not None:
                merged = dict(candidate)
                merged["shard_index"] = index
                by_index[candidate["index"]] = merged
        rejected.extend(shard.get("rejected", ()))
        batches.extend(
            {**batch, "shard_index": index} for batch in shard.get("batches", ())
        )
        for name, value in shard.get("phase_seconds", {}).items():
            phase_seconds[name] += value
        seconds = max(seconds, shard.get("seconds") or 0.0)
    run["candidates"] = [by_index[c["index"]] for c in run["candidates"]]
    measured = [c for c in run["candidates"] if c.get("latency_us") is not None]
    champion = (
        None
        if not measured
        else min(measured, key=lambda c: (c["latency_us"], c["index"]))
    )
    run["champion"] = {
        "index": None if champion is None else champion["index"],
        "latency_us": None if champion is None else champion["latency_us"],
    }
    run["rejected"] = rejected
    run["rejected_count"] = sum(shard.get("rejected_count", 0) for shard in shards)
    run["batches"] = batches
    run["phase_seconds"] = dict(phase_seconds)
    run["seconds"] = seconds
    run["_ledger"]["shards_merged"] = len(shards)
    run["_ledger"]["sessions"] = [shard["_ledger"]["session"] for shard in shards]
    return run


def merge_runs(records):
    """Return logical runs: unsharded records as they are, shard records merged.

    Shard records of one key and shard count are ordered by start time and
    grouped greedily; a repeated shard index starts the next run. A run whose
    shard count exceeds the records found is kept and marked incomplete via
    ``_ledger["shards_merged"]``.
    """
    runs = []
    sharded = defaultdict(list)
    for record in records:
        count = record["_ledger"]["shard_count"]
        if count == 1:
            runs.append(record)
        else:
            sharded[(record["key"], count)].append(record)
    for items in sharded.values():
        items.sort(key=lambda r: (r.get("started_at") or r["_ledger"]["wall_time"]))
        current = {}
        for item in items:
            index = item["_ledger"]["shard_index"]
            if index in current:
                runs.append(_merge_shards(current.values()))
                current = {}
            current[index] = item
        if current:
            runs.append(_merge_shards(current.values()))
    runs.sort(key=lambda r: (r.get("started_at") or r["_ledger"]["wall_time"]))
    return runs


def _complete(run):
    ledger = run["_ledger"]
    return (
        ledger["shard_count"] == 1
        or ledger.get("shards_merged") == ledger["shard_count"]
    )


def _cv(values):
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return None if mean <= 0 else statistics.pstdev(values) / mean


def analyze(record):
    """Per-race metrics; ``None`` when the race measured nothing."""
    candidates = record["candidates"]
    measured = [c for c in candidates if c.get("latency_us") is not None]
    if not measured:
        return None
    latencies = sorted(c["latency_us"] for c in measured)
    winner_us = latencies[0]
    winner = min(measured, key=lambda c: (c["latency_us"], c["index"]))
    default = next(
        (c for c in measured if c["index"] == record.get("default_index")), None
    )
    pruning = {}
    if all(c.get("admission") is not None for c in measured):
        effort_rank = EFFORT_RANK[record["effort"]]
        for level, rank in EFFORT_RANK.items():
            if rank > effort_rank:
                continue
            pool = [
                c["latency_us"] for c in measured if EFFORT_RANK[c["admission"]] <= rank
            ]
            pruning[level] = None if not pool else min(pool) / winner_us - 1
    noise = [
        value
        for value in (
            _cv([r["mean_us"] for r in c.get("rounds", ())]) for c in measured
        )
        if value is not None
    ]
    return {
        "component_id": record["component_id"],
        "key": record["key"],
        "kind": record["kind"],
        "effort": record["effort"],
        "candidates": len(candidates),
        "measured": len(measured),
        "rejected": record.get("rejected_count", 0),
        "winner_index": winner["index"],
        "winner_us": winner_us,
        "runner_up_ratio": None if len(latencies) < 2 else latencies[1] / winner_us,
        "median_ratio": statistics.median(latencies) / winner_us,
        "worst_ratio": latencies[-1] / winner_us,
        "default_index": record.get("default_index"),
        "default_regret": None
        if default is None
        else default["latency_us"] / winner_us - 1,
        "pruning_regret": pruning,
        "within": {
            str(percent): sum(
                1 for value in latencies if value <= winner_us * (1 + percent / 100)
            )
            for percent in MARGINS_PERCENT
        },
        "noise_cv": None if not noise else statistics.median(noise),
        "winner_cv": _cv([r["mean_us"] for r in winner.get("rounds", ())]),
        "compiled_candidates": sum(
            1 for c in candidates if c.get("precompiled") is False
        ),
        "seconds": record.get("seconds"),
        "phase_seconds": record.get("phase_seconds", {}),
        "legal_by_effort": record.get("coverage", {}).get("legal_by_effort"),
        "started_at": record.get("started_at"),
        "query": record.get("query"),
    }


def replications(records):
    """Compare repeated complete runs of one key."""
    groups = defaultdict(list)
    for record in records:
        if _complete(record) and record["champion"]["index"] is not None:
            groups[record["key"]].append(record)
    rows = []
    for key, items in groups.items():
        if len(items) < 2:
            continue
        winners = [item["champion"]["index"] for item in items]
        modal = max(set(winners), key=winners.count)
        by_index = defaultdict(list)
        tables = []
        for item in items:
            table = {
                c["index"]: c["latency_us"]
                for c in item["candidates"]
                if c.get("latency_us") is not None
            }
            tables.append(table)
            for index, latency in table.items():
                by_index[index].append(latency)
        spreads = [
            max(values) / min(values) - 1
            for values in by_index.values()
            if len(values) >= 2 and min(values) > 0
        ]
        cross = []
        for a, table_a in enumerate(tables):
            winner_a = items[a]["champion"]["index"]
            for b, table_b in enumerate(tables):
                if a == b or winner_a not in table_b or not table_b:
                    continue
                cross.append(table_b[winner_a] / min(table_b.values()) - 1)
        rows.append(
            {
                "key": key,
                "component_id": items[0]["component_id"],
                "effort": items[0]["effort"],
                "replications": len(items),
                "distinct_winners": len(set(winners)),
                "winner_agreement": winners.count(modal) / len(winners),
                "candidate_spread_p50": None
                if not spreads
                else statistics.median(spreads),
                "candidate_spread_p90": _quantile(spreads, 0.9),
                "cross_regret_p50": None if not cross else statistics.median(cross),
                "cross_regret_max": None if not cross else max(cross),
                "query": items[0]["query"],
            }
        )
    rows.sort(key=lambda row: (row["component_id"], -row["replications"]))
    return rows


def _quantile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def _pct(value):
    return "-" if value is None else f"{100 * value:.1f}%"


def _ratio(value):
    return "-" if value is None else f"{value:.3f}x"


def _count(value):
    return "-" if value is None else f"{value:.0f}"


def summarize(analyses):
    """Per-component table over race analyses."""
    groups = defaultdict(list)
    for item in analyses:
        groups[item["component_id"]].append(item)
    header = (
        f"{'component':<38} {'races':>5} {'cand':>6} {'meas':>6} "
        f"{'default regret p50/p90':>22} {'prune low/med/high p50':>23} "
        f"{'2nd/1st p50':>11} {'noise p50':>9} {'secs':>8}"
    )
    lines = [header, "-" * len(header)]
    for component in sorted(groups):
        items = groups[component]
        regrets = [i["default_regret"] for i in items]
        prune = "/".join(
            _pct(_quantile([i["pruning_regret"].get(level) for i in items], 0.5))
            for level in ("low", "medium", "high")
        )
        lines.append(
            f"{component[:38]:<38} {len(items):>5} "
            f"{_count(_quantile([i['candidates'] for i in items], 0.5)):>6} "
            f"{_count(_quantile([i['measured'] for i in items], 0.5)):>6} "
            f"{_pct(_quantile(regrets, 0.5)) + '/' + _pct(_quantile(regrets, 0.9)):>22} "
            f"{prune:>23} "
            f"{_ratio(_quantile([i['runner_up_ratio'] for i in items], 0.5)):>11} "
            f"{_pct(_quantile([i['noise_cv'] for i in items], 0.5)):>9} "
            f"{sum(i['seconds'] or 0 for i in items):>8.0f}"
        )
    return "\n".join(lines)


def describe_records(analyses):
    header = (
        f"{'component':<34} {'effort':<6} {'cand':>5} {'meas':>5} {'win':>5} "
        f"{'winner us':>10} {'2nd/1st':>8} {'default':>8} {'prune l/m/h':>20} "
        f"{'noise':>6} {'secs':>6}"
    )
    lines = [header, "-" * len(header)]
    for item in analyses:
        prune = "/".join(
            _pct(item["pruning_regret"].get(level))
            for level in ("low", "medium", "high")
        )
        lines.append(
            f"{item['component_id'][:34]:<34} {item['effort']:<6} "
            f"{item['candidates']:>5} {item['measured']:>5} {item['winner_index']:>5} "
            f"{item['winner_us']:>10.1f} {_ratio(item['runner_up_ratio']):>8} "
            f"{_pct(item['default_regret']):>8} {prune:>20} "
            f"{_pct(item['noise_cv']):>6} {(item['seconds'] or 0):>6.0f}"
        )
    return "\n".join(lines)


def describe_replications(rows):
    header = (
        f"{'component':<34} {'effort':<6} {'runs':>4} {'winners':>7} "
        f"{'agree':>6} {'spread p50/p90':>16} {'cross regret p50/max':>21}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['component_id'][:34]:<34} {row['effort']:<6} {row['replications']:>4} "
            f"{row['distinct_winners']:>7} {_pct(row['winner_agreement']):>6} "
            f"{_pct(row['candidate_spread_p50']) + '/' + _pct(row['candidate_spread_p90']):>16} "
            f"{_pct(row['cross_regret_p50']) + '/' + _pct(row['cross_regret_max']):>21}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--component", default=None, help="substring filter")
    parser.add_argument("--records", action="store_true", help="one line per race")
    parser.add_argument("--replications", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="write analyses")
    args = parser.parse_args(argv)
    root = default_root() if args.root is None else args.root
    records = load_records(root)
    if args.component:
        records = [r for r in records if args.component in r["component_id"]]
    runs = merge_runs(records)
    analyses = [item for item in map(analyze, runs) if item is not None]
    print(f"{len(records)} ledger records, {len(runs)} runs under {root}")
    if not analyses:
        return 0
    print(summarize(analyses))
    if args.records:
        print()
        print(describe_records(analyses))
    rows = replications(runs) if args.replications else []
    if args.replications:
        print()
        print(describe_replications(rows) if rows else "no repeated races")
    if args.json is not None:
        args.json.write_text(
            json.dumps(
                {"analyses": analyses, "replications": rows}, indent=1, allow_nan=False
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
