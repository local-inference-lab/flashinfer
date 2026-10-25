"""Transactional tuning progress survives process loss and concurrent ranks."""

import multiprocessing
import os

import pytest

from b12x.preparation._race_cache import RaceCache, RaceCheckpoint


def _crash_writer(path):
    cache = RaceCache(path)
    cache.save("race", RaceCheckpoint({0: 3.0, 1: 1.0}, 1))
    cache.connection.execute("BEGIN IMMEDIATE")
    cache.connection.execute("DELETE FROM races_v1")
    os._exit(17)


def _rank_writer(path, rank):
    cache = RaceCache(path)
    for count in range(1, 6):
        cache.save(str(rank), RaceCheckpoint({i: 1.0 + i for i in range(count)}, 0))
    cache.close()


def test_committed_batch_survives_process_exit_and_uncommitted_transaction(tmp_path):
    path = tmp_path / "progress.sqlite3"
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_writer, args=(path,)
    )
    process.start()
    process.join(30)
    try:
        assert process.exitcode == 17
        cache = RaceCache(path)
        try:
            assert cache.load("race") == RaceCheckpoint({0: 3.0, 1: 1.0}, 1)
            assert cache.load("other-race") is None
        finally:
            cache.close()
    finally:
        if process.is_alive():
            process.kill()
            process.join()


def test_concurrent_ranks_keep_independent_checkpoints(tmp_path):
    path = tmp_path / "progress.sqlite3"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_rank_writer, args=(path, rank)) for rank in range(4)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
        cache = RaceCache(path)
        try:
            for rank in range(4):
                assert cache.load(str(rank)) == RaceCheckpoint(
                    {i: 1.0 + i for i in range(5)}, 0
                )
            cache.save("0", RaceCheckpoint({0: 2.0}, 0))
            assert len(cache.load("0").measurements) == 5
        finally:
            cache.close()
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def test_ledger_records_append_in_order_and_filter_by_key(tmp_path):
    cache = RaceCache(tmp_path / "progress.sqlite3")
    try:
        cache.append_ledger(
            "a", shard_index=0, shard_count=1, session="s1", record={"n": 1}
        )
        cache.append_ledger(
            "b", shard_index=1, shard_count=2, session="s2", record={"n": 2}
        )
        cache.append_ledger(
            "a", shard_index=0, shard_count=1, session="s3", record={"n": 3}
        )
        rows = cache.ledger_records()
        assert [row["record"]["n"] for row in rows] == [1, 2, 3]
        assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
        assert [row["session"] for row in cache.ledger_records("a")] == ["s1", "s3"]
        assert cache.ledger_records("b")[0]["shard_count"] == 2
        assert cache.ledger_records("missing") == []
        with pytest.raises(ValueError):
            cache.append_ledger(
                "c",
                shard_index=0,
                shard_count=1,
                session="s",
                record={"x": float("nan")},
            )
        assert len(cache.ledger_records()) == 3
    finally:
        cache.close()
    # Reopening an existing database keeps the appended records.
    cache = RaceCache(tmp_path / "progress.sqlite3")
    try:
        assert len(cache.ledger_records()) == 3
    finally:
        cache.close()
