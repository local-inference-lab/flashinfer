"""Durable tournament checkpoints for independently progressing tuning ranks."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock


@dataclass
class RaceCheckpoint:
    measurements: dict[int, float]
    champion: int


class RaceCache:
    """Commit one rank's completed batches without a tuning collective."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            # Serialize WAL/schema initialization across simultaneous rank startups.
            with FileLock(str(self.path) + ".lock", timeout=30):
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=FULL")
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS races_v1 ("
                    "key TEXT PRIMARY KEY, measured_count INTEGER NOT NULL, "
                    "payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS discoveries_v1 ("
                    "key TEXT PRIMARY KEY, payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS ledger_v1 ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL, "
                    "shard_index INTEGER NOT NULL, shard_count INTEGER NOT NULL, "
                    "session TEXT NOT NULL, wall_time REAL NOT NULL, "
                    "payload TEXT NOT NULL)"
                )
                self.connection.execute(
                    "CREATE INDEX IF NOT EXISTS ledger_v1_key ON ledger_v1 (key)"
                )
        except BaseException:
            self.connection.close()
            raise

    def load(self, key: str) -> RaceCheckpoint | None:
        row = self.connection.execute(
            "SELECT payload FROM races_v1 WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return RaceCheckpoint(dict(payload["measurements"]), payload["champion"])

    def save(self, key: str, checkpoint: RaceCheckpoint):
        payload = json.dumps(
            {
                "measurements": sorted(checkpoint.measurements.items()),
                "champion": checkpoint.champion,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        # Concurrent identical exports may have different progress. A slower writer
        # must not replace a checkpoint containing more completed candidates.
        self.connection.execute(
            "INSERT INTO races_v1 (key, measured_count, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "measured_count=excluded.measured_count, payload=excluded.payload "
            "WHERE excluded.measured_count >= races_v1.measured_count",
            (key, len(checkpoint.measurements), payload),
        )

    def delete(self, key: str):
        self.connection.execute("DELETE FROM races_v1 WHERE key = ?", (key,))

    def discovery_lock(self, key: str):
        directory = self.path.with_suffix(".discovery-locks")
        directory.mkdir(exist_ok=True)
        return FileLock(directory / (key + ".lock"))

    def load_discovery(self, key: str):
        row = self.connection.execute(
            "SELECT payload FROM discoveries_v1 WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def save_discovery(self, key: str, record):
        self.connection.execute(
            "INSERT INTO discoveries_v1 (key, payload) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET payload=excluded.payload",
            (key, json.dumps(record, separators=(",", ":"), allow_nan=False)),
        )

    def append_ledger(self, key, *, shard_index, shard_count, session, record):
        """Append one completed race record; records are never replaced."""
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False)
        self.connection.execute(
            "INSERT INTO ledger_v1 "
            "(key, shard_index, shard_count, session, wall_time, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (key, int(shard_index), int(shard_count), session, time.time(), payload),
        )

    def ledger_records(self, key=None):
        """Return appended race records in insertion order."""
        query = (
            "SELECT id, key, shard_index, shard_count, session, wall_time, payload "
            "FROM ledger_v1"
        )
        parameters = ()
        if key is not None:
            query += " WHERE key = ?"
            parameters = (key,)
        rows = self.connection.execute(query + " ORDER BY id", parameters).fetchall()
        return [
            {
                "id": row[0],
                "key": row[1],
                "shard_index": row[2],
                "shard_count": row[3],
                "session": row[4],
                "wall_time": row[5],
                "record": json.loads(row[6]),
            }
            for row in rows
        ]

    def close(self):
        self.connection.close()
