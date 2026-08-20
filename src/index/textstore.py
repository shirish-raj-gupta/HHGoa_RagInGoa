"""
On-disk passage text, keyed by passage_id.

The index holds vectors; something still has to hold the words. Keeping
14.3M passage texts in RAM costs ~4.3GB on top of a ~12.8GB memory-mapped
index, which does not fit the machine or a Space. But only the handful of
passages actually retrieved need their text - typically five per query.

SQLite is the right tool here and not a placeholder: single file, no server,
sub-millisecond primary-key lookup, and the OS page cache keeps the hot set
resident without us managing it. Reads are the only operation on the critical
path, so the connection opens read-only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable


class TextStore:
    def __init__(self, path: Path, read_only: bool = True):
        self.path = Path(path)
        if read_only:
            self.conn = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro", uri=True,
                check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self.conn.executescript(
                "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;"
                "CREATE TABLE IF NOT EXISTS passages("
                "  passage_id TEXT PRIMARY KEY, lang TEXT, text TEXT) WITHOUT ROWID;")

    def add_many(self, rows: Iterable[tuple[str, str, str]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO passages(passage_id, lang, text) VALUES (?,?,?)",
            rows)
        self.conn.commit()

    def get(self, passage_ids: list[str]) -> dict[str, str]:
        """Batch lookup. One query, not N - this sits on the critical path."""
        if not passage_ids:
            return {}
        qs = ",".join("?" * len(passage_ids))
        cur = self.conn.execute(
            f"SELECT passage_id, text FROM passages WHERE passage_id IN ({qs})",
            passage_ids)
        return dict(cur.fetchall())

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]

    def close(self) -> None:
        self.conn.close()
