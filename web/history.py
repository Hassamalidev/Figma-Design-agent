"""Persistent log of past generations, for the dashboard's History tab.

Deliberately capped and append-only: a run record is small, but the final
screenshot is not, so only the most recent few keep their thumbnail.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_HISTORY_PATH = Path(__file__).parent / "history.json"
MAX_ENTRIES = 50
# Screenshots dominate the file size, so only the newest entries keep one.
MAX_THUMBNAILS = 8


@dataclass
class HistoryEntry:
    id: str
    instruction: str
    file_key: str
    file_name: str
    status: str  # done | error
    success: bool
    created_node_count: int
    failed_step_count: int
    started_at: str  # ISO 8601
    finished_at: str
    thumbnail_base64: str | None = None

    def summary(self) -> dict:
        """What the UI needs; the thumbnail is included only if still retained."""
        return asdict(self)


class History:
    """Reads/writes the run log. Newest first."""

    def __init__(self, path: Path = DEFAULT_HISTORY_PATH):
        self._path = path

    def list_entries(self) -> list[HistoryEntry]:
        return [HistoryEntry(**row) for row in self._read()]

    def add(self, entry: HistoryEntry) -> None:
        rows = self._read()
        rows.insert(0, asdict(entry))
        del rows[MAX_ENTRIES:]
        for index, row in enumerate(rows):
            if index >= MAX_THUMBNAILS:
                row["thumbnail_base64"] = None
        self._write(rows)

    def clear(self) -> None:
        self._write([])

    def _read(self) -> list[dict]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return [row for row in data if isinstance(row, dict)] if isinstance(data, list) else []

    def _write(self, rows: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
