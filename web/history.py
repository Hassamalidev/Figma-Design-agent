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
    # Everything below is optional so a history file written by an older build
    # still loads. A row is only useful if it says what actually happened --
    # "Success · 0 nodes" told the user nothing they could act on.
    duration_seconds: float = 0.0
    section_count: int = 0
    requirements_met: int = 0
    requirements_total: int = 0
    layout_defect_count: int = 0
    error: str = ""  # why it failed, when it did

    def summary(self) -> dict:
        """What the UI needs; the thumbnail is included only if still retained."""
        return asdict(self)


class History:
    """Reads/writes the run log. Newest first."""

    def __init__(self, path: Path = DEFAULT_HISTORY_PATH):
        self._path = path

    def list_entries(self) -> list[HistoryEntry]:
        """Newest first, ignoring anything a different build may have written.

        Unknown keys are dropped rather than raising: a history file is a log,
        and one unreadable row must never take the whole tab down.
        """
        known = set(HistoryEntry.__dataclass_fields__)
        entries = []
        for row in self._read():
            try:
                entries.append(HistoryEntry(**{k: v for k, v in row.items() if k in known}))
            except TypeError:
                continue  # missing a required field -- skip that row, keep the rest
        return entries

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
