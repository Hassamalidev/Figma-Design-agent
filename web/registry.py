"""Local history of Figma files the plugin has ever connected to, with a
thumbnail from the last time we saw each one. Populates the web dashboard's
file gallery.

This is deliberately dumb storage, not a Figma API client: it only knows
about files you've actually opened with the plugin running, and only what
the bridge's "hello" handshake + a screenshot told it -- consistent with
CLAUDE.md's "the agent doesn't know Figma from memory" (section 1).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_REGISTRY_PATH = Path(__file__).parent / "known_files.json"


@dataclass
class FileEntry:
    file_key: str
    file_name: str
    thumbnail_base64: str | None
    last_seen: str  # ISO 8601 timestamp; callers stamp it (see web/app.py's _now())


class Registry:
    """Reads/writes the JSON file history.

    Not thread-safe on its own -- a caller mutating it from multiple threads
    (web/app.py) is expected to hold its own lock around upsert/get calls
    that need to be atomic together.
    """

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH):
        self._path = path

    def list_files(self) -> list[FileEntry]:
        """All known files, most recently seen first."""
        entries = [FileEntry(**data) for data in self._read().values()]
        entries.sort(key=lambda e: e.last_seen, reverse=True)
        return entries

    def get(self, file_key: str) -> FileEntry | None:
        data = self._read().get(file_key)
        return FileEntry(**data) if data else None

    def remove(self, file_key: str) -> bool:
        """Forget a file. Returns False if it was not there.

        Local bookkeeping only -- this gallery is built from files the plugin
        has connected to, so a removed file reappears the next time you open it
        with the plugin running. Nothing in Figma is touched.
        """
        data = self._read()
        if file_key not in data:
            return False
        del data[file_key]
        self._write(data)
        return True

    def upsert(self, entry: FileEntry) -> None:
        data = self._read()
        data[entry.file_key] = asdict(entry)
        self._write(data)

    def _read(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
