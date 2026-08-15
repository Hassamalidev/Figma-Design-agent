"""Pure filesystem logic -- no network, no Figma."""
from __future__ import annotations

from web.registry import FileEntry, Registry


def test_upsert_then_get(tmp_path):
    registry = Registry(tmp_path / "known_files.json")
    entry = FileEntry(file_key="abc", file_name="Sign In", thumbnail_base64="Zm9v", last_seen="2026-08-13T00:00:00+00:00")

    registry.upsert(entry)

    assert registry.get("abc") == entry
    assert registry.get("missing") is None


def test_upsert_overwrites_same_key(tmp_path):
    registry = Registry(tmp_path / "known_files.json")
    registry.upsert(FileEntry("abc", "Old Name", None, "2026-08-13T00:00:00+00:00"))
    registry.upsert(FileEntry("abc", "New Name", "aGk=", "2026-08-13T01:00:00+00:00"))

    files = registry.list_files()
    assert len(files) == 1
    assert files[0].file_name == "New Name"


def test_list_files_sorted_most_recent_first(tmp_path):
    registry = Registry(tmp_path / "known_files.json")
    registry.upsert(FileEntry("a", "Older", None, "2026-08-13T00:00:00+00:00"))
    registry.upsert(FileEntry("b", "Newer", None, "2026-08-13T02:00:00+00:00"))
    registry.upsert(FileEntry("c", "Middle", None, "2026-08-13T01:00:00+00:00"))

    names = [f.file_name for f in registry.list_files()]
    assert names == ["Newer", "Middle", "Older"]


def test_missing_registry_file_is_empty(tmp_path):
    registry = Registry(tmp_path / "does_not_exist.json")
    assert registry.list_files() == []
    assert registry.get("anything") is None


def test_survives_a_corrupt_registry_file(tmp_path):
    path = tmp_path / "known_files.json"
    path.write_text("not json{{{", encoding="utf-8")
    registry = Registry(path)

    assert registry.list_files() == []
    registry.upsert(FileEntry("a", "Fresh Start", None, "2026-08-13T00:00:00+00:00"))
    assert registry.get("a").file_name == "Fresh Start"
