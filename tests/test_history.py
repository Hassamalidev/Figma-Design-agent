"""Run history + the preferences that back the Settings toggles.

Every toggle in the UI must change real behaviour -- these tests pin that
down so none of them becomes decoration.
"""
from __future__ import annotations

import pytest

from web.history import MAX_ENTRIES, MAX_THUMBNAILS, History, HistoryEntry
from web.settings_store import PREF_DEFAULTS, SettingsStore


def entry(n: int, thumb: str | None = "PNG") -> HistoryEntry:
    return HistoryEntry(
        id=f"id{n}", instruction=f"build thing {n}", file_key="k", file_name="Test File",
        status="done", success=True, created_node_count=n, failed_step_count=0,
        started_at=f"2026-08-15T10:{n:02d}:00+00:00",
        finished_at=f"2026-08-15T10:{n:02d}:30+00:00",
        thumbnail_base64=thumb,
    )


def test_newest_run_is_listed_first(tmp_path):
    history = History(tmp_path / "h.json")
    history.add(entry(1))
    history.add(entry(2))

    assert [e.id for e in history.list_entries()] == ["id2", "id1"]


def test_history_is_capped(tmp_path):
    history = History(tmp_path / "h.json")
    for i in range(MAX_ENTRIES + 10):
        history.add(entry(i))

    assert len(history.list_entries()) == MAX_ENTRIES


def test_only_the_newest_entries_keep_a_thumbnail(tmp_path):
    """Screenshots dominate the file size, so older ones are dropped."""
    history = History(tmp_path / "h.json")
    for i in range(MAX_THUMBNAILS + 4):
        history.add(entry(i))

    entries = history.list_entries()
    assert all(e.thumbnail_base64 for e in entries[:MAX_THUMBNAILS])
    assert all(e.thumbnail_base64 is None for e in entries[MAX_THUMBNAILS:])


def test_failed_runs_are_recorded_too(tmp_path):
    history = History(tmp_path / "h.json")
    failed = entry(1)
    failed.status, failed.success = "error", False
    history.add(failed)

    stored = history.list_entries()[0]
    assert stored.status == "error" and stored.success is False


def test_survives_a_corrupt_history_file(tmp_path):
    path = tmp_path / "h.json"
    path.write_text("not json {{{", encoding="utf-8")
    history = History(path)

    assert history.list_entries() == []
    history.add(entry(1))
    assert len(history.list_entries()) == 1


# ---- preferences --------------------------------------------------------


@pytest.fixture
def store(tmp_path, monkeypatch):
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    return SettingsStore(env_path=tmp_path / "absent.env", runtime_path=tmp_path / "runtime.json")


def test_defaults_apply_when_nothing_is_stored(store):
    assert store.prefs() == PREF_DEFAULTS


def test_prefs_round_trip_with_type_coercion(store):
    store.update_prefs({"visual_gate": False, "max_retries": "5"})

    prefs = store.prefs()
    assert prefs["visual_gate"] is False
    assert prefs["max_retries"] == 5  # coerced from the string the browser sends
    assert prefs["max_steps"] == PREF_DEFAULTS["max_steps"]  # untouched


def test_nonsense_values_are_ignored_not_stored(store):
    store.update_prefs({"max_retries": "banana"})
    assert store.prefs()["max_retries"] == PREF_DEFAULTS["max_retries"]

    store.update_prefs({"max_retries": 0})
    assert store.prefs()["max_retries"] == 1  # clamped, never zero


def test_saving_credentials_does_not_wipe_prefs(store):
    """Both live in one file -- a naive overwrite silently lost every setting."""
    store.update_prefs({"visual_gate": False, "max_steps": 12})
    store.update({"model_name": "some-model", "model_api_key": "k", "model_base_url": "http://x/v1"})

    assert store.prefs()["visual_gate"] is False
    assert store.prefs()["max_steps"] == 12
    assert store.effective().model_name == "some-model"


def test_saving_prefs_does_not_wipe_credentials(store):
    store.update({"model_name": "some-model"})
    store.update_prefs({"visual_gate": False})

    assert store.effective().model_name == "some-model"
