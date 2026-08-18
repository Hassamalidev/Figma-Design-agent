"""Run history + the preferences that back the Settings toggles.

Every toggle in the UI must change real behaviour -- these tests pin that
down so none of them becomes decoration.
"""
from __future__ import annotations

import json

import pytest

from web.history import MAX_ENTRIES, MAX_THUMBNAILS, History, HistoryEntry
from web.settings_store import SettingsStore


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
    assert store.prefs() == store.defaults()


def test_prefs_round_trip_with_type_coercion(store):
    store.update_prefs({"visual_gate": False, "max_retries": "5"})

    prefs = store.prefs()
    assert prefs["visual_gate"] is False
    assert prefs["max_retries"] == 5  # coerced from the string the browser sends
    assert prefs["max_steps"] == store.defaults()["max_steps"]  # untouched


def test_nonsense_values_are_ignored_not_stored(store):
    store.update_prefs({"max_retries": "banana"})
    assert store.prefs()["max_retries"] == store.defaults()["max_retries"]

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


# ---- preference coercion ---------------------------------------------------
#
# Every test below is a bug that was live in this file. The path had no test
# coverage at all, which is why a toggle could be turned on by the string
# "false" and a numeric field could be set to 999999.


def test_a_string_false_turns_a_toggle_off(store):
    """The browser and any JSON client may send "false" as a string.
    `bool("false")` is True, which inverted the setting."""
    assert store.update_prefs({"visual_gate": "false"}).prefs["visual_gate"] is False
    assert store.update_prefs({"visual_gate": "true"}).prefs["visual_gate"] is True
    assert store.update_prefs({"visual_gate": "off"}).prefs["visual_gate"] is False


def test_reads_and_writes_use_the_same_coercion(store, tmp_path):
    """Writes went through bool(), reads through str(v).lower() == "true", so a
    stored 1 was written True and read back False."""
    import json

    (tmp_path / "runtime.json").write_text(json.dumps({"visual_gate": 1}))

    assert store.prefs()["visual_gate"] is True


def test_numbers_are_bounded_and_the_user_is_told_why(store):
    update = store.update_prefs({"max_steps": 999999})

    assert update.prefs["max_steps"] == 100  # the bound the UI advertises
    assert update.errors and "between 1 and 100" in update.errors[0]


def test_an_unusable_value_reports_an_error_instead_of_vanishing(store):
    update = store.update_prefs({"max_retries": "abc"})

    assert update.prefs["max_retries"] == store.defaults()["max_retries"]
    assert update.errors and "whole number" in update.errors[0]


def test_a_boolean_is_refused_by_a_numeric_preference(store):
    """bool is a subclass of int, so int(True) is a silent 1 -- a toggle posted
    to the wrong field became "1 retry" with no complaint."""
    update = store.update_prefs({"max_retries": True})

    assert update.prefs["max_retries"] == store.defaults()["max_retries"]
    assert update.errors


def test_env_drives_the_dashboard_defaults(tmp_path, monkeypatch):
    """The CLI honoured MAX_RETRIES from .env and the dashboard ignored it,
    running 3 and displaying 3 from the same configuration file."""
    from web.settings_store import SettingsStore

    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME", "MAX_RETRIES", "MAX_STEPS"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / "with-limits.env"
    env.write_text("MAX_RETRIES=5\nMAX_STEPS=12\n")
    store = SettingsStore(env_path=env, runtime_path=tmp_path / "runtime.json")

    assert store.prefs()["max_retries"] == 5
    assert store.prefs()["max_steps"] == 12
    # ...and the same numbers the run itself will use.
    assert store.effective().max_retries == 5
    assert store.effective().max_steps == 12


def test_a_ui_override_still_beats_env(tmp_path, monkeypatch):
    from web.settings_store import SettingsStore

    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME", "MAX_RETRIES", "MAX_STEPS"):
        monkeypatch.delenv(key, raising=False)
    env = tmp_path / "with-limits.env"
    env.write_text("MAX_RETRIES=5\n")
    store = SettingsStore(env_path=env, runtime_path=tmp_path / "runtime.json")

    assert store.update_prefs({"max_retries": 2}).prefs["max_retries"] == 2


def test_a_corrupt_stored_preference_does_not_break_the_dashboard(store, tmp_path):
    (tmp_path / "runtime.json").write_text('{"max_retries": "banana", "visual_gate": "maybe"}')

    prefs = store.prefs()

    assert prefs["max_retries"] == store.defaults()["max_retries"]
    assert prefs["visual_gate"] is store.defaults()["visual_gate"]


def test_the_schema_the_ui_renders_matches_what_the_store_enforces(store):
    """One declaration. The dashboard previously hardcoded its own min/max in
    HTML attributes, giving the bounds a second place to be wrong."""
    schema = {row["key"]: row for row in store.schema()}

    assert schema["max_steps"]["maximum"] == 100
    assert store.update_prefs({"max_steps": 101}).prefs["max_steps"] == 100
    assert schema["visual_gate"]["kind"] == "bool"
    assert schema["max_retries"]["default"] == store.defaults()["max_retries"]


def test_saving_credentials_still_preserves_preferences(store):
    """Both live in one file; a naive overwrite wiped prefs on every key save."""
    store.update_prefs({"max_retries": 7, "visual_gate": False})
    store.update({"model_name": "some-model"})

    assert store.prefs()["max_retries"] == 7
    assert store.prefs()["visual_gate"] is False
    assert store.effective().model_name == "some-model"


def test_the_settings_file_is_written_atomically(store, tmp_path):
    """An interrupted save must not leave JSON that reads back as 'no settings'
    and loses the user's API key."""
    store.update_prefs({"max_retries": 4})

    assert not list(tmp_path.glob("*.tmp"))  # no debris left behind
    assert (tmp_path / "runtime.json").read_text().strip().endswith("}")


# ---- history rows must say what actually happened -------------------------


def test_old_history_files_still_load(tmp_path):
    """Fields were added to HistoryEntry. A log written by an earlier build must
    keep working -- losing the record because the schema grew is the one thing
    a log may never do."""
    from web.history import History

    path = tmp_path / "history.json"
    path.write_text(json.dumps([{
        "id": "a", "instruction": "a dashboard", "file_key": "fk", "file_name": "Untitled",
        "status": "done", "success": True, "created_node_count": 4, "failed_step_count": 0,
        "started_at": "2026-08-18T13:00:00+00:00", "finished_at": "2026-08-18T13:01:00+00:00",
    }]))

    entries = History(path).list_entries()

    assert len(entries) == 1
    assert entries[0].duration_seconds == 0.0     # absent -> default, not a crash
    assert entries[0].requirements_total == 0


def test_an_unreadable_row_does_not_take_the_whole_log_down(tmp_path):
    from web.history import History

    path = tmp_path / "history.json"
    path.write_text(json.dumps([
        {"nonsense": True},                                    # missing everything
        {"id": "b", "instruction": "a hero", "file_key": "fk", "file_name": "F",
         "status": "done", "success": True, "created_node_count": 2, "failed_step_count": 0,
         "started_at": "x", "finished_at": "y", "unknown_future_field": 1},
    ]))

    entries = History(path).list_entries()

    assert [e.id for e in entries] == ["b"]   # the good row survives


def test_a_recorded_run_carries_what_it_achieved(tmp_path, monkeypatch):
    """A row reading "Success - 0 nodes" is a log entry nobody can act on."""
    from types import SimpleNamespace

    from web.history import History

    history = History(tmp_path / "history.json")
    from tests.test_settings import make_dashboard

    dash = make_dashboard(tmp_path, monkeypatch)
    dash.history = history
    result = SimpleNamespace(
        success=True, created_node_ids=["1:1", "1:2"], failed_steps=[], warnings=[],
        layout_defects=["[overlap] Hero: overlaps Nav"],
        requirements_met=["email field", "button"], requirements_missing=["password field"],
        metrics={"elapsed_seconds": 214.0, "steps_completed": 7},
        final_screenshot_base64=None,
    )

    dash._record_history("fk", "Marketing Site", "a sign-in screen", "2026-08-18T13:00:00+00:00", result)

    entry = history.list_entries()[0]
    assert entry.duration_seconds == 214.0
    assert entry.section_count == 7
    assert (entry.requirements_met, entry.requirements_total) == (2, 3)
    assert entry.layout_defect_count == 1


def test_a_failed_run_records_why(tmp_path, monkeypatch):
    from web.history import History
    from tests.test_settings import make_dashboard

    history = History(tmp_path / "history.json")
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.history = history

    dash._record_history("fk", "F", "a hero", "2026-08-18T13:00:00+00:00", None,
                         error="Timed out waiting for 'F' to connect.")

    entry = history.list_entries()[0]
    assert entry.status == "error"
    assert "Timed out" in entry.error


# ---- a run without its own screenshot still gets a picture ----------------


def test_a_run_without_a_screenshot_borrows_its_files_thumbnail(tmp_path, monkeypatch):
    from web.registry import FileEntry
    from tests.test_settings import make_dashboard

    dash = make_dashboard(tmp_path, monkeypatch)
    dash.registry.upsert(FileEntry(file_key="fk", file_name="F",
                                   thumbnail_base64="FILE-PICTURE", last_seen="2026-01-01"))
    dash._record_history("fk", "F", "a hero", "2026-08-18T13:00:00+00:00", None)

    row = dash.history_snapshot()[0]

    assert row["thumbnail_base64"] == "FILE-PICTURE"
    # Flagged, because it is the file as it looks NOW -- not a picture of this run.
    assert row["thumbnail_is_file"] is True


def test_a_runs_own_screenshot_is_never_replaced(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from web.registry import FileEntry
    from tests.test_settings import make_dashboard

    dash = make_dashboard(tmp_path, monkeypatch)
    dash.registry.upsert(FileEntry(file_key="fk", file_name="F",
                                   thumbnail_base64="FILE-PICTURE", last_seen="2026-01-01"))
    result = SimpleNamespace(
        success=True, created_node_ids=["1:1"], failed_steps=[], warnings=[], layout_defects=[],
        requirements_met=[], requirements_missing=[], metrics={},
        final_screenshot_base64="THIS-RUNS-PICTURE",
    )
    dash._record_history("fk", "F", "a hero", "2026-08-18T13:00:00+00:00", result)

    row = dash.history_snapshot()[0]

    assert row["thumbnail_base64"] == "THIS-RUNS-PICTURE"
    assert "thumbnail_is_file" not in row


def test_borrowed_thumbnails_are_capped(tmp_path, monkeypatch):
    """Otherwise opening the tab ships one full-page PNG per row, duplicated."""
    from web.registry import FileEntry
    from tests.test_settings import make_dashboard

    dash = make_dashboard(tmp_path, monkeypatch)
    dash.registry.upsert(FileEntry(file_key="fk", file_name="F",
                                   thumbnail_base64="FILE-PICTURE", last_seen="2026-01-01"))
    for _ in range(dash.THUMBNAIL_FALLBACK_LIMIT + 4):
        dash._record_history("fk", "F", "a hero", "2026-08-18T13:00:00+00:00", None)

    rows = dash.history_snapshot()
    borrowed = [r for r in rows if r.get("thumbnail_is_file")]

    assert len(borrowed) == dash.THUMBNAIL_FALLBACK_LIMIT


def test_the_test_dashboard_never_writes_to_the_real_history(tmp_path, monkeypatch):
    """`History()` defaults to web/history.json. Every test that drove a run was
    writing a fake "a dashboard / Untitled / 5ms" row into the user's own
    History tab -- 50 of which had completely buried their real runs."""
    from tests.test_settings import make_dashboard

    dash = make_dashboard(tmp_path, monkeypatch)

    assert str(tmp_path) in str(dash.history._path)
