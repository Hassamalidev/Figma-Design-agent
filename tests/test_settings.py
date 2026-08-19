"""Settings precedence, masking, and the dashboard's settings API.

No network, no Figma, no model -- the model client is injected as a fake.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config import Settings, load_settings
from web.app import DashboardServer
from web.history import History
from web.registry import FileEntry, Registry
from web.settings_store import SettingsStore, mask


@pytest.fixture
def store(tmp_path, monkeypatch):
    # Isolate from the developer's real environment/.env.
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME", "BRIDGE_HOST", "BRIDGE_PORT"):
        monkeypatch.delenv(key, raising=False)
    return SettingsStore(env_path=tmp_path / "absent.env", runtime_path=tmp_path / "runtime.json")


def test_boots_with_no_credentials_at_all(store):
    """The dashboard must start unconfigured so the UI can collect the keys."""
    settings = store.effective()

    assert settings.is_model_configured is False
    assert settings.bridge_host == "localhost"  # defaults still apply
    assert settings.bridge_port == 9223


def test_cli_style_load_still_fails_loudly(tmp_path, monkeypatch):
    """main.py has nowhere to ask, so it must not start half-configured."""
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(RuntimeError):
        load_settings(tmp_path / "absent.env", require_model=True)


def test_ui_values_are_saved_and_take_effect(store):
    store.update(
        {
            "model_base_url": "http://localhost:11434/v1",
            "model_api_key": "ollama",
            "model_name": "gpt-oss:20b-cloud",
        }
    )
    settings = store.effective()

    assert settings.is_model_configured is True
    assert settings.model_name == "gpt-oss:20b-cloud"
    assert store.view().sources["model_api_key"] == "ui"


def test_ui_values_win_over_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "MODEL_BASE_URL=https://from-env/v1\nMODEL_API_KEY=env-key\nMODEL_NAME=env-model\n",
        encoding="utf-8",
    )
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    store = SettingsStore(env_path=env, runtime_path=tmp_path / "runtime.json")

    assert store.effective().model_name == "env-model"
    assert store.view().sources["model_name"] == "env"

    store.update({"model_name": "ui-model"})

    assert store.effective().model_name == "ui-model"
    assert store.view().sources["model_name"] == "ui"
    # Untouched fields still come from .env.
    assert store.effective().model_api_key == "env-key"
    assert store.view().sources["model_api_key"] == "env"


def test_blank_value_clears_an_override(store):
    store.update({"model_name": "temporary"})
    assert store.effective().model_name == "temporary"

    store.update({"model_name": ""})
    assert store.effective().model_name == ""


def test_survives_a_corrupt_runtime_file(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text("}}not json{{", encoding="utf-8")
    store = SettingsStore(env_path=tmp_path / "absent.env", runtime_path=path)

    assert store.effective().model_name == ""  # no crash
    store.update({"model_name": "recovered"})
    assert store.effective().model_name == "recovered"


def test_api_key_is_masked_never_sent_raw():
    assert mask("sk-or-v1-abcdefghijklmnop") == "sk-o••••••••mnop"
    assert mask("short") == "•••••"
    assert mask("") == ""


# ---- dashboard API ------------------------------------------------------


class FakeBridge:
    is_connected = False
    current_file = None
    connection_generation = 0


class FakeLLM:
    """Records construction args so we can prove credentials reach the client."""

    last = None

    def __init__(self, base_url, api_key, model):
        FakeLLM.last = (base_url, api_key, model)
        self._model = model

    def complete(self, messages, tools=None):
        if self._model == "explode":
            raise RuntimeError("bad credentials")
        return SimpleNamespace(content="ok", tool_calls=None)


def make_dashboard(tmp_path, monkeypatch):
    for key in ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    store = SettingsStore(env_path=tmp_path / "absent.env", runtime_path=tmp_path / "runtime.json")
    registry = Registry(tmp_path / "files.json")
    # History MUST be redirected too. It defaults to web/history.json -- the
    # real one -- so every test that drove a run was writing a fake "a dashboard
    # / Untitled / 5ms" entry into the user's own History tab, 50 of which had
    # completely buried their actual runs.
    history = History(tmp_path / "history.json")
    return DashboardServer(FakeBridge(), store, registry, llm_factory=FakeLLM, history=history)


def test_settings_snapshot_never_leaks_the_key(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "super-secret-key-1234", "model_name": "m"}
    )

    snapshot = dash.settings_snapshot()

    assert "super-secret-key-1234" not in json.dumps(snapshot)
    assert snapshot["has_api_key"] is True
    assert snapshot["configured"] is True


def test_omitting_the_key_keeps_the_existing_one(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings({"model_base_url": "http://x/v1", "model_api_key": "keep-me", "model_name": "m"})

    # The UI sends no key when the user leaves the field blank.
    dash.update_settings({"model_base_url": "http://y/v1", "model_name": "m2"})

    assert dash.settings_store.effective().model_api_key == "keep-me"
    assert dash.settings_store.effective().model_base_url == "http://y/v1"


def test_run_is_refused_until_a_model_is_configured(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)

    ok, message = dash.start_run("any-key", "build something")

    assert ok is False
    assert "settings" in message.lower()


def test_connection_test_reports_success_and_failure(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    assert dash.test_model_connection()["ok"] is False  # nothing configured yet

    dash.update_settings({"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "good"})
    assert dash.test_model_connection()["ok"] is True
    assert FakeLLM.last == ("http://x/v1", "k", "good")  # credentials really reached the client

    dash.update_settings({"model_name": "explode"})
    failure = dash.test_model_connection()
    assert failure["ok"] is False
    assert "bad credentials" in failure["error"]


def test_setup_snapshot_points_at_the_real_manifest(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)

    setup = dash.setup_snapshot()

    assert setup["manifest_path"].endswith("manifest.json")
    assert Settings  # imported for the type it documents
    assert setup["bridge_url"].startswith("ws://")


# ---- the dashboard's run path actually executes -----------------------------

def test_the_dashboard_run_worker_reaches_the_agent(tmp_path, monkeypatch):
    """Regression: `build_critic_client` was called in _run_worker but never
    imported, so every dashboard run died with a NameError the moment the
    plugin connected. Nothing caught it because no test executed this path --
    the CLI and benchmark wire the agent separately.
    """
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )

    seen = {}

    def fake_run(instruction, bridge, llm, max_retries, max_steps, **kwargs):
        seen["instruction"] = instruction
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            success=True, created_node_ids=[], failed_steps=[], warnings=[],
            layout_defects=[], final_screenshot_base64=None, step_results=[],
            design_notes=[], requirements_met=[], requirements_missing=[],
        )

    monkeypatch.setattr("web.app.loop.run", fake_run)
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: True)
    monkeypatch.setattr(dash, "_capture", lambda *a: None)

    dash._run_worker("fk", "Untitled", "a dashboard")

    assert seen["instruction"] == "a dashboard"
    assert "critic_llm" in seen["kwargs"]      # the critic is wired through
    assert dash._run_status == "done"


def test_no_vision_critic_configured_passes_none_rather_than_failing(tmp_path, monkeypatch):
    """A blank CRITIC_MODEL_NAME must be a normal, quiet outcome."""
    for key in ("CRITIC_BASE_URL", "CRITIC_API_KEY", "CRITIC_MODEL_NAME"):
        monkeypatch.delenv(key, raising=False)
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )

    seen = {}
    monkeypatch.setattr(
        "web.app.loop.run",
        lambda *a, **kw: (seen.update(kw), SimpleNamespace(
            success=True, created_node_ids=[], failed_steps=[], warnings=[],
            layout_defects=[], final_screenshot_base64=None, step_results=[],
            design_notes=[], requirements_met=[], requirements_missing=[],
        ))[1],
    )
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: True)
    monkeypatch.setattr(dash, "_capture", lambda *a: None)

    dash._run_worker("fk", "Untitled", "a dashboard")

    assert seen["critic_llm"] is None
    assert dash._run_status == "done"


def test_a_failed_thumbnail_refresh_does_not_discard_a_successful_run(tmp_path, monkeypatch):
    """The design is already built by then. Reporting "Run crashed" and losing
    the result because a screenshot failed afterwards is the wrong trade."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )

    monkeypatch.setattr(
        "web.app.loop.run",
        lambda *a, **kw: SimpleNamespace(
            success=True, created_node_ids=["1:1"], failed_steps=[], warnings=[],
            layout_defects=[], final_screenshot_base64=None, step_results=[],
            design_notes=[], requirements_met=[], requirements_missing=[],
        ),
    )
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: True)
    monkeypatch.setattr(dash, "_capture", lambda *a: (_ for _ in ()).throw(RuntimeError("no socket")))

    dash._run_worker("fk", "Untitled", "a dashboard")

    assert dash._run_status == "done"
    assert dash._run_result["created_node_count"] == 1


# ---- dashboard: preferences, progress and resilience -----------------------


def test_the_api_hands_the_ui_the_bounds_it_should_enforce(tmp_path, monkeypatch):
    """The page renders its number inputs from this, instead of hardcoding
    min/max in HTML where they can drift from what the store accepts."""
    dash = make_dashboard(tmp_path, monkeypatch)

    schema = {row["key"]: row for row in dash.settings_snapshot()["prefs_schema"]}

    assert schema["max_steps"]["maximum"] == 100
    assert schema["visual_gate"]["kind"] == "bool"


def test_a_rejected_preference_comes_back_with_its_reason(tmp_path, monkeypatch):
    """Silently reverting an input leaves the user with no idea why."""
    dash = make_dashboard(tmp_path, monkeypatch)

    reply = dash.update_prefs({"max_steps": 5000})

    assert reply["prefs"]["max_steps"] == 100
    assert reply["pref_errors"] and "between 1 and 100" in reply["pref_errors"][0]


def test_a_valid_preference_reports_no_errors(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)

    reply = dash.update_prefs({"visual_gate": "false"})

    assert reply["prefs"]["visual_gate"] is False
    assert reply["pref_errors"] == []


def test_status_carries_live_metrics_for_the_progress_display(tmp_path, monkeypatch):
    from agent.metrics import RunMetrics

    dash = make_dashboard(tmp_path, monkeypatch)
    measured = RunMetrics()
    measured.start_step("Add the hero section", index=3, total=8)
    dash._run_metrics = measured

    progress = dash.status_snapshot()["metrics"]["progress"]

    assert progress["index"] == 3 and progress["total"] == 8
    assert progress["step"] == "Add the hero section"


def test_a_broken_metrics_read_never_breaks_the_status_endpoint(tmp_path, monkeypatch):
    """It is read from the HTTP thread while the run thread writes. A status
    endpoint that 500s because a counter moved is worse than a skipped frame."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash._run_metrics = SimpleNamespace(snapshot=lambda: (_ for _ in ()).throw(RuntimeError("torn")))

    assert dash.status_snapshot()["metrics"] is None


def test_the_connection_watcher_survives_a_failed_snapshot(tmp_path, monkeypatch):
    """A screenshot can fail (plugin closed, Dev Mode). That used to kill the
    watcher thread outright, after which the gallery silently stopped updating
    for the rest of the process."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.bridge.current_file = SimpleNamespace(file_key="fk", file_name="Untitled")
    dash.bridge.connection_generation = 1
    calls = []

    def exploding_capture(*args):
        calls.append(args)
        raise RuntimeError("plugin went away")

    monkeypatch.setattr(dash, "_capture", exploding_capture)
    monkeypatch.setattr("web.app.time.sleep", lambda _s: (_ for _ in ()).throw(StopIteration))

    with pytest.raises(StopIteration):     # only our sentinel escapes the loop
        dash.watch_connections()

    assert calls, "the watcher must have tried"
    # ...and the generation is NOT marked done, so the next tick retries it.
    assert dash._last_captured_generation == -1


# ---- the file gallery must show each file's OWN design ---------------------


def _dashboard_in_file(tmp_path, monkeypatch, file_key, image):
    """A dashboard whose plugin is currently in `file_key`, rendering `image`."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.bridge.current_file = SimpleNamespace(file_key=file_key, file_name=file_key)
    monkeypatch.setattr(
        "web.app.get_screenshot",
        lambda bridge, node_id=None: {"ok": image is not None, "image_base64": image, "error": None},
    )
    return dash


def test_a_thumbnail_is_stored_for_the_file_that_is_actually_open(tmp_path, monkeypatch):
    dash = _dashboard_in_file(tmp_path, monkeypatch, "fk-a", "PICTURE-OF-A")

    dash._capture("fk-a", "Design A")

    assert dash.registry.get("fk-a").thumbnail_base64 == "PICTURE-OF-A"


def test_one_designs_picture_never_lands_on_another_designs_card(tmp_path, monkeypatch):
    """The reported bug. The screenshot comes from whatever file the plugin is
    in now; `file_key` is what we file it under. If the user switched files,
    those differ -- and storing it anyway mislabels the design."""
    dash = _dashboard_in_file(tmp_path, monkeypatch, "fk-b", "PICTURE-OF-B")

    dash._capture("fk-a", "Design A")   # asked for A, but the plugin is in B

    assert dash.registry.get("fk-a") is None


def test_switching_files_mid_render_is_also_caught(tmp_path, monkeypatch):
    """The check has to happen after the screenshot too: rendering takes time,
    and the plugin can move during it."""
    dash = _dashboard_in_file(tmp_path, monkeypatch, "fk-a", "PICTURE-OF-B")

    def wander(bridge, node_id=None):
        dash.bridge.current_file = SimpleNamespace(file_key="fk-b", file_name="B")
        return {"ok": True, "image_base64": "PICTURE-OF-B", "error": None}

    monkeypatch.setattr("web.app.get_screenshot", wander)
    dash._capture("fk-a", "Design A")

    assert dash.registry.get("fk-a") is None


def test_a_failed_screenshot_keeps_the_last_good_thumbnail(tmp_path, monkeypatch):
    """`upsert` replaces the whole entry, so writing None erased the picture --
    a card would go blank because one render happened to fail."""
    dash = _dashboard_in_file(tmp_path, monkeypatch, "fk-a", "GOOD-PICTURE")
    dash._capture("fk-a", "Design A")

    monkeypatch.setattr(
        "web.app.get_screenshot",
        lambda bridge, node_id=None: {"ok": False, "image_base64": None, "error": "Dev Mode"},
    )
    dash._capture("fk-a", "Design A")

    assert dash.registry.get("fk-a").thumbnail_base64 == "GOOD-PICTURE"


def test_no_plugin_connected_captures_nothing(tmp_path, monkeypatch):
    dash = _dashboard_in_file(tmp_path, monkeypatch, "fk-a", "PICTURE")
    dash.bridge.current_file = None

    dash._capture("fk-a", "Design A")

    assert dash.registry.get("fk-a") is None


# ---- removing a file from the gallery -------------------------------------
#
# There is no "delete the Figma file" anywhere here, because no such operation
# exists: a plugin runs INSIDE a file, the Plugin API has no deleteFile, and
# figma.fileKey is read-only. The two things that ARE possible -- forgetting the
# file locally, and emptying its canvas -- are separate, and the destructive one
# is opt-in.


def _gallery_with(tmp_path, monkeypatch, connected_key=None):
    dash = make_dashboard(tmp_path, monkeypatch)
    for key, name in (("fk-a", "Design A"), ("fk-b", "Design B")):
        dash.registry.upsert(
            FileEntry(file_key=key, file_name=name, thumbnail_base64="PIC", last_seen="2026-01-01")
        )
    if connected_key:
        dash.bridge.current_file = SimpleNamespace(file_key=connected_key, file_name=connected_key)
    return dash


def test_removing_a_file_forgets_it_without_touching_figma(tmp_path, monkeypatch):
    dash = _gallery_with(tmp_path, monkeypatch)
    ran = []
    monkeypatch.setattr("web.app.execute_figma_js", lambda *a, **k: ran.append(a) or {"ok": True})

    ok, message = dash.forget_file("fk-a")

    assert ok and "Removed" in message
    assert dash.registry.get("fk-a") is None
    assert dash.registry.get("fk-b") is not None      # only the one asked for
    assert ran == [], "nothing may be executed in Figma for a gallery-only removal"


def test_removing_an_unknown_file_is_refused(tmp_path, monkeypatch):
    dash = _gallery_with(tmp_path, monkeypatch)

    ok, message = dash.forget_file("nope")

    assert not ok and "not in the gallery" in message


def test_clearing_the_canvas_requires_that_file_to_be_open(tmp_path, monkeypatch):
    """The script always runs in whichever file the plugin is in, so clearing
    a file that is not open would wipe a different design entirely."""
    dash = _gallery_with(tmp_path, monkeypatch, connected_key="fk-b")
    ran = []
    monkeypatch.setattr("web.app.execute_figma_js", lambda *a, **k: ran.append(a) or {"ok": True})

    ok, message = dash.forget_file("fk-a", clear_canvas=True)

    assert not ok
    assert "not open with the plugin running" in message
    assert ran == []
    assert dash.registry.get("fk-a") is not None      # nothing removed either


def test_clearing_the_canvas_runs_only_for_the_connected_file(tmp_path, monkeypatch):
    dash = _gallery_with(tmp_path, monkeypatch, connected_key="fk-a")
    monkeypatch.setattr(
        "web.app.execute_figma_js",
        lambda bridge, code, **k: {"ok": True, "result": {"removed": 4}, "error": None},
    )

    ok, message = dash.forget_file("fk-a", clear_canvas=True)

    assert ok
    assert "Deleted 4 top-level layer(s)" in message
    assert dash.registry.get("fk-a") is None


def test_a_failed_clear_keeps_the_file_in_the_gallery(tmp_path, monkeypatch):
    """Reporting it as removed while the canvas is untouched would be a lie."""
    dash = _gallery_with(tmp_path, monkeypatch, connected_key="fk-a")
    monkeypatch.setattr(
        "web.app.execute_figma_js",
        lambda bridge, code, **k: {"ok": False, "result": None, "error": "Dev Mode is read-only"},
    )

    ok, message = dash.forget_file("fk-a", clear_canvas=True)

    assert not ok and "Dev Mode" in message
    assert dash.registry.get("fk-a") is not None


def test_nothing_is_deleted_while_a_run_is_in_progress(tmp_path, monkeypatch):
    dash = _gallery_with(tmp_path, monkeypatch, connected_key="fk-a")
    dash._run_status = "running"

    ok, message = dash.forget_file("fk-a")

    assert not ok and "run is in progress" in message
    assert dash.registry.get("fk-a") is not None


def test_the_delete_endpoint_reports_refusals(tmp_path, monkeypatch):
    dash = _gallery_with(tmp_path, monkeypatch)

    ok, _ = dash.forget_file("fk-a")
    assert ok
    ok_again, message = dash.forget_file("fk-a")
    assert not ok_again and message      # a second delete says why, rather than pretending


# ---- the dashboard pages through screens instead of embedding them --------


def test_the_status_payload_carries_screen_NAMES_not_images(tmp_path, monkeypatch):
    """/api/status is polled every 1.5s. Five full-page PNGs in that payload is
    megabytes a minute for pictures that never change."""
    from types import SimpleNamespace

    import web.app as app_module

    result = SimpleNamespace(
        success=True, created_node_ids=["1:1"], failed_steps=[], warnings=[],
        layout_defects=[], design_notes=[], requirements_met=[], requirements_missing=[],
        screens=["Login", "Dashboard"], metrics={},
        screen_shots=[
            {"name": "Login", "image_base64": "AAAA"},
            {"name": "Dashboard", "image_base64": "BBBB"},
        ],
        final_screenshot_base64="AAAA",
    )

    payload = app_module._result_payload(result)

    assert payload["screen_names"] == ["Login", "Dashboard"]
    assert "AAAA" not in json.dumps(payload)


def test_each_screen_is_served_on_its_own(tmp_path, monkeypatch):
    import base64

    dash = make_dashboard(tmp_path, monkeypatch)
    dash._run_screens = [
        {"name": "Login", "image_base64": base64.b64encode(b"login-png").decode()},
        {"name": "Dashboard", "image_base64": base64.b64encode(b"dash-png").decode()},
    ]

    assert dash.screen_image(0) == b"login-png"
    assert dash.screen_image(1) == b"dash-png"
    assert dash.screen_image(2) is None      # past the end
    assert dash.screen_image(-1) is None     # and before the start


def test_a_corrupt_screen_image_does_not_raise(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash._run_screens = [{"name": "Login", "image_base64": "not base64 at all!!"}]

    assert dash.screen_image(0) is None


def test_starting_a_run_clears_the_previous_runs_screens(tmp_path, monkeypatch):
    """Otherwise the pager shows the last design while the new one builds."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )
    dash._run_screens = [{"name": "Old", "image_base64": "AAAA"}]
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: False)

    dash.start_run("fk", "a new design")
    import time

    for _ in range(50):                      # the worker runs on its own thread
        if not dash._run_screens:
            break
        time.sleep(0.02)

    assert dash._run_screens == []


# ---- stopping a run from the dashboard ------------------------------------


def test_stopping_when_nothing_is_running_is_refused(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)

    ok, message = dash.stop_run()

    assert not ok and "no run in progress" in message


def test_stopping_sets_the_signal_the_run_reads(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash._run_status = "running"

    ok, message = dash.stop_run()

    assert ok and "Stopping" in message
    assert dash._stop.is_set()
    assert dash.status_snapshot()["status"] == "stopping"


def test_stopping_twice_is_harmless(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash._run_status = "running"
    dash.stop_run()

    ok, message = dash.stop_run()

    assert ok and "Already stopping" in message


def test_a_new_run_is_never_born_already_stopping(tmp_path, monkeypatch):
    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )
    dash._run_status = "running"
    dash.stop_run()
    dash._run_status = "idle"          # the previous run finished
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: False)

    dash.start_run("fk", "a new design")

    assert not dash._stop.is_set()


def test_waiting_for_a_file_can_be_stopped(tmp_path, monkeypatch):
    """The easiest thing of all to abandon -- nothing has happened yet."""
    dash = make_dashboard(tmp_path, monkeypatch)
    dash._stop.set()

    assert dash._wait_for_file("fk") is False


def test_a_stopped_run_is_recorded_as_stopped_not_failed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import web.app as app_module

    stopped = SimpleNamespace(
        success=False, stopped=True, created_node_ids=["1:1"], failed_steps=[], warnings=[],
        layout_defects=[], design_notes=[], requirements_met=[], requirements_missing=[],
        screens=["Login"], screen_shots=[], metrics={}, final_screenshot_base64=None,
    )

    assert app_module._history_status(stopped, "") == "stopped"
    assert app_module._result_payload(stopped)["stopped"] is True

    finished = SimpleNamespace(**{**stopped.__dict__, "stopped": False, "success": True})
    assert app_module._history_status(finished, "") == "done"
    assert app_module._result_payload(finished)["stopped"] is False


def test_a_crash_is_still_recorded_as_an_error(tmp_path, monkeypatch):
    import web.app as app_module

    assert app_module._history_status(None, "TypeError: boom") == "error"
    assert app_module._history_status(None, "Stopped before the run started.") == "stopped"


# ---- attachments through the dashboard ------------------------------------


def test_the_vision_model_can_be_set_from_the_dashboard(tmp_path, monkeypatch):
    """The refusal message for an unreadable image points at Settings, so
    Settings has to actually have the control."""
    dash = make_dashboard(tmp_path, monkeypatch)

    dash.update_settings({"vision_model_name": "gemma4:cloud"})

    snapshot = dash.settings_snapshot()
    assert snapshot["vision_model_name"] == "gemma4:cloud"
    assert snapshot["has_vision"] is True
    assert snapshot["sources"]["vision_model_name"] == "ui"


def test_a_configured_critic_already_counts_as_vision(tmp_path, monkeypatch):
    """Someone who set up screenshot critique should not have to configure a
    second model to attach a screenshot."""
    from config import Settings

    settings = Settings("http://x", "k", "m", critic_base_url="http://v",
                        critic_model_name="gemma4:cloud")

    assert settings.has_vision
    assert settings.vision_settings()[2] == "gemma4:cloud"


def test_an_attachment_that_cannot_be_decoded_is_a_400_not_a_started_run():
    """Starting a run, spending model calls and then reporting that the file
    could not be opened is the worst ordering available."""
    from agent import reference

    with pytest.raises(reference.ReferenceError):
        reference.from_payload([{"name": "x.png", "data_base64": "!!!!"}])


def test_an_attachment_reaches_the_run(tmp_path, monkeypatch):
    """The whole path: posted bytes -> decoded -> described -> into the loop."""
    import base64
    import time

    from agent import loop as loop_module
    from agent import reference

    dash = make_dashboard(tmp_path, monkeypatch)
    dash.update_settings(
        {"model_base_url": "http://x/v1", "model_api_key": "k", "model_name": "m",
         "vision_model_name": "seeing-model"}
    )
    dash.registry.upsert(
        FileEntry(file_key="fk", file_name="Untitled", thumbnail_base64=None, last_seen="2026-01-01")
    )
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: True)

    seen: dict = {}

    def capture(*args, **kwargs):
        seen["references"] = kwargs.get("references")
        return _blank_result()

    monkeypatch.setattr(loop_module, "run", capture)
    monkeypatch.setattr(
        "web.app.build_vision_client",
        lambda _s: SimpleNamespace(
            complete=lambda messages, tools=None: SimpleNamespace(
                content="COLORS\nAccent: #6C5CE7", tool_calls=None
            )
        ),
    )

    attachments = reference.from_payload(
        [{"name": "shot.png", "data_base64": base64.b64encode(b"png bytes").decode()}]
    )
    dash.start_run("fk", "rebuild this", "create", attachments)
    for _ in range(100):
        if "references" in seen:
            break
        time.sleep(0.02)

    assert "Accent: #6C5CE7" in (seen.get("references") or "")
    assert dash.status_snapshot()["status"] == "done"


def _blank_result():
    from agent.state import RunResult

    return RunResult(
        instruction="rebuild this", success=True, created_node_ids=["1:1"],
        failed_steps=[], warnings=[],
    )
