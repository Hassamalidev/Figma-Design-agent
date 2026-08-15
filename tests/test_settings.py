"""Settings precedence, masking, and the dashboard's settings API.

No network, no Figma, no model -- the model client is injected as a fake.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from config import Settings, load_settings
from web.app import DashboardServer
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
    return DashboardServer(FakeBridge(), store, registry, llm_factory=FakeLLM)


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
        ),
    )
    monkeypatch.setattr(dash, "_wait_for_file", lambda _key: True)
    monkeypatch.setattr(dash, "_capture", lambda *a: (_ for _ in ()).throw(RuntimeError("no socket")))

    dash._run_worker("fk", "Untitled", "a dashboard")

    assert dash._run_status == "done"
    assert dash._run_result["created_node_count"] == 1
