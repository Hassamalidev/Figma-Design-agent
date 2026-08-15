"""Runtime settings entered from the dashboard UI, layered over .env.

Precedence is deliberately "UI wins": if you type a key into the settings
panel it is used even when .env also has one, because the panel is the more
recent, more explicit act. Nothing here is ever written back to .env -- the
overrides live in their own git-ignored file so a shared .env stays clean.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import Settings, env_configured_keys, load_settings

DEFAULT_RUNTIME_PATH = Path(__file__).parent / "runtime_settings.json"

# Only these may be set from the UI. Bridge host/port are excluded on purpose:
# they must match the plugin manifest's networkAccess, which is a file edit.
EDITABLE = ("model_base_url", "model_api_key", "model_name")

# Run preferences, also editable from the UI. Each one changes real behaviour --
# nothing here is a decorative switch.
PREF_DEFAULTS: dict[str, object] = {
    "max_retries": 3,      # attempts per plan step
    "max_steps": 40,       # hard cap on plan length
    "visual_gate": True,   # run the layout gate after each visual step
    "auto_select": True,   # preselect whichever file is currently connected
}
INT_PREFS = ("max_retries", "max_steps")
BOOL_PREFS = ("visual_gate", "auto_select")
PREFS = tuple(PREF_DEFAULTS)


@dataclass
class SettingsView:
    """What the UI needs to render the settings panel."""

    settings: Settings
    sources: dict[str, str]  # field -> "env" | "ui" | "unset"


class SettingsStore:
    """Merges .env with UI-entered overrides and persists the overrides."""

    def __init__(self, env_path: str | Path = ".env", runtime_path: Path = DEFAULT_RUNTIME_PATH):
        self._env_path = env_path
        self._runtime_path = runtime_path

    def effective(self) -> Settings:
        base = load_settings(self._env_path, require_model=False)
        return base.with_overrides(**self._overrides())

    def view(self) -> SettingsView:
        overrides = self._overrides()
        from_env = env_configured_keys(self._env_path)
        sources = {}
        for field in EDITABLE:
            if overrides.get(field):
                sources[field] = "ui"
            elif field.upper() in from_env:
                sources[field] = "env"
            else:
                sources[field] = "unset"
        return SettingsView(settings=self.effective(), sources=sources)

    def update(self, values: dict[str, str]) -> Settings:
        """Persist the editable subset. An empty string clears an override."""
        overrides = self._overrides()
        for field in EDITABLE:
            if field not in values:
                continue
            value = (values.get(field) or "").strip()
            if value:
                overrides[field] = value
            else:
                overrides.pop(field, None)
        self._write(overrides)
        return self.effective()

    def clear(self) -> None:
        self._write({})

    # -- run preferences -------------------------------------------------

    def prefs(self) -> dict[str, object]:
        """Effective preferences: stored values over defaults, type-coerced."""
        stored = self._raw()
        result = dict(PREF_DEFAULTS)
        for key in PREFS:
            if key not in stored:
                continue
            value = stored[key]
            if key in INT_PREFS:
                try:
                    result[key] = max(1, int(value))
                except (TypeError, ValueError):
                    continue
            elif key in BOOL_PREFS:
                result[key] = value if isinstance(value, bool) else str(value).lower() == "true"
        return result

    def update_prefs(self, values: dict) -> dict[str, object]:
        raw = self._raw()
        for key in PREFS:
            if key not in values:
                continue
            if key in INT_PREFS:
                try:
                    raw[key] = max(1, int(values[key]))
                except (TypeError, ValueError):
                    continue
            elif key in BOOL_PREFS:
                raw[key] = bool(values[key])
        self._write_raw(raw)
        return self.prefs()

    # -- storage ---------------------------------------------------------

    def _raw(self) -> dict:
        if not self._runtime_path.exists():
            return {}
        try:
            data = json.loads(self._runtime_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _overrides(self) -> dict[str, str]:
        return {k: str(v) for k, v in self._raw().items() if k in EDITABLE and v}

    def _write(self, overrides: dict[str, str]) -> None:
        """Save credentials WITHOUT disturbing stored preferences.

        Both live in one file, so a naive overwrite here would silently wipe
        every preference each time the user saved their API key.
        """
        raw = {k: v for k, v in self._raw().items() if k not in EDITABLE}
        raw.update(overrides)
        self._write_raw(raw)

    def _write_raw(self, raw: dict) -> None:
        self._runtime_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def mask(secret: str) -> str:
    """Never send a full API key back to the browser."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}{'•' * 8}{secret[-4:]}"
