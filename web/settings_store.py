"""Runtime settings entered from the dashboard UI, layered over .env.

Precedence is deliberately "UI wins": if you type a key into the settings
panel it is used even when .env also has one, because the panel is the more
recent, more explicit act. Nothing here is ever written back to .env -- the
overrides live in their own git-ignored file so a shared .env stays clean.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from config import Settings, env_configured_keys, load_settings

logger = logging.getLogger(__name__)

DEFAULT_RUNTIME_PATH = Path(__file__).parent / "runtime_settings.json"

# Only these may be set from the UI. Bridge host/port are excluded on purpose:
# they must match the plugin manifest's networkAccess, which is a file edit.
# `vision_model_name` is here and its base URL / key are not, on purpose: the
# fallback chain in `Settings.vision_settings` already borrows those from the
# critic or the main model, so naming a multimodal model is usually the ONLY
# thing a user has to do to unlock reading an attached screenshot. Three more
# inputs to express what one usually says is a worse panel, not a fuller one.
EDITABLE = ("model_base_url", "model_api_key", "model_name", "vision_model_name")

# Run preferences, also editable from the UI. Each one changes real behaviour --
# nothing here is a decorative switch.
#
# Every preference is declared ONCE, here, with its type, its bounds and where
# its default comes from. The dashboard renders its inputs from this same
# declaration (`schema()` below), so the UI's min/max, the value the storage
# layer will accept, and the value the run actually uses cannot drift apart.
# They previously could, and did: the numeric inputs advertised max="100" while
# the store accepted any integer at all.


class PrefError(ValueError):
    """A preference value the UI sent that we will not store. The message is
    shown to the user -- silently dropping it left them staring at an input
    that would not keep what they typed."""


@dataclass(frozen=True)
class BoolPref:
    """A toggle. `default` applies until the user changes it."""

    default: bool

    kind = "bool"

    def coerce(self, key: str, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off", ""):
                return False
        raise PrefError(f"{key}: expected true or false, got {value!r}")


@dataclass(frozen=True)
class IntPref:
    """A bounded whole number whose default comes from the .env `Settings`.

    Taking the default from Settings is the point: `.env` MAX_RETRIES=5 used to
    apply to the CLI and be ignored by the dashboard, which ran 3 and displayed
    3, from the same configuration file.
    """

    settings_field: str
    minimum: int
    maximum: int

    kind = "int"

    def coerce(self, key: str, value: object) -> int:
        # bool is a subclass of int in Python, so `int(True)` is a silent 1 --
        # a toggle posted to a numeric field became "1 retry" with no complaint.
        if isinstance(value, bool):
            raise PrefError(f"{key}: expected a number, got a true/false value")
        try:
            number = int(str(value).strip())
        except (TypeError, ValueError):
            raise PrefError(f"{key}: expected a whole number, got {value!r}") from None
        if not self.minimum <= number <= self.maximum:
            raise PrefError(
                f"{key}: must be between {self.minimum} and {self.maximum} "
                f"(got {number}, using {min(max(number, self.minimum), self.maximum)})"
            )
        return number

    def clamp(self, number: int) -> int:
        return min(max(number, self.minimum), self.maximum)


PREF_SPECS: dict[str, BoolPref | IntPref] = {
    # attempts per plan step -- the upper bound matches the dashboard input
    "max_retries": IntPref(settings_field="max_retries", minimum=1, maximum=10),
    # hard cap on plan length
    "max_steps": IntPref(settings_field="max_steps", minimum=1, maximum=100),
    # run the layout gate after each visual step
    "visual_gate": BoolPref(default=True),
    # preselect whichever file is currently connected
    "auto_select": BoolPref(default=True),
    # after the design is built, go back and fix what the review found
    "final_repair": BoolPref(default=True),
    # wire the finished design up as a clickable prototype
    "prototype": BoolPref(default=True),
}
PREFS = tuple(PREF_SPECS)
INT_PREFS = tuple(k for k, s in PREF_SPECS.items() if isinstance(s, IntPref))
BOOL_PREFS = tuple(k for k, s in PREF_SPECS.items() if isinstance(s, BoolPref))


@dataclass
class PrefUpdate:
    """The effective preferences after an update, plus anything we refused.

    Errors are returned rather than swallowed so the dashboard can say why a
    value did not stick. An out-of-range number is still applied (clamped) --
    refusing it outright would leave the user with no way to fix it from a
    number input that had already accepted their keystrokes.
    """

    prefs: dict[str, object]
    errors: list[str] = dc_field(default_factory=list)


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

    def defaults(self) -> dict[str, object]:
        """What each preference is before the user touches it.

        Numeric defaults come from `.env` via Settings, so MAX_RETRIES=5 means
        five retries in the CLI *and* in the dashboard, and the dashboard shows
        5. Previously the dashboard hardcoded 3 and quietly ignored the file.
        """
        settings = self.effective()
        return {
            key: (
                getattr(settings, spec.settings_field)
                if isinstance(spec, IntPref)
                else spec.default
            )
            for key, spec in PREF_SPECS.items()
        }

    def prefs(self) -> dict[str, object]:
        """Effective preferences: stored values over defaults, type-coerced.

        Read and write use the SAME coercion, so a value can only round-trip to
        itself. They used to differ -- writes went through `bool(value)` and
        reads through `str(value).lower() == "true"` -- which meant a stored `1`
        was written as True and read back as False.
        """
        result = self.defaults()
        stored = self._raw()
        for key, spec in PREF_SPECS.items():
            if key not in stored:
                continue
            try:
                result[key] = spec.coerce(key, stored[key])
            except PrefError as exc:
                # A corrupt or hand-edited file must not break the dashboard.
                logger.info("Ignoring stored preference (%s); using the default.", exc)
                if isinstance(spec, IntPref):
                    # Out of range is still the user's intent, so clamp it.
                    # Unparseable is not, so leave the default in place --
                    # writing the None back would blank the preference.
                    clamped = self._clamped_or_default(key, spec, stored[key])
                    if clamped is not None:
                        result[key] = clamped
        return result

    @staticmethod
    def _clamped_or_default(key: str, spec: "IntPref", value: object) -> object | None:
        """An out-of-range stored number is still the user's intent -- clamp it."""
        try:
            return spec.clamp(int(str(value).strip()))
        except (TypeError, ValueError):
            return None

    def update_prefs(self, values: dict) -> PrefUpdate:
        """Validate, store and report. Nothing is dropped in silence.

        Returns the effective preferences plus the reason for anything we would
        not take at face value, so the dashboard can show it instead of leaving
        the user to wonder why their input reverted.
        """
        raw = self._raw()
        errors: list[str] = []
        for key, spec in PREF_SPECS.items():
            if key not in values:
                continue  # absent means "leave it alone"
            try:
                raw[key] = spec.coerce(key, values[key])
            except PrefError as exc:
                errors.append(str(exc))
                # Out of range is a bounds problem, not a type problem: keep the
                # user's intent by clamping. Anything unparseable is discarded.
                if isinstance(spec, IntPref):
                    clamped = self._clamped_or_default(key, spec, values[key])
                    if clamped is not None:
                        raw[key] = clamped
        self._write_raw(raw)
        return PrefUpdate(prefs=self.prefs(), errors=errors)

    def schema(self) -> list[dict]:
        """How the UI should render each preference. One declaration, one truth.

        The dashboard previously hardcoded its own min/max in HTML attributes,
        which is a second place for the bounds to live and a second place for
        them to be wrong.
        """
        defaults = self.defaults()
        rows = []
        for key, spec in PREF_SPECS.items():
            row = {"key": key, "kind": spec.kind, "default": defaults[key]}
            if isinstance(spec, IntPref):
                row["minimum"], row["maximum"] = spec.minimum, spec.maximum
            rows.append(row)
        return rows

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
        """Write via a temp file + rename, so an interrupted save cannot leave
        a half-written JSON file that reads back as "no settings at all" and
        loses the user's API key."""
        self._runtime_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._runtime_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        os.replace(temp_path, self._runtime_path)  # atomic on POSIX and Windows


def mask(secret: str) -> str:
    """Never send a full API key back to the browser."""
    if not secret:
        return ""
    if len(secret) <= 8:
        return "•" * len(secret)
    return f"{secret[:4]}{'•' * 8}{secret[-4:]}"
