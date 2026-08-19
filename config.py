"""Loads .env into a typed Settings object. No os.getenv scattered elsewhere."""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import dotenv_values


@dataclass(frozen=True)
class Settings:
    """Typed, immutable view of the configuration."""

    model_base_url: str
    model_api_key: str
    model_name: str

    # The VISION critic. Separate on purpose: the generator needs reliable tool
    # calling and runs ~50 times per design, while the critic needs to see
    # images and runs a handful of times. Forcing one model to do both means
    # settling for the worse of each. Blank = no vision critique.
    critic_base_url: str = ""
    critic_api_key: str = ""
    critic_model_name: str = ""

    # The model that READS attachments. A screenshot only becomes a design if
    # something can see it, and the generator (gpt-oss:20b) cannot. Defaults to
    # the critic, which is already a probed vision model -- set VISION_* only to
    # use a different one for transcription than for critique.
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_model_name: str = ""

    bridge_host: str = "localhost"
    bridge_port: int = 9223

    max_retries: int = 3
    max_steps: int = 40

    @property
    def is_model_configured(self) -> bool:
        """The dashboard can start without these and collect them from the UI."""
        return bool(self.model_base_url and self.model_api_key and self.model_name)

    @property
    def has_vision_critic(self) -> bool:
        """Only a model explicitly configured for vision is asked to see images."""
        return bool(self.critic_model_name and self.critic_base_url)

    def critic_settings(self) -> tuple[str, str, str]:
        """(base_url, api_key, model) for the critic, falling back to the main key."""
        return (
            self.critic_base_url or self.model_base_url,
            self.critic_api_key or self.model_api_key,
            self.critic_model_name,
        )

    @property
    def has_vision(self) -> bool:
        """Can this configuration look at an attached image at all?

        Attachments are refused with a clear message when it cannot, rather
        than being silently dropped -- a run that ignored the screenshot you
        attached and built something generic is the worst possible outcome.
        """
        return bool(self.vision_model_name or self.critic_model_name)

    def vision_settings(self) -> tuple[str, str, str]:
        """(base_url, api_key, model) for reading attachments.

        Falls through VISION_* -> CRITIC_* -> the main model, so configuring a
        vision critic is enough to unlock screenshot input too.
        """
        model = self.vision_model_name or self.critic_model_name
        return (
            self.vision_base_url or self.critic_base_url or self.model_base_url,
            self.vision_api_key or self.critic_api_key or self.model_api_key,
            model,
        )

    def with_overrides(self, **overrides: object) -> "Settings":
        """Return a copy with non-empty overrides applied (used by the UI settings)."""
        clean = {k: v for k, v in overrides.items() if v not in (None, "")}
        return replace(self, **clean) if clean else self


def _get(values: dict[str, str | None], key: str, default: str | None = None) -> str:
    val = values.get(key) or default
    if val is None:
        raise RuntimeError(
            f"Missing required setting {key}. Copy .env.example to .env and fill it in."
        )
    return val


def _int(values: dict[str, str | None], key: str, default: str) -> int:
    try:
        return int(_get(values, key, default))
    except ValueError:
        return int(default)


def load_settings(env_path: str | Path = ".env", require_model: bool = True) -> Settings:
    """Read .env (falling back to real environment variables) into a Settings object.

    `require_model=False` lets the web dashboard boot with no credentials at
    all and collect them from its settings panel instead -- the CLI still
    fails loudly, because it has nowhere to ask.
    """
    values = {**os.environ, **dotenv_values(env_path)}
    model_default = None if require_model else ""

    return Settings(
        model_base_url=_get(values, "MODEL_BASE_URL", model_default),
        model_api_key=_get(values, "MODEL_API_KEY", model_default),
        model_name=_get(values, "MODEL_NAME", model_default),
        critic_base_url=_get(values, "CRITIC_BASE_URL", ""),
        critic_api_key=_get(values, "CRITIC_API_KEY", ""),
        critic_model_name=_get(values, "CRITIC_MODEL_NAME", ""),
        vision_base_url=_get(values, "VISION_BASE_URL", ""),
        vision_api_key=_get(values, "VISION_API_KEY", ""),
        vision_model_name=_get(values, "VISION_MODEL_NAME", ""),
        bridge_host=_get(values, "BRIDGE_HOST", "localhost"),
        bridge_port=_int(values, "BRIDGE_PORT", "9223"),
        max_retries=_int(values, "MAX_RETRIES", "3"),
        max_steps=_int(values, "MAX_STEPS", "40"),
    )


def env_configured_keys(env_path: str | Path = ".env") -> set[str]:
    """Which model settings came from .env / the environment.

    The UI shows these as "from .env" and treats them as the source of truth
    unless the user deliberately overrides them.
    """
    values = {**os.environ, **dotenv_values(env_path)}
    known = ("MODEL_BASE_URL", "MODEL_API_KEY", "MODEL_NAME", "VISION_MODEL_NAME")
    found = {key for key in known if values.get(key)}
    # A critic model already makes attachments readable, so the UI must not show
    # "unset" next to a vision field that is, in effect, set.
    if values.get("CRITIC_MODEL_NAME"):
        found.add("VISION_MODEL_NAME")
    return found
