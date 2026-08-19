"""The ONLY place a model provider is configured.

This is the swap point: hosted free API today, local model tomorrow, changed
via .env only -- no code edits anywhere else. Never import a vendor SDK
outside this file.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from types import SimpleNamespace
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessage

from agent import metrics

logger = logging.getLogger(__name__)

# A hosted endpoint blipping is not a bug in the design agent. A single 500 from
# Ollama Cloud killed a six-task benchmark sweep mid-run and lost every result
# with it, so transient failures are absorbed here -- provider quirk handling,
# which is exactly what this file is for (CLAUDE.md golden rule 1).
TRANSIENT_ERRORS = (InternalServerError, RateLimitError, APIConnectionError, APITimeoutError)
MAX_TRANSIENT_RETRIES = 4
BACKOFF_SECONDS = 2.0


class ModelClient:
    """Wraps any OpenAI-compatible chat endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> Any:
        """Return the assistant message (which may contain tool calls).

        Timed and counted: model calls are the single most expensive thing a run
        does (~50 per design), so "did that change help" is unanswerable without
        this number. Recording here rather than at the call sites means every
        caller -- loop, planner, critic -- is measured by construction.
        """
        started = time.monotonic()
        try:
            resp = self._create_with_retry(messages, tools)
        except Exception:
            metrics.current().observe_model(time.monotonic() - started, ok=False)
            raise
        metrics.current().observe_model(time.monotonic() - started)
        _record_usage(resp)

        message: ChatCompletionMessage = resp.choices[0].message
        if tools and not message.tool_calls:
            message = _recover_tool_call_from_content(message, tools)
        return message

    def _create_with_retry(self, messages: list[dict], tools: list[dict] | None):
        """Absorb transient endpoint failures with exponential backoff.

        Only retries errors that are actually worth retrying: a 400 (bad
        request -- e.g. an image sent to a text-only model) is a real answer and
        must surface immediately rather than being retried four times.
        """
        for attempt in range(MAX_TRANSIENT_RETRIES + 1):
            try:
                return self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=tools or None,
                    temperature=0.2,  # low: reliable code + tool calls, not creativity
                )
            except TRANSIENT_ERRORS as exc:
                if attempt >= MAX_TRANSIENT_RETRIES:
                    logger.info(
                        "Model endpoint still failing after %d retries: %s",
                        MAX_TRANSIENT_RETRIES,
                        type(exc).__name__,
                    )
                    raise
                metrics.current().model_transient_retries += 1
                delay = BACKOFF_SECONDS * (2**attempt)
                logger.info(
                    "Model endpoint returned %s; retrying in %.0fs (%d/%d).",
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    MAX_TRANSIENT_RETRIES,
                )
                time.sleep(delay)
        raise RuntimeError("unreachable")  # pragma: no cover


def _record_usage(response) -> None:
    """Token counts, when the endpoint reports them.

    Not every OpenAI-compatible server fills `usage` in, so this is
    best-effort by design -- a missing field must never take a run down over a
    statistic.
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    metrics.current().observe_tokens(
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )


def build_critic_client(settings) -> "ModelClient | None":
    """The vision critic, or None when no critic model is configured.

    Kept here because it is model wiring, and every entry point (CLI, dashboard,
    benchmark) needs the same answer. Returning None is a first-class outcome:
    the run then skips screenshot critique entirely rather than sending images
    to a text-only endpoint and eating a 400 per step to find out.
    """
    if not getattr(settings, "has_vision_critic", False):
        return None
    base_url, api_key, model = settings.critic_settings()
    return ModelClient(base_url, api_key, model)


def build_vision_client(settings) -> "ModelClient | None":
    """The model that reads attached screenshots, or None if none is configured.

    Falls through VISION_* -> CRITIC_* -> nothing, so a user who already set up
    a vision critic gets screenshot input for free. Returning None is a
    first-class outcome: the run then REFUSES the attachment with a clear
    message instead of quietly ignoring it and building something generic,
    which is the worst way this could fail.
    """
    if not getattr(settings, "has_vision", False):
        return None
    base_url, api_key, model = settings.vision_settings()
    if not model:
        return None
    return ModelClient(base_url, api_key, model)


def _recover_tool_call_from_content(message: ChatCompletionMessage, tools: list[dict]) -> Any:
    """Fallback for local models with unreliable tool-calling (CLAUDE.md section 5:
    "tool-call reliability drops on small models... this is the single biggest
    reliability lever for local models"). Some servers/models emit a perfectly
    valid tool call as plain JSON *text* in `content` instead of populating the
    API's real `tool_calls` field. If that's what happened, parse it out into
    the same shape a real tool call would have, so agent/loop.py never has to
    know the difference. Returns `message` unchanged if `content` isn't a
    recognizable tool call.
    """
    if not message.content:
        return message

    known_names = {t["function"]["name"] for t in tools}
    by_argument = _tools_by_sole_argument(tools)
    calls = []
    for block in _json_candidates(message.content):
        calls.extend(_calls_from(block, known_names, by_argument))
    if not calls:
        return message
    return SimpleNamespace(content=None, tool_calls=calls)


def _tools_by_sole_argument(tools: list[dict]) -> dict[str, str]:
    """`{"spec": "render_ui", "code": "execute_figma_js"}` -- but only where the
    mapping is unambiguous.

    A model that emits a tool call as text often drops the envelope with it and
    prints just the ARGUMENTS: a ```json block holding `{"spec": {...}}` and no
    `name` field anywhere. It is a complete, correct call missing only its own
    label, and the label is recoverable because exactly one tool takes a
    required argument called `spec`. A run lost its whole correcting pass to
    this -- the loop saw text, decided the step was finished, and the fix the
    model had just written was never executed.
    """
    claims: dict[str, set[str]] = {}
    for tool in tools:
        function = tool.get("function") or {}
        required = ((function.get("parameters") or {}).get("required")) or []
        if len(required) == 1 and function.get("name"):
            claims.setdefault(str(required[0]), set()).add(str(function["name"]))
    return {argument: next(iter(names)) for argument, names in claims.items() if len(names) == 1}


def _calls_from(
    parsed: Any, known_names: set[str], by_argument: dict[str, str] | None = None
) -> list[Any]:
    """Pull tool calls out of one parsed JSON block.

    Handles both the bare `{"name": ..., "arguments": {...}}` shape and the
    invented `{"calls": [{"name": ..., "arguments": {...}}, ...]}` wrapper
    that models sometimes produce when batching several calls into one reply.
    """
    if not isinstance(parsed, dict):
        return []

    if isinstance(parsed.get("calls"), list):
        found: list[Any] = []
        for entry in parsed["calls"]:
            found.extend(_calls_from(entry, known_names, by_argument))
        return found

    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if name not in known_names or not isinstance(arguments, dict):
        # No envelope: is this block itself a tool's arguments? Only when one
        # tool -- and one only -- takes a required argument by that name.
        named = {
            (by_argument or {}).get(key) for key in parsed if key in (by_argument or {})
        }
        if len(named) != 1 or not isinstance(parsed.get(next(iter(named), "")), object):
            return []
        name, arguments = next(iter(named)), parsed
        if name not in known_names:
            return []
    return [
        SimpleNamespace(
            id=uuid.uuid4().hex,
            function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        )
    ]


def _json_candidates(content: str) -> list[Any]:
    """Parse the whole message as JSON, else every ```-fenced JSON block in it.

    Models that narrate ("Below are three calls: ```json ... ``` ```json ...
    ```") bury real calls inside prose, so a whole-string parse alone misses
    them.
    """
    blocks: list[Any] = []
    try:
        return [json.loads(_strip_code_fence(content))]
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"```(?:json)?\s*\n(.*?)```", content, re.DOTALL):
        try:
            blocks.append(json.loads(match.group(1).strip()))
        except json.JSONDecodeError:
            continue
    return blocks


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()
