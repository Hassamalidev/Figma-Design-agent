"""Prove a model can really CALL TOOLS, before a run depends on it.

Tool calling is the one hard requirement on the generator (CLAUDE.md section 5):
the whole loop is `render_ui` / `edit_ui` calls, and a model that cannot make
one produces a run where every step "replies with text instead of calling the
tool" and every step fails.

There is already `vision_probe` for the critic. This is its counterpart, and it
exists for the same reason: **verify with a real request, never assume from the
name**. Probing this project's own endpoint has found models that were retired,
models that needed a subscription, and -- the one that matters here -- models
advertising a `tools` capability that still answer in prose.

Four outcomes, deliberately kept apart because they call for different fixes:

  ok          the model called the tool, with sane arguments
  recovered   it printed the call as text; `llm.py` salvaged it. Usable, but
              every step pays for the round trip that recovery costs
  no_tools    it answered in prose and nothing was recoverable -- unusable
  error       the endpoint refused: rate limit, 403, dead model name

The task is one the answer is checkable for, so a model that calls the tool
with nonsense is not scored as a pass.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

# A tool shaped like the ones the agent really offers -- one required object
# argument -- so this measures the thing the run actually needs rather than a
# toy string parameter that a weaker model can manage.
PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "render_ui",
            "description": (
                "Build a UI section. Pass the whole section as one `spec` tree."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": (
                            'A UI tree, e.g. {"kind":"section","name":"...","children":['
                            '{"kind":"text","value":"..."}]}'
                        ),
                    }
                },
                "required": ["spec"],
            },
        },
    }
]

PROBE_PROMPT = (
    "Build a section named 'Sign in' containing one text node reading exactly "
    "'Welcome back'. Call render_ui once with the whole spec. Do not reply with prose."
)


@dataclass
class ToolProbeResult:
    model: str
    status: str          # ok | recovered | no_tools | error
    detail: str
    seconds: float = 0.0
    reply: str = ""

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "recovered")

    def __str__(self) -> str:
        mark = {"ok": "PASS", "recovered": "WEAK", "no_tools": "FAIL", "error": "ERROR"}[self.status]
        timing = f" ({self.seconds:.1f}s)" if self.seconds else ""
        return f"{mark}{timing}: {self.detail}"


def probe_tool_calling(client, model: str = "") -> ToolProbeResult:
    """One real request. Returns what happened, never raises."""
    started = time.monotonic()
    try:
        message = client.complete([{"role": "user", "content": PROBE_PROMPT}], tools=PROBE_TOOL)
    except Exception as exc:
        return ToolProbeResult(model, "error", _explain(exc), time.monotonic() - started)
    elapsed = time.monotonic() - started

    calls = getattr(message, "tool_calls", None)
    content = (getattr(message, "content", "") or "").strip()
    if not calls:
        return ToolProbeResult(
            model, "no_tools",
            "answered in prose -- no tool call, and nothing recoverable from the text",
            elapsed, content[:200],
        )

    call = calls[0]
    name = getattr(getattr(call, "function", None), "name", "")
    raw = getattr(getattr(call, "function", None), "arguments", "") or "{}"
    if name != "render_ui":
        return ToolProbeResult(model, "no_tools", f"called {name!r}, not the tool it was given", elapsed)
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return ToolProbeResult(model, "no_tools", "tool call arguments were not valid JSON", elapsed, raw[:200])

    spec = args.get("spec") if isinstance(args, dict) else None
    if not isinstance(spec, dict):
        # A call with no `spec` is the failure that cost a real run half its
        # tool budget, so it is a genuine result -- not a pass with a caveat.
        return ToolProbeResult(
            model, "no_tools",
            f"called render_ui but sent {sorted(args) if isinstance(args, dict) else type(args).__name__} "
            "instead of a `spec` object",
            elapsed, raw[:200],
        )
    if "welcome back" not in json.dumps(spec).lower():
        return ToolProbeResult(
            model, "no_tools", "called the tool but ignored what it was asked to build",
            elapsed, json.dumps(spec)[:200],
        )

    # `llm.py` normalises a call the model printed as text. It works, but the
    # model still burned a turn on prose, on every step -- so it is reported as
    # its own outcome rather than folded into a pass.
    status = "recovered" if getattr(message, "recovered_from_text", False) else "ok"
    detail = (
        "called render_ui correctly"
        if status == "ok"
        else "emitted the call as TEXT; llm.py recovered it (costs a round trip per step)"
    )
    return ToolProbeResult(model, status, detail, elapsed, json.dumps(spec)[:200])


def _explain(exc: Exception) -> str:
    """Turn an SDK exception into the thing the user has to do about it."""
    text = str(exc)
    status = getattr(exc, "status_code", None) or _status_in(text)
    if status == 429 or "rate limit" in text.lower() or "quota" in text.lower():
        return "rate limited / out of quota on this model right now"
    if status == 403:
        return "403 -- this model needs a paid subscription"
    if status == 404 or "not found" in text.lower():
        return "no such model at this endpoint (names rot; pick another)"
    if status == 410 or "retired" in text.lower():
        return "retired by the provider"
    if status == 401:
        return "401 -- the API key was rejected"
    return f"{type(exc).__name__}: {text[:160]}"


def _status_in(text: str) -> int | None:
    match = re.search(r"\b(4\d\d|5\d\d)\b", text)
    return int(match.group(1)) if match else None
