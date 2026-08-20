"""Verify a generator model can call tools -- before a run depends on it.

Usage:
    python check_model.py                     # the model in .env
    python check_model.py qwen3-coder:480b-cloud deepseek-v3.1:671b-cloud
    python check_model.py --list              # what this endpoint offers

Tool calling is the ONE hard requirement on the generator (CLAUDE.md section 5).
A model that cannot make a tool call produces a run where every step "replies
with text instead of calling the tool" and every step fails -- which reads like
an agent bug and is not one.

Names rot and free tiers move. Probing this project's own endpoint has found
models that were retired, models that turned out to need a subscription, and
models advertising a `tools` capability that still answer in prose. So: verify
with a real request, never assume from the name.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from agent.llm import ModelClient
from agent.tool_probe import ToolProbeResult, probe_tool_calling
from config import load_settings


def available_models(base_url: str) -> list[dict]:
    """What the endpoint says it has, with capabilities where it reports them.

    Ollama's native /api/tags carries a `capabilities` list, which is a useful
    FILTER -- but never a verdict. `tools` in that list means the model claims
    to support them, which is exactly the claim this script exists to check.
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    for path in ("/api/tags", "/v1/models"):
        try:
            with urllib.request.urlopen(root + path, timeout=15) as response:
                payload = json.loads(response.read())
        except Exception:
            continue
        entries = payload.get("models") or payload.get("data") or []
        return [
            {
                "name": str(entry.get("name") or entry.get("id") or ""),
                "capabilities": entry.get("capabilities") or [],
            }
            for entry in entries
            if entry.get("name") or entry.get("id")
        ]
    return []


def probe(model: str, settings) -> ToolProbeResult:
    client = ModelClient(settings.model_base_url, settings.model_api_key, model)
    return probe_tool_calling(client, model)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = [a for a in sys.argv[1:] if a not in ("--list", "-l")]
    listing = len(args) != len(sys.argv[1:])

    settings = load_settings(require_model=False)
    if not settings.model_base_url:
        print("No MODEL_BASE_URL set. Copy .env.example to .env first.")
        return 1
    print(f"Endpoint: {settings.model_base_url}\n")

    if listing:
        found = available_models(settings.model_base_url)
        if not found:
            print("This endpoint does not list its models. Name them explicitly instead.")
            return 1
        print(f"{len(found)} model(s) on this endpoint:")
        for entry in found:
            caps = ", ".join(entry["capabilities"]) or "capabilities not reported"
            marker = "  " if "tools" in entry["capabilities"] else "  (no tools) "
            print(f"{marker}{entry['name']:<34} {caps}")
        print("\n`tools` here is the model's CLAIM. Probe it:  python check_model.py <name>")
        return 0

    candidates = args or ([settings.model_name] if settings.model_name else [])
    if not candidates:
        print("No model to check. Set MODEL_NAME in .env, or pass names as arguments.")
        return 1

    print(f"Probing {len(candidates)} model(s) with a real tool call...\n")
    results = []
    for model in candidates:
        result = probe(model, settings)
        results.append(result)
        print(f"  {model:<34} {result}")
        if result.reply and not result.usable:
            print(f"       replied: {result.reply[:150]!r}")

    usable = [r for r in results if r.usable]
    print()
    if not usable:
        print("None of these can drive a run.")
        print("Find candidates with:  python check_model.py --list")
        return 1

    best = min(usable, key=lambda r: (r.status != "ok", r.seconds))
    print(f"Use this one:  MODEL_NAME={best.model}")
    if best.status == "recovered":
        print("  (it emits calls as text; llm.py recovers them, but every step pays for it)")
    print(f"  measured {best.seconds:.1f}s for one call -- a design is roughly 50 of those.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
