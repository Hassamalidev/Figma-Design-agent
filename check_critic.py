"""Verify the configured vision critic before a real run depends on it.

Usage:
    python check_critic.py

Checks three things in order, because each one can pass while the next fails:
  1. Is a critic configured at all?
  2. Does the endpoint accept an image?
  3. Does the model actually SEE it -- or is it answering from the text alone?
"""
from __future__ import annotations

import sys

from agent.llm import ModelClient, build_critic_client
from agent.vision_probe import probe_vision
from config import load_settings


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings(require_model=False)

    if not settings.has_vision_critic:
        print("No vision critic configured.\n")
        print("Set these in .env, then run this again:")
        print("  CRITIC_BASE_URL   e.g. http://localhost:11434/v1")
        print("  CRITIC_API_KEY    e.g. ollama")
        print("  CRITIC_MODEL_NAME the vision model to use")
        print("\nWithout one, runs still work -- geometry checks stay on and")
        print("screenshot critique is skipped. No images are ever sent.")
        return 1

    base_url, api_key, model = settings.critic_settings()
    print(f"Critic model : {model}")
    print(f"Endpoint     : {base_url}")
    print("Sending a test image...\n")

    client = build_critic_client(settings)
    result = probe_vision(client)
    print(result)
    if result.reply:
        print(f"  model replied: {result.reply[:200]!r}")

    if result.ok:
        print("\nVision critique is ready. It will run on each finished section.")
        return 0

    print("\nLeave CRITIC_MODEL_NAME blank to disable critique entirely,")
    print("or try a different vision model. Runs still work either way.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
