"""CLI entry point.

Usage:
    python main.py "a mobile sign-in screen with email, password, and a Google button"
"""
from __future__ import annotations

import logging
import sys

from agent import loop
from agent.llm import ModelClient, build_critic_client
from bridge.server import Bridge
from config import load_settings


def _log_progress_to_stdout() -> None:
    """Prints agent.loop/agent.planner's step-by-step progress live -- without
    this, the CLI looks hung for however long each model call takes."""
    # Windows consoles often default to a legacy codepage (e.g. cp1252) that
    # can't encode everything a model writes -- narrow no-break spaces,
    # em-dashes, smart quotes. Reconfigure to UTF-8 so an unusual character
    # in the model's own output can't crash the run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("  %(message)s"))
    logger = logging.getLogger("agent")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python main.py "your instruction"')
        return 1
    instruction = sys.argv[1]

    _log_progress_to_stdout()
    settings = load_settings()
    llm = ModelClient(settings.model_base_url, settings.model_api_key, settings.model_name)

    bridge = Bridge(settings.bridge_host, settings.bridge_port)
    bridge.start()
    print(f"Bridge listening on ws://{settings.bridge_host}:{settings.bridge_port}")
    print("In Figma Desktop: Plugins -> Development -> run this plugin inside a design file.")

    if not bridge.wait_for_plugin(timeout=120):
        print("Timed out waiting for the Figma plugin to connect.")
        bridge.stop()
        return 1
    print("Plugin connected. Running...\n")

    try:
        result = loop.run(
            instruction, bridge, llm, settings.max_retries, settings.max_steps,
            critic_llm=build_critic_client(settings),
        )
    finally:
        bridge.stop()

    print(f"Instruction: {result.instruction}")
    print(f"Success: {result.success}")
    print(f"Created {len(result.created_node_ids)} node(s).")
    if result.failed_steps:
        print("Failed steps:")
        for step in result.failed_steps:
            print(f"  - {step}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
