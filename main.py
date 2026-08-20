"""CLI entry point.

Usage:
    python main.py "a mobile sign-in screen with email, password, and a Google button"
    python main.py --edit "make every primary button purple and shorten the heading"
    python main.py --attach ref.png "rebuild this as a Figma design"

`--edit` changes the design that is already in the file instead of building a
new one. It honours whatever you have selected in Figma: select a card, ask for
a change, and the change applies there.

`--attach` takes a screenshot or a spec document (repeatable) and builds from
it. An image needs a vision model configured (CRITIC_MODEL_NAME is enough); if
none is, the run says so rather than quietly ignoring the file.
"""
from __future__ import annotations

import base64
import logging
import sys
from pathlib import Path

from agent import edit_loop, loop, reference
from agent.llm import ModelClient, build_critic_client, build_vision_client
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


def _load_attachments(paths: list[str]) -> list[reference.Attachment]:
    """Read the files named with --attach, checking the same limits the
    dashboard does. One code path for the rules, two ways in."""
    items = []
    for path in paths:
        data = Path(path).read_bytes()
        items.append({"name": Path(path).name, "data_base64": base64.b64encode(data).decode()})
    return reference.from_payload(items)


def main() -> int:
    args = sys.argv[1:]
    editing = False
    paths: list[str] = []
    while args:
        if args[0] in ("--edit", "-e"):
            editing, args = True, args[1:]
        elif args[0] in ("--attach", "-a") and len(args) > 1:
            paths.append(args[1])
            args = args[2:]
        else:
            break
    if not args:
        print('Usage: python main.py "your instruction"')
        print('       python main.py --edit "what to change about the existing design"')
        print('       python main.py --attach screenshot.png "rebuild this"')
        return 1
    instruction = args[0]

    try:
        attachments = _load_attachments(paths)
    except (OSError, reference.ReferenceError) as exc:
        print(f"Attachment problem: {exc}")
        return 1

    _log_progress_to_stdout()
    settings = load_settings()
    llm = ModelClient(settings.model_base_url, settings.model_api_key, settings.model_name)

    bridge = Bridge(settings.bridge_host, settings.bridge_port)
    bridge.start()
    print(f"Bridge listening on ws://{settings.bridge_host}:{settings.bridge_port}")
    print("In Figma Desktop: Plugins -> Development -> run this plugin inside a design file.")
    if editing:
        print("Edit mode: select the nodes you want changed, or leave nothing selected "
              "to let the agent find them.")

    if not bridge.wait_for_plugin(timeout=120):
        print("Timed out waiting for the Figma plugin to connect.")
        bridge.stop()
        return 1
    print("Plugin connected. Running...\n")

    references = ""
    if attachments:
        try:
            for note in reference.check_readable(attachments, settings.has_vision):
                print(f"  ! {note}")
        except reference.ReferenceError as exc:
            print(f"Attachment problem: {exc}")
            bridge.stop()
            return 1
        print(f"Reading {len(attachments)} attachment(s)...")
        references, attach_warnings = reference.describe(
            attachments, build_vision_client(settings)
        )
        for warning in attach_warnings:
            print(f"  ! {warning}")

    try:
        if editing:
            result = edit_loop.run(
                instruction, bridge, llm, settings.max_retries, references=references,
                attachments=attachments,
            )
        else:
            result = loop.run(
                instruction, bridge, llm, settings.max_retries, settings.max_steps,
                critic_llm=build_critic_client(settings),
                references=references,
                # The attachments travel BOTH ways: read as words above, and
                # placed on the canvas as real images (agent/assets.py).
                attachments=attachments,
            )
    finally:
        bridge.stop()

    print(f"Instruction: {result.instruction}")
    print(f"Success: {result.success}")
    print(f"{'Changed' if editing else 'Created'} {len(result.created_node_ids)} node(s).")

    # Whether the design contains what was ASKED for -- the CLI reported how the
    # build went and never this, so a run that matched none of the instruction
    # printed "Success: True" and nothing else.
    if result.requirements_met or result.requirements_missing:
        total = len(result.requirements_met) + len(result.requirements_missing)
        print(f"Requirements: {len(result.requirements_met)}/{total} met.")
        for label in result.requirements_missing:
            print(f"  - MISSING: {label}")
    if result.design_notes:
        print(f"Design-system notes ({len(result.design_notes)}):")
        for note in result.design_notes[:5]:
            print(f"  - {note}")
    if result.metrics:
        print(f"Cost: {result.metrics.get('elapsed_seconds')}s, "
              f"{result.metrics['model']['count']} model call(s), "
              f"{result.metrics['bridge_calls']} Figma call(s), "
              f"x{result.metrics['retry_rate']} attempts per step.")

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
