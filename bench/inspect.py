"""Score and diagnose the design that is ALREADY open in Figma.

    python -m bench.inspect                  # find the page's root frame and report
    python -m bench.inspect --task dashboard # also check that task's acceptance criteria
    python -m bench.inspect --node 1:2       # a specific frame

No model is involved and nothing is built -- it reads the finished file and
runs the same deterministic checks the benchmark uses. This exists because
"the design looks bad" is not actionable: this turns a screenshot into a list
of named nodes with measured problems.
"""
from __future__ import annotations

import argparse
import logging

from agent import critic, loop
from bench import capture as capture_mod
from bench.score import format_score, score_requirements, score_task
from bench.spec import load_task
from tools.figma_exec import execute_figma_js

logger = logging.getLogger(__name__)

# A band of dead space taller than this reads as a hole in the page.
EMPTY_BAND_PX = 120


def find_root(bridge) -> tuple[str | None, list[dict]]:
    """The biggest auto-layout frame on the page, plus everything else up there."""
    result = execute_figma_js(bridge, loop.INSPECT_SCRIPT)
    if not result["ok"]:
        raise SystemExit(f"Could not read the page: {result['error']}")
    nodes = (result.get("result") or {}).get("topLevelNodes") or []
    frames = [n for n in nodes if n.get("layoutMode") in ("VERTICAL", "HORIZONTAL")]
    if not frames:
        return None, nodes
    biggest = max(frames, key=lambda n: (n.get("width") or 0) * (n.get("height") or 0))
    return biggest["id"], nodes


def report_orphans(nodes: list[dict], root_id: str) -> list[str]:
    """Top-level nodes sitting outside the root frame.

    These are the stray squares and loose text that appear beside a design:
    a script created them and never parented them, so they land on the page
    itself instead of inside a section.
    """
    return [
        f"{n.get('name', '?')} ({n.get('type')}) at ({n.get('x')},{n.get('y')}) "
        f"{n.get('width')}x{n.get('height')}"
        for n in nodes
        if n.get("id") != root_id
    ]


def report_sections(tree: dict) -> list[str]:
    """Each section's height against how much of it is actually filled.

    Vertical emptiness is the most common thing that makes a generated page
    look unfinished, and it is invisible to every other check: a 400px section
    holding one 40px label passes geometry cleanly.
    """
    lines = []
    for child in tree.get("children") or []:
        height = child.get("height") or 0
        kids = child.get("children") or []
        used = max(((k.get("y") or 0) + (k.get("height") or 0)) for k in kids) if kids else 0
        slack = height - used
        flag = "  <-- mostly empty" if kids and slack > EMPTY_BAND_PX else ""
        if not kids and height > EMPTY_BAND_PX:
            flag = "  <-- EMPTY, no children"
        lines.append(
            f"  {child.get('name', '?')[:34]:<34} {height:>5}px tall, "
            f"{len(kids):>2} children, {slack:>5}px unused{flag}"
        )
    return lines


def count_nodes(tree: dict) -> int:
    return len(capture_mod.walk(tree))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the design open in Figma")
    parser.add_argument("--task", help="also check this benchmark task's criteria")
    parser.add_argument("--node", help="frame id to inspect (default: the page's root frame)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from bridge.server import Bridge
    from config import load_settings

    settings = load_settings(require_model=False)
    bridge = Bridge(settings.bridge_host, settings.bridge_port)
    bridge.start()
    print("Waiting for the Figma plugin...")
    if not bridge.wait_for_plugin(timeout=120):
        bridge.stop()
        raise SystemExit("The plugin never connected.")

    try:
        root_id, page_nodes = (args.node, []) if args.node else find_root(bridge)
        if not root_id:
            raise SystemExit("No auto-layout frame found on this page.")

        tree = capture_mod.capture(bridge, root_id)
        if tree is None:
            raise SystemExit(f"Could not read {root_id}.")

        print(f"\n{'=' * 66}")
        print(f"{tree.get('name')} ({root_id})  {tree.get('width')}x{tree.get('height')}px, "
              f"{count_nodes(tree)} nodes")
        print("=" * 66)

        print("\nSECTIONS (top to bottom)")
        for line in report_sections(tree) or ["  (no children)"]:
            print(line)

        if page_nodes:
            orphans = report_orphans(page_nodes, root_id)
            print(f"\nLOOSE NODES OUTSIDE THE FRAME ({len(orphans)})")
            for line in orphans or ["  none"]:
                print(f"  {line}")

        defects = critic.find_layout_defects(tree)
        print(f"\nGEOMETRY DEFECTS ({len(defects)})")
        for defect in defects or []:
            print(f"  {defect}")
        if not defects:
            print("  none")

        nodes = capture_mod.walk(tree)
        filled = [n for n in nodes if n.get("hasSolidFill")]
        bound = [n for n in filled if n.get("fillBound")]
        texts = [n for n in nodes if n.get("type") == "TEXT"]
        print("\nDESIGN SYSTEM")
        print(f"  {len(bound)}/{len(filled)} fills token-backed")
        print(f"  {len(texts)} text nodes; sizes used: "
              f"{sorted({n.get('fontSize') for n in texts if n.get('fontSize')})}")

        if args.task:
            task = load_task(args.task)
            _, unmet = score_requirements(task, tree, nodes)
            print(f"\nREQUIREMENTS ({len(task.criteria) - len(unmet)}/{len(task.criteria)})")
            for label in unmet:
                print(f"  MISSING: {label}")
            # No run history here: this file was inspected, not built by us,
            # so figma_correctness is excluded rather than invented.
            print()
            print(format_score(score_task(task, tree, result=None)))
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
