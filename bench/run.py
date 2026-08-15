"""Run benchmark tasks against the real agent, or re-score a saved capture.

    python -m bench.run --list
    python -m bench.run login dashboard        # needs Figma + a model
    python -m bench.run --all --repeat 3       # variance matters; see below
    python -m bench.run --rescore bench/results/2026-08-15T10-00-00_login.json

Every run saves its capture alongside its score, so re-scoring after a scorer
change never needs Figma again -- and an old result can always be recomputed
under a new rubric.

**Run each configuration at least 3 times.** Single-run variance on a small
model is wide enough to swallow most real improvements, so one run of A beating
one run of B tells you nothing.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

from bench import capture as capture_mod
from bench.score import Score, format_score, score_task
from bench.spec import Task, load_all_tasks, load_task

RESULTS_DIR = Path(__file__).parent / "results"

logger = logging.getLogger(__name__)


def run_task(task: Task, bridge, llm, max_retries: int, max_steps: int, critic_llm=None) -> dict:
    """Build one task on the real canvas, capture it, and score it."""
    from agent import loop

    logger.info("=== %s ===", task.task_id)
    result = loop.run(task.instruction, bridge, llm, max_retries, max_steps,
                      critic_llm=critic_llm)

    tree = capture_mod.capture(bridge, result.created_node_ids[0]) if result.created_node_ids else None
    if tree is None:
        logger.warning("%s: could not capture the design; scoring skipped.", task.task_id)
        return {"task_id": task.task_id, "error": "capture failed"}

    score = score_task(task, tree, result)
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "score": score.as_dict(),
        "tree": tree,
        "run": {
            "success": result.success,
            "failed_steps": result.failed_steps,
            "warnings": result.warnings,
            "layout_defects": result.layout_defects,
            "node_count": len(result.created_node_ids),
            # Kept so a saved result can be re-scored without Figma.
            "steps": [
                {"step": s.step_description, "ok": s.ok,
                 "summary": s.summary, "section_name": s.section_name}
                for s in result.step_results
            ],
        },
    }


def save(payload: dict, label: str) -> Path:
    """Persist a result. The timestamp is the run identity -- never overwrite."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = RESULTS_DIR / f"{stamp}_{label}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def result_from_payload(payload: dict):
    """Rebuild the RunResult a saved result was scored from.

    Scoring must be reproducible offline: a scorer change should be testable
    against every run already recorded, without rebuilding anything in Figma.
    """
    from agent.state import RunResult, StepResult

    run = payload.get("run") or {}
    return RunResult(
        instruction=payload.get("instruction", ""),
        success=run.get("success", False),
        created_node_ids=[],
        failed_steps=run.get("failed_steps", []),
        warnings=run.get("warnings", []),
        layout_defects=run.get("layout_defects", []),
        step_results=[
            StepResult(
                step_description=s.get("step", ""),
                ok=s.get("ok", False),
                summary=s.get("summary", ""),
                section_name=s.get("section_name", ""),
            )
            for s in run.get("steps", [])
        ],
    )


def rescore(path: Path) -> Score:
    """Recompute a saved capture under the current rubric, without Figma."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = load_task(payload["task_id"])
    return score_task(task, payload["tree"], result_from_payload(payload))


def summarise(scores: list[Score]) -> str:
    """Mean and spread per task -- the only fair way to compare two configs."""
    by_task: dict[str, list[float]] = {}
    for score in scores:
        by_task.setdefault(score.task_id, []).append(score.total)

    lines = ["", "=" * 62, "SUMMARY (mean over runs; spread matters as much as the mean)"]
    for task_id, totals in sorted(by_task.items()):
        mean = statistics.mean(totals)
        spread = f"{min(totals)}-{max(totals)}" if len(totals) > 1 else "single run"
        lines.append(f"  {task_id:<12} {mean:5.1f}/100   n={len(totals)}  range {spread}")
    if by_task:
        overall = statistics.mean([t for totals in by_task.values() for t in totals])
        lines.append(f"  {'OVERALL':<12} {overall:5.1f}/100")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Figma agent design benchmark")
    parser.add_argument("tasks", nargs="*", help="task ids to run")
    parser.add_argument("--list", action="store_true", help="list available tasks")
    parser.add_argument("--all", action="store_true", help="run every task")
    parser.add_argument("--repeat", type=int, default=1, help="runs per task (use >=3)")
    parser.add_argument("--rescore", help="re-score a saved result file, no Figma needed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.list:
        for task in load_all_tasks():
            print(f"  {task.task_id:<12} {task.page_type:<28} {len(task.criteria)} criteria")
        return

    if args.rescore:
        print(format_score(rescore(Path(args.rescore))))
        return

    task_ids = [t.task_id for t in load_all_tasks()] if args.all else args.tasks
    if not task_ids:
        parser.error("name at least one task, or pass --all / --list")
    if args.repeat < 3:
        logger.warning(
            "Running n=%d. Variance on a small model is wide -- use --repeat 3 "
            "or more before comparing two configurations.",
            args.repeat,
        )

    # Imported late so --list and --rescore work with no Figma and no model.
    from config import load_settings
    from bridge.server import Bridge
    from agent.llm import ModelClient, build_critic_client

    settings = load_settings()
    llm = ModelClient(settings.model_base_url, settings.model_api_key, settings.model_name)
    critic_llm = build_critic_client(settings)
    if critic_llm:
        logger.info("Vision critic: %s", settings.critic_model_name)
    bridge = Bridge(settings.bridge_host, settings.bridge_port)
    bridge.start()
    logger.info("Bridge on ws://%s:%d -- run the plugin in a DESIGN file.",
                settings.bridge_host, settings.bridge_port)
    if not bridge.wait_for_plugin(timeout=120):
        bridge.stop()
        raise SystemExit("Timed out waiting for the Figma plugin to connect.")

    scores: list[Score] = []
    failures: list[str] = []
    try:
        for task_id in task_ids:
            task = load_task(task_id)
            for attempt in range(args.repeat):
                run_label = f"{task_id} run {attempt + 1}/{args.repeat}"
                try:
                    payload = run_task(task, bridge, llm, settings.max_retries,
                                       settings.max_steps, critic_llm)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    # One dead run must not discard the results already earned.
                    # A 500 from the model endpoint used to end the whole sweep.
                    logger.warning("%s FAILED: %s: %s", run_label, type(exc).__name__, exc)
                    failures.append(f"{run_label}: {type(exc).__name__}: {exc}")
                    continue
                if "error" in payload:
                    failures.append(f"{run_label}: {payload['error']}")
                    continue
                path = save(payload, f"{task_id}_{attempt + 1}")
                score = score_task(task, payload["tree"], result_from_payload(payload))
                scores.append(score)
                print(format_score(score))
                print(f"  saved -> {path}")
    finally:
        bridge.stop()

    print(summarise(scores))
    if failures:
        # Never let a partial sweep read as a complete one.
        print(f"\n{len(failures)} run(s) did not complete:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
