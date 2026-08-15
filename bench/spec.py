"""Benchmark task definitions: a frozen instruction plus checkable criteria.

A task is data, not code, so the set can grow without touching the scorer. The
instruction string is FROZEN -- rewording it invalidates every earlier result,
which is the whole point of having a benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class Criterion:
    """One checkable statement about the finished design.

    Deliberately a small set of predicate kinds. These are a *proxy* for
    requirement adherence -- "a node's text matches /password/i" is evidence a
    password field exists, not proof it is well built. Keep that honest when
    reading a score.
    """

    label: str
    any_text: str = ""           # regex: some TEXT node's characters match
    any_name: str = ""           # regex: some node's name matches
    min_nodes: dict = field(default_factory=dict)   # {"type": "TEXT", "n": 6}
    min_sections: int = 0        # direct children of the root frame

    def kind(self) -> str:
        if self.any_text:
            return "any_text"
        if self.any_name:
            return "any_name"
        if self.min_nodes:
            return "min_nodes"
        if self.min_sections:
            return "min_sections"
        return "unknown"


@dataclass
class Task:
    task_id: str
    instruction: str          # FROZEN -- changing it invalidates past results
    page_type: str
    viewport: int
    criteria: list[Criterion]
    notes: str = ""


def _criterion(raw: dict) -> Criterion:
    return Criterion(
        label=raw["label"],
        any_text=raw.get("any_text", ""),
        any_name=raw.get("any_name", ""),
        min_nodes=raw.get("min_nodes", {}) or {},
        min_sections=int(raw.get("min_sections", 0) or 0),
    )


def load_task(task_id: str) -> Task:
    """Load one task by id, failing loudly if it isn't there."""
    path = TASKS_DIR / f"{task_id}.json"
    if not path.exists():
        available = ", ".join(t.task_id for t in load_all_tasks()) or "(none)"
        raise FileNotFoundError(f"No benchmark task '{task_id}'. Available: {available}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Task(
        task_id=raw["task_id"],
        instruction=raw["instruction"],
        page_type=raw.get("page_type", ""),
        viewport=int(raw.get("viewport", 1440)),
        criteria=[_criterion(c) for c in raw.get("criteria", [])],
        notes=raw.get("notes", ""),
    )


def load_all_tasks() -> list[Task]:
    """Every task on disk, in stable id order so runs are comparable."""
    return [load_task(p.stem) for p in sorted(TASKS_DIR.glob("*.json"))]
