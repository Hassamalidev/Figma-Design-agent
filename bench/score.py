"""Deterministic scoring of a finished design.

Five of the six dimensions are computed from the node tree with no model
involved, so they cannot drift, cannot flatter a run, and cost nothing. The
sixth (visual quality) needs a vision judge and is simply ABSENT when no judge
ran -- absent, not zero. A dimension you did not measure must never be scored,
or every architecture change looks like an improvement the moment you enable
the judge.

Read the numbers with their limits in mind: `requirements` checks that text
matching /password/i exists somewhere, which is evidence a password field was
built, not proof it was built well.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from agent import critic, scaffold
from bench.capture import walk
from bench.spec import Criterion, Task

# The scale and ramp the harness builds to, defined once in agent/critic.py so
# the benchmark cannot drift from the gate that runs during a build. Padding
# and gaps off this scale are what makes a generated page feel arrhythmic even
# when nothing overlaps.
SPACING_SCALE = critic.SPACING_SCALE
TYPE_RAMP = critic.TYPE_RAMP

# Above this many geometry defects a page reads as broken rather than blemished.
LAYOUT_DEFECT_FLOOR = 8

WEIGHTS = {
    "requirements": 0.30,
    "figma_correctness": 0.15,
    "layout": 0.15,
    "design_system": 0.15,
    "typography": 0.10,
    "visual": 0.15,
}


@dataclass
class Dimension:
    name: str
    weight: float
    score: float | None  # 0..1, or None when not measured
    detail: str


@dataclass
class Score:
    task_id: str
    total: float                 # 0..100, renormalised over measured dimensions
    dimensions: list[Dimension]
    unmet: list[str] = field(default_factory=list)
    measured_weight: float = 0.0

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "total": self.total,
            "measured_weight": self.measured_weight,
            "unmet": self.unmet,
            "dimensions": [asdict(d) for d in self.dimensions],
        }


# ---- individual criteria --------------------------------------------------


def _text_nodes(nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if n.get("type") == "TEXT"]


def criterion_met(criterion: Criterion, tree: dict, nodes: list[dict]) -> bool:
    """Check one acceptance statement against the captured tree."""
    if criterion.any_text:
        pattern = re.compile(criterion.any_text)
        return any(pattern.search(n.get("characters") or "") for n in _text_nodes(nodes))

    if criterion.any_name:
        pattern = re.compile(criterion.any_name)
        return any(pattern.search(n.get("name") or "") for n in nodes)

    if criterion.min_nodes:
        wanted_type = criterion.min_nodes.get("type")
        needed = int(criterion.min_nodes.get("n", 1))
        matching = [n for n in nodes if not wanted_type or n.get("type") == wanted_type]
        return len(matching) >= needed

    if criterion.min_sections:
        return len(tree.get("children") or []) >= criterion.min_sections

    return False


# ---- dimensions -----------------------------------------------------------


def score_requirements(task: Task, tree: dict, nodes: list[dict]) -> tuple[Dimension, list[str]]:
    """The share of the instruction that demonstrably made it onto the canvas."""
    if not task.criteria:
        return Dimension("requirements", WEIGHTS["requirements"], None, "no criteria"), []
    unmet = [c.label for c in task.criteria if not criterion_met(c, tree, nodes)]
    met = len(task.criteria) - len(unmet)
    ratio = met / len(task.criteria)
    return (
        Dimension(
            "requirements",
            WEIGHTS["requirements"],
            ratio,
            f"{met}/{len(task.criteria)} criteria met",
        ),
        unmet,
    )


def score_figma_correctness(result) -> Dimension:
    """Did the build itself go cleanly -- or limp to the end on placeholders?

    Needs the run's own history. Scoring a file we did not watch being built
    would report a perfect 100% from no evidence at all -- which is exactly the
    flattery `measured_weight` exists to prevent.
    """
    if result is None or not result.step_results:
        return Dimension(
            "figma_correctness",
            WEIGHTS["figma_correctness"],
            None,
            "not measured (no run history -- this file was inspected, not built here)",
        )
    steps = len(result.step_results) or 1
    failed = len(result.failed_steps)
    placeholders = sum(1 for s in result.step_results if "placeholder" in (s.summary or ""))
    duplicates = _duplicate_section_count(result)

    penalty = (failed / steps) + (placeholders / steps) * 0.5 + min(duplicates, 5) * 0.1
    value = max(0.0, 1.0 - penalty)
    return Dimension(
        "figma_correctness",
        WEIGHTS["figma_correctness"],
        value,
        f"{failed}/{steps} steps failed, {placeholders} placeholder(s), {duplicates} duplicate section(s)",
    )


def _duplicate_section_count(result) -> int:
    """Sections built more than once -- the signature of a regenerating retry."""
    seen: dict[str, int] = {}
    for step in result.step_results:
        name = (step.section_name or "").strip().lower()
        if name:
            seen[name] = seen.get(name, 0) + 1
    return sum(count - 1 for count in seen.values() if count > 1)


def score_layout(tree: dict, nodes: list[dict]) -> Dimension:
    """Geometry defects plus how much of the spacing sits on the scale."""
    defects = critic.find_layout_defects(tree)
    geometry = max(0.0, 1.0 - len(defects) / LAYOUT_DEFECT_FLOOR)

    values: list[int] = []
    for node in nodes:
        if node.get("layoutMode") in ("VERTICAL", "HORIZONTAL"):
            if isinstance(node.get("itemSpacing"), int):
                values.append(node["itemSpacing"])
            values.extend(v for v in (node.get("padding") or []) if isinstance(v, int))

    on_scale = sum(1 for v in values if v in SPACING_SCALE)
    rhythm = on_scale / len(values) if values else None

    value = geometry if rhythm is None else (geometry * 0.6 + rhythm * 0.4)
    detail = f"{len(defects)} geometry defect(s)"
    if rhythm is not None:
        detail += f", {on_scale}/{len(values)} spacing values on scale"
    return Dimension("layout", WEIGHTS["layout"], value, detail)


def score_design_system(nodes: list[dict]) -> Dimension:
    """The share of filled nodes whose colour is token-backed rather than ad hoc."""
    filled = [n for n in nodes if n.get("hasSolidFill")]
    if not filled:
        return Dimension("design_system", WEIGHTS["design_system"], None, "no filled nodes")
    bound = sum(1 for n in filled if n.get("fillBound"))
    return Dimension(
        "design_system",
        WEIGHTS["design_system"],
        bound / len(filled),
        f"{bound}/{len(filled)} fills token-backed",
    )


def score_typography(nodes: list[dict]) -> Dimension:
    """Ramp adherence, penalised by text that cannot actually be read."""
    texts = _text_nodes(nodes)
    if not texts:
        return Dimension("typography", WEIGHTS["typography"], None, "no text nodes")

    sized = [n for n in texts if isinstance(n.get("fontSize"), (int, float))]
    on_ramp = sum(1 for n in sized if n["fontSize"] in TYPE_RAMP)
    ramp = on_ramp / len(sized) if sized else 0.0

    broken = sum(
        1
        for n in texts
        if (n.get("width") or 0) < 8
        or not (n.get("characters") or "").strip()
        or (isinstance(n.get("fontSize"), (int, float)) and (n.get("height") or 0) + 1 < n["fontSize"])
    )
    readable = 1.0 - (broken / len(texts))

    return Dimension(
        "typography",
        WEIGHTS["typography"],
        ramp * 0.5 + readable * 0.5,
        f"{on_ramp}/{len(sized)} sizes on the ramp, {broken}/{len(texts)} unreadable",
    )


# ---- assembly -------------------------------------------------------------


def score_task(task: Task, tree: dict, result=None, visual: float | None = None) -> Score:
    """Score one finished design.

    `result` is the run that built it, or None when inspecting a file we did
    not build. `visual` is the vision judge's 0..1, or None. Either being
    absent removes that dimension from the total rather than zeroing it.
    """
    nodes = walk(tree)

    requirements, unmet = score_requirements(task, tree, nodes)
    dimensions = [
        requirements,
        score_figma_correctness(result),
        score_layout(tree, nodes),
        score_design_system(nodes),
        score_typography(nodes),
        Dimension(
            "visual",
            WEIGHTS["visual"],
            visual,
            "vision judge" if visual is not None else "not measured (no vision model)",
        ),
    ]

    # Renormalise over what was actually measured, so an unmeasured dimension
    # is excluded rather than silently counted as a zero.
    measured = [d for d in dimensions if d.score is not None]
    weight = sum(d.weight for d in measured)
    total = (sum(d.score * d.weight for d in measured) / weight * 100) if weight else 0.0

    return Score(
        task_id=task.task_id,
        total=round(total, 1),
        dimensions=dimensions,
        unmet=unmet,
        measured_weight=round(weight, 2),
    )


def format_score(score: Score) -> str:
    """A compact human-readable report -- what you actually read after a run."""
    lines = [f"{score.task_id}: {score.total}/100  (measured weight {score.measured_weight})"]
    for dim in score.dimensions:
        shown = "  n/a" if dim.score is None else f"{dim.score * 100:5.1f}"
        lines.append(f"  {dim.name:<18} {shown}%  x{dim.weight:<5} {dim.detail}")
    if score.unmet:
        lines.append("  unmet criteria:")
        lines.extend(f"    - {label}" for label in score.unmet)
    return "\n".join(lines)
