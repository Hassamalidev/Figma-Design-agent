"""Holds the plan, created node IDs, and per-step results.

Feeds the model concise summaries, never full history -- context is the
scarcest resource, especially on a local model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    step_description: str
    ok: bool
    created_node_ids: list[str] = field(default_factory=list)
    summary: str = ""  # short, model-facing description of what happened
    section_name: str = ""  # real Figma name of the node this step created, if known


@dataclass
class RunResult:
    instruction: str
    success: bool
    created_node_ids: list[str]
    failed_steps: list[str]
    warnings: list[str]
    final_screenshot_base64: str | None = None
    layout_defects: list[str] = field(default_factory=list)
    # Per-step outcomes. The benchmark scores build quality from these (how
    # many steps failed, how many fell back to a placeholder, whether any
    # section was built twice) -- none of which is visible from node ids alone.
    step_results: list[StepResult] = field(default_factory=list)


@dataclass
class RunState:
    """Mutable state for a single run of the agent loop."""

    instruction: str
    enhanced_brief: str = ""
    plan: list[str] = field(default_factory=list)
    inspection_summary: str = ""
    existing_nodes: list[dict] = field(default_factory=list)  # what was on the page before this run
    root_frame_id: str | None = None  # created (or reused) by the harness, not the model
    root_is_existing: bool = False  # True when continuing a previous run's design
    existing_sections: list[str] = field(default_factory=list)  # already built inside the root
    palette: list[tuple[str, str]] = field(default_factory=list)  # (name, hex) the harness created
    # (token_name, hex, role) -- roles derived from luminance, not guessed.
    palette_info: list[tuple[str, str, str]] = field(default_factory=list)
    readable_pairings: list[str] = field(default_factory=list)  # measured WCAG AA pairs
    bound_node_count: int = 0  # nodes auto-rebound from a hardcoded colour to a token
    token_names: list[str] = field(default_factory=list)  # paint styles the harness created
    text_style_names: list[str] = field(default_factory=list)
    font_styles: list[str] = field(default_factory=list)  # real Inter styles, read at runtime
    # None = not yet known. Set once, the first time we try a screenshot critique.
    model_sees_images: bool | None = None
    # A separate vision model for critique, or None to skip it entirely.
    critic_llm: object | None = None
    # Non-blocking visual observations, reported but never used to fail a step.
    minor_notes: list[str] = field(default_factory=list)
    layout_defects: list[str] = field(default_factory=list)  # from the final visual review
    visual_gate_enabled: bool = True  # user preference, from the dashboard
    created_node_ids: list[str] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_screenshot_base64: str | None = None

    def add_node_ids(self, node_ids: list[str]) -> None:
        """Track new nodes, never duplicating.

        A correcting retry legitimately returns ids we already have (it edits
        the same nodes), which used to double-count them -- the binding audit
        reported the same node twice.
        """
        known = set(self.created_node_ids)
        for node_id in node_ids:
            if node_id and node_id not in known:
                known.add(node_id)
                self.created_node_ids.append(node_id)

    def record_step_result(self, step: str, result: StepResult) -> None:
        self.step_results.append(result)
        self.add_node_ids(result.created_node_ids)

    def record_section(self, name: str) -> None:
        """Note a section as built, so later steps are told not to recreate it.

        This list used to be filled in once, only when reusing a root frame
        from an earlier run -- so during a fresh run it stayed empty and every
        step was told the page contained nothing. That is a direct cause of
        duplicated sections.
        """
        cleaned = (name or "").strip()
        if cleaned and cleaned not in self.existing_sections:
            self.existing_sections.append(cleaned)

    def mark_failed(self, step: str) -> None:
        self.failed_steps.append(step)

    def recent_summary(self, n: int = 5) -> str:
        """A short, model-facing digest of the last n steps -- never the full transcript."""
        lines = []
        for r in self.step_results[-n:]:
            status = "ok" if r.ok else "FAILED"
            lines.append(f"- [{status}] {r.step_description}: {r.summary}")
        return "\n".join(lines) if lines else "(no steps completed yet)"

    def built_section_count(self) -> int:
        """Sections that are real work, not `TODO` placeholders."""
        return len([s for s in self.existing_sections if not s.startswith("TODO")])

    def succeeded(self) -> bool:
        """A run only succeeded if a step actually PUT SOMETHING on the canvas.

        `not failed_steps` alone reported "Success" on runs that created zero
        nodes -- an empty plan, or steps that each ended without touching the
        canvas. A green tick on an empty page destroys trust in every other
        number the tool reports.

        Counted from the steps, not from `created_node_ids`, because the latter
        includes the harness's own root frame: a run that created nothing but
        the empty frame it built to hold the design has not succeeded.
        """
        built_something = any(r.created_node_ids for r in self.step_results if r.ok)
        return not self.failed_steps and built_something

    def result(self) -> RunResult:
        return RunResult(
            instruction=self.instruction,
            success=self.succeeded(),
            created_node_ids=list(self.created_node_ids),
            failed_steps=list(self.failed_steps),
            warnings=list(self.warnings),
            final_screenshot_base64=self.final_screenshot_base64,
            layout_defects=list(self.layout_defects),
            step_results=list(self.step_results),
        )
