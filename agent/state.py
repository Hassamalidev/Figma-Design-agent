"""Holds the plan, created node IDs, and per-step results.

Feeds the model concise summaries, never full history -- context is the
scarcest resource, especially on a local model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Screen:
    """One top-level Figma FRAME -- a single screen of the design.

    This is Figma's own model, which the agent used to ignore. A PAGE is a
    workspace; a FRAME is a screen. "Login, sign-up and a dashboard" is THREE
    sibling frames laid out side by side on ONE page -- not three sections
    stacked into a single tall frame, and not three Figma pages.

    Building everything into one frame is what produced designs where a
    sign-in form sat directly on top of a dashboard, and why a second run
    stamped new work over the first.
    """

    name: str
    frame_id: str | None = None
    width: int = 1440
    # Sections already inside THIS screen. Per-screen, because "the page
    # already has a header" is only true of the screen that has one.
    sections: list[str] = field(default_factory=list)
    is_existing: bool = False  # reused from a previous run rather than created

    def record_section(self, name: str) -> None:
        cleaned = (name or "").strip()
        if cleaned and cleaned not in self.sections:
            self.sections.append(cleaned)


@dataclass
class PlanStep:
    """One build step, and which screen it belongs to.

    The plan used to be a list of bare strings, so the only way to know what a
    step was for was to keyword-match its English (CLAUDE.md's known gap #3).
    A step now carries its target screen as data.
    """

    description: str
    screen_index: int = 0

    def __str__(self) -> str:
        return self.description


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
    # The screens this run produced, left to right -- one Figma frame each.
    screens: list[str] = field(default_factory=list)
    # Design-system adherence: contrast below AA, off-scale spacing, off-ramp
    # type, untokenised fills. Advisory by construction -- reported so the rules
    # are measurable, never used to fail a step (see agent/critic.py).
    design_notes: list[str] = field(default_factory=list)
    # What the INSTRUCTION asked for and what actually landed (agent/requirements.py).
    requirements_met: list[str] = field(default_factory=list)
    requirements_missing: list[str] = field(default_factory=list)
    # What the run cost: model calls, Figma round trips, retries, failure
    # reasons (agent/metrics.py). Filled in by `loop.run`, so it is present on
    # every result rather than only when someone remembered to ask for it.
    metrics: dict = field(default_factory=dict)
    # Per-step outcomes. The benchmark scores build quality from these (how
    # many steps failed, how many fell back to a placeholder, whether any
    # section was built twice) -- none of which is visible from node ids alone.
    step_results: list[StepResult] = field(default_factory=list)


@dataclass
class RunState:
    """Mutable state for a single run of the agent loop."""

    instruction: str
    enhanced_brief: str = ""
    plan: list[PlanStep] = field(default_factory=list)
    inspection_summary: str = ""
    existing_nodes: list[dict] = field(default_factory=list)  # what was on the page before this run
    # Every screen the instruction asks for, left to right. Created by the
    # harness (section 6a), never by the model.
    screens: list[Screen] = field(default_factory=list)
    root_frame_id: str | None = None  # the FIRST screen's frame; kept for whole-design fallbacks
    root_is_existing: bool = False  # True when continuing a previous run's design
    existing_sections: list[str] = field(default_factory=list)  # union across screens
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
    design_notes: list[str] = field(default_factory=list)  # advisory design-system findings
    requirements_met: list[str] = field(default_factory=list)
    requirements_missing: list[str] = field(default_factory=list)
    # True only when the instruction named several checkable things and NONE of
    # them reached the canvas -- the one coverage result confident enough to
    # fail a run over.
    satisfied_no_requirements: bool = False
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

    def screen_at(self, index: int) -> Screen | None:
        """The screen a step targets, or None if the plan outran the screens."""
        if 0 <= index < len(self.screens):
            return self.screens[index]
        return None

    def frame_for(self, index: int) -> str | None:
        """The frame id a step must build into."""
        screen = self.screen_at(index)
        return screen.frame_id if screen else self.root_frame_id

    def screen_names(self) -> list[str]:
        return [s.name for s in self.screens]

    def record_section(self, name: str, screen_index: int = 0) -> None:
        """Note a section as built, so later steps are told not to recreate it.

        This list used to be filled in once, only when reusing a root frame
        from an earlier run -- so during a fresh run it stayed empty and every
        step was told the page contained nothing. That is a direct cause of
        duplicated sections.

        Recorded against the SCREEN as well as globally: with several screens
        a shared list would tell the dashboard that the sign-in screen already
        has a header, because a different screen does.
        """
        cleaned = (name or "").strip()
        if not cleaned:
            return
        screen = self.screen_at(screen_index)
        if screen is not None:
            screen.record_section(cleaned)
        if cleaned not in self.existing_sections:
            self.existing_sections.append(cleaned)

    def mark_failed(self, step) -> None:
        self.failed_steps.append(str(step))

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

        It also fails a run that built a page matching NONE of a clearly
        specified instruction. Every gate before this one measured how well the
        nodes were built and none asked whether they were the nodes requested,
        so "10 sections, zero of them what you asked for" reported success.
        One missing requirement is a flaw, not a failure -- only zero-of-many
        is treated as the wrong design (see agent/requirements.py).
        """
        built_something = any(r.created_node_ids for r in self.step_results if r.ok)
        return not self.failed_steps and built_something and not self.satisfied_no_requirements

    def result(self) -> RunResult:
        return RunResult(
            instruction=self.instruction,
            success=self.succeeded(),
            created_node_ids=list(self.created_node_ids),
            failed_steps=list(self.failed_steps),
            warnings=list(self.warnings),
            final_screenshot_base64=self.final_screenshot_base64,
            layout_defects=list(self.layout_defects),
            screens=self.screen_names(),
            design_notes=list(self.design_notes),
            requirements_met=list(self.requirements_met),
            requirements_missing=list(self.requirements_missing),
            step_results=list(self.step_results),
        )
