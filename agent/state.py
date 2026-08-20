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
    height: int = 900  # starting height; the frame hugs its content afterwards
    # One line about what this screen is FOR, from the same answer that named
    # it. The per-screen plan is written from the whole design's brief, so
    # without this the plan for "Dashboard" was derived from a brief that
    # mostly talks about signing in.
    purpose: str = ""
    # desktop | tablet | mobile. Decides the frame width, so "a mobile
    # sign-in screen and a desktop dashboard" stops making both 1440 wide.
    device: str = "desktop"
    # One screenful, when the INSTRUCTION said what that is ("1440 x 900px").
    # Kept apart from `height`, which is only where the frame started: the
    # end-of-run fit pass rounds a frame up to a whole number of viewports, and
    # it used the DEVICE viewport (1024) even when the user had asked for 900 --
    # so a design specified as 1440x900 shipped as 1440x2048, with a screenful
    # of blank canvas under it and every full-height column stretched into it.
    viewport_height: int | None = None
    # Where the frame actually sits. Filled in by the harness when it places a
    # new screen, and read off the canvas when it adopts an existing one, so
    # the next screen is positioned against real coordinates.
    x: int | None = None
    y: int | None = None
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
    # None = work it out from the wording. Harness-authored steps set it
    # explicitly, because deciding whether a step may write raw JavaScript by
    # keyword-matching its English is exactly the guesswork this field removes.
    render_only: bool | None = None

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
    # True when the user stopped the run. Kept apart from `success` because
    # "you stopped it" and "it failed" are different things to be told.
    stopped: bool = False
    # The bridge died mid-run: the work is kept and reported, but nothing
    # was validated, so this is never a success.
    ended_early: bool = False
    final_screenshot_base64: str | None = None
    layout_defects: list[str] = field(default_factory=list)
    # The screens this run produced, left to right -- one Figma frame each.
    screens: list[str] = field(default_factory=list)
    # One rendered PNG per screen, so the dashboard can page through them
    # instead of showing every frame shrunk into a single wide strip.
    screen_shots: list[dict] = field(default_factory=list)
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
    # Prototype wiring: which element on which screen navigates where, as
    # readable lines. A static mockup has none; a design you can click through
    # has one per link (agent/interactions.py).
    interactions: list[str] = field(default_factory=list)
    # Per-step outcomes. The benchmark scores build quality from these (how
    # many steps failed, how many fell back to a placeholder, whether any
    # section was built twice) -- none of which is visible from node ids alone.
    step_results: list[StepResult] = field(default_factory=list)


@dataclass
class RunState:
    """Mutable state for a single run of the agent loop."""

    instruction: str
    # Text derived from the user's ATTACHMENTS (a screenshot read by a vision
    # model, a spec document). Kept apart from `instruction` on purpose: it
    # feeds the brief, the plan and the palette, but `agent/requirements.py`
    # must only ever grade the design against the user's OWN words, or the
    # agent would be setting its own homework (CLAUDE.md section 8d).
    references: str = ""
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
    # The user's attached pictures, uploaded into this Figma file once and then
    # painted onto as many nodes as the design wants (agent/assets.py). Empty
    # on a run with no attachments, which is what makes `kind: "image"` fall
    # back to a placeholder box instead of failing.
    assets: list = field(default_factory=list)
    # The raw attachments, held only until their images are uploaded into the
    # file. Everything after that reads `assets`, which is small.
    attachments: list = field(default_factory=list)
    # Wire the finished design up as a clickable prototype. A user preference,
    # on by default, because a design nobody can click through is half a design.
    prototype_enabled: bool = True
    # Prototype links that were actually wired, as readable lines
    # ("Login · 'Sign in' -> Dashboard"). See agent/interactions.py.
    interactions: list[str] = field(default_factory=list)
    font_styles: list[str] = field(default_factory=list)  # real Inter styles, read at runtime
    # The brand font the instruction asked for, IF this file really has it --
    # "Playfair Display" for an editorial bookstore, "" for a design that never
    # named one. Verified against Figma, never taken from the wording alone.
    display_family: str = ""
    # Ramp style name -> the (family, style) it really uses. Every string in
    # here came back from `listAvailableFontsAsync`, so nothing downstream has
    # to guess "SemiBold" vs "Semi Bold" (the two real spellings this project
    # has already lost a run to).
    text_fonts: dict = field(default_factory=dict)
    # None = not yet known. Set once, the first time we try a screenshot critique.
    model_sees_images: bool | None = None
    # A separate vision model for critique, or None to skip it entirely.
    critic_llm: object | None = None
    # Non-blocking visual observations, reported but never used to fail a step.
    minor_notes: list[str] = field(default_factory=list)
    layout_defects: list[str] = field(default_factory=list)  # from the final visual review
    screen_shots: list[dict] = field(default_factory=list)  # {name, image_base64} per screen
    design_notes: list[str] = field(default_factory=list)  # advisory design-system findings
    requirements_met: list[str] = field(default_factory=list)
    requirements_missing: list[str] = field(default_factory=list)
    # True only when the instruction named several checkable things and NONE of
    # them reached the canvas -- the one coverage result confident enough to
    # fail a run over.
    satisfied_no_requirements: bool = False
    visual_gate_enabled: bool = True  # user preference, from the dashboard
    # Asked at every checkpoint. A run is stopped COOPERATIVELY -- a model call
    # or a Figma round trip in flight cannot be interrupted, so the honest
    # promise is "no new work starts", not "everything halts this instant".
    should_stop: object | None = None
    stopped: bool = False  # the user asked for this run to end early
    # The bridge died mid-run: what was built is kept and reported, but
    # nothing was validated, so it is never reported as a success.
    ended_early: bool = False
    created_node_ids: list[str] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    failed_steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_screenshot_base64: str | None = None

    def stop_requested(self) -> bool:
        """Has the user asked this run to stop? Never raises -- a broken
        callback must not be able to take the run down."""
        if self.should_stop is None:
            return False
        try:
            return bool(self.should_stop())
        except Exception:
            return False

    def forget_node_ids(self, node_ids: list[str]) -> None:
        """Drop nodes that were removed from the canvas.

        The binding audit and the final review both walk `created_node_ids`, so
        a deleted node left in here means the run reports on, and tries to
        rebind, something that no longer exists.
        """
        gone = {node_id for node_id in node_ids if node_id}
        if not gone:
            return
        self.created_node_ids = [i for i in self.created_node_ids if i not in gone]

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

    def screen_map(self) -> dict[str, str]:
        """Screen name -> frame id, for wiring `"on_click": "Dashboard"`.

        Every screen frame exists before the first step runs (the harness
        creates them all up front), so a section can be wired to a screen that
        has not been designed yet -- which is exactly what a prototype link
        from the first screen to the last one needs.
        """
        return {s.name: s.frame_id for s in self.screens if s.frame_id}

    def sections_elsewhere(self, screen_index: int) -> list[str]:
        """Sections already built on the OTHER screens, most recent last.

        Every screen is planned and built in isolation, so the header on the
        shop page had no idea what the header on the home page looked like and
        the two came out different -- which is what makes a multi-screen file
        read as several designs rather than one. Now that screens are built
        round-robin, the first screen's chrome exists by the time the second
        screen's first step runs, so this is real information rather than an
        empty list.

        `TODO` placeholders are left out: a gap marker is not a design
        decision, and telling a screen to match one would spread it.
        """
        seen: list[str] = []
        for index, screen in enumerate(self.screens):
            if index == screen_index:
                continue
            for name in screen.sections:
                if not name.startswith("TODO") and name not in seen:
                    seen.append(name)
        return seen

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

    def completed_step_descriptions(self) -> set[str]:
        """Steps that finished, so a run cut short can name only what it owes."""
        return {r.step_description for r in self.step_results if r.ok}

    def mark_failed(self, step) -> None:
        description = str(step)
        if description not in self.failed_steps:
            self.failed_steps.append(description)

    def recent_summary(self, n: int = 5) -> str:
        """A short, model-facing digest of the last n steps -- never the full transcript."""
        lines = []
        for r in self.step_results[-n:]:
            status = "ok" if r.ok else "FAILED"
            lines.append(f"- [{status}] {r.step_description}: {r.summary}")
        return "\n".join(lines) if lines else "(no steps completed yet)"

    def design_source(self) -> str:
        """Everything describing the design to build: the instruction and the
        attachments. This is what the brief and the PALETTE are read from --
        a screenshot's colours are facts about a real image, and the best
        source available. It is deliberately NOT what requirements are read
        from; see `references`.
        """
        if not self.references:
            return self.instruction
        return self.instruction + "\n\n" + self.references

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
        if self.stopped:
            return False  # a design that was cut short is not a finished one
        if self.ended_early:
            # The plugin went away mid-run. The work already on the canvas is
            # real and is reported, but a run that never reached its own final
            # validation has not been checked and must not claim a green tick.
            return False
        built_something = any(r.created_node_ids for r in self.step_results if r.ok)
        return not self.failed_steps and built_something and not self.satisfied_no_requirements

    def result(self) -> RunResult:
        return RunResult(
            instruction=self.instruction,
            success=self.succeeded(),
            ended_early=self.ended_early,
            stopped=self.stopped,
            created_node_ids=list(self.created_node_ids),
            failed_steps=list(self.failed_steps),
            warnings=list(self.warnings),
            final_screenshot_base64=self.final_screenshot_base64,
            layout_defects=list(self.layout_defects),
            screens=self.screen_names(),
            screen_shots=list(self.screen_shots),
            design_notes=list(self.design_notes),
            requirements_met=list(self.requirements_met),
            requirements_missing=list(self.requirements_missing),
            step_results=list(self.step_results),
            interactions=list(self.interactions),
        )
