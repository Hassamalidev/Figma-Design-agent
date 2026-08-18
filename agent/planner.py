"""Turns the instruction into an ordered plan: tokens, then components, then composition."""
from __future__ import annotations

import json
import logging
import re

from agent.llm import ModelClient
from agent.prompts import (
    ENHANCE_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    SCREENS_SYSTEM_PROMPT,
    enhance_user_message,
    planning_user_message,
    screens_user_message,
)
from agent.state import RunState, Screen

logger = logging.getLogger(__name__)

# Used only if the model's response can't be parsed as JSON -- keeps the loop
# moving instead of crashing the whole run on a formatting slip.
_FALLBACK_STEP = "Build the requested design as a single composition step."


def enhance_instruction(instruction: str, llm: ModelClient) -> str:
    """Expand a short instruction into a detailed creative brief before planning.

    Keeps the user's explicit requirements intact and fills in everything
    they left unspecified with concrete design judgment (palette, layout,
    type) -- the goal is a plan built from a real design brief, not a raw
    one-liner. Falls back to the raw instruction if the model returns nothing
    usable, so a bad enhancement never blocks the run.
    """
    logger.info("Writing a design brief for: %s", instruction)
    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM_PROMPT},
        {"role": "user", "content": enhance_user_message(instruction)},
    ]
    message = llm.complete(messages, tools=None)
    brief = (message.content or "").strip()
    if not brief:
        logger.info("Brief came back empty -- building from the raw instruction instead.")
        return instruction
    logger.info("Design brief:\n%s", brief)
    return brief


# A design with more screens than this is a project, not a request -- and each
# screen costs its own planning call plus its own build steps.
MAX_SCREENS = 6

# Screen names the model returns that are really SECTIONS. Returning one of
# these means it misread the question, and building it as a separate top-level
# frame would scatter one screen's parts across the canvas.
_SECTION_WORDS = {
    "header", "footer", "nav", "navbar", "navigation", "hero", "sidebar",
    "banner", "cta", "form", "card", "section", "content", "body", "main",
    "features", "testimonials", "pricing table", "menu",
}


def plan_screens(instruction: str, brief: str, llm: ModelClient) -> list[Screen]:
    """Decide which SCREENS the request needs -- one Figma frame each.

    Asked as its own small question rather than inferred from the plan. "How
    many screens is this?" is a question even a small model answers reliably,
    while recovering the same structure by keyword-matching English plan steps
    is exactly the guesswork CLAUDE.md lists as a known weakness.

    Always returns at least one screen, so a failed or nonsense answer simply
    behaves like the single-screen design that most requests are.
    """
    messages = [
        {"role": "system", "content": SCREENS_SYSTEM_PROMPT},
        {"role": "user", "content": screens_user_message(instruction, brief)},
    ]
    try:
        message = llm.complete(messages, tools=None)
        names = _parse_steps(message.content or "")
    except Exception as exc:  # a screen list must never take the run down
        logger.info("Screen decomposition failed (%s); building a single screen.", exc)
        names = []

    names = _clean_screen_names(names)
    if not names:
        names = [_default_screen_name(instruction)]
    logger.info("Screens (%d): %s", len(names), ", ".join(names))
    return [Screen(name=name) for name in names]


def _clean_screen_names(names: list[str]) -> list[str]:
    """Keep real screen names, drop sections, dedupe, cap.

    Deterministic clean-up of a model answer (section 7: the model makes
    decisions, the harness does the arithmetic and the filtering).
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = re.sub(r"\s+", " ", str(raw)).strip(" .:-")
        name = re.sub(r"^\d+[\.\)]\s*", "", name)  # "1. Login" -> "Login"
        if not name or len(name) > 40:
            continue
        key = name.lower()
        if key in _SECTION_WORDS or key in seen:
            continue
        seen.add(key)
        cleaned.append(name[:40])
        if len(cleaned) >= MAX_SCREENS:
            break
    return cleaned


def _default_screen_name(instruction: str) -> str:
    """Name the single screen after a quoted product name, else generically."""
    match = re.search(r"[\"'\u201c]([^\"'\u201d]{2,30})[\"'\u201d]", instruction or "")
    return match.group(1) if match else "Screen"


def make_plan(
    instruction: str,
    state: RunState,
    llm: ModelClient,
    screen: str = "",
    other_screens: list[str] | None = None,
    existing_sections: list[str] | None = None,
) -> list[str]:
    """Ask the model for an ordered list of small, atomic build steps.

    Scoped to ONE screen when `screen` is given, so each screen gets a plan
    about itself rather than one plan trying to describe the whole design.
    """
    sections = state.existing_sections if existing_sections is None else existing_sections
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": planning_user_message(
                instruction,
                state.inspection_summary,
                sections,
                screen=screen,
                other_screens=other_screens,
            ),
        },
    ]
    message = llm.complete(messages, tools=None)
    steps = _drop_component_steps(_parse_steps(message.content or "")) or [_FALLBACK_STEP]
    label = f" for '{screen}'" if screen else ""
    logger.info(
        "Plan%s (%d step%s):\n%s", label, len(steps), "" if len(steps) == 1 else "s", _format_plan(steps)
    )
    return steps


# A step whose whole job is creating a Figma COMPONENT. Telling the planner not
# to emit these is not enough on its own -- a real run planned five of them
# across five screens, and every one either failed ("would create a component
# inside a component") or left a loose component sitting on the canvas that no
# section ever used.
_COMPONENT_STEP = re.compile(
    r"\b(component|variant|component set|shared library|design system|style guide)\b",
    re.IGNORECASE,
)
# ...but a step that USES a component to build something visible is fine.
_BUILDS_SOMETHING = re.compile(r"\b(add|append|compose|place|build the|section)\b", re.IGNORECASE)


def _drop_component_steps(steps: list[str]) -> list[str]:
    """Remove "create a Button component" steps; keep steps that build sections.

    Components add nothing to a static mockup: consistency comes from the
    colour and text styles the harness already created, which the renderer
    applies to every node it makes.
    """
    kept = []
    for step in steps:
        if _COMPONENT_STEP.search(step) and not _BUILDS_SOMETHING.search(step):
            logger.info("Dropped component step: %s", step)
            continue
        kept.append(step)
    return kept


def _format_plan(steps: list[str]) -> str:
    return "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, start=1))


def _parse_steps(content: str) -> list[str]:
    """Parse a JSON array of strings out of the model's reply, tolerating stray fences/prose."""
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(step).strip() for step in parsed if str(step).strip()]
