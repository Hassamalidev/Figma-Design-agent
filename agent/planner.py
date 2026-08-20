"""Turns the instruction into an ordered plan: tokens, then components, then composition."""
from __future__ import annotations

import json
import logging
import re
from collections import Counter

from agent import scaffold
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
    "features", "testimonials", "pricing table", "menu", "toolbar", "topbar",
    "appbar", "breadcrumbs", "search bar", "hero banner", "feature grid",
}

# Generic nouns tacked onto a section's name. "Hero Section" and "Navigation
# Bar" are the same mistake as "Hero" and "Nav", but an exact-match filter saw
# neither -- so they became top-level frames, and one screen's parts ended up
# scattered across the canvas as siblings of the screen they belong to.
#
# "page" is deliberately NOT here: "Landing Page" and "Settings Page" are real
# screen names, and stripping the noun would leave "Landing", not a section.
_SECTION_SUFFIXES = (
    "section", "area", "block", "band", "row", "bar", "panel", "component",
    "module", "region", "strip", "group", "content",
)


def _is_section_name(name: str) -> bool:
    """Is this the name of a SECTION rather than a screen?

    Deterministic, because promoting a section to its own top-level frame is
    not a judgement call -- a hero is part of a screen however it is spelled.
    """
    key = re.sub(r"[^a-z0-9 ]+", " ", (name or "").lower())
    key = re.sub(r"\s+", " ", key).strip()
    if not key:
        return True
    if key in _SECTION_WORDS or key.replace(" ", "") in _SECTION_WORDS:
        return True
    if "section" in key.split():
        return True  # anything that calls itself one
    words = key.split()
    while len(words) > 1 and words[-1] in _SECTION_SUFFIXES:
        words.pop()
        stem = " ".join(words)
        if stem in _SECTION_WORDS or stem.replace(" ", "") in _SECTION_WORDS:
            return True
    return False


_MOBILE = re.compile(r"\b(mobile|phone|iphone|android|handset|ios app)\b", re.IGNORECASE)
_TABLET = re.compile(r"\b(tablet|ipad)\b", re.IGNORECASE)


def _device_from_text(text: str) -> str:
    """The device a piece of writing points at, or "" if it points at none."""
    if _MOBILE.search(text or ""):
        return "mobile"
    if _TABLET.search(text or ""):
        return "tablet"
    return ""


def _infer_device(declared: str, name: str, purpose: str, fallback: str) -> str:
    """desktop | tablet | mobile, decided per screen rather than per request.

    The width used to come from a single scan of the whole instruction and was
    then applied to every screen, so a mobile screen asked for alongside a
    desktop one came out 1440px wide.

    A screen's own wording wins over the fallback, so "a marketing site with a
    mobile app preview screen" sizes only that one screen as a phone.
    """
    declared = (declared or "").strip().lower()
    if declared in scaffold.DEVICE_WIDTHS:
        return declared
    return _device_from_text(name + " " + purpose) or fallback


def _fallback_device(entries: list[dict], instruction: str) -> str:
    """What a screen that says nothing about its device should get.

    The instruction is read ONLY when no screen declared a device at all. A
    device word in the request usually belongs to one screen -- "a login, a
    phone dashboard and settings" made the settings screen 390px wide, because
    the word "phone" was about the screen beside it. When other screens did
    declare, the run's own dominant device is the far better guess.
    """
    declared = [
        device
        for device in (str(entry.get("device", "")).strip().lower() for entry in entries)
        if device in scaffold.DEVICE_WIDTHS
    ]
    if declared:
        return Counter(declared).most_common(1)[0][0]
    return _device_from_text(instruction) or "desktop"


def plan_screens(instruction: str, brief: str, llm: ModelClient) -> list[Screen]:
    """Decide which SCREENS the request needs -- one Figma frame each.

    Asked as its own small question rather than inferred from the plan. "How
    many screens is this?" is a question even a small model answers reliably,
    while recovering the same structure by keyword-matching English plan steps
    is exactly the guesswork CLAUDE.md lists as a known weakness.

    Each screen comes back with a PURPOSE and a DEVICE as well as a name, and
    both feed a stage that had to guess without them: the purpose is what that
    screen's own plan is written from, and the device is what decides its frame
    width.

    Always returns at least one screen, so a failed or nonsense answer simply
    behaves like the single-screen design that most requests are.
    """
    messages = [
        {"role": "system", "content": SCREENS_SYSTEM_PROMPT},
        {"role": "user", "content": screens_user_message(instruction, brief)},
    ]
    try:
        message = llm.complete(messages, tools=None)
        entries = _parse_screen_entries(message.content or "")
    except Exception as exc:  # a screen list must never take the run down
        logger.info("Screen decomposition failed (%s); building a single screen.", exc)
        entries = []

    screens = _clean_screens(entries, instruction)
    if not screens:
        screens = [_default_screen(instruction)]
    logger.info(
        "Screens (%d): %s",
        len(screens),
        ", ".join("%s [%s %dpx]" % (s.name, s.device, s.width) for s in screens),
    )
    return screens


def _parse_screen_entries(content: str) -> list[dict]:
    """Normalise the model's answer into {name, purpose, device} dicts.

    Both shapes are accepted -- a JSON array of objects, and the plain array of
    names smaller models answer with. Insisting on the richer shape would mean
    the screen list, the one question every run depends on, fails whenever the
    model gets casual about the format.
    """
    entries: list[dict] = []
    for raw in _parse_json_list(content):
        if isinstance(raw, dict):
            name = raw.get("name") or raw.get("screen") or raw.get("title") or ""
            entries.append(
                {
                    "name": str(name).strip(),
                    "purpose": str(raw.get("purpose") or raw.get("description") or "").strip(),
                    "device": str(raw.get("device") or raw.get("platform") or "").strip(),
                }
            )
        elif str(raw).strip():
            entries.append({"name": str(raw).strip(), "purpose": "", "device": ""})
    return entries


# A purpose is one line of context for that screen's planner, not a second brief.
MAX_PURPOSE_CHARS = 160


def _clean_screens(entries: list[dict], instruction: str) -> list[Screen]:
    """Keep real screens, drop sections, dedupe, size, cap.

    Deterministic clean-up of a model answer (section 7: the model makes
    decisions, the harness does the arithmetic and the filtering).
    """
    cleaned: list[Screen] = []
    seen: set[str] = set()
    fallback = _fallback_device(entries, instruction)
    for entry in entries:
        name = re.sub(r"\s+", " ", str(entry.get("name", ""))).strip(" .:-")
        name = re.sub(r"^\d+[\.\)]\s*", "", name)  # "1. Login" -> "Login"
        if not name or len(name) > 40:
            continue
        key = name.lower()
        if key in seen or _is_section_name(name):
            continue
        seen.add(key)
        purpose = re.sub(r"\s+", " ", str(entry.get("purpose", ""))).strip()
        device = _infer_device(str(entry.get("device", "")), name, purpose, fallback)
        width, height = scaffold.device_size(device)
        cleaned.append(
            Screen(
                name=name[:40],
                purpose=purpose[:MAX_PURPOSE_CHARS],
                device=device,
                width=width,
                height=height,
            )
        )
        if len(cleaned) >= MAX_SCREENS:
            break
    return cleaned


def _default_screen(instruction: str) -> Screen:
    """The single screen a request falls back to, sized from its own wording."""
    device = _device_from_text(instruction) or "desktop"
    width, height = scaffold.device_size(device)
    return Screen(
        name=_default_screen_name(instruction),
        device=device,
        width=width,
        height=height,
    )


def _default_screen_name(instruction: str) -> str:
    """Name the single screen after a quoted product name, else generically."""
    match = re.search("[\"'“]([^\"'”]{2,30})[\"'”]", instruction or "")
    return match.group(1) if match else "Screen"


def make_plan(
    instruction: str,
    state: RunState,
    llm: ModelClient,
    screen: str = "",
    other_screens: list[str] | None = None,
    existing_sections: list[str] | None = None,
    screen_purpose: str = "",
) -> list[str]:
    """Ask the model for an ordered list of small, atomic build steps.

    Scoped to ONE screen when `screen` is given, so each screen gets a plan
    about itself rather than one plan trying to describe the whole design.

    What comes back is then put into BUILD ORDER by the harness, because the
    order is not a preference: every step appends to the bottom of a vertical
    auto-layout frame, so a plan that names the footer before the hero builds
    the screen upside down.
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
                screen_purpose=screen_purpose,
            ),
        },
    ]
    message = llm.complete(messages, tools=None)
    steps = _drop_component_steps(_parse_steps(message.content or "")) or [_FALLBACK_STEP]
    steps = _collapse_side_by_side(steps, screen)
    steps = order_steps(steps)
    label = f" for '{screen}'" if screen else ""
    logger.info(
        "Plan%s (%d step%s):\n%s", label, len(steps), "" if len(steps) == 1 else "s", _format_plan(steps)
    )
    return steps


# How far into a step we look for a word that places it vertically. Plan steps
# read "Add the <region> ...", so the region is always at the front -- and
# matching the whole string would sort "a sign-in card with a footer link"
# to the bottom of the screen.
_RANK_WINDOW = 40

# Regions whose vertical position is not a matter of taste. Everything else
# keeps the order the model gave it.
_TOP_OF_SCREEN = re.compile(
    r"\b(nav|navbar|navigation|header|top bar|app bar|menu bar|breadcrumb)\b",
    re.IGNORECASE,
)
_NEAR_TOP = re.compile(r"\b(hero|masthead|above the fold)\b", re.IGNORECASE)
_BOTTOM_OF_SCREEN = re.compile(r"\b(footer)\b", re.IGNORECASE)

_DEFAULT_RANK = 2


def order_steps(steps: list[str]) -> list[str]:
    """Put the steps into the order they must be BUILT in, top to bottom.

    A screen frame is a vertical auto-layout and every section step appends
    into it, so step order IS visual order -- a plan that returns the footer
    first puts the footer at the top of the screen, and nothing downstream can
    see anything wrong: each band is individually well formed, exactly like the
    side-by-side failure `_collapse_side_by_side` exists for.

    Deliberately conservative. Only a step that clearly names the top or the
    bottom of a screen moves; the sort is stable, so everything else keeps the
    order the model chose. Reordering a plan we do not understand would be a
    worse failure than the one being fixed.
    """
    if len(steps) < 2:
        return steps
    ranked = sorted(enumerate(steps), key=lambda pair: (_rank_of(pair[1]), pair[0]))
    ordered = [step for _, step in ranked]
    if ordered != steps:
        logger.info("Reordered the plan into build order:\n%s", _format_plan(ordered))
    return ordered


def _rank_of(step: str) -> int:
    """Where this step's region sits vertically: lower is nearer the top."""
    head = str(step)[:_RANK_WINDOW]
    if _TOP_OF_SCREEN.search(head):
        return 0
    if _NEAR_TOP.search(head):
        return 1
    if _BOTTOM_OF_SCREEN.search(head):
        return 3
    return _DEFAULT_RANK


# A merged tail still has to read as one instruction, not as a paragraph.
MAX_MERGED_STEP_CHARS = 220


def fit_steps(steps: list[str], limit: int) -> list[str]:
    """Fit a plan into its step budget by MERGING the tail, never dropping it.

    The budget used to be applied with a slice, which silently threw the end of
    the screen away: a five-band landing page planned as hero/features/
    pricing/testimonials/footer, capped at three, shipped without its
    testimonials or its footer and reported success. Nothing else in the run
    could notice -- the steps it did build all passed.

    Merging keeps the work in the plan. The last step gets bigger, which the
    builder handles (it renders a whole screen in one call by default), and the
    design keeps the regions the plan asked for.
    """
    if limit < 1 or len(steps) <= limit:
        return list(steps)
    kept = list(steps[: limit - 1])
    tail = [_region_of(step) for step in steps[limit - 1 :]]
    merged = "Add these in order into the frame: " + "; ".join(tail)
    if len(merged) > MAX_MERGED_STEP_CHARS:
        merged = merged[: MAX_MERGED_STEP_CHARS - 1].rstrip() + "\u2026"
    logger.info("Merged the last %d steps to fit the budget: %s", len(tail), merged)
    return kept + [merged]


# Words that place a region HORIZONTALLY. A step that names one is describing
# half of a layout, not a band of a scrolling page.
_SIDE_BY_SIDE = re.compile(
    r"\b(left|right|side[- ]?bar|side panel|split|two[- ]column|split[- ]screen|"
    r"beside|alongside|left-hand|right-hand)\b",
    re.IGNORECASE,
)


def _collapse_side_by_side(steps: list[str], screen: str = "") -> list[str]:
    """A screen laid out side by side can only be built in ONE step.

    Every step appends a full-width band under the last, because a screen frame
    is a VERTICAL auto-layout. So "add the left visual panel" then "add the
    right auth panel" does not produce two halves -- it produces two stacked
    bands, with the form underneath the artwork instead of next to it. That is
    precisely what a live run shipped, and no later gate can see it: both bands
    are individually well-formed.

    The planning prompt says this too, and the prompt was ignored. This is the
    harness taking the mistake off the table (CLAUDE.md section 6a).
    """
    if len(steps) < 2 or not any(_SIDE_BY_SIDE.search(step) for step in steps):
        return steps
    regions = "; ".join(_region_of(step) for step in steps)
    merged = f"Build the whole {screen or 'screen'} in one side-by-side layout: {regions}"
    logger.info("Collapsed %d side-by-side steps into one: %s", len(steps), merged)
    return [merged]


def _region_of(step: str) -> str:
    """The part of a step that names its region, without the plumbing."""
    text = re.sub(
        r"^\s*(add|append|create|place|build)\s+(the\s+)?", "", step.strip(), flags=re.IGNORECASE
    )
    text = re.sub(
        r"[,\s]*(in)?to the (screen )?frame\.?\s*$", "", text, flags=re.IGNORECASE
    )
    return text.strip(" .") or step.strip()


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
    return [str(step).strip() for step in _parse_json_list(content) if str(step).strip()]


def _parse_json_list(content: str) -> list:
    """The first JSON array in the reply, with its entries left as they came.

    Entries stay raw so screen decomposition can read objects while step
    planning reads strings -- both answers arrive the same way, wrapped in the
    same stray prose and markdown fences.
    """
    match = re.search(r"\[.*\]", content or "", re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
