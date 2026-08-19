"""The EDIT pipeline: change a design that already exists.

Create mode and edit mode are the same shape -- read the canvas, plan, do one
step at a time, gate the result -- but they are not the same job, and merging
them would make both worse:

- Create owns its parent frame and every id in it. Edit owns nothing: every
  target is the user's existing work, named by an id the model has to copy
  correctly, and a mistake damages something rather than adding something ugly
  beside it.
- Create's gate asks "is this section well formed?". Edit's gate asks "is this
  node still well formed, and did the edits I asked for actually apply?" --
  which is a different question with a different answer available (the script
  reports exactly which edits took).
- Create plans screens. Edit must never create one.

What they DO share is imported rather than copied: the model plumbing from
`loop`, the palette roles from `scaffold`/`renderer`, the geometry and contrast
checks from `critic`, and the compiler from `editor`.

Two rules that shape everything here:

1. **The canvas listing is re-read between steps.** An edit changes what the
   ids mean -- a `replace` makes the old id dead -- so a stale listing would
   have the next step targeting a node that is gone.
2. **A gate can only fail an edit on what the edit touched.** Judging the whole
   page would fail step 2 for a defect that was already in the file before the
   user asked for anything, which is not a thing the agent did and not a thing
   it should undo.
"""
from __future__ import annotations

import functools
import logging

from agent import critic, editor, inventory, metrics, planner, renderer, scaffold
from agent.llm import ModelClient
from agent.loop import (
    Cancelled,
    _assistant_message_dict,
    _safe_json_loads,
    _stop_if_asked,
    _tool_result_message,
    read_layout,
)
from agent.prompts import (
    EDIT_PLANNING_SYSTEM_PROMPT,
    EDIT_SYSTEM_PROMPT,
    edit_planning_user_message,
    edit_step_user_message,
)
from agent.state import PlanStep, RunResult, RunState, Screen, StepResult
from bridge.server import Bridge
from tools.figma_exec import execute_figma_js
from tools.figma_read import get_screenshot
from tools.registry import EDIT_TOOLS, dispatch

logger = logging.getLogger(__name__)

MAX_TURNS_PER_EDIT = 6
MAX_REFUSALS = 3
# More than this and the request is a redesign, not an edit.
MAX_EDIT_STEPS = 5


def run(
    instruction: str,
    bridge: Bridge,
    llm: ModelClient,
    max_retries: int = 3,
    max_steps: int = MAX_EDIT_STEPS,
    run_metrics=None,
    should_stop=None,
    references: str = "",
) -> RunResult:
    """Apply the requested changes to whatever is already in the file."""
    with metrics.recording(run_metrics) as measured:
        result = _run(instruction, bridge, llm, max_retries, max_steps, should_stop, references)
    result.metrics = measured.snapshot()
    logger.info("Edit cost: %s", measured.summary())
    return result


def _run(instruction, bridge, llm, max_retries, max_steps, should_stop, references="") -> RunResult:
    state = RunState(instruction=instruction)
    state.references = references
    state.should_stop = should_stop
    state.visual_gate_enabled = False  # the edit gate is its own, below
    try:
        return _apply(state, bridge, llm, max_retries, max_steps)
    except Cancelled:
        state.stopped = True
        state.warnings.insert(0, "You stopped this run, so not every change was applied.")
        return state.result()
    except Exception as exc:
        # Same reasoning as create mode: the edits that landed are on the
        # user's canvas whether or not the run finished.
        logger.info("Edit run ended early (%s: %s)", type(exc).__name__, exc)
        state.ended_early = True
        state.warnings.insert(0, f"The run ended early -- {exc}. Changes made before that point stand.")
        return state.result()


def _apply(state: RunState, bridge: Bridge, llm: ModelClient, max_retries: int, max_steps: int) -> RunResult:
    logger.info("Reading the current design...")
    canvas = read_inventory(bridge)
    if canvas is None:
        state.warnings.append("Could not read the canvas, so nothing was changed.")
        state.mark_failed(state.instruction)
        return state.result()
    if canvas.is_empty():
        logger.info("The page is empty -- there is nothing to edit.")
        state.warnings.append(
            "This file is empty. Edit mode changes an existing design; use Create to build one."
        )
        state.mark_failed(state.instruction)
        return state.result()

    logger.info(
        "Canvas: %d node(s) across %d screen(s)%s.",
        len(canvas.nodes),
        len(canvas.roots),
        f", {len(canvas.selection)} selected" if canvas.selection else "",
    )
    adopt_existing_styles(state, bridge)

    steps = plan_edits(state, canvas, llm)[:max_steps]
    state.plan = [PlanStep(description=s, screen_index=0, render_only=False) for s in steps]
    metrics.current().steps_planned = len(state.plan)

    for index, step in enumerate(state.plan, start=1):
        _stop_if_asked(state)
        label = f"Edit {index}/{len(state.plan)}"
        logger.info("%s: %s", label, step.description)
        metrics.current().start_step(step.description, index, len(state.plan))
        # Re-read: the last step may have replaced or removed the very nodes
        # this one is about.
        canvas = read_inventory(bridge) or canvas
        run_edit_step(step, state, canvas, bridge, llm, max_retries, label, index)

    review(state, bridge)
    logger.info(
        "Done. %d change(s) applied, %d step(s) failed.",
        len(state.created_node_ids), len(state.failed_steps),
    )
    return state.result()


def read_inventory(bridge: Bridge) -> inventory.Inventory | None:
    """One round trip: the whole page, plus what the user has selected."""
    result = execute_figma_js(bridge, inventory.inventory_script())
    if not result["ok"]:
        logger.info("Could not read the canvas: %s", result["error"])
        return None
    return inventory.build(result.get("result") or {})


def adopt_existing_styles(state: RunState, bridge: Bridge) -> None:
    """Learn the file's OWN palette and text styles, rather than imposing ours.

    An edit run must bind to the styles the design already uses. Creating a
    fresh set would mean "make the button purple" introduces a second, slightly
    different purple that no other node references -- the exact untokenised
    drift the design-system checks exist to catch.
    """
    result = execute_figma_js(bridge, _READ_STYLES_SCRIPT)
    if not result["ok"]:
        state.warnings.append(f"Could not read the file's styles: {result['error']}")
        return
    payload = result.get("result") or {}
    state.token_names = [str(n) for n in (payload.get("paintStyles") or [])]
    state.text_style_names = [str(n) for n in (payload.get("textStyles") or [])]

    palette = [
        (name.replace("color/", "", 1), _hex(rgb))
        for name, rgb in (payload.get("paintColors") or {}).items()
        if isinstance(rgb, dict)
    ]
    state.palette = palette
    state.palette_info = scaffold.describe_palette(palette)
    state.readable_pairings = scaffold.readable_pairings(palette)
    logger.info(
        "Using the file's own styles: %d colour(s), %d text style(s).",
        len(state.token_names), len(state.text_style_names),
    )
    if not state.token_names:
        state.warnings.append(
            "This file has no colour styles, so colour changes cannot be token-backed."
        )


def _hex(rgb: dict) -> str:
    return "#" + "".join(
        f"{max(0, min(255, round(float(rgb.get(c, 0)) * 255))):02X}" for c in ("r", "g", "b")
    )


_READ_STYLES_SCRIPT = """\
const paints = await figma.getLocalPaintStylesAsync();
const texts = await figma.getLocalTextStylesAsync();
const names = [];
const colors = {};
for (const s of paints) {
  names.push(s.name);
  const paint = Array.isArray(s.paints) ? s.paints[0] : null;
  if (paint && paint.type === 'SOLID' && paint.color) {
    colors[s.name] = { r: paint.color.r, g: paint.color.g, b: paint.color.b };
  }
}
return {
  createdNodeIds: [],
  paintStyles: names,
  paintColors: colors,
  textStyles: texts.map(function (t) { return t.name; })
};
"""


def plan_edits(state: RunState, canvas: inventory.Inventory, llm: ModelClient) -> list[str]:
    """Break the request into ordered changes. One step is the common case."""
    selection_note = ""
    if canvas.selection:
        selection_note = (
            f"The user has {len(canvas.selection)} node(s) selected in Figma "
            f"({', '.join(canvas.selection[:8])}); the request is most likely about those.\n\n"
        )
    messages = [
        {"role": "system", "content": EDIT_PLANNING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": edit_planning_user_message(
                state.design_source(),
                inventory.format_listing(canvas, canvas.selection or None),
                selection_note,
            ),
        },
    ]
    try:
        steps = planner._parse_steps(llm.complete(messages, tools=None).content or "")
    except Exception as exc:
        logger.info("Edit planning failed (%s); treating it as a single change.", exc)
        steps = []
    steps = [s for s in steps if s.strip()] or [state.instruction]
    logger.info(
        "Plan (%d change%s):\n%s",
        len(steps), "" if len(steps) == 1 else "s",
        "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, start=1)),
    )
    return steps


def run_edit_step(
    step: PlanStep,
    state: RunState,
    canvas: inventory.Inventory,
    bridge: Bridge,
    llm: ModelClient,
    max_retries: int,
    label: str,
    index: int,
) -> None:
    """One change, with retries. A retry is told exactly which edits failed."""
    applied: list[str] = []
    problems: list[str] = []

    for attempt in range(max_retries):
        _stop_if_asked(state)
        metrics.current().start_attempt()
        if attempt:
            logger.info("%s: retrying (attempt %d/%d)...", label, attempt + 1, max_retries)
        outcome = converse_edit(
            step, state, canvas, bridge, llm, label, index, applied, problems
        )
        applied = list(dict.fromkeys(applied + outcome.applied))
        if outcome.ok and not outcome.problems:
            defects = edit_gate(outcome.touched, bridge, label)
            if not defects:
                state.add_node_ids(outcome.touched + outcome.created)
                state.record_step_result(
                    step.description,
                    StepResult(step.description, ok=True, created_node_ids=outcome.touched,
                               summary=f"{len(applied)} change(s) applied"),
                )
                metrics.current().steps_completed += 1
                logger.info("%s: done -- %d change(s) applied.", label, len(applied))
                return
            problems = defects
            logger.info("%s: the change left %d problem(s), correcting...", label, len(defects))
        else:
            problems = outcome.problems or [outcome.summary]
            metrics.current().record_failure("edit-failed")
            logger.info("%s failed (attempt %d/%d): %s", label, attempt + 1, max_retries, problems[0])
        # The canvas moved under us; re-read before the next attempt.
        canvas = read_inventory(bridge) or canvas

    state.mark_failed(step)
    metrics.current().steps_failed += 1
    if applied:
        # Partial is the honest answer. An edit batch is not atomic by design,
        # and pretending nothing happened would send the user looking for
        # changes that are really there.
        state.warnings.append(
            f"'{step.description}' only partly applied: {len(applied)} change(s) landed. "
            + (f"Still wrong: {problems[0]}" if problems else "")
        )
    else:
        state.warnings.append(f"'{step.description}' could not be applied. {problems[0] if problems else ''}")


class EditOutcome:
    def __init__(self, ok, applied=None, problems=None, touched=None, created=None, summary=""):
        self.ok = ok
        self.applied = applied or []
        self.problems = problems or []
        self.touched = touched or []
        self.created = created or []
        self.summary = summary


def converse_edit(
    step: PlanStep,
    state: RunState,
    canvas: inventory.Inventory,
    bridge: Bridge,
    llm: ModelClient,
    label: str,
    index: int,
    already_applied: list[str],
    problems: list[str],
) -> EditOutcome:
    """One attempt at one change: the model calls `edit_ui`, we run it."""
    messages = [
        {"role": "system", "content": EDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": edit_step_user_message(
                step.description,
                state.design_source(),
                inventory.format_listing(canvas, canvas.selection or None),
                plan=[s.description for s in state.plan],
                step_index=index,
                selection=canvas.selection,
                palette_info=state.palette_info,
                text_style_names=state.text_style_names,
                pairings=state.readable_pairings,
                applied=already_applied,
                problems=problems,
            ),
        },
    ]
    context = {
        "resolve": functools.partial(inventory.resolve, canvas),
        "color_roles": renderer.role_map(state.palette_info),
        "token_names": list(state.token_names),
        # Top-level frames are whole screens. Nothing here may delete one.
        "protected_ids": {str(r.get("id")) for r in canvas.roots if r.get("id")},
    }

    applied: list[str] = []
    touched: list[str] = []
    created: list[str] = []
    ran = False
    refusals = 0
    seen: set[str] = set()

    for turn in range(1, MAX_TURNS_PER_EDIT + 1):
        _stop_if_asked(state)
        logger.info("%s: thinking (turn %d/%d)...", label, turn, MAX_TURNS_PER_EDIT)
        message = llm.complete(messages, tools=EDIT_TOOLS)
        messages.append(_assistant_message_dict(message))

        calls = getattr(message, "tool_calls", None)
        if not calls:
            if ran:
                return EditOutcome(True, applied, [], touched, created, message.content or "done")
            return EditOutcome(
                False, applied, ["you replied with text instead of calling `edit_ui`"], touched
            )

        for call in calls:
            args = _safe_json_loads(call.function.arguments)
            logger.info("%s: -> %s", label, _describe(call.function.name, args))

            signature = (call.function.name, call.function.arguments or "")
            if call.function.name not in {t["function"]["name"] for t in EDIT_TOOLS}:
                refusals += 1
                messages.append(_tool_result_message(call.id, {"ok": False, "error": _WRONG_TOOL}))
                logger.info("%s: refused %s -- not available here", label, call.function.name)
            elif signature in seen:
                refusals += 1
                messages.append(_tool_result_message(call.id, {"ok": False, "error": _REPEAT}))
                logger.info("%s: refused a repeated identical call", label)
            else:
                seen.add(signature)
                result = dispatch(call.function.name, args, bridge, render_context=context)
                if call.function.name != "edit_ui":
                    messages.append(_tool_result_message(call.id, result))
                    continue
                if not result["ok"]:
                    logger.info("%s: edit rejected -- %s", label, result["error"])
                    messages.append(_tool_result_message(call.id, result))
                    if not result.get("recoverable"):
                        return EditOutcome(False, applied, [str(result["error"])], touched)
                    refusals += 1
                else:
                    ran = True
                    payload = result.get("result") or {}
                    landed = [str(a) for a in (payload.get("appliedEdits") or [])]
                    refused = [str(f) for f in (payload.get("failedEdits") or [])]
                    applied += landed
                    touched += [str(t) for t in (result.get("touchedNodeIds") or [])]
                    created += [str(c) for c in (payload.get("createdNodeIds") or [])]
                    for line in landed[:6]:
                        logger.info("%s: applied %s", label, line)
                    for line in refused[:6]:
                        logger.info("%s: FAILED %s", label, line)
                    if refused:
                        return EditOutcome(False, applied, refused, touched, created)
                    messages.append(_tool_result_message(call.id, result))

            if refusals >= MAX_REFUSALS:
                logger.info("%s: %d refused calls -- ending the attempt", label, refusals)
                return EditOutcome(ran, applied, [] if ran else ["the model could not make a valid call"],
                                   touched, created)

    return EditOutcome(ran, applied, [] if ran else ["ran out of turns without a valid edit"],
                       touched, created)


def edit_gate(touched: list[str], bridge: Bridge, label: str) -> list[str]:
    """Did the change leave the nodes it touched in a good state?

    Scoped to what THIS edit touched. The file is the user's own work and may
    well have had problems before the agent arrived; failing an edit for one of
    those would have the agent undoing changes it was asked to make in order to
    fix something nobody mentioned.
    """
    if not touched:
        return []
    defects: list[str] = []
    for node_id in touched[:6]:
        tree = read_layout(node_id, bridge)
        if tree is None:
            continue
        found = critic.find_layout_defects(tree)
        defects += [str(d) for d in found]
    unique = list(dict.fromkeys(defects))[:5]
    for defect in unique:
        logger.info("%s: defect %s", label, defect)
    return unique


def review(state: RunState, bridge: Bridge) -> None:
    """Report the finished file: a screenshot per screen, and what is still wrong."""
    canvas = read_inventory(bridge)
    if canvas is None:
        return
    # `Screen` objects, not names: RunResult builds its screen list from these,
    # and the dashboard's pager indexes screenshots against it.
    state.screens = [
        Screen(name=str(r.get("name") or "?"), frame_id=str(r.get("id") or ""))
        for r in canvas.roots
    ]
    defects: list[str] = []
    notes: list[str] = []
    for root in canvas.roots:
        tree = read_layout(str(root.get("id")), bridge)
        if tree is None:
            continue
        defects += [str(d) for d in critic.find_layout_defects(tree)]
        notes += [str(d) for d in critic.find_design_defects(tree)]
        shot = _screenshot(bridge, str(root.get("id")))
        if shot:
            state.screen_shots.append(
                {"name": str(root.get("name") or "Screen"), "image_base64": shot}
            )
    state.layout_defects = list(dict.fromkeys(defects))[:12]
    state.design_notes = list(dict.fromkeys(notes))[:12]
    if state.screen_shots:
        state.final_screenshot_base64 = state.screen_shots[0]["image_base64"]


def _screenshot(bridge: Bridge, node_id: str) -> str | None:
    """Best-effort: a picture is a nice-to-have and must not fail the run."""
    try:
        result = get_screenshot(bridge, node_id)
    except Exception as exc:
        logger.info("Could not photograph %s: %s", node_id, exc)
        return None
    return result.get("image_base64") if result.get("ok") else None


def _describe(name: str, args: dict) -> str:
    if name != "edit_ui":
        return f"{name}({args.get('node_id') or 'current page'})"
    edits = args.get("edits")
    if isinstance(edits, list):
        ops = ", ".join(str(e.get("op")) for e in edits[:6] if isinstance(e, dict))
        return f"edit_ui({len(edits)} edit(s): {ops})"
    return f"edit_ui(args: {', '.join(sorted(args)) or 'none'})"


_WRONG_TOOL = (
    "REFUSED: that tool is not available when editing. Use `edit_ui` with a list of edits, "
    "targeting node ids copied exactly from the canvas listing. To add something new, use "
    "the `insert` op with a `parent`; to swap a section, use `replace`."
)

_REPEAT = (
    "REFUSED: you already ran this exact call in this step. Repeating it cannot tell you "
    "anything new. Either make a DIFFERENT call, or stop and reply with a one-line summary "
    "and no tool call."
)
