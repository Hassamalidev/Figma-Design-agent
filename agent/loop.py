"""Drives a run: inspect -> plan -> per-step (generate -> execute -> observe ->
validate -> retry). Hand-rolled, no framework -- see CLAUDE.md section 6.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from agent import critic, planner, renderer, scaffold
from agent.llm import ModelClient
from agent.prompts import step_user_message, system_prompt
from agent.state import RunResult, RunState, StepResult
from bridge.server import Bridge
from knowledge import index as knowledge_index
from tools.docs import query_docs
from tools.figma_exec import execute_figma_js
from tools.figma_read import get_metadata, get_screenshot
from tools.registry import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)

# Read-only: lists what's already on the current page. Runs before anything
# else so the plan is grounded in the real canvas, not an assumed blank one.
INSPECT_SCRIPT = """\
const page = figma.currentPage;
const summary = page.children.slice(0, 50).map(function (n) {
  var info = {
    id: n.id, name: n.name, type: n.type,
    x: Math.round(n.x), y: Math.round(n.y),
    width: Math.round(n.width), height: Math.round(n.height),
    layoutMode: 'layoutMode' in n ? n.layoutMode : null,
    children: []
  };
  if ('children' in n) {
    info.children = n.children.slice(0, 30).map(function (c) {
      return { id: c.id, name: c.name, type: c.type, height: Math.round(c.height) };
    });
  }
  return info;
});
return { createdNodeIds: [], pageName: page.name, topLevelNodes: summary };
"""

# Never guess a font style string: a live run died on "Inter SemiBold" because
# the real style is "Semi Bold" with a space. Discover them at runtime.
FONT_SCRIPT = """\
const fonts = await figma.listAvailableFontsAsync();
const styles = [];
for (const f of fonts) {
  if (f.fontName.family === 'Inter') { styles.push(f.fontName.style); }
}
return { createdNodeIds: [], interStyles: styles };
"""

# The root frame is written by the harness, not the model. It is the same
# script every time, it is the single point every later step depends on, and
# getting it wrong cascades: in one live run the model failed it three times
# (trying to set FILL on a child of the PAGE, which is never legal), and all
# six "append into root frame" steps then had no parent to append to.
#
# Note it deliberately uses primaryAxisSizingMode/counterAxisSizingMode rather
# than layoutSizing*: those are the properties that work on a frame whose
# parent (the page) is not auto-layout.
ROOT_FRAME_SCRIPT = """\
const frame = figma.createFrame();
frame.name = {name};
frame.resize({width}, 900);
figma.currentPage.appendChild(frame);
frame.x = {x}; frame.y = {y};
frame.layoutMode = 'VERTICAL';
frame.counterAxisSizingMode = 'FIXED';
frame.primaryAxisSizingMode = 'AUTO';
frame.itemSpacing = 0;
frame.fills = [{{ type: 'SOLID', color: {{ r: 1, g: 1, b: 1 }} }}];
return {{ createdNodeIds: [frame.id] }};
"""

# Gap left between an existing design and a newly created one, so a second run
# never lands on top of the first.
NEW_FRAME_GAP = 200

DEFAULT_ROOT_WIDTH = 1440

# Bounds one step's tool-calling back-and-forth so a confused model can't spin forever.
MAX_TOOL_TURNS_PER_STEP = 8

# Retrieval is restricted to the typings: the gotchas are inlined in the
# system prompt, so pulling them in again per step would spend context twice.
STEP_DOC_SOURCES = ("api_types.d.ts",)


def run(
    instruction: str,
    bridge: Bridge,
    llm: ModelClient,
    max_retries: int,
    max_steps: int,
    visual_gate: bool = True,
    critic_llm: ModelClient | None = None,
) -> RunResult:
    state = RunState(instruction=instruction)
    state.visual_gate_enabled = visual_gate
    # A separate vision model, if one is configured. The generator needs
    # reliable tool calling; the critic needs eyes. Rarely the same model.
    state.critic_llm = critic_llm
    if critic_llm is None:
        # Never send images to a model not known to accept them: a text-only
        # endpoint 400s, which costs a round trip per step to rediscover.
        state.model_sees_images = False

    logger.info("Inspecting the current canvas...")
    inspect_file(state, bridge)  # 1. READ-ONLY: never assume canvas state

    state.enhanced_brief = planner.enhance_instruction(instruction, llm)  # fill in what's unspecified

    create_root_frame(state, bridge)  # deterministic: every section needs this parent
    bootstrap_tokens(state, bridge)   # deterministic: the model never has to write these

    state.plan = planner.make_plan(state.enhanced_brief, state, llm)[:max_steps]  # components -> composition

    total = len(state.plan)
    for index, step in enumerate(state.plan, start=1):  # 3. one step at a time
        logger.info("Step %d/%d: %s", index, total, step)
        run_step(step, state, bridge, llm, max_retries, index, total)

    logger.info("Final validation: screenshot + variable-binding audit...")
    final_validation(state, bridge)  # 4. whole-screen screenshot
    logger.info("Done. Success=%s, created %d node(s).", not state.failed_steps, len(state.created_node_ids))
    return state.result()


def inspect_file(state: RunState, bridge: Bridge) -> None:
    """The first action is always read-only: list what's already on the page."""
    result = execute_figma_js(bridge, INSPECT_SCRIPT)
    if not result["ok"]:
        state.inspection_summary = f"(inspection failed: {result['error']})"
        state.warnings.append(f"Initial inspection failed: {result['error']}")
        return
    payload = result.get("result") or {}
    state.existing_nodes = payload.get("topLevelNodes") or []
    state.inspection_summary = json.dumps(payload)
    logger.info("Canvas has %d top-level node(s).", len(state.existing_nodes))
    discover_fonts(state, bridge)


def discover_fonts(state: RunState, bridge: Bridge) -> None:
    """Read the real Inter style strings so no script has to guess them."""
    result = execute_figma_js(bridge, FONT_SCRIPT)
    if not result["ok"]:
        return
    styles = (result.get("result") or {}).get("interStyles") or []
    state.font_styles = [s for s in styles if isinstance(s, str)][:20]
    if state.font_styles:
        logger.info("Inter styles available: %s", ", ".join(state.font_styles))


def create_root_frame(state: RunState, bridge: Bridge) -> None:
    """Reuse the page's existing root frame, or build a fresh one beside it.

    Deterministic on purpose (CLAUDE.md: the harness is where quality comes
    from). Every section step then has a guaranteed-valid auto-layout parent
    whose id we hand it, which also makes FILL legal on its children.

    Reuse matters as much as creation: re-running on a file you already
    designed must CONTINUE that work, not stamp a second copy on top of it.
    """
    width = _root_width(state.instruction)
    existing = _find_reusable_root(state.existing_nodes, width)

    if existing is not None:
        state.root_frame_id = existing["id"]
        state.root_is_existing = True
        state.existing_sections = [c.get("name", "?") for c in existing.get("children") or []]
        logger.info(
            "Continuing in the existing frame %s (%s) with %d section(s): %s",
            existing.get("name"),
            existing["id"],
            len(state.existing_sections),
            ", ".join(state.existing_sections) or "none yet",
        )
        return

    script = ROOT_FRAME_SCRIPT.format(
        name=json.dumps(_root_name(state.instruction)),
        width=width,
        x=_free_x(state.existing_nodes),  # never stack on top of existing work
        y=200,
    )
    result = execute_figma_js(bridge, script)
    if not result["ok"]:
        state.warnings.append(f"Could not create the root frame: {result['error']}")
        logger.info("Root frame creation FAILED: %s", result["error"])
        return
    ids = _normalize_node_ids((result.get("result") or {}).get("createdNodeIds"))
    if ids:
        state.root_frame_id = ids[0]
        state.add_node_ids(ids)
        logger.info("New root frame: %s (%dpx wide)", state.root_frame_id, width)


def bootstrap_tokens(state: RunState, bridge: Bridge) -> None:
    """Create the colour + text tokens ourselves, before the plan exists.

    Token creation is the step that failed most often in live runs, and it is
    completely mechanical: the palette is already spelled out in the model's
    own brief. Doing it here means the plan starts at components/composition
    with tokens guaranteed to exist, and gives every later step real style
    names to bind to instead of hardcoded hexes.
    """
    palette = scaffold.extract_palette(state.enhanced_brief)
    state.palette = palette  # kept so hardcoded fills can be rebound to these later

    colors = execute_figma_js(bridge, scaffold.build_token_script(palette))
    if colors["ok"]:
        payload = colors.get("result") or {}
        state.token_names = payload.get("tokenNames") or []
        logger.info("Colour tokens ready (%d): %s", len(state.token_names), ", ".join(state.token_names))
        for failure in payload.get("failedTokens") or []:
            logger.info("Token skipped -- %s", failure)
            state.warnings.append(f"Colour token skipped: {failure}")
    else:
        state.warnings.append(f"Colour tokens could not be created: {colors['error']}")
        logger.info("Colour tokens FAILED: %s", colors["error"])

    describe_usable_palette(state, palette)

    text = execute_figma_js(bridge, scaffold.build_text_style_script())
    if text["ok"]:
        state.text_style_names = (text.get("result") or {}).get("textStyleNames") or []
        logger.info("Text styles ready (%d): %s", len(state.text_style_names), ", ".join(state.text_style_names))
    else:
        state.warnings.append(f"Text styles could not be created: {text['error']}")
        logger.info("Text styles FAILED: %s", text["error"])


def describe_usable_palette(state: RunState, palette: list[tuple[str, str]]) -> None:
    """Derive roles and legal pairings -- for the tokens that REALLY exist.

    Only ever advertise styles Figma actually created. When the token script
    failed, the prompt still listed every colour, so every generated script
    looked its style up, found nothing, and silently fell back to a hardcoded
    fill: a whole run of untokenised colour that no gate could see.
    """
    created = set(state.token_names)
    usable = [(name, hex_value) for name, hex_value in palette if f"color/{name}" in created]

    state.palette_info = scaffold.describe_palette(usable)
    state.readable_pairings = scaffold.readable_pairings(usable)

    if not usable:
        state.warnings.append(
            "No colour tokens exist, so this run cannot be token-backed -- "
            "every fill will be a hardcoded one-off."
        )
    elif state.readable_pairings:
        logger.info("Readable colour pairings: %s", "; ".join(state.readable_pairings[:3]))
    elif len(usable) > 1:
        # One colour has no pairs to measure; two or more that still can't make
        # readable text is a genuinely unusable palette.
        state.warnings.append(
            "No colour pair in this palette meets WCAG AA for text -- the brief's "
            "palette may be too low-contrast to build readable UI from."
        )


def _find_reusable_root(nodes: list[dict], width: int) -> dict | None:
    """Pick the frame a previous run built, so we extend it instead of duplicating.

    Only auto-layout FRAMEs qualify: that's what the harness creates, and it's
    the only shape a section can be safely appended into.
    """
    candidates = [
        n
        for n in nodes
        if n.get("type") == "FRAME" and n.get("layoutMode") in ("VERTICAL", "HORIZONTAL")
    ]
    if not candidates:
        return None
    # Prefer one this harness made (its name always ends with "Page"), then the
    # one matching the requested width, then simply the largest.
    ours = [n for n in candidates if str(n.get("name", "")).endswith("Page")]
    pool = ours or candidates
    same_width = [n for n in pool if abs(int(n.get("width") or 0) - width) <= 1]
    pool = same_width or pool
    return max(pool, key=lambda n: (n.get("width") or 0) * (n.get("height") or 0))


def _free_x(nodes: list[dict]) -> int:
    """X coordinate clear of everything already on the page."""
    if not nodes:
        return 200
    right_edge = max((n.get("x") or 0) + (n.get("width") or 0) for n in nodes)
    return int(right_edge) + NEW_FRAME_GAP


def _root_width(instruction: str) -> int:
    """Honour an explicit pixel width in the instruction (e.g. "1440px wide")."""
    match = re.search(r"(\d{3,4})\s*px", instruction, re.IGNORECASE)
    if match:
        width = int(match.group(1))
        if 200 <= width <= 4000:  # ignore incidental numbers like a 48px icon
            return width
    return DEFAULT_ROOT_WIDTH


def _root_name(instruction: str) -> str:
    """Name the frame after a quoted product name if the instruction has one."""
    match = re.search(r"[\"'“]([^\"'”]{2,40})[\"'”]", instruction)
    return f"{match.group(1)} — Page" if match else "Generated Page"


def run_step(
    step: str,
    state: RunState,
    bridge: Bridge,
    llm: ModelClient,
    max_retries: int,
    index: int = 0,
    total: int = 0,
) -> None:
    """Bounded retry loop for one plan step. Both gates must pass before we advance:
    the structural gate (do the nodes really exist as intended) and, for visual
    steps, the visual gate (does it actually look right) -- CLAUDE.md section 6.
    """
    label = f"Step {index}/{total}" if total else "Step"
    docs = query_docs(step, sources=STEP_DOC_SOURCES)
    # Anything a previous attempt put on the canvas. Scripts are atomic, so an
    # attempt that threw may still have landed earlier scripts -- those nodes
    # are real, and the retry must continue from them rather than build a
    # second copy of everything.
    landed_ids: list[str] = []
    prior_defects: list[str] = []
    prior_error = ""
    # Persisted across attempts on purpose: the repeat guard used to reset with
    # each new conversation, which is how a step produced seven identical
    # footer frames.
    seen_calls: set[tuple[str, str]] = set()

    for attempt in range(max_retries):
        if attempt:
            logger.info(
                "%s: %s (attempt %d/%d)...",
                label,
                "repairing" if landed_ids else "retrying",
                attempt + 1,
                max_retries,
            )
        outcome = converse_step(
            step, docs, state, bridge, llm, label, index,
            prior_node_ids=landed_ids,
            prior_defects=prior_defects,
            prior_error=prior_error,
            seen_calls=seen_calls,
        )
        _remember(landed_ids, outcome.created_node_ids)

        if outcome.ok:
            gate = visual_gate(step, state, bridge, llm, label, outcome.created_node_ids)
            if not gate:
                logger.info("%s: done -- %s", label, outcome.summary or "no summary")
                _accept(step, state, outcome)
                return

            # Last attempt, and only the vision critic is unhappy: keep the
            # section. Geometry defects are facts, but "it could look better"
            # must never demote a real section to a TODO placeholder -- that
            # trades a slightly imperfect section for a visibly empty one.
            if attempt == max_retries - 1 and not gate.geometry:
                logger.info("%s: accepted with %d visual note(s)", label, len(gate.vision))
                state.warnings.append(
                    f"'{_section_label(step)}' kept with unresolved visual notes: "
                    + "; ".join(gate.vision[:3])
                )
                _accept(step, state, outcome)
                return

            # The step ran, but the result is visibly wrong. Keep the nodes it
            # made (they exist) and spend the next attempt fixing them in place.
            state.add_node_ids(outcome.created_node_ids)
            prior_defects, prior_error = gate.all(), ""
            logger.info("%s: visual gate found %d issue(s), correcting...", label, len(gate.all()))
            continue

        # A script threw. Whatever ran before it still landed, so the next
        # attempt repairs from there; only the error goes into the docs.
        state.add_node_ids(outcome.created_node_ids)
        docs = augment_with_error(docs, outcome.summary)
        prior_defects, prior_error = [], outcome.summary
        logger.info(
            "%s failed (attempt %d/%d): %s", label, attempt + 1, max_retries, outcome.summary
        )
    state.mark_failed(step)
    recovered = fallback_for_step(step, state, bridge, label)
    state.record_step_result(
        step,
        StepResult(
            step_description=step,
            ok=False,
            created_node_ids=recovered,
            summary="exhausted retries" + (" (placeholder added)" if recovered else ""),
        ),
    )


_SYSTEM_PROMPT: str | None = None


def build_system_prompt() -> str:
    """System prompt + the inlined Plugin API reference, read once per process."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = system_prompt(knowledge_index.gotchas_text())
    return _SYSTEM_PROMPT


def _accept(step: str, state: RunState, outcome: StepResult) -> None:
    """Record a step as done, and register its section so later steps know."""
    state.record_step_result(step, outcome)
    if _is_section_step(step):
        # Later steps are told this is FINISHED, so they add what's missing
        # instead of building a second copy of it.
        state.record_section(outcome.section_name or _section_label(step))


def _remember(known: list[str], new_ids: list[str]) -> None:
    """Accumulate node ids across attempts at one step, without duplicates."""
    for node_id in new_ids:
        if node_id and node_id not in known:
            known.append(node_id)


@dataclass
class GateResult:
    """What the visual gate found, split by how much authority it has.

    Geometry defects are facts: a 0x0 node renders nothing. Vision defects are
    judgement. They are kept apart because they must be treated differently on
    the final attempt -- see `run_step`.
    """

    geometry: list[str] = field(default_factory=list)
    vision: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        return self.geometry + self.vision

    def __bool__(self) -> bool:
        return bool(self.geometry or self.vision)


def visual_gate(
    step: str,
    state: RunState,
    bridge: Bridge,
    llm: ModelClient,
    label: str,
    created_node_ids: list[str] | None = None,
) -> GateResult:
    """Checkpoint check after a visual step: does the result actually look right?

    Screenshots cost tokens, so this only runs for steps that put something on
    the canvas (section 8: "checkpoints, not every micro-step").

    Judged on THIS step's nodes only. Gating on the whole page meant one
    unfixable defect early in the run failed every later step, each of which
    then spent its full retry budget on someone else's problem. The whole page
    is still reviewed once, at the end, by `final_layout_review`.
    """
    if not state.visual_gate_enabled or not state.root_frame_id or not _is_section_step(step):
        return []

    scope = [i for i in (created_node_ids or []) if not _is_style_id(i)]
    if not scope:
        return GateResult()  # nothing of this step's own to judge

    tree = read_layout(state.root_frame_id, bridge)
    if tree is None:
        return GateResult()

    geometry = [str(d) for d in critic.find_layout_defects(tree, scope_ids=scope)]
    # Judge the section that was just built, not the whole page.
    subtree = critic.find_subtrees(tree, set(scope))
    vision = critique_with_model(state, bridge, llm, subtree[0] if subtree else tree, scope[0])

    for defect in geometry[:5]:
        logger.info("%s: defect %s", label, defect)
    for defect in vision[:5]:
        logger.info("%s: defect (vision) %s", label, defect)
    return GateResult(geometry=geometry, vision=vision)


def read_layout(node_id: str, bridge: Bridge) -> dict | None:
    """Batch-read the whole subtree's geometry in one round trip."""
    result = execute_figma_js(bridge, critic.layout_script(node_id))
    if not result["ok"]:
        return None
    return (result.get("result") or {}).get("tree")


def critique_with_model(
    state: RunState, bridge: Bridge, llm: ModelClient, tree: dict, node_id: str
) -> list[str]:
    """Ask the vision critic what is visibly wrong with the section just built.

    Returns only BLOCKING defects. Minor ones are collected for the final
    report but never fail a step -- a vision model always has something to say
    about a work in progress, and gating on all of it would fail every step.
    """
    critic_llm = state.critic_llm or (llm if state.model_sees_images else None)
    if critic_llm is None or state.model_sees_images is False:
        return []

    shot = get_screenshot(bridge, node_id)
    screenshot = shot["image_base64"] if shot["ok"] else None
    if not screenshot:
        return []

    messages = critic.build_critique_messages(
        json.dumps(tree)[:4000], screenshot, tree.get("name", "")
    )
    try:
        reply = critic_llm.complete(messages, tools=None)
    except Exception as exc:
        # A text-only endpoint rejects image content outright -- note it and
        # never pay for the attempt again this run.
        state.model_sees_images = False
        logger.info("Visual critique unavailable (model cannot see images): %s", type(exc).__name__)
        state.warnings.append(
            "Screenshot critique skipped: the configured critic cannot accept images. "
            "Layout was checked geometrically instead."
        )
        return []

    state.model_sees_images = True
    defects = critic.parse_critique(getattr(reply, "content", "") or "")
    for defect in defects:
        if defect.severity == "minor":
            state.minor_notes.append(str(defect))
    return critic.blocking_only(defects)


def fallback_for_step(step: str, state: RunState, bridge: Bridge, label: str) -> list[str]:
    """Last resort after every retry failed: keep the page coherent.

    For a step that was meant to add a section, drop a labelled placeholder in
    the right position so the layout still reads as a page and the gap is
    visible in the file. Steps that build tokens or components get nothing --
    a fake component is worse than none, and tokens already exist from
    `bootstrap_tokens`.
    """
    if not state.root_frame_id or not _is_section_step(step):
        return []
    script = scaffold.build_placeholder_section_script(state.root_frame_id, _section_label(step))
    result = execute_figma_js(bridge, script)
    if not result["ok"]:
        logger.info("%s: placeholder fallback also failed -- %s", label, result["error"])
        return []
    ids = _normalize_node_ids((result.get("result") or {}).get("createdNodeIds"))
    if ids:
        state.add_node_ids(ids)
        # It occupies the slot, so later steps must know it is there.
        state.record_section(f"TODO — {_section_label(step)}")
        logger.info("%s: added a TODO placeholder so the page keeps its structure", label)
    return ids


# Words that mean "this step puts a block on the page", as opposed to creating
# tokens or components off-canvas.
_SECTION_WORDS = ("section", "append", "add ", "compose", "build the", "place")


def _is_section_step(step: str) -> bool:
    lowered = step.lower()
    if any(w in lowered for w in ("color style", "colour style", "text style", "token", "variable")):
        return False
    return any(w in lowered for w in _SECTION_WORDS)


def _section_label(step: str) -> str:
    """Turn a plan step into a short placeholder label."""
    cleaned = re.sub(
        r"^(create|add|append|build|compose)\s+(the\s+)?", "", step.strip(), flags=re.IGNORECASE
    )
    cleaned = re.split(r"[:(]| into | to root| with ", cleaned, maxsplit=1)[0]
    return cleaned.strip(" .,-") or step[:40]


def converse_step(
    step: str,
    docs: str,
    state: RunState,
    bridge: Bridge,
    llm: ModelClient,
    label: str = "Step",
    index: int = 0,
    prior_node_ids: list[str] | None = None,
    prior_defects: list[str] | None = None,
    prior_error: str = "",
    seen_calls: set[tuple[str, str]] | None = None,
) -> StepResult:
    """Run one step's tool-calling conversation until the model finishes or a script fails.

    The step description alone is deliberately tiny, so the design brief and
    the surrounding plan travel with it -- otherwise the component making every
    visual decision is the only one that never sees the design.

    `prior_node_ids` turns this into a repair attempt rather than a fresh
    build, and `seen_calls` is owned by the caller so the repeat guard survives
    across attempts at the same step.
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {
            "role": "user",
            "content": step_user_message(
                step,
                docs,
                state.recent_summary(),
                state.root_frame_id,
                state.existing_sections,
                state.token_names,
                state.text_style_names,
                state.font_styles,
                instruction=state.instruction,
                brief=state.enhanced_brief,
                plan=state.plan,
                step_index=index,
                prior_node_ids=prior_node_ids,
                prior_defects=prior_defects,
                prior_error=prior_error,
                palette_info=state.palette_info,
                pairings=state.readable_pairings,
            ),
        },
    ]

    created_ids: list[str] = []
    section_name = ""  # real Figma name of the first node this step created
    ran_a_script = False
    if seen_calls is None:
        seen_calls = set()
    # A step gets to append ONE section to the root frame. A live dashboard run
    # built the nav bar three times inside a single attempt -- three near-identical
    # scripts, each slightly reworded so the identical-call guard let them through.
    # In repair mode the section already exists, so the budget starts used up.
    root_appends = 1 if prior_node_ids else 0
    for turn in range(1, MAX_TOOL_TURNS_PER_STEP + 1):
        logger.info("%s: thinking (turn %d/%d)...", label, turn, MAX_TOOL_TURNS_PER_STEP)
        message = llm.complete(messages, tools=TOOL_SCHEMAS)
        messages.append(_assistant_message_dict(message))

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            if not ran_a_script:
                # The model stopped without ever running a script -- usually it
                # printed the script as chat text instead of calling the tool.
                # Nothing reached the canvas, so this is NOT a completed step.
                logger.info("%s: stopped without running any script", label)
                return StepResult(
                    step,
                    ok=False,
                    created_node_ids=created_ids,
                    summary=(
                        "You replied with text instead of calling the tool, so nothing ran. "
                        "Call execute_figma_js as a real tool call, passing the script as `code`."
                    ),
                )
            # A script did run -- the model considers the step done.
            return StepResult(
                step,
                ok=True,
                created_node_ids=created_ids,
                summary=message.content or "done",
                section_name=section_name,
            )

        for call in tool_calls:
            args = _safe_json_loads(call.function.arguments)
            logger.info("%s: -> %s", label, _describe_call(call.function.name, args))

            # Identical repeated calls are always a stuck loop, never intent:
            # one live step ran the same createPaintStyle script 8 times, and
            # another created 7 duplicate footer frames. Refuse the repeat and
            # say so, rather than silently doing the work again.
            signature = (call.function.name, call.function.arguments or "")
            if signature in seen_calls:
                logger.info("%s: refused a repeated identical call", label)
                messages.append(
                    _tool_result_message(
                        call.id,
                        {
                            "ok": False,
                            "error": (
                                "You already ran this exact call in this step and it was handled. "
                                "Repeating it would duplicate work. Either move on to the next part "
                                "of the step with a DIFFERENT script, or stop and finish the step."
                            ),
                        },
                    )
                )
                continue
            seen_calls.add(signature)

            # One section per step. Reworded near-duplicates slip past the
            # identical-call guard, so cap the thing that actually matters:
            # how many times this step appends a new block to the page.
            appends_to_root = _appends_to_root(call.function.name, args, state.root_frame_id)
            if appends_to_root and root_appends >= 1:
                logger.info("%s: refused a second append into the root frame", label)
                messages.append(
                    _tool_result_message(call.id, {"ok": False, "error": _SECOND_APPEND_REFUSAL})
                )
                continue

            try:
                result = dispatch(
                    call.function.name, args, bridge,
                    render_context={
                        "parent_id": state.root_frame_id,
                        "color_roles": renderer.role_map(state.palette_info),
                    },
                )
            except Exception as exc:
                # One malformed tool call must not kill the run -- hand the
                # model the error and let it correct itself next turn.
                logger.info("%s: tool call failed -- %s", label, exc)
                result = {"ok": False, "error": f"Tool call failed: {exc}"}

            if call.function.name not in ("execute_figma_js", "render_ui"):
                messages.append(_tool_result_message(call.id, result))
                continue
            if not result["ok"]:
                messages.append(_tool_result_message(call.id, result))
                if result.get("recoverable"):
                    # Rejected before it reached Figma (a malformed UI spec).
                    # Nothing was attempted, so let the model fix its JSON on
                    # the next turn instead of spending a whole retry.
                    logger.info("%s: spec rejected -- %s", label, result["error"])
                    continue
                # A script failed: the document is unchanged (scripts are atomic).
                # Bubble the error up so the retry loop can feed it back as docs.
                logger.info("%s: script failed -- %s", label, result["error"])
                return StepResult(step, ok=False, created_node_ids=created_ids, summary=str(result["error"]))

            ran_a_script = True
            # Only a call that actually landed spends the one-section budget.
            # A rejected spec or a thrown script changes nothing (Figma scripts
            # are atomic), so it must not lock the step out of building at all.
            if appends_to_root:
                root_appends += 1
            new_ids = _normalize_node_ids((result.get("result") or {}).get("createdNodeIds"))
            if new_ids:
                logger.info("%s: created %s", label, ", ".join(new_ids))
            created_ids.extend(new_ids)
            name = validate_creation(new_ids, state, bridge)
            if name and not section_name:
                section_name = name

            # Say plainly that the work landed. Without this the model reads the
            # unchanged "Current step: Add the ... section" and builds it again.
            if created_ids:
                result = dict(result)
                result["note"] = _existing_nodes_note(created_ids, section_name)
            messages.append(_tool_result_message(call.id, result))

    return StepResult(
        step, ok=False, created_node_ids=created_ids, summary="step did not conclude within the tool-call budget"
    )


_SECOND_APPEND_REFUSAL = (
    "REFUSED: this step has already appended its section to the root frame, and that "
    "section is on the canvas now. Running this would put a SECOND copy of it on the "
    "page. If the section is finished, stop and reply with a one-line summary and no "
    "tool call. If something inside it is missing or wrong, append to or modify the "
    "nodes you already created (use their ids) -- never the root frame again."
)


def _appends_to_root(tool_name: str, args: dict, root_id: str | None) -> bool:
    # render_ui always appends its tree to the root frame, so it counts.
    if tool_name == "render_ui":
        return bool(root_id)
    """Does this script add a new block directly to the page's root frame?

    Deliberately narrow: a script that parents into a section the step already
    created is normal and must not be blocked. Only the root frame id combined
    with an append is treated as "starting the section over", and the prompt
    hands that id to the model literally, so it appears verbatim in the code.
    """
    if tool_name != "execute_figma_js" or not root_id:
        return False
    code = args.get("code") or ""
    if root_id not in code:
        return False
    return "appendchild" in code.lower() or "insertchild" in code.lower()


def _existing_nodes_note(created_ids: list[str], section_name: str) -> str:
    """Told after every successful script, so 'done' is unambiguous.

    The step description still reads "Add the top navigation bar...", and a
    model that has just built it sees no signal that the job is finished --
    which is how one attempt produced three nav bars.
    """
    named = f" ('{section_name}')" if section_name else ""
    return (
        f"These nodes now EXIST on the canvas{named}: {', '.join(created_ids[:20])}. "
        "The work for this step is on the page. Do NOT create or append it again. "
        "If the step is complete, reply with a short summary and NO tool call."
    )


def _describe_call(name: str, args: dict) -> str:
    if name == "execute_figma_js":
        code = (args.get("code") or "").strip().splitlines()
        preview = code[0][:80] if code else ""
        return f"execute_figma_js: {preview}"
    if name == "query_docs":
        return f"query_docs: {args.get('query', '')}"
    if name in ("get_metadata", "get_screenshot"):
        node = args.get("node_id") or "current page"
        return f"{name}({node})"
    return name


def _normalize_node_ids(raw) -> list[str]:
    """Clean up whatever the model's script actually returned.

    Scripts have returned ids with trailing commas ("1:2,") and occasionally a
    comma-joined string instead of a list, which then fails every later
    lookup. Normalize here rather than trusting generated JS to be tidy.
    """
    if not raw:
        return []
    items = raw.split(",") if isinstance(raw, str) else raw
    cleaned = []
    for item in items:
        if not isinstance(item, str):
            continue
        node_id = item.strip().strip(",").strip()
        if node_id:
            cleaned.append(node_id)
    return cleaned


def _is_style_id(node_id: str) -> bool:
    """Style ids (S:...) are not nodes -- getNodeByIdAsync never finds them."""
    return node_id.startswith("S:")


def validate_creation(new_ids: list[str], state: RunState, bridge: Bridge) -> str:
    """Structural check: read back the first touched node to confirm it really exists.

    Returns the node's real Figma name (the read is already happening, so the
    name is free) so the loop can tell later steps what is on the page. Visual
    correctness is left to the visual gate.
    """
    checkable = [i for i in new_ids if not _is_style_id(i)]
    if not checkable:
        return ""  # styles-only step: nothing node-shaped to verify
    check = get_metadata(bridge, checkable[0])
    if not check["ok"]:
        state.warnings.append(f"Could not verify node {checkable[0]}: {check['error']}")
        return ""
    return str((check.get("result") or {}).get("name") or "")


# Errors the model has repeatedly failed to self-correct from, paired with the
# exact fix. Retrieval alone didn't reach these -- the error text is the most
# reliable trigger we have, so map it straight to the correction.
ERROR_HINTS: list[tuple[str, str]] = [
    (
        "can only be set on children of auto-layout",
        "FIX: `layoutSizingHorizontal`/`layoutSizingVertical` are only legal once the node is "
        "ALREADY appended to a parent whose layoutMode is 'VERTICAL' or 'HORIZONTAL'. Order: "
        "create -> resize -> parent.appendChild(node) -> THEN set layoutSizing*. If you do not "
        "control the parent, drop FILL/HUG entirely and just use resize(w, h). NOTE: a node "
        "from `figma.createComponent()` or `figma.createFrame()` is NOT auto-layout until you "
        "set `node.layoutMode = 'VERTICAL'` on it -- so its children cannot use FILL/HUG "
        "before you do.",
    ),
    (
        "HUG can only be set on",
        "FIX: HUG is only valid on an auto-layout frame itself or on a TEXT child of one. Set "
        "`node.layoutMode` first, or use `primaryAxisSizingMode = 'AUTO'` instead.",
    ),
    (
        "could not be loaded",
        "FIX: that font/style string does not exist. Inter's styles are 'Regular', 'Medium', "
        "'Semi Bold' (WITH a space), 'Bold'. Use one of those exact strings, and load the font "
        "in the SAME script that uses it.",
    ),
    (
        "unexpected token in expression: '?'",
        "FIX: this sandbox does not support optional chaining (`?.`) or nullish coalescing "
        "(`??`). Write explicit checks instead: `a && a.b ? a.b : fallback`.",
    ),
    (
        'Property "lineHeight" failed validation',
        "FIX: lineHeight is an object, not a number: `{ unit: 'PIXELS', value: 56 }` "
        "(or `{ unit: 'PERCENT', value: 140 }`). Same for letterSpacing.",
    ),
    (
        "without calling figma.loadAllPagesAsync",
        "FIX: do not search the document. Use `await figma.getNodeByIdAsync(id)` with an id you "
        "already have, or `figma.currentPage.children` -- never `figma.root.findOne`.",
    ),
    (
        "not a function",
        "FIX: you called an API that does not exist. The usual causes, in order: (1) you "
        "added 'Async' to a CREATOR -- `createPaintStyleAsync`, `createTextStyleAsync`, "
        "`createVariableAsync` and `createVariableCollectionAsync` are all invented; "
        "creators are SYNC. (2) `figma.createInstance(component)` -- it is "
        "`component.createInstance()`. (3) `node.setFillStyleId(id)` -- it is "
        "`await node.setFillStyleIdAsync(id)`. (4) `figma.variables.getVariableByName` / "
        "`figma.getLocalVariableByName` do not exist -- list with "
        "`getLocalVariablesAsync()` and filter. Re-read the reference and use the exact "
        "name; do not guess a variant of it.",
    ),
    (
        "would create a component inside a component",
        "FIX: a COMPONENT cannot be nested inside another COMPONENT. You are appending a "
        "component (or an instance's source) into a component. Build the child as a plain "
        "FRAME instead, or append `component.createInstance()` rather than the component "
        "itself. For a static mockup you rarely need components at all -- a plain frame is "
        "always safe.",
    ),
    (
        "createComponentSet",
        "FIX: `figma.createComponentSet()` does not exist. Build the variants as separate "
        "COMPONENT nodes, then combine them: `figma.combineAsVariants([a, b], parent)`. For a "
        "static mockup, a single component without variants is usually enough -- hover/active "
        "states are not required.",
    ),
]


def augment_with_error(docs: str, error: str) -> str:
    """Feed the failure back as context for the next attempt, with a targeted
    fix when we recognise the error."""
    parts = [f"### Previous attempt failed\n{error}"]
    for needle, hint in ERROR_HINTS:
        if needle.lower() in (error or "").lower():
            parts.append(hint)
            break
    addendum = "\n\n".join(parts)
    return f"{docs}\n\n{addendum}" if docs else addendum


def final_validation(state: RunState, bridge: Bridge) -> None:
    """Whole-screen review. Fix what is mechanically fixable, then report the rest."""
    enforce_root_hug(state, bridge)          # fix clipping BEFORE judging the layout
    audit_variable_bindings(state, bridge)   # rebind hardcoded colours to tokens

    shot = get_screenshot(bridge, state.root_frame_id)
    if shot["ok"]:
        state.final_screenshot_base64 = shot["image_base64"]
    else:
        state.warnings.append(f"Final screenshot failed: {shot['error']}")

    final_layout_review(state, bridge)


def final_layout_review(state: RunState, bridge: Bridge) -> None:
    """Report what's visibly wrong with the finished screen, rather than
    leaving the user to spot it. Deterministic, so it works with any model."""
    if not state.root_frame_id:
        return
    tree = read_layout(state.root_frame_id, bridge)
    if tree is None:
        return
    defects = critic.find_layout_defects(tree)
    state.layout_defects = [str(d) for d in defects]
    if defects:
        logger.info("Final review found %d layout issue(s).", len(defects))
        for defect in defects[:8]:
            logger.info("  %s", defect)
        state.warnings.append(
            f"{len(defects)} layout issue(s) in the finished screen: "
            + "; ".join(str(d) for d in defects[:5])
            + ("; ..." if len(defects) > 5 else "")
        )
    else:
        logger.info("Final review: layout is CLEAN.")


def audit_variable_bindings(state: RunState, bridge: Bridge) -> None:
    """Make hardcoded fills dynamic, then report only what's genuinely left.

    Golden rule 5 says never hardcode colours -- but *warning* about it does
    not make a design dynamic. So the harness rebinds any fill that matches a
    token (the model meant the token and wrote the hex), and only reports
    colours far enough from every token that changing them would be a design
    decision rather than a fix.
    """
    ids = [i for i in state.created_node_ids if not _is_style_id(i)][:200]  # cap so the script stays small
    if not ids or not state.palette:
        return

    result = execute_figma_js(bridge, scaffold.build_bind_fills_script(ids, state.palette))
    if not result["ok"]:
        state.warnings.append(f"Variable-binding pass failed: {result['error']}")
        return

    payload = result.get("result") or {}
    bound = payload.get("boundNodes") or []
    unbound = payload.get("stillUnbound") or []

    if bound:
        state.bound_node_count = len(bound)
        tokens = sorted({b.get("token", "?") for b in bound if isinstance(b, dict)})
        logger.info("Bound %d node(s) to tokens: %s", len(bound), ", ".join(tokens))
    if unbound:
        logger.info("%d node(s) use a colour that matches no token.", len(unbound))
        state.warnings.append(
            f"{len(unbound)} node(s) use a one-off colour not in the token set "
            f"(left as-is -- add it to the palette if it should be a token)."
        )


def enforce_root_hug(state: RunState, bridge: Bridge) -> None:
    """Restore the root frame's hug sizing before reviewing the layout.

    `resize()` resets sizing to FIXED, after which the frame clips everything
    below the fold and every stacked section reads as an overflow defect.
    """
    if not state.root_frame_id:
        return
    result = execute_figma_js(bridge, scaffold.build_hug_fix_script(state.root_frame_id))
    if result["ok"] and (result.get("result") or {}).get("changed"):
        logger.info(
            "Root frame was clipping its content; restored hug sizing (now %spx tall).",
            (result.get("result") or {}).get("height"),
        )


_BINDING_AUDIT_SCRIPT = """\
const ids = {ids};
const unbound = [];
for (const id of ids) {{
  const node = await figma.getNodeByIdAsync(id);
  if (!node || !('fills' in node)) continue;
  // A node using one of our paint styles is token-backed by definition --
  // those styles are themselves bound to variables in bootstrap_tokens.
  if ('fillStyleId' in node && node.fillStyleId) continue;
  const fills = node.fills;
  if (!Array.isArray(fills)) continue;
  for (const fill of fills) {{
    if (fill.type === 'SOLID' && !(fill.boundVariables && fill.boundVariables.color)) {{
      unbound.push(id);
      break;
    }}
  }}
}}
return {{ createdNodeIds: [], unboundFillNodeIds: unbound }};
"""


def _assistant_message_dict(message) -> dict:
    d: dict = {"role": "assistant", "content": message.content}
    if getattr(message, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return d


def _tool_result_message(tool_call_id: str, result: dict) -> dict:
    # Strip large binary payloads (screenshots) before feeding results back as text.
    trimmed = {k: v for k, v in result.items() if k != "image_base64"}
    if result.get("image_base64"):
        trimmed["image_base64"] = "(omitted -- binary)"
    return {"role": "tool", "tool_call_id": tool_call_id, "content": json.dumps(trimmed)}


def _safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
