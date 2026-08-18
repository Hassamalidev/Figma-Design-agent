"""The visual gate (CLAUDE.md section 8).

Two complementary checks, because they catch different bugs:

1. `find_layout_defects` -- deterministic geometry analysis in Python. Reads
   the real node tree and looks for the specific things that make a generated
   screen look broken: collapsed text, children overflowing their parent,
   overlapping siblings, empty frames, invisible text. No model, no tokens,
   works with a text-only model, and never hallucinates a defect.

2. `critique_screenshot` -- shows the model the rendered PNG and asks for
   concrete defects. Only a multimodal model can do this; we detect that
   automatically and skip it otherwise rather than pretending.

Arithmetic lives here rather than in the prompt (section 7: "do arithmetic in
Python, not the model") -- overlap and overflow are geometry, not judgement.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from agent import renderer, scaffold

logger = logging.getLogger(__name__)

# Reads the subtree with the geometry needed to reason about layout, PLUS the
# properties the design-system checks need: the resolved fill, whether that
# fill is token-backed, and the auto-layout spacing values. Reading them here
# costs nothing extra -- it is the same single round trip -- and it is what
# lets contrast, spacing scale and type ramp be checked as arithmetic instead
# of being asserted in a prompt and hoped for.
LAYOUT_SCRIPT = """const root = await figma.getNodeByIdAsync({node_id});
if (!root) {{ throw new Error('node not found'); }}

// The first opaque SOLID fill, as plain numbers. Figma hands back a read-only
// array, and `fills` is the `mixed` SYMBOL on a node with differing fills --
// neither is safe to index without checking.
function solidFill(node) {{
  if (!('fills' in node)) {{ return null; }}
  const fills = node.fills;
  if (!Array.isArray(fills)) {{ return null; }}
  for (const paint of fills) {{
    if (!paint || paint.type !== 'SOLID' || paint.visible === false) {{ continue; }}
    const opacity = typeof paint.opacity === 'number' ? paint.opacity : 1;
    if (opacity < 0.9) {{ continue; }}   // see-through: not the effective colour
    return {{ r: paint.color.r, g: paint.color.g, b: paint.color.b }};
  }}
  return null;
}}

// Token-backed means "changing the token changes this node": either it uses one
// of our paint styles, or its paint is bound to a variable directly.
function tokenBacked(node) {{
  if ('fillStyleId' in node && typeof node.fillStyleId === 'string' && node.fillStyleId) {{
    return true;
  }}
  if (!('fills' in node)) {{ return false; }}
  const fills = node.fills;
  if (!Array.isArray(fills)) {{ return false; }}
  for (const paint of fills) {{
    if (paint && paint.boundVariables && paint.boundVariables.color) {{ return true; }}
  }}
  return false;
}}

function describe(node, depth) {{
  const info = {{
    id: node.id, name: node.name, type: node.type,
    x: Math.round(node.x), y: Math.round(node.y),
    width: Math.round(node.width), height: Math.round(node.height),
    visible: node.visible !== false,
    layoutMode: 'layoutMode' in node ? node.layoutMode : null,
    fill: solidFill(node),
    tokenBacked: tokenBacked(node),
    children: []
  }};
  if (info.layoutMode && info.layoutMode !== 'NONE') {{
    info.itemSpacing = Math.round(node.itemSpacing);
    info.padding = [
      Math.round(node.paddingTop), Math.round(node.paddingRight),
      Math.round(node.paddingBottom), Math.round(node.paddingLeft)
    ];
  }}
  if (node.type === 'TEXT') {{
    info.characters = String(node.characters).slice(0, 120);
    info.fontSize = typeof node.fontSize === 'number' ? node.fontSize : null;
    info.fontStyle = (node.fontName && node.fontName.style) ? String(node.fontName.style) : '';
  }}
  if ('children' in node && depth < 4) {{
    info.children = node.children.slice(0, 40).map(function (c) {{ return describe(c, depth + 1); }});
  }}
  return info;
}}
return {{ createdNodeIds: [], tree: describe(root, 0) }};
"""


@dataclass
class Defect:
    node_id: str
    node_name: str
    kind: str
    detail: str
    # Advisory defects are REPORTED but never fail a step. The split matters:
    # a 0x0 node renders nothing (fact, worth a retry), while "this 20px gap is
    # off the 8px scale" is a polish note. Gating on the latter would burn a
    # step's whole retry budget on something no user would call broken --
    # exactly the failure mode CLAUDE.md section 8b warns about for the vision
    # critic, arrived at from the other direction.
    advisory: bool = False

    def __str__(self) -> str:
        return f"[{self.kind}] {self.node_name}: {self.detail}"


# A couple of pixels of overlap is normal from rounding; this is the threshold
# at which it reads as a real visual bug.
OVERLAP_TOLERANCE = 2
OVERFLOW_TOLERANCE = 2

# ---- design-system thresholds --------------------------------------------
#
# CLAUDE.md computes contrast to build the PROMPT (the "readable pairings"
# handed to the builder) but never checked the result, so a model that ignored
# the list produced invisible copy that passed every gate. These are the same
# numbers, applied to what actually landed on the canvas.

# WCAG AA. Large text is a lower bar because size compensates for contrast.
NORMAL_TEXT_AA = 4.5
LARGE_TEXT_AA = 3.0
LARGE_TEXT_PX = 24            # or 18.66px when bold -- the WCAG definition
LARGE_TEXT_BOLD_PX = 18.66

# Below AA is a defect; below THIS is unreadable. Only the unreadable band
# blocks a step. 4.4:1 is a real AA failure and worth reporting, but failing a
# section over it -- and eventually replacing it with a TODO placeholder --
# would be a worse design outcome than shipping it.
NORMAL_TEXT_BROKEN = 3.0
LARGE_TEXT_BROKEN = 2.0

# The spacing scale the harness itself builds to (agent/renderer.SPACING),
# widened with the intermediate 8px-grid values a hand-written script may
# legitimately use. Anything outside this is what makes a generated page feel
# arrhythmic even when nothing overlaps.
SPACING_SCALE = frozenset(renderer.SPACING.values()) | {12, 20, 40, 56, 64, 80, 96, 120}

# The type ramp bootstrap_tokens actually creates. A size outside it means the
# text style was ignored in favour of an ad-hoc fontSize.
TYPE_RAMP = frozenset(size for _, _, size, _ in scaffold.TEXT_STYLES)

# Below this a frame is an icon/divider, where a one-off colour is normal and
# flagging it would be pure noise.
MIN_TOKEN_AUDIT_AREA = 24 * 24


# A long defect list is unreadable and, worse, unactionable. Repeats of the
# same problem on identically-named nodes are collapsed.
MAX_DEFECTS = 12
MAX_PER_KIND = 3


def find_layout_defects(tree: dict, scope_ids: list[str] | None = None) -> list[Defect]:
    """Walk the node tree and report concrete, checkable layout problems.

    Deduplicated and capped: a live run produced 28 defects that were all the
    same underlying issue, which buries the signal.

    `scope_ids` narrows the walk to the subtrees a single step actually
    touched. Without it, a defect left behind by step 2 fails the gate for
    step 7 -- which then burns its retries on a problem it did not cause and
    cannot see. The whole-tree form (no scope) is what the final review uses.
    """
    return _condense([d for d in analyze(tree, scope_ids) if not d.advisory])


def find_design_defects(tree: dict, scope_ids: list[str] | None = None) -> list[Defect]:
    """Design-system adherence: contrast, spacing scale, type ramp, tokens.

    Everything here is advisory -- reported in the run summary, never used to
    fail a step. These are the rules CLAUDE.md states as prose ("tokens ->
    components -> composition", "never hardcode a colour"), measured against
    what actually landed rather than asserted in a prompt and hoped for.
    """
    return _condense([d for d in analyze(tree, scope_ids) if d.advisory])


def analyze(tree: dict, scope_ids: list[str] | None = None) -> list[Defect]:
    """Every defect in ONE walk, blocking and advisory together.

    One traversal rather than two: the checks share the work of resolving each
    node's inherited background, which is the only genuinely awkward part.
    """
    roots = [tree] if not scope_ids else find_subtrees(tree, set(scope_ids))
    found: list[Defect] = []
    for root in roots:
        _walk(root, found, None)
    return found


def find_subtrees(tree: dict, wanted: set[str]) -> list[dict]:
    """The outermost subtrees for `wanted`, so a step is judged on its own work.

    Stops descending once a node matches: its children are part of that same
    subtree and would otherwise be walked twice. An id that isn't in the tree
    contributes nothing -- an empty result means "nothing of this step's to
    judge", which must not fall back to judging the whole page.
    """
    if tree.get("id") in wanted:
        return [tree]
    found: list[dict] = []
    for child in tree.get("children") or []:
        found.extend(find_subtrees(child, wanted))
    return found


def _condense(defects: list[Defect]) -> list[Defect]:
    seen: set[tuple[str, str]] = set()
    per_kind: dict[str, int] = {}
    result: list[Defect] = []
    for defect in defects:
        key = (defect.kind, defect.node_name)
        if key in seen:
            continue
        seen.add(key)
        count = per_kind.get(defect.kind, 0)
        if count >= MAX_PER_KIND:
            continue
        per_kind[defect.kind] = count + 1
        result.append(defect)
        if len(result) >= MAX_DEFECTS:
            break
    return result


def _walk(node: dict, defects: list[Defect], background: tuple | None) -> None:
    children = node.get("children") or []
    _check_self(node, defects, background)
    _check_design(node, defects)

    if children:
        _check_children_fit(node, children, defects)
        # Overlap only means something when the parent isn't laying children
        # out itself -- auto-layout cannot produce overlapping siblings.
        if node.get("layoutMode") in (None, "NONE"):
            _check_overlaps(children, defects)

    # A node with no fill of its own shows whatever is behind it, so the
    # background a text node sits on is the nearest FILLED ancestor -- not
    # necessarily its direct parent.
    inherited = fill_rgb(node) or background
    for child in children:
        _walk(child, defects, inherited)


def fill_rgb(node: dict) -> tuple[float, float, float] | None:
    """The node's own opaque solid fill as 0-1 RGB, or None if it has none."""
    fill = node.get("fill")
    if not isinstance(fill, dict):
        return None
    try:
        return (float(fill["r"]), float(fill["g"]), float(fill["b"]))
    except (KeyError, TypeError, ValueError):
        return None


def _check_self(node: dict, defects: list[Defect], background: tuple | None = None) -> None:
    name = node.get("name", "?")
    node_id = node.get("id", "?")
    width = node.get("width") or 0
    height = node.get("height") or 0

    if node.get("visible") is False:
        defects.append(Defect(node_id, name, "invisible", "node is hidden"))
        return

    if width <= 0 or height <= 0:
        defects.append(
            Defect(node_id, name, "collapsed", f"has no area ({width}x{height}) so nothing renders")
        )
        return

    if node.get("type") == "TEXT":
        text = (node.get("characters") or "").strip()
        if not text:
            defects.append(Defect(node_id, name, "empty-text", "text node has no characters"))
        font_size = node.get("fontSize")
        if font_size and height + 1 < font_size:
            defects.append(
                Defect(node_id, name, "clipped-text",
                       f"height {height}px is smaller than its {font_size}px font, so text is cut off")
            )
        if width < 8 and text:
            defects.append(
                Defect(node_id, name, "collapsed-text",
                       f"only {width}px wide -- set textAutoResize and an explicit width")
            )
        # A box far narrower than its own font cannot fit characters side by
        # side, so Figma stacks them ONE PER LINE. It renders as a vertical
        # column of letters and is unmissable on the canvas -- but it slipped
        # past every check, because 20px is not "collapsed" and the node's
        # height is large, not small.
        elif (
            font_size
            and len(text) > 3
            and width < font_size * 3
            and height > font_size * 2.5
        ):
            defects.append(
                Defect(node_id, name, "vertical-text",
                       f"{width}x{height}px at {font_size}px font -- too narrow for its text, so "
                       f"it is wrapping one character per line. Set an explicit width wide enough "
                       f"for the content (or layoutSizingHorizontal='FILL' inside an auto-layout parent)")
            )

        _check_contrast(node, defects, background)

    if node.get("type") == "FRAME" and not (node.get("children") or []):
        if width > 40 and height > 40:
            defects.append(
                Defect(node_id, name, "empty-frame",
                       f"{width}x{height} frame is empty, leaving a blank region")
            )


def _check_contrast(node: dict, defects: list[Defect], background: tuple | None) -> None:
    """Is this text actually readable against what is behind it?

    Geometry cannot see contrast: text bound to the wrong token is the right
    size, in the right place, and completely invisible -- and every existing
    check passes it. The palette's legal pairings are already MEASURED for the
    prompt (scaffold.readable_pairings); this measures the result.

    Silent when either colour is unknown. Text over an image, a gradient or a
    frame with no fill anywhere above it has no single background colour, and
    inventing one would produce confident nonsense.
    """
    foreground = fill_rgb(node)
    if foreground is None or background is None:
        return
    if not (node.get("characters") or "").strip():
        return  # empty text is already reported as its own defect

    ratio = scaffold.contrast_ratio_rgb(foreground, background)
    size = node.get("fontSize") or 0
    bold = "bold" in str(node.get("fontStyle") or "").lower()
    large = size >= LARGE_TEXT_PX or (bold and size >= LARGE_TEXT_BOLD_PX)

    required = LARGE_TEXT_AA if large else NORMAL_TEXT_AA
    if ratio >= required:
        return

    unreadable = LARGE_TEXT_BROKEN if large else NORMAL_TEXT_BROKEN
    if ratio < unreadable:
        defects.append(
            Defect(
                node.get("id", "?"), node.get("name", "?"), "contrast",
                f"text is {ratio}:1 against its background -- effectively invisible "
                f"(WCAG AA needs {required}:1). Bind its fill to one of the colour "
                f"pairings listed as readable, not to whichever token sounds right",
            )
        )
    else:
        defects.append(
            Defect(
                node.get("id", "?"), node.get("name", "?"), "contrast-aa",
                f"text is {ratio}:1 against its background, below the {required}:1 "
                f"WCAG AA minimum -- legible but not accessible",
                advisory=True,
            )
        )


def _check_design(node: dict, defects: list[Defect]) -> None:
    """Spacing scale, type ramp and token backing -- the rules that were prose.

    All three are arithmetic over values already in the tree, so they cost
    nothing and cannot hallucinate. All three are advisory: a page that is
    correct but 4px off the scale is still a working page.
    """
    node_id, name = node.get("id", "?"), node.get("name", "?")

    if node.get("layoutMode") in ("VERTICAL", "HORIZONTAL"):
        off_scale = sorted(
            {v for v in _spacing_values(node) if v not in SPACING_SCALE}
        )
        if off_scale:
            defects.append(
                Defect(
                    node_id, name, "off-scale-spacing",
                    f"uses spacing {', '.join(f'{v}px' for v in off_scale[:4])} which is "
                    f"not on the 8px scale -- the layout reads as arrhythmic next to "
                    f"sections that are",
                    advisory=True,
                )
            )

    if node.get("type") == "TEXT":
        size = node.get("fontSize")
        if size and int(size) not in TYPE_RAMP:
            defects.append(
                Defect(
                    node_id, name, "off-ramp-type",
                    f"{int(size)}px is not on the type ramp "
                    f"({', '.join(str(s) for s in sorted(TYPE_RAMP))}) -- a text style "
                    f"was set aside in favour of an ad-hoc size",
                    advisory=True,
                )
            )

    # Golden rule 5: never hardcode a colour into a final node. The harness
    # rebinds anything close to a token in `audit_variable_bindings`, so what
    # survives to here is genuinely off-palette.
    if node.get("fill") and not node.get("tokenBacked"):
        area = (node.get("width") or 0) * (node.get("height") or 0)
        if area >= MIN_TOKEN_AUDIT_AREA:
            defects.append(
                Defect(
                    node_id, name, "untokenised-fill",
                    "has a hardcoded colour that matches no token, so changing the "
                    "palette will not change it",
                    advisory=True,
                )
            )


def _spacing_values(node: dict) -> list[int]:
    """Auto-layout gaps and paddings, as plain ints. Zero is always on-scale."""
    values: list[int] = []
    spacing = node.get("itemSpacing")
    # Only meaningful once there is a gap to space: a single child never shows it.
    if isinstance(spacing, (int, float)) and len(node.get("children") or []) > 1:
        values.append(int(spacing))
    padding = node.get("padding")
    if isinstance(padding, list):
        values.extend(int(v) for v in padding if isinstance(v, (int, float)))
    return [v for v in values if v]


def _check_children_fit(parent: dict, children: list[dict], defects: list[Defect]) -> None:
    """Children are positioned relative to the parent -- flag ones sticking out.

    Auto-layout is handled separately: Figma owns child positions there, so
    per-child overflow is not a per-child bug. A frame whose content exceeds
    it is ONE problem (the frame isn't growing), not one per child -- a live
    run reported the same stacked-content issue 28 times.
    """
    pw = parent.get("width") or 0
    ph = parent.get("height") or 0
    if pw <= 0 or ph <= 0:
        return

    if parent.get("layoutMode") in ("VERTICAL", "HORIZONTAL"):
        _check_autolayout_fits(parent, children, defects, pw, ph)
        return

    for child in children:
        cx, cy = child.get("x") or 0, child.get("y") or 0
        cw, ch = child.get("width") or 0, child.get("height") or 0
        if cx + cw > pw + OVERFLOW_TOLERANCE or cy + ch > ph + OVERFLOW_TOLERANCE or cx < -OVERFLOW_TOLERANCE or cy < -OVERFLOW_TOLERANCE:
            defects.append(
                Defect(
                    child.get("id", "?"), child.get("name", "?"), "overflow",
                    f"extends outside '{parent.get('name', '?')}' "
                    f"({cx},{cy} {cw}x{ch} vs parent {pw}x{ph})",
                )
            )


def _check_autolayout_fits(
    parent: dict, children: list[dict], defects: list[Defect], pw: int, ph: int
) -> None:
    """One defect if an auto-layout frame is too small for its own content."""
    vertical = parent.get("layoutMode") == "VERTICAL"
    extent = max(
        ((c.get("y") or 0) + (c.get("height") or 0)) if vertical
        else ((c.get("x") or 0) + (c.get("width") or 0))
        for c in children
    )
    limit = ph if vertical else pw
    if extent > limit + OVERFLOW_TOLERANCE:
        axis = "taller" if vertical else "wider"
        prop = "primaryAxisSizingMode = 'AUTO'" if vertical else "counterAxisSizingMode = 'AUTO'"
        defects.append(
            Defect(
                parent.get("id", "?"), parent.get("name", "?"), "clipped-content",
                f"content is {extent - limit}px {axis} than the frame "
                f"({extent} vs {limit}) -- the frame is FIXED and clipping its children; "
                f"set {prop} so it hugs, and do not call resize() on it afterwards",
            )
        )


def _check_overlaps(children: list[dict], defects: list[Defect]) -> None:
    visible = [c for c in children if c.get("visible") is not False and (c.get("width") or 0) > 0]
    for i, a in enumerate(visible):
        for b in visible[i + 1 :]:
            overlap = _overlap_area(a, b)
            if overlap is None:
                continue
            ow, oh = overlap
            if ow > OVERLAP_TOLERANCE and oh > OVERLAP_TOLERANCE:
                defects.append(
                    Defect(
                        a.get("id", "?"), a.get("name", "?"), "overlap",
                        f"overlaps '{b.get('name', '?')}' by {ow}x{oh}px",
                    )
                )


def _overlap_area(a: dict, b: dict) -> tuple[int, int] | None:
    ax, ay = a.get("x") or 0, a.get("y") or 0
    bx, by = b.get("x") or 0, b.get("y") or 0
    aw, ah = a.get("width") or 0, a.get("height") or 0
    bw, bh = b.get("width") or 0, b.get("height") or 0
    ow = min(ax + aw, bx + bw) - max(ax, bx)
    oh = min(ay + ah, by + bh) - max(ay, by)
    if ow <= 0 or oh <= 0:
        return None
    return int(ow), int(oh)


# ---- model-driven critique (needs a multimodal model) ---------------------

CRITIQUE_SYSTEM = """\
You are a senior product designer reviewing ONE section of a UI, rendered from \
Figma. You are looking at the real pixels.

Reply with ONLY a JSON array. No prose, no markdown fences. Each item:

  {"severity": "blocking", "element": "<what it is>", "problem": "<one sentence>"}

Return [] if the section reads as acceptable UI. That is a normal answer -- do \
not invent problems to seem useful.

"blocking" means the section is visibly BROKEN. Only these count:
  - text unreadable against its background (too low contrast)
  - elements overlapping or colliding
  - text or content cut off / clipped by its container
  - a large region that is empty when it should hold content
  - content spilling outside its container
  - an element so misaligned it reads as a mistake, not a choice

"minor" is everything else: spacing could be tighter, a colour could be nicer, \
hierarchy could be stronger, it could be more polished. Use "minor" freely -- \
minor items are recorded but never block.

Do NOT report: missing features or content ("it needs a logo"), suggestions, \
praise, or anything you cannot literally SEE in the image. You are judging \
whether what is here is rendered correctly, not whether more should exist.

At most 6 items.
"""


@dataclass
class VisualDefect:
    severity: str  # "blocking" | "minor"
    element: str
    problem: str

    def __str__(self) -> str:
        return f"[visual] {self.element}: {self.problem}"


def build_critique_messages(
    metadata_json: str, screenshot_b64: str | None, section_name: str = ""
) -> list[dict]:
    """Both signals when we have them: metadata is structural ground truth,
    the screenshot is visual ground truth (CLAUDE.md section 8)."""
    target = f" (the '{section_name}' section)" if section_name else ""
    content: list[dict] = [
        {
            "type": "text",
            "text": f"Review this section{target}. Reply with the JSON array only.",
        },
        {"type": "text", "text": f"metadata: {metadata_json}"},
    ]
    if screenshot_b64:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
        )
    return [
        {"role": "system", "content": CRITIQUE_SYSTEM},
        {"role": "user", "content": content},
    ]


MAX_VISUAL_DEFECTS = 6


def parse_critique(reply: str) -> list[VisualDefect]:
    """Parse the critic's JSON array into typed defects.

    Severity is the whole point. The previous version treated EVERY non-"CLEAN"
    line as blocking, which meant a vision model -- which will nearly always
    find something to say about a half-built page -- would fail essentially
    every step. Anything we cannot confidently read as blocking is treated as
    minor, so ambiguity never blocks.
    """
    payload = _first_json_array(reply or "")
    if payload is None:
        return []

    defects: list[VisualDefect] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        problem = str(item.get("problem") or item.get("issue") or "").strip()
        if not problem:
            continue
        severity = str(item.get("severity", "minor")).strip().lower()
        defects.append(
            VisualDefect(
                severity="blocking" if severity == "blocking" else "minor",
                element=str(item.get("element") or "section").strip()[:60],
                problem=problem[:200],
            )
        )
        if len(defects) >= MAX_VISUAL_DEFECTS:
            break
    return defects


def _first_json_array(text: str) -> list | None:
    """Pull the JSON array out of a reply, tolerating fences and stray prose."""
    stripped = text.strip()
    if stripped.upper().startswith("CLEAN"):
        return []
    match = re.search(r"\[.*\]", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def blocking_only(defects: list[VisualDefect]) -> list[str]:
    """The defects that are allowed to fail a step."""
    return [str(d) for d in defects if d.severity == "blocking"]


def summarize(defects: list[Defect], model_defects: list[str], limit: int = 6) -> str:
    """One compact block to hand back to the model as a correction brief."""
    lines = [str(d) for d in defects[:limit]]
    lines.extend(model_defects[: max(0, limit - len(lines))])
    return "\n".join(f"- {line}" for line in lines)


def layout_script(node_id: str) -> str:
    return LAYOUT_SCRIPT.format(node_id=json.dumps(node_id))
