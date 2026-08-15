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

logger = logging.getLogger(__name__)

# Reads the subtree with the geometry needed to reason about layout.
LAYOUT_SCRIPT = """\
const root = await figma.getNodeByIdAsync({node_id});
if (!root) {{ throw new Error('node not found'); }}

function describe(node, depth) {{
  const info = {{
    id: node.id, name: node.name, type: node.type,
    x: Math.round(node.x), y: Math.round(node.y),
    width: Math.round(node.width), height: Math.round(node.height),
    visible: node.visible !== false,
    layoutMode: 'layoutMode' in node ? node.layoutMode : null,
    children: []
  }};
  if (node.type === 'TEXT') {{
    info.characters = String(node.characters).slice(0, 60);
    info.fontSize = typeof node.fontSize === 'number' ? node.fontSize : null;
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

    def __str__(self) -> str:
        return f"[{self.kind}] {self.node_name}: {self.detail}"


# A couple of pixels of overlap is normal from rounding; this is the threshold
# at which it reads as a real visual bug.
OVERLAP_TOLERANCE = 2
OVERFLOW_TOLERANCE = 2


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
    roots = [tree] if not scope_ids else find_subtrees(tree, set(scope_ids))
    found: list[Defect] = []
    for root in roots:
        _walk(root, found)
    return _condense(found)


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


def _walk(node: dict, defects: list[Defect]) -> None:
    children = node.get("children") or []
    _check_self(node, defects)

    if children:
        _check_children_fit(node, children, defects)
        # Overlap only means something when the parent isn't laying children
        # out itself -- auto-layout cannot produce overlapping siblings.
        if node.get("layoutMode") in (None, "NONE"):
            _check_overlaps(children, defects)

    for child in children:
        _walk(child, defects)


def _check_self(node: dict, defects: list[Defect]) -> None:
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

    if node.get("type") == "FRAME" and not (node.get("children") or []):
        if width > 40 and height > 40:
            defects.append(
                Defect(node_id, name, "empty-frame",
                       f"{width}x{height} frame is empty, leaving a blank region")
            )


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
