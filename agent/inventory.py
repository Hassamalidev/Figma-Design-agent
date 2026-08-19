"""What is already on the canvas, as something a model can name and target.

Create mode never needed this: it builds into a frame it made itself, so the
only ids it has to know are the ones it just returned. Edit mode is the
opposite problem -- every instruction ("make the login button purple") is about
a node that already exists, and the agent has no way to refer to it.

So the harness reads the canvas and hands over an INDEX: one line per node,
carrying its real id. The model picks ids out of that listing rather than
inventing selectors, and `resolve` checks every id it picks against the index
before a single Plugin API call is made. An id that is not in the index is a
hallucination and is refused with the listing attached, not run.

Two further rules, both learned from create mode:

- The reader is `critic.NODE_READER_JS`, the same one the visual gate uses.
  A second reader would drift, and "what the agent sees" would stop matching
  "what is checked".
- `find` is arithmetic, so Python does it (CLAUDE.md section 7). The model says
  what it is looking for; matching, ranking and disambiguation happen here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from agent import critic

# One line per node in the prompt. A big file is thousands of nodes and the
# listing is resent on every turn, so it is capped -- and the cap is announced
# rather than silently truncating, because a model that cannot see a node will
# confidently conclude it does not exist.
MAX_LISTED_NODES = 400

# Text longer than this is trimmed in the listing; the full string is still
# what `find` matches against.
MAX_TEXT_PREVIEW = 60

INVENTORY_SCRIPT = """const MAX_TREE_DEPTH = __MAX_DEPTH__;
__READER__
const page = figma.currentPage;
const selection = page.selection.map(function (n) { return n.id; });
const roots = page.children.slice(0, 30).map(function (n) { return describe(n, 0); });
return {
  createdNodeIds: [],
  pageId: page.id,
  pageName: page.name,
  selection: selection,
  roots: roots
};
"""


@dataclass
class Node:
    """One node, flattened out of the tree with its ancestry kept."""

    id: str
    name: str
    type: str
    depth: int
    width: int = 0
    height: int = 0
    text: str = ""
    font_size: float | None = None
    fill: tuple[float, float, float] | None = None
    layout_mode: str | None = None
    parent_id: str = ""
    screen: str = ""  # the top-level frame this node lives in


@dataclass
class Inventory:
    """The canvas as the editor sees it."""

    page_id: str = ""
    page_name: str = ""
    selection: list[str] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    roots: list[dict] = field(default_factory=list)  # the raw trees, for the gate
    truncated: bool = False

    def by_id(self, node_id: str) -> Node | None:
        return self._index.get(node_id)

    @property
    def _index(self) -> dict[str, Node]:
        if not hasattr(self, "_cached_index"):
            object.__setattr__(self, "_cached_index", {n.id: n for n in self.nodes})
        return getattr(self, "_cached_index")

    def is_empty(self) -> bool:
        return not self.nodes


def inventory_script() -> str:
    """The one round trip that reads the whole page."""
    return (
        INVENTORY_SCRIPT.replace("__MAX_DEPTH__", str(critic.MAX_TREE_DEPTH))
        .replace("__READER__", critic.NODE_READER_JS)
    )


def build(payload: dict) -> Inventory:
    """Flatten the reader's nested trees into an addressable list."""
    inventory = Inventory(
        page_id=str(payload.get("pageId") or ""),
        page_name=str(payload.get("pageName") or "Page"),
        selection=[str(i) for i in (payload.get("selection") or []) if i],
        roots=list(payload.get("roots") or []),
    )
    for root in inventory.roots:
        screen = str(root.get("name") or "?")
        _flatten(root, inventory.nodes, depth=0, parent_id="", screen=screen)
        if len(inventory.nodes) >= MAX_LISTED_NODES:
            inventory.truncated = True
            del inventory.nodes[MAX_LISTED_NODES:]
            break
    return inventory


def _flatten(node: dict, out: list[Node], depth: int, parent_id: str, screen: str) -> None:
    if len(out) >= MAX_LISTED_NODES:
        return
    fill = critic.fill_rgb(node)
    out.append(
        Node(
            id=str(node.get("id") or ""),
            name=str(node.get("name") or "?"),
            type=str(node.get("type") or "?"),
            depth=depth,
            width=int(node.get("width") or 0),
            height=int(node.get("height") or 0),
            text=str(node.get("characters") or ""),
            font_size=node.get("fontSize"),
            fill=fill,
            layout_mode=node.get("layoutMode"),
            parent_id=parent_id,
            screen=screen,
        )
    )
    for child in node.get("children") or []:
        _flatten(child, out, depth + 1, str(node.get("id") or ""), screen)


# ---- the listing the model reads -----------------------------------------


def format_listing(inventory: Inventory, scope_ids: list[str] | None = None) -> str:
    """The canvas as an indented, id-first listing.

    Id FIRST on every line, because that is the only part the model has to copy
    exactly and burying it after the name made it likelier to be paraphrased.
    """
    if inventory.is_empty():
        return "(the page is empty)"
    wanted = _scoped(inventory, scope_ids)
    lines = [f"PAGE '{inventory.page_name}'"]
    for node in inventory.nodes:
        if wanted is not None and node.id not in wanted:
            continue
        lines.append("  " + "  " * node.depth + _describe(node))
    if inventory.truncated:
        lines.append(
            f"  ... only the first {MAX_LISTED_NODES} nodes are listed; "
            "ask for metadata on a specific node if you need more"
        )
    return "\n".join(lines)


def _scoped(inventory: Inventory, scope_ids: list[str] | None) -> set[str] | None:
    """Ids in scope, plus every descendant -- selecting a card means its contents."""
    if not scope_ids:
        return None
    wanted = set(scope_ids)
    for node in inventory.nodes:  # parents precede children, so one pass is enough
        if node.parent_id in wanted:
            wanted.add(node.id)
    return wanted


def _describe(node: Node) -> str:
    parts = [f"{node.id}", node.type, f'"{node.name}"']
    if node.text:
        preview = node.text[:MAX_TEXT_PREVIEW]
        parts.append(f"text={preview!r}")
    if node.font_size:
        parts.append(f"{int(node.font_size)}px")
    if node.width or node.height:
        parts.append(f"{node.width}x{node.height}")
    if node.fill:
        parts.append(_hex(node.fill))
    return "  ".join(parts)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * 255))):02X}" for c in rgb)


# ---- finding nodes without the model guessing -----------------------------

_STOP_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "all", "every",
    "make", "change", "set", "update", "edit", "colour", "color", "please",
}


def find(inventory: Inventory, query: dict) -> list[str]:
    """Resolve a `find` selector to real node ids, most relevant first.

    Deterministic on purpose. Letting the model write its own `findAll`
    callback is exactly the class of thing that crashed live runs, and a
    selector it cannot express is one it cannot get wrong.
    """
    if not isinstance(query, dict):
        return []
    name = str(query.get("name") or "").strip().lower()
    text = str(query.get("text") or "").strip().lower()
    node_type = str(query.get("type") or "").strip().upper()
    screen = str(query.get("screen") or "").strip().lower()
    limit = int(query.get("limit") or 0) or None

    matches: list[tuple[int, str]] = []
    for node in inventory.nodes:
        if node_type and node.type != node_type:
            continue
        if screen and screen not in node.screen.lower():
            continue
        score = 0
        if name:
            score += _score(name, node.name)
            if not score:
                continue
        if text:
            hit = _score(text, node.text)
            if not hit:
                continue
            score += hit
        if not name and not text and not node_type and not screen:
            continue
        matches.append((score, node.id))

    matches.sort(key=lambda pair: (-pair[0], inventory.nodes.index(inventory.by_id(pair[1]))))
    found = [node_id for _, node_id in matches]
    return found[:limit] if limit else found


def _score(needle: str, haystack: str) -> int:
    """3 for an exact match, 2 for a substring, 1 for shared significant words."""
    target = (haystack or "").strip().lower()
    if not target:
        return 0
    if target == needle:
        return 3
    if needle in target:
        return 2
    words = {w for w in re.split(r"\W+", needle) if w and w not in _STOP_WORDS}
    target_words = {w for w in re.split(r"\W+", target) if w}
    return 1 if words and words & target_words else 0


def resolve(inventory: Inventory, target) -> tuple[list[str], str]:
    """Turn whatever the model named into verified node ids.

    Returns `(ids, error)`. An id that is not on the canvas comes back as an
    error rather than being sent to Figma: the Plugin API's own message for a
    missing node says nothing about which id was wrong or what the real ones
    are, and a model that gets it will simply invent another.
    """
    if isinstance(target, dict):
        found = find(inventory, target)
        if not found:
            return [], f"nothing on the canvas matches {json.dumps(target)}"
        return found, ""
    wanted = [target] if isinstance(target, str) else list(target or [])
    ids, missing = [], []
    for raw in wanted:
        node_id = str(raw).strip()
        if inventory.by_id(node_id) is not None:
            ids.append(node_id)
        else:
            missing.append(node_id)
    if missing:
        return [], (
            f"no node with id {', '.join(missing)} is on this page -- use an id "
            f"from the canvas listing exactly as it appears there"
        )
    return ids, ""
