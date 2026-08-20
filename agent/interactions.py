"""A design you can click through, not a picture of one.

Everything else in this project produces a STATIC mockup: frames, text, fills.
Open it in Figma and press play and nothing happens -- the sign-in button is a
purple rectangle with the word "Sign in" on it. A real Figma design is wired:
clicking that button takes you to the dashboard, "Back" goes back, a long page
scrolls. That wiring is Figma's prototype layer, and it is as much a part of a
design file as the pixels are.

The Plugin API calls it a `Reaction`: a TRIGGER (`ON_CLICK`) plus ACTIONS
(navigate to this frame, with this transition). Under
`documentAccess: "dynamic-page"` the `reactions` property is read-only, so the
only way to set one is `await node.setReactionsAsync([...])`.

    ┌── Login ──────────┐            ┌── Dashboard ──────┐
    │  [ Sign in ] ─────┼─ ON_CLICK ─┼──►                │
    │  Create account ──┼────────────┼─► Sign Up         │
    └───────────────────┘            └───────────────────┘

Who decides what links to what, and why it is split that way:

- **Matching a label to a screen is arithmetic, so Python does it**
  (CLAUDE.md section 7). "Sign up" on the Login screen goes to the Sign Up
  screen; that is not a design judgement, it is string matching, and a model
  asked to do it gets it wrong often enough to matter.
- **The model wires what it is building, as it builds it.** A `render_ui` spec
  may carry `"on_click": "Dashboard"` on any node, because the model knows
  which button is the primary action of the section it just designed and
  nothing downstream can work that out from a label.
- **The gaps are filled once, at the end, from the real canvas.** Reading what
  was actually built is the only honest source -- CLAUDE.md's "never assume
  canvas state" applies to interactions exactly as it does to geometry.

Two properties that keep this safe on someone else's file:

1. **A node that already has a reaction is never overwritten.** It is either
   the user's own wiring or the model's, and both are more specific than a
   name match.
2. **Flow starting points are MERGED, not replaced.** Assigning
   `flowStartingPoints` overwrites the whole list, and a re-run doing that
   would throw away flows the user set by hand.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# What a spec may say, and the Plugin API value it means. `ON_HOVER` and
# `ON_PRESS` are real triggers; anything else is refused rather than guessed.
TRIGGERS = {
    "click": {"type": "ON_CLICK"},
    "tap": {"type": "ON_CLICK"},
    "hover": {"type": "ON_HOVER"},
    "press": {"type": "ON_PRESS"},
    "drag": {"type": "ON_DRAG"},
}
DEFAULT_TRIGGER = "click"

_EASE = {"type": "EASE_OUT"}
# Named transitions, so a spec picks a NAME and an invalid easing/duration
# combination is not expressible. Exact enum values verified against the
# typings: SimpleTransition and DirectionalTransition are different shapes.
TRANSITIONS: dict[str, dict | None] = {
    "instant": None,
    "dissolve": {"type": "DISSOLVE", "easing": _EASE, "duration": 0.3},
    "smart": {"type": "SMART_ANIMATE", "easing": _EASE, "duration": 0.3},
    "slide-left": {
        "type": "SLIDE_IN", "direction": "LEFT", "matchLayers": False,
        "easing": _EASE, "duration": 0.3,
    },
    "slide-right": {
        "type": "SLIDE_IN", "direction": "RIGHT", "matchLayers": False,
        "easing": _EASE, "duration": 0.3,
    },
    "push-left": {
        "type": "PUSH", "direction": "LEFT", "matchLayers": False,
        "easing": _EASE, "duration": 0.3,
    },
    "move-in": {
        "type": "MOVE_IN", "direction": "BOTTOM", "matchLayers": False,
        "easing": _EASE, "duration": 0.3,
    },
}
# Dissolve rather than Figma's own "instant" default: it reads as a designed
# transition everywhere, and unlike SMART_ANIMATE it cannot look broken when
# the two screens share no layers.
DEFAULT_TRANSITION = "dissolve"

# Actions a link may take. Deliberately four: a vocabulary the model cannot
# overreach is worth more than one that covers every prototype feature.
ACTIONS = ("navigate", "back", "url", "scroll_to")

# One design should not become a switchboard. Well past what a real six-screen
# design needs, and a hard stop on a runaway match.
MAX_LINKS = 60


@dataclass(frozen=True)
class Link:
    """One interaction: this node, when clicked, does this."""

    source_id: str
    label: str = ""
    action: str = "navigate"
    destination_id: str = ""
    destination_name: str = ""
    trigger: str = DEFAULT_TRIGGER
    transition: str = DEFAULT_TRANSITION
    url: str = ""
    # Which screen the SOURCE sits on. Only used for reporting, but it is what
    # makes the report readable: "Login · 'Sign in' -> Dashboard".
    screen_name: str = ""

    def describe(self) -> str:
        where = f"{self.screen_name} · " if self.screen_name else ""
        label = f"'{self.label}'" if self.label else self.source_id
        if self.action == "back":
            return f"{where}{label} -> back"
        if self.action == "url":
            return f"{where}{label} -> {self.url}"
        if self.action == "scroll_to":
            return f"{where}{label} -> scrolls to {self.destination_name}"
        return f"{where}{label} -> {self.destination_name or self.destination_id}"


def reaction(link: Link) -> dict:
    """The Plugin API `Reaction` for one link, as plain data.

    Built in Python and passed to the script as JSON, so the generated
    JavaScript has no object construction in it to get wrong.
    """
    trigger = TRIGGERS.get(link.trigger, TRIGGERS[DEFAULT_TRIGGER])
    if link.action == "back":
        # BACK takes no transition -- it replays the one that brought you here.
        return {"trigger": trigger, "actions": [{"type": "BACK"}]}
    if link.action == "url":
        return {
            "trigger": trigger,
            "actions": [{"type": "URL", "url": link.url, "openInNewTab": True}],
        }
    navigation = "SCROLL_TO" if link.action == "scroll_to" else "NAVIGATE"
    return {
        "trigger": trigger,
        "actions": [
            {
                "type": "NODE",
                "destinationId": link.destination_id,
                "navigation": navigation,
                "transition": TRANSITIONS.get(link.transition, TRANSITIONS[DEFAULT_TRANSITION]),
            }
        ],
    }


def normalize_trigger(value: str | None) -> str:
    key = str(value or "").strip().lower().replace("on_", "").replace(" ", "")
    return key if key in TRIGGERS else DEFAULT_TRIGGER


def normalize_transition(value: str | None) -> str:
    key = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "none": "instant", "immediate": "instant", "fade": "dissolve",
        "smart-animate": "smart", "animate": "smart", "slide": "slide-left",
        "push": "push-left",
    }
    key = aliases.get(key, key)
    return key if key in TRANSITIONS else DEFAULT_TRANSITION


# -- reading the canvas ------------------------------------------------------

# What might plausibly be clicked. Filtered in JavaScript so the payload stays
# small: a five-screen design is thousands of nodes and only a handful of them
# are ever a button.
_CANDIDATES_SCRIPT = """const screens = __SCREENS__;
const CLICKABLE = /button|btn|cta|link|nav|tab|menu|item|card|chip|pill|action|icon|logo|back/i;
const MAX_CANDIDATES = 400;
const MAX_DEPTH = 8;

function firstText(node, depth) {
  if (node.type === 'TEXT') { return String(node.characters); }
  if (!('children' in node) || depth > 3) { return ''; }
  for (const child of node.children) {
    const found = firstText(child, depth + 1);
    if (found) { return found; }
  }
  return '';
}

// `reactions` is READ-ONLY under documentAccess: "dynamic-page", but readable.
// A node that already has one is never touched again.
function isWired(node) {
  if (!('reactions' in node)) { return false; }
  const existing = node.reactions;
  return Array.isArray(existing) && existing.length > 0;
}

function interesting(node) {
  if (node.type === 'TEXT') { return true; }
  if (isWired(node)) { return true; }
  return CLICKABLE.test(node.name);
}

const candidates = [];
for (const screen of screens) {
  const frame = await figma.getNodeByIdAsync(screen.id);
  if (!frame || frame.removed || !('children' in frame)) { continue; }
  // Breadth-first, so a Button frame is seen BEFORE the text inside it and
  // wins the link -- clicking the label alone is not what anyone means.
  const queue = [{ node: frame, path: [], depth: 0 }];
  while (queue.length > 0 && candidates.length < MAX_CANDIDATES) {
    const entry = queue.shift();
    const node = entry.node;
    if (node.id !== frame.id && interesting(node)) {
      candidates.push({
        id: node.id,
        name: String(node.name).slice(0, 60),
        type: node.type,
        label: firstText(node, 0).slice(0, 60),
        screenId: screen.id,
        path: entry.path,
        wired: isWired(node),
        width: Math.round(node.width),
        height: Math.round(node.height)
      });
    }
    if ('children' in node && entry.depth < MAX_DEPTH) {
      const path = entry.path.concat([node.id]);
      for (const child of node.children) {
        queue.push({ node: child, path: path, depth: entry.depth + 1 });
      }
    }
  }
}
return { createdNodeIds: [], candidates: candidates };
"""


def build_candidates_script(screens: list[dict]) -> str:
    """One round trip: everything on the page that could be clicked."""
    return _CANDIDATES_SCRIPT.replace(
        "__SCREENS__",
        json.dumps([{"id": str(s["id"]), "name": str(s.get("name", ""))} for s in screens]),
    )


@dataclass(frozen=True)
class Candidate:
    """A node that might be a button, read off the real canvas."""

    id: str
    name: str
    type: str
    label: str
    screen_id: str
    path: tuple[str, ...] = ()
    wired: bool = False
    width: int = 0
    height: int = 0

    @property
    def text(self) -> str:
        """What this thing SAYS -- its own text, or the tail of its name.

        The renderer names a button "Button / Sign in", so the name carries the
        label even when the text node is nested deeper than the reader looked.
        """
        if self.label.strip():
            return self.label.strip()
        return self.name.split("/")[-1].strip()


def parse_candidates(payload: dict | None) -> list[Candidate]:
    rows = (payload or {}).get("candidates") or []
    found = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        found.append(
            Candidate(
                id=str(row["id"]),
                name=str(row.get("name") or ""),
                type=str(row.get("type") or ""),
                label=str(row.get("label") or ""),
                screen_id=str(row.get("screenId") or ""),
                path=tuple(str(p) for p in (row.get("path") or [])),
                wired=bool(row.get("wired")),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
            )
        )
    return found


# -- deciding what links to what --------------------------------------------

# Words that mean "the previous screen", whatever that turns out to be. BACK is
# right for all of them: CLOSE only does anything inside an overlay.
_BACK_WORDS = {
    "back", "go back", "cancel", "close", "dismiss", "x", "return", "back to home",
    "previous", "skip",
}

# A label that names its destination in different words. The VALUES are matched
# against screen names, so nothing is invented -- if the design has no screen
# like that, no link is made.
_DESTINATION_HINTS: dict[str, tuple[str, ...]] = {
    # Ordered: the screen a successful sign-in LANDS on, and only if there is
    # no such screen, the sign-in screen itself. Both readings are real --
    # "Sign In" on a login screen submits, "Sign in" on a sign-up screen is the
    # link back -- and the source screen is excluded from matching, so the same
    # entry answers both without ever linking a screen to itself.
    "sign in": ("dashboard", "home", "overview", "feed", "login", "log in", "sign in"),
    "log in": ("dashboard", "home", "overview", "feed", "login", "log in", "sign in"),
    "login": ("dashboard", "home", "overview", "feed", "login", "log in", "sign in"),
    "continue": ("dashboard", "home", "overview", "next"),
    "submit": ("dashboard", "home", "confirmation", "success"),
    "get started": ("sign up", "signup", "register", "onboarding", "home"),
    "start free trial": ("sign up", "signup", "register"),
    "try it free": ("sign up", "signup", "register"),
    "sign up": ("sign up", "signup", "register", "create account"),
    "create account": ("sign up", "signup", "register"),
    "create one": ("sign up", "signup", "register", "create account"),
    "create an account": ("sign up", "signup", "register"),
    "sign up now": ("sign up", "signup", "register"),
    "register": ("sign up", "signup", "register"),
    "join": ("sign up", "signup", "register"),
    "forgot password": ("forgot", "reset"),
    "log out": ("login", "log in", "sign in", "welcome", "landing"),
    "sign out": ("login", "log in", "sign in", "welcome", "landing"),
    "checkout": ("checkout", "payment", "cart"),
    "add to cart": ("cart", "basket", "checkout"),
    "buy now": ("checkout", "cart", "payment"),
    "view cart": ("cart", "basket"),
    "search": ("search", "results", "explore", "browse"),
    "shop now": ("shop", "store", "products", "catalogue", "catalog"),
    "browse": ("shop", "browse", "explore", "catalogue", "catalog"),
    "view all": ("shop", "browse", "results", "list"),
    "learn more": ("about", "details", "features", "product"),
    "settings": ("settings", "preferences"),
    "profile": ("profile", "account"),
    "my account": ("account", "profile", "settings"),
    "home": ("home", "dashboard", "feed", "landing"),
}

# A long string is prose, not a button. "Welcome back to your dashboard" must
# not become a link to the Dashboard screen because it contains the word.
MAX_LABEL_CHARS = 30
MAX_LABEL_WORDS = 4

# ...with one exception, because it is how every auth pair in existence is
# written: "Don't have an account? Create one". The whole string is the link,
# it is longer than a button label, and it ends with the action. Only an
# ACTION tail counts -- a sentence ending in a screen's NAME ("Welcome back to
# your dashboard") is still prose, which is the false positive this guards.
MAX_TAIL_LABEL_CHARS = 64
_ACTION_TAILS = (
    "create one", "create an account", "create account", "sign up", "sign up now",
    "register", "join", "sign in", "log in", "login", "get started", "learn more",
    "view all", "shop now", "browse", "checkout", "search",
)

# Below this, a substring match is coincidence rather than a name.
MIN_NAME_MATCH_CHARS = 4

_BUTTONISH = re.compile(r"button|btn|cta|link|nav|tab|menu|item|chip|pill|action", re.IGNORECASE)


def _clean(text: str) -> str:
    """Lowercase, punctuation-free, single-spaced -- so "Sign up!" == "sign up"."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())).strip()


def _is_clickable(candidate: Candidate) -> bool:
    """Would a person expect this to do something when clicked?"""
    if candidate.type == "TEXT":
        return len(candidate.text) <= MAX_TAIL_LABEL_CHARS
    return bool(_BUTTONISH.search(candidate.name))


def _screen_for(label: str, screens: list, current_id: str):
    """Which screen this label points at, or None. Pure string matching."""
    cleaned = _clean(label)
    if not cleaned:
        return None
    others = [s for s in screens if str(getattr(s, "frame_id", "")) != current_id]
    names = [(s, _clean(s.name)) for s in others]

    if len(cleaned) <= MAX_LABEL_CHARS and len(cleaned.split()) <= MAX_LABEL_WORDS:
        for screen, name in names:
            if name and name == cleaned:
                return screen
        for screen, name in names:
            if len(name) >= MIN_NAME_MATCH_CHARS and (name in cleaned or cleaned in name):
                return screen
        for phrase, keywords in _DESTINATION_HINTS.items():
            if cleaned == phrase or cleaned.startswith(phrase + " "):
                found = _by_keywords(keywords, names)
                if found is not None:
                    return found
        return None

    # A longer string is prose unless it ENDS with an action -- the
    # "Don't have an account? Create one" shape.
    if len(cleaned) <= MAX_TAIL_LABEL_CHARS:
        for phrase in _ACTION_TAILS:
            if cleaned.endswith(phrase):
                found = _by_keywords(_DESTINATION_HINTS.get(phrase, (phrase,)), names)
                if found is not None:
                    return found
    return None


def _by_keywords(keywords, names):
    """The first screen whose name answers to one of these keywords, in order.

    Order is the whole point: "sign in" means the dashboard when there is one
    and the login screen when there is not.
    """
    for keyword in keywords:
        for screen, name in names:
            if keyword in name or name in keyword:
                return screen
    return None


def auto_link(candidates: list[Candidate], screens: list) -> list[Link]:
    """Wire every button whose label names a screen. No model involved.

    Three rules keep this from wiring the whole page together:

    - The outermost match wins. A `Button / Sign in` frame and the TEXT inside
      it both say "Sign in"; linking both means the label swallows the click
      and the button's own reaction never fires.
    - A node that already has a reaction is left alone -- it was wired by the
      model while building, or by the user, and both know more than a name.
    - Only short, button-shaped labels are considered at all, so a heading that
      happens to contain a screen's name is not a navigation link.
    """
    by_id = {str(getattr(s, "frame_id", "")): s for s in screens if getattr(s, "frame_id", None)}
    links: list[Link] = []
    claimed: set[str] = set()
    for candidate in candidates:
        if len(links) >= MAX_LINKS:
            break
        if candidate.wired:
            claimed.add(candidate.id)  # its children must not be wired either
            continue
        if any(parent in claimed for parent in candidate.path):
            continue
        if not _is_clickable(candidate):
            continue
        screen = by_id.get(candidate.screen_id)
        screen_name = screen.name if screen else ""
        label = candidate.text
        cleaned = _clean(label)
        if cleaned in _BACK_WORDS:
            links.append(
                Link(source_id=candidate.id, label=label, action="back", screen_name=screen_name)
            )
            claimed.add(candidate.id)
            continue
        target = _screen_for(label, screens, candidate.screen_id)
        if target is None:
            continue
        links.append(
            Link(
                source_id=candidate.id,
                label=label,
                destination_id=str(target.frame_id),
                destination_name=target.name,
                screen_name=screen_name,
            )
        )
        claimed.add(candidate.id)
    return links


def unlinked(candidates: list[Candidate], links: list[Link]) -> list[Candidate]:
    """Button-shaped things nothing has wired yet -- what the model is asked about."""
    wired_ids = {link.source_id for link in links}
    return [
        c
        for c in candidates
        if not c.wired
        and c.id not in wired_ids
        and _BUTTONISH.search(c.name)
        and c.text
    ]


def unreachable(screens: list, links: list[Link]) -> list:
    """Screens nothing navigates to. The first screen is the entry point, so it
    is not counted -- a design's home screen having no way IN is normal."""
    arrived = {link.destination_id for link in links if link.action == "navigate"}
    return [s for s in screens[1:] if str(getattr(s, "frame_id", "")) not in arrived]


# -- writing it back ---------------------------------------------------------

_APPLY_SCRIPT = """const links = __LINKS__;
const applied = [];
const failed = [];
for (const link of links) {
  try {
    const node = await figma.getNodeByIdAsync(link.id);
    if (!node || node.removed) { throw new Error('node is gone'); }
    if (!('setReactionsAsync' in node)) {
      throw new Error(node.type + ' cannot hold an interaction');
    }
    // reactions is read-only under dynamic-page access; this is the only setter.
    await node.setReactionsAsync(link.reactions);
    applied.push(link.label);
  } catch (e) {
    failed.push(link.label + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: [], applied: applied, failed: failed };
"""


def build_apply_script(links: list[Link]) -> str:
    """Wire every link, each in its own try/catch.

    Not atomic, deliberately, and for the same reason `agent/editor.py` is not:
    one stale node id must not discard nineteen good interactions.
    """
    payload = [
        {"id": link.source_id, "label": link.describe(), "reactions": [reaction(link)]}
        for link in links[:MAX_LINKS]
    ]
    return _APPLY_SCRIPT.replace("__LINKS__", json.dumps(payload))


_FLOW_SCRIPT = """const specs = __SPECS__;
const page = figma.currentPage;
const wanted = [];
const scrolled = [];
for (const spec of specs) {
  const node = await figma.getNodeByIdAsync(spec.id);
  if (!node || node.removed) { continue; }
  if (spec.start) { wanted.push({ nodeId: node.id, name: spec.name }); }
  // A page taller than its own viewport should SCROLL in the prototype
  // instead of being cut off at the fold.
  if (spec.scrolls && 'overflowDirection' in node) {
    node.overflowDirection = 'VERTICAL';
    scrolled.push(node.name);
  }
}
// MERGED, never replaced: assigning this property overwrites the whole list,
// and a re-run must not throw away flows the user set up by hand.
const ours = {};
for (const entry of wanted) { ours[entry.nodeId] = true; }
const kept = [];
for (const existing of page.flowStartingPoints) {
  if (ours[existing.nodeId]) { continue; }
  const node = await figma.getNodeByIdAsync(existing.nodeId);
  if (node && !node.removed) { kept.push({ nodeId: existing.nodeId, name: existing.name }); }
}
page.flowStartingPoints = wanted.concat(kept);
return {
  createdNodeIds: [],
  flows: wanted.map(function (entry) { return entry.name; }),
  scrolled: scrolled
};
"""


def build_flow_script(specs: list[dict]) -> str:
    """Prototype entry points, and which screens scroll.

    `specs` is `[{id, name, start: bool, scrolls: bool}]`. A starting point is
    what makes a frame playable on its own: without one, a screen nothing links
    to cannot be reached in Presentation view at all.
    """
    return _FLOW_SCRIPT.replace(
        "__SPECS__",
        json.dumps(
            [
                {
                    "id": str(spec["id"]),
                    "name": str(spec.get("name") or "Flow"),
                    "start": bool(spec.get("start")),
                    "scrolls": bool(spec.get("scrolls")),
                }
                for spec in specs
                if spec.get("id")
            ]
        ),
    )


# -- the model's answer, checked ---------------------------------------------


@dataclass
class LinkPlan:
    """What the model proposed, after everything unusable was dropped."""

    links: list[Link] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


def parse_link_plan(
    content: str, candidates: list[Candidate], screens: list
) -> LinkPlan:
    """Turn the model's JSON into links, verifying every id and every screen.

    Same contract as `agent/inventory.py`'s `resolve`: an id the model invented
    never reaches Figma, and a destination that is not a real screen is dropped
    with a reason rather than guessed at.
    """
    plan = LinkPlan()
    rows = _parse_json_list(content)
    by_id = {c.id: c for c in candidates}
    screen_by_id = {
        str(getattr(s, "frame_id", "")): s for s in screens if getattr(s, "frame_id", None)
    }
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = str(row.get("id") or row.get("source") or "").strip()
        candidate = by_id.get(source)
        if candidate is None:
            plan.rejected.append(f"no node {source!r} on the canvas")
            continue
        if source in seen:
            continue
        seen.add(source)
        action = str(row.get("action") or "navigate").strip().lower()
        source_screen = screen_by_id.get(candidate.screen_id)
        where = source_screen.name if source_screen else ""
        if action == "back":
            plan.links.append(
                Link(source_id=source, label=candidate.text, action="back", screen_name=where)
            )
            continue
        target = _named_screen(row.get("to") or row.get("screen"), screens)
        if target is None:
            plan.rejected.append(f"no screen named {str(row.get('to') or '')!r}")
            continue
        if str(target.frame_id) == candidate.screen_id:
            plan.rejected.append(f"{candidate.text!r} links to its own screen")
            continue
        plan.links.append(
            Link(
                source_id=source,
                label=candidate.text,
                destination_id=str(target.frame_id),
                destination_name=target.name,
                trigger=normalize_trigger(row.get("trigger")),
                transition=normalize_transition(row.get("transition")),
                screen_name=where,
            )
        )
    return plan


def resolve_screen(name, screens: dict[str, str]) -> str:
    """A screen frame id from a name the model typed. "" when nothing matches.

    Loose on purpose: the model reads the screen names out of a prompt and
    retypes them, so "Dashboard", "dashboard" and "Dashboard screen" have to
    reach the same frame. Never a fuzzy guess, though -- a name that matches
    nothing returns nothing, and the caller drops the link.
    """
    wanted = _clean(name)
    if not wanted:
        return ""
    cleaned = {_clean(key): value for key, value in (screens or {}).items()}
    if wanted in cleaned:
        return cleaned[wanted]
    for key, value in cleaned.items():
        if len(key) >= MIN_NAME_MATCH_CHARS and (key in wanted or wanted in key):
            return value
    return ""


def _named_screen(name, screens):
    """A screen object by name, for the model's own link plan."""
    mapping = {
        s.name: str(getattr(s, "frame_id", "")) for s in screens if getattr(s, "frame_id", None)
    }
    frame_id = resolve_screen(name, mapping)
    if not frame_id:
        return None
    return next((s for s in screens if str(getattr(s, "frame_id", "")) == frame_id), None)


def _parse_json_list(content: str) -> list:
    """The JSON array in a reply that may be wrapped in prose or a code fence."""
    text = (content or "").strip()
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "").strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
