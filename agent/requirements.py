"""Did the finished design actually contain what the user asked for?

Every other gate in this project measures *how* something was built -- geometry,
contrast, tokens. None of them asks whether the sign-in screen has a password
field. That is CLAUDE.md's known gap #2: `success` meant "no step exhausted its
retries", so a run could satisfy none of the instruction and still report a
green tick.

Two deliberate design choices keep this honest:

1. **Requirements are derived from the USER'S instruction, never from the
   brief.** The brief is written by the model, so triggering on it would let
   the agent invent its own homework and then mark it complete.
2. **A requirement is only asserted when the user named it.** Nothing is
   assumed about what a "landing page" ought to contain, so a missing item is
   always something that was literally asked for and is literally absent.

Like `bench.score`'s requirements dimension, this is a PROXY: it checks that a
node named or reading "password" exists, which is evidence a password field was
built -- not proof it was built well. Read the number that way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# (label, trigger on the instruction, evidence in the finished tree).
#
# Triggers are deliberately narrow and evidence deliberately generous: a false
# "missing" is far more damaging than a missed check, because it tells the user
# their design is broken when it is fine.
_ELEMENTS: list[tuple[str, str, str]] = [
    ("password field", r"\bpasswords?\b", r"password"),
    ("email field", r"\be-?mails?\b", r"e-?mail|@"),
    ("search field", r"\bsearch\b", r"search|find"),
    ("navigation bar", r"\bnav(?:bar|igation)?\b", r"nav|menu|header|logo"),
    ("header", r"\bheader\b", r"header|nav|top ?bar"),
    ("footer", r"\bfooter\b", r"footer|copyright|©"),
    ("hero section", r"\bhero\b", r"hero|headline|banner"),
    ("sidebar", r"\bside ?(?:bar|nav)\b", r"side ?bar|side ?nav|rail"),
    ("chart", r"\b(?:charts?|graphs?)\b", r"chart|graph|plot|trend"),
    ("data table", r"\btables?\b", r"table|row|column|cell"),
    ("avatar", r"\bavatars?\b|\bprofile (?:picture|photo)\b", r"avatar|profile|user"),
    ("tab bar", r"\btabs?\b|\btab bar\b", r"tab"),
    ("cards", r"\bcards?\b", r"card|tile|item"),
    ("logo", r"\blogos?\b", r"logo|brand|mark"),
    ("button", r"\bbuttons?\b|\bcta\b|\bcall to action\b", r"button|btn|cta|sign|get started|continue"),
    ("checkbox", r"\bcheck ?box(?:es)?\b", r"check ?box|remember"),
    ("dropdown", r"\bdrop ?downs?\b|\bselects?\b", r"drop ?down|select|picker"),
    ("toggle", r"\btoggles?\b|\bswitch(?:es)?\b", r"toggle|switch"),
    ("badge", r"\bbadges?\b|\bchips?\b|\btags?\b|\bpills?\b", r"badge|chip|tag|pill|status"),
    ("breadcrumb", r"\bbread ?crumbs?\b", r"bread ?crumb|\/"),
    ("pagination", r"\bpaginat\w*\b", r"pagin|next|previous|page"),
    ("modal", r"\b(?:modals?|dialogs?)\b", r"modal|dialog|overlay"),
    ("input field", r"\b(?:inputs?|text fields?|form fields?)\b", r"input|field|placeholder"),
    ("form", r"\bforms?\b", r"form|field|input|submit"),
    ("Google sign-in", r"\bgoogle\b", r"google"),
    ("Apple sign-in", r"\bapple\b", r"apple"),
    ("pricing", r"\bpricing?\b|\bpricing\b", r"pric|\$|\bmo\b|month|free|plan"),
    ("testimonial", r"\btestimonials?\b", r"testimonial|quote|review|says"),
    ("FAQ", r"\bfaqs?\b|\bfrequently asked\b", r"faq|question|\?"),
    ("newsletter signup", r"\bnewsletter\b", r"newsletter|subscribe|updates"),
    ("metric cards", r"\b(?:metrics?|kpis?|stats?|statistics)\b", r"metric|kpi|stat|total|revenue|users|%"),
    ("notifications", r"\bnotifications?\b|\bbell\b", r"notif|bell|alert"),
    ("copyright line", r"\bcopyright\b", r"copyright|©|rights reserved"),
    ("sign-in action", r"\bsign ?in\b|\blog ?in\b", r"sign ?in|log ?in"),
    ("sign-up action", r"\bsign ?up\b|\bregister\b|\bcreate an account\b", r"sign ?up|register|create account"),
]

# Copy the user put in quotes is a literal requirement -- they wrote the words,
# so they expect to see those words. Capped so one heavily-quoted instruction
# cannot drown out every element check.
_QUOTED = re.compile(r"['\"‘“]([^'\"’”]{2,40})['\"’”]")
MAX_COPY_REQUIREMENTS = 8

# Fewer than this and coverage is too small a sample to draw a conclusion from.
MIN_REQUIREMENTS_TO_JUDGE = 3


@dataclass(frozen=True)
class Requirement:
    """One checkable thing the instruction asked for."""

    label: str
    evidence: str  # regex matched against every node's name and text


@dataclass
class Coverage:
    """What was asked for, and what is demonstrably on the canvas."""

    met: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def expected(self) -> int:
        return len(self.met) + len(self.missing)

    @property
    def ratio(self) -> float | None:
        """0..1, or None when the instruction named nothing checkable."""
        return len(self.met) / self.expected if self.expected else None

    @property
    def satisfied_nothing(self) -> bool:
        """Built a page that matches none of a clearly-specified instruction.

        This is the case CLAUDE.md calls out by name, and the only one confident
        enough to fail a run over -- one missing field is a flaw, but zero of
        five requirements met is not the design that was requested.
        """
        return self.expected >= MIN_REQUIREMENTS_TO_JUDGE and not self.met

    def summary(self) -> str:
        return f"{len(self.met)}/{self.expected} requirements met"


def expected_requirements(instruction: str) -> list[Requirement]:
    """The checkable requirements the USER'S OWN WORDS asked for."""
    text = instruction or ""
    found: list[Requirement] = []
    seen: set[str] = set()

    for label, trigger, evidence in _ELEMENTS:
        if label in seen or not re.search(trigger, text, re.IGNORECASE):
            continue
        seen.add(label)
        found.append(Requirement(label=label, evidence=evidence))

    for phrase in _QUOTED.findall(text)[:MAX_COPY_REQUIREMENTS]:
        cleaned = phrase.strip()
        # A quoted number or symbol is not copy, and a bare word already
        # covered by an element check would double-count it.
        if not re.search(r"[A-Za-z]{2}", cleaned):
            continue
        label = f'copy "{cleaned}"'
        if label in seen:
            continue
        seen.add(label)
        found.append(Requirement(label=label, evidence=re.escape(cleaned)))

    return found


def flatten(tree: dict | list | None) -> list[dict]:
    """Every node as a flat list. Accepts one tree or several.

    Several, because a design is now one frame PER SCREEN: a password field on
    the sign-in screen satisfies the instruction whether or not the dashboard
    beside it has one.
    """
    if isinstance(tree, list):
        return [node for item in tree for node in flatten(item)]
    if not isinstance(tree, dict):
        return []
    nodes = [tree]
    for child in tree.get("children") or []:
        nodes.extend(flatten(child))
    return nodes


def check_coverage(instruction: str, tree: dict | list | None) -> Coverage:
    """Match the instruction's requirements against the finished node tree(s)."""
    requirements = expected_requirements(instruction)
    if not requirements:
        return Coverage()

    # One haystack per node (name + its text), so evidence can be satisfied by
    # either a designer-ish node name or the visible copy.
    haystacks = [
        f"{node.get('name') or ''} {node.get('characters') or ''}".lower()
        for node in flatten(tree)
    ]

    coverage = Coverage()
    for requirement in requirements:
        pattern = re.compile(requirement.evidence, re.IGNORECASE)
        if any(pattern.search(hay) for hay in haystacks):
            coverage.met.append(requirement.label)
        else:
            coverage.missing.append(requirement.label)
    return coverage
