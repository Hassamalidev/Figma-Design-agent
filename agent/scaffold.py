"""Deterministic scaffolding: the work the harness does itself.

Token creation is mechanical, identical every time, and the single most
common step to fail -- live runs lost whole runs to `createVariableSet`,
`figma.createStyle`, wrong mode ids and bad enum strings before ever reaching
composition. So we don't ask the model to do it. We read the palette out of
its own design brief and write the Plugin API calls ourselves.

This is CLAUDE.md section 5 ("tokens -> components -> composition") enforced
in code rather than hoped for in a prompt.
"""
from __future__ import annotations

import json
import re

# name: hex -- used when the brief has no usable colours at all.
FALLBACK_PALETTE: list[tuple[str, str]] = [
    ("accent", "#4F46E5"),
    ("accent-strong", "#4338CA"),
    ("bg", "#FFFFFF"),
    ("surface", "#F7F8FA"),
    ("border", "#E3E6EC"),
    ("text", "#101828"),
    ("text-muted", "#667085"),
]

# A type scale is far less variable than colour, and the brief's wording for it
# is inconsistent, so a sane default beats parsing. Inter is preloaded.
TEXT_STYLES: list[tuple[str, str, int, float]] = [
    # (name, Inter style, size, line height)
    ("Display", "Bold", 48, 56),
    ("Heading", "Semi Bold", 32, 40),
    ("Subheading", "Semi Bold", 20, 28),
    ("Body", "Regular", 16, 24),
    ("Caption", "Regular", 13, 20),
    ("Button", "Semi Bold", 15, 20),
]

# The ramp entries that carry the BRAND voice. A display font belongs on the
# wordmark and the headings; labels, inputs and buttons stay in the UI font,
# because a serif at 13px inside an input is what makes a page hard to read.
DISPLAY_STYLES = ("Display", "Heading", "Subheading")

DEFAULT_FONT_FAMILY = "Inter"

# "Primary Display Font: Playfair Display", "UI Font / Inter", "Font: Playfair
# Display". The captured phrase is deliberately greedy and is then offered to
# Figma longest-first (see `candidate_fonts`): guessing where a font name ends
# is exactly the kind of thing to ask the real API about.
_FONT_HINT = re.compile(
    r"\bfont\b[^A-Za-z]{0,12}([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})",
    re.IGNORECASE,   # the KEYWORD is case-insensitive; the NAME must be capitalised
)
MAX_FONT_CANDIDATES = 10


def candidate_fonts(text: str) -> list[str]:
    """Font families the instruction might be naming, in the order it names them.

    Every one is a GUESS. `loop.discover_fonts` asks Figma which of them really
    exist, which is the same rule this project already applies to font STYLE
    strings -- "Inter SemiBold" killed a live run because the real style is
    "Semi Bold". Nothing is ever used because it looked like a font name.
    """
    flat = re.sub(r"[*#`_>|]", " ", str(text or ""))
    flat = re.sub(r"\s+", " ", flat)
    found: list[str] = []
    for match in _FONT_HINT.finditer(flat):
        words = match.group(1).split()
        # Longest first: "Playfair Display" beats "Playfair", and the trailing
        # word a greedy match picks up ("Playfair Display Use") is dropped by
        # Figma simply not having a family called that.
        for length in range(len(words), 0, -1):
            name = " ".join(words[:length])
            if name not in found:
                found.append(name)
    return found[:MAX_FONT_CANDIDATES]


def pick_display_family(candidates: list[str], available: dict) -> str:
    """The brand font: the first CANDIDATE that Figma actually has.

    Never Inter -- that is the fallback, not a choice -- and never a name the
    file cannot render, which would throw the moment a text style used it.
    """
    real = {str(name).strip().lower(): str(name) for name in (available or {})}
    for candidate in candidates or []:
        key = candidate.strip().lower()
        if key == DEFAULT_FONT_FAMILY.lower():
            continue
        if key in real:
            return real[key]
    return ""


def match_style(wanted: str, available: list[str]) -> str:
    """The real style string for a weight, out of what the family actually has.

    Families spell the same weight differently -- Inter has "Semi Bold" with a
    space, Playfair Display has "SemiBold" without one -- and guessing throws
    "font not loaded". So the wanted weight is matched against the strings
    Figma reported, then degraded to the nearest weight the family does have.
    """
    def key(value: str) -> str:
        return re.sub(r"[^a-z]", "", str(value).lower())

    have = {key(style): style for style in available or []}
    if not have:
        return wanted
    ladder = {
        "bold": ("bold", "semibold", "medium", "regular"),
        "semibold": ("semibold", "demibold", "bold", "medium", "regular"),
        "medium": ("medium", "semibold", "regular"),
        "regular": ("regular", "book", "normal", "medium"),
    }
    for step in ladder.get(key(wanted), (key(wanted), "regular")):
        if step in have:
            return have[step]
    return next(iter(have.values()))


def ramp_fonts(display_family: str, families: dict) -> dict[str, tuple[str, str]]:
    """Ramp style name -> the (family, style) that style should really use.

    Resolved in Python from what Figma reported, so no generated script ever
    contains a font string nobody has checked.
    """
    inter_styles = list((families or {}).get(DEFAULT_FONT_FAMILY) or [])
    display_styles = list((families or {}).get(display_family) or []) if display_family else []
    resolved: dict[str, tuple[str, str]] = {}
    for name, style, _size, _line in TEXT_STYLES:
        if display_family and display_styles and name in DISPLAY_STYLES:
            resolved[name] = (display_family, match_style(style, display_styles))
        elif inter_styles:
            resolved[name] = (DEFAULT_FONT_FAMILY, match_style(style, inter_styles))
        else:
            resolved[name] = (DEFAULT_FONT_FAMILY, style)
    return resolved


_HEX = r"#[0-9a-fA-F]{6}"
# "--color-primary: #0066FF", "Primary Blue (#0066FF)", "Accent — #0066FF"
_NAMED = re.compile(rf"([A-Za-z][\w \-/]{{1,38}}?)\s*[:(\-–—]\s*[`'\"]?({_HEX})")


def extract_palette(brief: str, instruction: str = "", limit: int = 12) -> list[tuple[str, str]]:
    """Pull `name -> hex` pairs out of the instruction first, then the brief.

    The USER'S instruction is read first and is authoritative. A real run asked
    for nine named colours in a clean `* Deep background: #0B1020` list and got
    three tokens, because the palette was only ever read out of the model's
    rewritten brief -- which had scattered the same colours into table cells
    ("1 px solid #E5E7EB") that no `name: #hex` pattern can see. Six colours
    vanished, `border` and `surface` collapsed onto the page fill, and the
    result was white dividers on white panels.

    The brief still contributes whatever the instruction did not name, since
    most instructions specify no palette at all. Falls back to a neutral
    palette when neither has anything parseable.
    """
    found: list[tuple[str, str]] = []
    seen_hex: set[str] = set()
    seen_names: set[str] = set()

    pairs = _NAMED.findall(instruction or "") + _NAMED.findall(brief or "")
    for raw_name, hex_value in pairs:
        hex_value = hex_value.upper()
        if hex_value in seen_hex:
            continue
        name = _clean_name(raw_name)
        if not name:
            continue
        # Two different colours can clean to the same name -- a real brief had
        # "Accent: Orange #FF6600" and "Orange (#FFC107) for In Transit". Figma
        # rejects a duplicate variable name, and because scripts are atomic that
        # one collision destroyed the ENTIRE palette. Disambiguate instead.
        name = _unique_name(name, seen_names)
        seen_hex.add(hex_value)
        seen_names.add(name)
        found.append((name, hex_value))
        if len(found) >= limit:
            break

    return found or list(FALLBACK_PALETTE)


def _unique_name(name: str, taken: set[str]) -> str:
    """`orange` -> `orange-2` when `orange` is already used."""
    if name not in taken:
        return name
    for suffix in range(2, 100):
        candidate = f"{name}-{suffix}"
        if candidate not in taken:
            return candidate
    return name


def _clean_name(raw: str) -> str:
    """`--color-primary` / `Primary Blue` -> `primary`, `primary-blue`."""
    name = raw.strip().strip("-*`•").strip()
    name = re.sub(r"^(color|colour|token)[\s\-/]*", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^\w\-/ ]", "", name).strip()
    name = re.sub(r"[\s_]+", "-", name).lower()
    return name[:40].strip("-")


def hex_to_rgb(hex_value: str) -> tuple[float, float, float]:
    """Figma wants 0-1 floats, not 0-255 ints."""
    h = hex_value.lstrip("#")
    return tuple(round(int(h[i : i + 2], 16) / 255, 6) for i in (0, 2, 4))  # type: ignore[return-value]


# Roles are derived, never guessed. The palette comes out of the model's own
# brief with names like "deep-navy" or "cta", which tell the builder nothing
# about whether a colour is a background or a foreground -- so it bound text to
# whatever token sounded right and produced invisible copy that passed every
# gate. Luminance and contrast are arithmetic, so the harness computes them
# (CLAUDE.md section 7) and hands over the answer.

# WCAG AA for body text. Anything below this is not a legal text pairing.
MIN_TEXT_CONTRAST = 4.5

# Below this, a colour reads as a neutral rather than a brand accent.
ACCENT_MIN_CHROMA = 0.15


def relative_luminance_rgb(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance from 0-1 RGB (0 = black, 1 = white).

    The RGB form is the primitive: the critic reads colours back off the canvas
    as 0-1 floats (that is what the Plugin API stores), so making hex the only
    entry point would mean converting real colours to text and back to compare
    them.
    """
    channels = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio_rgb(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> float:
    """WCAG contrast ratio between two 0-1 RGB colours, 1.0 to 21.0."""
    la, lb = relative_luminance_rgb(a), relative_luminance_rgb(b)
    lighter, darker = max(la, lb), min(la, lb)
    return round((lighter + 0.05) / (darker + 0.05), 1)


def relative_luminance(hex_value: str) -> float:
    """WCAG relative luminance (0 = black, 1 = white)."""
    return relative_luminance_rgb(hex_to_rgb(hex_value))


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two colours, 1.0 (identical) to 21.0."""
    return contrast_ratio_rgb(hex_to_rgb(hex_a), hex_to_rgb(hex_b))


def _chroma(hex_value: str) -> float:
    rgb = hex_to_rgb(hex_value)
    return max(rgb) - min(rgb)


# Words in a token name that DECLARE what the colour is for. A brief that says
# "Border: #E5E7EB" has told us the answer, and luminance ordering alone got it
# wrong every time: in a real run #E5E7EB was ranked a "surface", #111827 became
# "text-muted", and the near-black page background was labelled "text". Names
# are only trusted for these unambiguous words -- everything else still falls
# out of the arithmetic below.
_ROLE_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("text", ("text", "ink", "foreground", "copy", "type")),
    ("border", ("border", "divider", "outline", "stroke", "rule")),
    ("surface", ("surface", "card", "panel", "input", "field", "elevated", "raised")),
    ("background", ("background", "bg", "canvas", "page", "base")),
    ("accent", ("accent", "primary", "cta", "brand", "highlight")),
]
# A modifier that turns "text" into "secondary copy".
_MUTED_WORDS = ("muted", "secondary", "subtle", "tertiary", "placeholder", "hint")

ROLE_LABELS = {
    "accent": "accent (buttons, links, emphasis)",
    "background": "background (the page fill this screen sits on)",
    "background-alt": "background-alt (inverse/dark panels, hero halves)",
    "surface": "surface (cards, inputs, raised areas)",
    "border": "border / divider",
    "text": "text (body copy on the background)",
    "text-muted": "text-muted (secondary copy)",
    "text-on-alt": "text-on-alt (copy sitting on background-alt)",
}
# The roles a palette must fill for the renderer to draw a distinguishable UI.
# Anything still missing after parsing is DERIVED (see complete_palette).
REQUIRED_ROLES = ("background", "surface", "border", "text", "text-muted", "accent")


def _name_hint(token_name: str) -> str | None:
    """The role a token's own name declares, or None if it declares nothing."""
    words = set(re.split(r"[-/_ ]+", token_name.lower()))
    for role, vocabulary in _ROLE_WORDS:
        if words & set(vocabulary):
            if role == "text" and words & set(_MUTED_WORDS):
                return "text-muted"
            return role
    return None


def assign_roles(entries: list[tuple[str, str]]) -> dict[str, str]:
    """token name -> role key, from the names the brief used and the arithmetic.

    Two sources, in that order. A name that says what it is wins, because it is
    the designer's own statement of intent; everything else is decided by
    luminance and chroma, which is what CLAUDE.md section 7 means by doing the
    arithmetic in Python rather than letting the model guess.
    """
    if not entries:
        return {}
    hexes = dict(entries)
    claims: dict[str, list[str]] = {}
    for name, _hex in entries:
        hint = _name_hint(name)
        if hint:
            claims.setdefault(hint, []).append(name)

    by_light = sorted(entries, key=lambda e: relative_luminance(e[1]), reverse=True)
    roles: dict[str, str] = {}
    taken: set[str] = set()

    def claim(role: str, name: str | None) -> None:
        if name and name not in taken:
            roles[name] = role
            taken.add(name)

    # Accent: the name wins, else the most saturated colour in the palette.
    accent = next(
        (n for n in claims.get("accent", []) if _chroma(hexes[n]) >= ACCENT_MIN_CHROMA), None
    )
    if accent is None:
        boldest = max(entries, key=lambda e: _chroma(e[1]))
        accent = boldest[0] if _chroma(boldest[1]) >= ACCENT_MIN_CHROMA else None
    claim("accent", accent)

    # Text: the darkest of whatever claims to be text, else the darkest colour.
    text = min(claims.get("text", []), key=lambda n: relative_luminance(hexes[n]), default=None)
    if text is None:
        text = next((n for n, _ in reversed(by_light) if n not in taken), None)
    claim("text", text)

    # Background: of everything claiming it, the one this screen's text can
    # actually be read on. A brief with a dark hero AND a white form panel has
    # two backgrounds, and picking the wrong one paints near-black text on a
    # near-black page -- which is exactly what a live run produced.
    backgrounds = [n for n in claims.get("background", []) if n not in taken]
    if text and backgrounds:
        background = max(backgrounds, key=lambda n: contrast_ratio(hexes[n], hexes[text]))
    elif backgrounds:
        background = backgrounds[0]
    else:
        background = next((n for n, _ in by_light if n not in taken), None)
    claim("background", background)

    # background-alt is the SECOND background: the inverse panel a modern
    # split-screen layout is built from. Without it the dark half of a
    # dark-to-light design is not expressible at all, so the model asked for
    # "background", got the white one, and rendered a 1440x900 white void.
    if background:
        rest = [n for n in backgrounds if n not in taken]
        alt = max(rest, key=lambda n: contrast_ratio(hexes[n], hexes[background]), default=None)
        claim("background-alt", alt)

    claim("surface", next((n for n in claims.get("surface", []) if n not in taken), None))
    claim("border", next((n for n in claims.get("border", []) if n not in taken), None))
    claim("text-muted", next((n for n in claims.get("text-muted", []) if n not in taken), None))

    # Whatever is left is ranked by luminance between the background and the
    # text -- the same reasoning as before, applied only to the colours that
    # did not declare themselves.
    filled = set(roles.values())
    leftover = [(n, h) for n, h in by_light if n not in taken]
    for position, (name, hex_value) in enumerate(leftover):
        if "surface" not in filled and relative_luminance(hex_value) >= 0.5:
            claim("surface", name)
            filled.add("surface")
        elif "text-muted" not in filled and position == len(leftover) - 1:
            claim("text-muted", name)
            filled.add("text-muted")
        elif "border" not in filled:
            claim("border", name)
            filled.add("border")

    # Copy that sits on the inverse panel. Measured, never assumed: whichever
    # colour in the palette is genuinely readable there.
    alt_name = next((n for n, r in roles.items() if r == "background-alt"), None)
    if alt_name:
        readable = [
            (contrast_ratio(h, hexes[alt_name]), n)
            for n, h in entries
            if n != alt_name and contrast_ratio(h, hexes[alt_name]) >= MIN_TEXT_CONTRAST
        ]
        if readable:
            roles.setdefault(max(readable)[1], "text-on-alt")
    return roles


def complete_palette(palette: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Add derived tokens for any role the brief left unfilled.

    A palette missing `border` used to alias it onto `surface`, which aliased
    onto `background` -- so a divider was painted white on a white panel and
    the input boxes had no visible edge at all. Aliasing a role onto the page
    fill is never a usable answer, so the missing colours are MIXED here
    instead: deterministic arithmetic from the two colours we do have, created
    as real tokens the renderer binds to like any other.
    """
    if not palette:
        return palette
    entries = [(f"color/{n}", h) for n, h in palette]
    roles = assign_roles(entries)
    have = set(roles.values())
    hexes = dict(entries)
    background = next((hexes[n] for n, r in roles.items() if r == "background"), "#FFFFFF")
    text = next((hexes[n] for n, r in roles.items() if r == "text"), "#111827")

    # How far each derived colour sits from the page fill, towards the ink.
    recipe = {"surface": 0.04, "border": 0.16, "text-muted": 0.55}
    existing_hexes = {h.upper() for _, h in palette}
    taken = {name for name, _ in palette}
    extra: list[tuple[str, str]] = []
    for role in ("surface", "border", "text-muted"):
        if role in have:
            continue
        derived = _mix(background, text, recipe[role])
        if derived in existing_hexes:
            continue
        existing_hexes.add(derived)
        name = _unique_name(role, taken)
        taken.add(name)
        extra.append((name, derived))
    return list(palette) + extra


def _mix(hex_a: str, hex_b: str, amount: float) -> str:
    """`amount` of the way from A to B, as a hex string."""
    a, b = hex_to_rgb(hex_a), hex_to_rgb(hex_b)
    channels = [round((x + (y - x) * amount) * 255) for x, y in zip(a, b)]
    return "#" + "".join(f"{max(0, min(255, c)):02X}" for c in channels)


def describe_palette(palette: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Annotate each token with a functional role: (token_name, hex, role).

    Token names keep whatever the brief called them -- the paint styles are
    already created under those names, so renaming here would break every
    lookup. The role is added alongside.
    """
    if not palette:
        return []
    entries = [(f"color/{name}", hex_value) for name, hex_value in palette]
    roles = assign_roles(entries)
    return [
        (name, hex_value, ROLE_LABELS.get(roles.get(name, ""), "decorative"))
        for name, hex_value in entries
    ]


def readable_pairings(palette: list[tuple[str, str]], limit: int = 8) -> list[str]:
    """Foreground/background pairs that actually meet WCAG AA, as prompt lines.

    Stating these as fact is what stops the builder inventing a pairing; every
    ratio here is computed, not asserted.
    """
    entries = [(f"color/{name}", hex_value) for name, hex_value in palette]
    scored = []
    for fg_name, fg_hex in entries:
        for bg_name, bg_hex in entries:
            if fg_name == bg_name:
                continue
            ratio = contrast_ratio(fg_hex, bg_hex)
            if ratio >= MIN_TEXT_CONTRAST:
                scored.append((ratio, f"{fg_name} text on {bg_name} background ({ratio}:1)"))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [line for _, line in scored[:limit]]


def build_token_script(palette: list[tuple[str, str]]) -> str:
    """One atomic script creating a variable collection, COLOR variables, and
    paint styles bound to them.

    Binding the styles to variables is what satisfies golden rule 5: nodes can
    then just use a style id and still be variable-backed.
    """
    entries = [
        {"name": name, "r": rgb[0], "g": rgb[1], "b": rgb[2]}
        for name, hex_value in palette
        for rgb in [hex_to_rgb(hex_value)]
    ]
    return _TOKEN_SCRIPT.replace("__ENTRIES__", json.dumps(entries))


# Two hard-won details:
#  - lookups are MAPS built once and updated as we go. Re-scanning a snapshot
#    taken before the loop meant a name created during the loop was invisible,
#    so a repeated name hit createVariable twice -> "duplicate variable name".
#  - each entry is wrapped in try/catch. Scripts are atomic, so without this a
#    single malformed colour discards every token in the palette, and the run
#    silently continues with no styles at all to bind to.
_TOKEN_SCRIPT = """\
const entries = __ENTRIES__;
const collections = await figma.variables.getLocalVariableCollectionsAsync();
let collection = null;
for (const c of collections) { if (c.name === 'Tokens') { collection = c; } }
if (!collection) { collection = figma.variables.createVariableCollection('Tokens'); }
const modeId = collection.modes[0].modeId;

const existingVars = await figma.variables.getLocalVariablesAsync();
const existingStyles = await figma.getLocalPaintStylesAsync();

const varByName = {};
for (const v of existingVars) { varByName[v.name] = v; }
const styleByName = {};
for (const s of existingStyles) { styleByName[s.name] = s; }

const names = [];
const failed = [];
const unbound = [];

for (const entry of entries) {
  const varName = 'color/' + entry.name;
  let paint = { type: 'SOLID', color: { r: entry.r, g: entry.g, b: entry.b } };

  // The VARIABLE is the nice-to-have. Its own try block, so a collection that
  // rejects one variable cannot cost us the paint style too -- a run that ends
  // up with one usable colour has nothing to bind text to, and produces
  // invisible copy on an accent background.
  try {
    let variable = varByName[varName];
    if (!variable) {
      variable = figma.variables.createVariable(varName, collection, 'COLOR');
      varByName[varName] = variable;
    }
    variable.setValueForMode(modeId, { r: entry.r, g: entry.g, b: entry.b, a: 1 });
    paint = figma.variables.setBoundVariableForPaint(paint, 'color', variable);
  } catch (e) {
    unbound.push(varName + ': ' + String(e && e.message ? e.message : e));
  }

  // The STYLE is what every later script actually looks up by name, so this is
  // the part that must not fail.
  try {
    let style = styleByName[varName];
    if (!style) {
      style = figma.createPaintStyle();
      style.name = varName;
      styleByName[varName] = style;
    }
    style.paints = [paint];
    names.push(varName);
  } catch (e) {
    failed.push(varName + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: [], tokenNames: names, failedTokens: failed, unboundTokens: unbound };
"""


def build_text_style_script(fonts: dict[str, tuple[str, str]] | None = None) -> str:
    """Create the type scale, skipping any that already exist.

    `fonts` maps a ramp style NAME to the (family, style) it should use,
    resolved against the fonts Figma really has (`ramp_fonts`). So a design
    that asks for Playfair Display headings gets them, and one that asks for
    nothing gets Inter exactly as before.
    """
    resolved = fonts or {}
    entries = []
    for name, style, size, line in TEXT_STYLES:
        family, real_style = resolved.get(name, (DEFAULT_FONT_FAMILY, style))
        entries.append({
            "name": name, "family": family, "style": real_style,
            "fallback": style, "size": size, "line": line,
        })
    return _TEXT_STYLE_SCRIPT.replace("__ENTRIES__", json.dumps(entries))


_TEXT_STYLE_SCRIPT = """\
const entries = __ENTRIES__;
const existing = await figma.getLocalTextStylesAsync();
const names = [];
const fellBack = [];
for (const entry of entries) {
  let font = { family: entry.family, style: entry.style };
  try {
    await figma.loadFontAsync(font);
  } catch (e) {
    // A family this file turns out not to have must cost ONE style, not the
    // whole ramp -- and the run is told, rather than quietly shipping Inter.
    font = { family: 'Inter', style: entry.fallback };
    await figma.loadFontAsync(font);
    fellBack.push(entry.name + ' (' + entry.family + ')');
  }
  let style = null;
  for (const s of existing) { if (s.name === entry.name) { style = s; } }
  if (!style) { style = figma.createTextStyle(); style.name = entry.name; }
  style.fontName = font;
  style.fontSize = entry.size;
  style.lineHeight = { unit: 'PIXELS', value: entry.line };
  names.push(entry.name);
}
return { createdNodeIds: [], textStyleNames: names, fellBack: fellBack };
"""


# How close a hardcoded colour must be to a token before we bind it to that
# token automatically. 0.06 in 0-1 RGB space is visually indistinguishable, so
# this only ever "fixes" a colour the model clearly meant to be a token -- it
# never silently restyles a deliberate one-off colour.
BIND_TOLERANCE = 0.06


def build_bind_fills_script(node_ids: list[str], palette: list[tuple[str, str]]) -> str:
    """Replace hardcoded fills with the token they were obviously meant to be.

    Warning about unbound fills doesn't make a design dynamic -- rebinding
    them does. Anything not close to a token is left alone and still reported,
    because changing it would be a design decision, not a fix.
    """
    tokens = [
        {"name": f"color/{name}", "r": rgb[0], "g": rgb[1], "b": rgb[2]}
        for name, hex_value in palette
        for rgb in [hex_to_rgb(hex_value)]
    ]
    return (
        _BIND_FILLS_SCRIPT
        .replace("__IDS__", json.dumps(node_ids))
        .replace("__TOKENS__", json.dumps(tokens))
        .replace("__TOL__", str(BIND_TOLERANCE))
    )


_BIND_FILLS_SCRIPT = """\
const ids = __IDS__;
const tokens = __TOKENS__;
const tolerance = __TOL__;

const styles = await figma.getLocalPaintStylesAsync();
const styleByName = {};
for (const s of styles) { styleByName[s.name] = s; }

const bound = [];
const unbound = [];

for (const id of ids) {
  const node = await figma.getNodeByIdAsync(id);
  if (!node || !('fills' in node)) continue;
  if ('fillStyleId' in node && node.fillStyleId) continue;   // already token-backed
  const fills = node.fills;
  if (!Array.isArray(fills) || fills.length === 0) continue;

  const paint = fills[0];
  if (!paint || paint.type !== 'SOLID') continue;
  if (paint.boundVariables && paint.boundVariables.color) continue;

  let best = null;
  let bestDistance = Infinity;
  for (const token of tokens) {
    const dr = paint.color.r - token.r;
    const dg = paint.color.g - token.g;
    const db = paint.color.b - token.b;
    const distance = Math.sqrt(dr * dr + dg * dg + db * db);
    if (distance < bestDistance) { bestDistance = distance; best = token; }
  }

  if (best && bestDistance <= tolerance && styleByName[best.name]) {
    await node.setFillStyleIdAsync(styleByName[best.name].id);
    bound.push({ id: id, token: best.name });
  } else {
    unbound.push(id);
  }
}
return { createdNodeIds: [], boundNodes: bound, stillUnbound: unbound };
"""


def build_hug_fix_script(root_id: str) -> str:
    """Make an auto-layout frame hug its content again.

    `resize()` silently resets sizing to FIXED, after which the frame clips
    everything added below the fold -- the cause of a run reporting content
    2px past a 2233px-tall root.
    """
    return _HUG_FIX_SCRIPT.format(root_id=json.dumps(root_id))


_HUG_FIX_SCRIPT = """\
const root = await figma.getNodeByIdAsync({root_id});
if (!root || !('layoutMode' in root) || root.layoutMode === 'NONE') {{
  return {{ createdNodeIds: [], changed: false }};
}}
const before = Math.round(root.height);
root.primaryAxisSizingMode = 'AUTO';
return {{ createdNodeIds: [], changed: Math.round(root.height) !== before,
          height: Math.round(root.height) }};
"""


def build_remove_nodes_script(node_ids: list[str]) -> str:
    """Delete specific nodes, tolerating ones that are already gone.

    A step that exhausted its retries used to leave its broken section on the
    canvas AND get a TODO placeholder appended beneath it, so the page showed
    both the failure and the marker for it -- a 1440x900 white void followed by
    a "TODO: left visual panel" band. A failed final repair was worse: it had
    already replaced the section, then ran out of turns, and its half-built
    copy simply stayed. Four stacked copies of one sign-in form is what that
    looks like on the canvas.

    Wrapped per id, because scripts are atomic and one stale id must not cost
    the whole cleanup.
    """
    return _REMOVE_NODES_SCRIPT.replace("__IDS__", json.dumps(list(node_ids)))


_REMOVE_NODES_SCRIPT = """const ids = __IDS__;
const removed = [];
const skipped = [];
for (const id of ids) {
  try {
    const node = await figma.getNodeByIdAsync(id);
    if (!node || node.removed) { continue; }
    // A top-level frame is a whole SCREEN. No cleanup pass has any business
    // removing one, and refusing here closes the class of mistake rather than
    // the one instance of it that was found.
    if (node.parent && node.parent.type === 'PAGE') { skipped.push(id); continue; }
    node.remove(); removed.push(id);
  } catch (e) {}
}
return { createdNodeIds: [], removedNodeIds: removed, skippedNodeIds: skipped };
"""


def build_placeholder_section_script(
    root_id: str,
    label: str,
    height: int = 160,
    surface_style: str = "",
    text_style: str = "",
) -> str:
    """A labelled stand-in for a section whose step failed.

    Keeps the page's vertical rhythm intact and makes the gap obvious in the
    file, instead of silently leaving a hole where a section should be.

    The two styles are passed in RESOLVED. The script used to hunt for a style
    whose name merely contained "surface" or "text", which in a real run picked
    a near-black page background and then found nothing at all for the label --
    so the marker rendered as grey-on-navy at 2.3:1, and the run reported its
    own scaffolding as a contrast defect.
    """
    return _PLACEHOLDER_SCRIPT.format(
        root_id=json.dumps(root_id),
        label=json.dumps(label[:60]),
        height=height,
        surface_style=json.dumps(surface_style),
        text_style=json.dumps(text_style),
    )


# The placeholder used a 14px label and two hardcoded greys, so the harness's
# own scaffolding failed the harness's own design checks -- a finished run
# reported six "issues" that were all about its own TODO frames. It now uses
# the token styles and the 13px Caption size from the type ramp, and falls back
# to a legible grey only when a run has no styles at all.
_PLACEHOLDER_SCRIPT = """\
const root = await figma.getNodeByIdAsync({root_id});
if (!root) {{ throw new Error('root frame not found'); }}
await figma.loadFontAsync({{ family: 'Inter', style: 'Regular' }});

const paints = await figma.getLocalPaintStylesAsync();
const byName = {{}};
for (const s of paints) {{ byName[s.name] = s; }}
// Exact names, resolved by the harness from the palette ROLES. Substring
// matching picked whichever style happened to contain the word.
function styleNamed(wanted) {{
  return wanted && byName[wanted] ? byName[wanted] : null;
}}

const section = figma.createFrame();
section.name = 'TODO — ' + {label};
section.resize(root.width, {height});
root.appendChild(section);
section.layoutMode = 'VERTICAL';
section.primaryAxisAlignItems = 'CENTER';
section.counterAxisAlignItems = 'CENTER';
section.paddingTop = 24; section.paddingBottom = 24;
section.layoutSizingHorizontal = 'FILL';
section.layoutSizingVertical = 'FIXED';
const surface = styleNamed({surface_style});
if (surface) {{ await section.setFillStyleIdAsync(surface.id); }}
else {{ section.fills = [{{ type: 'SOLID', color: {{ r: 0.97, g: 0.97, b: 0.98 }} }}]; }}

const label = figma.createText();
label.fontName = {{ family: 'Inter', style: 'Regular' }};
label.characters = 'TODO: ' + {label};
label.fontSize = 13;                       // on the type ramp
const ink = styleNamed({text_style});
if (ink) {{ await label.setFillStyleIdAsync(ink.id); }}
else {{ label.fills = [{{ type: 'SOLID', color: {{ r: 0.29, g: 0.31, b: 0.36 }} }}]; }}
section.appendChild(label);

return {{ createdNodeIds: [section.id] }};
"""


# Gap between two screen frames on the page. Wide enough that the frames read
# as separate screens rather than one design that happens to have a seam.
SCREEN_GAP = 160

# Where the first screen goes on an empty page. Never (0,0) -- gotchas.md.
SCREEN_START_X = 200
SCREEN_START_Y = 200

# Clearance left between work that was already on the page and anything this
# run creates, so a second run never lands on top of the first.
EXISTING_CLEARANCE = 200

# Frame width per device. A screen's device is decided once, in the planner,
# because "a mobile sign-in screen and a desktop dashboard" used to produce two
# frames of identical width -- the instruction was read for ONE width and every
# screen got it.
# The VIEWPORT each device shows: one screenful. A page frame is a whole
# number of these tall (see `build_fit_screens_script`), so a screen that fits
# is exactly the size of the real thing.
DEVICE_WIDTHS: dict[str, tuple[int, int]] = {
    "mobile": (390, 844),      # iPhone 14
    "tablet": (834, 1194),     # iPad Pro 11"
    "desktop": (1440, 1024),   # Figma's own Desktop preset -- a laptop screen
}


def device_size(device: str) -> tuple[int, int]:
    """(width, height) for a device name, defaulting to desktop."""
    return DEVICE_WIDTHS.get((device or "").strip().lower(), DEVICE_WIDTHS["desktop"])


def occupied_rects(nodes: list[dict]) -> list[tuple[int, int, int, int]]:
    """(left, top, right, bottom) for everything already on the page.

    Anything without real geometry is skipped rather than treated as a rect at
    the origin -- a node we cannot measure must not push every screen to the
    right of a phantom box.
    """
    rects: list[tuple[int, int, int, int]] = []
    for node in nodes or []:
        try:
            x = int(node.get("x") or 0)
            y = int(node.get("y") or 0)
            width = int(node.get("width") or 0)
            height = int(node.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        rects.append((x, y, x + width, y + height))
    return rects


def place_screens(
    sizes: list[tuple[int, int]],
    occupied: list[tuple[int, int, int, int]] | None = None,
    y: int = SCREEN_START_Y,
    gap: int = SCREEN_GAP,
    clearance: int = EXISTING_CLEARANCE,
) -> list[tuple[int, int]]:
    """Left-to-right positions for new screens that can overlap NOTHING.

    Placement is arithmetic, so Python owns it (CLAUDE.md section 7) and the
    guarantee is checkable rather than hoped for: no returned rect touches
    another, and none touches anything already on the page.

    Two holes this closes that a running `x += width + gap` had:

    - **Screens of different widths.** The advance used one shared width, so a
      390px mobile frame beside a 1440px desktop one either overlapped it or
      left a chasm. Each screen now advances by its OWN width.
    - **Existing work below or beside the row.** The start was "right of the
      widest thing on the page", which is clear of a tall sketch but not of a
      second one further right, and not of anything that appears at the row's
      own height. Every candidate is now tested against every rect and slid
      right until it is genuinely clear.
    """
    taken = list(occupied or [])
    placed: list[tuple[int, int]] = []
    x = _first_free_x(taken, clearance)

    for width, height in sizes:
        x = _slide_clear(x, y, int(width), int(height), taken, gap)
        placed.append((x, y))
        taken.append((x, y, x + int(width), y + int(height)))
        x += int(width) + gap
    return placed


def _first_free_x(taken: list[tuple[int, int, int, int]], clearance: int) -> int:
    """Start to the right of everything already on the page."""
    if not taken:
        return SCREEN_START_X
    return max(right for _, _, right, _ in taken) + clearance


def _slide_clear(
    x: int,
    y: int,
    width: int,
    height: int,
    taken: list[tuple[int, int, int, int]],
    gap: int,
) -> int:
    """Push x right until this rect clears every rect already taken.

    Bounded by the number of rects: each pass moves past at least one of them,
    so it cannot loop -- an unbounded "try again" here would hang the run
    before a single node was created.
    """
    for _ in range(len(taken) + 1):
        blocker = _first_blocker(x, y, width, height, taken, gap)
        if blocker is None:
            return x
        x = blocker[2] + gap
    return x


def _first_blocker(
    x: int,
    y: int,
    width: int,
    height: int,
    taken: list[tuple[int, int, int, int]],
    gap: int,
) -> tuple[int, int, int, int] | None:
    """The furthest-right rect this candidate collides with, gap included.

    Furthest-right, so one pass jumps past a whole cluster rather than landing
    between two overlapping boxes and having to slide again.
    """
    hits = [
        rect
        for rect in taken
        if _collides((x, y, x + width, y + height), rect, gap)
    ]
    return max(hits, key=lambda r: r[2]) if hits else None


def _collides(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    """Do two rects overlap, counting the gap that must stay between them?"""
    return (
        a[0] < b[2] + gap
        and a[2] + gap > b[0]
        and a[1] < b[3] + gap
        and a[3] + gap > b[1]
    )


def build_screen_frames_script(specs: list[dict]) -> str:
    """Create one top-level FRAME per screen, laid out left to right.

    This is Figma's own structure, which the agent used to ignore: a PAGE is a
    workspace and a FRAME is a screen. Several screens are sibling frames side
    by side on one page -- never sections stacked into a single frame, and
    never separate Figma pages.

    Harness-authored (CLAUDE.md section 6a) because it is completely
    mechanical and because getting it wrong cascades: every later step appends
    into one of these frames, so a frame at the wrong coordinates means an
    entire screen rendered on top of another one.

    Each spec is {name, x, y, width}. Positions are computed in Python (section
    7: do arithmetic in Python), so two screens can never overlap.
    """
    return _SCREEN_FRAMES_SCRIPT.replace("__SPECS__", json.dumps(specs))


# Wrapped per entry: scripts are atomic, so one bad spec would otherwise
# discard every screen and leave the run with nowhere to build.
_SCREEN_FRAMES_SCRIPT = """const specs = __SPECS__;
const made = [];
const ids = [];
const failed = [];

for (const spec of specs) {
  try {
    const frame = figma.createFrame();
    frame.name = spec.name;
    frame.resize(spec.width, spec.height || 900);
    figma.currentPage.appendChild(frame);
    frame.x = spec.x;
    frame.y = spec.y;
    frame.layoutMode = 'VERTICAL';
    frame.counterAxisSizingMode = 'FIXED';
    frame.primaryAxisSizingMode = 'AUTO';
    frame.itemSpacing = 0;
    frame.fills = [{ type: 'SOLID', color: { r: 1, g: 1, b: 1 } }];
    made.push({ name: spec.name, id: frame.id });
    ids.push(frame.id);
  } catch (e) {
    failed.push(spec.name + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: ids, screens: made, failedScreens: failed };
"""


# A page may be taller than one screenful -- a landing page always is -- but it
# should be a whole number of screenfuls, not a ragged number ending wherever
# the last section happened to stop.
MAX_VIEWPORTS_PER_SCREEN = 6

# How much blank canvas rounding up to ANOTHER whole viewport may add, as a
# share of one viewport. Past this the frame keeps its content height instead.
#
# "A page frame is a whole number of screenfuls tall" is the right rule and it
# had one bad case, which a real run hit squarely: content a little OVER one
# viewport rounds to two, so a design specified as 1440x900 shipped 1440x2048 --
# a screenful of empty canvas under it, and every full-height column stretched
# down into the void. Snapping is worth some slack; it is not worth most of a
# screen.
#
# It only applies from the SECOND screenful on. A page whose content fits
# inside one viewport is always snapped up to it: that is the property that
# makes every page that fits come out exactly 1440x1024.
MAX_SNAP_SLACK = 0.5


def build_fit_screens_script(specs: list[dict]) -> str:
    """Size every screen frame to a whole number of its own viewports.

    Two failures this replaces, both from real runs and both visible as blank
    canvas:

    - **Frames hugged their content**, so a row of pages ended at assorted
      heights and read as a mistake rather than a design.
    - **Levelling them all to the TALLEST** fixed that and created something
      worse: a Category page whose content stops halfway down a frame sized for
      a five-section landing page, more than half of it empty. "Equal" bought
      at the price of a screen and a half of nothing is not a good trade.

    Rounding up to a viewport gives both. A page that fits is EXACTLY
    1440x1024 -- a real laptop screen, which is what a mockup is supposed to
    look like -- and pages that fit are also all identical, which is what
    "equal" was asking for. A long page becomes 2048 or 3072: still a clean
    number, still obviously the same design system, and never more than one
    screenful of slack.

    ...and it snaps only when snapping is nearly free. Rounding content that
    is a little over one viewport up to TWO adds most of a screen of blank
    canvas, which is the very thing this pass replaced. Past `MAX_SNAP_SLACK`
    the frame keeps its content height instead: slightly ragged beats visibly
    empty.

    Each spec is {id, viewport}. Arithmetic, so Python owns it (section 7).
    """
    return (
        _FIT_SCREENS_SCRIPT.replace("__SPECS__", json.dumps(specs))
        .replace("__MAX_VIEWPORTS__", str(int(MAX_VIEWPORTS_PER_SCREEN)))
        .replace("__MAX_SLACK__", repr(float(MAX_SNAP_SLACK)))
    )


# Wrapped per frame: one screen that cannot be resized must not cost the rest
# their sizing.
_FIT_SCREENS_SCRIPT = """const specs = __SPECS__;
const MAX_VIEWPORTS = __MAX_VIEWPORTS__;
const MAX_SLACK = __MAX_SLACK__;
const sized = [];
const failed = [];

for (const spec of specs) {
  try {
    const frame = await figma.getNodeByIdAsync(spec.id);
    if (!frame || frame.removed || !('resize' in frame)) { continue; }
    const viewport = Math.max(1, Math.round(spec.viewport));
    // What the content needs right now. The frame has been hugging until this
    // point, which is what stopped a long page clipping.
    const content = Math.ceil(frame.height);
    let screens = Math.ceil(content / viewport);
    if (screens < 1) { screens = 1; }
    if (screens > MAX_VIEWPORTS) { screens = MAX_VIEWPORTS; }
    const target = viewport * screens;
    // Snap only when it is nearly free. Content 150px over one viewport would
    // otherwise round to two, and the extra screenful is blank canvas.
    const snapped = screens <= 1 || (target - content) <= viewport * MAX_SLACK;
    const height = snapped ? Math.max(target, content) : content;
    // A hugging frame ignores a resize on that axis, so the mode goes first.
    if ('primaryAxisSizingMode' in frame) { frame.primaryAxisSizingMode = 'FIXED'; }
    // Never SHORTER than the content: cutting a page down is the clipping the
    // hugging existed to avoid, and the cap only bites on a runaway page.
    frame.resize(frame.width, height);
    sized.push({
      id: frame.id, name: frame.name, screens: screens, snapped: snapped,
      height: Math.round(frame.height)
    });
  } catch (e) {
    failed.push(spec.id + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: [], screens: sized, failed: failed };
"""


def build_screen_content_script(frame_ids: list[str]) -> str:
    """How many children each screen frame really has.

    Ground truth, because the in-memory answer is wrong. `record_section` only
    fires for steps the keyword heuristic calls a section (`_is_section_step`),
    so a screen built by a step phrased any other way looks empty from Python
    while being full on the canvas. Reading it back is CLAUDE.md's rule for
    exactly this: never assume canvas state.
    """
    return _SCREEN_CONTENT_SCRIPT.replace(
        "__IDS__", json.dumps([str(i) for i in frame_ids if i])
    )


_SCREEN_CONTENT_SCRIPT = """const ids = __IDS__;
const CONTROLS = /button|input|field|checkbox|toggle|select|link|cta|nav|tab/i;
const MAX_DEPTH = 8;

// Not just "does this frame have children". A screen that got its artwork and
// none of its form has one child and looks built from the outside -- which is
// exactly how a Sign Up screen shipped as a picture and a quote.
function tally(node, depth, acc) {
  if (!('children' in node) || depth > MAX_DEPTH) { return; }
  for (const child of node.children) {
    acc.nodes += 1;
    if (child.type === 'TEXT' && String(child.characters).trim()) { acc.texts += 1; }
    if (CONTROLS.test(child.name)) { acc.controls += 1; }
    tally(child, depth + 1, acc);
  }
}

const screens = [];
for (const id of ids) {
  const node = await figma.getNodeByIdAsync(id);
  if (!node || node.removed) { continue; }
  const acc = { nodes: 0, texts: 0, controls: 0 };
  tally(node, 0, acc);
  screens.push({
    id: node.id,
    name: node.name,
    children: 'children' in node ? node.children.length : 0,
    nodes: acc.nodes,
    texts: acc.texts,
    controls: acc.controls
  });
}
return { createdNodeIds: [], screens: screens };
"""


# A real screen has more copy on it than this. Below it, whatever is there is
# scaffolding -- a picture, a heading, a stray label -- and the screen the user
# asked for was never built.
MIN_SCREEN_TEXTS = 3
# ...and a screen with a fraction of what its siblings got is unfinished even
# when it clears that bar on its own.
THIN_VS_SIBLING = 4


def thin_screen_reason(row: dict, richest_texts: int) -> str:
    """Why this screen looks half-built, or "" if it looks finished.

    Arithmetic on what is really on the canvas (CLAUDE.md section 7), and
    deliberately conservative: the cost of a false positive is one extra build
    attempt, and the cost of a false NEGATIVE is shipping a screen that is a
    picture and a quote where a sign-up form should be.
    """
    texts = int(row.get("texts") or 0)
    controls = int(row.get("controls") or 0)
    if texts <= 0:
        return "nothing on it reads as content"
    if texts < MIN_SCREEN_TEXTS and controls == 0:
        return f"only {texts} piece(s) of copy and nothing to interact with"
    if richest_texts >= MIN_SCREEN_TEXTS * THIN_VS_SIBLING and texts * THIN_VS_SIBLING <= richest_texts:
        return f"{texts} pieces of copy where another screen has {richest_texts}"
    return ""


def build_clear_page_script() -> str:
    """Remove every top-level layer from the current page.

    This is as close to "delete the design" as a plugin can get. The Plugin API
    has no way to delete a FILE -- there is no `figma.deleteFile`, and
    `figma.fileKey` is read-only (verified against the real typings) -- so what
    a plugin can offer is emptying the canvas it is running in.

    Destructive, and only ever called behind an explicit confirmation. Figma's
    own undo still applies, which is the safety net that makes it reasonable to
    offer at all.
    """
    return _CLEAR_PAGE_SCRIPT


# `children` is live while we remove from it, so iterate over a COPY -- removing
# in place skips every other node. Each removal is wrapped because one locked or
# already-detached node must not abandon the rest.
_CLEAR_PAGE_SCRIPT = """const page = figma.currentPage;
const nodes = page.children.slice();
let removed = 0;
const failed = [];
for (const node of nodes) {
  try {
    node.remove();
    removed = removed + 1;
  } catch (e) {
    failed.push(node.name + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: [], removed: removed, failed: failed, pageName: page.name };
"""
