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

_HEX = r"#[0-9a-fA-F]{6}"
# "--color-primary: #0066FF", "Primary Blue (#0066FF)", "Accent — #0066FF"
_NAMED = re.compile(rf"([A-Za-z][\w \-/]{{1,38}}?)\s*[:(\-–—]\s*[`'\"]?({_HEX})")


def extract_palette(brief: str, limit: int = 12) -> list[tuple[str, str]]:
    """Pull `name -> hex` pairs out of the model's own design brief.

    The brief reliably lists colours (that is what we asked it for); it is
    turning them into Plugin API calls that it fails at. Falls back to a
    neutral palette if nothing parseable is there.
    """
    found: list[tuple[str, str]] = []
    seen_hex: set[str] = set()
    seen_names: set[str] = set()

    for raw_name, hex_value in _NAMED.findall(brief or ""):
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


def relative_luminance(hex_value: str) -> float:
    """WCAG relative luminance (0 = black, 1 = white)."""
    channels = []
    for value in hex_to_rgb(hex_value):
        channels.append(value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)
    r, g, b = channels
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex_a: str, hex_b: str) -> float:
    """WCAG contrast ratio between two colours, 1.0 (identical) to 21.0."""
    a, b = relative_luminance(hex_a), relative_luminance(hex_b)
    lighter, darker = max(a, b), min(a, b)
    return round((lighter + 0.05) / (darker + 0.05), 1)


def _chroma(hex_value: str) -> float:
    rgb = hex_to_rgb(hex_value)
    return max(rgb) - min(rgb)


def describe_palette(palette: list[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """Annotate each token with a functional role: (token_name, hex, role).

    Token names keep whatever the brief called them -- the paint styles are
    already created under those names, so renaming here would break every
    lookup. The role is added alongside.
    """
    if not palette:
        return []

    entries = [(f"color/{name}", hex_value) for name, hex_value in palette]
    accent = max(entries, key=lambda e: _chroma(e[1]))
    has_accent = _chroma(accent[1]) >= ACCENT_MIN_CHROMA

    neutrals = [e for e in entries if not (has_accent and e[0] == accent[0])]
    neutrals.sort(key=lambda e: relative_luminance(e[1]), reverse=True)  # lightest first

    roles: dict[str, str] = {}
    if has_accent:
        roles[accent[0]] = "accent (buttons, links, emphasis)"
    for position, (name, _) in enumerate(neutrals):
        from_end = len(neutrals) - 1 - position
        hex_value = neutrals[position][1]
        if position == 0:
            roles[name] = "background (page/section fill)"
        elif from_end == 0:
            roles[name] = "text (body copy on a light background)"
        elif relative_luminance(hex_value) >= 0.5:
            # Light enough to sit under content.
            roles[name] = "surface (cards, inputs, raised areas)"
        elif from_end == 1:
            # Dark, but not the darkest -- that reads as secondary copy, not a
            # card fill, however close to the top of the list it happens to be.
            roles[name] = "text-muted (secondary copy)"
        else:
            roles[name] = "border / divider"

    return [(name, hex_value, roles.get(name, "decorative")) for name, hex_value in entries]


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

for (const entry of entries) {
  const varName = 'color/' + entry.name;
  try {
    let variable = varByName[varName];
    if (!variable) {
      variable = figma.variables.createVariable(varName, collection, 'COLOR');
      varByName[varName] = variable;
    }
    variable.setValueForMode(modeId, { r: entry.r, g: entry.g, b: entry.b, a: 1 });

    let style = styleByName[varName];
    if (!style) {
      style = figma.createPaintStyle();
      style.name = varName;
      styleByName[varName] = style;
    }
    let paint = { type: 'SOLID', color: { r: entry.r, g: entry.g, b: entry.b } };
    paint = figma.variables.setBoundVariableForPaint(paint, 'color', variable);
    style.paints = [paint];
    names.push(varName);
  } catch (e) {
    failed.push(varName + ': ' + String(e && e.message ? e.message : e));
  }
}
return { createdNodeIds: [], tokenNames: names, failedTokens: failed };
"""


def build_text_style_script() -> str:
    """Create the default type scale, skipping any that already exist."""
    entries = [
        {"name": name, "style": style, "size": size, "line": line}
        for name, style, size, line in TEXT_STYLES
    ]
    return _TEXT_STYLE_SCRIPT.replace("__ENTRIES__", json.dumps(entries))


_TEXT_STYLE_SCRIPT = """\
const entries = __ENTRIES__;
const existing = await figma.getLocalTextStylesAsync();
const names = [];
for (const entry of entries) {
  await figma.loadFontAsync({ family: 'Inter', style: entry.style });
  let style = null;
  for (const s of existing) { if (s.name === entry.name) { style = s; } }
  if (!style) { style = figma.createTextStyle(); style.name = entry.name; }
  style.fontName = { family: 'Inter', style: entry.style };
  style.fontSize = entry.size;
  style.lineHeight = { unit: 'PIXELS', value: entry.line };
  names.push(entry.name);
}
return { createdNodeIds: [], textStyleNames: names };
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


def build_placeholder_section_script(root_id: str, label: str, height: int = 160) -> str:
    """A labelled stand-in for a section whose step failed.

    Keeps the page's vertical rhythm intact and makes the gap obvious in the
    file, instead of silently leaving a hole where a section should be.
    """
    return _PLACEHOLDER_SCRIPT.format(
        root_id=json.dumps(root_id), label=json.dumps(label[:60]), height=height
    )


_PLACEHOLDER_SCRIPT = """\
const root = await figma.getNodeByIdAsync({root_id});
if (!root) {{ throw new Error('root frame not found'); }}
await figma.loadFontAsync({{ family: 'Inter', style: 'Regular' }});

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
section.fills = [{{ type: 'SOLID', color: {{ r: 0.97, g: 0.97, b: 0.98 }} }}];

const label = figma.createText();
label.fontName = {{ family: 'Inter', style: 'Regular' }};
label.characters = 'TODO: ' + {label};
label.fontSize = 14;
label.fills = [{{ type: 'SOLID', color: {{ r: 0.45, g: 0.47, b: 0.52 }} }}];
section.appendChild(label);

return {{ createdNodeIds: [section.id] }};
"""
