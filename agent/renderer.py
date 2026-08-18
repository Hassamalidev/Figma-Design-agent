"""Compile a declarative UI tree into correct Figma Plugin API JavaScript.

This is CLAUDE.md rule 7 ("if it is mechanical, the harness writes it") applied
to the last and largest thing still delegated to the model: the JavaScript
itself.

Asking a 20B model to write Plugin API code produced the same failures every
run -- `FILL can only be set on children of auto-layout frames`, `not a
function`, components nested inside components, text nodes collapsed to 0px,
sections built three times. None of those are design mistakes. They are
mistakes about an API, and the API never changes, so the harness can own it.

The model now emits WHAT to build:

    {"kind": "section", "name": "KPI Cards", "direction": "row", "gap": "md",
     "children": [
       {"kind": "card", "children": [
         {"kind": "text", "style": "Caption", "color": "text-muted", "value": "Total Revenue"},
         {"kind": "text", "style": "Display", "value": "$128,430"},
         {"kind": "badge", "tone": "success", "label": "+12.5%"}]}]}

...and this module decides HOW: font loading, creation order, resize before
sizing modes, append before FILL/HUG, style lookup, hug-by-default text.

Python does the recursion and emits a flat sequence of statements, so the
generated JavaScript has no control flow to get wrong.
"""
from __future__ import annotations

import json

# The 8px scale. The model picks a NAME, never a number, so off-scale spacing
# is not expressible rather than merely discouraged.
SPACING = {"none": 0, "xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 32, "2xl": 48}
RADIUS = {"none": 0, "sm": 6, "md": 8, "lg": 12, "xl": 16}

# Maps to the text styles bootstrap_tokens creates, plus the Inter weight each
# one needs loaded. Keep in step with scaffold.TEXT_STYLES.
TEXT_STYLES = {
    "Display": ("Bold", 48),
    "Heading": ("Semi Bold", 32),
    "Subheading": ("Semi Bold", 20),
    "Body": ("Regular", 16),
    "Caption": ("Regular", 13),
    "Button": ("Semi Bold", 15),
}
DEFAULT_TEXT_STYLE = "Body"

# Semantic tones a badge/status can use, resolved against the palette roles.
TONES = ("success", "warning", "error", "info", "accent", "neutral")

MAX_NODES = 400  # a runaway spec must not lock up Figma


class SpecError(ValueError):
    """The spec is not renderable. The message goes back to the model verbatim."""


def _js(value) -> str:
    """JSON is valid JS for the literals we emit, and it escapes quotes for us."""
    return json.dumps(value)


class _Compiler:
    def __init__(self, parent_id: str, color_roles: dict[str, str]):
        self.parent_id = parent_id
        # role -> real paint style name, e.g. {"text": "color/dark-gray"}
        self.roles = color_roles
        self.lines: list[str] = []
        self.fonts: set[tuple[str, str]] = set()
        self.created: list[str] = []
        self.count = 0

    def var(self) -> str:
        self.count += 1
        if self.count > MAX_NODES:
            raise SpecError(f"spec has more than {MAX_NODES} nodes; split it across steps")
        return f"n{self.count}"

    # -- helpers -----------------------------------------------------------

    def fill(self, var: str, role: str | None) -> None:
        """Apply a paint STYLE by name. Styles are variable-bound already, so
        this is what keeps the design token-backed without the model ever
        touching a hex value."""
        if not role:
            return
        style = self.roles.get(role)
        if not style:
            return  # unknown role: leave the default fill rather than guessing
        self.lines.append(f"  await applyFill({var}, {_js(style)});")

    def frame(self, node: dict, parent: str, default_dir: str = "col") -> str:
        var = self.var()
        direction = node.get("direction", default_dir)
        layout = "HORIZONTAL" if direction == "row" else "VERTICAL"
        gap = SPACING.get(str(node.get("gap", "md")), SPACING["md"])
        pad = SPACING.get(str(node.get("padding", "none")), 0)
        radius = RADIUS.get(str(node.get("radius", "none")), 0)

        self.lines.append(f"  const {var} = figma.createFrame();")
        self.lines.append(f"  {var}.name = {_js(str(node.get('name', 'Frame'))[:60])};")
        self.lines.append(f"  {var}.layoutMode = {_js(layout)};")
        self.lines.append(f"  {var}.itemSpacing = {gap};")
        if pad:
            self.lines.append(
                f"  {var}.paddingTop = {pad}; {var}.paddingBottom = {pad};"
                f" {var}.paddingLeft = {pad}; {var}.paddingRight = {pad};"
            )
        if radius:
            self.lines.append(f"  {var}.cornerRadius = {radius};")
        # Hug by default in BOTH axes: a frame that hugs can never clip its own
        # content, which is the defect the geometry gate reports most often.
        self.lines.append(f"  {var}.primaryAxisSizingMode = 'AUTO';")
        self.lines.append(f"  {var}.counterAxisSizingMode = 'AUTO';")
        if node.get("align"):
            self.lines.append(f"  {var}.counterAxisAlignItems = {_js(node['align'])};")
        self.fill(var, node.get("background"))
        self.lines.append(f"  {parent}.appendChild({var});")
        # Stretch to the parent's width unless the node opts out (buttons and
        # badges hug their label). `setFill` re-checks at runtime that the
        # parent really is auto-layout, so this can never throw.
        # A FIXED width is what a sidebar is: 240px beside a filling content
        # column. It must not also stretch, so it opts out of FILL.
        if node.get("width"):
            width = int(node["width"])
            axis = "primaryAxisSizingMode" if direction == "row" else "counterAxisSizingMode"
            self.lines.append(f"  {var}.{axis} = 'FIXED';")
            self.lines.append(f"  {var}.resize({width}, {var}.height);")
        elif node.get("fill", True):
            self.lines.append(f"  setFill({var});")
        if node.get("height"):
            self.lines.append(
                f"  {var}.counterAxisSizingMode = 'FIXED';"
                if direction == "row"
                else f"  {var}.primaryAxisSizingMode = 'FIXED';"
            )
            self.lines.append(f"  {var}.resize({var}.width, {int(node['height'])});")
        self.created.append(var)
        return var

    def text(self, node: dict, parent: str) -> str:
        var = self.var()
        style_name = node.get("style", DEFAULT_TEXT_STYLE)
        if style_name not in TEXT_STYLES:
            style_name = DEFAULT_TEXT_STYLE
        weight, size = TEXT_STYLES[style_name]
        self.fonts.add(("Inter", weight))
        value = str(node.get("value", ""))[:400]

        self.lines.append(f"  const {var} = figma.createText();")
        self.lines.append(f"  {var}.fontName = {{ family: 'Inter', style: {_js(weight)} }};")
        self.lines.append(f"  {var}.characters = {_js(value)};")
        self.lines.append(f"  {var}.fontSize = {size};")
        # Hug by default. This is the fix for text that renders as a vertical
        # column of letters: a hugging text node sizes itself to its content
        # and cannot be squeezed narrower than one character.
        self.lines.append(f"  {var}.textAutoResize = 'WIDTH_AND_HEIGHT';")
        self.fill(var, node.get("color", "text"))
        self.lines.append(f"  {parent}.appendChild({var});")
        if node.get("wrap"):
            # Explicit wrapping: fill the parent's width, grow downwards.
            self.lines.append(f"  setFill({var});")
            self.lines.append(f"  {var}.textAutoResize = 'HEIGHT';")
        self.lines.append(f"  await applyTextStyle({var}, {_js(style_name)});")
        self.created.append(var)
        return var

    def button(self, node: dict, parent: str) -> str:
        primary = node.get("variant", "primary") == "primary"
        frame = {
            "name": f"Button / {node.get('label', '')}"[:60],
            "direction": "row", "gap": "sm", "padding": "md", "radius": "md",
            "background": "accent" if primary else "surface",
            "align": "CENTER", "fill": False,
        }
        var = self.frame(frame, parent)
        self.text(
            {"style": "Button", "value": node.get("label", "Button"),
             "color": "on-accent" if primary else "text"},
            var,
        )
        return var

    def badge(self, node: dict, parent: str) -> str:
        tone = node.get("tone", "neutral")
        frame = {
            "name": f"Badge / {node.get('label', '')}"[:60],
            "direction": "row", "padding": "sm", "radius": "sm",
            "background": f"{tone}-bg" if tone in TONES else "surface",
            "align": "CENTER", "fill": False,
        }
        var = self.frame(frame, parent)
        self.text(
            {"style": "Caption", "value": node.get("label", ""),
             "color": tone if tone in TONES else "text"},
            var,
        )
        return var

    def input(self, node: dict, parent: str) -> str:
        wrapper = self.frame(
            {"name": f"Field / {node.get('label', '')}"[:60], "direction": "col", "gap": "xs"},
            parent,
        )
        if node.get("label"):
            self.text({"style": "Caption", "value": node["label"], "color": "text-muted"}, wrapper)
        box = self.frame(
            {"name": "Input", "direction": "row", "padding": "md", "radius": "md",
             "background": "surface", "align": "CENTER"},
            wrapper,
        )
        self.text(
            {"style": "Body", "value": node.get("placeholder", ""), "color": "text-muted"}, box
        )
        return wrapper

    def avatar(self, node: dict, parent: str) -> str:
        var = self.var()
        size = int(node.get("size", 40))
        self.lines.append(f"  const {var} = figma.createEllipse();")
        self.lines.append(f"  {var}.name = {_js('Avatar')};")
        self.lines.append(f"  {var}.resize({size}, {size});")
        self.fill(var, node.get("color", "accent"))
        self.lines.append(f"  {parent}.appendChild({var});")
        self.created.append(var)
        return var

    def divider(self, node: dict, parent: str) -> str:
        var = self.var()
        self.lines.append(f"  const {var} = figma.createRectangle();")
        self.lines.append(f"  {var}.name = 'Divider';")
        self.lines.append(f"  {var}.resize(100, 1);")
        self.fill(var, node.get("color", "border"))
        self.lines.append(f"  {parent}.appendChild({var});")
        self.lines.append(f"  setFill({var});")
        self.created.append(var)
        return var

    def box(self, node: dict, parent: str) -> str:
        """A plain filled block -- chart areas, image placeholders, spacers."""
        var = self.frame(
            {"name": node.get("name", "Box"), "direction": "col",
             "background": node.get("background", "surface"),
             "radius": node.get("radius", "md")},
            parent,
        )
        height = int(node.get("height", 160))
        self.lines.append(f"  {var}.primaryAxisSizingMode = 'FIXED';")
        self.lines.append(f"  {var}.resize({var}.width, {height});")
        return var

    # -- dispatch ----------------------------------------------------------

    def node(self, spec: dict, parent: str) -> str:
        if not isinstance(spec, dict):
            raise SpecError(f"every node must be an object, got {type(spec).__name__}")
        kind = spec.get("kind")
        if kind in ("section", "col", "row", "card", "stack"):
            defaults = {"row": "row"}.get(kind, "col")
            node = dict(spec)
            if kind == "card":
                node.setdefault("background", "surface")
                node.setdefault("padding", "lg")
                node.setdefault("radius", "lg")
            if kind == "section":
                node.setdefault("padding", "xl")
                node.setdefault("gap", "lg")
            var = self.frame(node, parent, default_dir=defaults)
            for child in spec.get("children") or []:
                self.node(child, var)
            return var
        if kind == "text":
            return self.text(spec, parent)
        if kind == "button":
            return self.button(spec, parent)
        if kind == "badge":
            return self.badge(spec, parent)
        if kind == "input":
            return self.input(spec, parent)
        if kind == "avatar":
            return self.avatar(spec, parent)
        if kind == "divider":
            return self.divider(spec, parent)
        if kind == "box":
            return self.box(spec, parent)
        raise SpecError(
            f"unknown kind {kind!r}. Valid kinds: section, col, row, card, text, "
            "button, badge, input, avatar, divider, box"
        )


PREAMBLE = """\
const root0 = await figma.getNodeByIdAsync({parent_id});
if (!root0) {{ throw new Error('parent frame not found'); }}
// A repair REPLACES the section a previous attempt built. Without this, the
// only way to correct a section would be to append a second copy of it beside
// the broken one -- and `render_ui` cannot edit nodes in place.
for (const _oldId of {replace_ids}) {{
  const _old = await figma.getNodeByIdAsync(_oldId);
  if (_old && !_old.removed) {{ try {{ _old.remove(); }} catch (e) {{}} }}
}}
{fonts}
const _paint = await figma.getLocalPaintStylesAsync();
const _paintByName = {{}};
for (const s of _paint) {{ _paintByName[s.name] = s; }}
const _texts = await figma.getLocalTextStylesAsync();
const _textByName = {{}};
for (const t of _texts) {{ _textByName[t.name] = t; }}

async function applyFill(node, styleName) {{
  const style = _paintByName[styleName];
  if (style && 'setFillStyleIdAsync' in node) {{ await node.setFillStyleIdAsync(style.id); }}
}}
async function applyTextStyle(node, styleName) {{
  const style = _textByName[styleName];
  if (style) {{ await node.setTextStyleIdAsync(style.id); }}
}}
// FILL is only legal inside an auto-layout parent, and only after appending.
function setFill(node) {{
  const p = node.parent;
  if (p && 'layoutMode' in p && p.layoutMode !== 'NONE') {{
    node.layoutSizingHorizontal = 'FILL';
  }}
}}

const created = [];
{{
"""


def compile_spec(
    spec: dict,
    parent_id: str,
    color_roles: dict[str, str],
    replace_ids: list[str] | None = None,
) -> tuple[str, list[str]]:
    """Turn one UI tree into an atomic Figma script. Returns (js, node_var_names).

    `color_roles` maps a role the model can name ("text", "accent") to the real
    paint style created by bootstrap_tokens ("color/dark-gray"). An unknown
    role is skipped rather than guessed, so a bad role never invents a colour.

    `replace_ids` are nodes a previous attempt at this step created. They are
    removed before the new section is built, which is what makes a correcting
    retry a REPLACEMENT rather than a second copy appended beside the first.
    Scripts are atomic, so a failed repair leaves the original section intact.
    """
    compiler = _Compiler(parent_id, color_roles)
    compiler.node(spec, "root0")

    fonts = "\n".join(
        f"await figma.loadFontAsync({{ family: {_js(f)}, style: {_js(s)} }});"
        for f, s in sorted(compiler.fonts)
    )
    body = "\n".join(compiler.lines)
    pushes = "\n".join(f"  created.push({v}.id);" for v in compiler.created)
    return (
        PREAMBLE.format(
            parent_id=_js(parent_id), fonts=fonts, replace_ids=_js(list(replace_ids or []))
        )
        + body
        + "\n"
        + pushes
        + "\n}\nreturn { createdNodeIds: created };\n"
    ), compiler.created


def role_map(palette_info: list[tuple[str, str, str]]) -> dict[str, str]:
    """Build role -> paint-style-name from the harness's derived palette roles.

    The model names a ROLE; the harness resolves which token that is. Token
    names come from the brief and can be anything ("deep-navy", "cta"), so a
    model choosing them directly is guessing -- which is how invisible text on
    an accent background kept happening.
    """
    mapping: dict[str, str] = {}
    for token_name, _hex, role in palette_info:
        key = role.split(" ")[0].strip().lower()
        mapping.setdefault(key, token_name)
    # Sensible aliases so a reasonable spec never silently loses its colour.
    if "background" in mapping:
        mapping.setdefault("surface", mapping["background"])
        mapping.setdefault("on-accent", mapping["background"])
    if "text" in mapping:
        mapping.setdefault("text-muted", mapping["text"])
        mapping.setdefault("neutral", mapping["text"])
    if "surface" in mapping:
        mapping.setdefault("border", mapping["surface"])
        for tone in TONES:
            mapping.setdefault(f"{tone}-bg", mapping["surface"])
    for tone in TONES:
        mapping.setdefault(tone, mapping.get("accent", mapping.get("text", "")))
    return {k: v for k, v in mapping.items() if v}
