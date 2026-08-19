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

from agent import scaffold

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

# Synonyms for the kinds that really exist. A live run lost turns to
# `unknown kind 'ellipse'`, `'frame'`, `'checkbox'` and `'image'` -- all
# reasonable names for things the renderer can already draw. Rejecting a
# synonym teaches the model nothing and costs a whole turn, so they are mapped.
ALIASES = {
    "frame": "col", "group": "col", "container": "col", "div": "col",
    "stack": "col", "vstack": "col", "list": "col", "form": "col",
    "hstack": "row", "flex": "row", "inline": "row",
    "panel": "card", "surface": "card",
    "heading": "text", "title": "text", "label": "text", "paragraph": "text",
    "link": "text", "caption": "text", "span": "text",
    "cta": "button", "submit": "button",
    "textfield": "input", "textbox": "input", "field": "input",
    "ellipse": "avatar", "circle": "avatar", "dot": "avatar",
    "icon": "avatar", "logo": "avatar",
    "image": "box", "img": "box", "illustration": "box", "graphic": "box",
    "rectangle": "box", "rect": "box", "shape": "box", "spacer": "box",
    "gradient": "box", "chart": "box", "placeholder": "box",
    "line": "divider", "rule": "divider", "separator": "divider",
    "radio": "checkbox", "toggle": "checkbox", "switch": "checkbox",
    "tag": "badge", "chip": "badge", "pill": "badge",
}

MAX_NODES = 400  # a runaway spec must not lock up Figma

# Below this ratio an aliased role is indistinguishable from what it sits on,
# so the alias is dropped rather than painting a divider white on white.
MIN_ALIAS_CONTRAST = 1.35


class SpecError(ValueError):
    """The spec is not renderable. The message goes back to the model verbatim."""


def _js(value) -> str:
    """JSON is valid JS for the literals we emit, and it escapes quotes for us."""
    return json.dumps(value)


class _Compiler:
    def __init__(self, parent_id: str, color_roles: dict[str, str], token_names=()):
        self.parent_id = parent_id
        # role -> real paint style name, e.g. {"text": "color/dark-gray"}
        self.roles = color_roles
        # Every token that really exists, keyed loosely so "color/deep-navy",
        # "deep-navy" and "Deep Navy" all resolve to the same paint style.
        self.tokens = {_token_key(n): n for n in (token_names or ())}
        # Row frames with an explicit height: their children stretch to it.
        self.fixed_rows: set[str] = set()
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
        touching a hex value.

        A ROLE is the normal way in. A real TOKEN name is also accepted, because
        roles cannot express everything a palette contains: a dark-to-light
        split screen needs the near-black token by name, and asking for the
        `background` role -- which is by definition the lightest colour -- gave
        it white and rendered the whole panel as a blank void.
        """
        if not role:
            return
        style = self.roles.get(role) or self.tokens.get(_token_key(role))
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
        # SIZE FIRST, THEN SIZING MODES. `resize()` resets HUG/FILL back to
        # FIXED (knowledge/gotchas.md), so a FILL applied before a resize is
        # silently thrown away -- which is how a full-width row ended up frozen
        # at whatever width it happened to have when it was measured.
        # A FIXED width is what a sidebar is: 240px beside a filling content
        # column. It must not also stretch, so it opts out of FILL.
        if node.get("width"):
            width = int(node["width"])
            axis = "primaryAxisSizingMode" if direction == "row" else "counterAxisSizingMode"
            self.lines.append(f"  {var}.{axis} = 'FIXED';")
            self.lines.append(f"  {var}.resize({width}, {var}.height);")
        if node.get("height"):
            self.lines.append(
                f"  {var}.counterAxisSizingMode = 'FIXED';"
                if direction == "row"
                else f"  {var}.primaryAxisSizingMode = 'FIXED';"
            )
            self.lines.append(f"  {var}.resize({var}.width, {int(node['height'])});")
        if not node.get("width") and node.get("fill", True):
            # `setFill` re-checks at runtime that the parent really is
            # auto-layout, so this can never throw.
            self.lines.append(f"  setFill({var});")
        # A column beside another column has to be as tall as the row holding
        # them, or a 55/45 split screen renders as two short bands with the
        # page showing through underneath. Automatic for a container inside a
        # fixed-height ROW -- which is exactly the split-screen case and never
        # a row of buttons.
        stretches = node.get("stretch") or (
            # `fill: False` marks something that hugs its own content -- a
            # button or a badge. One of those dropped straight into a
            # fixed-height row must not become 900px tall.
            parent in self.fixed_rows and node.get("fill", True)
        )
        if stretches:
            self.lines.append(f"  setFillV({var});")
        if direction == "row" and node.get("height"):
            self.fixed_rows.add(var)
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
        # An input with no placeholder rendered an EMPTY text node, so the box
        # came back as "408x56 frame is empty, leaving a blank region" -- a
        # defect the harness caused and then reported. A field always shows
        # something; fall back to the label, then to a neutral hint.
        placeholder = str(node.get("placeholder") or "").strip()
        if not placeholder:
            label = str(node.get("label") or "").strip()
            placeholder = f"Enter {label.lower()}" if label else "Enter a value"
        self.text({"style": "Body", "value": placeholder, "color": "text-muted"}, box)
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

    def checkbox(self, node: dict, parent: str) -> str:
        """A 20px box beside its label. Every auth screen wants one ("Remember
        me", "I agree to the Terms"), and rejecting the kind cost a step three
        turns each time it asked."""
        row = self.frame(
            {"name": f"Checkbox / {node.get('label', '')}"[:60], "direction": "row",
             "gap": "sm", "align": "CENTER", "fill": False},
            parent,
        )
        mark = self.var()
        checked = bool(node.get("checked"))
        self.lines.append(f"  const {mark} = figma.createFrame();")
        self.lines.append(f"  {mark}.name = 'Box';")
        self.lines.append(f"  {mark}.resize(20, 20);")
        self.lines.append(f"  {mark}.cornerRadius = {RADIUS['sm'] // 2};")
        self.fill(mark, node.get("color") or ("accent" if checked else "surface"))
        self.lines.append(f"  {row}.appendChild({mark});")
        self.created.append(mark)
        if node.get("label"):
            self.text({"style": "Caption", "value": node["label"], "color": "text"}, row)
        return row

    # -- dispatch ----------------------------------------------------------

    def node(self, spec: dict, parent: str) -> str:
        if not isinstance(spec, dict):
            raise SpecError(f"every node must be an object, got {type(spec).__name__}")
        kind = ALIASES.get(str(spec.get("kind") or "").strip().lower(), spec.get("kind"))
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
            for child in _without_duplicate_labels(spec.get("children") or []):
                self.node(child, var)
            return var
        if kind == "text":
            if not str(spec.get("value") or "").strip():
                raise SpecError(
                    "a text node needs a non-empty `value` -- an empty one renders "
                    "as a blank region and is reported as a defect"
                )
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
        if kind == "checkbox":
            return self.checkbox(spec, parent)
        raise SpecError(
            f"unknown kind {kind!r}. Valid kinds: section, col, row, card, text, "
            "button, badge, input, checkbox, avatar, divider, box"
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
// The cross-axis twin: what makes both halves of a split screen full height.
function setFillV(node) {{
  const p = node.parent;
  if (p && 'layoutMode' in p && p.layoutMode !== 'NONE') {{
    try {{ node.layoutSizingVertical = 'FILL'; }} catch (e) {{}}
  }}
}}

const created = [];
{{
"""


def _without_duplicate_labels(children: list) -> list:
    """Drop a text node that only repeats the label of the input right after it.

    `input` renders its own label, so a spec that writes the label out AND
    labels the field produces "Email" twice, stacked. The vision critic caught
    it correctly ("the label is duplicated immediately above the input field")
    -- but a judgement call cannot fail a step, so the run just spent its whole
    repair budget on it. The renderer owns field layout, so the renderer
    resolves the collision (CLAUDE.md section 6a).
    """
    kept: list = []
    for index, child in enumerate(children):
        following = children[index + 1] if index + 1 < len(children) else None
        if (
            isinstance(child, dict)
            and isinstance(following, dict)
            and child.get("kind") == "text"
            and following.get("kind") == "input"
            and _same_label(child.get("value"), following.get("label"))
        ):
            continue
        kept.append(child)
    return kept


def _same_label(a, b) -> bool:
    return bool(a) and bool(b) and str(a).strip().strip(":*").lower() == str(b).strip().lower()


def _token_key(name: str) -> str:
    """Loose key for a token name, so near-misses still find the real style."""
    return str(name).strip().lower().replace("color/", "").replace(" ", "-").replace("_", "-")


def compile_spec(
    spec: dict,
    parent_id: str,
    color_roles: dict[str, str],
    replace_ids: list[str] | None = None,
    token_names: list[str] | None = None,
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
    compiler = _Compiler(parent_id, color_roles, token_names or [])
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

    **A role is never aliased onto a colour it cannot be seen against.** The old
    chain was `border -> surface -> background`, so a palette with no border
    colour painted every divider in the page fill: white on white, invisible,
    and the input boxes lost their edges with it. An alias that collapses to the
    background is worse than no fill at all -- an unfilled divider keeps Figma's
    default grey, which is at least visible -- so those aliases are now
    contrast-checked and dropped when they would disappear.
    """
    mapping: dict[str, str] = {}
    hex_of: dict[str, str] = {}
    for token_name, hex_value, role in palette_info:
        key = role.split(" ")[0].strip().lower()
        hex_of.setdefault(token_name, hex_value)
        if key == "decorative":
            # A colour with no role is still a real token the spec can name
            # directly -- it just is not something `background` may resolve to.
            continue
        mapping.setdefault(key, token_name)

    def visible_against(candidate: str, ground_role: str | None) -> bool:
        """Would this alias still be distinguishable from what sits behind it?

        `ground_role` is a ROLE, so it is resolved through the mapping first --
        comparing it as a token name silently matched nothing and let every
        alias through, which is the bug this guard exists to stop.
        """
        if not ground_role:
            return True
        a, b = hex_of.get(candidate), hex_of.get(mapping.get(ground_role, ""))
        if not a or not b:
            return True
        return scaffold.contrast_ratio(a, b) >= MIN_ALIAS_CONTRAST

    def alias(role: str, source: str, *, against: str | None = None) -> None:
        target = mapping.get(source)
        if role in mapping or not target:
            return
        if visible_against(target, against):
            mapping[role] = target

    # A card on the page fill reads as the page; that is a flat design, not a
    # broken one, so this alias stands unconditionally.
    alias("surface", "background")
    alias("on-accent", "background")
    alias("text-muted", "text")
    alias("neutral", "text")
    # An inverse panel with no second background is simply unavailable, and
    # copy for it falls back to whatever is readable there rather than to the
    # ordinary text colour, which would vanish on a dark panel.
    alias("text-on-alt", "background")
    # These MUST stay distinguishable from what they sit on.
    alias("border", "surface", against="background")
    alias("border", "text-muted", against="background")
    for tone in TONES:
        alias(f"{tone}-bg", "surface")
    for tone in TONES:
        alias(tone, "accent")
    for tone in TONES:
        alias(tone, "text")
    return {k: v for k, v in mapping.items() if v}
