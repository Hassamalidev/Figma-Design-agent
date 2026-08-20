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
import logging

from agent import assets as assets_module
from agent import interactions, scaffold

logger = logging.getLogger(__name__)

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
    "illustration": "box", "graphic": "box",
    "rectangle": "box", "rect": "box", "shape": "box", "spacer": "box",
    "gradient": "box", "chart": "box", "placeholder": "box",
    "line": "divider", "rule": "divider", "separator": "divider",
    "radio": "checkbox", "toggle": "checkbox", "switch": "checkbox",
    "tag": "badge", "chip": "badge", "pill": "badge",
}

# Kinds that mean "the picture the user attached". Handled before ALIASES,
# because with no attachment they fall back to a `box` placeholder and with one
# they are the real thing -- the same word meaning both is the point.
IMAGE_KINDS = {"image", "img", "photo", "picture", "screenshot", "cover", "thumbnail"}

# Where a node says what clicking it does. Several spellings, because a refused
# synonym costs a whole turn and teaches the model nothing.
NAV_KEYS = ("on_click", "navigate", "goto", "go_to", "link_to", "onClick", "href")

MAX_NODES = 400  # a runaway spec must not lock up Figma

# An image with no stated height. Tall enough to read as a picture rather than
# a band, short enough not to push the rest of a section below the fold.
DEFAULT_IMAGE_HEIGHT = 240

# Below this ratio an aliased role is indistinguishable from what it sits on,
# so the alias is dropped rather than painting a divider white on white.
MIN_ALIAS_CONTRAST = 1.35


class SpecError(ValueError):
    """The spec is not renderable. The message goes back to the model verbatim."""


def _js(value) -> str:
    """JSON is valid JS for the literals we emit, and it escapes quotes for us."""
    return json.dumps(value)


class _Compiler:
    def __init__(
        self,
        parent_id: str,
        color_roles: dict[str, str],
        token_names=(),
        assets=(),
        screens: dict[str, str] | None = None,
        text_fonts: dict | None = None,
    ):
        self.parent_id = parent_id
        # role -> real paint style name, e.g. {"text": "color/dark-gray"}
        self.roles = color_roles
        # The images the user attached, already uploaded to this Figma file
        # (agent/assets.py). Empty on a run with no attachments, which is what
        # makes `kind: "image"` fall back to a placeholder box.
        self.assets = list(assets or [])
        # screen name -> frame id, so `"on_click": "Dashboard"` can become a
        # real prototype link while the section is being built.
        self.screens = dict(screens or {})
        # Ramp style name -> the (family, style) it really uses. Defaults to
        # the Inter table, so nothing changes for a design that named no font.
        self.text_fonts = dict(text_fonts or {})
        # (var, description, reaction) for every interaction this spec wires.
        self.reactions: list[tuple[str, str, dict]] = []
        # Every token that really exists, keyed loosely so "color/deep-navy",
        # "deep-navy" and "Deep Navy" all resolve to the same paint style.
        self.tokens = {_token_key(n): n for n in (token_names or ())}
        # Row frames with an explicit height: their children stretch to it.
        self.fixed_rows: set[str] = set()
        # Nodes that must FILL their parent, applied in ONE pass at the very
        # end (see `sizing_pass`). Collected in creation order, which is
        # outermost first -- a child can only fill a parent that already has a
        # resolved width.
        self.fills: list[str] = []
        self.vfills: list[str] = []
        self.lines: list[str] = []
        self.fonts: set[tuple[str, str]] = set()
        self.created: list[str] = []
        self.count = 0

    def sizing_pass(self) -> list[str]:
        """Every FILL, emitted once, after the whole subtree exists.

        Width is the one property in this compiler that cannot be decided
        locally: `resize()` resets sizing modes, and a frame's own resolved
        width depends on children that do not exist yet when it is created. So
        it is decided here, last, in outermost-first order -- a child can only
        fill a parent that already has a width.
        """
        if not self.fills and not self.vfills:
            return []
        lines = [
            "  // Width LAST. resize() resets sizing modes, and a frame has no",
            "  // resolved width until its children exist, so every FILL is",
            "  // applied here rather than while the tree was being built.",
        ]
        lines += [f"  setFill({var});" for var in self.fills]
        lines += [f"  setFillV({var});" for var in self.vfills]
        return lines

    def wiring_pass(self) -> list[str]:
        """Every prototype interaction, applied once the whole tree exists.

        Each one is wrapped on its own: a node type that cannot hold a reaction
        must cost one interaction, not the section it belongs to. `reactions`
        is read-only under `documentAccess: "dynamic-page"`, so
        `setReactionsAsync` is the only setter there is.
        """
        lines: list[str] = []
        for var, description, payload in self.reactions:
            lines.append("  try {")
            lines.append(f"    await {var}.setReactionsAsync({_js([payload])});")
            lines.append(f"    wired.push({_js(description)});")
            lines.append("  } catch (e) { wireFailed.push(" + _js(description) + "); }")
        return lines

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

    def paint_image(self, var: str, node: dict) -> bool:
        """Paint one of the user's attached images onto this node.

        Returns False when there is nothing to paint, so the caller can fall
        back to the placeholder it would have drawn anyway. An asset name that
        matches NOTHING is an error rather than a silent grey box: the user
        attached a picture and would otherwise never learn it went unused.
        """
        wanted = node.get("asset") or node.get("image") or node.get("src")
        if not wanted and str(node.get("kind") or "").lower() not in IMAGE_KINDS:
            return False
        if not self.assets:
            return False
        asset = assets_module.find(self.assets, wanted if isinstance(wanted, str) else None)
        if asset is None:
            raise SpecError(
                f"no attached image called {str(wanted)!r}. The images available are: "
                f"{assets_module.names(self.assets)}"
            )
        mode = "FIT" if str(node.get("fit", "")).lower() in ("fit", "contain") else "FILL"
        self.lines.append(
            f"  applyImage({var}, {_js(asset.image_hash)}, {_js(mode)});"
        )
        return True

    @staticmethod
    def _default_justify(node: dict, layout: str) -> str | None:
        """Spread a full-width row's children instead of packing them left.

        A header is a row of things that each hug -- a logo, some nav links,
        some icons. The row fills the page, the children do not, and the
        default `MIN` packs all of them against the left with the rest of the
        1440px blank. That is half of the empty space a real design showed.

        Only for a row that actually FILLS: on a row that hugs its content
        there is no free space, so this is a no-op, and on a grid whose cards
        all fill there is none either. It changes exactly the case it is for.
        """
        if layout != "HORIZONTAL":
            return None
        if node.get("width") or not node.get("fill", True):
            return None  # a hugging or fixed-width row has nothing to spread
        if len(node.get("children") or []) < 2:
            return None
        return "SPACE_BETWEEN"

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
        justify = node.get("justify") or self._default_justify(node, layout)
        if justify:
            self.lines.append(f"  {var}.primaryAxisAlignItems = {_js(justify)};")
        self.fill(var, node.get("background"))
        # After the role fill, so a frame given both shows the picture. A
        # section with a photo behind it is the commonest thing an attachment
        # is for, and it must not be undone by the background token.
        self.paint_image(var, node)
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
            # DEFERRED, not emitted here. This is the bug that left a book grid
            # occupying a third of the page with the rest blank:
            #
            #   n4.appendChild(n5);
            #   setFill(n5);                  <- width = FILL
            #   n5.resize(n5.width, 180);     <- resets BOTH axes to FIXED
            #
            # `resize()` resets sizing modes (knowledge/gotchas.md), so a fill
            # applied here is destroyed by the node's own height resize two
            # lines later -- freezing it at whatever width it happened to have
            # while the frame was still empty. Worse, every fill was applied
            # BEFORE the frame had any children at all, so the width it
            # resolved against was mid-build in every case.
            #
            # Deciding width last removes the ordering question entirely.
            self.fills.append(var)
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
            self.vfills.append(var)
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
        # The BRAND font for headings when the design asked for one and this
        # file has it -- resolved by the harness, never guessed here.
        family, weight = self.text_fonts.get(style_name, ("Inter", weight))
        self.fonts.add((family, weight))
        value = str(node.get("value", ""))[:400]

        self.lines.append(f"  const {var} = figma.createText();")
        self.lines.append(
            f"  {var}.fontName = {{ family: {_js(family)}, style: {_js(weight)} }};"
        )
        self.lines.append(f"  {var}.characters = {_js(value)};")
        self.lines.append(f"  {var}.fontSize = {size};")
        # Hug by default. This is the fix for text that renders as a vertical
        # column of letters: a hugging text node sizes itself to its content
        # and cannot be squeezed narrower than one character.
        self.lines.append(f"  {var}.textAutoResize = 'WIDTH_AND_HEIGHT';")
        self.fill(var, node.get("color", "text"))
        self.lines.append(f"  {parent}.appendChild({var});")
        if node.get("wrap"):
            # Explicit wrapping: fill the parent's width, grow downwards. The
            # fill is deferred with all the others so a later resize anywhere
            # in the subtree cannot undo it.
            self.lines.append(f"  {var}.textAutoResize = 'HEIGHT';")
            self.fills.append(var)
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
        # A real profile picture or logo, cropped to the circle, when one was
        # attached -- otherwise the flat accent disc.
        if not self.paint_image(var, node):
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

    def image(self, node: dict, parent: str) -> str:
        """One of the user's attached pictures, at a real size on the canvas.

        With no attachment this IS `box` -- the same grey placeholder the
        renderer has always drawn for `kind: "image"`. The word means "a
        picture goes here" either way; only one of them has a picture.
        """
        if not self.assets:
            return self.box(node, parent)
        frame = dict(node)
        frame.setdefault("name", str(node.get("asset") or "Image"))
        frame["radius"] = node.get("radius", "md")
        # No background token: the image IS the fill, and a role underneath it
        # would only show through if the picture failed to paint. Height is
        # applied below instead, so it is stated exactly once.
        frame.pop("background", None)
        frame.pop("height", None)
        var = self.frame(frame, parent)
        self.lines.append(f"  {var}.primaryAxisSizingMode = 'FIXED';")
        self.lines.append(f"  {var}.resize({var}.width, {self._image_height(node)});")
        return var

    def _image_height(self, node: dict) -> int:
        """How tall the picture is. Width is decided last (`sizing_pass`), so an
        explicit height is the only thing that can be honoured exactly; from a
        stated width the image's own aspect ratio gives the rest."""
        if node.get("height"):
            return max(8, int(node["height"]))
        asset = assets_module.find(self.assets, node.get("asset") or node.get("image"))
        if node.get("width") and asset is not None:
            return max(40, round(int(node["width"]) / asset.aspect))
        return DEFAULT_IMAGE_HEIGHT

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
        """Build one node, then wire whatever clicking it should do."""
        var = self._build(spec, parent)
        self.wire(spec, var)
        return var

    def wire(self, spec: dict, var: str) -> None:
        """Turn `"on_click": "Dashboard"` into a real prototype interaction.

        The model is the only thing that knows the primary action of the
        section it just designed, so it says so while building. Everything
        mechanical about it -- which frame that name means, which transition,
        what a bare "back" does -- is decided here (CLAUDE.md rule 7).
        """
        target = next(
            (str(spec[key]) for key in NAV_KEYS if str(spec.get(key) or "").strip()), ""
        ).strip()
        if not target:
            return
        label = str(spec.get("label") or spec.get("value") or spec.get("name") or "")[:60]
        trigger = interactions.normalize_trigger(spec.get("trigger"))
        transition = interactions.normalize_transition(spec.get("transition"))
        lowered = target.lower()
        if lowered in ("back", "go back", "previous"):
            link = interactions.Link(
                source_id="", label=label, action="back", trigger=trigger
            )
        elif lowered.startswith("http://") or lowered.startswith("https://"):
            link = interactions.Link(
                source_id="", label=label, action="url", url=target, trigger=trigger
            )
        else:
            destination = interactions.resolve_screen(target, self.screens)
            if not destination or destination == self.parent_id:
                # An unknown screen name, or a link from a screen to itself.
                # Never fatal: a missing interaction is worth far less than the
                # section it would take down, and the end-of-run wiring pass
                # reads the real canvas and gets another go at it.
                logger.info("Skipped an interaction to %r -- no such screen.", target)
                return
            link = interactions.Link(
                source_id="",
                label=label,
                destination_id=destination,
                destination_name=target,
                trigger=trigger,
                transition=transition,
            )
        self.reactions.append((var, link.describe(), interactions.reaction(link)))

    def _build(self, spec: dict, parent: str) -> str:
        if not isinstance(spec, dict):
            raise SpecError(f"every node must be an object, got {type(spec).__name__}")
        raw_kind = str(spec.get("kind") or "").strip().lower()
        if raw_kind in IMAGE_KINDS:
            return self.image(spec, parent)
        kind = ALIASES.get(raw_kind, spec.get("kind"))
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
            "button, badge, input, checkbox, avatar, divider, box, image"
        )


PREAMBLE = """\
const root0 = await figma.getNodeByIdAsync({parent_id});
if (!root0) {{ throw new Error('parent frame not found'); }}
// A repair REPLACES the section a previous attempt built. Without this, the
// only way to correct a section would be to append a second copy of it beside
// the broken one -- and `render_ui` cannot edit nodes in place.
for (const _oldId of {replace_ids}) {{
  const _old = await figma.getNodeByIdAsync(_oldId);
  if (!_old || _old.removed) {{ continue; }}
  // Never a top-level frame: that is a whole SCREEN, and replacing one empties
  // the page. Callers already check this, and checking again here means no
  // future caller can get it wrong -- an edit run emptied a real user's file
  // through exactly this line.
  if (_old.parent && _old.parent.type === 'PAGE') {{ continue; }}
  try {{ _old.remove(); }} catch (e) {{}}
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
// One of the user's attached pictures. The image itself is already stored in
// this file (agent/assets.py uploaded it once); a paint only references it by
// hash, which is why the same photo costs nothing to reuse on five screens.
function applyImage(node, hash, mode) {{
  if (!('fills' in node)) {{ return; }}
  node.fills = [{{ type: 'IMAGE', scaleMode: mode, imageHash: hash }}];
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
const wired = [];
const wireFailed = [];
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
    assets: list | None = None,
    screens: dict[str, str] | None = None,
    text_fonts: dict | None = None,
) -> tuple[str, list[str]]:
    """Turn one UI tree into an atomic Figma script. Returns (js, node_var_names).

    `color_roles` maps a role the model can name ("text", "accent") to the real
    paint style created by bootstrap_tokens ("color/dark-gray"). An unknown
    role is skipped rather than guessed, so a bad role never invents a colour.

    `replace_ids` are nodes a previous attempt at this step created. They are
    removed before the new section is built, which is what makes a correcting
    retry a REPLACEMENT rather than a second copy appended beside the first.
    Scripts are atomic, so a failed repair leaves the original section intact.

    `assets` are the user's attached images, already uploaded to this file, and
    `screens` maps a screen NAME to its frame id -- which is what lets a spec
    say `"on_click": "Dashboard"` and get a working prototype link out of it.
    """
    compiler = _Compiler(
        parent_id, color_roles, token_names or [], assets, screens, text_fonts
    )
    compiler.node(spec, "root0")

    fonts = "\n".join(
        f"await figma.loadFontAsync({{ family: {_js(f)}, style: {_js(s)} }});"
        for f, s in sorted(compiler.fonts)
    )
    body = "\n".join(compiler.lines + compiler.sizing_pass() + compiler.wiring_pass())
    pushes = "\n".join(f"  created.push({v}.id);" for v in compiler.created)
    return (
        PREAMBLE.format(
            parent_id=_js(parent_id), fonts=fonts, replace_ids=_js(list(replace_ids or []))
        )
        + body
        + "\n"
        + pushes
        + "\n}\nreturn { createdNodeIds: created, wired: wired, wireFailed: wireFailed };\n"
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
