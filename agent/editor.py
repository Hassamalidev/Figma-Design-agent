"""Compile a list of declarative EDITS into correct Figma Plugin API JavaScript.

The mirror of `agent/renderer.py`, and it exists for the same reason. Create
mode proved that asking a small model to write Plugin API code produces API
mistakes, not design mistakes -- `FILL can only be set on children of
auto-layout frames`, `fills` mutated in place, font styles guessed. Editing has
its own version of every one of those, plus a worse failure mode: a bad create
leaves an ugly section next to the good ones, while a bad edit damages work the
user already has.

So the model says WHAT to change:

    [{"op": "set_fill",  "target": "1:9",  "color": "accent"},
     {"op": "set_text",  "target": "1:10", "value": "Sign in"},
     {"op": "insert",    "parent": "1:2",  "spec": {...}}]

...and this module decides HOW: loading the font a text node already uses,
cloning paints instead of mutating them, refusing x/y on an auto-layout child,
and never touching a node that was not named.

Three safety properties, all deliberate:

1. **Every target is verified against the inventory before compiling.** An id
   the model invented never reaches Figma.
2. **Each edit is wrapped in its own try/catch.** Scripts are atomic, which is
   right for building one section and wrong for a batch of independent edits:
   one bad target would otherwise discard nine good changes. The script reports
   `applied` and `failed` separately so the loop can tell the model precisely
   which one did not take.
3. **Nothing is destructive by default.** `delete` is the only op that removes
   anything, it is never implied by another op, and it refuses to delete a
   whole screen frame.
"""
from __future__ import annotations

import json

from agent import renderer
from agent.renderer import RADIUS, SPACING, TEXT_STYLES, SpecError

# Ops that change something. Kept small on purpose: a vocabulary the model
# cannot overreach is worth more than one that covers every Figma property.
OPS = (
    "set_fill",
    "set_text",
    "set_text_style",
    "set_size",
    "set_spacing",
    "set_radius",
    "set_visible",
    "set_name",
    "reorder",
    "delete",
    "insert",
    "replace",
)

# A batch bigger than this is a rebuild, not an edit.
MAX_EDITS = 40

# The destructive budget. An edit run wiped a user's page by expanding one
# `replace` across every frame on it, so removal is now capped twice: by how
# many nodes a whole batch may remove, and -- much more tightly -- by how many
# a SELECTOR may remove. An over-broad selector is precisely how "swap the old
# label" becomes "swap everything", and unlike a list of ids the model never
# sees how far it reaches.
MAX_REMOVALS_PER_BATCH = 6
MAX_SELECTOR_REMOVALS = 2

# Ops that take something off the canvas. The guard hangs off this set rather
# than off the op name, so a new destructive op cannot be added without one.
DESTRUCTIVE_OPS = ("delete", "replace")


def _js(value) -> str:
    return json.dumps(value)


class _EditCompiler:
    def __init__(self, roles: dict[str, str], tokens, protected: set[str]):
        self.roles = roles
        self.tokens = {renderer._token_key(t): t for t in (tokens or ())}
        # Screen frames. Deleting one throws a whole screen away over a
        # phrase like "remove the old login" -- far too much damage for a
        # single mis-parsed word.
        self.protected = protected
        self.lines: list[str] = []
        # How many nodes this batch removes, across every op that removes one.
        self.removals = 0
        self.fonts: set[tuple[str, str]] = set()
        self.touched: list[str] = []

    # -- helpers -----------------------------------------------------------

    def style_for(self, colour: str | None) -> str | None:
        """A role name, or a real token name. Same resolution as the renderer,
        so a colour that works when building works when editing."""
        if not colour:
            return None
        return self.roles.get(colour) or self.tokens.get(renderer._token_key(colour))

    def begin(self, node_id: str, label: str) -> str:
        var = f"n{len(self.touched)}"
        self.touched.append(node_id)
        self.lines.append(f"try {{")
        self.lines.append(f"  const {var} = await figma.getNodeByIdAsync({_js(node_id)});")
        self.lines.append(f"  if (!{var} || {var}.removed) {{ throw new Error('node is gone'); }}")
        return var

    def end(self, node_id: str, label: str) -> None:
        self.lines.append(f"  applied.push({_js(f'{label} on {node_id}')});")
        self.lines.append("} catch (e) {")
        self.lines.append(
            f"  failed.push({_js(f'{label} on {node_id}')} + ': ' "
            "+ String(e && e.message ? e.message : e));"
        )
        self.lines.append("}")

    # -- the operations ----------------------------------------------------

    def set_fill(self, node_id: str, edit: dict) -> None:
        style = self.style_for(edit.get("color") or edit.get("colour"))
        if not style:
            raise SpecError(
                f"set_fill needs a `color` that is a role or a real token; "
                f"got {edit.get('color')!r}"
            )
        var = self.begin(node_id, "set_fill")
        self.lines.append(f"  const style = _paintByName[{_js(style)}];")
        self.lines.append("  if (!style) { throw new Error('no paint style ' + " + _js(style) + "); }")
        self.lines.append(f"  if (!('setFillStyleIdAsync' in {var})) {{ throw new Error('cannot be filled'); }}")
        self.lines.append(f"  await {var}.setFillStyleIdAsync(style.id);")
        self.end(node_id, "set_fill")

    def set_text(self, node_id: str, edit: dict) -> None:
        value = str(edit.get("value") if edit.get("value") is not None else "").strip()
        if not value:
            raise SpecError("set_text needs a non-empty `value`; use `delete` to remove a node")
        var = self.begin(node_id, "set_text")
        self.lines.append(f"  if ({var}.type !== 'TEXT') {{ throw new Error('not a text node'); }}")
        # Load the font the node ALREADY uses. Guessing a style string is what
        # killed a live run ("Inter SemiBold" -- the real style has a space),
        # and here we never have to guess: the node knows.
        self.lines.append(f"  if ({var}.fontName === figma.mixed) {{")
        self.lines.append("    await figma.loadFontAsync({ family: 'Inter', style: 'Regular' });")
        self.lines.append(f"    {var}.fontName = {{ family: 'Inter', style: 'Regular' }};")
        self.lines.append("  } else {")
        self.lines.append(f"    await figma.loadFontAsync({var}.fontName);")
        self.lines.append("  }")
        self.lines.append(f"  {var}.characters = {_js(value[:400])};")
        self.end(node_id, "set_text")

    def set_text_style(self, node_id: str, edit: dict) -> None:
        name = str(edit.get("style") or "")
        if name not in TEXT_STYLES:
            raise SpecError(
                f"unknown text style {name!r}. Available: {', '.join(TEXT_STYLES)}"
            )
        self.fonts.add(("Inter", TEXT_STYLES[name][0]))
        var = self.begin(node_id, "set_text_style")
        self.lines.append(f"  if ({var}.type !== 'TEXT') {{ throw new Error('not a text node'); }}")
        self.lines.append(f"  const ts = _textByName[{_js(name)}];")
        self.lines.append("  if (!ts) { throw new Error('no text style ' + " + _js(name) + "); }")
        self.lines.append(f"  await {var}.setTextStyleIdAsync(ts.id);")
        self.end(node_id, "set_text_style")

    def set_size(self, node_id: str, edit: dict) -> None:
        width, height = edit.get("width"), edit.get("height")
        if width is None and height is None:
            raise SpecError("set_size needs `width`, `height`, or both")
        var = self.begin(node_id, "set_size")
        self.lines.append(f"  if (!('resize' in {var})) {{ throw new Error('cannot be resized'); }}")
        w = int(width) if width is not None else f"{var}.width"
        h = int(height) if height is not None else f"{var}.height"
        # A child of an auto-layout parent is sized by the layout, so a resize
        # is silently undone. Switch it to a fixed size on that axis first.
        self.lines.append(f"  const p = {var}.parent;")
        self.lines.append("  if (p && 'layoutMode' in p && p.layoutMode !== 'NONE') {")
        if width is not None:
            self.lines.append(f"    try {{ {var}.layoutSizingHorizontal = 'FIXED'; }} catch (e) {{}}")
        if height is not None:
            self.lines.append(f"    try {{ {var}.layoutSizingVertical = 'FIXED'; }} catch (e) {{}}")
        self.lines.append("  }")
        self.lines.append(f"  {var}.resize({w}, {h});")
        self.end(node_id, "set_size")

    def set_spacing(self, node_id: str, edit: dict) -> None:
        gap = edit.get("gap")
        padding = edit.get("padding")
        if gap is None and padding is None:
            raise SpecError("set_spacing needs `gap`, `padding`, or both")
        var = self.begin(node_id, "set_spacing")
        self.lines.append(
            f"  if (!('layoutMode' in {var}) || {var}.layoutMode === 'NONE') "
            "{ throw new Error('not an auto-layout frame'); }"
        )
        if gap is not None:
            self.lines.append(f"  {var}.itemSpacing = {_spacing(gap, 'gap')};")
        if padding is not None:
            pad = _spacing(padding, "padding")
            self.lines.append(
                f"  {var}.paddingTop = {pad}; {var}.paddingBottom = {pad};"
                f" {var}.paddingLeft = {pad}; {var}.paddingRight = {pad};"
            )
        self.end(node_id, "set_spacing")

    def set_radius(self, node_id: str, edit: dict) -> None:
        name = str(edit.get("radius", "md"))
        if name not in RADIUS:
            raise SpecError(f"radius must be one of {', '.join(RADIUS)}; got {name!r}")
        var = self.begin(node_id, "set_radius")
        self.lines.append(f"  if (!('cornerRadius' in {var})) {{ throw new Error('has no corners'); }}")
        self.lines.append(f"  {var}.cornerRadius = {RADIUS[name]};")
        self.end(node_id, "set_radius")

    def set_visible(self, node_id: str, edit: dict) -> None:
        visible = bool(edit.get("visible", True))
        var = self.begin(node_id, "set_visible")
        self.lines.append(f"  {var}.visible = {_js(visible)};")
        self.end(node_id, "set_visible")

    def set_name(self, node_id: str, edit: dict) -> None:
        name = str(edit.get("name") or "").strip()
        if not name:
            raise SpecError("set_name needs a non-empty `name`")
        var = self.begin(node_id, "set_name")
        self.lines.append(f"  {var}.name = {_js(name[:100])};")
        self.end(node_id, "set_name")

    def reorder(self, node_id: str, edit: dict) -> None:
        index = edit.get("index")
        if index is None:
            raise SpecError("reorder needs an `index` (0 is first)")
        var = self.begin(node_id, "reorder")
        self.lines.append(f"  const parent = {var}.parent;")
        self.lines.append("  if (!parent) { throw new Error('node has no parent'); }")
        self.lines.append(
            f"  parent.insertChild(Math.max(0, Math.min({int(index)}, parent.children.length - 1)), {var});"
        )
        self.end(node_id, "reorder")

    # -- the destructive ops -----------------------------------------------
    #
    # `delete` was guarded and `replace` was not, and `replace` removes its
    # target just as surely -- so `{"op":"replace","target":"<a screen frame>"}`
    # threw a whole screen away, and the same op with a `{"type":"FRAME"}`
    # selector fanned out and emptied the page. Both now go through one guard,
    # because the property that matters is "this removes something", not which
    # word was used for it.

    def guard_removal(self, node_id: str, op: str) -> None:
        """Refuse a removal that is too big to be an edit."""
        if not self.protected:
            # A plumbing mistake (an inventory that failed to read, a caller
            # that forgot to pass the screens) must not silently unlock
            # deleting screens. Fail closed.
            raise SpecError(
                f"{op} is not available: the harness could not work out which frames are "
                "whole screens, and will not risk removing one."
            )
        if node_id in self.protected:
            fix = (
                "Rebuild a whole screen with Create mode"
                if op == "replace"
                else "Remove the specific section inside it instead"
            )
            raise SpecError(
                f"{node_id} is a whole screen frame, and {op} removes what it targets -- "
                f"this would empty the page. {fix}, or target a section inside it."
            )
        self.removals += 1
        if self.removals > MAX_REMOVALS_PER_BATCH:
            raise SpecError(
                f"this batch would remove {self.removals} nodes, more than the "
                f"{MAX_REMOVALS_PER_BATCH} an edit is allowed. Removing more than that is a "
                "redesign, not an edit -- do it in smaller, explicit steps."
            )

    def delete(self, node_id: str, edit: dict) -> None:
        self.guard_removal(node_id, "delete")
        var = self.begin(node_id, "delete")
        self.lines.append(f"  {var}.remove();")
        self.end(node_id, "delete")

    # -- the structural ops, which reuse the renderer ----------------------

    def insert(self, parent_id: str, edit: dict) -> None:
        spec = edit.get("spec")
        if not isinstance(spec, dict):
            raise SpecError("insert needs a `spec` (a UI tree, same shape as render_ui)")
        self._render_into(parent_id, spec, replace_ids=[], label="insert", index=edit.get("index"))

    def replace(self, node_id: str, edit: dict) -> None:
        spec = edit.get("spec")
        if not isinstance(spec, dict):
            raise SpecError("replace needs a `spec` (a UI tree, same shape as render_ui)")
        self.guard_removal(node_id, "replace")
        self.touched.append(node_id)
        # Build into the node's PARENT and remove the old one, which is what
        # "replace" has to mean -- a node cannot be rendered into itself.
        #
        # Everything is scoped to a block: `const old` at the top level meant a
        # second replace in one batch was a redeclaration, and the whole script
        # died as a syntax error before anything ran.
        self.lines.append("try {")
        self.lines.append("{")
        self.lines.append(f"  const old = await figma.getNodeByIdAsync({_js(node_id)});")
        self.lines.append("  if (!old || old.removed) { throw new Error('node to replace is gone'); }")
        self.lines.append("  const target = old.parent;")
        self.lines.append("  if (!target) { throw new Error('node to replace has no parent'); }")
        # The last line of defence, checked against Figma itself rather than
        # against the inventory we read a moment ago: a node whose parent is the
        # PAGE is a screen, and replacing one is never an edit.
        self.lines.append(
            "  if (target.type === 'PAGE') { throw new Error("
            "'that node is a whole screen -- replacing it would empty the page'); }"
        )
        self.lines.append("  const at = target.children.indexOf(old);")
        self.lines.append(self._render_body("target", spec, [node_id], raw_parent=True))
        self.lines.append("  await _placeLast(at);")
        self.lines.append("}")
        self.lines.append(f"  applied.push({_js(f'replace on {node_id}')});")
        self.lines.append("} catch (e) {")
        self.lines.append(
            f"  failed.push({_js(f'replace on {node_id}')} + ': ' "
            "+ String(e && e.message ? e.message : e));"
        )
        self.lines.append("}")

    def _render_body(self, parent, spec, replace_ids, raw_parent=False) -> str:
        code, _ = renderer.compile_spec(
            spec,
            parent,
            self.roles,
            replace_ids=replace_ids,
            token_names=list(self.tokens.values()),
        )
        return _inline_render(code, parent, raw_parent)

    def _render_into(self, parent, spec, replace_ids, label, index=None, raw_parent=False):
        # The renderer emits a standalone script. Inline its body, keeping the
        # ids it creates, so a batch of edits stays one round trip.
        body = self._render_body(parent, spec, replace_ids, raw_parent)
        self.lines.append("try {")
        self.lines.append(body)
        if index is not None:
            self.lines.append(f"  await _placeLast({index});")
        self.lines.append(f"  applied.push({_js(label)});")
        self.lines.append("} catch (e) {")
        self.lines.append(
            f"  failed.push({_js(label)} + ': ' + String(e && e.message ? e.message : e));"
        )
        self.lines.append("}")


def _spacing(value, field_name: str) -> int:
    name = str(value)
    if name not in SPACING:
        raise SpecError(f"{field_name} must be one of {', '.join(SPACING)}; got {name!r}")
    return SPACING[name]


def _inline_render(code: str, parent, raw_parent: bool) -> str:
    """Turn a standalone render script into a block that runs inside the batch.

    The renderer's preamble resolves the parent, loads fonts and builds the
    style lookups -- all of which the batch has already done, and two `const
    _paintByName` declarations in one scope is a syntax error, not a subtle bug.
    """
    body = code
    # Its own `return` would end the whole batch.
    body = body.replace("return { createdNodeIds: created };", "for (const id of created) { madeIds.push(id); }")
    # Its preamble redeclares things the batch owns. Rename the whole block's
    # bindings by scoping it -- a block statement gives every `const` in the
    # inlined script its own scope, so nothing collides.
    parent_expr = parent if raw_parent else f"await figma.getNodeByIdAsync({_js(parent)})"
    body = body.replace(
        f"const root0 = await figma.getNodeByIdAsync({_js(parent)});", f"const root0 = {parent_expr};"
    )
    if raw_parent:
        body = body.replace("const root0 = await figma.getNodeByIdAsync(target);", "const root0 = target;")
    lines = ["  {"] + [f"  {line}" for line in body.splitlines()] + ["  }"]
    return "\n".join(lines)


PREAMBLE = """\
const applied = [];
const failed = [];
const madeIds = [];
__FONTS__
const _paint = await figma.getLocalPaintStylesAsync();
const _paintByName = {};
for (const s of _paint) { _paintByName[s.name] = s; }
const _texts = await figma.getLocalTextStylesAsync();
const _textByName = {};
for (const t of _texts) { _textByName[t.name] = t; }
// Move whatever was just inserted to a specific position among its siblings.
// ASYNC on purpose: the sync `figma.getNodeById` throws under this plugin's
// documentAccess: "dynamic-page" (knowledge/gotchas.md), and it threw AFTER the
// replace had already removed the old node -- so the edit reported failure with
// the original gone.
async function _placeLast(at) {
  if (typeof at !== 'number' || at < 0 || !madeIds.length) { return; }
  const node = await figma.getNodeByIdAsync(madeIds[madeIds.length - 1]);
  if (node && node.parent) {
    node.parent.insertChild(Math.max(0, Math.min(at, node.parent.children.length - 1)), node);
  }
}
"""


def compile_edits(
    edits: list[dict],
    resolve,
    color_roles: dict[str, str],
    token_names: list[str] | None = None,
    protected_ids: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Turn a list of edits into ONE script. Returns `(js, touched_node_ids)`.

    `resolve` is `inventory.resolve` bound to the current canvas: it turns
    whatever the model named -- an id, a list of ids, or a `find` selector --
    into verified ids, or an error naming what was wrong. A target that does
    not resolve raises here, before anything reaches Figma.
    """
    if not isinstance(edits, list) or not edits:
        raise SpecError("`edits` must be a non-empty list of edit objects")
    if len(edits) > MAX_EDITS:
        raise SpecError(f"{len(edits)} edits is more than {MAX_EDITS}; split this across steps")

    compiler = _EditCompiler(color_roles, token_names or [], protected_ids or set())
    for position, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            raise SpecError(f"edit {position} is not an object")
        op = str(edit.get("op") or "").strip().lower()
        if op not in OPS:
            raise SpecError(f"edit {position}: unknown op {op!r}. Valid ops: {', '.join(OPS)}")

        if op == "insert":
            ids, error = resolve(edit.get("parent") or edit.get("target"))
            if error:
                raise SpecError(f"edit {position} (insert): {error}")
            compiler.insert(ids[0], edit)
            continue

        target = edit.get("target")
        ids, error = resolve(target)
        if error:
            raise SpecError(f"edit {position} ({op}): {error}")
        # A SELECTOR that removes things is the dangerous shape. `set_fill` on
        # everything matching "Button" is a fine bulk edit; `replace` on the
        # same match silently deleted every frame on a real user's page. The
        # model can still remove several nodes -- it just has to name them, so
        # the count is something it chose rather than something it discovered.
        if op in DESTRUCTIVE_OPS and isinstance(target, dict) and len(ids) > MAX_SELECTOR_REMOVALS:
            raise SpecError(
                f"edit {position} ({op}): that selector matches {len(ids)} nodes, and "
                f"{op} removes what it matches. Name the specific ids you mean "
                f"(at most {MAX_SELECTOR_REMOVALS} per selector), so nothing is removed "
                "by accident."
            )
        for node_id in ids:
            getattr(compiler, op)(node_id, edit)

    fonts = "\n".join(
        f"await figma.loadFontAsync({{ family: {_js(f)}, style: {_js(s)} }});"
        for f, s in sorted(compiler.fonts)
    )
    script = (
        PREAMBLE.replace("__FONTS__", fonts)
        + "\n".join(compiler.lines)
        + "\nreturn { createdNodeIds: madeIds, appliedEdits: applied, failedEdits: failed };\n"
    )
    return script, list(dict.fromkeys(compiler.touched))
