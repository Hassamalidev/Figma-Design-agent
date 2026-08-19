"""Tool JSON schemas + dispatch to the functions the model may call.

Keep this set small (CLAUDE.md section 3): execute_figma_js, get_metadata,
get_screenshot, query_docs. Adding a tool means adding it here and nowhere
else -- the loop just dispatches by name.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from agent import editor, renderer
from bridge.server import Bridge
from tools.docs import query_docs
from tools.figma_exec import execute_figma_js
from tools.figma_read import get_metadata, get_screenshot

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "render_ui",
            "description": (
                "PREFERRED: build a section by describing WHAT it contains, as a UI tree. "
                "The harness compiles it into correct Figma API calls -- it handles font "
                "loading, sizing modes, auto-layout, spacing and token colours for you, so "
                "none of those can go wrong. Use this for every visible section. "
                "Only fall back to execute_figma_js for something this cannot express."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spec": {
                        "type": "object",
                        "description": (
                            "One UI tree. Every node has a `kind`: "
                            "section/card/row/col (containers, take `children`), "
                            "text, button, badge, input, avatar, divider, box. "
                            'Example: {"kind":"section","name":"Sign in","children":['
                            '{"kind":"text","style":"Heading","value":"Welcome back"},'
                            '{"kind":"input","label":"Email","placeholder":"you@co.com"},'
                            '{"kind":"button","label":"Sign in","variant":"primary"}]}'
                        ),
                    }
                },
                "required": ["spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_ui",
            "description": (
                "Change nodes that ALREADY EXIST on the canvas. Pass a list of small, "
                "explicit edits; the harness compiles them into correct Figma API calls, "
                "verifies every target id against the canvas first, and reports which "
                "edits took and which did not. This is the only way to modify an "
                "existing design -- never rebuild a screen to change one thing about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "description": (
                            "Each edit is {op, target, ...}. `target` is a node id copied "
                            "EXACTLY from the canvas listing (or a list of ids, or "
                            '{"name": "...", "type": "TEXT"} to match several). Ops: '
                            'set_fill {color: a ROLE or token}, set_text {value}, '
                            "set_text_style {style}, set_size {width, height}, "
                            "set_spacing {gap, padding}, set_radius {radius}, "
                            "set_visible {visible}, set_name {name}, reorder {index}, "
                            "delete, insert {parent, spec, index}, replace {target, spec}. "
                            "`spec` is a render_ui UI tree. Example: "
                            '[{"op":"set_fill","target":"1:9","color":"accent"},'
                            '{"op":"set_text","target":"1:10","value":"Sign in"}]'
                        ),
                        "items": {"type": "object"},
                    }
                },
                "required": ["edits"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_figma_js",
            "description": (
                "Run ONE small, atomic script inside Figma's Plugin API sandbox. "
                "The script is the body of an async function: it may `await` "
                "Plugin API calls and MUST end with `return { createdNodeIds: [...] }`. "
                "Do one logical operation per call (create a node, set its props, "
                "parent it) -- never batch many operations into one script."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The JavaScript to run, as an async function body.",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metadata",
            "description": (
                "Read structural metadata (id, name, type, children) for a node, "
                "or for the whole document if node_id is omitted. Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Node id to inspect. Omit to inspect the current page.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_screenshot",
            "description": (
                "Render a node (or the current page if node_id is omitted) as a PNG "
                "so you can visually check the result. Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Node id to render. Omit to render the current page.",
                    }
                },
            },
        },
    },
]
# ---- which tools a step may use -------------------------------------------
#
# A section step gets `render_ui` and the read-only tools -- and NOT
# `execute_figma_js`. This is CLAUDE.md rule 7 ("if it is mechanical, the
# harness writes it") taken to its conclusion.
#
# The renderer already handles font loading, creation order, resize-before-
# sizing-mode, append-before-FILL, style lookup and token colours. Yet in a
# real 29-step run the model reached for `execute_figma_js` every single time
# and lost steps to exactly those things:
#
#   in set_layoutSizingVertical: HUG can only be set on auto-layout frames
#   in set_layoutSizingHorizontal: FILL can only be set on children of ...
#   in appendChild: Reparenting would create a component inside a component
#   counterAxisAlignItems ... received 'END'
#   findAll callback crashed: TypeError: not a function
#
# Every one of those is impossible to express through `render_ui`. Telling the
# model to prefer it did not work; removing the alternative does. Steps that
# are not building a section keep the escape hatch.

_READ_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in ("get_metadata", "get_screenshot")]
_RENDER_TOOL = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "render_ui"]
_EDIT_TOOL = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "edit_ui"]
_EXEC_TOOL = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "execute_figma_js"]

SECTION_TOOLS: list[dict[str, Any]] = _RENDER_TOOL + _READ_TOOLS
# Edit mode gets `edit_ui` and the read-only tools -- and NOT `render_ui`,
# because building a fresh section is how "make the button purple" turns into a
# second copy of the whole screen. Structural changes go through edit_ui's own
# `insert` and `replace`, which are anchored to a node that already exists.
EDIT_TOOLS: list[dict[str, Any]] = _EDIT_TOOL + _READ_TOOLS
ALL_TOOLS: list[dict[str, Any]] = TOOL_SCHEMAS


def tools_for(section_step: bool) -> list[dict[str, Any]]:
    """The tools this step is allowed to call.

    Narrowing the toolset is a harness decision, so it lives here rather than
    in the loop: the loop asks "is this a section?" and gets back a list.
    """
    return SECTION_TOOLS if section_step else ALL_TOOLS


# `query_docs` is deliberately NOT offered as a tool. The whole gotchas corpus
# is ~4k tokens and is inlined in the system prompt instead, and the loop
# retrieves type signatures for each step automatically. As a tool it cost two
# or three round trips per step and needed its own budget guardrail to stop
# steps searching until they ran out of turns. `dispatch` still answers it, so
# a model that hallucinates the call gets documentation rather than an error.


def dispatch(
    name: str,
    arguments: dict[str, Any],
    bridge: Bridge,
    render_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a model tool call to the matching function and return its result.

    A malformed tool call (missing a required argument) must come back as a
    normal error the model can read and correct on the next turn -- never as
    an exception, which would kill the whole run over one bad turn.
    """
    if name == "render_ui":
        return _render_ui(arguments, bridge, render_context or {})
    if name == "edit_ui":
        return _edit_ui(arguments, bridge, render_context or {})
    if name == "execute_figma_js" and not arguments.get("code"):
        return {"ok": False, "error": "Missing required argument 'code'. Pass the script as the 'code' argument of the tool call."}
    if name == "query_docs" and not arguments.get("query"):
        return {"ok": False, "error": "Missing required argument 'query'."}

    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "execute_figma_js": lambda: execute_figma_js(bridge, arguments["code"]),
        "get_metadata": lambda: get_metadata(bridge, arguments.get("node_id")),
        "get_screenshot": lambda: get_screenshot(bridge, arguments.get("node_id")),
        "query_docs": lambda: {"ok": True, "result": query_docs(arguments["query"])},
    }
    handler = handlers.get(name)
    if handler is None:
        return {"ok": False, "error": f"Unknown tool: {name}"}
    return handler()


def coerce_spec(arguments: Any) -> dict | None:
    """Find the UI tree in whatever envelope the model wrapped it in.

    The schema asks for `{"spec": {...}}`, but the prompt teaches the shape by
    showing the bare tree `{"kind":"section", ...}` -- so a small model sends
    the tree AS the arguments object. In a real 5-step run that produced
    "Missing required argument 'spec'" twenty-five times and consumed roughly
    half of every step's tool-call budget, for a call that was otherwise
    perfectly correct.

    Being strict about the envelope buys nothing: the tree is unmistakable
    (`kind` at the top, or `children`), so every shape that unambiguously
    contains one is accepted. Anything genuinely unreadable still comes back as
    a normal error the model can correct.
    """
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None

    # The documented shape, including a spec handed over as a JSON string.
    for key in ("spec", "ui", "tree", "node", "root"):
        value = arguments.get(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if isinstance(value, dict):
            return coerce_spec(value) if "spec" in value else value
        if isinstance(value, list) and value:
            # A bare list of sections: wrap it, rather than losing all but one.
            return {"kind": "col", "name": "Section", "children": value}

    # The tree sent as the arguments object itself.
    if "kind" in arguments or "children" in arguments:
        return arguments
    return None


def _render_ui(arguments: dict[str, Any], bridge: Bridge, context: dict[str, Any]) -> dict[str, Any]:
    """Compile a UI tree and run it.

    A malformed spec comes back as a normal, readable error so the model can
    correct it -- and crucially it never reaches Figma, so a bad spec cannot
    leave half a section on the canvas.
    """
    spec = coerce_spec(arguments)
    if spec is None:
        return _bad_spec(
            "Missing the UI tree. Call render_ui with one argument named `spec` whose "
            'value is the tree object, e.g. {"spec": {"kind": "section", "name": "...", '
            '"children": [...]}}.'
        )
    parent_id = context.get("parent_id")
    if not parent_id:
        return _bad_spec("No parent frame is available for render_ui.")
    try:
        code, _ = renderer.compile_spec(
            spec,
            parent_id,
            context.get("color_roles") or {},
            replace_ids=context.get("replace_ids"),
            token_names=context.get("token_names"),
        )
    except renderer.SpecError as exc:
        return _bad_spec(f"Invalid UI spec: {exc}")
    return execute_figma_js(bridge, code)


def _edit_ui(arguments: dict[str, Any], bridge: Bridge, context: dict[str, Any]) -> dict[str, Any]:
    """Compile a batch of edits against the CURRENT canvas and run it.

    Every target is checked against the inventory here, in Python, before a
    single Plugin API call is made. Figma's own error for a bad id says nothing
    about which id was wrong or what the real ones are, so a model that gets it
    just invents another; the message this returns names the mistake and points
    back at the listing.
    """
    edits = arguments.get("edits")
    if isinstance(edits, dict):
        edits = [edits]  # one edit, unwrapped -- an obvious intent, not an error
    if isinstance(edits, str):
        try:
            edits = json.loads(edits)
        except json.JSONDecodeError:
            edits = None
    if not isinstance(edits, list) or not edits:
        return _bad_spec(
            "Missing required argument 'edits' (a non-empty list of edit objects, "
            'e.g. [{"op": "set_fill", "target": "1:9", "color": "accent"}]).'
        )
    resolve = context.get("resolve")
    if resolve is None:
        return _bad_spec("No canvas inventory is available for edit_ui.")
    try:
        code, touched = editor.compile_edits(
            edits,
            resolve,
            context.get("color_roles") or {},
            token_names=context.get("token_names"),
            protected_ids=context.get("protected_ids"),
        )
    except renderer.SpecError as exc:
        return _bad_spec(f"Invalid edit: {exc}")
    result = execute_figma_js(bridge, code)
    if result["ok"]:
        result = dict(result)
        result["touchedNodeIds"] = touched
    return result


def _bad_spec(message: str) -> dict[str, Any]:
    """A spec rejected in Python, before anything reached Figma.

    Flagged so the loop can tell it apart from a Figma runtime error. Nothing
    was attempted on the canvas, so the model should simply correct the JSON on
    its next turn rather than burning a whole retry of the step.
    """
    return {"ok": False, "error": message, "recoverable": True}
