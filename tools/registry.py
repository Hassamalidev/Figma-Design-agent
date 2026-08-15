"""Tool JSON schemas + dispatch to the functions the model may call.

Keep this set small (CLAUDE.md section 3): execute_figma_js, get_metadata,
get_screenshot, query_docs. Adding a tool means adding it here and nowhere
else -- the loop just dispatches by name.
"""
from __future__ import annotations

from typing import Any, Callable

from agent import renderer
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


def _render_ui(arguments: dict[str, Any], bridge: Bridge, context: dict[str, Any]) -> dict[str, Any]:
    """Compile a UI tree and run it.

    A malformed spec comes back as a normal, readable error so the model can
    correct it -- and crucially it never reaches Figma, so a bad spec cannot
    leave half a section on the canvas.
    """
    spec = arguments.get("spec")
    if not isinstance(spec, dict):
        return _bad_spec("Missing required argument 'spec' (a UI tree object).")
    parent_id = context.get("parent_id")
    if not parent_id:
        return _bad_spec("No parent frame is available for render_ui.")
    try:
        code, _ = renderer.compile_spec(spec, parent_id, context.get("color_roles") or {})
    except renderer.SpecError as exc:
        return _bad_spec(f"Invalid UI spec: {exc}")
    return execute_figma_js(bridge, code)


def _bad_spec(message: str) -> dict[str, Any]:
    """A spec rejected in Python, before anything reached Figma.

    Flagged so the loop can tell it apart from a Figma runtime error. Nothing
    was attempted on the canvas, so the model should simply correct the JSON on
    its next turn rather than burning a whole retry of the step.
    """
    return {"ok": False, "error": message, "recoverable": True}
