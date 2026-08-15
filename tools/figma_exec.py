"""execute_figma_js -- run one small, atomic script inside Figma.

The script body runs as the body of an async function inside the plugin
(see figma_plugin/code.js): it may `await` Plugin API calls and MUST `return`
the node ids it touched, e.g. `return { createdNodeIds: [frame.id] }`.

A failed script changes nothing in the document, so retrying is always safe.
"""
from __future__ import annotations

from typing import Any

from bridge.protocol import Request
from bridge.server import Bridge, new_request_id


def execute_figma_js(bridge: Bridge, code: str, timeout: float = 30.0) -> dict[str, Any]:
    """Send one script to the plugin and return its outcome.

    Returns a dict with at least `ok: bool`. On success, `result` holds
    whatever the script returned. On failure, `error` holds the message the
    Plugin API raised -- feed this straight back to the model, it is usually
    enough to fix the script.
    """
    request = Request(id=new_request_id(), type="exec", code=code)
    response = bridge.send(request, timeout=timeout)
    return {"ok": response.ok, "result": response.result, "error": response.error}
