"""Read-only observation tools: get_metadata, get_screenshot.

These are how the loop grounds itself in what's actually on the canvas,
rather than trusting the model's memory of what it just did.
"""
from __future__ import annotations

from typing import Any

from bridge.protocol import Request
from bridge.server import Bridge, new_request_id


def get_metadata(bridge: Bridge, node_id: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Return structural metadata for a node (or the whole document if node_id is None)."""
    request = Request(id=new_request_id(), type="metadata", node_id=node_id)
    response = bridge.send(request, timeout=timeout)
    return {"ok": response.ok, "result": response.result, "error": response.error}


def get_screenshot(bridge: Bridge, node_id: str | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """Return a base64 PNG of a node (or the current page if node_id is None)."""
    request = Request(id=new_request_id(), type="screenshot", node_id=node_id)
    response = bridge.send(request, timeout=timeout)
    return {
        "ok": response.ok,
        "image_base64": response.image_base64,
        "error": response.error,
    }
