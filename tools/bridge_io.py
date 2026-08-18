"""One timed, measured way to talk to the Figma plugin.

Every tool in this package sends a request and waits for a response, and each
one used to call `bridge.send` directly -- so the round trip that dominates a
run's wall clock was counted nowhere.

The timing lives here rather than inside `bridge/server.py` on purpose:
CLAUDE.md's rule is that the bridge moves messages and matches ids, with no
agent-side concerns in it. This is the agent side.
"""
from __future__ import annotations

import time

from agent import metrics
from bridge.protocol import Request, Response
from bridge.server import Bridge


def send(bridge: Bridge, request: Request, timeout: float) -> Response:
    """Send one request, recording its latency and whether it worked.

    A transport failure (timeout, plugin disconnected) is recorded and then
    re-raised unchanged: swallowing it here would turn "Figma is gone" into a
    confusing empty result several layers away from the cause.
    """
    started = time.monotonic()
    try:
        response = bridge.send(request, timeout=timeout)
    except Exception:
        metrics.current().observe_bridge(request.type, time.monotonic() - started, ok=False)
        raise
    metrics.current().observe_bridge(request.type, time.monotonic() - started, ok=response.ok)
    return response
