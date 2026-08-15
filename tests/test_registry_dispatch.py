"""Malformed tool calls must come back as readable errors, never exceptions.

A live run died with `Run crashed: 'code'` when the model emitted an
execute_figma_js call with no `code` argument -- one bad turn killed the
whole run. These lock that behaviour down.
"""
from __future__ import annotations

from tools.registry import dispatch


class UnusedBridge:
    """Dispatch should reject malformed calls before touching the bridge."""

    def send(self, request, timeout: float = 30.0):
        raise AssertionError("dispatch should not have reached the bridge")


def test_execute_figma_js_without_code_returns_an_error():
    result = dispatch("execute_figma_js", {}, UnusedBridge())

    assert result["ok"] is False
    assert "code" in result["error"]


def test_execute_figma_js_with_empty_code_returns_an_error():
    result = dispatch("execute_figma_js", {"code": ""}, UnusedBridge())

    assert result["ok"] is False


def test_query_docs_without_query_returns_an_error():
    result = dispatch("query_docs", {}, UnusedBridge())

    assert result["ok"] is False


def test_unknown_tool_returns_an_error():
    result = dispatch("no_such_tool", {}, UnusedBridge())

    assert result["ok"] is False
    assert "Unknown tool" in result["error"]
