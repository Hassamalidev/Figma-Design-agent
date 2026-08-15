"""The vision probe: does a configured critic actually SEE?

The dangerous failure is not a rejected image -- that is loud. It is an
endpoint that accepts the request and silently ignores the image, so critique
still comes back and every defect in it is invented.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from agent import vision_probe


class FakeClient:
    def __init__(self, reply=None, error=None):
        self._reply, self._error = reply, error
        self.messages = None

    def complete(self, messages, tools=None):
        self.messages = messages
        if self._error:
            raise self._error
        return SimpleNamespace(content=self._reply, tool_calls=None)


def test_the_probe_image_is_a_real_png():
    raw = base64.b64decode(vision_probe.make_test_png((40, 90, 220)))
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in raw and b"IDAT" in raw and raw.endswith(b"IEND\xae\x42\x60\x82")


def test_the_probe_sends_the_image_as_an_image_not_as_text():
    client = FakeClient(reply="blue")
    vision_probe.probe_vision(client)

    content = client.messages[0]["content"]
    assert any(part.get("type") == "image_url" for part in content)
    assert "data:image/png;base64," in str(content)


def test_the_prompt_never_reveals_the_answer():
    """Otherwise a text-only model passes by reading the question."""
    assert "blue" not in vision_probe.PROBE_PROMPT.lower().split("one word:")[0]


def test_a_model_that_sees_the_image_passes():
    assert vision_probe.probe_vision(FakeClient(reply="Blue.")).ok


def test_a_rejected_image_fails_with_a_clear_reason():
    result = vision_probe.probe_vision(
        FakeClient(error=RuntimeError("Error code: 400 - image input not supported"))
    )
    assert not result.ok
    assert "text-only" in result.detail


def test_a_retired_model_is_not_reported_as_an_image_problem():
    """A real probe hit 410 "qwen3-vl:235b was retired". Calling that an
    image-support failure sends you hunting for a different vision model when
    the actual problem is a dead model name."""
    result = vision_probe.probe_vision(
        FakeClient(error=RuntimeError("Error code: 410 - qwen3-vl:235b was retired at 2026-06-16"))
    )
    assert not result.ok
    assert "RETIRED" in result.detail
    assert "image" not in result.detail.lower().replace("image-support", "")


def test_an_unknown_model_says_so_rather_than_blaming_vision():
    result = vision_probe.probe_vision(
        FakeClient(error=RuntimeError('Error code: 404 - model "x:1b" not found'))
    )
    assert not result.ok
    assert "no such model" in result.detail


def test_a_credentials_failure_is_named_as_one():
    result = vision_probe.probe_vision(
        FakeClient(error=RuntimeError("Error code: 403 - requires a subscription"))
    )
    assert not result.ok
    assert "credentials or plan" in result.detail


def test_a_model_that_guesses_the_wrong_colour_fails():
    """The whole point: the endpoint answered, but it was not looking."""
    result = vision_probe.probe_vision(FakeClient(reply="red"), expected="blue")
    assert not result.ok
    assert "without looking" in result.detail


def test_a_reply_naming_no_colour_fails():
    result = vision_probe.probe_vision(FakeClient(reply="I cannot process images."))
    assert not result.ok
    assert "not seeing the image" in result.detail


def test_an_empty_reply_fails():
    assert not vision_probe.probe_vision(FakeClient(reply="")).ok


def test_every_probe_colour_can_be_rendered_and_checked():
    for name in vision_probe.PROBE_COLORS:
        assert vision_probe.probe_vision(FakeClient(reply=name), expected=name).ok


def test_an_unknown_probe_colour_is_rejected_loudly():
    with pytest.raises(ValueError):
        vision_probe.probe_vision(FakeClient(reply="x"), expected="chartreuse")
