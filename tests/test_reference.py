"""Attachments: decoding them, refusing them, and turning them into text.

Weighted towards refusal, for the same reason the editor's tests are: the worst
outcome here is not an error, it is a run that silently ignored the screenshot
you attached and built something generic instead.
"""
from __future__ import annotations

import base64
import pathlib
from types import SimpleNamespace

import pytest

from agent import reference
from agent.state import RunState
from config import Settings


def payload(name: str, data: bytes) -> dict:
    return {"name": name, "data_base64": base64.b64encode(data).decode()}


class FakeVision:
    def __init__(self, reply="SCREENS\none screen\nCOLORS\nAccent: #6C5CE7"):
        self._reply = reply
        self.calls: list[list] = []

    def complete(self, messages, tools=None):
        self.calls.append(messages)
        return SimpleNamespace(content=self._reply, tool_calls=None)


# ---- decoding -------------------------------------------------------------


def test_an_attachment_round_trips_through_base64():
    items = reference.from_payload([payload("shot.png", b"\x89PNG fake")])

    assert len(items) == 1
    assert items[0].name == "shot.png" and items[0].is_image
    assert items[0].data == b"\x89PNG fake"


def test_a_text_file_is_recognised_as_text_not_an_image():
    item = reference.from_payload([payload("spec.md", b"# Spec")])[0]

    assert item.is_text and not item.is_image


@pytest.mark.parametrize(
    "items, expected",
    [
        ([payload("a.png", b"x")] * (reference.MAX_ATTACHMENTS + 1), "more than"),
        ([payload("big.png", b"x" * (reference.MAX_FILE_BYTES + 1))], "over the"),
        ([{"name": "bad.png", "data_base64": "!!!not base64!!!"}], "could not be decoded"),
        ([payload("empty.png", b"")], "is empty"),
    ],
)
def test_an_unusable_payload_fails_as_a_readable_message(items, expected):
    """A 200MB file must fail as something the user can act on, not as a
    MemoryError three layers down."""
    with pytest.raises(reference.ReferenceError) as caught:
        reference.from_payload(items)

    assert expected in str(caught.value)


def test_the_total_size_is_capped_even_when_each_file_is_legal():
    chunk = b"x" * (reference.MAX_FILE_BYTES - 1)
    many = [payload(f"{i}.png", chunk) for i in range(5)]

    with pytest.raises(reference.ReferenceError) as caught:
        reference.from_payload(many)

    assert "total" in str(caught.value)


# ---- refusing what cannot be read -----------------------------------------


def test_a_placeable_image_with_no_vision_model_warns_rather_than_refusing():
    """This line moved when images started being PLACED as well as read.

    A PNG with no vision model used to be a dead end. It is now a picture the
    design can genuinely show -- it just cannot be built to RESEMBLE it -- and
    refusing to start over that is refusing work the agent can really do. The
    warning is loud and names the setting that fixes it.
    """
    items = reference.from_payload([payload("shot.png", b"data")])

    warnings = reference.check_readable(items, has_vision=False)

    assert len(warnings) == 1
    assert "PLACED" in warnings[0] and "not read" in warnings[0]
    assert "Settings -> Vision model" in warnings[0]   # ...and how to fix it


def test_an_unplaceable_image_with_no_vision_model_is_still_refused():
    """A WEBP can neither be read nor put on the canvas, so accepting it would
    mean building something generic while the user watches."""
    items = reference.from_payload([payload("shot.webp", b"data")])

    with pytest.raises(reference.ReferenceError) as caught:
        reference.check_readable(items, has_vision=False)

    assert "no vision model" in str(caught.value)
    assert "Settings -> Vision model" in str(caught.value)


def test_an_image_with_a_vision_model_is_accepted():
    items = reference.from_payload([payload("shot.png", b"data")])

    reference.check_readable(items, has_vision=True)   # does not raise


def test_a_pdf_is_refused_with_the_thing_to_do_instead():
    items = reference.from_payload([payload("spec.pdf", b"%PDF-1.4")])

    with pytest.raises(reference.ReferenceError) as caught:
        reference.check_readable(items, has_vision=True)

    assert "export the page as PNG" in str(caught.value).lower() or "PNG" in str(caught.value)


def test_a_text_file_needs_no_vision_model():
    items = reference.from_payload([payload("spec.md", b"# Spec")])

    reference.check_readable(items, has_vision=False)   # does not raise


# ---- turning them into text -----------------------------------------------


def test_a_screenshot_becomes_a_brief_the_pipeline_can_build_from():
    vision = FakeVision()
    items = reference.from_payload([payload("login.png", b"data")])

    text, warnings = reference.describe(items, vision)

    assert warnings == []
    assert "REFERENCE 1 (login.png)" in text
    assert "Accent: #6C5CE7" in text
    # The image really was sent, not just its filename.
    content = vision.calls[0][1]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_the_vision_prompt_asks_for_the_shape_the_palette_parser_reads():
    """`scaffold.extract_palette` reads `Name: #RRGGBB`. Asking for anything
    else means a screenshot's colours never become tokens."""
    from agent.prompts import IMAGE_REFERENCE_PROMPT

    assert "#RRGGBB" in IMAGE_REFERENCE_PROMPT
    for role in ("Background:", "Surface:", "Border:", "Text:", "Accent:"):
        assert role in IMAGE_REFERENCE_PROMPT


def test_a_screenshots_colours_really_do_become_tokens():
    """The whole chain: image -> vision text -> design source -> palette."""
    from agent import scaffold

    vision = FakeVision(
        "COLORS\nBackground: #0B1020\nSurface: #F9FAFB\nBorder: #E5E7EB\n"
        "Text: #111827\nText muted: #6B7280\nAccent: #6C5CE7"
    )
    text, _ = reference.describe(reference.from_payload([payload("s.png", b"d")]), vision)
    state = RunState(instruction="rebuild this")
    state.references = text

    palette = scaffold.extract_palette("", state.design_source())

    assert {h for _, h in palette} >= {"#0B1020", "#F9FAFB", "#E5E7EB", "#111827", "#6C5CE7"}


def test_a_text_attachment_is_carried_through_and_trimmed():
    long_spec = b"# Spec\n" + b"detail line\n" * 5000

    text, _ = reference.describe(reference.from_payload([payload("spec.md", long_spec)]), None)

    assert "# Spec" in text
    assert len(text) < len(long_spec)
    assert "trimmed" in text


def test_a_vision_model_that_errors_warns_rather_than_taking_the_run_down():
    class Broken:
        def complete(self, messages, tools=None):
            raise RuntimeError("400 image rejected")

    text, warnings = reference.describe(
        reference.from_payload([payload("s.png", b"d")]), Broken()
    )

    assert text == ""
    assert warnings and "s.png" in warnings[0]


# ---- what the reference text is allowed to influence ----------------------


def test_attachments_feed_the_design_but_never_the_requirement_check():
    """CLAUDE.md section 8d: requirements come from the USER'S words, so the
    agent cannot write its own homework and then mark it complete."""
    from agent import requirements

    state = RunState(instruction="a sign-in screen")
    state.references = "REFERENCE 1: the screen has a search field and a data table."

    assert "search field" in state.design_source()
    labels = {r.label for r in requirements.expected_requirements(state.instruction)}
    assert "search field" not in labels
    assert "data table" not in labels
    # ...and the run really does grade against the instruction alone.
    source = pathlib.Path("agent/loop.py").read_text(encoding="utf-8")
    assert "requirements.check_coverage(state.instruction" in source


def test_vision_settings_fall_back_to_the_critic_so_one_key_unlocks_both():
    configured = Settings("http://m", "k", "gen", critic_model_name="gemma4:cloud",
                          critic_base_url="http://v")
    assert configured.has_vision
    assert configured.vision_settings() == ("http://v", "k", "gemma4:cloud")

    explicit = configured.with_overrides(vision_model_name="qwen-vl", vision_base_url="http://w")
    assert explicit.vision_settings() == ("http://w", "k", "qwen-vl")

    assert not Settings("http://m", "k", "gen").has_vision
