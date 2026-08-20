"""The user's attached pictures, placed on the canvas as real images.

Weighted towards what must NOT happen. Two failures here are silent and both
are worse than an error: an attachment that is quietly ignored (the design is
built with grey boxes and the user watches it happen), and an image name that
resolves to the wrong picture.
"""
from __future__ import annotations

import base64
import json

import pytest

from agent import assets
from agent.reference import Attachment


def png(size: int = 64) -> Attachment:
    return Attachment(name="hero.png", data=b"\x89PNG\r\n\x1a\n" + b"x" * size)


def test_a_webp_is_readable_but_not_placeable_and_says_so():
    """figma.createImage takes PNG, JPEG and GIF only -- verified against the
    typings. A WEBP can still be READ by the vision model, so this is a warning
    about one file, never a refused run."""
    keep, warnings = assets.placeable([Attachment(name="shot.webp", data=b"RIFF....")])

    assert keep == []
    assert len(warnings) == 1
    assert "shot.webp" in warnings[0]
    assert "PNG" in warnings[0]  # says what to do about it


def test_a_text_attachment_is_not_an_asset_and_is_not_complained_about():
    """A spec document reaches the run as text. It is not missing; it simply
    is not a picture, and warning about it would be noise on every run."""
    keep, warnings = assets.placeable([Attachment(name="brand.md", data=b"# Brand")])

    assert keep == [] and warnings == []


def test_an_oversized_image_is_refused_with_its_size():
    huge = Attachment(name="big.png", data=b"x" * (assets.MAX_ASSET_BYTES + 1))

    keep, warnings = assets.placeable([huge])

    assert keep == []
    assert "big.png" in warnings[0]


def test_a_normal_png_is_kept():
    keep, warnings = assets.placeable([png()])

    assert [a.name for a in keep] == ["hero.png"]
    assert warnings == []


@pytest.mark.parametrize(
    "typed",
    ["hero.png", "hero", "Hero", "HERO.PNG", "hero image", "hero_png"],
)
def test_an_asset_is_found_however_the_model_retypes_its_name(typed):
    """The model reads the name out of a prompt and types it again. Failing on
    a changed hyphen would cost a whole step for nothing."""
    hero = assets.ImageAsset(name="hero.png", key="hero", image_hash="h1")

    assert assets.find([hero], typed) is hero


def test_one_asset_and_no_name_is_unambiguous():
    hero = assets.ImageAsset(name="hero.png", key="hero", image_hash="h1")

    assert assets.find([hero], None) is hero


def test_several_assets_and_no_name_is_a_choice_the_harness_will_not_make():
    """Which of three photographs goes in the hero is a design decision. The
    renderer turns this into a readable error naming the real files."""
    two = [
        assets.ImageAsset(name="a.png", key="a", image_hash="h1"),
        assets.ImageAsset(name="b.png", key="b", image_hash="h2"),
    ]

    assert assets.find(two, None) is None


def test_an_asset_can_be_named_by_position():
    two = [
        assets.ImageAsset(name="a.png", key="a", image_hash="h1"),
        assets.ImageAsset(name="b.png", key="b", image_hash="h2"),
    ]

    assert assets.find(two, "2") is two[1]
    assert assets.find(two, "image 1") is two[0]


def test_a_name_matching_nothing_finds_nothing():
    """Never a near-enough guess: painting the wrong picture is worse than an
    error the model can correct."""
    hero = assets.ImageAsset(name="hero.png", key="hero", image_hash="h1")

    assert assets.find([hero], "sidebar-illustration") is None


def test_the_upload_script_carries_the_bytes_and_returns_a_hash():
    script = assets.build_upload_script(b"\x89PNG\r\n\x1a\n")

    assert "figma.base64Decode(" in script
    assert "figma.createImage(" in script
    assert "getSizeAsync()" in script
    assert base64.b64encode(b"\x89PNG\r\n\x1a\n").decode() in script


def test_the_upload_script_is_valid_javascript():
    from tests.test_scaffold import compiles_as_async_body

    assert compiles_as_async_body(assets.build_upload_script(b"\x89PNG\r\n\x1a\n"))


def test_the_base64_payload_is_json_encoded_not_interpolated():
    """It is megabytes of untrusted-ish text going into a script body. It must
    be a JSON string literal, not spliced in raw."""
    script = assets.build_upload_script(b"hello")

    quoted = json.dumps(base64.b64encode(b"hello").decode())
    assert f"figma.base64Decode({quoted})" in script


def test_a_successful_upload_becomes_an_addressable_asset():
    asset = assets.parse_upload(
        "hero.png", {"imageHash": "abc123", "width": 1600, "height": 900}
    )

    assert asset is not None
    assert asset.key == "hero" and asset.image_hash == "abc123"
    assert round(asset.aspect, 2) == 1.78
    assert "1600x900" in asset.describe()


def test_an_upload_that_returned_no_hash_is_not_an_asset():
    """Reporting a picture as available when Figma did not store it means the
    design references an image that is not there."""
    assert assets.parse_upload("hero.png", {"width": 10}) is None
    assert assets.parse_upload("hero.png", None) is None


def test_an_asset_with_no_reported_size_still_has_a_usable_aspect():
    asset = assets.ImageAsset(name="a.png", key="a", image_hash="h")

    assert asset.aspect > 1  # 16:9, rather than a division by zero


def test_figmas_error_is_turned_into_something_to_do_about_it():
    """Same idea as loop.ERROR_HINTS: the raw message names the symptom and
    the fix is knowledge this project already has."""
    assert str(assets.MAX_IMAGE_PIXELS) in assets.upload_hint("Image is too large")
    assert "PNG" in assets.upload_hint("Invalid image")
    assert assets.upload_hint("something else entirely") == ""
