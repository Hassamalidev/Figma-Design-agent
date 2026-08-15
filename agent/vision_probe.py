"""Prove a configured critic can really SEE, before a run depends on it.

Two failure modes matter, and they look identical from the outside:

  1. The endpoint rejects images outright (a 400). Obvious.
  2. The endpoint ACCEPTS the request and silently ignores the image, answering
     from the text alone. This is the dangerous one -- critique still comes
     back, it just has nothing to do with the screen, and every defect it
     reports is invented.

So the probe sends an image whose content cannot be guessed from the prompt and
checks the answer against it. CLAUDE.md's rule: verify with a real request
rather than assuming from the model's name.
"""
from __future__ import annotations

import base64
import struct
import zlib
from dataclasses import dataclass

# The probe image is a solid block of ONE of these. The prompt never says which,
# so a model answering from text alone cannot do better than chance.
PROBE_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (220, 40, 40),
    "green": (40, 180, 80),
    "blue": (40, 90, 220),
    "yellow": (240, 210, 50),
}

PROBE_PROMPT = (
    "What is the single dominant colour of this image? "
    "Answer with exactly one word: red, green, blue, or yellow."
)


@dataclass
class ProbeResult:
    ok: bool
    detail: str
    reply: str = ""

    def __str__(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'}: {self.detail}"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_test_png(color: tuple[int, int, int], size: int = 64) -> str:
    """A base64 PNG of one solid colour, built with the standard library only.

    Hand-rolled rather than adding Pillow: this is the only image the project
    ever creates, and a new dependency for 20 lines is a bad trade.
    """
    raw = b"".join(b"\x00" + bytes(color) * size for _ in range(size))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode("ascii")


def build_probe_messages(image_b64: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROBE_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }
    ]


def _explain_failure(exc: Exception) -> str:
    """Say which thing went wrong, because the fixes are completely different.

    A live probe returned 410 "retired" and 404 "not found" for models this
    project would otherwise have recommended. Reporting those as "cannot accept
    images" sends you looking for a different vision model when the real
    problem is that the name is dead.
    """
    text = str(exc)
    lowered = text.lower()

    if "retired" in lowered or "410" in lowered:
        return (
            f"that model has been RETIRED by the provider ({text[:140]}). "
            "Pick a current model name -- this is not an image-support problem."
        )
    if "not found" in lowered or "404" in lowered:
        return (
            "no such model at this endpoint. Check the exact name (for Ollama, "
            "`ollama list` shows what you have; cloud names end in `-cloud` and "
            "must be pulled first)."
        )
    if "403" in lowered or "subscription" in lowered or "unauthor" in lowered or "401" in lowered:
        return f"the endpoint refused the request -- credentials or plan ({text[:140]})."
    if "400" in lowered or "image" in lowered:
        return (
            f"the endpoint rejected the IMAGE ({text[:140]}). This model is "
            "text-only -- pick a vision model."
        )
    return f"the request failed ({type(exc).__name__}: {text[:140]})."


def probe_vision(client, expected: str = "blue") -> ProbeResult:
    """Send a known image and check the model reports what is actually in it."""
    if expected not in PROBE_COLORS:
        raise ValueError(f"unknown probe colour {expected!r}")

    image = make_test_png(PROBE_COLORS[expected])
    try:
        message = client.complete(build_probe_messages(image), tools=None)
    except Exception as exc:
        return ProbeResult(False, _explain_failure(exc))

    reply = (getattr(message, "content", "") or "").strip()
    if not reply:
        return ProbeResult(False, "the model returned an empty reply.", reply)

    lowered = reply.lower()
    if expected in lowered:
        return ProbeResult(True, f"the model correctly identified the image as {expected}.", reply)

    others = [c for c in PROBE_COLORS if c != expected and c in lowered]
    if others:
        return ProbeResult(
            False,
            f"the model said {others[0]!r} but the image is {expected!r}. It is answering "
            "without looking -- critique from it would be invented, not observed.",
            reply,
        )
    return ProbeResult(
        False,
        "the model did not name a colour at all, so it is probably not seeing the image.",
        reply,
    )
