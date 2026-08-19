"""Attachments: a screenshot, a spec document, a design brief -- turned into text.

The pipeline already knows how to build from a written instruction, and it is
very good at exactly one thing the user cannot always provide: words. So rather
than teaching the loop, the planner, the scaffold and the renderer about
images, attachments are converted ONCE, at the front, into the thing everything
downstream already consumes.

    screenshot.png ---> [vision model] ---> "REFERENCE 1 (screenshot): a dark
    spec.md        ---> [read as text] ---> sign-in screen, 1440x900... Deep
                                            background: #0B1020 ..."

Two rules that decide where that text is allowed to go:

1. **It feeds the BRIEF and the PALETTE, never the requirement check.**
   `agent/requirements.py` derives what the design must contain from the USER'S
   own words, precisely so the agent cannot set its own homework and mark it
   complete. Vision output is the model's words. It describes the reference
   faithfully, and it is genuinely the best source for the palette -- a
   screenshot's colours are facts about a real image -- but letting it become
   the instruction would quietly remove the one check the model does not grade.

2. **An attachment is never silently dropped.** If nothing can read it, the run
   says so and stops. A run that ignored the screenshot you attached and built
   something generic instead is worse than one that refused.
"""
from __future__ import annotations

import base64
import binascii
import logging
from dataclasses import dataclass

from agent.llm import ModelClient
from agent.prompts import IMAGE_REFERENCE_PROMPT, TEXT_REFERENCE_HEADER

logger = logging.getLogger(__name__)

# What the vision model is shown, and what is read straight through as text.
IMAGE_TYPES = {"png", "jpg", "jpeg", "webp", "gif"}
TEXT_TYPES = {
    "txt", "md", "markdown", "json", "csv", "css", "scss", "html", "htm",
    "svg", "yaml", "yml", "ts", "tsx", "js", "jsx", "xml",
}

# Per file and in total. A base64 image is ~4/3 of its bytes and the whole
# payload is held in memory by the stdlib HTTP server, so this is a real limit
# rather than a formality.
MAX_FILE_BYTES = 6 * 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
MAX_ATTACHMENTS = 6
# A long spec would otherwise crowd out the canvas listing and the palette in
# every single prompt, since the whole conversation is resent on every turn.
MAX_TEXT_CHARS = 6000


class ReferenceError(ValueError):
    """The attachment cannot be used. The message is shown to the user verbatim."""


@dataclass
class Attachment:
    name: str
    data: bytes

    @property
    def extension(self) -> str:
        return self.name.rsplit(".", 1)[-1].lower() if "." in self.name else ""

    @property
    def is_image(self) -> bool:
        return self.extension in IMAGE_TYPES

    @property
    def is_text(self) -> bool:
        return self.extension in TEXT_TYPES


def from_payload(items: list[dict]) -> list[Attachment]:
    """Decode what the dashboard posted, refusing anything unusable.

    Every limit is checked here, in one place, before any of it is held or
    sent anywhere. A 200MB file must fail as a readable message, not as a
    MemoryError three layers down.
    """
    if not items:
        return []
    if len(items) > MAX_ATTACHMENTS:
        raise ReferenceError(
            f"{len(items)} attachments is more than {MAX_ATTACHMENTS}. "
            "Attach the ones that matter most."
        )
    attachments: list[Attachment] = []
    total = 0
    for item in items:
        name = str((item or {}).get("name") or "attachment").strip()[:120]
        raw = (item or {}).get("data_base64") or ""
        try:
            data = base64.b64decode(str(raw), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ReferenceError(f"'{name}' could not be decoded ({exc}).") from exc
        if not data:
            raise ReferenceError(f"'{name}' is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise ReferenceError(
                f"'{name}' is {_size(len(data))}, over the {_size(MAX_FILE_BYTES)} limit."
            )
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise ReferenceError(
                f"The attachments total more than {_size(MAX_TOTAL_BYTES)}."
            )
        attachments.append(Attachment(name=name, data=data))
    return attachments


def check_readable(attachments: list[Attachment], has_vision: bool) -> None:
    """Refuse now, clearly, rather than ignoring the attachment mid-run."""
    for item in attachments:
        if item.is_text:
            continue
        if item.is_image:
            if not has_vision:
                raise ReferenceError(
                    f"'{item.name}' is an image, and no vision model is configured to read "
                    "it. Name one in Settings -> Vision model (or set VISION_MODEL_NAME in "
                    ".env), or describe the design in words instead."
                )
            continue
        raise ReferenceError(
            f"'{item.name}' is a .{item.extension or '?'} file, which cannot be read. "
            f"Attach an image ({', '.join(sorted(IMAGE_TYPES))}) or a text file "
            f"({', '.join(sorted(list(TEXT_TYPES)[:6]))}, ...). For a PDF, export the "
            "page as PNG."
        )


def describe(
    attachments: list[Attachment], vision: ModelClient | None
) -> tuple[str, list[str]]:
    """Turn every attachment into reference text. Returns `(text, warnings)`.

    Images cost one model call each -- far cheaper than the ~50 a build makes,
    and it is the call that decides whether the design resembles what the user
    actually showed you.
    """
    blocks: list[str] = []
    warnings: list[str] = []
    for index, item in enumerate(attachments, start=1):
        label = f"REFERENCE {index} ({item.name})"
        if item.is_image:
            described = _describe_image(item, vision)
            if described is None:
                warnings.append(f"Could not read the image '{item.name}'.")
                continue
            blocks.append(f"{label} -- a screenshot the user attached:\n{described}")
            logger.info("Read '%s' (%s).", item.name, _size(len(item.data)))
        elif item.is_text:
            blocks.append(f"{label} -- a document the user attached:\n{_as_text(item)}")
            logger.info("Read '%s' as text (%s).", item.name, _size(len(item.data)))
    if not blocks:
        return "", warnings
    return TEXT_REFERENCE_HEADER + "\n\n".join(blocks), warnings


def _describe_image(item: Attachment, vision: ModelClient | None) -> str | None:
    """Ask the vision model to write the screenshot down as a design brief.

    The output is deliberately shaped to feed the existing pipeline: an explicit
    `Name: #RRGGBB` colour list, because that is the shape `scaffold.
    extract_palette` reads, and a screen-by-screen breakdown, because that is
    what `planner.plan_screens` is asked next.
    """
    if vision is None:
        return None
    encoded = base64.b64encode(item.data).decode("ascii")
    mime = "jpeg" if item.extension in ("jpg", "jpeg") else item.extension
    messages = [
        {"role": "system", "content": IMAGE_REFERENCE_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Describe this design ({item.name})."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/{mime};base64,{encoded}"},
                },
            ],
        },
    ]
    try:
        reply = vision.complete(messages, tools=None)
    except Exception as exc:
        logger.info("Vision model could not read '%s': %s", item.name, exc)
        return None
    text = (getattr(reply, "content", "") or "").strip()
    return text or None


def _as_text(item: Attachment) -> str:
    """Decode a text attachment, trimming it to something a prompt can carry."""
    body = item.data.decode("utf-8", errors="replace").strip()
    if len(body) <= MAX_TEXT_CHARS:
        return body
    return body[:MAX_TEXT_CHARS] + f"\n... (trimmed at {MAX_TEXT_CHARS} characters)"


def _size(byte_count: int) -> str:
    if byte_count >= 1024 * 1024:
        return f"{byte_count / (1024 * 1024):.1f}MB"
    return f"{max(1, byte_count // 1024)}KB"
