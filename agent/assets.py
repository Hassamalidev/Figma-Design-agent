"""An attached picture, placed on the canvas as a real image -- not just words.

`agent/reference.py` turns an attachment into TEXT: a vision model writes the
screenshot down as a brief, and the whole pipeline builds from that. That is
the right answer for "make it look like this", and it is the only answer for a
spec document. It is the wrong answer for "put my logo in the header": the
design ends up with a grey box where the picture should be, and no amount of
describing a photograph reproduces the photograph.

So an attachment now travels down BOTH paths:

    hero.jpg ──► [vision model] ──► reference text ──► brief, screens, palette
             └─► [figma.createImage] ──► image hash ──► an IMAGE fill on a node

The second path is this module. It is deliberately thin -- upload the bytes
once per run, remember the hash, and let `agent/renderer.py` paint it onto any
node the spec asks for. An image in Figma is not a node: it is a handle stored
in the file, referenced by hash from a paint. Uploading it once and reusing the
hash is why a logo can appear on five screens for one round trip.

Three rules, all learned from the Plugin API's own limits:

1. **PNG, JPEG and GIF only.** `figma.createImage` rejects everything else,
   including the WEBP that `reference.IMAGE_TYPES` is happy to *read*. A file
   that can be described but not placed is a warning about that one file, never
   a refusal of the run.
2. **4096px is Figma's ceiling**, and the error it throws does not say so. The
   hint travels with the failure, the way `loop.ERROR_HINTS` does.
3. **An asset that cannot be uploaded is never silently absent.** The step
   prompt lists exactly the images that really exist, so the model cannot place
   one that failed -- and `renderer` refuses an unknown asset name rather than
   quietly rendering an empty box.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# What figma.createImage actually accepts. Verified against the typings:
# "The data passed in must be encoded as a PNG, JPEG, or GIF."
PLACEABLE_TYPES = {"png", "jpg", "jpeg", "gif"}

# Figma's own limit, from the same doc comment. Bigger throws.
MAX_IMAGE_PIXELS = 4096

# The base64 of one image travels inside the script that uploads it, so this is
# a real transport limit rather than a formality. `reference.MAX_FILE_BYTES`
# (6MB) is the outer bound; this is what may be PLACED.
MAX_ASSET_BYTES = 6 * 1024 * 1024


@dataclass(frozen=True)
class ImageAsset:
    """One uploaded image: what the model names, and what Figma needs."""

    name: str  # the file the user attached, e.g. "hero.jpg"
    key: str  # normalized, e.g. "hero" -- what a spec may say
    image_hash: str  # Figma's handle; goes straight into an IMAGE paint
    width: int = 0
    height: int = 0

    @property
    def aspect(self) -> float:
        """Width / height, defaulting to 16:9 when Figma did not report a size."""
        if self.width > 0 and self.height > 0:
            return self.width / self.height
        return 16 / 9

    def describe(self) -> str:
        """One line for the step prompt."""
        size = f"{self.width}x{self.height}" if self.width and self.height else "size unknown"
        return f'"{self.name}" ({size})'


def asset_key(name: str) -> str:
    """A loose key, so "Hero Image.PNG", "hero-image" and "hero image" match.

    The model retypes the file name from a prompt, and a design run must not
    fail because it dropped the extension or changed a hyphen to a space.
    """
    stem = str(name or "").rsplit(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def placeable(attachments: list) -> tuple[list, list[str]]:
    """Split the attachments into what can go on the canvas and why not.

    Returns `(placeable, warnings)`. A file that cannot be placed is still
    perfectly readable as a reference, so this never raises -- it explains.
    """
    keep, warnings = [], []
    for item in attachments or []:
        if not getattr(item, "is_image", False):
            continue  # a spec document; it reaches the run as text
        extension = item.extension
        if extension not in PLACEABLE_TYPES:
            warnings.append(
                f"'{item.name}' was read as a reference but cannot be placed on the "
                f"canvas -- Figma accepts PNG, JPEG and GIF images only. Export it as "
                f"a PNG to use it in the design itself."
            )
            continue
        if len(item.data) > MAX_ASSET_BYTES:
            warnings.append(
                f"'{item.name}' is too large to place on the canvas "
                f"({len(item.data) // (1024 * 1024)}MB)."
            )
            continue
        keep.append(item)
    return keep, warnings


def find(assets: list[ImageAsset], name: str | None) -> ImageAsset | None:
    """The asset a spec is asking for, matched the way a person would.

    Exact key first, then the file name, then a 1-based index ("2", "image 2"),
    then a substring. With exactly one asset uploaded and no name given at all,
    that one is the answer -- there is nothing else it could mean.
    """
    if not assets:
        return None
    wanted = str(name or "").strip()
    if not wanted:
        return assets[0] if len(assets) == 1 else None
    key = asset_key(wanted)
    for asset in assets:
        if asset.key == key or asset.name.lower() == wanted.lower():
            return asset
    # "2", "image 2", "reference 2" -- an index into the list the prompt shows.
    digits = re.findall(r"\d+", wanted)
    if digits:
        index = int(digits[0]) - 1
        if 0 <= index < len(assets):
            return assets[index]
    for asset in assets:
        if key and (key in asset.key or asset.key in key):
            return asset
    return None


def names(assets: list[ImageAsset]) -> str:
    """The valid asset names, for an error message the model can act on."""
    return ", ".join(f'"{a.name}"' for a in assets) or "(none)"


_UPLOAD_SCRIPT = """const bytes = figma.base64Decode(__DATA__);
const image = figma.createImage(bytes);
const size = await image.getSizeAsync();
return {
  createdNodeIds: [],
  imageHash: image.hash,
  width: Math.round(size.width),
  height: Math.round(size.height)
};
"""


def build_upload_script(data: bytes) -> str:
    """Store one image in the Figma file and hand back its hash.

    ONE image per script on purpose. They are few (six at most) and each is
    megabytes of base64 inside the script body, so batching them would make a
    single failure lose every image and a single message enormous.
    """
    return _UPLOAD_SCRIPT.replace("__DATA__", json.dumps(base64.b64encode(data).decode("ascii")))


def parse_upload(name: str, payload: dict | None) -> ImageAsset | None:
    """The uploaded image, or None if Figma did not give us a hash."""
    data = payload or {}
    image_hash = str(data.get("imageHash") or "")
    if not image_hash:
        return None
    return ImageAsset(
        name=name,
        key=asset_key(name),
        image_hash=image_hash,
        width=int(data.get("width") or 0),
        height=int(data.get("height") or 0),
    )


def upload_hint(error: str) -> str:
    """Turn Figma's message into one the user can act on.

    Same idea as `loop.ERROR_HINTS`: the raw error names a symptom, and the
    thing to do about it is knowledge this project already has.
    """
    lowered = (error or "").lower()
    if "too large" in lowered or "size" in lowered:
        return (
            f" Figma images are limited to {MAX_IMAGE_PIXELS}px in width and height -- "
            "resize the file and attach it again."
        )
    if "invalid" in lowered or "unsupported" in lowered or "decode" in lowered:
        return " Figma accepts PNG, JPEG and GIF only; re-export the file as a PNG."
    return ""
