"""The wire contract between the Python agent and the Figma plugin. Keep it tiny and stable."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

RequestType = Literal["exec", "screenshot", "metadata", "ping", "hello"]


@dataclass
class Request:
    id: str
    type: RequestType
    code: str | None = None  # for "exec"
    node_id: str | None = None  # for "screenshot"/"metadata"; None means current page

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Response:
    id: str
    ok: bool
    result: Any = None
    image_base64: str | None = None  # for screenshots
    error: str | None = None

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Response":
        return Response(
            id=data["id"],
            ok=data.get("ok", False),
            result=data.get("result"),
            image_base64=data.get("image_base64"),
            error=data.get("error"),
        )
