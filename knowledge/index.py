"""Embed + retrieve (RAG) over the Plugin API knowledge base.

CLAUDE.md's Phase 3 calls for real embeddings (sentence-transformers + a
vector store). Until that's earned its keep, this module exposes the same
`retrieve(query) -> str` interface with plain keyword-overlap scoring over
`gotchas.md` and `api_types.d.ts` -- zero extra dependencies, and good enough
to hand the model the right section. Swapping in embeddings later only
touches this file; no caller changes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def _words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if len(w) > 2}


@dataclass
class Chunk:
    source: str
    heading: str
    text: str

    @property
    def keywords(self) -> set[str]:
        return _words(self.heading + " " + self.text)


def _split_markdown(text: str, source: str) -> list[Chunk]:
    """Split a gotchas-style doc into chunks at each `## heading`."""
    chunks: list[Chunk] = []
    heading = "intro"
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if buffer:
                chunks.append(Chunk(source, heading, "\n".join(buffer).strip()))
            heading = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    if buffer:
        chunks.append(Chunk(source, heading, "\n".join(buffer).strip()))
    return [c for c in chunks if c.text]


def _split_dts(text: str, source: str) -> list[Chunk]:
    """Split a .d.ts file into chunks at each top-level interface/type declaration."""
    chunks: list[Chunk] = []
    heading = "header"
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"(interface|type)\s+(\w+)", line)
        if match:
            if buffer:
                chunks.append(Chunk(source, heading, "\n".join(buffer).strip()))
            heading = match.group(2)
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        chunks.append(Chunk(source, heading, "\n".join(buffer).strip()))
    return [c for c in chunks if c.text]


def _load_chunks() -> list[Chunk]:
    gotchas = (KNOWLEDGE_DIR / "gotchas.md").read_text(encoding="utf-8")
    types = (KNOWLEDGE_DIR / "api_types.d.ts").read_text(encoding="utf-8")
    return _split_markdown(gotchas, "gotchas.md") + _split_dts(types, "api_types.d.ts")


_CHUNKS: list[Chunk] | None = None


def _chunks() -> list[Chunk]:
    global _CHUNKS
    if _CHUNKS is None:
        _CHUNKS = _load_chunks()
    return _CHUNKS


def gotchas_text() -> str:
    """The whole gotchas file, for inlining into the system prompt.

    The corpus is ~4k tokens -- small enough to simply always be in context.
    Retrieving it through a tool call cost the model two or three round trips
    per step, which is far more expensive than just carrying it.
    """
    return (KNOWLEDGE_DIR / "gotchas.md").read_text(encoding="utf-8")


def retrieve(
    query: str, max_chars: int = 2000, top_k: int = 4, sources: tuple[str, ...] | None = None
) -> str:
    """Return the top_k knowledge chunks most relevant to `query`, joined and trimmed.

    `sources` restricts the search to particular files. The loop uses it to ask
    for type signatures only, since the gotchas are already inlined in the
    system prompt and repeating them here would just burn context twice.
    """
    query_words = _words(query)
    if not query_words:
        return ""

    pool = [c for c in _chunks() if not sources or c.source in sources]
    scored = [
        (len(query_words & chunk.keywords), chunk)
        for chunk in pool
        if query_words & chunk.keywords
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    picked = [chunk for _, chunk in scored[:top_k]]
    if not picked:
        return ""

    joined = "\n\n".join(f"### {c.heading} ({c.source})\n{c.text}" for c in picked)
    return joined[:max_chars]
