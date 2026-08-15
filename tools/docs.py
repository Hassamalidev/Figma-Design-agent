"""query_docs -- pulls the relevant slice of Plugin API knowledge into context.

This is what lets the harness carry Figma expertise instead of the model
having to know it from memory (see CLAUDE.md section 1).
"""
from __future__ import annotations

from knowledge.index import retrieve


NO_MATCH = (
    "No documentation matched that query. The knowledge base only covers Plugin API "
    "gotchas and core type definitions -- it is not a full API reference, and a rephrased "
    "search will not find more. Write your best attempt and call execute_figma_js: the "
    "error it returns is more informative than another search."
)


def query_docs(
    query: str, max_chars: int = 2000, sources: tuple[str, ...] | None = None
) -> str:
    """Return the most relevant gotchas + API type snippets for `query`.

    An empty result told the model nothing, so it would rephrase and search
    again until the step's whole turn budget was gone (seen repeatedly in
    live runs). Say "no match, stop searching" explicitly instead.
    """
    return retrieve(query, max_chars=max_chars, sources=sources) or NO_MATCH
