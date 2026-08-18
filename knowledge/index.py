"""Embed + retrieve (RAG) over the Plugin API knowledge base.

`retrieve(query) -> str` is the whole public interface, and everything behind
it is a swappable backend (see `set_backend`). CLAUDE.md promised that swapping
in embeddings would touch this file and nothing else; the seam is now real
rather than aspirational.

**Default backend: BM25.** The previous scorer counted how many query words a
chunk contained, which rewards LONG chunks -- and the longest chunk in the
corpus is the typings file's preamble, so "set the line height on a text node"
retrieved the file header instead of `TextNode`. BM25 fixes exactly that: it
weights rare terms above common ones (IDF) and divides out chunk length, both
of which the old score ignored. Still zero dependencies and still pure Python.

**Identifier-aware tokenization.** `setBoundVariableForPaint` is now indexed
both whole and as set/bound/variable/for/paint, so plain-English step
descriptions -- which is all the planner ever emits -- can reach the API names
that answer them. That single change is what makes the queries in the tests
below resolve.

**Caching.** Three layers, because the same text is otherwise re-derived on
every step of every run: the parsed corpus, the term statistics computed from
it, and the answers themselves (`retrieve` is memoized -- a retrying step asks
the identical question up to `max_retries` times).
"""
from __future__ import annotations

import functools
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")
# Splits camelCase and PascalCase: "setBoundVariableForPaint" -> the words in it.
_CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

# Words that appear in almost every chunk of this corpus carry no signal, and
# in a corpus this small IDF alone does not fully discount them.
_STOPWORDS = frozenset(
    """the and for with that this from you your not are was were will can may
    use used using set get any all one two into out per its it's their there
    when what which while must never always only also than then them they has
    have had does did done else new old off via etc how why who whom whose""".split()
)

# BM25 parameters. These are the standard defaults and there is no tuning set
# to fit them to, so they stay at the values the literature reports.
BM25_K1 = 1.2
BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    """Words plus the parts of every compound identifier.

    Indexing `setBoundVariableForPaint` only as itself means a step that says
    "bind a paint to a colour variable" can never reach it -- which is the
    normal case, since plan steps are plain English by design.
    """
    tokens: list[str] = []
    for match in _WORD_RE.findall(text):
        lowered = match.lower()
        if len(lowered) > 2 and lowered not in _STOPWORDS:
            tokens.append(lowered)
        parts = _CAMEL_RE.findall(match)
        if len(parts) > 1:
            tokens.extend(
                part.lower()
                for part in parts
                if len(part) > 2 and part.lower() not in _STOPWORDS
            )
    return tokens


def _words(text: str) -> set[str]:
    """The distinct tokens in a string. Kept for callers that want a set."""
    return set(_tokenize(text))


@dataclass
class Chunk:
    source: str
    heading: str
    text: str
    # Computed once when the corpus loads. This used to be a @property, so
    # every scored chunk re-ran the tokenizer on every single retrieve() call.
    counts: dict[str, int] = field(default_factory=dict, repr=False)
    length: int = 0

    def index(self) -> "Chunk":
        """Populate the term counts. Called once, at load."""
        tokens = _tokenize(f"{self.heading} {self.heading} {self.text}")  # heading counts double
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self.counts = counts
        self.length = len(tokens)
        return self

    @property
    def keywords(self) -> set[str]:
        return set(self.counts)


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
    chunks = _split_markdown(gotchas, "gotchas.md") + _split_dts(types, "api_types.d.ts")
    return [c.index() for c in chunks]


# ---- the corpus and its term statistics, computed once ---------------------


@dataclass
class Corpus:
    """The parsed knowledge base plus everything BM25 needs to score against it."""

    chunks: list[Chunk]
    idf: dict[str, float]
    average_length: float


def _build_corpus() -> Corpus:
    chunks = _load_chunks()
    total = len(chunks) or 1
    document_frequency: dict[str, int] = {}
    for chunk in chunks:
        for term in chunk.counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    # Standard BM25 IDF. A term in every chunk scores ~0; a term in one scores high.
    idf = {
        term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
        for term, freq in document_frequency.items()
    }
    average = sum(c.length for c in chunks) / total
    return Corpus(chunks=chunks, idf=idf, average_length=average or 1.0)


_CORPUS: Corpus | None = None


def _corpus() -> Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _build_corpus()
    return _CORPUS


def _chunks() -> list[Chunk]:
    return _corpus().chunks


@functools.lru_cache(maxsize=1)
def gotchas_text() -> str:
    """The whole gotchas file, for inlining into the system prompt.

    The corpus is ~4k tokens -- small enough to simply always be in context.
    Retrieving it through a tool call cost the model two or three round trips
    per step, which is far more expensive than just carrying it. Cached: it is
    a fixed prefix read on every process, and re-reading a file to produce a
    byte-identical string is pure waste.
    """
    return (KNOWLEDGE_DIR / "gotchas.md").read_text(encoding="utf-8")


# ---- scoring backends ------------------------------------------------------


def _bm25_scores(query: str, pool: list[Chunk]) -> list[tuple[float, Chunk]]:
    """Rank `pool` against `query` with BM25.

    Two properties the old overlap count lacked, and both are why it misfired:
    rare terms outweigh common ones, and a long chunk no longer wins simply by
    containing more words.
    """
    corpus = _corpus()
    terms = _tokenize(query)
    if not terms:
        return []

    scored: list[tuple[float, Chunk]] = []
    for chunk in pool:
        score = 0.0
        for term in terms:
            frequency = chunk.counts.get(term)
            if not frequency:
                continue
            idf = corpus.idf.get(term, 0.0)
            norm = 1 - BM25_B + BM25_B * (chunk.length / corpus.average_length)
            score += idf * (frequency * (BM25_K1 + 1)) / (frequency + BM25_K1 * norm)
        if score > 0:
            scored.append((score, chunk))
    return scored


class KeywordBackend:
    """BM25 over the parsed corpus. Default, dependency-free, deterministic."""

    name = "keyword"

    def rank(self, query: str, pool: list[Chunk]) -> list[tuple[float, Chunk]]:
        return _bm25_scores(query, pool)


class EmbeddingBackend:
    """Dense retrieval over the same chunks, for when the corpus outgrows BM25.

    OPT-IN and self-disabling. `sentence-transformers` pulls in PyTorch, which
    is a heavier dependency than the rest of this project combined, so it is
    never imported unless someone asks for this backend by name -- and if the
    import fails we fall back to keyword scoring with a logged warning rather
    than taking a run down over a search index.

    Worth switching on when `api_types.d.ts` grows past a few thousand lines or
    the corpus gains prose that shares no vocabulary with the queries. At the
    current ~650 lines, BM25 measurably wins on the queries in tests/test_docs.py
    and costs nothing, which is why it stays the default.
    """

    name = "embeddings"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None
        self._vectors = None
        self._fallback = KeywordBackend()

    def _ensure_loaded(self) -> bool:
        if self._vectors is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            logger.info(
                "Embedding retrieval requested but sentence-transformers is not "
                "installed; falling back to keyword search. `pip install "
                "sentence-transformers` to enable it."
            )
            return False
        self._model = SentenceTransformer(self._model_name)
        chunks = _chunks()
        self._vectors = self._model.encode(
            [f"{c.heading}\n{c.text}" for c in chunks], normalize_embeddings=True
        )
        return True

    def rank(self, query: str, pool: list[Chunk]) -> list[tuple[float, Chunk]]:
        if not self._ensure_loaded():
            return self._fallback.rank(query, pool)
        wanted = {id(c) for c in pool}
        query_vector = self._model.encode([query], normalize_embeddings=True)[0]
        # Vectors are normalized, so the dot product IS cosine similarity.
        similarities = self._vectors @ query_vector
        return [
            (float(score), chunk)
            for score, chunk in zip(similarities, _chunks())
            if id(chunk) in wanted and score > 0
        ]


_BACKEND: KeywordBackend | EmbeddingBackend = KeywordBackend()


def set_backend(name: str) -> str:
    """Choose the retrieval backend. Returns the name actually in effect.

    An unknown name is not fatal: retrieval is a quality feature, and refusing
    to start a run over a typo in a config value would be a worse trade than
    searching slightly less well.
    """
    global _BACKEND
    retrieve.cache_clear()
    if name == "embeddings":
        _BACKEND = EmbeddingBackend()
    elif name == "keyword":
        _BACKEND = KeywordBackend()
    else:
        logger.info("Unknown retrieval backend %r; keeping keyword search.", name)
    return _BACKEND.name


def backend_name() -> str:
    return _BACKEND.name


@functools.lru_cache(maxsize=256)
def retrieve(
    query: str, max_chars: int = 2000, top_k: int = 4, sources: tuple[str, ...] | None = None
) -> str:
    """Return the top_k knowledge chunks most relevant to `query`, joined and trimmed.

    `sources` restricts the search to particular files. The loop uses it to ask
    for type signatures only, since the gotchas are already inlined in the
    system prompt and repeating them here would just burn context twice.

    Memoized: every retry of a step asks the identical question, and the corpus
    is immutable for the life of the process, so the second answer can only be
    the first one recomputed.
    """
    pool = [c for c in _chunks() if not sources or c.source in sources]
    scored = _BACKEND.rank(query, pool)
    if not scored:
        return ""

    scored.sort(key=lambda pair: pair[0], reverse=True)
    picked = [chunk for _, chunk in scored[:top_k]]

    joined = "\n\n".join(f"### {c.heading} ({c.source})\n{c.text}" for c in picked)
    return joined[:max_chars]
