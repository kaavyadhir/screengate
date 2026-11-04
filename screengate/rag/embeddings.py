"""Embedding backends.

Two implementations behind one interface, and the difference between them
matters for how results are reported.

GeminiEmbedder is the real one. It produces semantic embeddings, which is
what the semantic defense layer needs: two sentences that mean the same
thing land near each other even when they share no words.

HashingEmbedder is a deterministic offline stand-in built from character
n-gram hashing. It is NOT semantic - it measures surface overlap, so it
catches near-duplicates and misses genuine paraphrase. It exists so the
test suite and CI can exercise the plumbing without network or an API key.

Any number produced with HashingEmbedder is a plumbing check, not a
measurement of paraphrase resistance. The benchmark refuses to report
semantic-layer results when running on it, rather than quietly publishing
a figure that means nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    name: str
    semantic: bool
    """False for stand-ins whose numbers must not be reported as results."""

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]: ...


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("dimension mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashingEmbedder:
    """Offline, deterministic, non-semantic. For tests and CI only."""

    name = "hashing"
    semantic = False

    def __init__(self, dim: int = 512, ngram: int = 4) -> None:
        self.dim = dim
        self.ngram = ngram

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        norm = re.sub(r"\s+", " ", text.lower().strip())
        vec = [0.0] * self.dim
        if not norm:
            return vec
        for i in range(max(len(norm) - self.ngram + 1, 1)):
            gram = norm[i : i + self.ngram]
            h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        norm_len = math.sqrt(sum(v * v for v in vec))
        return [v / norm_len for v in vec] if norm_len else vec


class GeminiEmbedder:
    """Real semantic embeddings, on the supported `google-genai` SDK.

    `text-embedding-004` was retired along with the old SDK; the current
    stable embedding model is `gemini-embedding-001`.
    """

    name = "gemini"
    semantic = True

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-embedding-001",
    ) -> None:
        from ..providers.gemini import build_client

        self._client = build_client(api_key)
        self.model = model
        self._batch_ok = True
        """Cleared the first time a batched call returns the wrong count."""

    def _config(self, kind: str):
        """Task-type config, tolerating SDK shape changes."""
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        try:
            from google.genai import types

            return types.EmbedContentConfig(task_type=task)
        except Exception:
            return {"task_type": task}

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        from ..providers.gemini import with_retries

        if not texts:
            return []

        # Cache first. On a free tier this is the difference between a
        # benchmark run taking an hour and taking seconds: the corpus is
        # fixed, so every re-run would otherwise re-pay for identical
        # embeddings at one API call per line.
        cached = [_cache_get(self.model, kind, t) for t in texts]
        missing = [i for i, c in enumerate(cached) if c is None]
        if not missing:
            return [c for c in cached if c is not None]

        todo = [texts[i] for i in missing]
        fresh = self._embed_uncached(todo, kind)
        for i, vec in zip(missing, fresh):
            cached[i] = vec
            _cache_put(self.model, kind, texts[i], vec)
        return [c for c in cached if c is not None]

    def _embed_uncached(self, texts: list[str], kind: str) -> list[list[float]]:
        from ..providers.gemini import with_retries

        config = self._config(kind)

        def call(batch: list[str]):
            kwargs = {"model": self.model, "contents": batch}
            if config is not None:
                kwargs["config"] = config
            return self._client.models.embed_content(**kwargs)

        # Batching matters: the semantic layer embeds every line of every
        # document, so one call per line would multiply a benchmark run by
        # an order of magnitude and exhaust a free-tier quota immediately.
        #
        # But not every model batches. gemini-embedding-2 accepts a list
        # of three strings and returns ONE vector - it concatenates them
        # rather than embedding each. That is worse than an error: without
        # the count check in _vectors_from it would silently compare every
        # line against one blended vector, and layer 4's numbers would be
        # meaningless while looking perfectly healthy.
        #
        # So: try batching once, verify the count, and fall back to one
        # call per text for the life of this embedder if it lied.
        if self._batch_ok:
            try:
                resp = with_retries(lambda: call(texts))
                return _vectors_from(resp, len(texts))
            except _CountMismatch:
                self._batch_ok = False
                print(
                    f"  note: {self.model} does not batch embeddings; "
                    "falling back to one call per text",
                    flush=True,
                )

        out: list[list[float]] = []
        for text in texts:
            resp = with_retries(lambda t=text: call([t]))
            out.extend(_vectors_from(resp, 1))
        return out


_CACHE_DIR = Path(
    os.getenv("SCREENGATE_EMBED_CACHE", Path(__file__).resolve().parents[2] / ".embed_cache")
)
_CACHE: dict[str, list[float]] | None = None
_CACHE_PATH: Path | None = None


def _cache_key(model: str, kind: str, text: str) -> str:
    h = hashlib.sha256(f"{model}\x00{kind}\x00{text}".encode("utf-8")).hexdigest()
    return h[:32]


def _cache_load(model: str) -> dict[str, list[float]]:
    global _CACHE, _CACHE_PATH
    path = _CACHE_DIR / f"{model.replace('/', '_')}.json"
    if _CACHE is not None and _CACHE_PATH == path:
        return _CACHE
    try:
        _CACHE = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _CACHE = {}
    _CACHE_PATH = path
    return _CACHE


def _cache_get(model: str, kind: str, text: str) -> list[float] | None:
    return _cache_load(model).get(_cache_key(model, kind, text))


def _cache_put(model: str, kind: str, text: str, vec: list[float]) -> None:
    cache = _cache_load(model)
    cache[_cache_key(model, kind, text)] = vec
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        assert _CACHE_PATH is not None
        _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except OSError:
        pass  # a cache that cannot be written is a slowdown, not a failure


class _CountMismatch(RuntimeError):
    """The API returned a different number of vectors than texts sent.

    Its own exception type so the batching fallback can catch exactly
    this and nothing else - an auth failure or a retired model must still
    propagate rather than being retried one text at a time.
    """


def _vectors_from(resp, expected: int) -> list[list[float]]:
    """Pull plain float lists out of whatever shape the SDK returned.

    Written defensively on purpose: this project has already lost a run
    to an SDK change, and an embedding response is a container whose
    exact shape has moved between versions.
    """
    items = getattr(resp, "embeddings", None)
    if items is None and isinstance(resp, dict):
        items = resp.get("embeddings") or resp.get("embedding")
    if items is None:
        raise RuntimeError(f"unexpected embedding response: {type(resp)!r}")

    out: list[list[float]] = []
    for item in items:
        values = getattr(item, "values", None)
        if values is None and isinstance(item, dict):
            values = item.get("values")
        if values is None:
            values = item  # already a bare sequence of floats
        out.append([float(v) for v in values])

    if len(out) != expected:
        raise _CountMismatch(f"expected {expected} embeddings, got {len(out)}")
    return out


def build_embedder(settings) -> Embedder:
    """Pick a backend. 'auto' uses Gemini when a key is present."""
    choice = settings.embedder
    if choice == "hashing":
        return HashingEmbedder()
    if choice == "gemini":
        return GeminiEmbedder(
            api_key=settings.gemini_api_key, model=settings.embedding_model
        )
    if choice == "auto":
        if settings.gemini_api_key:
            try:
                return GeminiEmbedder(
                    api_key=settings.gemini_api_key, model=settings.embedding_model
                )
            except RuntimeError:
                return HashingEmbedder()
        return HashingEmbedder()
    raise ValueError(f"unknown embedder: {choice}")
