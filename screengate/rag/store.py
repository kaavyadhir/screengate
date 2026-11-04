"""Vector store.

ChromaDB when it is installed, an in-memory cosine store otherwise. The
interface is the same, so the retrieval path and the semantic defense layer
do not care which is behind them - and the test suite runs without Chroma.

Being honest about the sizing: at a few hundred vectors the in-memory store
is perfectly adequate and the cosine loop is not the bottleneck. Chroma is
here because it is how this is actually built at scale, with persistence
and metadata filtering, and switching stores later should not mean
rewriting the callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .embeddings import cosine


@dataclass
class Hit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(Protocol):
    name: str

    def add(
        self,
        ids: list[str],
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def query(self, vector: list[float], k: int = 5) -> list[Hit]: ...

    def count(self) -> int: ...


class InMemoryStore:
    """Exact cosine search. No dependencies, deterministic."""

    name = "memory"

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._vecs: list[list[float]] = []
        self._meta: list[dict[str, Any]] = []

    def add(self, ids, texts, vectors, metadatas=None) -> None:
        metadatas = metadatas or [{} for _ in ids]
        for i, t, v, m in zip(ids, texts, vectors, metadatas):
            if i in self._ids:
                pos = self._ids.index(i)
                self._texts[pos], self._vecs[pos], self._meta[pos] = t, v, m
                continue
            self._ids.append(i)
            self._texts.append(t)
            self._vecs.append(v)
            self._meta.append(m)

    def query(self, vector, k: int = 5) -> list[Hit]:
        scored = [
            Hit(self._ids[i], self._texts[i], cosine(vector, self._vecs[i]), self._meta[i])
            for i in range(len(self._ids))
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._ids)


class ChromaStore:
    """Persistent Chroma collection."""

    name = "chroma"

    def __init__(self, path: str = ".chroma", collection: str = "screengate") -> None:
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "chromadb is not installed. pip install -r requirements.txt"
            ) from exc
        self._client = chromadb.PersistentClient(path=path)
        # Embeddings are supplied by us, so Chroma's own embedding function
        # is explicitly disabled - otherwise it would silently use a
        # different model from the one the semantic layer was tuned on.
        self._col = self._client.get_or_create_collection(
            name=collection, embedding_function=None
        )

    def add(self, ids, texts, vectors, metadatas=None) -> None:
        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=vectors,
            metadatas=metadatas or [{} for _ in ids],
        )

    def query(self, vector, k: int = 5) -> list[Hit]:
        res = self._col.query(query_embeddings=[vector], n_results=k)
        hits: list[Hit] = []
        for i in range(len(res["ids"][0])):
            # Chroma returns squared L2 distance by default; convert to a
            # similarity in the same direction as cosine for a shared API.
            dist = res["distances"][0][i]
            hits.append(
                Hit(
                    id=res["ids"][0][i],
                    text=res["documents"][0][i],
                    score=1.0 / (1.0 + dist),
                    metadata=(res.get("metadatas") or [[{}]])[0][i] or {},
                )
            )
        return hits

    def count(self) -> int:
        return self._col.count()


def build_store(settings, collection: str = "screengate") -> VectorStore:
    choice = settings.vector_store
    if choice == "memory":
        return InMemoryStore()
    if choice == "chroma":
        return ChromaStore(path=settings.chroma_path, collection=collection)
    if choice == "auto":
        try:
            return ChromaStore(path=settings.chroma_path, collection=collection)
        except RuntimeError:
            return InMemoryStore()
    raise ValueError(f"unknown vector store: {choice}")
