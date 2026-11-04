"""Resume ingestion and retrieval.

This is the path that carries the attack. A resume is chunked, embedded and
stored; at screening time the chunks most relevant to the job description
are retrieved and placed in the prompt. The poisoned chunk arrives through
retrieval, exactly as it would in a production system - which is why this
is built as real retrieval rather than string concatenation.

One detail worth noticing: retrieval is *adversarially helpful* to the
attacker. An injected line written to look relevant to the job description
scores well on similarity and is therefore more likely to be retrieved than
the candidate's genuine experience. Retrieval does not merely fail to stop
the attack - it can actively select for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Segment, Trust
from .embeddings import Embedder
from .store import VectorStore


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    index: int


def chunk_text(
    text: str, source: str, max_chars: int = 480, overlap: int = 60
) -> list[Chunk]:
    """Split on paragraph boundaries, then pack to a character budget.

    Paragraph-first keeps an injected block intact rather than slicing it
    across chunks, which would hide it from both retrieval and detection.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(
                Chunk(
                    id=f"{source}::{len(chunks)}",
                    text=buf.strip(),
                    source=source,
                    index=len(chunks),
                )
            )
        buf = ""

    for para in paras:
        if len(para) > max_chars:
            flush()
            start = 0
            while start < len(para):
                piece = para[start : start + max_chars]
                chunks.append(
                    Chunk(
                        id=f"{source}::{len(chunks)}",
                        text=piece.strip(),
                        source=source,
                        index=len(chunks),
                    )
                )
                start += max_chars - overlap
            continue
        if len(buf) + len(para) + 2 > max_chars:
            flush()
        buf = f"{buf}\n\n{para}" if buf else para

    flush()
    return chunks


class ResumeIndex:
    """Ingest resumes, retrieve the chunks relevant to a role."""

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def ingest(self, text: str, source: str) -> list[Chunk]:
        chunks = chunk_text(text, source)
        if not chunks:
            return []
        vectors = self.embedder.embed([c.text for c in chunks], kind="document")
        self.store.add(
            ids=[c.id for c in chunks],
            texts=[c.text for c in chunks],
            vectors=vectors,
            metadatas=[{"source": c.source, "index": c.index} for c in chunks],
        )
        return chunks

    def retrieve(self, job_description: str, k: int = 4) -> list[Segment]:
        if self.store.count() == 0:
            return []
        qvec = self.embedder.embed([job_description], kind="query")[0]
        hits = self.store.query(qvec, k=k)
        return [
            Segment(
                trust=Trust.DOCUMENT,
                content=h.text,
                label=str(h.metadata.get("source", h.id)),
            )
            for h in hits
        ]
