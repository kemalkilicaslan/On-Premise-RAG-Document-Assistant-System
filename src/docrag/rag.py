"""Retrieval and answer generation.

The role filter is applied inside the vector query, so chunks from documents a user may not
read never reach the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from . import llm
from .models import Chunk, Document, User
from .prompt import NO_ANSWER, build_context, build_messages

__all__ = ["NO_ANSWER", "RetrievedChunk", "answer", "build_context", "search", "version_at_least"]


_iterative_scan: bool | None = None  # cached per process: does the server's pgvector support it?


def version_at_least(version: str, minimum: tuple[int, ...]) -> bool:
    parts = tuple(int(p) for p in version.split(".")[: len(minimum)] if p.isdigit())
    return parts >= minimum


def widen_filtered_scan(db: Session) -> None:
    """Keep a role-filtered HNSW scan from returning fewer than top_k rows.

    pgvector >= 0.8 can keep scanning until enough rows pass the filter (iterative scan);
    older versions only allow a larger candidate list (ef_search, default 40).
    """
    global _iterative_scan
    if _iterative_scan is None:
        version = db.scalar(text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")) or "0"
        _iterative_scan = version_at_least(version, (0, 8))
    if _iterative_scan:
        db.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
    else:
        db.execute(text("SET LOCAL hnsw.ef_search = 200"))


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    document_title: str
    page: int
    content: str
    score: float


def search(db: Session, user: User, query: str, top_k: int) -> list[RetrievedChunk]:
    [query_vector] = llm.embed([query])
    distance = Chunk.embedding.cosine_distance(query_vector)

    stmt = (
        select(Chunk, Document.title, distance.label("distance"))
        .join(Document)
        .order_by(distance)
        .limit(top_k)
    )
    if user.role != "admin":
        stmt = stmt.where(Document.allowed_roles.any(user.role))
        widen_filtered_scan(db)

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_title=title,
            page=chunk.page,
            content=chunk.content,
            score=1 - dist,
        )
        for chunk, title, dist in db.execute(stmt)
    ]


def answer(db: Session, user: User, question: str, top_k: int) -> tuple[str, list[RetrievedChunk]]:
    chunks = search(db, user, question, top_k)
    if not chunks:
        return NO_ANSWER, []
    return llm.chat(build_messages(chunks, question)), chunks
