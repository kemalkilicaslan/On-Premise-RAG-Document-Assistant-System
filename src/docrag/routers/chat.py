"""Question answering over the documents the current user may read."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import rag
from ..config import get_settings
from ..db import get_db
from ..models import User
from ..schemas import ChatRequest, ChatResponse, Source
from ..security import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def ask(body: ChatRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    answer, chunks = rag.answer(db, user, body.question, body.top_k or get_settings().top_k)
    return ChatResponse(
        answer=answer,
        sources=[
            Source(
                index=i,
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                page=chunk.page,
                score=round(chunk.score, 4),
                excerpt=chunk.content[:300],
            )
            for i, chunk in enumerate(chunks, start=1)
        ],
    )
