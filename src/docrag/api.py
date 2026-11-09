"""FastAPI application.

    uvicorn docrag.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

from . import __version__, llm
from .db import SessionLocal, init_db
from .routers import auth, chat, documents
from .security import ensure_admin


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    with SessionLocal() as db:
        ensure_admin(db)
    yield


app = FastAPI(title="On-Premise RAG Document Assistant", version=__version__, lifespan=lifespan)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.exception_handler(httpx.HTTPError)
def llm_unavailable(_: Request, exc: httpx.HTTPError):
    return JSONResponse(status_code=503, content={"detail": f"LLM service unavailable: {exc}"})


@app.get("/health", tags=["health"])
def health():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    ollama_ok = llm.is_up()
    return JSONResponse(
        status_code=200 if db_ok and ollama_ok else 503,
        content={"database": db_ok, "ollama": ollama_ok},
    )
