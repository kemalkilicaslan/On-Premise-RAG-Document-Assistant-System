"""PDF ingestion: extract text per page, chunk, embed and store with the document's allowed roles."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import llm
from .chunking import chunk_pages, normalize
from .config import get_settings
from .models import Chunk, Document


class DuplicateDocumentError(Exception):
    pass


def extract_pages(pdf_bytes: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [normalize(page.extract_text() or "") for page in reader.pages]


def ingest_pdf(
    db: Session,
    *,
    pdf_bytes: bytes,
    filename: str,
    title: str,
    allowed_roles: list[str],
    uploaded_by: int | None,
) -> Document:
    settings = get_settings()
    sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    if db.scalar(select(Document.id).where(Document.sha256 == sha256)) is not None:
        raise DuplicateDocumentError("This document has already been uploaded")

    pages = extract_pages(pdf_bytes)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No text could be extracted; the PDF may be scanned and need OCR first")

    vectors = llm.embed([chunk.content for chunk in chunks])

    document = Document(
        title=title,
        filename=filename,
        sha256=sha256,
        allowed_roles=allowed_roles,
        page_count=len(pages),
        uploaded_by=uploaded_by,
        chunks=[
            Chunk(chunk_index=chunk.index, page=chunk.page, content=chunk.content, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    )
    db.add(document)
    db.commit()

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / f"{sha256}.pdf").write_bytes(pdf_bytes)
    return document
