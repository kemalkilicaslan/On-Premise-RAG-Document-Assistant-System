"""Upload (admin), list (filtered by role) and delete (admin) documents."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..ingest import DuplicateDocumentError, ingest_pdf
from ..models import Document, User
from ..schemas import ROLES, DocumentOut
from ..security import get_current_user, require_roles

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: Annotated[UploadFile, File()],
    allowed_roles: Annotated[list[str], Form(description="Roles that may read the document")],
    title: Annotated[str | None, Form()] = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles("admin")),
):
    settings = get_settings()
    unknown = sorted(set(allowed_roles) - set(ROLES))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown role: {', '.join(unknown)}")
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are supported")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    pdf_bytes = file.file.read(max_bytes + 1)
    if len(pdf_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds the {settings.max_upload_mb} MB limit")

    try:
        return ingest_pdf(
            db,
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            title=title or Path(file.filename).stem,
            allowed_roles=sorted(set(allowed_roles)),
            uploaded_by=user.id,
        )
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("", response_model=list[DocumentOut])
def list_documents(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    stmt = select(Document).order_by(Document.created_at.desc())
    if user.role != "admin":
        stmt = stmt.where(Document.allowed_roles.any(user.role))
    return db.scalars(stmt).all()


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int, db: Session = Depends(get_db), _: User = Depends(require_roles("admin"))
):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(document)
    db.commit()
    (Path(get_settings().upload_dir) / f"{document.sha256}.pdf").unlink(missing_ok=True)
