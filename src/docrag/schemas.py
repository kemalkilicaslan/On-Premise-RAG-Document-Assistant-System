"""Request and response models, and the role list."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

# admin: everything · isg: occupational safety · ik: HR · uretim: production · personel: all staff
Role = Literal["admin", "isg", "ik", "uretim", "personel"]
ROLES: tuple[str, ...] = get_args(Role)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=72)
    role: Role


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    created_at: datetime


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    filename: str
    allowed_roles: list[str]
    page_count: int
    created_at: datetime


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    index: int
    document_id: int
    document_title: str
    page: int
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
