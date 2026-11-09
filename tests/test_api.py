"""Route and security checks that need neither a database nor Ollama."""

import pytest
from fastapi import HTTPException

from docrag.api import app
from docrag.models import Document, User
from docrag.security import can_read, create_access_token, hash_password, require_roles, verify_password


def test_openapi_lists_every_route():
    paths = set(app.openapi()["paths"])
    assert {"/auth/token", "/auth/me", "/auth/users", "/documents", "/documents/{document_id}", "/chat",
            "/health"} <= paths


def test_password_hash_roundtrip():
    hashed = hash_password("Sifre1234")
    assert verify_password("Sifre1234", hashed)
    assert not verify_password("yanlis-sifre", hashed)


def test_token_is_a_jwt():
    assert create_access_token(User(username="ayse", role="ik")).count(".") == 2


def test_role_based_read_access():
    isg_doc = Document(allowed_roles=["isg", "personel"])
    assert can_read(User(role="isg"), isg_doc)
    assert can_read(User(role="admin"), isg_doc)
    assert not can_read(User(role="ik"), isg_doc)


def test_require_roles_rejects_other_roles():
    check = require_roles("admin")
    assert check(User(role="admin")).role == "admin"
    with pytest.raises(HTTPException) as exc:
        check(User(role="personel"))
    assert exc.value.status_code == 403
