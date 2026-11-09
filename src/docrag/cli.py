"""Command-line client for the API.

    python -m docrag.cli upload data/raw/*.pdf --roles isg personel
    python -m docrag.cli ask "İşveren risk değerlendirmesi yapmakla yükümlü müdür?"
    python -m docrag.cli add-user ayse Sifre1234 ik

The password is read from --password or the RAG_PASSWORD environment variable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import httpx

ROLES = ["admin", "isg", "ik", "uretim", "personel"]  # kept local so the CLI needs no server settings


def login(client: httpx.Client, username: str, password: str) -> None:
    response = client.post("/auth/token", data={"username": username, "password": password})
    if response.status_code != 200:
        sys.exit(f"Login failed: {response.text}")
    client.headers["Authorization"] = f"Bearer {response.json()['access_token']}"


def upload(client: httpx.Client, args: argparse.Namespace) -> None:
    for path in map(Path, args.files):
        with path.open("rb") as f:
            response = client.post(
                "/documents",
                files={"file": (path.name, f, "application/pdf")},
                data={"allowed_roles": args.roles},
            )
        status = "ok     " if response.status_code == 201 else f"error {response.status_code}"
        print(f"{status} {path.name}: {response.json()}")


def ask(client: httpx.Client, args: argparse.Namespace) -> None:
    response = client.post("/chat", json={"question": args.question, "top_k": args.top_k})
    response.raise_for_status()
    body = response.json()
    print(body["answer"], "\n\nSources:")
    for s in body["sources"]:
        print(f"  [{s['index']}] {s['document_title']}, p.{s['page']} (score {s['score']})")


def add_user(client: httpx.Client, args: argparse.Namespace) -> None:
    payload = {"username": args.name, "password": args.new_password, "role": args.role}
    response = client.post("/auth/users", json=payload)
    print(response.status_code, response.json())


def main() -> None:
    parser = argparse.ArgumentParser(description="On-premise RAG document assistant client")
    parser.add_argument("--api", default=os.getenv("RAG_API", "http://localhost:8000"))
    parser.add_argument("--username", default=os.getenv("RAG_USERNAME", "admin"))
    parser.add_argument("--password", default=os.getenv("RAG_PASSWORD"))
    sub = parser.add_subparsers(dest="command", required=True)

    p_upload = sub.add_parser("upload", help="upload PDFs")
    p_upload.add_argument("files", nargs="+")
    p_upload.add_argument("--roles", nargs="+", required=True, choices=ROLES, help="roles that may read them")
    p_upload.set_defaults(func=upload)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--top-k", type=int, default=None)
    p_ask.set_defaults(func=ask)

    p_user = sub.add_parser("add-user", help="create a user (admin)")
    p_user.add_argument("name")
    p_user.add_argument("new_password")
    p_user.add_argument("role", choices=ROLES)
    p_user.set_defaults(func=add_user)

    args = parser.parse_args()
    if not args.password:
        sys.exit("Password required: --password or RAG_PASSWORD")

    with httpx.Client(base_url=args.api, timeout=600) as client:
        login(client, args.username, args.password)
        args.func(client, args)


if __name__ == "__main__":
    main()
