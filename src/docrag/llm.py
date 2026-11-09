"""Ollama client: batched embeddings and non-streaming chat completions."""

from __future__ import annotations

import httpx

from .config import get_settings

_settings = get_settings()
_client = httpx.Client(base_url=_settings.ollama_base_url, timeout=_settings.ollama_timeout)


def embed(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = _client.post(
            "/api/embed",
            json={"model": _settings.embed_model, "input": texts[start : start + batch_size]},
        )
        response.raise_for_status()
        vectors.extend(response.json()["embeddings"])

    if vectors and len(vectors[0]) != _settings.embed_dim:
        raise ValueError(
            f"{_settings.embed_model} returns {len(vectors[0])}-dimensional vectors but "
            f"EMBED_DIM={_settings.embed_dim}; set them to the same value"
        )
    return vectors


def chat(messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    response = _client.post(
        "/api/chat",
        json={
            "model": _settings.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def is_up() -> bool:
    try:
        return _client.get("/api/tags", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False
