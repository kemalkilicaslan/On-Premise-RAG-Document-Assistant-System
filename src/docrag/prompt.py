"""The system prompt, the fallback answer and context formatting.

Kept free of settings and database imports so the evaluation sends the LLM exactly what the API sends.
"""

from __future__ import annotations

from typing import Protocol

# The documents are Turkish, so the prompt and the fallback answer are Turkish too
NO_ANSWER = "Bu sorunun cevabı erişebildiğiniz dokümanlarda bulunmuyor."

SYSTEM_PROMPT = f"""Sen bir şirket içi doküman asistanısın. Yalnızca sana verilen BAĞLAM bölümündeki bilgilere dayanarak Türkçe cevap ver.

Kurallar:
- Bağlamda olmayan hiçbir bilgiyi uydurma. Cevap bağlamda yoksa yalnızca şunu yaz: "{NO_ANSWER}"
- Kullandığın her bilginin sonuna kaynağını köşeli parantezle yaz, örneğin [1] veya [2][3].
- Kısa ve net ol. Bağlamda madde numarası geçiyorsa cevapta belirt."""  # noqa: E501


class ContextChunk(Protocol):
    document_title: str
    page: int | None
    content: str


def build_context(chunks: list[ContextChunk]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk.document_title}" + (f", sayfa {chunk.page}" if chunk.page else "")
        parts.append(f"{header}\n{chunk.content}")
    return "\n\n".join(parts)


def build_messages(chunks: list[ContextChunk], question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"BAĞLAM:\n{build_context(chunks)}\n\nSORU: {question}"},
    ]
