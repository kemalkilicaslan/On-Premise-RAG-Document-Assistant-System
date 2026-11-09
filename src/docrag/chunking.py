"""Text normalization and chunking.

Chunks never cross page boundaries, so every answer can cite an exact page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class TextChunk:
    page: int
    index: int
    content: str


def normalize(text: str) -> str:
    text = re.sub(r"-\n(?=\w)", "", text)  # re-join words hyphenated across lines
    # Some PDF fonts emit a space before "ğ" ("de ğer"); no Turkish word starts with ğ, so it is safe to drop
    text = re.sub(r"(?<=\w) (?=ğ)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Pack sentences/lines into chunks of about `size` characters.

    Consecutive chunks share up to `overlap` trailing characters, so a chunk can be at
    most size + overlap long. Pieces longer than `size` are cut hard.
    """
    units: list[str] = []
    for unit in re.split(r"\n+|(?<=[.!?;:])\s+", text):
        unit = unit.strip()
        units.extend(unit[i : i + size] for i in range(0, len(unit), size))

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail.split(" ", 1)[-1] if " " in tail else tail  # start on a word boundary
        current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page_number, text in enumerate(pages, start=1):
        for content in split_text(text, size, overlap):
            chunks.append(TextChunk(page=page_number, index=len(chunks), content=content))
    return chunks
