"""Section-aware chunking of markdown policy documents."""

from __future__ import annotations

import re

from careagent.data.policies import PolicyDocument
from careagent.models import PolicyChunk

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on markdown headings; returns (section_title, body) pairs."""
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(None, text.strip())]
    sections: list[tuple[str | None, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            sections.append((m.group(2).strip(), body))
    return sections


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Split long text into overlapping windows on sentence boundaries where possible."""
    if len(text) <= size:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    windows: list[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > size and buf:
            windows.append(buf.strip())
            buf = buf[-overlap:] if overlap else ""
        buf = f"{buf} {s}".strip()
    if buf:
        windows.append(buf.strip())
    return windows


def chunk_documents(
    docs: list[PolicyDocument], chunk_size: int = 600, overlap: int = 80
) -> list[PolicyChunk]:
    chunks: list[PolicyChunk] = []
    for doc in docs:
        for s_idx, (section, body) in enumerate(_split_sections(doc.text)):
            for w_idx, window in enumerate(_window(body, chunk_size, overlap)):
                # Prefix with doc/section title so the embedding carries context.
                prefix = f"{doc.title}" + (f" — {section}" if section else "")
                chunks.append(
                    PolicyChunk(
                        chunk_id=f"{doc.doc_id}#{s_idx}.{w_idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        section=section,
                        text=f"{prefix}\n{window}",
                    )
                )
    return chunks
