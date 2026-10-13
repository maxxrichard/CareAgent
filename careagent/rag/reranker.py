"""Rerankers: a cross-encoder in production, a lexical-overlap reranker as fallback."""

from __future__ import annotations

from typing import Protocol

from careagent.models import PolicyChunk
from careagent.rag.text import tokenize


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, chunks: list[PolicyChunk], top_n: int) -> list[PolicyChunk]: ...


class OverlapReranker:
    """Scores by weighted token overlap, with a bonus for matches in the section title."""

    name = "lexical-overlap"

    def rerank(self, query: str, chunks: list[PolicyChunk], top_n: int) -> list[PolicyChunk]:
        q = set(tokenize(query))
        scored = []
        for c in chunks:
            body = set(tokenize(c.text))
            title = set(tokenize(c.section or ""))
            overlap = len(q & body) / (len(q) or 1)
            title_bonus = 0.5 * len(q & title) / (len(q) or 1)
            scored.append(c.model_copy(update={"score": overlap + title_bonus + 0.05 * c.score}))
        return sorted(scored, key=lambda c: -c.score)[:top_n]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # lazy

        self.name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[PolicyChunk], top_n: int) -> list[PolicyChunk]:
        if not chunks:
            return []
        scores = self._model.predict([(query, c.text) for c in chunks])
        scored = [
            c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores, strict=True)
        ]
        return sorted(scored, key=lambda c: -c.score)[:top_n]


def build_reranker(model_name: str | None, allow_fallback: bool = True) -> Reranker:
    if model_name:
        try:
            return CrossEncoderReranker(model_name)
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "cross-encoder unavailable (%s); using overlap reranker", exc
            )
    return OverlapReranker()
