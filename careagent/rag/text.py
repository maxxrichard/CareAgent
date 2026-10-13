"""Shared lexical helpers for the BM25 index and the overlap reranker."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "will",
        "you",
        "your",
        "we",
        "our",
        "it",
        "its",
        "this",
        "that",
        "with",
        "as",
        "at",
        "by",
        "from",
        "can",
        "could",
        "would",
        "should",
        "has",
        "have",
        "had",
        "was",
        "were",
        "been",
        "please",
        "i",
        "my",
        "me",
        "not",
        "if",
        "any",
        "do",
        "does",
        "did",
        "how",
        "many",
        "much",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "there",
        "here",
        "about",
        "into",
        "than",
        "then",
        "them",
        "they",
        "their",
        "also",
        "just",
        "get",
        "got",
    ]
)


def stem(token: str) -> str:
    """Very small suffix stripper; enough to align 'items'/'item', 'returns'/'return', 'shipped'/'ship'."""
    for suffix in ("ies", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]
