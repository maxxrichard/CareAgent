"""Embedding backends.

`SentenceTransformerEmbedder` is used in production. `HashingEmbedder` is a dependency-free
fallback (feature-hashed token/bigram vectors) so the pipeline and tests run without model
downloads; it is noticeably weaker but deterministic.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing-bigram"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            toks = _TOKEN.findall(text.lower())
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
            for f in feats:
                h = int(hashlib.blake2b(f.encode(), digest_size=8).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 63) else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def build_embedder(model_name: str | None, allow_fallback: bool = True) -> Embedder:
    if model_name:
        try:
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            import logging

            logging.getLogger(__name__).warning(
                "sentence-transformers unavailable (%s); using hashing embedder", exc
            )
    return HashingEmbedder()
