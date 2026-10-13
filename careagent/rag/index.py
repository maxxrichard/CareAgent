"""Hybrid index: dense cosine search over numpy vectors + BM25, fused with reciprocal rank fusion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from careagent.models import PolicyChunk
from careagent.rag.embeddings import Embedder
from careagent.rag.text import tokenize as _tokenize


class HybridIndex:
    def __init__(
        self, chunks: list[PolicyChunk], embedder: Embedder, vectors: np.ndarray | None = None
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = vectors if vectors is not None else embedder.encode([c.text for c in chunks])
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])
        self._by_id = {c.chunk_id: c for c in chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> PolicyChunk | None:
        return self._by_id.get(chunk_id)

    # ---- search -------------------------------------------------------------
    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.embedder.encode([query])[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 8, rrf_k: int = 60) -> list[PolicyChunk]:
        """Reciprocal-rank fusion of dense and lexical rankings."""
        fused: dict[int, float] = {}
        for ranking in (self.dense_search(query, k * 2), self.bm25_search(query, k * 2)):
            for rank, (idx, _) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [self.chunks[i].model_copy(update={"score": s}) for i, s in ordered]

    # ---- persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(c.model_dump_json() + "\n")
        (directory / "meta.json").write_text(
            json.dumps(
                {"embedder": self.embedder.name, "dim": self.embedder.dim, "n": len(self.chunks)}
            )
        )

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> HybridIndex:
        meta = json.loads((directory / "meta.json").read_text())
        if meta.get("embedder") != embedder.name:
            raise ValueError(
                f"Index built with {meta.get('embedder')!r} but embedder is {embedder.name!r}; rebuild"
            )
        chunks = [
            PolicyChunk.model_validate_json(line)
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(chunks, embedder, vectors=np.load(directory / "vectors.npy"))
