"""End-to-end policy retriever: chunk -> embed -> hybrid search -> rerank."""

from __future__ import annotations

from pathlib import Path

from careagent.config import Settings, get_settings
from careagent.data.policies import load_policy_documents
from careagent.models import PolicyChunk
from careagent.rag.chunking import chunk_documents
from careagent.rag.embeddings import Embedder, build_embedder
from careagent.rag.index import HybridIndex
from careagent.rag.reranker import Reranker, build_reranker


class PolicyRetriever:
    def __init__(
        self, index: HybridIndex, reranker: Reranker, top_k: int = 8, rerank_top_n: int = 4
    ):
        self.index = index
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

    # ---- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        lightweight: bool = False,
        policies_dir: Path | None = None,
    ) -> PolicyRetriever:
        """Build from the policy documents. `lightweight=True` forces the dependency-free
        hashing embedder + overlap reranker (used in tests and CI)."""
        settings = settings or get_settings()
        embedder = embedder or build_embedder(None if lightweight else settings.embedding_model)
        reranker = reranker or build_reranker(None if lightweight else settings.reranker_model)
        docs = load_policy_documents(policies_dir or settings.policies_dir)
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        return cls(HybridIndex(chunks, embedder), reranker, settings.top_k, settings.rerank_top_n)

    @classmethod
    def load_or_build(
        cls, settings: Settings | None = None, lightweight: bool = False
    ) -> PolicyRetriever:
        settings = settings or get_settings()
        embedder = build_embedder(None if lightweight else settings.embedding_model)
        reranker = build_reranker(None if lightweight else settings.reranker_model)
        index_dir = settings.index_dir
        if (index_dir / "meta.json").exists():
            try:
                index = HybridIndex.load(index_dir, embedder)
                return cls(index, reranker, settings.top_k, settings.rerank_top_n)
            except ValueError:
                pass
        retriever = cls.build(settings, embedder=embedder, reranker=reranker)
        retriever.index.save(index_dir)
        return retriever

    # ---- query --------------------------------------------------------------
    def retrieve(self, query: str, top_n: int | None = None) -> list[PolicyChunk]:
        candidates = self.index.search(query, k=self.top_k)
        return self.reranker.rerank(query, candidates, top_n or self.rerank_top_n)

    def get_chunk(self, chunk_id: str) -> PolicyChunk | None:
        return self.index.get(chunk_id)
