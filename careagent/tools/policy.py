"""Policy retrieval tool (RAG)."""

from __future__ import annotations

from pydantic_ai import RunContext

from careagent.models import PolicyChunk, PolicySearchResult
from careagent.tools.deps import AgentDeps


def search_policies(ctx: RunContext[AgentDeps], query: str) -> PolicySearchResult:
    """Search the official return, refund, shipping, cancellation, payment and account
    policies. Returns the most relevant passages with their `chunk_id`, which you must
    cite verbatim when stating any rule, deadline, fee or timeframe.

    Args:
        query: A short natural-language description of the policy question.
    """
    ctx.deps.record("search_policies")
    chunks: list[PolicyChunk] = ctx.deps.retriever.retrieve(query)
    ctx.deps.evidence.add_chunks(chunks)
    return PolicySearchResult(query=query, chunks=chunks)
