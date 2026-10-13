"""Dependencies injected into every agent run (PydanticAI `deps`)."""

from __future__ import annotations

from dataclasses import dataclass, field

from careagent.data.orders import OrderStore
from careagent.grounding.citation_check import EvidenceStore
from careagent.rag.retriever import PolicyRetriever
from careagent.tools.escalation import EscalationDesk


@dataclass
class AgentDeps:
    orders: OrderStore
    retriever: PolicyRetriever
    desk: EscalationDesk
    session_id: str = "anonymous"
    customer_email: str | None = None
    evidence: EvidenceStore = field(default_factory=EvidenceStore)
    tools_used: list[str] = field(default_factory=list)
    escalation_ticket_id: str | None = None

    def record(self, tool_name: str) -> None:
        self.tools_used.append(tool_name)
