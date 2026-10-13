"""Request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field

from careagent.models import Citation, EscalationTicket, GroundingReport, UsageInfo


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="The customer's message")
    session_id: str | None = Field(default=None, description="Opaque conversation id")
    customer_email: EmailStr | None = Field(
        default=None, description="Authenticated customer's email; scopes order lookups"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation]
    escalated: bool
    escalation: EscalationTicket | None
    grounding: GroundingReport
    tools_used: list[str]
    backend: str
    model: str
    latency_ms: float
    cost_eur: float
    usage: UsageInfo


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    model: str
    index_size: int
    orders: int
