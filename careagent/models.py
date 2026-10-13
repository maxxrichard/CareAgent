"""Typed domain objects shared by tools, the agent, the API and the evaluation harness."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- orders
class OrderStatus(str, Enum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    RETURN_REQUESTED = "return_requested"
    REFUNDED = "refunded"


class OrderItem(BaseModel):
    sku: str
    name: str
    category: Literal["electronics", "clothing", "home", "beauty", "books", "perishable"]
    quantity: int = Field(ge=1)
    unit_price_eur: float = Field(ge=0)


class Order(BaseModel):
    order_id: str = Field(description="Public order number, e.g. ORD-100042")
    customer_email: str
    status: OrderStatus
    placed_on: date
    estimated_delivery: date | None = None
    delivered_on: date | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    items: list[OrderItem]
    total_eur: float

    @property
    def is_returnable_window_open(self) -> bool:
        if self.delivered_on is None:
            return False
        return (date.today() - self.delivered_on).days <= 30


class OrderLookupResult(BaseModel):
    """Result of the order-status tool. `found=False` means no matching order."""

    found: bool
    order: Order | None = None
    message: str = ""


# ------------------------------------------------------------------------ retrieval
class PolicyChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    section: str | None = None
    score: float = 0.0


class PolicySearchResult(BaseModel):
    query: str
    chunks: list[PolicyChunk]


# ----------------------------------------------------------------------- escalation
class EscalationTicket(BaseModel):
    ticket_id: str
    reason: str
    summary: str
    priority: Literal["low", "normal", "high"] = "normal"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -------------------------------------------------------------------- agent output
class Citation(BaseModel):
    """A pointer from a claim in the answer to the evidence that supports it."""

    source_id: str = Field(
        description="chunk_id of a retrieved policy chunk, or 'order:<order_id>' for order facts"
    )
    quote: str = Field(description="Short verbatim excerpt from the source that supports the claim")


class AgentAnswer(BaseModel):
    """Structured output the LLM must produce."""

    answer: str = Field(description="Customer-facing reply, concise and friendly")
    citations: list[Citation] = Field(
        default_factory=list,
        description="One citation per factual claim about policies, dates, fees or order details",
    )
    needs_human: bool = Field(
        default=False,
        description="True when the request cannot be resolved with the available tools/evidence",
    )
    handoff_reason: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class GroundingReport(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Share of checkable claims that are supported")
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    hallucination_flag: bool = False


class UsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    requests: int = 0
    tool_calls: int = 0


class AgentResponse(BaseModel):
    """What the service returns to callers: the grounded answer plus observability data."""

    session_id: str
    query: str
    answer: str
    citations: list[Citation]
    escalated: bool
    escalation: EscalationTicket | None = None
    grounding: GroundingReport
    tools_used: list[str]
    backend: str
    model: str
    latency_ms: float
    cost_eur: float
    usage: UsageInfo
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)
