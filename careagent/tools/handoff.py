"""Escalation tool: hand the conversation to a human agent."""

from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from careagent.models import EscalationTicket
from careagent.tools.deps import AgentDeps


def escalate_to_human(
    ctx: RunContext[AgentDeps],
    reason: Literal[
        "missing_information",
        "out_of_scope",
        "policy_exception",
        "customer_request",
        "complaint",
        "damaged_or_missing_parcel",
        "payment_dispute",
        "low_confidence",
    ],
    summary: str,
    priority: Literal["low", "normal", "high"] = "normal",
) -> EscalationTicket:
    """Hand the conversation to a human agent by opening a ticket. Use this when the request
    cannot be resolved with the available tools and policies, when the customer explicitly
    asks for a person, for complaints, or when you are not confident in the answer.

    Args:
        reason: Why a human is needed.
        summary: A two-sentence summary of the customer's situation for the human agent.
        priority: "high" for damaged/missing parcels, payment disputes or angry customers.
    """
    ctx.deps.record("escalate_to_human")
    ticket = ctx.deps.desk.open_ticket(reason=reason, summary=summary, priority=priority)
    ctx.deps.escalation_ticket_id = ticket.ticket_id
    return ticket
