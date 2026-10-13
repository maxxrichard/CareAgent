"""Human hand-off: creates a ticket in an (in-memory, pluggable) escalation queue."""

from __future__ import annotations

import itertools
import threading
from typing import Literal

from careagent.models import EscalationTicket


class EscalationDesk:
    """Minimal ticket queue. Swap for a Zendesk/Freshdesk client in production."""

    def __init__(self) -> None:
        self._tickets: dict[str, EscalationTicket] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def open_ticket(
        self, reason: str, summary: str, priority: Literal["low", "normal", "high"] = "normal"
    ) -> EscalationTicket:
        with self._lock:
            ticket = EscalationTicket(
                ticket_id=f"TCK-{next(self._counter):06d}",
                reason=reason,
                summary=summary,
                priority=priority,
            )
            self._tickets[ticket.ticket_id] = ticket
        return ticket

    def get(self, ticket_id: str) -> EscalationTicket | None:
        return self._tickets.get(ticket_id)

    def all(self) -> list[EscalationTicket]:
        return list(self._tickets.values())
