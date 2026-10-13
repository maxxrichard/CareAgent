"""Typed tools exposed to the agent: order lookup, policy retrieval, human escalation."""

from careagent.tools.deps import AgentDeps
from careagent.tools.escalation import EscalationDesk

__all__ = ["AgentDeps", "EscalationDesk"]
