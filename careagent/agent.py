"""The CareAgent: a PydanticAI agent with typed tools, wrapped in a service that enforces
grounding and hand-off rules around every run."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models import Model

from careagent.backends import BackendSpec, build_backend, price_usd
from careagent.config import Settings, get_settings
from careagent.data.orders import OrderStore
from careagent.grounding.citation_check import CitationChecker, order_as_text
from careagent.models import AgentAnswer, AgentResponse, Citation, GroundingReport, UsageInfo
from careagent.prompts import INSTRUCTIONS
from careagent.rag.retriever import PolicyRetriever
from careagent.tools.deps import AgentDeps
from careagent.tools.escalation import EscalationDesk
from careagent.tools.handoff import escalate_to_human
from careagent.tools.orders import list_recent_orders, lookup_order
from careagent.tools.policy import search_policies

log = logging.getLogger(__name__)

SAFE_FALLBACK = (
    "I want to make sure you get accurate information, and I can't confirm every detail of this "
    "from our policies. I've passed your request to a colleague on our customer-service team, who "
    "will follow up with you shortly."
)


def build_agent(
    model: Model | None = None, backend: BackendSpec | None = None
) -> Agent[AgentDeps, AgentAnswer]:
    """Construct the PydanticAI agent. Pass `model` (e.g. a FunctionModel) for offline tests."""
    spec = backend if model is None else None
    agent: Agent[AgentDeps, AgentAnswer] = Agent(
        model or (spec.model if spec else None),
        deps_type=AgentDeps,
        output_type=AgentAnswer,
        instructions=INSTRUCTIONS,
        model_settings=spec.settings if spec else None,
        name="careagent",
        retries=2,
    )
    agent.tool(lookup_order)
    agent.tool(list_recent_orders)
    agent.tool(search_policies)
    agent.tool(escalate_to_human)
    return agent


@dataclass
class CareAgentService:
    """Runs the agent and applies the citation check + hand-off policy to its output."""

    agent: Agent[AgentDeps, AgentAnswer]
    orders: OrderStore
    retriever: PolicyRetriever
    desk: EscalationDesk
    settings: Settings
    backend_name: str
    model_name: str
    checker: CitationChecker

    @classmethod
    def create(
        cls,
        settings: Settings | None = None,
        *,
        backend: str | None = None,
        model: Model | None = None,
        orders: OrderStore | None = None,
        retriever: PolicyRetriever | None = None,
        lightweight_rag: bool = False,
    ) -> CareAgentService:
        settings = settings or get_settings()
        if model is not None:
            agent = build_agent(model=model)
            backend_name, model_name = "test", getattr(model, "model_name", "test-model")
        else:
            spec = build_backend(settings, backend)
            agent = build_agent(backend=spec)
            backend_name, model_name = spec.name, spec.model_name
        return cls(
            agent=agent,
            orders=orders or OrderStore.seeded(),
            retriever=retriever
            or PolicyRetriever.load_or_build(settings, lightweight=lightweight_rag),
            desk=EscalationDesk(),
            settings=settings,
            backend_name=backend_name,
            model_name=model_name,
            checker=CitationChecker(),
        )

    def _deps(self, session_id: str | None, customer_email: str | None) -> AgentDeps:
        return AgentDeps(
            orders=self.orders,
            retriever=self.retriever,
            desk=self.desk,
            session_id=session_id or uuid.uuid4().hex[:12],
            customer_email=customer_email,
        )

    async def answer(
        self, query: str, *, session_id: str | None = None, customer_email: str | None = None
    ) -> AgentResponse:
        deps = self._deps(session_id, customer_email)
        t0 = time.perf_counter()
        result = await self.agent.run(
            query,
            deps=deps,
            usage_limits=UsageLimits(request_limit=self.settings.max_tool_calls + 2),
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        output: AgentAnswer = result.output
        usage = result.usage

        report = self.checker.check(output, deps.evidence)
        answer, citations, escalated = self._apply_policy(output, report, deps)

        cost_usd = price_usd(
            self.model_name, usage.input_tokens, usage.output_tokens, usage.cache_read_tokens
        )
        ticket = self.desk.get(deps.escalation_ticket_id) if deps.escalation_ticket_id else None
        return AgentResponse(
            session_id=deps.session_id,
            query=query,
            answer=answer,
            citations=citations,
            escalated=escalated,
            escalation=ticket,
            grounding=report,
            tools_used=deps.tools_used,
            backend=self.backend_name,
            model=self.model_name,
            latency_ms=round(latency_ms, 1),
            cost_eur=round(cost_usd * self.settings.usd_to_eur, 6),
            usage=UsageInfo(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_tokens,
                requests=usage.requests,
                tool_calls=usage.tool_calls,
            ),
            raw={
                "model_output": output.model_dump(),
                "evidence_ids": list(deps.evidence.chunks) + list(deps.evidence.orders),
                "evidence_texts": [c.text for c in deps.evidence.chunks.values()]
                + [order_as_text(o) for o in deps.evidence.orders.values()],
            },
        )

    def answer_sync(self, query: str, **kwargs) -> AgentResponse:
        import asyncio

        return asyncio.run(self.answer(query, **kwargs))

    def _apply_policy(
        self, output: AgentAnswer, report: GroundingReport, deps: AgentDeps
    ) -> tuple[str, list[Citation], bool]:
        """Decide what the customer finally sees.

        * Model asked for a human -> make sure a ticket exists, keep its answer.
        * Grounding failed under strict mode -> refuse the unsupported answer, hand off.
        * Low grounding score -> keep the answer but hand off as a safety net.
        """
        escalated = output.needs_human or deps.escalation_ticket_id is not None
        answer, citations = output.answer, output.citations

        def ensure_ticket(reason: str, summary: str) -> None:
            if deps.escalation_ticket_id is None:
                t = self.desk.open_ticket(reason=reason, summary=summary)
                deps.escalation_ticket_id = t.ticket_id
                deps.record("escalate_to_human[auto]")

        if escalated:
            ensure_ticket(output.handoff_reason or "model_requested", output.answer[:300])

        if report.hallucination_flag and self.settings.strict_grounding:
            log.info(
                "refusing unsupported claims: %s | invalid=%s",
                report.unsupported_claims,
                report.invalid_citations,
            )
            ensure_ticket(
                "unsupported_claims",
                f"Agent produced unsupported claims for: {deps.session_id}. "
                f"Unsupported: {report.unsupported_claims[:3]}",
            )
            return SAFE_FALLBACK, [], True

        if report.score < self.settings.escalation_threshold:
            ensure_ticket("low_grounding_score", output.answer[:300])
            escalated = True

        return answer, citations, escalated
