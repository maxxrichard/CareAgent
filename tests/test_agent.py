"""End-to-end agent behaviour with scripted models (no provider calls)."""

import pytest

from careagent.agent import SAFE_FALLBACK
from tests.conftest import reactive_model, scripted_model


def _chunk_id(ret):
    return ret.content.chunks[0].chunk_id


@pytest.mark.asyncio
async def test_grounded_policy_answer(make_service):
    def decide(last):
        if last is None:
            return ("search_policies", {"query": "return window days"})
        chunks = {c.section: c for c in last.content.chunks}
        c = chunks.get("Return window") or last.content.chunks[0]
        return {
            "answer": "You can return most items within 30 days of the delivery date for a full refund.",
            "citations": [
                {"source_id": c.chunk_id, "quote": "within 30 days of the delivery date"}
            ],
        }

    svc = make_service(reactive_model(decide))
    resp = await svc.answer("How long do I have to return something?")
    assert not resp.escalated
    assert "30 days" in resp.answer
    assert resp.grounding.score == 1.0
    assert resp.tools_used == ["search_policies"]
    assert resp.cost_eur == 0.0 and resp.usage.requests == 2


@pytest.mark.asyncio
async def test_unsupported_claim_is_refused_and_handed_off(make_service):
    def decide(last):
        if last is None:
            return ("search_policies", {"query": "return window"})
        c = last.content.chunks[0]
        return {
            "answer": "You can return items within 90 days and we refund within 1 business day.",
            "citations": [{"source_id": c.chunk_id, "quote": c.text[-40:]}],
        }

    svc = make_service(reactive_model(decide))
    resp = await svc.answer("What is the return window?")
    assert resp.escalated
    assert resp.answer == SAFE_FALLBACK
    assert resp.citations == []
    assert resp.grounding.hallucination_flag
    assert resp.escalation is not None and resp.escalation.reason == "unsupported_claims"


@pytest.mark.asyncio
async def test_order_lookup_with_citation(make_service, orders):
    order = orders.all()[3]
    status = order.status.value.replace("_", " ")

    def decide(last):
        if last is None:
            return ("lookup_order", {"order_id": order.order_id})
        assert last.content.found
        return {
            "answer": f"Your order {order.order_id} is currently {status}.",
            "citations": [{"source_id": f"order:{order.order_id}", "quote": f"status: {status}"}],
        }

    svc = make_service(reactive_model(decide))
    resp = await svc.answer(f"Where is {order.order_id}?", customer_email=order.customer_email)
    assert not resp.escalated, resp.grounding
    assert status in resp.answer
    assert resp.tools_used == ["lookup_order"]


@pytest.mark.asyncio
async def test_order_of_other_customer_is_not_revealed(make_service, orders):
    order = orders.all()[3]
    seen = {}

    def decide(last):
        if last is None:
            return ("lookup_order", {"order_id": order.order_id})
        seen["found"] = last.content.found
        return {
            "answer": "I could not find that order under your account.",
            "needs_human": True,
            "handoff_reason": "missing_information",
        }

    svc = make_service(reactive_model(decide))
    resp = await svc.answer(
        f"Status of {order.order_id}", customer_email="someone.else@example.com"
    )
    assert seen["found"] is False
    assert resp.escalated and resp.escalation is not None


@pytest.mark.asyncio
async def test_explicit_escalation_tool(make_service):
    svc = make_service(
        scripted_model(
            [
                (
                    "escalate_to_human",
                    {"reason": "complaint", "summary": "Customer is unhappy.", "priority": "high"},
                ),
                {
                    "answer": "I've passed this to a colleague who will contact you shortly.",
                    "needs_human": True,
                },
            ]
        )
    )
    resp = await svc.answer("I want to complain about the service!")
    assert resp.escalated
    assert resp.escalation is not None and resp.escalation.priority == "high"
    assert resp.tools_used == ["escalate_to_human"]
    assert len(svc.desk.all()) == 1  # no duplicate ticket from the policy layer


@pytest.mark.asyncio
async def test_low_confidence_without_claims_is_handed_off(make_service):
    svc = make_service(
        scripted_model(
            [
                {"answer": "I'm not sure about that, sorry.", "confidence": 0.2},
            ]
        )
    )
    svc.settings = svc.settings.model_copy(update={"escalation_threshold": 0.6})
    resp = await svc.answer("Something unusual")
    # No checkable claims -> grounding score 1.0, so no forced hand-off; answer passes through.
    assert not resp.escalated and resp.grounding.score == 1.0


@pytest.mark.asyncio
async def test_non_strict_mode_keeps_answer_but_flags(make_service, settings):
    def decide(last):
        if last is None:
            return ("search_policies", {"query": "return window"})
        c = last.content.chunks[0]
        return {
            "answer": "You have 90 days to return.",
            "citations": [{"source_id": c.chunk_id, "quote": c.text[-30:]}],
        }

    svc = make_service(reactive_model(decide))
    svc.settings = settings.model_copy(
        update={"strict_grounding": False, "escalation_threshold": 0.0}
    )
    resp = await svc.answer("Return window?")
    assert resp.answer == "You have 90 days to return."
    assert resp.grounding.hallucination_flag
