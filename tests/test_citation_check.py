from careagent.grounding.citation_check import CitationChecker, EvidenceStore, order_as_text
from careagent.models import AgentAnswer, Citation, PolicyChunk

CHUNK = PolicyChunk(
    chunk_id="return_policy#0.0",
    doc_id="return_policy",
    title="Return & Refund Policy",
    section="Return window",
    text="Return & Refund Policy — Return window\nYou can return most items within 30 days of the "
    "delivery date for a full refund. The return window starts on the day the carrier marks "
    "the parcel as delivered, not on the order date.",
)


def _store():
    ev = EvidenceStore()
    ev.add_chunks([CHUNK])
    return ev


def test_supported_answer_passes():
    ans = AgentAnswer(
        answer="You can return most items within 30 days of the delivery date for a full refund.",
        citations=[
            Citation(source_id="return_policy#0.0", quote="within 30 days of the delivery date")
        ],
    )
    rep = CitationChecker().check(ans, _store())
    assert rep.score == 1.0 and not rep.hallucination_flag and not rep.invalid_citations


def test_wrong_number_is_flagged():
    ans = AgentAnswer(
        answer="You can return most items within 14 days of the delivery date for a full refund.",
        citations=[
            Citation(source_id="return_policy#0.0", quote="within 30 days of the delivery date")
        ],
    )
    rep = CitationChecker().check(ans, _store())
    assert rep.hallucination_flag and rep.unsupported_claims


def test_invented_source_is_flagged():
    ans = AgentAnswer(
        answer="Returns are free within 30 days.",
        citations=[Citation(source_id="return_policy#99.0", quote="free")],
    )
    rep = CitationChecker().check(ans, _store())
    assert rep.invalid_citations and rep.hallucination_flag


def test_invented_quote_is_flagged():
    ans = AgentAnswer(
        answer="You can return items within 30 days.",
        citations=[
            Citation(source_id="return_policy#0.0", quote="returns are always free of charge")
        ],
    )
    rep = CitationChecker().check(ans, _store())
    assert rep.invalid_citations


def test_fuzzy_quote_tolerates_minor_differences():
    c = CitationChecker()
    assert c.quote_in_source("within 30 days of the delivery date for a full refund", CHUNK.text)
    assert c.quote_in_source("Within 30 days  of the delivery-date for a full refund", CHUNK.text)
    assert not c.quote_in_source("within 60 days of purchase", CHUNK.text)


def test_non_factual_answer_without_citations_is_fine():
    ans = AgentAnswer(
        answer="I'm sorry to hear that. I've passed this to a colleague who will follow up."
    )
    rep = CitationChecker().check(ans, EvidenceStore())
    assert rep.score == 1.0 and not rep.hallucination_flag


def test_order_facts_can_be_cited(orders):
    order = orders.all()[0]
    ev = EvidenceStore()
    ev.add_order(order)
    status = order.status.value.replace("_", " ")
    ans = AgentAnswer(
        answer=f"Your order {order.order_id} is currently {status}.",
        citations=[Citation(source_id=f"order:{order.order_id}", quote=f"status: {status}")],
    )
    rep = CitationChecker().check(ans, ev)
    assert not rep.hallucination_flag, rep
    assert order.order_id in order_as_text(order)
