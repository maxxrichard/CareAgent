import pytest
from fastapi.testclient import TestClient

from careagent.api.app import create_app
from careagent.api.monitoring import MetricsRegistry
from tests.conftest import reactive_model


@pytest.fixture()
def client(make_service):
    def decide(last):
        if last is None:
            return ("search_policies", {"query": "return window"})
        chunks = {c.section: c for c in last.content.chunks}
        c = chunks.get("Return window") or last.content.chunks[0]
        return {
            "answer": "You can return most items within 30 days of the delivery date for a full refund.",
            "citations": [
                {"source_id": c.chunk_id, "quote": "within 30 days of the delivery date"}
            ],
        }

    app = create_app(service=make_service(reactive_model(decide)))
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["index_size"] > 0 and body["orders"] > 0


def test_chat_and_metrics(client):
    r = client.post("/chat", json={"message": "How long can I return?", "session_id": "s1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == "s1" and "30 days" in body["answer"] and not body["escalated"]
    assert body["citations"][0]["source_id"].startswith("return_policy#")

    stats = client.get("/stats").json()
    assert stats["total_requests"] == 1 and stats["hallucination_rate"] == 0.0
    assert stats["containment_rate"] == 1.0 and stats["latency_ms"]["p95"] >= 0

    prom = client.get("/metrics").text
    assert "careagent_requests_total 1" in prom and "careagent_hallucination_rate 0.0" in prom


def test_validation_errors(client):
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert (
        client.post("/chat", json={"message": "hi", "customer_email": "not-an-email"}).status_code
        == 422
    )


def test_metrics_registry_window():
    from careagent.models import AgentResponse, GroundingReport, UsageInfo

    reg = MetricsRegistry(window=3)
    for i in range(5):
        reg.record_response(
            AgentResponse(
                session_id="s",
                query="q",
                answer="a",
                citations=[],
                escalated=i % 2 == 0,
                grounding=GroundingReport(score=1.0, hallucination_flag=i == 4),
                tools_used=[],
                backend="b",
                model="m",
                latency_ms=100 + i,
                cost_eur=0.001,
                usage=UsageInfo(),
            )
        )
    reg.record_error(50)
    s = reg.snapshot()
    assert s["total_requests"] == 6 and s["total_errors"] == 1 and s["window_size"] == 2
    assert s["cost_eur"]["total"] == 0.005
