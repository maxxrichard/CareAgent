import json

import pytest

from careagent.evaluation import dataset as ds
from careagent.evaluation.databricks import analytics_queries, write_results
from careagent.evaluation.metrics import aggregate, by_area, score_case
from careagent.evaluation.runner import run_evaluation
from careagent.models import AgentResponse, Citation, GroundingReport, UsageInfo
from tests.conftest import reactive_model


def _resp(answer, escalated=False, tools=(), flag=False, cost=0.001, latency=100.0):
    return AgentResponse(
        session_id="s",
        query="q",
        answer=answer,
        citations=[Citation(source_id="x", quote="y")],
        escalated=escalated,
        grounding=GroundingReport(score=0.0 if flag else 1.0, hallucination_flag=flag),
        tools_used=list(tools),
        backend="b",
        model="m",
        latency_ms=latency,
        cost_eur=cost,
        usage=UsageInfo(),
    )


def test_dataset_is_deterministic_and_stratified(orders):
    a = ds.generate_cases(120, seed=5, store=orders)
    b = ds.generate_cases(120, seed=5, store=orders)
    assert [c.case_id for c in a] == [c.case_id for c in b]
    s = ds.stats(a)
    assert s["n"] == 120 and s["by_outcome"]["escalate"] > 0 and s["by_area"]["order_status"] > 0


def test_committed_dataset_has_500_cases():
    cases = ds.load_cases()
    assert len(cases) == 500
    assert len({c.case_id for c in cases}) == 500
    assert all(
        c.order_id for c in cases if c.area == "order_status" and c.case_id.startswith("order-")
    )


def test_score_case_rules():
    case = ds.EvalCase(
        case_id="c",
        query="q",
        area="returns",
        expected_outcome="answer",
        expected_tools=["search_policies"],
        required_facts=["30 days"],
        forbidden_facts=["14 days"],
        gold_chunk_prefix="return_policy#0",
    )
    good = score_case(
        case,
        _resp("Within 30 days.", tools=["search_policies"]),
        retrieved_ids=["return_policy#0.0"],
    )
    assert good.task_success and good.retrieval_hit
    wrong = score_case(case, _resp("Within 14 days.", tools=["search_policies"]))
    assert not wrong.task_success and wrong.forbidden_present and wrong.hallucination
    escalated = score_case(case, _resp("Handing off", escalated=True, tools=["search_policies"]))
    assert not escalated.task_success
    judged = score_case(
        case, _resp("Within 30 days.", tools=["search_policies"]), judge_hallucination=True
    )
    assert not judged.task_success and judged.hallucination

    esc_case = case.model_copy(
        update={"expected_outcome": "escalate", "expected_tools": ["escalate_to_human"]}
    )
    assert score_case(esc_case, _resp("…", escalated=True)).task_success


def test_aggregate_and_by_area():
    case_a = ds.EvalCase(case_id="a", query="q", area="returns", expected_outcome="answer")
    case_e = ds.EvalCase(case_id="e", query="q", area="escalation", expected_outcome="escalate")
    results = [
        score_case(case_a, _resp("ok", latency=100)),
        score_case(case_a, _resp("bad", flag=True, latency=300)),
        score_case(case_e, _resp("x", escalated=True, latency=200)),
    ]
    agg = aggregate(results)
    assert agg["n_cases"] == 3
    assert agg["task_success_rate"] == pytest.approx(2 / 3)
    assert agg["hallucination_rate"] == pytest.approx(0.5)
    assert agg["containment_rate"] == pytest.approx(2 / 3)
    assert agg["escalation_precision"] == 1.0 and agg["escalation_recall"] == 1.0
    assert agg["latency_p95_ms"] > agg["latency_p50_ms"]
    assert set(by_area(results)) == {"returns", "escalation"}


def test_write_results_local_fallback(tmp_path):
    loc = write_results([{"a": 1}, {"a": 2}], "main.x.y", tmp_path, "run1")
    with open(loc, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    assert rows[0]["run_id"] == "run1" and len(rows) == 2


def test_analytics_queries_substitute_table():
    q = analytics_queries("main.careagent.eval_results")
    assert {"backend_comparison", "success_by_area", "worst_cases"} <= set(q)
    assert all("${table}" not in v and "main.careagent.eval_results" in v for v in q.values())


def test_run_evaluation_end_to_end(tmp_path, make_service, monkeypatch):
    """Runs the harness on a slice of the committed dataset with a scripted model."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("careagent.evaluation.tracking._mlflow", lambda: None)

    def decide(last):
        if last is None:
            return ("escalate_to_human", {"reason": "low_confidence", "summary": "scripted"})
        return {"answer": "A colleague will follow up.", "needs_human": True}

    svc = make_service(reactive_model(decide))
    summary = run_evaluation(
        ["scripted"],
        limit=12,
        use_judge=False,
        lightweight_rag=True,
        output_dir=tmp_path,
        services={"scripted": svc},
    )
    m = summary["backends"]["scripted"]["metrics"]
    assert m["n_cases"] == 12 and m["n_errors"] == 0
    assert m["containment_rate"] == 0.0 and m["hallucination_rate"] == 0.0
    assert (tmp_path / f"summary-{summary['run_group']}-scripted.json").exists()
