"""Per-case scoring and aggregate metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from careagent.evaluation.dataset import EvalCase
from careagent.models import AgentResponse


@dataclass
class CaseResult:
    case_id: str
    area: str
    backend: str
    model: str
    query: str
    answer: str
    expected_outcome: str
    escalated: bool
    tools_used: list[str]
    expected_tools: list[str]
    task_success: bool
    tool_trajectory_ok: bool
    facts_present: bool
    forbidden_present: bool
    retrieval_hit: bool | None
    citation_flag: bool
    judge_hallucination: bool | None
    grounding_score: float
    latency_ms: float
    cost_eur: float
    input_tokens: int
    output_tokens: int
    error: str | None = None

    @property
    def hallucination(self) -> bool:
        """Gold hallucination label: judge verdict when available, else the citation check."""
        if self.judge_hallucination is not None:
            return self.judge_hallucination
        return self.citation_flag or self.forbidden_present

    def to_row(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["hallucination"] = self.hallucination
        d["tools_used"] = ",".join(self.tools_used)
        d["expected_tools"] = ",".join(self.expected_tools)
        return d


def score_case(
    case: EvalCase,
    resp: AgentResponse,
    judge_hallucination: bool | None = None,
    retrieved_ids: list[str] | None = None,
) -> CaseResult:
    answer_l = resp.answer.lower()
    facts_present = all(f.lower() in answer_l for f in case.required_facts)
    forbidden_present = any(f.lower() in answer_l for f in case.forbidden_facts)
    tools_ok = all(t in resp.tools_used for t in case.expected_tools)

    retrieval_hit: bool | None = None
    if case.gold_chunk_prefix and retrieved_ids is not None:
        retrieval_hit = any(r.startswith(case.gold_chunk_prefix) for r in retrieved_ids)

    hallucinated = (
        judge_hallucination
        if judge_hallucination is not None
        else (resp.grounding.hallucination_flag or forbidden_present)
    )
    if case.expected_outcome == "escalate":
        success = resp.escalated
    else:
        success = (
            (not resp.escalated)
            and facts_present
            and not forbidden_present
            and tools_ok
            and not hallucinated
        )

    return CaseResult(
        case_id=case.case_id,
        area=case.area,
        backend=resp.backend,
        model=resp.model,
        query=case.query,
        answer=resp.answer,
        expected_outcome=case.expected_outcome,
        escalated=resp.escalated,
        tools_used=resp.tools_used,
        expected_tools=case.expected_tools,
        task_success=success,
        tool_trajectory_ok=tools_ok,
        facts_present=facts_present,
        forbidden_present=forbidden_present,
        retrieval_hit=retrieval_hit,
        citation_flag=resp.grounding.hallucination_flag,
        judge_hallucination=judge_hallucination,
        grounding_score=resp.grounding.score,
        latency_ms=resp.latency_ms,
        cost_eur=resp.cost_eur,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )


def error_case(
    case: EvalCase, backend: str, model: str, error: str, latency_ms: float
) -> CaseResult:
    return CaseResult(
        case_id=case.case_id,
        area=case.area,
        backend=backend,
        model=model,
        query=case.query,
        answer="",
        expected_outcome=case.expected_outcome,
        escalated=False,
        tools_used=[],
        expected_tools=case.expected_tools,
        task_success=False,
        tool_trajectory_ok=False,
        facts_present=False,
        forbidden_present=False,
        retrieval_hit=None,
        citation_flag=False,
        judge_hallucination=None,
        grounding_score=0.0,
        latency_ms=latency_ms,
        cost_eur=0.0,
        input_tokens=0,
        output_tokens=0,
        error=error,
    )


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    n = len(results)
    if n == 0:
        return {}
    ok = [r for r in results if r.error is None]
    lat = np.array([r.latency_ms for r in ok]) if ok else np.array([0.0])
    answerable = [r for r in ok if r.expected_outcome == "answer"]
    retrieval = [r for r in ok if r.retrieval_hit is not None]
    return {
        "n_cases": n,
        "n_errors": n - len(ok),
        "task_success_rate": sum(r.task_success for r in results) / n,
        "containment_rate": sum(not r.escalated for r in ok) / max(len(ok), 1),
        "hallucination_rate": sum(r.hallucination for r in answerable) / max(len(answerable), 1),
        "citation_flag_rate": sum(r.citation_flag for r in answerable) / max(len(answerable), 1),
        "tool_trajectory_accuracy": sum(r.tool_trajectory_ok for r in ok) / max(len(ok), 1),
        "retrieval_recall": sum(bool(r.retrieval_hit) for r in retrieval) / max(len(retrieval), 1),
        "escalation_precision": _escalation_precision(ok),
        "escalation_recall": _escalation_recall(ok),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_mean_ms": float(lat.mean()),
        "cost_eur_per_query": sum(r.cost_eur for r in ok) / max(len(ok), 1),
        "cost_eur_total": sum(r.cost_eur for r in ok),
        "mean_grounding_score": sum(r.grounding_score for r in ok) / max(len(ok), 1),
        "input_tokens_mean": sum(r.input_tokens for r in ok) / max(len(ok), 1),
        "output_tokens_mean": sum(r.output_tokens for r in ok) / max(len(ok), 1),
    }


def _escalation_precision(results: list[CaseResult]) -> float:
    escalated = [r for r in results if r.escalated]
    return sum(r.expected_outcome == "escalate" for r in escalated) / max(len(escalated), 1)


def _escalation_recall(results: list[CaseResult]) -> float:
    should = [r for r in results if r.expected_outcome == "escalate"]
    return sum(r.escalated for r in should) / max(len(should), 1)


def by_area(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    areas = sorted({r.area for r in results})
    return {a: aggregate([r for r in results if r.area == a]) for a in areas}
