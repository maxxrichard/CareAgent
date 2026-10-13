"""Run the labelled query set against one or more backends and log everything."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from careagent.agent import CareAgentService
from careagent.config import get_settings
from careagent.data.orders import OrderStore
from careagent.evaluation.databricks import write_results
from careagent.evaluation.dataset import DEFAULT_PATH, EvalCase, load_cases, stats
from careagent.evaluation.judge import build_judge, judge_answer
from careagent.evaluation.metrics import CaseResult, aggregate, by_area, error_case, score_case
from careagent.evaluation.tracking import tracked_run
from careagent.rag.retriever import PolicyRetriever

log = logging.getLogger(__name__)


async def evaluate_backend(
    service: CareAgentService,
    cases: list[EvalCase],
    *,
    judge=None,
    concurrency: int = 4,
    on_progress=None,
) -> list[CaseResult]:
    sem = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await service.answer(
                    case.query,
                    session_id=f"eval-{case.case_id}",
                    customer_email=case.customer_email,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("case %s failed: %s", case.case_id, exc)
                return error_case(
                    case,
                    service.backend_name,
                    service.model_name,
                    f"{type(exc).__name__}: {exc}",
                    (time.perf_counter() - t0) * 1000,
                )
            evidence_ids = _evidence_ids(service, resp)
            verdict = None
            if judge is not None and case.expected_outcome == "answer" and not resp.escalated:
                try:
                    v = await judge_answer(
                        judge, case.query, resp.answer, _evidence_texts(service, resp)
                    )
                    verdict = v.hallucinated
                except Exception as exc:  # noqa: BLE001
                    log.warning("judge failed for %s: %s", case.case_id, exc)
            result = score_case(case, resp, judge_hallucination=verdict, retrieved_ids=evidence_ids)
            if on_progress:
                on_progress(result)
            return result

    return await asyncio.gather(*(run_one(c) for c in cases))


def _evidence_ids(service: CareAgentService, resp) -> list[str]:
    """Ids of every chunk/order the agent actually saw during the run."""
    return list(resp.raw.get("evidence_ids", []))


def _evidence_texts(service: CareAgentService, resp) -> list[str]:
    return list(resp.raw.get("evidence_texts", []))


def run_evaluation(
    backends: list[str],
    *,
    cases_path: Path = DEFAULT_PATH,
    limit: int | None = None,
    use_judge: bool = True,
    judge_model=None,
    lightweight_rag: bool = False,
    output_dir: str | Path = "data/eval/results",
    concurrency: int = 4,
    services: dict[str, CareAgentService] | None = None,
) -> dict[str, Any]:
    """Evaluate each backend on the labelled set; returns the per-backend summary.

    `services` lets tests inject pre-built services (e.g. with a FunctionModel) keyed by
    backend name, bypassing real providers.
    """
    settings = get_settings()
    cases = load_cases(cases_path, limit=limit)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_group = uuid.uuid4().hex[:8]
    retriever = PolicyRetriever.load_or_build(settings, lightweight=lightweight_rag)
    orders = OrderStore.seeded()
    judge = build_judge(judge_model) if use_judge else None
    summary: dict[str, Any] = {"run_group": run_group, "dataset": stats(cases), "backends": {}}

    for backend in backends:
        service = (services or {}).get(backend) or CareAgentService.create(
            settings, backend=backend, orders=orders, retriever=retriever
        )
        run_id = f"{run_group}-{backend}"
        params = {
            "backend": service.backend_name,
            "model": service.model_name,
            "n_cases": len(cases),
            "effort": settings.effort,
            "strict_grounding": settings.strict_grounding,
            "embedding_model": service.retriever.index.embedder.name,
            "reranker": service.retriever.reranker.name,
            "top_k": settings.top_k,
            "rerank_top_n": settings.rerank_top_n,
            "judge": "llm" if judge else "citation_check",
        }
        with tracked_run(
            run_name=run_id,
            experiment=settings.mlflow_experiment,
            params=params,
            tags={"run_group": run_group},
        ) as log_metrics:
            t0 = time.perf_counter()
            results = asyncio.run(
                evaluate_backend(service, cases, judge=judge, concurrency=concurrency)
            )
            wall = time.perf_counter() - t0
            agg = aggregate(results)
            agg["wall_time_s"] = wall
            area = by_area(results)
            rows = [r.to_row() for r in results]
            location = write_results(rows, settings.delta_table, output_dir, run_id)
            summary_path = output_dir / f"summary-{run_id}.json"
            summary_path.write_text(
                json.dumps(
                    {"params": params, "metrics": agg, "by_area": area}, indent=2, default=str
                )
            )
            log_metrics(
                agg,
                artifacts=[summary_path] + ([Path(location)] if Path(location).exists() else []),
            )
            for a, m in area.items():
                log_metrics(
                    {
                        f"{a}/{k}": v
                        for k, v in m.items()
                        if k in {"task_success_rate", "hallucination_rate", "containment_rate"}
                    }
                )
        summary["backends"][backend] = {"params": params, "metrics": agg, "results": location}
        log.info(
            "%s: success=%.3f containment=%.3f halluc=%.3f p95=%.0fms cost=€%.4f",
            backend,
            agg["task_success_rate"],
            agg["containment_rate"],
            agg["hallucination_rate"],
            agg["latency_p95_ms"],
            agg["cost_eur_per_query"],
        )
    return summary
