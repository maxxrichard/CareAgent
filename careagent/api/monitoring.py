"""In-process metrics for the REST API.

Keeps a sliding window of recent requests so `/stats` can report live p50/p95 latency, cost
per query, hallucination rate and containment, and exposes the same counters in Prometheus
text format on `/metrics` for scraping.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from careagent.models import AgentResponse


@dataclass
class RequestRecord:
    ts: float
    latency_ms: float
    cost_eur: float
    hallucination: bool
    escalated: bool
    grounding_score: float
    backend: str
    model: str
    input_tokens: int
    output_tokens: int
    error: bool = False


@dataclass
class MetricsRegistry:
    window: int = 1000
    _records: deque[RequestRecord] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_requests: int = 0
    total_errors: int = 0
    total_cost_eur: float = 0.0
    total_hallucinations: int = 0
    total_escalations: int = 0
    started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._records = deque(maxlen=self.window)

    def record_response(self, resp: AgentResponse) -> None:
        self._add(
            RequestRecord(
                ts=time.time(),
                latency_ms=resp.latency_ms,
                cost_eur=resp.cost_eur,
                hallucination=resp.grounding.hallucination_flag,
                escalated=resp.escalated,
                grounding_score=resp.grounding.score,
                backend=resp.backend,
                model=resp.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        )

    def record_error(self, latency_ms: float, backend: str = "", model: str = "") -> None:
        self._add(
            RequestRecord(
                ts=time.time(),
                latency_ms=latency_ms,
                cost_eur=0.0,
                hallucination=False,
                escalated=False,
                grounding_score=0.0,
                backend=backend,
                model=model,
                input_tokens=0,
                output_tokens=0,
                error=True,
            )
        )

    def _add(self, rec: RequestRecord) -> None:
        with self._lock:
            self._records.append(rec)
            self.total_requests += 1
            self.total_cost_eur += rec.cost_eur
            self.total_errors += int(rec.error)
            self.total_hallucinations += int(rec.hallucination)
            self.total_escalations += int(rec.escalated)

    def snapshot(self) -> dict:
        with self._lock:
            recs = [r for r in self._records if not r.error]
            n = len(recs)
            lat = np.array([r.latency_ms for r in recs]) if n else np.array([0.0])
            return {
                "uptime_s": round(time.time() - self.started_at, 1),
                "window_size": n,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "latency_ms": {
                    "p50": round(float(np.percentile(lat, 50)), 1),
                    "p95": round(float(np.percentile(lat, 95)), 1),
                    "p99": round(float(np.percentile(lat, 99)), 1),
                    "mean": round(float(lat.mean()), 1),
                },
                "cost_eur": {
                    "per_query_mean": round(sum(r.cost_eur for r in recs) / n, 6) if n else 0.0,
                    "total": round(self.total_cost_eur, 4),
                },
                "hallucination_rate": round(sum(r.hallucination for r in recs) / n, 4)
                if n
                else 0.0,
                "containment_rate": round(sum(not r.escalated for r in recs) / n, 4) if n else 0.0,
                "mean_grounding_score": round(sum(r.grounding_score for r in recs) / n, 4)
                if n
                else 0.0,
                "tokens": {
                    "input_mean": round(sum(r.input_tokens for r in recs) / n, 1) if n else 0.0,
                    "output_mean": round(sum(r.output_tokens for r in recs) / n, 1) if n else 0.0,
                },
            }

    def prometheus(self) -> str:
        s = self.snapshot()
        lines = [
            "# HELP careagent_requests_total Total agent requests.",
            "# TYPE careagent_requests_total counter",
            f"careagent_requests_total {s['total_requests']}",
            "# TYPE careagent_errors_total counter",
            f"careagent_errors_total {s['total_errors']}",
            "# TYPE careagent_hallucinations_total counter",
            f"careagent_hallucinations_total {self.total_hallucinations}",
            "# TYPE careagent_escalations_total counter",
            f"careagent_escalations_total {self.total_escalations}",
            "# TYPE careagent_cost_eur_total counter",
            f"careagent_cost_eur_total {s['cost_eur']['total']}",
            "# TYPE careagent_latency_ms gauge",
            f'careagent_latency_ms{{quantile="0.5"}} {s["latency_ms"]["p50"]}',
            f'careagent_latency_ms{{quantile="0.95"}} {s["latency_ms"]["p95"]}',
            f'careagent_latency_ms{{quantile="0.99"}} {s["latency_ms"]["p99"]}',
            "# TYPE careagent_hallucination_rate gauge",
            f"careagent_hallucination_rate {s['hallucination_rate']}",
            "# TYPE careagent_containment_rate gauge",
            f"careagent_containment_rate {s['containment_rate']}",
            "# TYPE careagent_cost_eur_per_query gauge",
            f"careagent_cost_eur_per_query {s['cost_eur']['per_query_mean']}",
        ]
        return "\n".join(lines) + "\n"
