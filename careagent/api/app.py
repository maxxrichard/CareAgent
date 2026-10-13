"""FastAPI application.

Endpoints
---------
POST /chat      – answer a customer message (grounded, with citations and hand-off)
GET  /health    – liveness + configuration summary
GET  /stats     – live latency / cost / hallucination / containment metrics (JSON)
GET  /metrics   – the same in Prometheus text format
GET  /tickets   – escalation tickets opened in this process
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

from careagent import __version__
from careagent.agent import CareAgentService
from careagent.api.monitoring import MetricsRegistry
from careagent.api.schemas import ChatRequest, ChatResponse, HealthResponse
from careagent.config import get_settings
from careagent.models import EscalationTicket

log = logging.getLogger("careagent.api")


def create_app(service: CareAgentService | None = None) -> FastAPI:
    metrics = MetricsRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or CareAgentService.create(get_settings())
        app.state.metrics = metrics
        _maybe_enable_tracing()
        log.info(
            "CareAgent ready: backend=%s model=%s",
            app.state.service.backend_name,
            app.state.service.model_name,
        )
        yield

    app = FastAPI(
        title="CareAgent",
        version=__version__,
        description="Agentic customer self-service assistant with grounded answers and human hand-off.",
        lifespan=lifespan,
    )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest, request: Request) -> ChatResponse:
        svc: CareAgentService = request.app.state.service
        t0 = time.perf_counter()
        try:
            resp = await svc.answer(
                req.message, session_id=req.session_id, customer_email=req.customer_email
            )
        except Exception as exc:  # noqa: BLE001
            metrics.record_error(
                (time.perf_counter() - t0) * 1000, svc.backend_name, svc.model_name
            )
            log.exception("agent run failed")
            raise HTTPException(
                status_code=502, detail=f"agent error: {type(exc).__name__}"
            ) from exc
        metrics.record_response(resp)
        return ChatResponse(**resp.model_dump(exclude={"query", "raw"}))

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        svc: CareAgentService = request.app.state.service
        return HealthResponse(
            status="ok",
            version=__version__,
            backend=svc.backend_name,
            model=svc.model_name,
            index_size=len(svc.retriever.index),
            orders=svc.orders.count(),
        )

    @app.get("/stats")
    async def stats() -> dict:
        return metrics.snapshot()

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus() -> str:
        return metrics.prometheus()

    @app.get("/tickets", response_model=list[EscalationTicket])
    async def tickets(request: Request) -> list[EscalationTicket]:
        return request.app.state.service.desk.all()

    return app


def _maybe_enable_tracing() -> None:
    """Turn on MLflow tracing of agent runs when MLflow is installed and a tracking URI is set."""
    import os

    if not os.getenv("MLFLOW_TRACKING_URI"):
        return
    try:
        import mlflow

        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "careagent"))
        mlflow.pydantic_ai.autolog()
        log.info("MLflow tracing enabled")
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow tracing not enabled: %s", exc)


app = create_app()
