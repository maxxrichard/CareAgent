"""MLflow tracking helpers (local file store by default, Databricks when configured)."""

from __future__ import annotations

import contextlib
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _mlflow():
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


@contextmanager
def tracked_run(
    run_name: str, experiment: str, params: dict[str, Any], tags: dict[str, str] | None = None
):
    """Context manager that yields a `log(metrics, artifacts)` callable; no-op without MLflow."""
    mlflow = _mlflow()
    if mlflow is None:
        log.warning("mlflow not installed; results are written to disk only")

        def _noop(
            metrics: dict[str, float], artifacts: list[Path] | None = None, step: int | None = None
        ) -> None:
            return None

        yield _noop
        return

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", experiment))
    with contextlib.suppress(Exception):
        mlflow.pydantic_ai.autolog(silent=True)
    with mlflow.start_run(run_name=run_name, tags=tags or {}):
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

        def _log(
            metrics: dict[str, float], artifacts: list[Path] | None = None, step: int | None = None
        ) -> None:
            mlflow.log_metrics(
                {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)}, step=step
            )
            for a in artifacts or []:
                mlflow.log_artifact(str(a))

        yield _log
