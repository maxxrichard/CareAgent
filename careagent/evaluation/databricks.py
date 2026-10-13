"""Databricks integration: persist evaluation results as a Delta table and run SQL analytics.

When executed inside a Databricks runtime (a `spark` session exists) results go straight to
Unity Catalog. Elsewhere, `write_results` falls back to local Parquet/JSONL so the same
code path is testable offline; `sql/analytics.sql` can then be run against the table with
Databricks SQL (or DuckDB locally for a quick look).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
SQL_DIR = Path(__file__).parent / "sql"


def _spark():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession()
    except Exception:  # noqa: BLE001
        return None


def write_results(rows: list[dict[str, Any]], table: str, local_dir: Path, run_id: str) -> str:
    """Append rows to a Delta table (if Spark is available) or to local files. Returns the location."""
    for r in rows:
        r["run_id"] = run_id
    spark = _spark()
    if spark is not None:
        df = spark.createDataFrame(rows)
        df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
        log.info("appended %d rows to Delta table %s", len(rows), table)
        return table
    local_dir.mkdir(parents=True, exist_ok=True)
    out = local_dir / f"results-{run_id}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(local_dir / f"results-{run_id}.parquet", index=False)
    except Exception:  # noqa: BLE001 - pandas/pyarrow optional
        pass
    return str(out)


def analytics_queries(table: str) -> dict[str, str]:
    """Return the named SQL analytics queries with the table name substituted."""
    sql = (SQL_DIR / "analytics.sql").read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    name = None
    buf: list[str] = []
    for line in sql.splitlines():
        if line.startswith("-- name:"):
            if name and buf:
                queries[name] = "\n".join(buf).strip()
            name, buf = line.split(":", 1)[1].strip(), []
        elif name is not None:
            buf.append(line)
    if name and buf:
        queries[name] = "\n".join(buf).strip()
    return {k: v.replace("${table}", table) for k, v in queries.items()}


def run_analytics(table: str) -> dict[str, Any]:
    spark = _spark()
    if spark is None:
        raise RuntimeError("No active Spark session; run inside Databricks or use DuckDB locally")
    return {
        name: spark.sql(q).toPandas().to_dict(orient="records")
        for name, q in analytics_queries(table).items()
    }
