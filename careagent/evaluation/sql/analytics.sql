-- Evaluation analytics over the Delta table written by careagent.evaluation.databricks.
-- Each block is addressable by name via `analytics_queries(table)`.

-- name: backend_comparison
SELECT
  backend,
  model,
  COUNT(*)                                            AS n,
  AVG(CAST(task_success AS INT))                      AS task_success_rate,
  AVG(CAST(NOT escalated AS INT))                     AS containment_rate,
  AVG(CASE WHEN expected_outcome = 'answer'
           THEN CAST(hallucination AS INT) END)       AS hallucination_rate,
  PERCENTILE_APPROX(latency_ms, 0.5)                  AS latency_p50_ms,
  PERCENTILE_APPROX(latency_ms, 0.95)                 AS latency_p95_ms,
  AVG(cost_eur)                                       AS cost_eur_per_query,
  SUM(cost_eur)                                       AS cost_eur_total
FROM ${table}
GROUP BY backend, model
ORDER BY task_success_rate DESC;

-- name: success_by_area
SELECT
  backend,
  area,
  COUNT(*)                                            AS n,
  AVG(CAST(task_success AS INT))                      AS task_success_rate,
  AVG(CAST(tool_trajectory_ok AS INT))                AS tool_trajectory_accuracy,
  AVG(CASE WHEN retrieval_hit IS NOT NULL
           THEN CAST(retrieval_hit AS INT) END)       AS retrieval_recall
FROM ${table}
GROUP BY backend, area
ORDER BY backend, area;

-- name: hallucinations_by_area
SELECT
  backend,
  area,
  AVG(CAST(hallucination AS INT))                     AS hallucination_rate,
  AVG(CAST(citation_flag AS INT))                     AS citation_flag_rate,
  AVG(grounding_score)                                AS mean_grounding_score
FROM ${table}
WHERE expected_outcome = 'answer'
GROUP BY backend, area
ORDER BY hallucination_rate DESC;

-- name: escalation_quality
SELECT
  backend,
  SUM(CASE WHEN escalated AND expected_outcome = 'escalate' THEN 1 ELSE 0 END)
    / NULLIF(SUM(CASE WHEN escalated THEN 1 ELSE 0 END), 0)                     AS escalation_precision,
  SUM(CASE WHEN escalated AND expected_outcome = 'escalate' THEN 1 ELSE 0 END)
    / NULLIF(SUM(CASE WHEN expected_outcome = 'escalate' THEN 1 ELSE 0 END), 0) AS escalation_recall
FROM ${table}
GROUP BY backend;

-- name: cost_latency_tradeoff
SELECT
  backend,
  model,
  AVG(input_tokens)   AS input_tokens_mean,
  AVG(output_tokens)  AS output_tokens_mean,
  AVG(cost_eur)       AS cost_eur_per_query,
  AVG(latency_ms)     AS latency_mean_ms,
  AVG(cost_eur) / NULLIF(AVG(CAST(task_success AS INT)), 0) AS cost_eur_per_successful_query
FROM ${table}
GROUP BY backend, model
ORDER BY cost_eur_per_successful_query;

-- name: worst_cases
SELECT run_id, backend, case_id, area, query, answer, grounding_score
FROM ${table}
WHERE hallucination OR (NOT task_success AND expected_outcome = 'answer')
ORDER BY grounding_score ASC
LIMIT 50;
