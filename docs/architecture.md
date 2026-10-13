# Architecture

## Request flow

```
POST /chat {message, session_id?, customer_email?}
  │
  ▼
CareAgentService.answer()
  ├─ builds AgentDeps (order store, retriever, escalation desk, EvidenceStore)
  ├─ agent.run(query, deps)                    ← PydanticAI, UsageLimits caps tool loops
  │     model ⇄ tools:
  │       lookup_order(order_id)      → OrderLookupResult   (adds order to evidence)
  │       list_recent_orders()        → [order summaries]   (scoped by customer_email)
  │       search_policies(query)      → PolicySearchResult  (adds chunks to evidence)
  │       escalate_to_human(reason…)  → EscalationTicket
  │     final_result → AgentAnswer{answer, citations[], needs_human, handoff_reason, confidence}
  ├─ CitationChecker.check(answer, evidence)  → GroundingReport
  ├─ _apply_policy()                           → final text, citations, escalated flag
  └─ AgentResponse (+ latency, cost, usage, tools_used)
  │
  ▼
MetricsRegistry.record_response()  → /stats, /metrics
```

## Components

### Agent (`careagent/agent.py`, `prompts.py`, `tools/`)
A single PydanticAI `Agent[AgentDeps, AgentAnswer]`. Tools are plain typed functions whose
docstrings become the tool descriptions; return values are Pydantic models so the model
sees stable JSON. `AgentDeps` carries per-run state (evidence, tools used, ticket id).

### Backends (`careagent/backends.py`)
`build_backend()` returns a `BackendSpec(model, settings)`. Anthropic uses the official SDK
through PydanticAI with adaptive thinking, `effort`, prompt caching of instructions/tools and
the server-side refusal fallback (`fallbacks: "default"`). OpenAI uses the official OpenAI
SDK. Ollama uses the OpenAI-compatible provider pointed at a local server. Pricing is a
small table used for € per query accounting.

### Retrieval (`careagent/rag/`)
- `chunking.py` — splits markdown on headings, windows long sections on sentence boundaries,
  prefixes each chunk with document + section title.
- `embeddings.py` — `SentenceTransformerEmbedder` (all-MiniLM-L6-v2) with a deterministic
  `HashingEmbedder` fallback (feature-hashed unigrams + bigrams) for CI/offline use.
- `index.py` — `HybridIndex`: cosine search over a numpy matrix + BM25 (`rank-bm25`), fused
  with reciprocal-rank fusion; persisted as `vectors.npy` + `chunks.jsonl` + `meta.json`.
- `reranker.py` — `CrossEncoderReranker` (ms-marco-MiniLM) with an `OverlapReranker`
  fallback.
- `retriever.py` — `PolicyRetriever.retrieve(query)` = search top-k → rerank → top-n.

### Grounding (`careagent/grounding/citation_check.py`)
Three checks, all deterministic:
1. **source validity** — `source_id` must be a chunk or order the agent actually saw;
2. **quote validity** — the quote must be a (lightly fuzzy, normalised) substring of the
   source;
3. **claim coverage** — every checkable sentence (numbers, durations, fees, dates, policy
   keywords) must share ≥ 34 % of its keywords and *all* of its numbers with at least one
   validly-cited source.

Score = supported / checkable claims (× 0.8 if any citation was invalid). Any unsupported
claim or invalid citation sets `hallucination_flag`.

### Hand-off policy (`CareAgentService._apply_policy`)
| Condition | Outcome |
|---|---|
| model set `needs_human` or called `escalate_to_human` | answer kept, ticket ensured, `escalated=true` |
| `hallucination_flag` and strict grounding | answer replaced by safe fallback, ticket opened |
| grounding score < `escalation_threshold` | answer kept, ticket opened as safety net |
| otherwise | answer returned as-is |

### API & monitoring (`careagent/api/`)
FastAPI with lifespan-managed service. `MetricsRegistry` keeps a sliding window of request
records and derives p50/p95/p99 latency, mean € per query, hallucination rate, containment
rate and mean grounding score; `/metrics` renders Prometheus text. When
`MLFLOW_TRACKING_URI` is set, `mlflow.pydantic_ai.autolog()` traces every run.

### Evaluation (`careagent/evaluation/`)
- `dataset.py` — `EvalCase` schema and a deterministic generator mixing a policy question
  bank, public-dataset phrasings, order-specific templates over synthetic orders, and
  hand-off cases. 500 cases are committed in `data/eval/labelled_queries.jsonl`.
- `metrics.py` — per-case scoring (`score_case`) and aggregates (`aggregate`, `by_area`).
- `judge.py` — Claude-based LLM judge producing `JudgeVerdict{hallucinated, …}` from the
  exact evidence the agent saw.
- `runner.py` — runs cases concurrently per backend, scores, logs to MLflow, writes rows.
- `tracking.py` — MLflow helper that works with a local store or Databricks.
- `databricks.py` + `sql/analytics.sql` — Delta table writer (Spark) with local fallback and
  named SQL analytics queries.

## Testing strategy
All tests are offline. PydanticAI's `FunctionModel` plays the LLM: scripted functions emit
tool calls and the `final_result` structured output, so the tool loop, evidence capture,
citation check, hand-off policy, API and evaluation harness are exercised end-to-end
without network access.
