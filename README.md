# CareAgent — Agentic Customer Self-Service Assistant

[![CI](https://github.com/maxxrichard/CareAgent/actions/workflows/ci.yml/badge.svg)](https://github.com/maxxrichard/CareAgent/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A multi-step customer self-service agent for high-volume customer care. It resolves order,
return and delivery questions end-to-end by calling **typed tools**, grounds every answer in
retrieved policy text with **RAG + a citation check that refuses unsupported claims**, and
**hands off to a human** when it is uncertain. An evaluation harness on **Databricks /
MLflow** scores 500 labelled queries for task success, containment, hallucination rate,
latency and cost across three LLM backends, and the agent is served as a **Docker REST API**
with real-time monitoring.

```
customer ──▶ FastAPI ──▶ PydanticAI agent ──▶ typed tools ──▶ citation check ──▶ answer | hand-off
                │             │   lookup_order · search_policies · escalate_to_human      │
                │             └── Anthropic Claude | OpenAI | Ollama (pluggable)          │
                └── /stats /metrics: p95 latency · €/query · hallucination · containment ◀┘
```

## Highlights

| Capability | How |
|---|---|
| Agentic orchestration | [PydanticAI](https://ai.pydantic.dev) agent with structured output (`AgentAnswer`) and four typed tools; the model decides the trajectory, the service enforces the policy |
| Typed tools | `lookup_order` / `list_recent_orders` (SQLite order store, scoped to the authenticated customer), `search_policies` (RAG), `escalate_to_human` (ticket queue) |
| Grounded answers | Hybrid retrieval (dense vectors + BM25, reciprocal-rank fusion) → cross-encoder reranking → the model must cite a `chunk_id` + verbatim quote for every factual claim |
| Citation check | Deterministic verifier: unknown sources, invented quotes, or claims (numbers, deadlines, fees) not covered by cited evidence are rejected and the answer is replaced with a safe hand-off |
| Human hand-off | Explicit tool call by the model, or automatic when grounding fails or the grounding score is below a threshold; every hand-off creates a ticket |
| Three LLM backends | `anthropic` (Claude, default, adaptive thinking + effort control + server-side refusal fallback), `openai`, `ollama` (local models) — one flag switches them |
| Evaluation harness | 500 labelled queries, task-success / containment / hallucination / tool-trajectory / retrieval-recall / latency / cost metrics, LLM judge, MLflow tracking, Delta table + SQL analytics on Databricks |
| Serving & monitoring | FastAPI + Docker; `/stats` (JSON) and `/metrics` (Prometheus) expose live p50/p95 latency, cost per query, hallucination and containment rates |

## Results

Evaluated on 500 labelled queries (orders, delivery, returns, cancellations, payments,
account questions, and hand-off cases) drawn from a public e-commerce customer-support
dataset and synthetic order records.

| Metric | Value |
|---|---|
| Task-success rate | **87 %** |
| Containment (resolved without human hand-off) | **71 %** |
| Hallucination rate (LLM-judged) | **9.8 % → 2.4 %** after RAG grounding + citation check |
| p95 latency | **1.8 s** |
| Cost per query | **€0.004** |

The harness compares the same agent across the Anthropic, OpenAI and Ollama backends; see
[`careagent/evaluation`](careagent/evaluation) and the SQL analytics in
[`analytics.sql`](careagent/evaluation/sql/analytics.sql).

## Quick start

```bash
git clone https://github.com/maxxrichard/CareAgent.git && cd CareAgent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,tracking,dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY (or OPENAI_API_KEY / an Ollama URL)

careagent seed                  # synthetic orders -> data/orders/orders.db
careagent build-index           # chunk + embed policies -> data/index/
careagent chat "How long do I have to return an item?"
```

Talk to it:

```
╭─ CareAgent ───────────────────────────────────────────────────────────────╮
│ You can return most items within 30 days of the delivery date for a full  │
│ refund; the window starts on the day the parcel is marked as delivered.   │
│ Returns within Germany are free. Open the order in your account and       │
│ choose "Return items" to get a prepaid label.                             │
│                                                                           │
│ Sources: return_policy#0.0, return_policy#4.0, return_policy#3.0          │
╰────────────────────────── 1430 ms · €0.0039 · grounding 1.00 ─────────────╯
```

### Run the API

```bash
careagent serve                                   # http://localhost:8000/docs
curl -s localhost:8000/chat -H 'content-type: application/json' \
  -d '{"message":"Where is my order ORD-100042?","customer_email":"anna.ben7@example.com"}' | jq
curl -s localhost:8000/stats | jq                 # live latency / cost / hallucination / containment
curl -s localhost:8000/metrics                    # Prometheus format
```

### Docker

```bash
docker build -t careagent .
docker run --rm -p 8000:8000 --env-file .env careagent
# or the full stack with MLflow + Prometheus:
docker compose up --build
```

## Switching LLM backends

```bash
CAREAGENT_LLM_BACKEND=anthropic CAREAGENT_ANTHROPIC_MODEL=claude-opus-5-5 careagent chat "..."
CAREAGENT_LLM_BACKEND=openai    CAREAGENT_OPENAI_MODEL=gpt-5              careagent chat "..."
CAREAGENT_LLM_BACKEND=ollama    CAREAGENT_OLLAMA_MODEL=llama3.3           careagent chat "..."
```

The Anthropic backend uses adaptive thinking with `CAREAGENT_EFFORT` (`low|medium|high`)
controlling reasoning depth, caches the instructions and tool definitions, and enables the
server-side refusal fallback so safety-classifier refusals are retried automatically.

## Evaluation

```bash
python scripts/run_eval.py --backends anthropic,openai,ollama          # full run, LLM judge
python scripts/run_eval.py --backends anthropic --limit 50 --no-judge  # quick, citation-check labels
mlflow ui                                                              # browse runs in ./mlruns
```

Each run logs parameters (backend, model, effort, embedder, reranker, k) and metrics to
MLflow, writes per-case rows to a Delta table when a Spark session is present (otherwise to
`data/eval/results/*.jsonl|parquet`), and saves a per-area breakdown. On Databricks, open
[`notebooks/databricks_eval_dashboard.py`](notebooks/databricks_eval_dashboard.py) to run the
comparison and render the SQL analytics (backend comparison, success by area, hallucinations
by area, escalation precision/recall, cost-latency trade-off, worst cases).

Metrics:

- **task success** — answer cases: not escalated, required facts present, no forbidden
  facts, expected tools called, no hallucination; hand-off cases: escalated.
- **containment** — share of queries resolved without a human.
- **hallucination rate** — LLM judge over the evidence the agent actually saw (falls back to
  the citation check); reported on answerable cases.
- **tool-trajectory accuracy**, **retrieval recall** (gold passage in the evidence),
  **escalation precision / recall**, **p50/p95 latency**, **€ per query**.

Regenerate the labelled set deterministically with `python scripts/generate_eval_set.py 500`.

## How grounding works

1. The model must call `search_policies` before stating any rule, and `lookup_order` for
   order facts. Everything a tool returns is recorded as the run's *evidence*.
2. The structured output carries `citations: [{source_id, quote}]`.
3. `CitationChecker` verifies, without another LLM call, that each `source_id` is in the
   evidence, each `quote` is a verbatim excerpt of it, and every *checkable* sentence (one
   with a number, duration, fee, date or policy keyword) is covered by cited evidence that
   contains the same numbers.
4. If anything fails under `CAREAGENT_STRICT_GROUNDING=true`, the answer is replaced with a
   safe reply, a ticket is opened and the response is flagged — so unsupported claims never
   reach the customer. The flag feeds the live hallucination-rate metric.

## Project layout

```
careagent/
  agent.py            PydanticAI agent + CareAgentService (grounding & hand-off policy)
  backends.py         anthropic | openai | ollama model factory, pricing
  prompts.py          system instructions
  tools/              lookup_order, list_recent_orders, search_policies, escalate_to_human
  rag/                chunking, embeddings, hybrid index (vectors + BM25), reranker, retriever
  grounding/          citation checker
  data/               policy documents loader, SQLite order store, public dataset loader
  api/                FastAPI app, request/response schemas, metrics registry
  evaluation/         labelled dataset, metrics, LLM judge, MLflow tracking, Databricks/Delta, SQL
  cli.py              careagent serve | chat | build-index | seed | eval
data/policies/        return, shipping, cancellation, payment, account policies (markdown)
data/eval/            labelled_queries.jsonl (500 cases), public_sample.jsonl
notebooks/            Databricks evaluation dashboard
scripts/              build_index, generate_eval_set, run_eval, load_test
tests/                offline tests with scripted models (no provider calls)
```

## Development

```bash
make dev        # editable install + pre-commit
make lint       # ruff
make test       # pytest (fully offline: PydanticAI FunctionModel plays the LLM)
```

## Data

- Policy documents are written for this project and live in `data/policies/`.
- Customer phrasings and intents come from the public
  [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
  (loaded on demand; a small bundled sample keeps everything runnable offline).
- Orders are synthetic and reproducible from a fixed seed, so evaluation cases can reference
  concrete order numbers.

## License

MIT — see [LICENSE](LICENSE).
