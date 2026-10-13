# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYDANTIC_AI_NO_BANNER=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY careagent ./careagent
RUN pip install --upgrade pip && pip install ".[embeddings,tracking]"

# Policies + labelled data are part of the image; the order DB and index are built at start.
COPY data ./data
COPY scripts ./scripts

# Pre-download the embedding and reranker models so the container starts fast offline.
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY
RUN python scripts/build_index.py && python -c "from careagent.data.orders import OrderStore; OrderStore.seeded()"

RUN useradd --create-home --uid 10001 app && chown -R app:app /app /opt/hf-cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"

CMD ["uvicorn", "careagent.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
