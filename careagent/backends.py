"""LLM backend factory.

Three backends are supported so the same agent can be benchmarked across providers:

* ``anthropic`` – Claude via the official Anthropic SDK (default). Adaptive thinking is
  always on; the ``effort`` setting controls reasoning depth, and server-side refusal
  fallbacks are enabled so safety-classifier refusals are retried on a fallback model.
* ``openai``    – OpenAI chat models via the official OpenAI SDK.
* ``ollama``    – any local model served through Ollama's OpenAI-compatible endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from careagent.config import Settings, get_settings

# USD per 1M tokens: (input, output). Used for per-query cost accounting.
PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


def price_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0
) -> float:
    """Estimate request cost in USD. Local (Ollama) models cost zero; unknown models fall back
    to a conservative mid-tier price so cost is never silently reported as 0."""
    if model.startswith(("ollama/", "function:", "test")) or model in {
        "llama3.3",
        "llama3.1",
        "mistral",
        "qwen2.5",
    }:
        return 0.0
    inp, out = PRICING_USD_PER_MTOK.get(model, (2.00, 10.00))
    # Cache reads are billed at ~10% of the input price on Anthropic; approximate uniformly.
    uncached = max(input_tokens - cache_read_tokens, 0)
    return (uncached * inp + cache_read_tokens * inp * 0.1 + output_tokens * out) / 1_000_000


@dataclass(frozen=True)
class BackendSpec:
    name: str
    model_name: str
    model: Model
    settings: ModelSettings


def build_backend(settings: Settings | None = None, backend: str | None = None) -> BackendSpec:
    """Create the PydanticAI model object for the configured backend."""
    settings = settings or get_settings()
    backend = backend or settings.llm_backend

    if backend == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        model_settings = AnthropicModelSettings(
            max_tokens=settings.max_output_tokens,
            timeout=settings.request_timeout_s,
            anthropic_thinking={"type": "adaptive"},
            anthropic_effort=settings.effort,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
        )
        if settings.anthropic_fallbacks:
            # Server-side refusal fallback: if a safety classifier declines the request the
            # API transparently retries on a fallback model instead of returning a refusal.
            model_settings["anthropic_betas"] = ["server-side-fallback-2026-07-01"]
            model_settings["extra_body"] = {"fallbacks": "default"}
        return BackendSpec(
            name="anthropic",
            model_name=settings.anthropic_model,
            model=AnthropicModel(settings.anthropic_model),
            settings=model_settings,
        )

    if backend == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel

        return BackendSpec(
            name="openai",
            model_name=settings.openai_model,
            model=OpenAIChatModel(settings.openai_model),
            settings=ModelSettings(
                max_tokens=settings.max_output_tokens, timeout=settings.request_timeout_s
            ),
        )

    if backend == "ollama":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.ollama import OllamaProvider

        return BackendSpec(
            name="ollama",
            model_name=f"ollama/{settings.ollama_model}",
            model=OpenAIChatModel(
                settings.ollama_model, provider=OllamaProvider(base_url=settings.ollama_base_url)
            ),
            settings=ModelSettings(
                max_tokens=settings.max_output_tokens, timeout=settings.request_timeout_s
            ),
        )

    raise ValueError(f"Unknown LLM backend: {backend!r} (expected anthropic | openai | ollama)")
