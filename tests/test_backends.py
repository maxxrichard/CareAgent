import pytest

from careagent.backends import build_backend, price_usd
from careagent.config import Settings


def test_price_usd():
    assert price_usd("claude-opus-5-5", 1_000_000, 0) == pytest.approx(4.0)
    assert price_usd("claude-opus-5-5", 0, 1_000_000) == pytest.approx(20.0)
    assert price_usd("claude-opus-5-5", 1_000_000, 0, cache_read_tokens=1_000_000) == pytest.approx(
        0.4
    )
    assert price_usd("ollama/llama3.3", 10_000, 10_000) == 0.0
    assert price_usd("some-unknown-model", 1000, 0) > 0


def test_build_anthropic_backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    spec = build_backend(Settings(llm_backend="anthropic", effort="low"))
    assert spec.name == "anthropic" and spec.model_name == "claude-opus-5-5"
    assert spec.settings["anthropic_thinking"] == {"type": "adaptive"}
    assert spec.settings["anthropic_effort"] == "low"
    assert spec.settings["extra_body"] == {"fallbacks": "default"}


def test_build_ollama_backend():
    spec = build_backend(Settings(llm_backend="ollama", ollama_model="llama3.3"))
    assert spec.name == "ollama" and spec.model_name == "ollama/llama3.3"


def test_unknown_backend():
    with pytest.raises(ValueError):
        build_backend(Settings(), backend="nope")
