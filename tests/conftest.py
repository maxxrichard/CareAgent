"""Shared fixtures: an offline retriever, an in-memory order store and scripted models.

No test talks to a real LLM provider. Agent behaviour is exercised through PydanticAI's
`FunctionModel`, which lets a Python function play the model and emit tool calls.
"""

from __future__ import annotations

import datetime
import os

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from careagent.agent import CareAgentService
from careagent.config import Settings
from careagent.data.orders import OrderStore
from careagent.rag.retriever import PolicyRetriever

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TODAY = datetime.date(2026, 10, 13)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(strict_grounding=True, escalation_threshold=0.6)


@pytest.fixture(scope="session")
def retriever(settings) -> PolicyRetriever:
    return PolicyRetriever.build(settings, lightweight=True)


@pytest.fixture()
def orders() -> OrderStore:
    return OrderStore.seeded(db_path=":memory:", n=50, seed=7, today=TODAY)


def _last_tool_return(messages):
    for part in reversed(messages[-1].parts):
        if isinstance(part, ToolReturnPart):
            return part
    return None


def scripted_model(script):
    """Build a FunctionModel from a list of steps.

    Each step is either a (tool_name, args) tuple -> emits a tool call, or a dict -> emits the
    final structured answer. Steps are consumed in order, one per model request.
    """
    steps = list(script)

    def fn(messages, info: AgentInfo) -> ModelResponse:
        if not steps:
            return ModelResponse(parts=[TextPart("no more steps")])
        step = steps.pop(0)
        if isinstance(step, tuple):
            name, args = step
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[ToolCallPart("final_result", step)])

    return FunctionModel(fn, model_name="test-scripted")


def reactive_model(decide):
    """FunctionModel whose behaviour depends on the last tool result (via `decide(last_return)`)."""

    def fn(messages, info: AgentInfo) -> ModelResponse:
        step = decide(_last_tool_return(messages))
        if isinstance(step, tuple):
            return ModelResponse(parts=[ToolCallPart(step[0], step[1])])
        return ModelResponse(parts=[ToolCallPart("final_result", step)])

    return FunctionModel(fn, model_name="test-reactive")


@pytest.fixture()
def make_service(settings, retriever, orders):
    def _make(model, **kw) -> CareAgentService:
        return CareAgentService.create(
            settings, model=model, orders=orders, retriever=retriever, **kw
        )

    return _make
