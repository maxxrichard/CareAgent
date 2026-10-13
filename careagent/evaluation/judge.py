"""LLM-as-judge for hallucination labelling.

The deterministic citation checker is the production guardrail; this judge is the
*evaluation* label. It sees the customer question, the agent's answer and all evidence the
agent retrieved, and decides whether the answer contains any statement not supported by
that evidence. Judged with Claude (Opus 5.5 by default) regardless of which backend is
being evaluated, so labels are comparable across backends.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from careagent.config import get_settings

JUDGE_INSTRUCTIONS = """\
You are a strict evaluator of a customer-support assistant. You are given the customer's
question, the assistant's answer, and the ONLY evidence the assistant had access to (policy
passages and order records). Decide whether the answer contains any factual statement about
policies, deadlines, fees, timeframes, order status or dates that is NOT supported by the
evidence. Polite phrases, apologies, and generic next-step suggestions are not factual claims.
A statement that contradicts the evidence is a hallucination. A statement that is plausible but
absent from the evidence is also a hallucination. If the assistant declined to answer or handed
off to a human without asserting facts, it did not hallucinate.
"""


class JudgeVerdict(BaseModel):
    hallucinated: bool = Field(
        description="True if any unsupported or contradicting factual claim is present"
    )
    unsupported_statements: list[str] = Field(default_factory=list)
    rationale: str = Field(description="One or two sentences")


def build_judge(model: Model | str | None = None) -> Agent[None, JudgeVerdict]:
    settings = get_settings()
    if model is None:
        from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

        model = AnthropicModel(settings.anthropic_model)
        model_settings = AnthropicModelSettings(
            max_tokens=1024,
            anthropic_thinking={"type": "adaptive"},
            anthropic_effort="low",
            anthropic_cache_instructions=True,
        )
    else:
        model_settings = None
    return Agent(
        model,
        output_type=JudgeVerdict,
        instructions=JUDGE_INSTRUCTIONS,
        model_settings=model_settings,
        name="careagent-judge",
        retries=2,
    )


def format_judge_prompt(question: str, answer: str, evidence: list[str]) -> str:
    ev = (
        "\n\n".join(f"[EVIDENCE {i + 1}]\n{e}" for i, e in enumerate(evidence))
        or "(no evidence retrieved)"
    )
    return f"CUSTOMER QUESTION:\n{question}\n\nASSISTANT ANSWER:\n{answer}\n\nEVIDENCE:\n{ev}"


async def judge_answer(
    judge: Agent[None, JudgeVerdict], question: str, answer: str, evidence: list[str]
) -> JudgeVerdict:
    result = await judge.run(format_judge_prompt(question, answer, evidence))
    return result.output
