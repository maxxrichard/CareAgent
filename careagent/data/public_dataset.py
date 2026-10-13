"""Loader for a public e-commerce customer-support dataset (Hugging Face).

We use the open Bitext customer-support dataset for realistic intents and customer
phrasings (ORDER / REFUND / DELIVERY / CANCEL / PAYMENT / ACCOUNT categories). The loader
is optional: when the `datasets` package or network access is unavailable we fall back to
a small bundled sample so that the evaluation harness still runs offline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from careagent.config import DATA_DIR

HF_DATASET = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SAMPLE_PATH = DATA_DIR / "eval" / "public_sample.jsonl"

# Map dataset intents onto the agent's capability areas.
INTENT_TO_AREA = {
    "track_order": "order_status",
    "change_order": "cancellation",
    "cancel_order": "cancellation",
    "place_order": "out_of_scope",
    "get_refund": "returns",
    "check_refund_policy": "returns",
    "track_refund": "returns",
    "delivery_options": "delivery",
    "delivery_period": "delivery",
    "change_shipping_address": "delivery",
    "set_up_shipping_address": "delivery",
    "payment_issue": "payment",
    "check_payment_methods": "payment",
    "check_invoice": "payment",
    "get_invoice": "payment",
    "contact_customer_service": "account",
    "contact_human_agent": "escalation",
    "complaint": "escalation",
    "review": "out_of_scope",
    "create_account": "account",
    "delete_account": "account",
    "edit_account": "account",
    "recover_password": "account",
    "registration_problems": "account",
    "switch_account": "account",
    "newsletter_subscription": "out_of_scope",
    "check_cancellation_fee": "cancellation",
}


@dataclass(frozen=True)
class SupportExample:
    utterance: str
    intent: str
    category: str
    area: str


def load_public_examples(
    limit: int | None = None, prefer_remote: bool = True
) -> list[SupportExample]:
    examples: list[SupportExample] = []
    if prefer_remote:
        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset(HF_DATASET, split="train")
            for row in ds:
                intent = str(row["intent"])
                examples.append(
                    SupportExample(
                        utterance=str(row["instruction"]),
                        intent=intent,
                        category=str(row["category"]),
                        area=INTENT_TO_AREA.get(intent, "out_of_scope"),
                    )
                )
                if limit and len(examples) >= limit:
                    break
            return examples
        except Exception:  # noqa: BLE001 - offline / package missing: use bundled sample
            examples = []
    with SAMPLE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                examples.append(
                    SupportExample(
                        utterance=row["utterance"],
                        intent=row["intent"],
                        category=row["category"],
                        area=INTENT_TO_AREA.get(row["intent"], "out_of_scope"),
                    )
                )
    return examples[:limit] if limit else examples


def write_sample(examples: list[SupportExample], path: Path = SAMPLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {"utterance": ex.utterance, "intent": ex.intent, "category": ex.category}
                )
                + "\n"
            )
