"""Labelled evaluation queries.

Each `EvalCase` carries the expected outcome (answer vs. hand-off), the tools that a correct
trajectory must call, facts that must appear in a correct answer, and the gold policy chunk
(for retrieval recall). The generator is deterministic: `generate_cases(n, seed)` always
yields the same cases, so results are comparable across backends and runs.

Phrasings come from the public customer-support dataset (through `public_dataset`) where
available, augmented with order-specific templates that reference synthetic orders.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from careagent.config import DATA_DIR, get_settings
from careagent.data.orders import OrderStore
from careagent.data.public_dataset import load_public_examples
from careagent.models import OrderStatus

DEFAULT_PATH = DATA_DIR / "eval" / "labelled_queries.jsonl"

Area = Literal[
    "order_status",
    "delivery",
    "returns",
    "cancellation",
    "payment",
    "account",
    "escalation",
    "out_of_scope",
]


class EvalCase(BaseModel):
    case_id: str
    query: str
    area: Area
    expected_outcome: Literal["answer", "escalate"]
    expected_tools: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(
        default_factory=list, description="substrings a correct answer contains"
    )
    forbidden_facts: list[str] = Field(
        default_factory=list, description="substrings that indicate a hallucination"
    )
    gold_chunk_prefix: str | None = Field(
        default=None, description="doc_id#section prefix of the gold passage"
    )
    order_id: str | None = None
    customer_email: str | None = None
    source: str = "template"


# ---- policy question bank: (query variants, area, required facts, gold chunk prefix, forbidden)
POLICY_QUESTIONS: list[tuple[list[str], Area, list[str], str, list[str]]] = [
    (
        [
            "How long do I have to return an item?",
            "What is your return window?",
            "Until when can I send something back?",
            "how many days to return a product",
        ],
        "returns",
        ["30 days"],
        "return_policy#0",
        ["14 days", "60 days", "90 days"],
    ),
    (
        [
            "Do I have to pay for return shipping?",
            "Is returning an item free?",
            "what does a return cost from Germany",
            "return shipping fee from Austria",
        ],
        "returns",
        ["free"],
        "return_policy#4",
        ["9.99", "5.99"],
    ),
    (
        [
            "When will I get my refund after returning?",
            "how long does a refund take on PayPal",
            "refund timing credit card",
            "my return arrived at the warehouse, when is the money back",
        ],
        "returns",
        ["business days"],
        "return_policy#5",
        ["30 business days", "immediately"],
    ),
    (
        [
            "Can I return opened cosmetics?",
            "is it possible to return an opened beauty product",
            "can I send back a plant I bought",
            "are gift cards refundable",
        ],
        "returns",
        ["cannot be returned"],
        "return_policy#2",
        ["14 days"],
    ),
    (
        [
            "Can I exchange a sweater for a different size?",
            "how do I exchange an item for another colour",
        ],
        "returns",
        ["do not offer direct exchanges"],
        "return_policy#7",
        ["exchange it for free"],
    ),
    (
        [
            "How do I start a return?",
            "what is the process to return something",
            "where do I get a return label",
        ],
        "returns",
        ["return label"],
        "return_policy#3",
        ["print"],
    ),
    (
        [
            "How long does standard delivery take?",
            "delivery time within Germany",
            "when does express delivery arrive",
            "how much does shipping cost",
        ],
        "delivery",
        ["business day"],
        "shipping_policy#0",
        ["7 to 10", "14 days"],
    ),
    (
        [
            "My parcel is late, what can I do?",
            "the estimated delivery date passed and nothing arrived",
            "delivery is delayed, what happens now",
        ],
        "delivery",
        ["3 business days"],
        "shipping_policy#3",
        ["24 hours", "14 days"],
    ),
    (
        [
            "Can I change my delivery address?",
            "I entered the wrong address, can you update it",
            "change shipping address after ordering",
        ],
        "delivery",
        ["dispatched"],
        "shipping_policy#5",
        ["fee"],
    ),
    (
        [
            "What happens if I miss the delivery?",
            "the courier could not deliver, where is my parcel",
            "how long is a parcel held at the pick-up point",
        ],
        "delivery",
        ["7 days"],
        "shipping_policy#6",
        ["14 days", "3 days"],
    ),
    (
        ["When do I get a tracking number?", "where can I track my parcel"],
        "delivery",
        ["tracking"],
        "shipping_policy#2",
        ["48 hours"],
    ),
    (
        ["Can I cancel my order?", "how do I cancel an order I just placed", "is cancelling free"],
        "cancellation",
        ["dispatched"],
        "cancellation_policy#0",
        ["fee", "24 hours"],
    ),
    (
        ["Can I add an item to my order?", "I want to change the items in my order"],
        "cancellation",
        ["cancel"],
        "cancellation_policy#1",
        [],
    ),
    (
        ["When will I be refunded for a cancelled order?", "cancelled order refund time"],
        "cancellation",
        ["3 to 5 business days"],
        "cancellation_policy#2",
        ["14 days", "10 business days"],
    ),
    (
        [
            "Which payment methods do you accept?",
            "can I pay with Apple Pay",
            "do you offer payment by invoice",
        ],
        "payment",
        ["PayPal"],
        "payment_policy#0",
        ["Klarna", "cash on delivery"],
    ),
    (
        ["When is my card charged?", "why has my payment not been taken yet"],
        "payment",
        ["dispatched"],
        "payment_policy#1",
        [],
    ),
    (
        ["Where can I find my invoice?", "I need a PDF invoice for my order"],
        "payment",
        ["PDF"],
        "payment_policy#2",
        [],
    ),
    (
        ["Can I use two discount codes?", "how do vouchers work with returns"],
        "payment",
        ["one promotional code"],
        "payment_policy#3",
        ["two codes"],
    ),
    (
        ["My payment was declined, what now?", "payment failed, will my order be cancelled"],
        "payment",
        ["48 hours"],
        "payment_policy#4",
        ["24 hours", "72 hours"],
    ),
    (
        ["What are your customer service hours?", "when can I reach support by phone"],
        "account",
        ["08:00"],
        "account_policy#0",
        ["24/7"],
    ),
    (
        ["How do I delete my account?", "how long does account deletion take"],
        "account",
        ["30 days"],
        "account_policy#1",
        ["immediately", "7 days"],
    ),
    (
        ["How long is the warranty?", "my headphones broke after a year, am I covered"],
        "account",
        ["two-year"],
        "account_policy#4",
        ["one-year", "5 years"],
    ),
    (
        ["Can I download my personal data?", "I want a copy of my data"],
        "account",
        ["72 hours"],
        "account_policy#2",
        ["30 days"],
    ),
]

ESCALATION_QUERIES: list[tuple[str, Area]] = [
    ("I want to speak to a human agent right now.", "escalation"),
    ("Connect me with a real person please.", "escalation"),
    (
        "My parcel arrived completely crushed and the espresso machine inside is broken.",
        "escalation",
    ),
    (
        "The tracking says delivered but there is nothing at my door and the neighbours have nothing.",
        "escalation",
    ),
    ("I was charged twice for the same order, I want this fixed today.", "escalation"),
    (
        "This is the third time I'm writing and nobody helps me. I want to file a complaint.",
        "escalation",
    ),
    ("I received a completely different product than what I ordered.", "escalation"),
    ("You charged my card but I never placed this order. Is this fraud?", "escalation"),
    ("I need an exception: can I return a sweater after 45 days? It was a gift.", "escalation"),
    ("Can you give me a discount on my next order because of the delay?", "escalation"),
    ("Can you tell me what the weather will be like tomorrow in Berlin?", "out_of_scope"),
    ("Write me a poem about my new headphones.", "out_of_scope"),
    ("Can you recommend which laptop I should buy?", "out_of_scope"),
    ("I'd like to subscribe to the newsletter and get a 10% code.", "out_of_scope"),
    ("What is the status of order ORD-999999?", "order_status"),
]

ORDER_TEMPLATES = [
    "Where is my order {oid}?",
    "What's the status of {oid}?",
    "Has {oid} shipped yet?",
    "Can you check order {oid} for me?",
    "When will {oid} arrive?",
    "Track {oid} please",
    "I ordered last week, order number {oid}, any update?",
    "Is {oid} delivered already?",
]


def _facts_for_order(store: OrderStore, oid: str) -> tuple[list[str], str, list[str]]:
    order = store.get(oid)
    assert order is not None
    status_word = order.status.value.replace("_", " ")
    facts = [
        status_word.split()[0]
        if order.status != OrderStatus.OUT_FOR_DELIVERY
        else "out for delivery"
    ]
    outcome = "answer"
    extra_tools: list[str] = []
    if order.status == OrderStatus.DELAYED:
        extra_tools = ["search_policies"]
    return facts, outcome, extra_tools


def generate_cases(
    n: int = 500, seed: int | None = None, store: OrderStore | None = None, use_public: bool = True
) -> list[EvalCase]:
    settings = get_settings()
    rng = random.Random(seed if seed is not None else settings.seed)
    store = store or OrderStore.seeded()
    orders = store.all()
    cases: list[EvalCase] = []

    # ---- 1. policy questions (templated phrasings) -------------------------
    pool: list[EvalCase] = []
    for qi, (variants, area, facts, gold, forbidden) in enumerate(POLICY_QUESTIONS):
        for vi, q in enumerate(variants):
            pool.append(
                EvalCase(
                    case_id=f"policy-{qi:02d}-{vi}",
                    query=q,
                    area=area,
                    expected_outcome="answer",
                    expected_tools=["search_policies"],
                    required_facts=facts,
                    forbidden_facts=forbidden,
                    gold_chunk_prefix=gold,
                )
            )

    # ---- 2. public-dataset phrasings mapped to the question bank ------------
    if use_public:
        try:
            public = load_public_examples(limit=2000, prefer_remote=False)
        except FileNotFoundError:
            public = []
        area_to_bank = {}
        for qi, (_, area, facts, gold, forbidden) in enumerate(POLICY_QUESTIONS):
            area_to_bank.setdefault(area, []).append((qi, facts, gold, forbidden))
        for pi, ex in enumerate(public):
            if ex.area in ("escalation", "out_of_scope"):
                pool.append(
                    EvalCase(
                        case_id=f"public-{pi:04d}",
                        query=ex.utterance,
                        area=ex.area,
                        expected_outcome="escalate",
                        expected_tools=["escalate_to_human"],
                        source=ex.intent,
                    )
                )
            elif ex.area in area_to_bank and ex.intent in {
                "check_refund_policy",
                "delivery_period",
                "delivery_options",
                "check_payment_methods",
                "check_cancellation_fee",
                "contact_customer_service",
                "delete_account",
                "get_invoice",
            }:
                qi, facts, gold, forbidden = rng.choice(area_to_bank[ex.area])
                pool.append(
                    EvalCase(
                        case_id=f"public-{pi:04d}",
                        query=ex.utterance,
                        area=ex.area,
                        expected_outcome="answer",
                        expected_tools=["search_policies"],
                        required_facts=[],
                        gold_chunk_prefix=gold,
                        forbidden_facts=forbidden,
                        source=ex.intent,
                    )
                )

    # ---- 3. order-status questions on real synthetic orders -----------------
    order_cases: list[EvalCase] = []
    for i, order in enumerate(rng.sample(orders, min(len(orders), n // 3))):
        tmpl = rng.choice(ORDER_TEMPLATES)
        facts, outcome, extra = _facts_for_order(store, order.order_id)
        order_cases.append(
            EvalCase(
                case_id=f"order-{i:03d}",
                query=tmpl.format(oid=order.order_id),
                area="order_status",
                expected_outcome=outcome,
                expected_tools=["lookup_order", *extra],
                required_facts=facts,
                order_id=order.order_id,
                customer_email=order.customer_email,
            )
        )

    # ---- 4. escalation / out-of-scope ---------------------------------------
    esc_cases = [
        EvalCase(
            case_id=f"escalate-{i:02d}",
            query=q,
            area=area,
            expected_outcome="escalate",
            expected_tools=["escalate_to_human"]
            if area != "order_status"
            else ["lookup_order", "escalate_to_human"],
        )
        for i, (q, area) in enumerate(ESCALATION_QUERIES)
    ]

    # ---- assemble to exactly n, stratified -----------------------------------
    rng.shuffle(pool)
    n_orders = len(order_cases)
    n_esc = min(len(esc_cases) * 3, n // 10)  # ~10% hand-off cases, repeated phrasings allowed
    esc_sel = [
        esc_cases[i % len(esc_cases)].model_copy(update={"case_id": f"escalate-{i:03d}"})
        for i in range(n_esc)
    ]
    n_policy = n - n_orders - len(esc_sel)
    policy_sel = [
        pool[i % len(pool)].model_copy(
            update={"case_id": f"{pool[i % len(pool)].case_id}-r{i // len(pool)}"}
        )
        for i in range(n_policy)
    ]
    cases = policy_sel + order_cases + esc_sel
    rng.shuffle(cases)
    return cases[:n]


def save_cases(cases: list[EvalCase], path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(c.model_dump_json(exclude_none=True) + "\n")


def load_cases(path: Path = DEFAULT_PATH, limit: int | None = None) -> list[EvalCase]:
    cases = [
        EvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return cases[:limit] if limit else cases


def stats(cases: list[EvalCase]) -> dict:
    by_area: dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    for c in cases:
        by_area[c.area] = by_area.get(c.area, 0) + 1
        by_outcome[c.expected_outcome] = by_outcome.get(c.expected_outcome, 0) + 1
    return {
        "n": len(cases),
        "by_area": by_area,
        "by_outcome": by_outcome,
        "sources": json.dumps(sorted({c.source for c in cases})),
    }
