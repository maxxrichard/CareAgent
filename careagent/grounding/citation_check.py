"""Citation checker.

The agent must emit a `Citation(source_id, quote)` for every factual claim. This module
verifies, deterministically and without another LLM call, that:

1. every cited `source_id` refers to evidence that was actually retrieved/looked up in
   this run (no invented sources);
2. every `quote` is a verbatim (whitespace/case-insensitive, lightly fuzzy) substring of
   that evidence (no invented quotes);
3. every *checkable claim* in the answer - a sentence containing a number, duration,
   price, date, carrier name, or policy keyword - is supported by at least one valid
   citation whose evidence shares the claim's key facts.

Unsupported claims are reported so the agent wrapper can refuse them or hand off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from careagent.models import AgentAnswer, Citation, GroundingReport, Order, PolicyChunk

_WS = re.compile(r"\s+")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_FACT_WORDS = re.compile(
    r"\b(day|days|business days|week|weeks|hour|hours|eur|€|%|percent|fee|free|refund|refunded|"
    r"return|returns|window|within|deadline|cancel|cancelled|dispatch|dispatched|shipped|delivered|"
    r"delivery|tracking|carrier|dhl|dpd|hermes|ups|gls|warranty|invoice|voucher|paypal|card|"
    r"processing|out for delivery|delayed|express|standard)\b",
    re.IGNORECASE,
)
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "will",
    "you",
    "your",
    "we",
    "our",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "from",
    "can",
    "has",
    "have",
    "was",
    "were",
    "been",
    "please",
    "i",
    "my",
    "me",
    "not",
    "if",
    "any",
}


def _norm(text: str) -> str:
    return _WS.sub(" ", text.lower().replace("’", "'").replace("—", "-")).strip()


def _keywords(text: str) -> set[str]:
    toks = re.findall(r"[a-z0-9€%]+(?:[.,]\d+)?", _norm(text))
    return {t for t in toks if t not in _STOP and len(t) > 1}


@dataclass
class EvidenceStore:
    """Everything the agent saw during a run: retrieved chunks and looked-up orders."""

    chunks: dict[str, PolicyChunk] = field(default_factory=dict)
    orders: dict[str, Order] = field(default_factory=dict)

    def add_chunks(self, chunks: list[PolicyChunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def add_order(self, order: Order) -> None:
        self.orders[f"order:{order.order_id}"] = order

    def text_for(self, source_id: str) -> str | None:
        if source_id in self.chunks:
            return self.chunks[source_id].text
        if source_id in self.orders:
            return order_as_text(self.orders[source_id])
        return None

    def is_empty(self) -> bool:
        return not self.chunks and not self.orders


def order_as_text(order: Order) -> str:
    """Render an order as prose so quotes against it can be verified verbatim."""
    items = ", ".join(f"{it.quantity} x {it.name}" for it in order.items)
    parts = [
        f"Order {order.order_id} status: {order.status.value.replace('_', ' ')}.",
        f"Placed on {order.placed_on.isoformat()}.",
        f"Items: {items}.",
        f"Total: {order.total_eur:.2f} EUR.",
    ]
    if order.estimated_delivery:
        parts.append(f"Estimated delivery: {order.estimated_delivery.isoformat()}.")
    if order.delivered_on:
        parts.append(f"Delivered on {order.delivered_on.isoformat()}.")
    if order.carrier:
        parts.append(f"Carrier: {order.carrier}.")
    if order.tracking_number:
        parts.append(f"Tracking number: {order.tracking_number}.")
    return " ".join(parts)


class CitationChecker:
    def __init__(self, min_quote_ratio: float = 0.85, min_claim_overlap: float = 0.34):
        self.min_quote_ratio = min_quote_ratio
        self.min_claim_overlap = min_claim_overlap

    # ---- helpers ------------------------------------------------------------
    def quote_in_source(self, quote: str, source: str) -> bool:
        q, s = _norm(quote), _norm(source)
        if not q:
            return False
        if q in s:
            return True
        # tolerate small transcription differences (punctuation, elisions)
        window = len(q)
        best = 0.0
        step = max(1, window // 4)
        for start in range(0, max(1, len(s) - window + 1), step):
            ratio = SequenceMatcher(None, q, s[start : start + window + 10]).ratio()
            if ratio > best:
                best = ratio
                if best >= self.min_quote_ratio:
                    return True
        return best >= self.min_quote_ratio

    @staticmethod
    def is_checkable(sentence: str) -> bool:
        return bool(_NUMBER.search(sentence)) or bool(_FACT_WORDS.search(sentence))

    @staticmethod
    def split_claims(answer: str) -> list[str]:
        sents = [s.strip() for s in _SENT.split(answer.replace("\n", " ")) if s.strip()]
        return [s for s in sents if len(s) > 12]

    def claim_supported(self, claim: str, evidence_texts: list[str]) -> bool:
        ck = _keywords(claim)
        if not ck:
            return True
        nums = set(_NUMBER.findall(claim))
        for ev in evidence_texts:
            ek = _keywords(ev)
            overlap = len(ck & ek) / len(ck)
            ev_nums = set(_NUMBER.findall(ev))
            # Every number in the claim must appear in the evidence - numbers are where
            # hallucinations hurt most (wrong refund window, wrong fee).
            numbers_ok = nums.issubset(ev_nums) if nums else True
            if numbers_ok and overlap >= self.min_claim_overlap:
                return True
        return False

    # ---- main entry point ---------------------------------------------------
    def check(self, answer: AgentAnswer, evidence: EvidenceStore) -> GroundingReport:
        valid: list[tuple[Citation, str]] = []
        invalid: list[str] = []
        for cit in answer.citations:
            src = evidence.text_for(cit.source_id)
            if src is None:
                invalid.append(f"unknown source {cit.source_id!r}")
            elif not self.quote_in_source(cit.quote, src):
                invalid.append(f"quote not found in {cit.source_id!r}: {cit.quote[:60]!r}")
            else:
                valid.append((cit, src))

        claims = self.split_claims(answer.answer)
        checkable = [c for c in claims if self.is_checkable(c)]
        evidence_texts = [src for _, src in valid]
        supported, unsupported = [], []
        for claim in checkable:
            (supported if self.claim_supported(claim, evidence_texts) else unsupported).append(
                claim
            )

        if not checkable:
            score = 1.0 if not invalid else 0.5
        else:
            score = len(supported) / len(checkable)
            if invalid:
                score *= 0.8
        flag = bool(unsupported) or bool(invalid)
        return GroundingReport(
            score=round(score, 3),
            supported_claims=supported,
            unsupported_claims=unsupported,
            invalid_citations=invalid,
            hallucination_flag=flag,
        )
