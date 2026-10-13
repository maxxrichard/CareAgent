"""Grounding: verify that every factual claim in an answer is backed by cited evidence."""

from careagent.grounding.citation_check import CitationChecker, EvidenceStore

__all__ = ["CitationChecker", "EvidenceStore"]
