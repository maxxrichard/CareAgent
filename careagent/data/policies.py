"""Load markdown policy documents (with a small YAML front-matter header)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from careagent.config import get_settings


@dataclass(frozen=True)
class PolicyDocument:
    doc_id: str
    title: str
    text: str
    path: Path


def _split_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        _, header, body = raw.split("---", 2)
        return yaml.safe_load(header) or {}, body.strip()
    return {}, raw.strip()


def load_policy_documents(policies_dir: Path | None = None) -> list[PolicyDocument]:
    policies_dir = policies_dir or get_settings().policies_dir
    docs: list[PolicyDocument] = []
    for path in sorted(policies_dir.glob("*.md")):
        meta, body = _split_front_matter(path.read_text(encoding="utf-8"))
        docs.append(
            PolicyDocument(
                doc_id=str(meta.get("id", path.stem)),
                title=str(meta.get("title", path.stem.replace("_", " ").title())),
                text=body,
                path=path,
            )
        )
    if not docs:
        raise FileNotFoundError(f"No policy documents found in {policies_dir}")
    return docs
