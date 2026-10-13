#!/usr/bin/env python
"""Build and persist the policy vector index. Usage: python scripts/build_index.py [--lightweight]"""

import sys

from careagent.config import get_settings
from careagent.rag import PolicyRetriever

if __name__ == "__main__":
    settings = get_settings()
    r = PolicyRetriever.build(settings, lightweight="--lightweight" in sys.argv)
    r.index.save(settings.index_dir)
    print(f"indexed {len(r.index)} chunks with {r.index.embedder.name} -> {settings.index_dir}")
