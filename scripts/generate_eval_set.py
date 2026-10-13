#!/usr/bin/env python
"""Regenerate data/eval/labelled_queries.jsonl (deterministic). Usage: python scripts/generate_eval_set.py [N]"""

import sys

from careagent.evaluation.dataset import DEFAULT_PATH, generate_cases, save_cases, stats

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    cases = generate_cases(n)
    save_cases(cases, DEFAULT_PATH)
    print(stats(cases))
