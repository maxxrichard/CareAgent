#!/usr/bin/env python
"""Run the evaluation harness across backends.

Examples:
    python scripts/run_eval.py --backends anthropic,openai,ollama
    python scripts/run_eval.py --backends anthropic --limit 50 --no-judge
"""

import argparse
import json
import logging

from careagent.evaluation.runner import run_evaluation

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="anthropic")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--lightweight", action="store_true", help="dependency-free embedder/reranker")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output", default="data/eval/results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    summary = run_evaluation(
        backends=args.backends.split(","),
        limit=args.limit,
        use_judge=not args.no_judge,
        lightweight_rag=args.lightweight,
        output_dir=args.output,
        concurrency=args.concurrency,
    )
    print(
        json.dumps({b: v["metrics"] for b, v in summary["backends"].items()}, indent=2, default=str)
    )
