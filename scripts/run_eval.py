#!/usr/bin/env python3
"""Run the retrieval evaluation pipeline against a saved index and labeled query set.

Usage:
    python scripts/run_eval.py <index-dir> <labeled-set.json> [--rerank]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.dataset import load_labeled_set
from app.eval.run_eval import evaluate, print_report
from app.retrieval.embed import embed_query
from app.retrieval.index import SegmentIndex
from app.retrieval.rerank import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval pipeline against a labeled query set."
    )
    parser.add_argument("index_dir", help="Directory containing a saved SegmentIndex.")
    parser.add_argument("labeled_set", help="Path to the JSON labeled query set.")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply CrossEncoderReranker after bi-encoder retrieval.",
    )
    args = parser.parse_args()

    print(f"Loading index from {args.index_dir!r} …", flush=True)
    index = SegmentIndex.load(args.index_dir)

    print(f"Loading labeled set from {args.labeled_set!r} …", flush=True)
    labeled_set = load_labeled_set(args.labeled_set)
    print(f"  {len(labeled_set)} queries.", flush=True)

    reranker: CrossEncoderReranker | None = None
    if args.rerank:
        print("Mode: bi-encoder + CrossEncoderReranker", flush=True)
        reranker = CrossEncoderReranker()
    else:
        print("Mode: bi-encoder only (baseline)", flush=True)

    report = evaluate(index, labeled_set, embed_query, reranker=reranker)
    print_report(report)


if __name__ == "__main__":
    main()
