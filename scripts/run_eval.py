#!/usr/bin/env python3
"""Run the retrieval evaluation pipeline against a saved index and labeled query set.

Usage:
    python scripts/run_eval.py <index-dir> <labeled-set.json>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval.dataset import load_labeled_set
from app.eval.run_eval import evaluate, print_report
from app.retrieval.embed import embed_query
from app.retrieval.index import SegmentIndex


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/run_eval.py <index-dir> <labeled-set.json>",
            file=sys.stderr,
        )
        sys.exit(1)

    index_path, labeled_set_path = sys.argv[1], sys.argv[2]

    print(f"Loading index from {index_path!r} …", flush=True)
    index = SegmentIndex.load(index_path)

    print(f"Loading labeled set from {labeled_set_path!r} …", flush=True)
    labeled_set = load_labeled_set(labeled_set_path)
    print(f"  {len(labeled_set)} queries.", flush=True)

    report = evaluate(index, labeled_set, embed_query)
    print_report(report)


if __name__ == "__main__":
    main()
