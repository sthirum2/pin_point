"""Analyze retrieval.jsonl and print summary statistics."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def analyze(log_path: Path, threshold: float) -> dict[str, float | int]:
    records: list[dict] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {
            "total_queries": 0,
            "mean_top1_score": 0.0,
            "median_top1_score": 0.0,
            "below_threshold_count": 0,
        }

    top1_scores: list[float] = []
    for record in records:
        if record.get("results"):
            top1_scores.append(float(record["results"][0]["score"]))
        else:
            top1_scores.append(0.0)

    return {
        "total_queries": len(records),
        "mean_top1_score": statistics.mean(top1_scores),
        "median_top1_score": statistics.median(top1_scores),
        "below_threshold_count": sum(1 for s in top1_scores if s < threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze retrieval logs")
    parser.add_argument(
        "--log",
        default=Path("data/logs/retrieval.jsonl"),
        type=Path,
        help="Path to retrieval.jsonl",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Top-1 score below this is flagged as low-quality retrieval",
    )
    args = parser.parse_args()

    stats = analyze(args.log, args.threshold)
    print(f"Total queries:          {stats['total_queries']}")
    print(f"Mean top-1 score:       {stats['mean_top1_score']:.4f}")
    print(f"Median top-1 score:     {stats['median_top1_score']:.4f}")
    print(
        f"Below threshold ({args.threshold:.2f}):  "
        f"{stats['below_threshold_count']}"
    )


if __name__ == "__main__":
    main()
