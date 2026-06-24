from collections.abc import Callable

import numpy as np

from app.eval.dataset import LabeledQuery
from app.eval.metrics import mrr as _mrr
from app.eval.metrics import ndcg_at_k, recall_at_k
from app.models import SearchResult
from app.retrieval.index import SegmentIndex
from app.telemetry import log_retrieval

type EmbedFn = Callable[[str], np.ndarray]
type RerankerFn = Callable[[list[SearchResult], str], list[SearchResult]]
type EvalReport = dict[str, float | int]


def evaluate(
    index: SegmentIndex,
    labeled_set: list[LabeledQuery],
    embed_query_fn: EmbedFn,
    k_values: tuple[int, ...] = (1, 5, 10),
    reranker: RerankerFn | None = None,
) -> EvalReport:
    """Score the retrieval pipeline against a labeled query set.

    Returns a report dict with mean recall@k, nDCG@k, MRR, and num_queries.
    """
    if not labeled_set:
        raise ValueError("labeled_set is empty")

    max_k = max(k_values)

    accum: dict[str, list[float]] = {
        **{f"recall@{k}": [] for k in k_values},
        **{f"ndcg@{k}": [] for k in k_values},
        "mrr": [],
    }

    for item in labeled_set:
        query: str = item["query"]
        relevant: list[str] = item["relevant_ids"]

        vec = embed_query_fn(query)
        results = index.query(vec, k=max_k)

        if reranker is not None:
            results = reranker(results, query)

        log_retrieval(query, results, reranked=reranker is not None, k=max_k)

        retrieved_ids = [r.segment.id for r in results]

        for k in k_values:
            accum[f"recall@{k}"].append(recall_at_k(relevant, retrieved_ids, k))
            accum[f"ndcg@{k}"].append(ndcg_at_k(relevant, retrieved_ids, k))
        accum["mrr"].append(_mrr(relevant, retrieved_ids))

    n = len(labeled_set)
    report: EvalReport = {"num_queries": n}
    for key, values in accum.items():
        report[key] = sum(values) / len(values)

    return report


def print_report(report: EvalReport) -> None:
    """Print a formatted eval report table to stdout."""
    n = report.get("num_queries", "?")
    print(f"\nEvaluation report  ({n} queries)")
    print(f"  {'Metric':<14}  {'Score':>8}")
    print("  " + "-" * 24)

    def _sort_key(key: str) -> tuple[str, int]:
        if "@" in key:
            name, num = key.split("@", 1)
            return (name, int(num))
        return (key, 0)

    metric_keys = sorted(
        (k for k in report if k != "num_queries"),
        key=_sort_key,
    )
    for key in metric_keys:
        print(f"  {key:<14}  {float(report[key]):>8.4f}")
    print("  " + "-" * 24)
