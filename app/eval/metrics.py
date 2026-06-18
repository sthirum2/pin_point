import math


def recall_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved results."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top_k)
    return hits / len(relevant)


def mrr(relevant: list[str], retrieved: list[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant item."""
    relevant_set = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def _dcg(relevant_set: set[str], retrieved: list[str], k: int) -> float:
    return sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant_set
    )


def ndcg_at_k(relevant: list[str], retrieved: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = _dcg(relevant_set, retrieved, k)
    # Ideal ranking: all relevant items ranked first
    ideal = _dcg(relevant_set, relevant, k)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal
