import math

import pytest

from app.eval.metrics import mrr, ndcg_at_k, recall_at_k


# ── recall_at_k ────────────────────────────────────────────────────────────────

def test_recall_perfect() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "b", "c", "d"], k=3) == 1.0


def test_recall_partial() -> None:
    assert recall_at_k(["a", "b", "c"], ["a", "x", "y", "b"], k=4) == pytest.approx(2 / 3)


def test_recall_zero_when_no_hit() -> None:
    assert recall_at_k(["a", "b"], ["x", "y", "z"], k=3) == 0.0


def test_recall_k_truncates() -> None:
    # only "a" appears in top-1; "b" is at rank 2 but k=1
    assert recall_at_k(["a", "b"], ["a", "b"], k=1) == pytest.approx(0.5)


def test_recall_empty_relevant() -> None:
    assert recall_at_k([], ["a", "b"], k=2) == 0.0


# ── mrr ───────────────────────────────────────────────────────────────────────

def test_mrr_first_hit_at_rank_1() -> None:
    assert mrr(["a"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_mrr_first_hit_at_rank_3() -> None:
    assert mrr(["c"], ["a", "b", "c"]) == pytest.approx(1 / 3)


def test_mrr_no_hit() -> None:
    assert mrr(["z"], ["a", "b", "c"]) == 0.0


def test_mrr_multiple_relevant_uses_first() -> None:
    # "b" is rank 2, "c" is rank 3 — MRR should be 1/2
    assert mrr(["b", "c"], ["a", "b", "c"]) == pytest.approx(0.5)


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking() -> None:
    assert ndcg_at_k(["a", "b"], ["a", "b", "c"], k=2) == pytest.approx(1.0)


def test_ndcg_reversed_ranking() -> None:
    # ideal: [a, b] → DCG = 1 + 1/log2(3)
    # actual: [b, a] → DCG = 1 + 1/log2(3)  (same gain, order symmetric for binary)
    assert ndcg_at_k(["a", "b"], ["b", "a", "c"], k=2) == pytest.approx(1.0)


def test_ndcg_partial_hit() -> None:
    # relevant: [a, b]; retrieved top-2: [a, x]
    # DCG   = 1/log2(2) = 1.0
    # IDCG  = 1/log2(2) + 1/log2(3) = 1 + 0.6309...
    dcg = 1.0
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(["a", "b"], ["a", "x", "b"], k=2) == pytest.approx(dcg / idcg)


def test_ndcg_no_hit() -> None:
    assert ndcg_at_k(["a"], ["x", "y", "z"], k=3) == 0.0


def test_ndcg_empty_relevant() -> None:
    assert ndcg_at_k([], ["a", "b"], k=2) == 0.0
