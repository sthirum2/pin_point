"""Offline tests for the evaluation harness — no network, no model loading."""
import json
import math

import numpy as np
import pytest

from app.eval.dataset import LabeledQuery, load_labeled_set
from app.eval.run_eval import evaluate, print_report
from app.models import Segment, SearchResult
from app.retrieval.index import SegmentIndex


# ── shared fixtures ───────────────────────────────────────────────────────────

def make_segment(seg_id: str, vec: list[float]) -> Segment:
    s = Segment(id=seg_id, video_id="v0", start=0.0, end=1.0, text=seg_id)
    s.embedding = np.array(vec, dtype=np.float32)
    return s


def build_index() -> SegmentIndex:
    """Three orthogonal unit-vector segments."""
    segs = [
        make_segment("a", [1.0, 0.0, 0.0]),
        make_segment("b", [0.0, 1.0, 0.0]),
        make_segment("c", [0.0, 0.0, 1.0]),
    ]
    idx = SegmentIndex()
    idx.build(segs)
    return idx


# Two queries with distinct, non-tied retrieval orderings:
#   q_a      → embed [1, 0, 0]        → retrieves a first  (relevant: a)
#   q_c_far  → embed [0.9, 0.4, 0.1]  → retrieves a>b>c    (relevant: c — c at rank 3)
_EMBED: dict[str, list[float]] = {
    "q_a":     [1.0, 0.0, 0.0],
    "q_c_far": [0.9, 0.4, 0.1],
}


def fake_embed(text: str) -> np.ndarray:
    return np.array(_EMBED[text], dtype=np.float32)


LABELED_SET: list[LabeledQuery] = [
    {"query": "q_a",     "relevant_ids": ["a"]},
    {"query": "q_c_far", "relevant_ids": ["c"]},
]

# Expected per-query values (k_values=(1, 3)):
#
#   q_a (relevant=["a"], retrieved=[a, ?, ?]):
#     recall@1=1.0  recall@3=1.0  mrr=1.0  ndcg@1=1.0  ndcg@3=1.0
#
#   q_c_far (relevant=["c"], retrieved=[a, b, c]):
#     recall@1=0.0  recall@3=1.0  mrr=1/3  ndcg@1=0.0
#     ndcg@3: DCG=1/log2(4)=0.5, IDCG=1/log2(2)=1.0  → 0.5
#
#   Aggregated means:
#     recall@1=0.5  recall@3=1.0  mrr=2/3  ndcg@1=0.5  ndcg@3=0.75


# ── report structure ──────────────────────────────────────────────────────────

def test_report_contains_expected_keys() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    for key in ("recall@1", "recall@3", "ndcg@1", "ndcg@3", "mrr", "num_queries"):
        assert key in report, f"missing key: {key!r}"


def test_no_extra_k_keys_for_unrequested_k() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    assert "recall@5" not in report
    assert "ndcg@5" not in report


def test_num_queries() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    assert report["num_queries"] == 2


# ── aggregate metric values ───────────────────────────────────────────────────

def test_recall_at_1() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    assert report["recall@1"] == pytest.approx(0.5)


def test_recall_at_3_perfect() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    assert report["recall@3"] == pytest.approx(1.0)


def test_mrr() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    # q_a: mrr=1.0;  q_c_far: mrr=1/3  →  mean = 2/3
    assert report["mrr"] == pytest.approx(2.0 / 3.0)


def test_ndcg_at_1() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    assert report["ndcg@1"] == pytest.approx(0.5)


def test_ndcg_at_3() -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    # q_c_far: DCG=1/log2(4), IDCG=1/log2(2) → nDCG=0.5;  q_a: 1.0  →  mean=0.75
    assert report["ndcg@3"] == pytest.approx(0.75)


# ── single perfect query ──────────────────────────────────────────────────────

def test_single_query_perfect_scores() -> None:
    report = evaluate(
        build_index(),
        [{"query": "q_a", "relevant_ids": ["a"]}],
        fake_embed,
        k_values=(1, 3),
    )
    assert report["recall@1"] == pytest.approx(1.0)
    assert report["recall@3"] == pytest.approx(1.0)
    assert report["mrr"] == pytest.approx(1.0)
    assert report["ndcg@1"] == pytest.approx(1.0)
    assert report["ndcg@3"] == pytest.approx(1.0)


# ── k larger than index size ──────────────────────────────────────────────────

def test_k_larger_than_index_does_not_crash() -> None:
    report = evaluate(
        build_index(),
        [{"query": "q_a", "relevant_ids": ["a"]}],
        fake_embed,
        k_values=(1, 50),
    )
    # Only 3 segments in index; recall@50 should equal recall@3
    assert report["recall@50"] == pytest.approx(1.0)


# ── reranker injection ────────────────────────────────────────────────────────

def test_reranker_is_applied() -> None:
    """A reranker that reverses results should change MRR."""

    def reverse(results: list[SearchResult], query: str) -> list[SearchResult]:
        return list(reversed(results))

    # Without reranker: q_a → [a, ?, ?] → a at rank 1 → mrr=1.0
    # With reversal:    q_a → [?, ?, a] → a at rank 3 → mrr=1/3
    report = evaluate(
        build_index(),
        [{"query": "q_a", "relevant_ids": ["a"]}],
        fake_embed,
        k_values=(1, 3),
        reranker=reverse,
    )
    assert report["mrr"] == pytest.approx(1.0 / 3.0)


def test_reranker_none_baseline_unchanged() -> None:
    """Passing reranker=None must give the same result as omitting it."""
    idx = build_index()
    r1 = evaluate(idx, LABELED_SET, fake_embed, k_values=(1, 3), reranker=None)
    r2 = evaluate(idx, LABELED_SET, fake_embed, k_values=(1, 3))
    assert r1["mrr"] == pytest.approx(r2["mrr"])


# ── empty labeled set guard ───────────────────────────────────────────────────

def test_empty_labeled_set_raises() -> None:
    with pytest.raises(ValueError):
        evaluate(build_index(), [], fake_embed, k_values=(1,))


# ── dataset I/O ──────────────────────────────────────────────────────────────

def test_load_labeled_set_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    data = [
        {"query": "bicycle kick", "relevant_ids": ["seg_01", "seg_07"]},
        {"query": "penalty save",  "relevant_ids": ["seg_03"]},
    ]
    p = tmp_path / "eval.json"
    p.write_text(json.dumps(data), encoding="utf-8")

    loaded = load_labeled_set(p)
    assert len(loaded) == 2
    assert loaded[0]["query"] == "bicycle kick"
    assert loaded[0]["relevant_ids"] == ["seg_01", "seg_07"]
    assert loaded[1]["relevant_ids"] == ["seg_03"]


def test_load_labeled_set_rejects_non_array(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"query": "oops"}), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON array"):
        load_labeled_set(p)


def test_example_json_is_valid_labeled_set() -> None:
    """Smoke-test the committed example file."""
    from pathlib import Path
    example = Path(__file__).resolve().parents[1] / "data" / "eval" / "example.json"
    records = load_labeled_set(example)
    assert len(records) >= 1
    for rec in records:
        assert "query" in rec and isinstance(rec["query"], str)
        assert "relevant_ids" in rec and isinstance(rec["relevant_ids"], list)


# ── print_report smoke test ───────────────────────────────────────────────────

def test_print_report_contains_key_metrics(capsys: pytest.CaptureFixture) -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    print_report(report)
    out = capsys.readouterr().out
    assert "recall@1" in out
    assert "recall@3" in out
    assert "ndcg@1" in out
    assert "ndcg@3" in out
    assert "mrr" in out


def test_print_report_shows_query_count(capsys: pytest.CaptureFixture) -> None:
    report = evaluate(build_index(), LABELED_SET, fake_embed, k_values=(1, 3))
    print_report(report)
    out = capsys.readouterr().out
    assert "2" in out  # num_queries
