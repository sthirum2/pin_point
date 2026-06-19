"""Offline tests for CrossEncoderReranker — no model download, no network."""
import dataclasses

import numpy as np
import pytest

from app.eval.dataset import LabeledQuery
from app.eval.run_eval import evaluate
from app.models import Segment, SearchResult
from app.retrieval.index import SegmentIndex
from app.retrieval.rerank import CrossEncoderReranker


# ── helpers ───────────────────────────────────────────────────────────────────

def make_result(seg_id: str, text: str, score: float = 0.5, rank: int = 0) -> SearchResult:
    seg = Segment(id=seg_id, video_id="v0", start=0.0, end=1.0, text=text)
    return SearchResult(segment=seg, score=score, rank=rank)


def fake_scorer(pairs: list[tuple[str, str]]) -> list[float]:
    """Score by keyword: 'best'→10, 'good'→5, otherwise 1."""
    out: list[float] = []
    for _q, text in pairs:
        if "best" in text:
            out.append(10.0)
        elif "good" in text:
            out.append(5.0)
        else:
            out.append(1.0)
    return out


# Three candidates in bi-encoder order: low, high, mid.
_CANDIDATES = [
    make_result("low",  "bad result",       score=0.9, rank=0),
    make_result("high", "best result ever", score=0.5, rank=1),
    make_result("mid",  "good result",      score=0.3, rank=2),
]


@pytest.fixture
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(scorer=fake_scorer)


# ── sorting ───────────────────────────────────────────────────────────────────

def test_rerank_sorts_by_cross_encoder_score(reranker: CrossEncoderReranker) -> None:
    results = reranker.rerank("q", list(_CANDIDATES))
    assert [r.segment.id for r in results] == ["high", "mid", "low"]


def test_rerank_sets_cross_encoder_scores(reranker: CrossEncoderReranker) -> None:
    results = reranker.rerank("q", list(_CANDIDATES))
    scores = {r.segment.id: r.score for r in results}
    assert scores["high"] == pytest.approx(10.0)
    assert scores["mid"]  == pytest.approx(5.0)
    assert scores["low"]  == pytest.approx(1.0)


def test_rerank_updates_rank_field(reranker: CrossEncoderReranker) -> None:
    results = reranker.rerank("q", list(_CANDIDATES))
    assert [r.rank for r in results] == [0, 1, 2]


# ── top_n ────────────────────────────────────────────────────────────────────

def test_top_n_arg_truncates(reranker: CrossEncoderReranker) -> None:
    results = reranker.rerank("q", list(_CANDIDATES), top_n=2)
    assert len(results) == 2
    assert results[0].segment.id == "high"
    assert results[1].segment.id == "mid"


def test_top_n_larger_than_input_returns_all(reranker: CrossEncoderReranker) -> None:
    results = reranker.rerank("q", list(_CANDIDATES), top_n=100)
    assert len(results) == 3


def test_top_n_from_init() -> None:
    r = CrossEncoderReranker(scorer=fake_scorer, top_n=1)
    results = r.rerank("q", list(_CANDIDATES))
    assert len(results) == 1
    assert results[0].segment.id == "high"


def test_top_n_arg_overrides_init_top_n() -> None:
    r = CrossEncoderReranker(scorer=fake_scorer, top_n=1)
    results = r.rerank("q", list(_CANDIDATES), top_n=2)
    assert len(results) == 2


# ── empty input ───────────────────────────────────────────────────────────────

def test_empty_results_returns_empty(reranker: CrossEncoderReranker) -> None:
    assert reranker.rerank("q", []) == []


# ── callable interface (RerankerFn) ───────────────────────────────────────────

def test_call_signature_is_results_then_query(reranker: CrossEncoderReranker) -> None:
    """__call__(results, query) must match RerankerFn."""
    results = reranker(list(_CANDIDATES), "q")
    assert results[0].segment.id == "high"


def test_call_uses_init_top_n() -> None:
    r = CrossEncoderReranker(scorer=fake_scorer, top_n=1)
    results = r(list(_CANDIDATES), "q")
    assert len(results) == 1
    assert results[0].segment.id == "high"


# ── immutability ──────────────────────────────────────────────────────────────

def test_does_not_mutate_input_list_order(reranker: CrossEncoderReranker) -> None:
    original = list(_CANDIDATES)
    original_ids = [r.segment.id for r in original]
    reranker.rerank("q", original)
    assert [r.segment.id for r in original] == original_ids


def test_does_not_mutate_original_scores(reranker: CrossEncoderReranker) -> None:
    original = list(_CANDIDATES)
    old_scores = [r.score for r in original]
    reranker.rerank("q", original)
    assert [r.score for r in original] == old_scores


# ── integration with evaluate() ──────────────────────────────────────────────

def _make_seg(seg_id: str, vec: list[float], text: str) -> Segment:
    s = Segment(id=seg_id, video_id="v", start=0.0, end=1.0, text=text)
    s.embedding = np.array(vec, dtype=np.float32)
    return s


def test_reranker_corrects_biencoder_ordering() -> None:
    """Cross-encoder should surface 'best' at rank 1 even when bi-encoder ranks it second."""
    # bi-encoder embeds: "low" is closer to the query vector than "best"
    segs = [
        _make_seg("best", [0.0, 1.0], "best result ever"),
        _make_seg("low",  [1.0, 0.0], "bad result"),
    ]
    idx = SegmentIndex()
    idx.build(segs)

    def fake_embed(_: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)  # cosine-closest to "low"

    labeled: list[LabeledQuery] = [{"query": "q", "relevant_ids": ["best"]}]

    # k_values=(1, 2) so max_k=2 — both candidates are fetched; recall@1 tests top slot only.
    # Without reranker: bi-encoder order is ["low", "best"] → recall@1 = 0
    report_baseline = evaluate(idx, labeled, fake_embed, k_values=(1, 2))
    assert report_baseline["recall@1"] == pytest.approx(0.0)

    # With reranker: cross-encoder reorders to ["best", "low"] → recall@1 = 1
    r = CrossEncoderReranker(scorer=fake_scorer)
    report_reranked = evaluate(idx, labeled, fake_embed, k_values=(1, 2), reranker=r)
    assert report_reranked["recall@1"] == pytest.approx(1.0)


def test_reranker_callable_signature_in_evaluate() -> None:
    """evaluate() calls reranker(results, query) — verify __call__ wiring is correct."""
    segs = [
        _make_seg("best", [1.0, 0.0], "best result ever"),
        _make_seg("low",  [0.9, 0.1], "bad result"),
    ]
    idx = SegmentIndex()
    idx.build(segs)

    def fake_embed(_: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    labeled: list[LabeledQuery] = [{"query": "q", "relevant_ids": ["best"]}]
    r = CrossEncoderReranker(scorer=fake_scorer)
    report = evaluate(idx, labeled, fake_embed, k_values=(1,), reranker=r)
    assert report["recall@1"] == pytest.approx(1.0)
