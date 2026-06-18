"""Offline tests for SegmentIndex — no network, no sentence-transformers model."""
import numpy as np
import pytest

from app.models import Segment, SearchResult
from app.retrieval.index import SegmentIndex


# ── helpers ──────────────────────────────────────────────────────────────────

def make_segment(seg_id: str, vec: list[float], text: str = "") -> Segment:
    s = Segment(id=seg_id, video_id="v0", start=0.0, end=1.0, text=text or seg_id)
    s.embedding = np.array(vec, dtype=np.float32)
    return s


def fresh_index() -> tuple[SegmentIndex, list[Segment]]:
    """Three orthogonal unit-vector segments."""
    segs = [
        make_segment("a", [1.0, 0.0, 0.0]),
        make_segment("b", [0.0, 1.0, 0.0]),
        make_segment("c", [0.0, 0.0, 1.0]),
    ]
    idx = SegmentIndex()
    idx.build(segs)
    return idx, segs


# ── build / query ─────────────────────────────────────────────────────────────

def test_query_returns_closest() -> None:
    idx, _ = fresh_index()
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert len(results) == 1
    assert results[0].segment.id == "a"


def test_query_exact_match_score_is_one() -> None:
    idx, _ = fresh_index()
    results = idx.query(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_query_orthogonal_score_is_zero() -> None:
    idx, _ = fresh_index()
    # Query along "a"; "b" and "c" are orthogonal → inner-product == 0
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3)
    scores = {r.segment.id: r.score for r in results}
    assert scores["b"] == pytest.approx(0.0, abs=1e-6)
    assert scores["c"] == pytest.approx(0.0, abs=1e-6)


def test_query_ranking_order() -> None:
    idx, _ = fresh_index()
    # Mixed query: closer to "a" than "b", "b" closer than "c"
    q = np.array([0.9, 0.4, 0.1], dtype=np.float32)
    results = idx.query(q, k=3)
    ids = [r.segment.id for r in results]
    assert ids[0] == "a"
    assert ids[1] == "b"
    assert ids[2] == "c"


def test_query_rank_field_sequential() -> None:
    idx, _ = fresh_index()
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3)
    assert [r.rank for r in results] == [0, 1, 2]


def test_query_k_limits_result_count() -> None:
    idx, _ = fresh_index()
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=2)
    assert len(results) == 2


def test_query_k_larger_than_index_clamps() -> None:
    idx, _ = fresh_index()
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=100)
    assert len(results) == 3  # only 3 segments in index


def test_query_empty_index_returns_empty() -> None:
    idx = SegmentIndex()
    results = idx.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
    assert results == []


def test_build_empty_raises() -> None:
    idx = SegmentIndex()
    with pytest.raises(ValueError):
        idx.build([])


# ── save / load round-trip ────────────────────────────────────────────────────

def test_save_load_returns_same_top_result(tmp_path: pytest.TempPathFactory) -> None:
    idx, _ = fresh_index()
    idx.save(tmp_path / "idx")
    loaded = SegmentIndex.load(tmp_path / "idx")
    results = loaded.query(np.array([0.0, 0.0, 1.0], dtype=np.float32), k=1)
    assert results[0].segment.id == "c"


def test_save_load_score_preserved(tmp_path: pytest.TempPathFactory) -> None:
    idx, _ = fresh_index()
    idx.save(tmp_path / "idx")
    loaded = SegmentIndex.load(tmp_path / "idx")
    results = loaded.query(np.array([0.0, 1.0, 0.0], dtype=np.float32), k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_save_load_preserves_metadata(tmp_path: pytest.TempPathFactory) -> None:
    seg = make_segment("x", [1.0, 0.0, 0.0])
    seg.metadata["speaker"] = "Alice"
    idx = SegmentIndex()
    idx.build([seg])
    idx.save(tmp_path / "idx")
    loaded = SegmentIndex.load(tmp_path / "idx")
    results = loaded.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)
    assert results[0].segment.metadata["speaker"] == "Alice"


def test_save_load_all_segments_present(tmp_path: pytest.TempPathFactory) -> None:
    idx, segs = fresh_index()
    idx.save(tmp_path / "idx")
    loaded = SegmentIndex.load(tmp_path / "idx")
    results = loaded.query(np.array([1.0, 1.0, 1.0], dtype=np.float32), k=10)
    returned_ids = {r.segment.id for r in results}
    assert returned_ids == {"a", "b", "c"}
