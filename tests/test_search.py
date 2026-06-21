"""Tests for GET /search — synthetic index, no model/network required."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import Segment
from app.retrieval.index import SegmentIndex
from app.retrieval.loader import get_embed_fn, get_index, get_reranker


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_index(*entries: tuple[str, list[float], str]) -> SegmentIndex:
    segs = []
    for i, (seg_id, vec, text) in enumerate(entries):
        s = Segment(
            id=seg_id, video_id="v0",
            start=float(i), end=float(i + 1),
            text=text, metadata={"speaker": f"S{i}"},
        )
        s.embedding = np.array(vec, dtype=np.float32)
        segs.append(s)
    idx = SegmentIndex()
    idx.build(segs)
    return idx


# Three segments with distinct inner products against [1, 0, 0]:
#   seg_a → 1.0   seg_b → 0.5   seg_c → 0.1
_IDX = _build_index(
    ("seg_a", [1.0, 0.0, 0.0], "alpha text"),
    ("seg_b", [0.5, 0.5, 0.0], "beta text"),
    ("seg_c", [0.1, 0.9, 0.0], "gamma text"),
)

# Embed always returns [1, 0, 0] → seg_a is always the top bi-encoder result.
_EMBED_A: np.ndarray = np.array([1.0, 0.0, 0.0], dtype=np.float32)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def client() -> TestClient:
    app.dependency_overrides[get_index] = lambda: _IDX
    app.dependency_overrides[get_embed_fn] = lambda: (lambda _: _EMBED_A)
    return TestClient(app)


# ── 503 when no index ─────────────────────────────────────────────────────────

def test_search_503_when_no_index() -> None:
    app.dependency_overrides[get_index] = lambda: None
    assert TestClient(app).get("/search", params={"q": "x"}).status_code == 503


def test_search_503_detail_mentions_index() -> None:
    app.dependency_overrides[get_index] = lambda: None
    detail = TestClient(app).get("/search", params={"q": "x"}).json()["detail"]
    assert "index" in detail.lower()


# ── response structure ────────────────────────────────────────────────────────

def test_search_returns_list(client: TestClient) -> None:
    assert isinstance(client.get("/search", params={"q": "q"}).json(), list)


def test_search_result_has_required_fields(client: TestClient) -> None:
    result = client.get("/search", params={"q": "q", "k": 1}).json()[0]
    for field in ("segment_id", "start", "end", "speaker", "text", "score"):
        assert field in result, f"missing field: {field!r}"


def test_search_speaker_in_result(client: TestClient) -> None:
    result = client.get("/search", params={"q": "q", "k": 1}).json()[0]
    assert result["speaker"] == "S0"  # seg_a's metadata


# ── retrieval correctness ─────────────────────────────────────────────────────

def test_search_top_result_is_closest(client: TestClient) -> None:
    """Embed always returns [1,0,0]; seg_a has embedding [1,0,0] → must be first."""
    top = client.get("/search", params={"q": "q", "k": 1}).json()[0]
    assert top["segment_id"] == "seg_a"


def test_search_k_limits_results(client: TestClient) -> None:
    for k in (1, 2, 3):
        results = client.get("/search", params={"q": "q", "k": k}).json()
        assert len(results) == k


def test_search_scores_descending(client: TestClient) -> None:
    scores = [r["score"] for r in client.get("/search", params={"q": "q", "k": 3}).json()]
    assert scores == sorted(scores, reverse=True)


# ── rerank flag ───────────────────────────────────────────────────────────────

def test_rerank_flag_invokes_reranker() -> None:
    """rerank=true calls the reranker; rerank=false does not."""
    called_with: list[str] = []

    def tracking_reranker(results, query):
        called_with.append(query)
        return results

    app.dependency_overrides[get_index] = lambda: _IDX
    app.dependency_overrides[get_embed_fn] = lambda: (lambda _: _EMBED_A)
    app.dependency_overrides[get_reranker] = lambda: tracking_reranker
    c = TestClient(app)

    c.get("/search", params={"q": "hello", "rerank": "false"})
    assert called_with == []

    c.get("/search", params={"q": "hello", "rerank": "true"})
    assert called_with == ["hello"]


def test_rerank_flag_changes_order() -> None:
    """A reranker that reverses results should move seg_a from first to last."""
    def reverse_reranker(results, query):
        return list(reversed(results))

    app.dependency_overrides[get_index] = lambda: _IDX
    app.dependency_overrides[get_embed_fn] = lambda: (lambda _: _EMBED_A)
    app.dependency_overrides[get_reranker] = lambda: reverse_reranker
    c = TestClient(app)

    # Without rerank: seg_a first.
    assert c.get("/search", params={"q": "q", "k": 3}).json()[0]["segment_id"] == "seg_a"

    # With rerank: reversed → seg_c first (lowest bi-encoder score), seg_a last.
    reranked = c.get("/search", params={"q": "q", "k": 3, "rerank": "true"}).json()
    assert reranked[0]["segment_id"] == "seg_c"
    assert reranked[-1]["segment_id"] == "seg_a"
