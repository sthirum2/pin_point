"""Offline tests for telemetry logging and log analysis — no network/model required."""
import json
from pathlib import Path

import pytest

from app.models import SearchResult, Segment
from app.telemetry import log_retrieval
from scripts.analyze_logs import analyze


# ── helpers ───────────────────────────────────────────────────────────────────

def make_result(seg_id: str, score: float) -> SearchResult:
    s = Segment(id=seg_id, video_id="v0", start=0.0, end=1.0, text=seg_id)
    return SearchResult(segment=s, score=score)


@pytest.fixture()
def log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / "retrieval.jsonl"
    monkeypatch.setenv("LOG_PATH", str(p))
    return p


# ── log_retrieval: one JSON line per call ─────────────────────────────────────

def test_log_creates_one_line_per_call(log_file: Path) -> None:
    log_retrieval("hello", [make_result("a", 0.9), make_result("b", 0.5)], reranked=False, k=5)
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1


def test_log_multiple_calls_produce_multiple_lines(log_file: Path) -> None:
    log_retrieval("q1", [make_result("a", 0.9)], reranked=False, k=1)
    log_retrieval("q2", [make_result("b", 0.7)], reranked=False, k=1)
    lines = [l for l in log_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2


# ── log_retrieval: required fields ───────────────────────────────────────────

def test_log_top_level_fields(log_file: Path) -> None:
    log_retrieval("test query", [make_result("x", 0.8)], reranked=True, k=3)
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["query"] == "test query"
    assert record["k"] == 3
    assert record["reranked"] is True
    assert "timestamp" in record
    assert "results" in record


def test_log_result_fields(log_file: Path) -> None:
    log_retrieval("q", [make_result("seg1", 0.75), make_result("seg2", 0.4)], reranked=False, k=2)
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert len(record["results"]) == 2
    for entry in record["results"]:
        assert "segment_id" in entry
        assert "score" in entry
        assert "rank" in entry


def test_log_ranks_are_1_indexed(log_file: Path) -> None:
    log_retrieval("q", [make_result("a", 0.9), make_result("b", 0.5)], reranked=False, k=2)
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["results"][0]["rank"] == 1
    assert record["results"][1]["rank"] == 2


def test_log_segment_ids_match(log_file: Path) -> None:
    results = [make_result("alpha", 0.9), make_result("beta", 0.3)]
    log_retrieval("q", results, reranked=False, k=2)
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["results"][0]["segment_id"] == "alpha"
    assert record["results"][1]["segment_id"] == "beta"


# ── log_retrieval: dir creation ───────────────────────────────────────────────

def test_log_creates_nested_dir_if_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "nested" / "deep" / "retrieval.jsonl"
    monkeypatch.setenv("LOG_PATH", str(p))
    log_retrieval("q", [make_result("a", 0.5)], reranked=False, k=1)
    assert p.exists()


# ── log_retrieval: non-fatal on error ─────────────────────────────────────────

def test_log_does_not_raise_on_bad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_PATH", "\\\\invalid\\\\??path\\\\nul")
    # Should not raise
    log_retrieval("q", [make_result("a", 0.5)], reranked=False, k=1)


# ── analyze: total queries and threshold flagging ─────────────────────────────

def _write_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _make_log_record(query: str, top_score: float) -> dict:
    return {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "query": query,
        "k": 5,
        "reranked": False,
        "results": [{"segment_id": "s1", "score": top_score, "rank": 1}],
    }


def test_analyze_total_queries(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [_make_log_record("q1", 0.9), _make_log_record("q2", 0.3)])
    stats = analyze(log, threshold=0.5)
    assert stats["total_queries"] == 2


def test_analyze_mean_top1_score(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [_make_log_record("q1", 0.8), _make_log_record("q2", 0.4)])
    stats = analyze(log, threshold=0.5)
    assert abs(stats["mean_top1_score"] - 0.6) < 1e-9


def test_analyze_median_top1_score(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [
        _make_log_record("q1", 0.9),
        _make_log_record("q2", 0.5),
        _make_log_record("q3", 0.1),
    ])
    stats = analyze(log, threshold=0.5)
    assert abs(stats["median_top1_score"] - 0.5) < 1e-9


def test_analyze_below_threshold_count(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [
        _make_log_record("q1", 0.9),  # above
        _make_log_record("q2", 0.3),  # below
        _make_log_record("q3", 0.1),  # below
    ])
    stats = analyze(log, threshold=0.5)
    assert stats["below_threshold_count"] == 2


def test_analyze_zero_below_threshold_when_all_pass(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [_make_log_record("q1", 0.8), _make_log_record("q2", 0.9)])
    stats = analyze(log, threshold=0.5)
    assert stats["below_threshold_count"] == 0


def test_analyze_all_below_threshold(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [_make_log_record("q1", 0.1), _make_log_record("q2", 0.2)])
    stats = analyze(log, threshold=0.5)
    assert stats["below_threshold_count"] == 2


def test_analyze_custom_threshold(tmp_path: Path) -> None:
    log = tmp_path / "retrieval.jsonl"
    _write_log(log, [
        _make_log_record("q1", 0.95),
        _make_log_record("q2", 0.85),
        _make_log_record("q3", 0.75),
    ])
    stats = analyze(log, threshold=0.9)
    assert stats["below_threshold_count"] == 2
