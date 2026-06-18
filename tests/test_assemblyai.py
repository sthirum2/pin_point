"""Unit tests for _utterances_to_segments — no network, no API key required."""
from dataclasses import dataclass

import pytest

from app.ingest.assemblyai import _utterances_to_segments, _video_id_from_source


@dataclass
class FakeUtterance:
    """Minimal stand-in for aai.Utterance — satisfies the _Utterance Protocol."""
    start: int   # milliseconds
    end: int     # milliseconds
    text: str
    speaker: str


# ── _video_id_from_source ────────────────────────────────────────────────────

def test_video_id_from_local_path() -> None:
    assert _video_id_from_source("/data/interview.mp4") == "interview"


def test_video_id_from_url() -> None:
    assert _video_id_from_source("https://example.com/clips/match.mp3") == "match"


def test_video_id_from_url_no_extension() -> None:
    assert _video_id_from_source("https://cdn.example.com/audio/clip42") == "clip42"


# ── _utterances_to_segments ──────────────────────────────────────────────────

UTTERANCES = [
    FakeUtterance(start=0,      end=4500,  text="Hello world.",    speaker="A"),
    FakeUtterance(start=5000,   end=9800,  text="How are you?",    speaker="B"),
    FakeUtterance(start=10200,  end=14000, text="I am doing well.", speaker="A"),
]


def test_segment_count() -> None:
    segments = _utterances_to_segments(UTTERANCES, "clip")
    assert len(segments) == 3


def test_segment_ids_are_sequential() -> None:
    segments = _utterances_to_segments(UTTERANCES, "clip")
    assert [s.id for s in segments] == ["clip_0", "clip_1", "clip_2"]


def test_video_id_propagated() -> None:
    segments = _utterances_to_segments(UTTERANCES, "myvideo")
    assert all(s.video_id == "myvideo" for s in segments)


def test_milliseconds_converted_to_seconds() -> None:
    segments = _utterances_to_segments(UTTERANCES, "clip")
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].end == pytest.approx(4.5)
    assert segments[1].start == pytest.approx(5.0)
    assert segments[1].end == pytest.approx(9.8)


def test_text_preserved() -> None:
    segments = _utterances_to_segments(UTTERANCES, "clip")
    assert segments[0].text == "Hello world."
    assert segments[2].text == "I am doing well."


def test_speaker_in_metadata() -> None:
    segments = _utterances_to_segments(UTTERANCES, "clip")
    assert segments[0].metadata["speaker"] == "A"
    assert segments[1].metadata["speaker"] == "B"


def test_empty_utterances_returns_empty_list() -> None:
    assert _utterances_to_segments([], "clip") == []


def test_single_utterance() -> None:
    u = FakeUtterance(start=1000, end=3000, text="Just one.", speaker="A")
    segments = _utterances_to_segments([u], "solo")
    assert len(segments) == 1
    assert segments[0].id == "solo_0"
    assert segments[0].start == pytest.approx(1.0)
    assert segments[0].end == pytest.approx(3.0)


def test_zero_duration_utterance() -> None:
    u = FakeUtterance(start=2000, end=2000, text="[noise]", speaker="A")
    segments = _utterances_to_segments([u], "clip")
    assert segments[0].start == pytest.approx(segments[0].end)
