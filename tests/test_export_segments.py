"""Offline tests for export_segments.format_segment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_segments import format_segment
from app.models import Segment


def _seg(
    seg_id: str = "v0_0",
    start: float = 1.5,
    end: float = 5.25,
    text: str = "Hello world",
    speaker: str | None = "A",
) -> Segment:
    meta: dict[str, object] = {}
    if speaker is not None:
        meta["speaker"] = speaker
    return Segment(id=seg_id, video_id="v", start=start, end=end, text=text, metadata=meta)


def test_format_contains_all_fields() -> None:
    line = format_segment(_seg())
    assert "v0_0" in line
    assert "[1.50-5.25]" in line
    assert "A" in line
    assert "Hello world" in line


def test_format_uses_pipe_delimiter() -> None:
    parts = format_segment(_seg()).split(" | ")
    assert len(parts) == 4


def test_field_order() -> None:
    parts = format_segment(_seg(seg_id="x1", start=2.0, end=4.0, text="hi", speaker="B")).split(" | ")
    assert parts[0] == "x1"
    assert parts[1] == "[2.00-4.00]"
    assert parts[2] == "B"
    assert parts[3] == "hi"


def test_missing_speaker_defaults_to_question_mark() -> None:
    line = format_segment(_seg(speaker=None))
    parts = line.split(" | ")
    assert parts[2] == "?"


def test_time_formatted_to_two_decimal_places() -> None:
    line = format_segment(_seg(start=0.0, end=12.345))
    assert "[0.00-12.35]" in line


def test_segments_property_returns_copy() -> None:
    """SegmentIndex.segments should return a fresh list each time."""
    import numpy as np
    from app.retrieval.index import SegmentIndex

    s = Segment(id="a", video_id="v", start=0.0, end=1.0, text="t")
    s.embedding = np.array([1.0, 0.0], dtype=np.float32)
    idx = SegmentIndex()
    idx.build([s])

    first = idx.segments
    first.append(s)  # mutate the returned copy
    assert len(idx.segments) == 1  # internal list is unchanged
