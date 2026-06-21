#!/usr/bin/env python3
"""Export segments from a saved index to a human-readable text file for labeling.

Usage:
    python scripts/export_segments.py <index-dir> <out.txt>

Output format (one line per segment):
    segment_id | [start-end] | speaker | text
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Segment
from app.retrieval.index import SegmentIndex


def format_segment(seg: Segment) -> str:
    """Format a Segment as a single pipe-delimited line for human review."""
    speaker = seg.metadata.get("speaker", "?")
    return f"{seg.id} | [{seg.start:.2f}-{seg.end:.2f}] | {speaker} | {seg.text}"


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/export_segments.py <index-dir> <out.txt>",
            file=sys.stderr,
        )
        sys.exit(1)

    index_dir, out_path = sys.argv[1], sys.argv[2]

    print(f"Loading index from {index_dir!r} …", flush=True)
    idx = SegmentIndex.load(index_dir)

    segs = idx.segments
    print(f"  {len(segs)} segments loaded.", flush=True)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for seg in segs:
            f.write(format_segment(seg) + "\n")

    print(f"Wrote {len(segs)} lines to {out_path!r}.")


if __name__ == "__main__":
    main()
