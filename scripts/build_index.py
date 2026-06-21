#!/usr/bin/env python3
"""Transcribe a video/audio file, embed segments, and save a SegmentIndex.

Usage:
    python scripts/build_index.py <video-or-audio> <output-dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.assemblyai import transcribe
from app.retrieval.embed import embed_segments
from app.retrieval.index import SegmentIndex


def main() -> None:
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/build_index.py <video-or-audio> <output-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    source, output_dir = sys.argv[1], sys.argv[2]

    print(f"Transcribing {source!r} …", flush=True)
    segments = transcribe(source)
    print(f"  {len(segments)} segments from transcription.", flush=True)

    print("Embedding segments …", flush=True)
    embed_segments(segments)

    print("Building index …", flush=True)
    idx = SegmentIndex()
    idx.build(segments)

    print(f"Saving index to {output_dir!r} …", flush=True)
    idx.save(output_dir)

    print(f"\nDone. Indexed {len(segments)} segments → {output_dir}")


if __name__ == "__main__":
    main()
