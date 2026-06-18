#!/usr/bin/env python3
"""Transcribe a file, embed segments, build a FAISS index, and search.

Usage:
    python scripts/try_search.py <audio-file-or-url> <query>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.assemblyai import transcribe
from app.retrieval.embed import embed_query, embed_segments
from app.retrieval.index import SegmentIndex


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/try_search.py <audio-file-or-url> <query>", file=sys.stderr)
        sys.exit(1)

    source, query = sys.argv[1], sys.argv[2]

    print(f"Transcribing {source!r} …", flush=True)
    segments = transcribe(source)
    print(f"  {len(segments)} segments returned.", flush=True)

    print("Embedding segments …", flush=True)
    embed_segments(segments)

    print("Building index …", flush=True)
    idx = SegmentIndex()
    idx.build(segments)

    query_vec = embed_query(query)
    results = idx.query(query_vec, k=5)

    print(f"\nTop results for {query!r}:")
    for r in results:
        seg = r.segment
        print(f"  [{r.score:.4f}] [{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}")


if __name__ == "__main__":
    main()
