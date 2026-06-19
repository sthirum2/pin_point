#!/usr/bin/env python3
"""Show bi-encoder retrieval order vs. cross-encoder reranked order on sample segments.

Usage:
    python scripts/try_rerank.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Segment
from app.retrieval.embed import embed_query, embed_segments
from app.retrieval.index import SegmentIndex
from app.retrieval.rerank import CrossEncoderReranker

_SEGMENTS = [
    Segment(id="s0", video_id="demo", start=0.0,  end=5.0,
            text="The goalkeeper makes a stunning save in the final minute."),
    Segment(id="s1", video_id="demo", start=5.0,  end=10.0,
            text="A bicycle kick goal from outside the penalty box."),
    Segment(id="s2", video_id="demo", start=10.0, end=15.0,
            text="The referee shows a red card after a dangerous tackle."),
    Segment(id="s3", video_id="demo", start=15.0, end=20.0,
            text="The striker converts a penalty kick in stoppage time."),
    Segment(id="s4", video_id="demo", start=20.0, end=25.0,
            text="A header from a corner kick rattles the crossbar."),
]

_QUERY = "goalkeeper saves the ball"


def _fmt(r: object) -> str:
    return f"  [{r.score:+.4f}]  {r.segment.id}  {r.segment.text[:60]}"


def main() -> None:
    print("Embedding segments …", flush=True)
    embed_segments(_SEGMENTS)

    idx = SegmentIndex()
    idx.build(_SEGMENTS)

    q_vec = embed_query(_QUERY)
    candidates = idx.query(q_vec, k=len(_SEGMENTS))

    print(f"\nQuery: {_QUERY!r}")
    print("\nBi-encoder order:")
    for r in candidates:
        print(_fmt(r))

    print("\nLoading cross-encoder …", flush=True)
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(_QUERY, candidates)

    print("\nCross-encoder order:")
    for r in reranked:
        print(_fmt(r))


if __name__ == "__main__":
    main()
