from collections.abc import Callable

import numpy as np
from fastapi import Depends, FastAPI, HTTPException

from app.models import SearchResult
from app.retrieval.index import SegmentIndex
from app.retrieval.loader import get_embed_fn, get_index, get_reranker
from app.retrieval.rerank import CrossEncoderReranker
from app.telemetry import log_retrieval

app = FastAPI(title="Video Search API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str,
    k: int = 5,
    rerank: bool = False,
    index: SegmentIndex | None = Depends(get_index),
    embed_fn: Callable[[str], np.ndarray] = Depends(get_embed_fn),
    reranker: CrossEncoderReranker = Depends(get_reranker),
) -> list[dict]:
    if index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not available — run scripts/build_index.py first.",
        )

    vec = embed_fn(q)
    results: list[SearchResult] = index.query(vec, k=k)

    if rerank:
        results = reranker(results, q)

    log_retrieval(q, results, reranked=rerank, k=k)

    return [
        {
            "segment_id": r.segment.id,
            "start": r.segment.start,
            "end": r.segment.end,
            "speaker": r.segment.metadata.get("speaker"),
            "text": r.segment.text,
            "score": r.score,
        }
        for r in results
    ]
