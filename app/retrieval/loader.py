import functools
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np

from app.retrieval.index import SegmentIndex
from app.retrieval.rerank import CrossEncoderReranker

_DEFAULT_INDEX_DIR = "data/index/demo"
_SEGMENTS_FILE = "segments.pkl"


@functools.lru_cache(maxsize=1)
def get_index() -> SegmentIndex | None:
    """Load and cache the SegmentIndex; returns None when no saved index exists."""
    index_dir = Path(os.getenv("INDEX_DIR", _DEFAULT_INDEX_DIR))
    if not (index_dir / _SEGMENTS_FILE).exists():
        return None
    return SegmentIndex.load(index_dir)


def get_embed_fn() -> Callable[[str], np.ndarray]:
    """Return embed_query (deferred import keeps sentence-transformers off the import path)."""
    from app.retrieval.embed import embed_query
    return embed_query


def get_reranker() -> CrossEncoderReranker:
    """Return a CrossEncoderReranker (model loads lazily on the first score call)."""
    return CrossEncoderReranker()
