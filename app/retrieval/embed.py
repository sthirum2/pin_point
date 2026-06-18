import functools

import numpy as np
from sentence_transformers import SentenceTransformer

from app.models import Segment

_MODEL_NAME = "all-MiniLM-L6-v2"


@functools.lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(_MODEL_NAME)


def embed_query(text: str) -> np.ndarray:
    """Return a single L2-normalized float32 embedding for the given query text."""
    vec: np.ndarray = _get_model().encode(
        [text], normalize_embeddings=True, show_progress_bar=False
    )
    return vec[0].astype(np.float32)


def embed_segments(segments: list[Segment]) -> None:
    """Fill each segment's .embedding in-place with an L2-normalized float32 vector."""
    if not segments:
        return
    texts = [s.text for s in segments]
    vecs: np.ndarray = _get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
    )
    for seg, vec in zip(segments, vecs):
        seg.embedding = vec.astype(np.float32)
