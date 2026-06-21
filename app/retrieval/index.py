import pickle
from pathlib import Path

import faiss
import numpy as np

from app.models import Segment, SearchResult

_INDEX_FILE = "index.faiss"
_SEGMENTS_FILE = "segments.pkl"


class SegmentIndex:
    """FAISS IndexFlatIP (inner-product / cosine on normalized vectors) over Segments."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._segments: list[Segment] = []

    # ── build ────────────────────────────────────────────────────────────────

    def build(self, segments: list[Segment]) -> None:
        """Populate the index from a list of already-embedded Segments."""
        if not segments:
            raise ValueError("Cannot build index from an empty segment list.")
        matrix = np.stack([s.embedding for s in segments]).astype(np.float32)
        dim = matrix.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(matrix)
        self._segments = list(segments)

    @property
    def segments(self) -> list[Segment]:
        """Return a copy of the indexed segments in build order."""
        return list(self._segments)

    # ── query ────────────────────────────────────────────────────────────────

    def query(self, query_vec: np.ndarray, k: int = 10) -> list[SearchResult]:
        """Return the top-k segments by cosine similarity (descending score)."""
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        vec = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(vec, k)
        return [
            SearchResult(segment=self._segments[idx], score=float(scores[0][rank]), rank=rank)
            for rank, idx in enumerate(indices[0])
            if idx >= 0
        ]

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the index and segment list to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(p / _INDEX_FILE))
        with open(p / _SEGMENTS_FILE, "wb") as f:
            pickle.dump(self._segments, f)

    @classmethod
    def load(cls, path: str | Path) -> "SegmentIndex":
        """Restore a previously saved SegmentIndex from a directory."""
        p = Path(path)
        obj = cls()
        index_path = p / _INDEX_FILE
        if index_path.exists():
            obj._index = faiss.read_index(str(index_path))
        with open(p / _SEGMENTS_FILE, "rb") as f:
            obj._segments = pickle.load(f)
        return obj
