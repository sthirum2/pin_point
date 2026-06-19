import dataclasses
import functools
from collections.abc import Callable

from sentence_transformers import CrossEncoder

from app.models import SearchResult

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ScorerFn: receives a list of (query, passage) pairs, returns a score per pair.
type ScorerFn = Callable[[list[tuple[str, str]]], list[float]]


@functools.lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


class CrossEncoderReranker:
    """Re-scores retrieval candidates with a cross-encoder and re-sorts by that score.

    Pass `scorer` to inject a fake scoring function for offline testing — when
    omitted, the real CrossEncoder model is used (loaded and cached on first call).

    The `top_n` set at construction is the default for `__call__`; it can be
    overridden per-call via `rerank(query, results, top_n=...)`.
    """

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        top_n: int | None = None,
        scorer: ScorerFn | None = None,
    ) -> None:
        self._model_name = model_name
        self._top_n = top_n
        self._scorer = scorer

    # ── internal scoring ──────────────────────────────────────────────────────

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._scorer is not None:
            return self._scorer(pairs)
        raw = _load_cross_encoder(self._model_name).predict(pairs)
        return raw.tolist()

    # ── public rerank method ──────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_n: int | None = None,
    ) -> list[SearchResult]:
        """Score every (query, segment.text) pair and return results sorted by score.

        Creates new SearchResult objects so the originals are never mutated.
        `top_n` overrides the instance default when provided.
        """
        if not results:
            return []

        pairs = [(query, r.segment.text) for r in results]
        scores = self._score(pairs)

        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)

        limit = top_n if top_n is not None else (self._top_n or len(results))
        return [
            dataclasses.replace(result, score=score, rank=rank)
            for rank, (score, result) in enumerate(ranked[:limit])
        ]

    # ── callable hook (matches run_eval.RerankerFn) ───────────────────────────

    def __call__(self, results: list[SearchResult], query: str) -> list[SearchResult]:
        """Callable interface: (results, query) → matches RerankerFn in run_eval.py."""
        return self.rerank(query, results)
