from app.models import SearchResult


def rerank(results: list[SearchResult], query: str, k: int | None = None) -> list[SearchResult]:
    """Re-score and sort results using a cross-encoder model, returning top-k."""
    raise NotImplementedError
