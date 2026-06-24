import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.models import SearchResult

_DEFAULT_LOG_PATH = Path("data/logs/retrieval.jsonl")


def _log_path() -> Path:
    return Path(os.environ.get("LOG_PATH", _DEFAULT_LOG_PATH))


def log_retrieval(
    query: str,
    results: list[SearchResult],
    reranked: bool,
    k: int,
) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "k": k,
            "reranked": reranked,
            "results": [
                {
                    "segment_id": r.segment.id,
                    "score": float(r.score),
                    "rank": rank,
                }
                for rank, r in enumerate(results, start=1)
            ],
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
