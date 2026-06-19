import json
from pathlib import Path
from typing import TypedDict


class LabeledQuery(TypedDict):
    query: str
    relevant_ids: list[str]


def load_labeled_set(path: str | Path) -> list[LabeledQuery]:
    """Load a JSON array of {query, relevant_ids} records from disk."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {path!r}, got {type(data).__name__}")
    return data
