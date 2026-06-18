from dataclasses import dataclass, field

import numpy as np


@dataclass
class Segment:
    id: str
    video_id: str
    start: float
    end: float
    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32),
        compare=False,
        repr=False,
    )


@dataclass
class SearchResult:
    segment: Segment
    score: float
    rank: int = 0
