from app.models import Segment


def save_segment(segment: Segment) -> None:
    """Upsert a segment document into MongoDB."""
    raise NotImplementedError


def get_segment(segment_id: str) -> Segment:
    """Fetch a segment by id; raises KeyError if not found."""
    raise NotImplementedError


def list_segments(video_id: str) -> list[Segment]:
    """Return all segments belonging to a given video."""
    raise NotImplementedError


def delete_video(video_id: str) -> int:
    """Delete all segments for a video; returns count of deleted documents."""
    raise NotImplementedError
