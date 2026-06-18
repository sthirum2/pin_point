from app.models import Segment


def create_index(index_name: str) -> str:
    """Create a Twelve Labs index and return its index_id."""
    raise NotImplementedError


def upload_video(index_id: str, video_url: str) -> str:
    """Upload a video to an index and return its video_id."""
    raise NotImplementedError


def search(query: str, index_id: str, k: int = 5) -> list[Segment]:
    """Run a semantic search against a Twelve Labs index."""
    raise NotImplementedError
