from pathlib import Path
from typing import Protocol

import assemblyai as aai

from app.config import get_settings
from app.models import Segment


class _Utterance(Protocol):
    """Structural type for anything with the utterance fields we need."""
    start: int
    end: int
    text: str
    speaker: str


def _utterances_to_segments(
    utterances: list[_Utterance],
    video_id: str,
) -> list[Segment]:
    """Pure mapping from utterance objects to Segments — no network, fully testable."""
    return [
        Segment(
            id=f"{video_id}_{i}",
            video_id=video_id,
            start=u.start / 1000.0,
            end=u.end / 1000.0,
            text=u.text,
            metadata={"speaker": u.speaker},
        )
        for i, u in enumerate(utterances)
    ]


def _video_id_from_source(source: str) -> str:
    """Derive a stable video_id from a file path stem or URL's final path component."""
    if source.startswith(("http://", "https://")):
        return Path(source.rstrip("/").split("/")[-1]).stem
    return Path(source).stem


def transcribe(source: str) -> list[Segment]:
    """Transcribe an audio/video file (local path or URL) with speaker diarization.

    Raises RuntimeError if the transcription job fails.
    """
    aai.settings.api_key = get_settings().assemblyai_api_key

    config = aai.TranscriptionConfig(speaker_labels=True)
    transcript = aai.Transcriber().transcribe(source, config)

    if transcript.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")

    utterances = transcript.utterances or []
    return _utterances_to_segments(utterances, _video_id_from_source(source))
