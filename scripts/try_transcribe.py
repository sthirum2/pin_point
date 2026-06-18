#!/usr/bin/env python3
"""Quick CLI to transcribe a file and print speaker-labeled segments.

Usage:
    python scripts/try_transcribe.py <file-path-or-url>
"""
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingest.assemblyai import transcribe


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_transcribe.py <file-path-or-url>", file=sys.stderr)
        sys.exit(1)

    source = sys.argv[1]
    print(f"Transcribing {source!r} …", flush=True)

    segments = transcribe(source)

    if not segments:
        print("No segments returned.")
        return

    for seg in segments:
        speaker = seg.metadata.get("speaker", "?")
        print(f"[{seg.start:.2f}-{seg.end:.2f}] {speaker}: {seg.text}")


if __name__ == "__main__":
    main()
