#!/usr/bin/env python3
"""Build a demo SegmentIndex from hard-coded factual sentences.

Saves to data/index/demo (the default path used by get_index()).

Usage:
    python scripts/build_demo_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models import Segment
from app.retrieval.embed import embed_segments
from app.retrieval.index import SegmentIndex

_OUTPUT_DIR = "data/index/demo"

_SEGMENTS: list[Segment] = [
    Segment(id="demo_01", video_id="demo", start=0.0, end=5.0,
            text="Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of sugar."),
    Segment(id="demo_02", video_id="demo", start=5.0, end=10.0,
            text="Chlorophyll, the green pigment in plant leaves, absorbs light energy and drives the chemical reactions of photosynthesis."),
    Segment(id="demo_03", video_id="demo", start=10.0, end=15.0,
            text="Gravity is a fundamental force that attracts objects with mass toward one another; on Earth it gives weight to physical objects."),
    Segment(id="demo_04", video_id="demo", start=15.0, end=20.0,
            text="Isaac Newton formulated the law of universal gravitation: every mass attracts every other mass with a force proportional to the product of their masses."),
    Segment(id="demo_05", video_id="demo", start=20.0, end=25.0,
            text="The water cycle describes the continuous movement of water through evaporation, condensation, precipitation, and collection back into bodies of water."),
    Segment(id="demo_06", video_id="demo", start=25.0, end=30.0,
            text="When liquid water is heated by the sun it evaporates into water vapor, rises into the atmosphere, and eventually cools to form clouds."),
    Segment(id="demo_07", video_id="demo", start=30.0, end=35.0,
            text="The French Revolution began in 1789 as a period of radical political and social transformation in France, overthrowing the monarchy and establishing a republic."),
    Segment(id="demo_08", video_id="demo", start=35.0, end=40.0,
            text="The storming of the Bastille on July 14 1789 is considered the symbolic start of the French Revolution and is commemorated as Bastille Day."),
    Segment(id="demo_09", video_id="demo", start=40.0, end=45.0,
            text="Vaccines work by introducing a weakened or inactivated form of a pathogen, training the immune system to recognize and fight it without causing disease."),
    Segment(id="demo_10", video_id="demo", start=45.0, end=50.0,
            text="After vaccination the immune system produces antibodies and memory cells that enable a rapid response if the real pathogen is encountered in the future."),
    Segment(id="demo_11", video_id="demo", start=50.0, end=55.0,
            text="The speed of light in a vacuum is approximately 299792 kilometers per second; according to special relativity nothing with mass can reach or exceed that speed."),
    Segment(id="demo_12", video_id="demo", start=55.0, end=60.0,
            text="Mitosis is the process of cell division that produces two daughter cells each with the same number and kind of chromosomes as the parent nucleus."),
    Segment(id="demo_13", video_id="demo", start=60.0, end=65.0,
            text="The Roman Empire at its peak stretched from Britain in the west to Mesopotamia in the east, encompassing large parts of Europe, North Africa, and the Middle East."),
    Segment(id="demo_14", video_id="demo", start=65.0, end=70.0,
            text="Plate tectonics explains that Earth's outer shell is divided into several large moving plates whose interactions shape the continents, mountains, and ocean basins."),
    Segment(id="demo_15", video_id="demo", start=70.0, end=75.0,
            text="DNA carries genetic information in sequences of four nucleotide bases—adenine, thymine, guanine, and cytosine—arranged in a double helix structure."),
]


def main() -> None:
    print(f"Embedding {len(_SEGMENTS)} segments …", flush=True)
    embed_segments(_SEGMENTS)

    print("Building index …", flush=True)
    idx = SegmentIndex()
    idx.build(_SEGMENTS)

    print(f"Saving index to {_OUTPUT_DIR!r} …", flush=True)
    idx.save(_OUTPUT_DIR)

    print(f"\nDone. Indexed {len(_SEGMENTS)} segments -> {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
