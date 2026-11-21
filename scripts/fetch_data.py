#!/usr/bin/env python3
"""Get the CC BY 4.0 dual-view Taegeuk dataset into the layout the pipeline expects.

Mendeley Data has no stable public download endpoint, so this script does not pretend to
download the archive. It tells you exactly what to fetch, where to put it, and then **verifies
the layout** — because a pipeline that silently runs on an empty data/ directory is worse than
one that refuses.

    python scripts/fetch_data.py           # check local layout
    python scripts/fetch_data.py --manual  # print the download instructions
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Data", "video")

DOI = "10.17632/bjy7vr4xkt.1"
URL = f"https://doi.org/{DOI}"
FORMS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
VIEWS = ["Frontal", "Lateral"]

MANUAL = f"""\
The dataset is CC BY 4.0 and is not redistributed in this repository.

  1. Open   {URL}
  2. Download the archive (Mendeley Data -> "Download all").
  3. Unpack so that this exists:

       data/Data/video/Taegeuk_1st/Frontal/*.mp4
       data/Data/video/Taegeuk_1st/Lateral/*.mp4
       ...                                   (through Taegeuk_8th)

  4. Re-run: python scripts/fetch_data.py

Cite it as: Zheng, Q. (2025). Taekwondo Poomsae (8 Forms) - Front & Side Videos with
12-Keypoint Pose Annotations (MediaPipe). Mendeley Data, V1. doi:{DOI}
"""


def main(check=True) -> int:
    if "--manual" in sys.argv:
        print(MANUAL)
        check = True

    missing, found = [], []
    for form in FORMS:
        for view in VIEWS:
            d = os.path.join(DATA, f"Taegeuk_{form}", view)
            if os.path.isdir(d):
                clips = [f for f in os.listdir(d) if f.lower().endswith(".mp4")]
                (found if clips else missing).append(f"Taegeuk_{form}/{view}" + ("" if clips else " (empty)"))
            else:
                missing.append(f"Taegeuk_{form}/{view}")

    print(f"data root: {DATA}")
    print(f"present : {len(found)}/16 form-view directories with clips")
    if missing:
        print(f"missing : {len(missing)} -> {', '.join(missing)}")
        print()
        print(MANUAL)
        return 1

    total = 0
    for form in FORMS:
        for view in VIEWS:
            d = os.path.join(DATA, f"Taegeuk_{form}", view)
            for f in os.listdir(d):
                if f.lower().endswith(".mp4"):
                    total += os.path.getsize(os.path.join(d, f))
    print(f"all 16 present, {total/1e6:.0f} MB of video. Run: .venv/bin/bash run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
