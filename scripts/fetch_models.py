#!/usr/bin/env python3
"""Download the two MediaPipe pose models this project measured against.

Prints the SHA-256 of every file it writes, so a reader can confirm they are running the same
estimators whose disagreement defines the tolerance in this repo.

    python scripts/fetch_models.py            # download what is missing
    python scripts/fetch_models.py --check    # verify what is present, no network
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
FILES = {
    "pose_landmarker_full.task": f"{BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    # only used by the gesture experiments; run_all.sh checks for it, so fetch it
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_only() -> int:
    missing = 0
    for name in FILES:
        path = os.path.join(MODELS, name)
        if not os.path.exists(path):
            print(f"MISSING {name}")
            missing += 1
            continue
        size = os.path.getsize(path)
        print(f"OK      {name:28s} {size/1e6:6.1f} MB  sha256 {sha256(path)}")
    return 1 if missing else 0


def fetch() -> int:
    os.makedirs(MODELS, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(MODELS, name)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print(f"present {name} ({os.path.getsize(path)/1e6:.1f} MB) - not re-downloading")
        else:
            print(f"downloading {name} <- {url}")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            if os.path.getsize(tmp) < 1_000_000:
                os.unlink(tmp)
                print(f"FAILED: {name} came back too small; check the URL", file=sys.stderr)
                return 1
            os.replace(tmp, path)
    print()
    return check_only()


if __name__ == "__main__":
    sys.exit(check_only() if "--check" in sys.argv else fetch())
