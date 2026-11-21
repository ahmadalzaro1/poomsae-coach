#!/usr/bin/env python3
"""Check that a re-run reproduced the certified numbers, not the certified clock.

`verify_claims.py` asks "does the README match the committed artifacts?". This asks the other
question: "does *my* run match the certified artifacts?" A full `run_all.sh` rerun on the same
machine reproduces every substantive number and changes exactly three kinds of field: when it ran,
how long it took, and how fast the pose model happened to extract. Those are measurements of the
machine, not of the athlete, so they are dropped before hashing — explicitly, and reported.

    python3 scripts/check_reproducible.py            # compare against out/manifest.sha256
    python3 scripts/check_reproducible.py --write    # regenerate the manifest (review the diff first)
    python3 scripts/check_reproducible.py --explain  # show which keys were ignored and why
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
MANIFEST = os.path.join(OUT, "manifest.sha256")

# Fields that describe the run, not the form. Dropped before hashing; anything else that differs
# is a real difference and fails the check.
TIMING_EXACT = {
    "generated_at", "runtime_s", "extraction_ms_per_frame", "wall_ms", "elapsed_s",
    "elapsed_s_per_frame", "ms_per_frame", "created_at", "timestamp", "host", "machine",
}
TIMING_SUFFIX = ("_ms", "_ms_per_frame", "_seconds")


def strip_timing(node, dropped):
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in TIMING_EXACT or str(key).endswith(TIMING_SUFFIX):
                dropped.add(str(key))
                continue
            out[key] = strip_timing(value, dropped)
        return out
    if isinstance(node, list):
        return [strip_timing(v, dropped) for v in node]
    return node


def canonical(path, dropped):
    with open(path) as fh:
        data = json.load(fh)
    payload = json.dumps(strip_timing(data, dropped), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def artifacts():
    names = sorted(n for n in os.listdir(OUT) if n.endswith(".json") and n != "manifest.sha256")
    return [(os.path.join(OUT, n), n) for n in names]


def read_manifest():
    if not os.path.exists(MANIFEST):
        sys.exit("no out/manifest.sha256 - run with --write once, then commit it")
    rows = {}
    for line in open(MANIFEST):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        rows[name.strip()] = digest
    return rows


def main(argv) -> int:
    dropped: set[str] = set()
    if "--write" in argv:
        lines = ["# canonical hash of out/*.json with run-timing fields removed",
                 "# regenerate with: python3 scripts/check_reproducible.py --write"]
        for path, name in artifacts():
            lines.append(f"{canonical(path, dropped)}  {name}")
        with open(MANIFEST, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"wrote {MANIFEST} ({len(lines) - 2} artifacts)")
        print("ignored fields: " + ", ".join(sorted(dropped)))
        return 0

    expected = read_manifest()
    bad, missing = [], []
    for path, name in artifacts():
        if name not in expected:
            missing.append(name)
            continue
        got = canonical(path, dropped)
        if got != expected[name]:
            bad.append(name)

    if "--explain" in argv:
        print("ignored as run-timing (not a difference in the athlete):")
        for key in sorted(dropped):
            print(f"  - {key}")

    extra = sorted(set(expected) - {n for _, n in artifacts()})
    if bad:
        print(f"DIFFERENT: {len(bad)} artifact(s) changed beyond run timing:")
        for name in bad:
            print(f"  - out/{name}")
        print("investigate before republishing: this is a change in the numbers, not the clock.")
        return 1
    print(f"{len(artifacts())}/{len(artifacts())} artifacts reproduce "
          f"({len(dropped)} run-timing fields ignored)")
    if missing:
        print("not in the manifest (new artifacts): " + ", ".join(missing))
    if extra:
        print("in the manifest but absent now: " + ", ".join(extra))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
