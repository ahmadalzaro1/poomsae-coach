#!/usr/bin/env python3
"""Fail if any local link or image in the markdown points at a file that is not committed.

A repo is 'neat' in proportion to how few of its links 404. Stdlib only.

    python scripts/check_links.py
"""
from __future__ import annotations

import os
import re
import sys
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP = ("http://", "https://", "mailto:", "#")


def tracked() -> set[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
        return set(out.stdout.split("\n"))
    except Exception:
        return set()


def main() -> int:
    files = [f for f in tracked() if f.endswith(".md")]
    if not files:
        files = [os.path.relpath(os.path.join(d, n), ROOT)
                 for d, _, ns in os.walk(ROOT) for n in ns
                 if n.endswith(".md") and ".git" not in d]
    ok = bad = 0
    for rel in sorted(files):
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for target in LINK.findall(text):
            target = target.split(" ")[0].strip("<>")
            if target.startswith(SKIP):
                continue
            target = target.split("#")[0]
            if not target:
                continue
            resolved = os.path.normpath(os.path.join(os.path.dirname(path), target))
            if os.path.exists(resolved):
                ok += 1
            else:
                print(f"BROKEN {rel} -> {target}")
                bad += 1
    print(f"\n{ok} local links resolve, {bad} broken")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
