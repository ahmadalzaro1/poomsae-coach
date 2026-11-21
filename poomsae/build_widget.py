"""Build a self-contained demo widget (skeleton replay + per-move coaching).

Two real scenarios, both from the CC-BY dual-view dataset:
  A. same clip, two pose estimators  (published annotation vs our MediaPipe run)
  B. same athlete, two camera angles (frontal vs lateral)
Nothing is synthesised: every number in the widget comes from out/*.json runs.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (FEATURE_NAMES, JOINTS, TOLERANCE_DEG, analyse, features,
                          load_csv, segment_moves)
from poomsae.run_naming import attach_names

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(HERE, "data", "Data")
STEP = 2  # keep every 2nd frame (15 fps) to hold the payload small

BONES = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 8), (8, 10),
         (7, 9), (9, 11), (0, 1), (6, 7), (0, 7), (1, 6)]


def seq_payload(seq, W=1920, H=1080):
    px = seq.px[::STEP]
    out = []
    for f in px:
        out.append([round(float(x) / W, 4) if i % 2 == 0 else round(float(x) / H, 4)
                    for i, x in enumerate(f.reshape(-1))])
    return out


def _moves_payload(moves, frames, fps_data=15.0):
    """Frame ranges must tile the whole clip with no gaps: each move runs from
    its own start to the next move's start. (A fixed +1.2 s window left holes on
    long moves, and the feedback panel went blank inside them.)"""
    out = []
    for i, m in enumerate(moves):
        start = int(round(m["t"] * fps_data))
        end = (int(round(moves[i + 1]["t"] * fps_data)) - 1) if i + 1 < len(moves) \
            else frames - 1
        out.append({"move": m["move"], "t": m["t"], "mean": m["mean_abs_delta"],
                    "max": m["max_abs_delta"], "exceed": m["max_exceedance"],
                    "tech": m.get("technique", ""), "tech_conf": m.get("technique_confidence", ""),
                    "flags": m["flags"], "seg": [start, max(start, end)],
                    "deltas": m["deltas"]})
    return out


def build_case(ref_seq, stu_seq, label, note):
    res = analyse(ref_seq, stu_seq, threshold=None)
    attach_names(res["moves"], stu_seq)
    frames = min(len(ref_seq.px[::STEP]), len(stu_seq.px[::STEP]))
    return {
        "label": label,
        "note": note,
        "ref_name": f"{ref_seq.form} / {ref_seq.view}",
        "stu_name": f"{stu_seq.form} / {stu_seq.view}",
        "dtw": res["dtw_cost_per_frame"],
        "mean_delta": res["mean_abs_delta"],
        "flagged": res["flagged_moves"],
        "n_moves": res["n_moves"],
        "frames": frames,
        "ref": seq_payload(ref_seq)[:frames],
        "stu": seq_payload(stu_seq)[:frames],
        "moves": _moves_payload(res["moves"], frames),
    }


def main():
    ref = load_csv(os.path.join(BASE, "annotations", "Taegeuk_1st", "Frontal", "1st_Frontal.csv"))
    lat = load_csv(os.path.join(BASE, "annotations", "Taegeuk_1st", "Lateral", "1st_Lateral.csv"))
    ours = load_csv(os.path.join(HERE, "out", "pose_taegeuk1_frontal.csv"))

    cases = [
        build_case(ref, ours, "A · same clip, two pose estimators",
                   "Reference = published annotation. Student = our own MediaPipe run on the same "
                   "video. Legs agree to ~2°, arms drift ~17° because the two estimators place the "
                   "wrists differently (20-24 px)."),
        build_case(ref, lat, "B · same athlete, two camera angles",
                   "Reference = frontal annotation. Student = the SAME performance filmed from the "
                   "side. Every difference here is the camera angle, not the athlete."),
    ]
    data = {"form": "Taegeuk 1 Jang", "tolerance": TOLERANCE_DEG,
            "joints": JOINTS, "bones": BONES, "cases": cases}
    payload = json.dumps(data, separators=(",", ":"))
    tpl = open(os.path.join(HERE, "poomsae", "widget_template.html")).read()
    html = tpl.replace("/*__DATA__*/null", payload)
    out = os.path.join(HERE, "out", "poomsae_widget.html")
    open(out, "w").write(html)
    print(f"wrote {out} ({len(html)/1024:.0f} KB, {len(cases)} cases, "
          f"{cases[0]['frames']} frames each)")
    for c in cases:
        print(f"  {c['label']}: mean|Δ| {c['mean_delta']}°, flagged {c['flagged']}/{c['n_moves']}")


if __name__ == "__main__":
    main()
