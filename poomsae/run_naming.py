"""Name the movements a performance is made of, and score the naming.

Segmentation gives move boundaries. This step answers the coach's question --
"which technique is move 7?" -- by classifying each segment against the
TUHAD A1-A8 taxonomy (plus the one Taegeuk-1 technique TUHAD omits) and then
validating the resulting name sequence against the published 18-count sequence
of Taegeuk Il Jang.

Validation design
-----------------
The reference is the published count list, not another classifier: Taegeuk Il
Jang has a canonically fixed order, so a correct naming layer must reproduce
it. Because the detector segments pauses rather than counts (a count can hold
two segments, and counts 14/16 hold kick+punch), the two sequences are aligned
with Needleman-Wunsch before scoring. Unaligned predicted segments are reported
as insertions rather than silently dropped.

Usage
-----
    python -m poomsae.run_naming            # score, write out/naming_taegeuk1.json
    python -m poomsae.run_naming --dump     # also print every segment's
                                            # descriptor evidence (calibration)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from .core import features, load_dataset, segment_moves
from .techniques import (CALIBRATION_FORM, FORM_COUNTS, TH, classify, describe,
                         label_name)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def align(a: list[str], b: list[str]):
    """Needleman-Wunsch. match +1, mismatch -1, gap -1. Returns [(i|None, j|None)]."""
    n, m = len(a), len(b)
    S = np.zeros((n + 1, m + 1))
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        S[i][0], P[i][0] = -i, 1
    for j in range(1, m + 1):
        S[0][j], P[0][j] = -j, 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = S[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else -1)
            u = S[i - 1][j] - 1
            l = S[i][j - 1] - 1
            best, arg = d, 0
            if u > best:
                best, arg = u, 1
            if l > best:
                best, arg = l, 2
            S[i][j], P[i][j] = best, arg
    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        arg = P[i][j]
        if arg == 0:
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif arg == 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


def classify_segment(seq, s0: int, s1: int):
    """Classify the technique a segment ends on.

    A pause is a hold, and a held technique is the technique that was just
    executed -- so the segment's label is read at its END boundary, not in the
    middle of the motion.

    A segment can hold TWO techniques: the published count tables put "front
    kick, then punch" under a single count (Taegeuk 2 counts 8, 10, 16, 17, 18),
    and that count is exactly two taxonomy entries. So this returns a list of
    labels: the kick first (found by its airborne peak, which never appears at a
    pause -- measured 7-10 frames), then whatever the arm is doing at the hold.
    """
    from .techniques import kick_peak
    b = min(s1, seq.n - 1)
    c = describe(seq, max(0, b - 2), b)
    peak = kick_peak(seq, s0, b)

    arm = classify(c)                       # arm technique held at the pause
    res = classify(c, peak=peak)            # same pose, kick test applied
    if peak > TH["kick_peak"] and arm["label"] != "JUMBI":
        labels, name = ["A8", arm["label"]], f"abchagi + {arm['name']}"
        confidence = arm["confidence"]
    else:
        labels, name, confidence = [res["label"]], res["name"], res["confidence"]

    out = dict(res)
    out.update({"labels": labels, "name": name, "confidence": confidence,
                "label": labels[-1], "t": round(b / 30.0, 2), "frames": [s0, s1]})
    out["descriptor"] = {
        "elbow": {k: round(v, 0) for k, v in c.elbow.items()},
        "wrist_above_shoulder": {k: round(v, 2) for k, v in c.wrist_above_shoulder.items()},
        "wrist_above_hip": {k: round(v, 2) for k, v in c.wrist_above_hip.items()},
        "reach": {k: round(v, 2) for k, v in c.reach.items()},
        "lateral": {k: round(v, 2) for k, v in c.lateral.items()},
        "kick_peak": round(peak, 2),
        "stance_width": round(c.stance_width, 2),
    }
    return out


def run_form(ds, form: str, view: str, dump: bool = False):
    counts = FORM_COUNTS[form]
    CANON = [lab for _, lab, _ in counts]
    CANON_DESC = [d for _, _, d in counts]
    seq = ds[(form, view)]
    feats = features(seq)
    segs = segment_moves(feats)
    # The final boundary is the return to joonbi after the last count, not a
    # technique, so it is excluded rather than scored as a mistake.
    seg_preds = [classify_segment(seq, s0, s1) for s0, s1 in segs[:-1]]
    flat = []
    for p in seg_preds:
        for lab in p["labels"]:
            flat.append({"label": lab, "t": p["t"], "confidence": p["confidence"],
                         "name": label_name(lab), "english": label_name(lab),
                         "segment": p})
    pairs = align([f["label"] for f in flat], CANON)
    preds = flat

    rows, confusions, matched, n_aligned = [], [], 0, 0
    for i, j in pairs:
        if i is not None and j is not None:
            n_aligned += 1
            ok = preds[i]["label"] == CANON[j]
            matched += ok
            rows.append({
                "count": counts[j][0],
                "canonical": CANON[j],
                "canonical_desc": CANON_DESC[j],
                "predicted": preds[i]["label"],
                "predicted_name": preds[i]["name"],
                "confidence": preds[i]["confidence"],
                "t": preds[i]["t"],
                "match": bool(ok),
            })
            if not ok:
                confusions.append(f"count {counts[j][0]} {CANON[j]}({CANON_DESC[j]}) -> "
                                  f"{preds[i]['label']}({preds[i]['english']})")
        elif i is not None:
            rows.append({"count": None, "canonical": None, "canonical_desc": None,
                         "predicted": preds[i]["label"],
                         "predicted_name": preds[i]["name"],
                         "confidence": preds[i]["confidence"], "t": preds[i]["t"],
                         "match": False, "insertion": True})
        else:
            rows.append({"count": counts[j][0], "canonical": CANON[j],
                         "canonical_desc": CANON_DESC[j], "predicted": None,
                         "predicted_name": None, "confidence": None, "t": None,
                         "match": False, "missed": True})

    if dump:
        print(f"\n--- {form} {view}: {len(segs)} segments ---")
        for k, p in enumerate(seg_preds):
            d = p["descriptor"]
            print(f"{k+1:2d} t={p['t']:6.2f} {'+'.join(p['labels']):8s} {p['confidence']:6s} "
                  f"elbow={d['elbow']['L']:.0f}/{d['elbow']['R']:.0f} "
                  f"hipY={d['wrist_above_hip']['L']:+.2f}/{d['wrist_above_hip']['R']:+.2f} "
                  f"reach={d['reach']['L']:.2f}/{d['reach']['R']:.2f} "
                  f"kick={d['kick_peak']:+.2f}")

    # Selective accuracy: the classifier's own confidence, restricted to aligned
    # rows. This separates "the segmenter put the boundary in the wrong place"
    # from "the rule set cannot tell these two techniques apart".
    sel = [r for r in rows if not r.get("insertion") and not r.get("missed")
           and r.get("confidence") in ("high", "medium")]
    sel_match = sum(1 for r in sel if r["match"])

    return {
        "form": form, "view": view,
        "n_segments": len(segs),
        "n_canonical_techniques": len(CANON),
        "n_aligned": n_aligned,
        "n_match": matched,
        "accuracy": round(matched / max(n_aligned, 1), 3),
        "selective_n": len(sel),
        "selective_accuracy": round(sel_match / max(len(sel), 1), 3),
        "insertions": sum(1 for r in rows if r.get("insertion")),
        "missed": sum(1 for r in rows if r.get("missed")),
        "confusions": confusions,
        "predicted_sequence": [p["label"] for p in preds],
        "canonical_sequence": CANON,
        "rows": rows,
    }


def move_names(seq) -> list[dict]:
    """Name every segment of a performance, keyed by the segment's end time."""
    out = []
    for s0, s1 in segment_moves(features(seq)):
        p = classify_segment(seq, s0, s1)
        out.append({"t": p["t"], "frames": p["frames"], "name": p["name"],
                    "labels": p["labels"], "confidence": p["confidence"],
                    "why": p["why"]})
    return out


def attach_names(moves: list[dict], seq) -> list[dict]:
    """Attach a technique name to each scored move (nearest segment hold in time).

    Used by the coach-facing outputs so a move reads "momtong an makki (inward
    middle block)" rather than "move 7". The name is a best-effort label with a
    measured accuracy (see out/naming_forms.json), not a referee decision.
    """
    named = move_names(seq)
    for m in moves:
        if not named:
            continue
        best = min(named, key=lambda r: abs(r["t"] - m["t"]))
        m["technique"] = best["name"]
        m["technique_labels"] = best["labels"]
        m["technique_confidence"] = best["confidence"]
    return moves


def main() -> int:
    dump = "--dump" in sys.argv
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")
    ds = load_dataset(root)
    payload = {"reference": "published 18-count sequences (British Taekwondo rules; "
                            "taekwondo-ai.com count tables; Albanian federation chart)",
               "calibration_form": CALIBRATION_FORM, "thresholds": TH, "forms": {}}
    for form in FORM_COUNTS:
        per_view = {}
        for view in ("Frontal", "Lateral"):
            if (form, view) not in ds:
                print(f"missing {form} {view}")
                continue
            r = run_form(ds, form, view, dump=dump)
            per_view[view] = r
            print(f"\n{form} {view}: {r['n_segments']} segments vs "
                  f"{r['n_canonical_techniques']} canonical techniques | matched "
                  f"{r['n_match']}/{r['n_aligned']} = {r['accuracy']*100:.1f}% "
                  f"| selective {r['selective_accuracy']*100:.1f}% "
                  f"(n={r['selective_n']}) | insertions {r['insertions']} "
                  f"| missed {r['missed']}")
            for c in r["confusions"]:
                print("   confusion:", c)
        if per_view:
            acc = sum(v["accuracy"] for v in per_view.values()) / len(per_view)
            sel = sum(v["selective_accuracy"] for v in per_view.values()) / len(per_view)
            payload["forms"][form] = {"views": per_view, "mean_accuracy": round(acc, 3),
                                      "mean_selective_accuracy": round(sel, 3)}
            print(f"{form}: mean accuracy {acc*100:.1f}% "
                  f"(calibration={form == CALIBRATION_FORM})")

    held = [f for f in payload["forms"] if f != CALIBRATION_FORM]
    if held:
        payload["heldout_mean_accuracy"] = round(
            sum(payload["forms"][f]["mean_accuracy"] for f in held) / len(held), 3)
        print(f"\nHELD-OUT mean accuracy: {payload['heldout_mean_accuracy']*100:.1f}% "
              f"(thresholds tuned on {CALIBRATION_FORM}, never adjusted for {held})")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "naming_forms.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote out/naming_forms.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
