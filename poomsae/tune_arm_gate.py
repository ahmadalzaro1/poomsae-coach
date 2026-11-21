"""Tuning: does gating the low-confidence ARM joints improve elbow accuracy?

Pose estimators disagree most at the wrists (20-24 px p90 ~60 px on this clip),
and that error propagates straight into the elbow angle. Test: interpolate the
arm chain more aggressively (higher confidence threshold) and see whether the
elbow features converge toward the published annotation.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poomsae.core as core
from poomsae.core import FEATURE_NAMES, JOINTS, analyse, load_csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(HERE, "data", "Data")
OURS = os.path.join(HERE, "out", "pose_taegeuk1_frontal.csv")
ANNOT = os.path.join(base, "annotations", "Taegeuk_1st", "Frontal", "1st_Frontal.csv")


def patched_features(seq, min_score=0.5, arm_gate=None, smooth_w=5):
    """Same as core.features but with a stricter gate on the arm chain."""
    px = seq.px.copy()
    thr = np.full(12, min_score)
    if arm_gate is not None:
        for i, name in enumerate(JOINTS):
            if name in ("L_wrist", "R_wrist", "L_elbow", "R_elbow"):
                thr[i] = arm_gate
    bad = (seq.scores < thr[None, :]) | np.isnan(px).any(-1)
    for j in range(12):
        if bad[:, j].any():
            ok = ~bad[:, j]
            if ok.sum() >= 2:
                for d in range(2):
                    px[:, j, d] = np.interp(np.arange(len(px)), np.flatnonzero(ok), px[ok, j, d])
    feats = []
    for _, a, b, c in core.ANGLE_TRIPLETS:
        feats.append(core.angle_at(px[:, core.J[a]], px[:, core.J[b]], px[:, core.J[c]]))
    hip = (px[:, core.J["L_hip"]] + px[:, core.J["R_hip"]]) / 2
    sho = (px[:, core.J["L_shoulder"]] + px[:, core.J["R_shoulder"]]) / 2
    spine = core._unit(sho - hip)
    feats.append(np.degrees(np.arccos(np.clip(spine[:, 1], -1, 1))))
    return core.smooth(np.stack(feats, axis=1), smooth_w)


def main():
    ours, annot = load_csv(OURS), load_csv(ANNOT)
    n = min(ours.n, annot.n)
    truth = patched_features(annot)
    results = []
    for gate in (0.5, 0.7, 0.8, 0.85, 0.9, 0.95):
        F = patched_features(ours, arm_gate=gate)
        d = np.abs(F[:n] - truth[:n])
        arm = d[:, [FEATURE_NAMES.index("L_elbow"), FEATURE_NAMES.index("R_elbow")]].mean()
        row = {"arm_gate": gate,
               "elbow_mae_deg": round(float(arm), 1),
               "all_feature_mae_deg": round(float(d.mean()), 2),
               "leg_mae_deg": round(float(d[:, [2, 3, 4, 5]].mean()), 2)}
        results.append(row)
        print(f"arm gate {gate:<5} elbow MAE {row['elbow_mae_deg']:>5.1f}°  "
              f"legs {row['leg_mae_deg']:>4.2f}°  all {row['all_feature_mae_deg']:>4.2f}°")

    best = min(results, key=lambda r: r["elbow_mae_deg"])
    print(f"\nbest arm gate: {best['arm_gate']} (elbow MAE {best['elbow_mae_deg']}°, "
          f"was {results[0]['elbow_mae_deg']}° at 0.5)")

    # does the improvement change the same-view coaching card?
    for gate in dict.fromkeys((0.5, best["arm_gate"])):
        F_ref, F_stu = patched_features(annot), patched_features(ours, arm_gate=gate)
        cost, path = core.dtw(F_ref, F_stu)
        segs = core.segment_moves(F_ref)
        j_of = {}
        for i, j in path:
            j_of.setdefault(i, j)
        segs_stu = []
        for r0, r1 in segs:
            js = [j_of[i] for i in range(r0, r1 + 1) if i in j_of] or [0]
            segs_stu.append((min(js), max(js)))
        moves = core.score_moves(F_ref, F_stu, segs, segs_stu)
        md = np.mean([m["mean_abs_delta"] for m in moves])
        fl = sum(1 for m in moves if m["flags"])
        print(f"gate {gate}: same-view mean|Δ| {md:.1f}°  flagged {fl}/{len(moves)}")
        results.append({"gate": gate, "same_view_mean_delta": round(float(md), 1),
                        "same_view_flags": fl, "moves": len(moves)})

    json.dump(results, open(os.path.join(HERE, "out", "arm_gate_tuning.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
