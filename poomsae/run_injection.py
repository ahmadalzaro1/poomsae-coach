"""Error-injection validation of the feedback engine.

We cannot invent a second athlete, so we do the next honest thing: take a real
performance, corrupt ONE joint by a known amount across ONE detected move, and
ask the scorer whether it points at the right move and the right joint.

Reported:
  top1_move    — injected move is the highest-deviation move
  flagged_move — injected move is flagged at all
  right_joint  — the top flag on that move names the injected joint
  collateral   — number of OTHER moves also flagged (specificity cost)
plus a sensitivity curve over injection magnitudes.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (FEATURE_NAMES, analyse, features, load_dataset,
                          segment_moves)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def inject(feats, seg, joint_idx, magnitude):
    f = feats.copy()
    f[seg[0]:seg[1] + 1, joint_idx] += magnitude
    return f


def trial(ref, seg_feats, segs, move_i, joint_i, magnitude, threshold=15.0):
    stu = inject(seg_feats, segs[move_i], joint_i, magnitude)
    rf = ref
    cost, path = _dtw_fast(rf, stu)
    j_of_i = {}
    for i, j in path:
        j_of_i.setdefault(i, j)
    segs_stu = []
    for r0, r1 in segs:
        js = [j_of_i[i] for i in range(r0, r1 + 1) if i in j_of_i] or [0]
        segs_stu.append((min(js), max(js)))
    from poomsae.core import score_moves
    moves = score_moves(rf, stu, segs, segs_stu, threshold=threshold)
    ranked = sorted(moves, key=lambda m: -m["mean_abs_delta"])
    top1 = ranked[0]["move"] - 1 == move_i
    flagged = bool(moves[move_i]["flags"])
    right_joint = bool(moves[move_i]["flags"]) and moves[move_i]["flags"][0]["key"] == FEATURE_NAMES[joint_i]
    collateral = sum(1 for m in moves if m["flags"]) - (1 if flagged else 0)
    return top1, flagged, right_joint, collateral


def _dtw_fast(r, s):
    """Same-length, near-identity alignment: tiny band keeps 1000+ trials cheap."""
    from poomsae.core import dtw
    return dtw(r, s, band=4)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset(ROOT)
    forms = sorted({f for f, _ in ds})
    magnitudes = [10.0, 15.0, 20.0, 25.0, 35.0]
    results = {}
    for mag in magnitudes:
        n_top1 = n_flag = n_joint = 0
        coll = []
        for form in forms:
            seq = ds[(form, "Lateral")]
            F = features(seq)
            segs = segment_moves(F)
            for mi in range(len(segs)):
                if segs[mi][1] - segs[mi][0] < 5:
                    continue
                for ji in range(len(FEATURE_NAMES)):
                    t1, fl, rj, c = trial(F, F, segs, mi, ji, mag)
                    n_top1 += t1
                    n_flag += fl
                    n_joint += rj
                    coll.append(c)
        total = len(coll)
        results[f"{mag:.0f}deg"] = {
            "trials": total,
            "top1_move": round(100 * n_top1 / total, 1),
            "flagged_move": round(100 * n_flag / total, 1),
            "right_joint": round(100 * n_joint / total, 1),
            "mean_collateral_flags": round(float(np.mean(coll)), 2),
        }
        print(f"inject ±{mag:>4.0f}°  trials {total:>5}  top1-move {100*n_top1/total:>5.1f}%  "
              f"flagged {100*n_flag/total:>5.1f}%  right-joint {100*n_joint/total:>5.1f}%  "
              f"collateral {np.mean(coll):.2f}")

    # specificity control: unmodified sequence must produce zero flags
    zero = 0
    for form in forms:
        F = features(ds[(form, "Lateral")])
        res = analyse(ds[(form, "Lateral")], ds[(form, "Lateral")], threshold=15.0)
        zero += res["flagged_moves"]
    results["control_unmodified_flags"] = zero
    print(f"\ncontrol (unmodified vs itself): {zero} flags across {len(forms)} forms")
    json.dump(results, open(os.path.join(OUT, "injection_test.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
