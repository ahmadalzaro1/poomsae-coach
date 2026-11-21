"""Repeat observations without a second take.

The rule this answers: a tolerance must be justified against two independent observations of
the same movement, not against a noisy label. Our datasets contain no second take (Mendeley =
8 forms x 2 views, one recording each; the labelled set = one recording per player per class),
so the second observation has to come from inside a take.

It does. Each Taegeuk form performs the same technique several times, seconds apart, by the
same athlete, on the same camera. Grouping the card's moves by technique and measuring the
spread inside a group gives:

    repeat spread = how much the SAME technique reads differently across instances

That is strictly stronger than the estimator-swap band (which reads identical frames twice) and
strictly weaker than two recordings days apart (it cannot see fatigue, clothing, camera moves
or a new room). It is reported as what it is: a BETWEEN-REPEAT estimate, not between-session.

Side confound: consecutive instances are usually mirrored (left arm, then right). To avoid
measuring inter-limb asymmetry and calling it noise, each move contributes the magnitude of its
ACTIVE side (max of |L|, |R|) for the joint under test, so both instances are on the same
footing.

Output: out/repeatability.json plus a printed table.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")
TOL = {"L_elbow": 18.0, "R_elbow": 18.0, "L_knee": 10.0, "R_knee": 10.0, "torso_tilt": 5.0}
PAIRS = [("L_elbow", "R_elbow"), ("L_knee", "R_knee")]


def active_side(move: dict, left: str, right: str) -> float:
    """Magnitude of the active side for a mirrored joint pair."""
    return max(abs(float(move["deltas"].get(left, 0.0))), abs(float(move["deltas"].get(right, 0.0))))


def group(card: dict, key: str) -> dict:
    out = defaultdict(list)
    for m in card["moves"]:
        tech = (m.get("technique") or "unnamed").split(" (")[0]
        out[tech].append(active_side(m, *key) if key else abs(float(m["deltas"].get("torso_tilt", 0.0))))
    return out


def spread(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": round(float(v.mean()), 1),
            "sd": round(float(v.std(ddof=1)) if v.size > 1 else float("nan"), 1),
            "range": round(float(v.max() - v.min()), 1)}


def main() -> int:
    res = {"generated": "2026-11-21",
           "kind": "between-repeat spread (same athlete, same take, same camera)",
           "not": "between-session: cannot see fatigue, clothing, camera move or a new room",
           "side_rule": "mirrored pairs contribute the active side only (max of |L|,|R|)",
           "cards": {}}
    for tag in ("sameview", "crossview"):
        card = json.load(open(os.path.join(OUT, f"card_{tag}.json")))
        res["cards"][tag] = {"n_moves": len(card["moves"]), "pairs": {}}
        print(f"\n=== [{tag}] repeat spread of the same technique inside one take ===")
        for pair in PAIRS:
            groups = group(card, pair)
            rows = {t: spread(v) for t, v in groups.items() if len(v) > 1}
            pooled = [x for v in groups.values() if len(v) > 1 for x in v]
            print(f"\n  {pair[0][2:]} (tol {TOL[pair[0]]:g}°)")
            for t, s in sorted(rows.items(), key=lambda kv: -kv[1]["sd"] if kv[1]["sd"] == kv[1]["sd"] else 0):
                print(f"    {t[:44]:<44} n={s['n']}  mean {s['mean']:>5.1f}°  sd {s['sd']:>5.1f}°  range {s['range']:>5.1f}°")
            if pooled:
                p = spread(pooled)
                print(f"    {'POOLED':<44} n={p['n']}  mean {p['mean']:>5.1f}°  sd {p['sd']:>5.1f}°  range {p['range']:>5.1f}°")
                res["cards"][tag]["pairs"][pair[0][2:]] = {"groups": rows, "pooled": p,
                                                           "tolerance_deg": TOL[pair[0]]}
        tg = group(card, None)
        tp = [x for v in tg.values() if len(v) > 1 for x in v]
        if tp:
            p = spread(tp)
            print(f"\n  torso_tilt (tol {TOL['torso_tilt']:g}°): POOLED n={p['n']} mean {p['mean']}° sd {p['sd']}° range {p['range']}°")
            res["cards"][tag]["pairs"]["torso_tilt"] = {"groups": {t: spread(v) for t, v in tg.items() if len(v) > 1},
                                                       "pooled": p, "tolerance_deg": TOL["torso_tilt"]}
        # estimator-swap band, for side-by-side comparison
        unc = card.get("uncertainty") or {}
        res["cards"][tag]["estimator_swap_band_p50_deg"] = unc.get("per_joint_band_p50_deg")
        res["cards"][tag]["estimator_swap_band_p95_deg"] = unc.get("per_joint_band_p95_deg")

    json.dump(res, open(os.path.join(OUT, "repeatability.json"), "w"), indent=1)
    print("\nwrote out/repeatability.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
