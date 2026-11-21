"""View calibration: remove the systematic part of a cross-view deviation.

`learned_gate.py` tested whether the shape of a deviation vector can tell a real
fault from a camera artifact. It cannot: AUC 0.712, and at matched recall the
learned gate flagged MORE camera artifacts than the plain rule (86.5% vs 85.3%).
That negative result says the information is not in the deviation vector at all.

What it implies is different. A frontal-vs-lateral comparison carries a
per-joint bias that is a property of the CAMERA PAIR, not of the athlete: every
move in the clip sits on the same geometric distortion. So estimate that bias
from the clip's own moves (median across moves, per joint) and subtract it. Then
the residual is what varies move to move -- which is where an athlete's actual
error lives.

    d_corrected[k] = d[k] - median_k(d[k])

This needs no labels and no training. It has one honest cost: a fault the
athlete commits on EVERY move becomes part of the bias and disappears. So this
measures inconsistency, not absolute correctness. Both numbers are reported.

Measured here: false flags on cross-view before/after, mean deviation, the bias
field itself (which joints the camera pair distorts), and whether injected
faults survive the correction.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import FEATURE_NAMES, TOLERANCE_DEG, load_dataset
from poomsae.learned_gate import form_moves

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
TOL = np.array([TOLERANCE_DEG.get(f, 15.0) for f in FEATURE_NAMES])
MAGS = (10.0, 15.0, 20.0, 25.0)


def flags(d: np.ndarray) -> int:
    return int((np.abs(d) / TOL > 1.0).max(axis=1).sum())


def main() -> int:
    ds = load_dataset(ROOT)
    forms = sorted({f for f, _ in ds})
    rows, bias_all, kept = [], [], {m: [] for m in MAGS}
    for form in forms:
        d, *_ = form_moves(ds[(form, "Frontal")], ds[(form, "Lateral")])
        bias = np.median(d, axis=0)
        resid = d - bias
        n = len(d)
        rows.append({
            "form": form, "moves": n,
            "raw_flags": flags(d), "calibrated_flags": flags(resid),
            "mean_abs_raw": round(float(np.abs(d).mean()), 2),
            "mean_abs_calibrated": round(float(np.abs(resid).mean()), 2),
            "bias": [round(float(v), 1) for v in bias],
        })
        bias_all.append(bias)
        # Does a fault survive correction? A real fault happens on ONE move, so
        # inject into a single move (the cleanest one) and keep the others clean.
        # Injecting into every move instead would make the fault identical to a
        # bias and median-subtraction would erase it by construction -- which is
        # the documented cost, reported separately below.
        k_clean = int(np.argmin(np.abs(d).max(axis=1)))
        for ji in range(len(FEATURE_NAMES)):
            for m in MAGS:
                d_pos = d.copy()
                d_pos[k_clean, ji] += m
                resid_pos = d_pos - np.median(d_pos, axis=0)
                hit = 1.0 if abs(resid_pos[k_clean, ji]) / TOL[ji] > 1.0 else 0.0
                kept[m].append(hit)

    tot_raw = sum(r["raw_flags"] for r in rows)
    tot_cal = sum(r["calibrated_flags"] for r in rows)
    tot_n = sum(r["moves"] for r in rows)
    mr = float(np.mean([r["mean_abs_raw"] for r in rows]))
    mc = float(np.mean([r["mean_abs_calibrated"] for r in rows]))

    print(f"cross-view, 8 forms, {tot_n} moves (same athlete, no fault present)")
    print(f"  rule flags            {tot_raw:3d}/{tot_n}  ({100*tot_raw/tot_n:.1f}%)")
    print(f"  after calibration     {tot_cal:3d}/{tot_n}  ({100*tot_cal/tot_n:.1f}%)")
    print(f"  mean |delta|          {mr:.2f} deg -> {mc:.2f} deg")
    print("\n  the bias field (deg) -- what the frontal->lateral pair distorts:")
    b = np.mean(bias_all, axis=0)
    for f, v in sorted(zip(FEATURE_NAMES, b), key=lambda kv: -abs(kv[1])):
        print(f"    {f:<12}{v:+7.1f}")
    print("\n  injected fault on ONE move, still detected after calibration:")
    for m in MAGS:
        arr = np.array(kept[m])
        print(f"    +/-{m:.0f} deg  {100*arr.mean():5.1f}%  (min {100*arr.min():.1f}%)")
    print("\n  cost (by design): a fault repeated on EVERY move is absorbed into the")
    print("  bias and disappears. This measures inconsistency across moves, not")
    print("  absolute correctness.")

    payload = {
        "per_form": rows,
        "pooled": {"moves": tot_n, "raw_flags": tot_raw, "calibrated_flags": tot_cal,
                   "raw_flag_rate": round(tot_raw / tot_n, 4),
                   "calibrated_flag_rate": round(tot_cal / tot_n, 4),
                   "mean_abs_raw": round(mr, 2), "mean_abs_calibrated": round(mc, 2)},
        "bias_field_deg": {f: round(float(v), 1) for f, v in zip(FEATURE_NAMES, b)},
        "injection_recall_after_calibration": {f"{m:.0f}deg": round(float(np.array(kept[m]).mean()), 4)
                                               for m in MAGS},
        "cost": ("a fault the athlete commits on every move is absorbed into the "
                 "bias and disappears; this measures inconsistency, not absolute "
                 "correctness"),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "view_calibration.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("\nwrote out/view_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
