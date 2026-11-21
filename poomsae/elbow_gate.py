"""Selective elbow feedback: say nothing where the signal cannot be trusted.

The elbow-signal study measured two things that pull in opposite directions:

  * typical agreement between independent estimators is small (~4 deg per move),
  * the tail is enormous (p95 20-40 deg on the worst move of a form).

A single tolerance cannot serve both: set it to the tail and nothing is detectable;
set it to the typical case and the scorer emits coaching sentences that are really
model disagreement. The fix is not a number, it is a decision rule per move -- emit
elbow feedback only when two independent readings of the same move agree closely
enough for the sentence to be true.

This module measures that rule on the 16 cached clips:

  1. per-move disagreement between pose_full and pose_heavy (the two readings),
  2. coverage and residual disagreement for a grid of gates,
  3. whether a FREE within-model proxy (jitter of the elbow angle inside the hold,
     which needs only one model) predicts that disagreement -- because paying for two
     models per student video is a product cost that has to be earned.

No model inference here: it reads out/elbow_signal/*.npz from poomsae/elbow_signal.py.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poomsae.core import (TOLERANCE_DEG, features, load_dataset,  # noqa: E402
                          segment_moves)
from poomsae.elbow_signal import FORMS, OUT, VIEWS, elbow_angles  # noqa: E402

NPZ_DIR = os.path.join(OUT, "elbow_signal")


def npz_path(ordinal: str, view: str, suffix: str = "") -> str:
    return os.path.join(NPZ_DIR, f"{ordinal}_{view}{suffix}.npz")

GATES = (5.0, 10.0, 15.0, 20.0, 25.0)


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared -- enough for Spearman without scipy."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def within_move_jitter(ang: np.ndarray, a0: int, a1: int) -> float:
    """Median absolute frame-to-frame change of the elbow angle inside one move.

    Free: needs a single model. A move where the wrist flickers in and out of
    detection shows a large value; a clean hold shows a small one.
    """
    seg = ang[a0:a1 + 1]
    seg = seg[np.isfinite(seg)]
    if len(seg) < 3:
        return float("nan")
    return float(np.median(np.abs(np.diff(seg))))


def main() -> None:
    ds = load_dataset("data/Data")
    rows, moves = {}, []
    for form, ordinal in FORMS:
        for view in VIEWS:
            path = npz_path(ordinal, view)
            if not os.path.exists(path):
                print(f"missing {path}: run poomsae.elbow_signal first")
                continue
            rec = {k: v for k, v in np.load(path).items()}
            rec["T"] = int(rec["T"])
            T = rec["T"]
            full = elbow_angles(rec, "pose_full")
            heavy = elbow_angles(rec, "pose_heavy")
            segs = segment_moves(features(ds[(form, view)]))
            clip = {"form": form, "view": view, "n_moves": len(segs), "per_move": []}
            for k, (a0, a1) in enumerate(segs):
                a1 = min(a1, T - 1)
                if a1 - a0 < 3:
                    continue
                for side in (0, 1):
                    mf = float(np.nanmean(full[side][a0:a1 + 1]))
                    mh = float(np.nanmean(heavy[side][a0:a1 + 1]))
                    if not (np.isfinite(mf) and np.isfinite(mh)):
                        continue
                    d = abs(mf - mh)
                    jit = within_move_jitter(heavy[side], a0, a1)
                    mv = {"move": k, "side": "LR"[side], "disagree_deg": round(d, 2),
                          "jitter_deg": None if not np.isfinite(jit) else round(jit, 3),
                          "full_deg": round(mf, 1), "heavy_deg": round(mh, 1)}
                    clip["per_move"].append(mv)
                    moves.append((dict(form=form, view=view), mv))
            rows[f"{ordinal}_{view}"] = clip

    d = np.array([m["disagree_deg"] for _, m in moves], float)
    j = np.array([m["jitter_deg"] if m["jitter_deg"] is not None else np.nan
                  for _, m in moves], float)
    ok = np.isfinite(j)
    rho = spearman(d[ok], j[ok])

    print(f"\nelbow moves with two readings: {len(d)} "
          f"(from {len(rows)} clips, {len({k for k, _ in [(f, v) for f in rows for v in [0]]})} forms)")
    print("disagreement percentiles (deg):",
          {f"p{q}": round(float(np.percentile(d, q)), 1) for q in (50, 75, 90, 95, 99)})
    print(f"max {d.max():.1f} deg | share above 10 deg {np.mean(d > 10):.1%} "
          f"| above 20 deg {np.mean(d > 20):.1%}")
    print(f"free proxy (within-move jitter) vs disagreement: Spearman rho = {rho:.3f} "
          f"on {ok.sum()} moves")

    gate_rows = {}
    print(f"\n{'gate (deg)':>11}{'coverage':>10}{'emitted':>9}{'max em.':>9}"
          f"{'jitter-gate cov':>17}{'jitter-gate max':>17}")
    for tau in GATES:
        keep = d <= tau
        # the free gate: refuse moves whose own jitter is in the worst decile
        thr_j = np.nanpercentile(j, 90) if ok.any() else np.nan
        keep_j = ok & (j <= thr_j)
        gate_rows[str(int(tau))] = {
            "coverage": round(float(keep.mean()), 3),
            "emitted": int(keep.sum()),
            "max_disagreement_emitted_deg": round(float(d[keep].max()), 2) if keep.any() else None,
            "jitter_gate_coverage": round(float(keep_j.mean()), 3),
            "jitter_gate_max_disagreement_deg": round(float(d[keep_j].max()), 2) if keep_j.any() else None,
        }
        print(f"{tau:>11.0f}{keep.mean():>9.1%}{keep.sum():>9d}"
              f"{(d[keep].max() if keep.any() else float('nan')):>9.1f}"
              f"{keep_j.mean():>16.1%}"
              f"{(d[keep_j].max() if keep_j.any() else float('nan')):>17.1f}")

    # The product question, stated as a number: at what gate does "your punch is
    # 20 deg short" become a sentence the measurement supports?
    # Recommended gate: the strictest grid point that still emits most moves with a
    # clear margin under the shipped tolerance. Margin is stated, not assumed.
    tol = float(TOLERANCE_DEG["L_elbow"])
    tau = 10.0
    supported = d <= tau
    margin = tol - float(d[supported].max())
    verdict = (f"Recommended gate {tau:.0f} deg: {supported.mean():.1%} of elbow moves are "
               f"emitted and the worst emitted reading disagrees by "
               f"{d[supported].max():.1f} deg, a {margin:.1f} deg margin under the {tol:.0f} deg "
               f"tolerance. Without the gate, {np.mean(d > tol):.1%} of moves disagree by more "
               f"than the tolerance itself and would be reported as athlete error.")
    payload_extra = {"recommended_gate_deg": tau, "tolerance_deg": tol,
                     "margin_deg": round(margin, 2)}

    payload = {"moves": len(d), "clips": len(rows), "per_move": rows,
               "disagreement_percentiles_deg":
                   {f"p{q}": round(float(np.percentile(d, q)), 2) for q in (50, 75, 90, 95, 99)},
               "disagreement_max_deg": round(float(d.max()), 2),
               "share_above_10_deg": round(float(np.mean(d > 10)), 4),
               "share_above_20_deg": round(float(np.mean(d > 20)), 4),
               "free_proxy": {"name": "within-move median |frame-to-frame d(elbow)|",
                              "spearman_vs_disagreement": round(rho, 4), "n": int(ok.sum())},
               "gates": gate_rows, **payload_extra, "verdict": verdict}
    with open("out/elbow_gate.json", "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\n{verdict}\nwrote out/elbow_gate.json")


if __name__ == "__main__":
    main()
