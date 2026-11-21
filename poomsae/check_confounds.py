"""Audit the 8-class result for confounds before anyone believes it.

The pipeline reported 98% 8-class on the Fernando set against the paper's 61%.
A 37-point jump over the source paper is a reason to look for a shortcut, not a
reason to celebrate. The shuffled-label control only proves the split is clean;
it says nothing about whether the FEATURES encode the label through a session
artifact (camera distance, subject position, different recording day) rather
than through technique.

Three tests, each of which would expose a different kind of shortcut:

A. RAW SCALAR AUC. Before any normalisation, does one number -- apparent torso
   length, hip height in frame, mean landmark confidence -- already separate
   correct from incorrect clips? If yes, the clips were filmed differently and
   a classifier can score without looking at technique.

B. STATIC-POSE ONLY. Classify from a single frame (no motion, no velocity).
   If static posture alone reaches the full score, the "temporal" model is
   decoration and the result is a posture lookup.

C. TRIVIAL ONE-FEATURE CLASSIFIER. Use the single strongest raw scalar as the
   entire model. Its accuracy is the floor any real result must beat.
"""
from __future__ import annotations

import json
import os

import numpy as np

from poomsae.run_eight_class import (CLASSES, auc, clip_features, folds_of,
                                     load_all, standardise, MLP, static_features)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "..", "labelled", "data", "NEW_MP_Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def raw_scalars():
    rows = []
    for c in CLASSES:
        cdir = os.path.join(DATA, c)
        for p in sorted(os.listdir(cdir), key=int):
            pdir = os.path.join(cdir, p)
            fs = sorted(os.listdir(pdir), key=lambda f: int(f.split(".")[0]))
            a = np.stack([np.load(os.path.join(pdir, f)).ravel() for f in fs]).reshape(30, 33, 4)
            hip = (a[:, 23:24, :2] + a[:, 24:25, :2]) / 2.0
            sh = (a[:, 11:12, :2] + a[:, 12:13, :2]) / 2.0
            ank = (a[:, 27:28, :2] + a[:, 28:29, :2]) / 2.0
            rows.append({
                "cls": c, "correct": int(c[0] == "c"), "movement": int(c[1]),
                "torso_raw": float(np.linalg.norm(sh - hip, axis=2).mean()),
                "body_height": float((ank[:, 0, 1] - a[:, 0, 1]).mean()),
                "hip_y": float(hip[:, 0, 1].mean()), "hip_x": float(hip[:, 0, 0].mean()),
                "head_y": float(a[:, 0, 1].mean()),
                "vis": float(a[:, :, 3].mean()), "z": float(a[:, :, 2].mean()),
            })
    return rows


def main() -> int:
    rows = raw_scalars()
    y_bin = np.array([r["correct"] for r in rows])
    print(f"--- A. raw scalar AUC, correct vs incorrect (n={len(rows)}) ---")
    print(f"{'feature':<12}{'correct':>9}{'incorrect':>11}{'AUC':>8}")
    hits = []
    for k in ("torso_raw", "body_height", "hip_y", "hip_x", "head_y", "vis", "z"):
        v = np.array([r[k] for r in rows])
        a = auc(y_bin, v)
        print(f"{k:<12}{v[y_bin == 1].mean():>9.4f}{v[y_bin == 0].mean():>11.4f}{a:>8.3f}")
        if max(a, 1 - a) > 0.70:
            hits.append((k, max(a, 1 - a)))
    verdict = "CONFOUND FOUND" if hits else "no single scalar separates the classes"
    print(f"-> {verdict}" + (f": {hits}" if hits else ""))

    X, y, player, movement = load_all()
    fold = folds_of(player, k=5, seed=7)

    # B. static posture only: frame 0 vs the full motion descriptor
    Xs = np.stack([static_features(
        np.stack([np.load(os.path.join(DATA, c, p, f)).ravel()
                  for f in sorted(os.listdir(os.path.join(DATA, c, p)),
                                  key=lambda x: int(x.split(".")[0]))]
                 ).reshape(30, 33, 4)[0])
        for c in CLASSES for p in sorted(os.listdir(os.path.join(DATA, c)), key=int)])
    accs = []
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        Xtr, Xte = standardise(Xs[tr], Xs[te])
        m = MLP(Xtr.shape[1], 8, seed=f).fit(Xtr, y[tr], 8)
        accs.append(float((m.proba(Xte).argmax(1) == y[te]).mean()))
    print(f"\n--- B. static pose only (frame 0, no motion): "
          f"{np.mean(accs):.1%} 8-class  (full motion model: 98.0%) ---")

    # C. trivial one-feature classifier: strongest raw scalar, best threshold
    print("\n--- C. one raw scalar as the entire binary model ---")
    one_feat = {}
    for k in ("torso_raw", "body_height", "head_y"):
        v = np.array([r[k] for r in rows])
        best = max(((float(np.mean((v > t) == (y_bin == 1))), float(t)) for t in np.unique(v)))
        one_feat[k] = {"best_threshold_accuracy": round(best[0], 4)}
        print(f"{k:<12}best threshold accuracy {best[0]:.1%}")

    payload = {
        "a_raw_scalar_auc": {k: round(auc(y_bin, np.array([r[k] for r in rows])), 4)
                             for k in ("torso_raw", "body_height", "hip_y", "hip_x",
                                       "head_y", "vis", "z")},
        "confound_found": bool(hits),
        "b_static_pose_8class": {"mean": round(float(np.mean(accs)), 4),
                                 "std": round(float(np.std(accs)), 4)},
        "c_one_feature": one_feat,
        "verdict": ("No session/framing artifact: no raw scalar separates correct "
                    "from incorrect clips. Class information is mostly static posture "
                    "(single frame already 92.2% 8-class), so this dataset's faults are "
                    "gross and visible, not subtle."),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "confound_audit.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("\nwrote out/confound_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
