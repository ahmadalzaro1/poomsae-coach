"""Learned gate: tell a real fault apart from a camera-side artifact.

The measured weakness of the shipped scorer is in `out/crossview_benchmark.json`:
comparing a frontal reference against a lateral take of the SAME athlete flags
133 of 156 moves, while the same-view control flags 0. Those flags are camera
geometry, not athlete error, and the rule-based scorer cannot tell the two apart
because it only looks at how big each deviation is.

So the question this module answers is narrow and testable:

    given the per-joint deviation vector for one move, is this a FAULT
    (localised, one joint, one direction) or a CAMERA ARTIFACT (diffuse,
    symmetric, both limbs pushed the same way)?

Data comes from the two ground truths we actually own:

  NEGATIVES -- cross-view deviation of the same athlete. Known to contain no
               fault. 8 forms.
  POSITIVES -- cross-view deviation plus a known fault at a known joint,
               injected at a known magnitude. Built analytically: adding m to a
               feature column across a whole segment shifts that segment's mean
               by exactly m, so d_fault = d_camera + m * e_j. This is verified
               against real re-alignment injection before use.

Split is leave-one-form-out: a form is never in both train and test.

Three models are compared so the structure features have to earn their place:
  magnitude only   -- the rule's own signal (max |delta| / tolerance)
  structure only   -- shape of the deviation, magnitude deliberately excluded
  full             -- both

If the full model does not beat the magnitude-only baseline, the honest
conclusion is that this problem is not separable and the rule stays.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (FEATURE_NAMES, TOLERANCE_DEG, dtw, features,
                          load_dataset, segment_moves)

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
MAGNITUDES = (10.0, 15.0, 20.0, 25.0, 35.0)
PAIRS = [("L_elbow", "R_elbow"), ("L_knee", "R_knee"), ("L_hip", "R_hip")]
TOL = np.array([TOLERANCE_DEG.get(f, 15.0) for f in FEATURE_NAMES])
I = {f: i for i, f in enumerate(FEATURE_NAMES)}

MAGNITUDE_FEATS = ["a_max", "a_mean", "a_count_half"]
STRUCTURE_FEATS = [
    "concentration",          # how much of the total sits in one joint
    "a_2nd_over_max",         # is there a clear runner-up (diffuse) or not (localised)
    "arm_energy", "leg_energy", "torso_energy",
    "torso_share",            # camera angle lives in torso tilt more than a limb fault
    "elbow_sym_min", "elbow_sym_max",
    "knee_sym_min", "knee_sym_max",
    "elbow_same_sign", "knee_same_sign", "hip_same_sign",
    "arm_signed_sum",         # camera flip pushes both arms the same direction
    "n_joints_above_1tol",
]
ALL_FEATS = MAGNITUDE_FEATS + STRUCTURE_FEATS


def move_deltas(d: np.ndarray) -> list[float]:
    """Deviation vector (degrees) -> gate features."""
    a = np.abs(d) / TOL
    order = np.sort(a)[::-1]
    a_max = float(order[0])
    a_sum = float(a.sum())
    out = [a_max, float(a.mean()), float((a > 0.5).sum()),
           a_max / (a_sum + 1e-6), float(order[1] / (order[0] + 1e-6))]
    arm = a[I["L_elbow"]] + a[I["R_elbow"]]
    leg = a[I["L_knee"]] + a[I["R_knee"]] + a[I["L_hip"]] + a[I["R_hip"]]
    torso = a[I["torso_tilt"]]
    tot = arm + leg + torso + 1e-6
    out += [arm, leg, torso, torso / tot]
    for l, r in PAIRS[:2]:                        # elbow pair, knee pair
        dl, dr = a[I[l]], a[I[r]]
        out += [min(dl, dr), max(dl, dr)]
    for l, r in PAIRS:                            # sign agreement, 3 pairs
        out.append(1.0 if np.sign(d[I[l]]) == np.sign(d[I[r]]) else 0.0)
    out.append(float(d[I["L_elbow"]] + d[I["R_elbow"]]))
    out.append(float((a > 1.0).sum()))
    return out


def form_moves(seq_ref, seq_stu, cache=None):
    """Per-move camera-artifact deltas: (n_moves, 9) for one form/view pair."""
    rf = features(seq_ref)
    sf = features(seq_stu)
    cost, path = dtw(rf, sf)
    segs_ref = segment_moves(rf)
    j_of_i = {}
    for i, j in path:
        j_of_i.setdefault(i, j)
    segs_stu = []
    for r0, r1 in segs_ref:
        js = [j_of_i[i] for i in range(r0, r1 + 1) if i in j_of_i] or [0]
        segs_stu.append((min(js), max(js)))
    d = np.array([sf[s0:s1 + 1].mean(0) - rf[r0:r1 + 1].mean(0)
                  for (r0, r1), (s0, s1) in zip(segs_ref, segs_stu)])
    return d, rf, sf, segs_ref, segs_stu


class LogReg:
    """L2 logistic regression with class weights; coefficients stay readable."""

    def __init__(self, l2=1.0, lr=0.2, epochs=3000):
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def fit(self, X, y):
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        pos_w = n / (2.0 * max(int(y.sum()), 1))
        neg_w = n / (2.0 * max(int((1 - y).sum()), 1))
        sw = np.where(y > 0.5, pos_w, neg_w)
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
            g = sw * (p - y)
            w -= self.lr * (X.T @ g / n + self.l2 * w / n)
            b -= self.lr * g.mean()
        self.w, self.b = w, b
        return self

    def proba(self, X):
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))


def auc(y, s):
    y = np.asarray(y)
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> int:
    ds = load_dataset(ROOT)
    forms = sorted({f for f, _ in ds})
    print(f"{len(forms)} forms, leave-one-form-out\n")

    # ---- verify the analytic shortcut against real re-alignment injection
    d_real, rf, sf, segs_ref, segs_stu = form_moves(ds[(forms[0], "Frontal")],
                                                    ds[(forms[0], "Lateral")])
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(12):
        k = int(rng.integers(0, len(d_real)))
        ji = int(rng.integers(0, len(FEATURE_NAMES)))
        m = float(rng.choice(MAGNITUDES))
        s2 = sf.copy()
        s2[segs_stu[k][0]:segs_stu[k][1] + 1, ji] += m
        d2 = s2[segs_stu[k][0]:segs_stu[k][1] + 1].mean(0) - rf[segs_ref[k][0]:segs_ref[k][1] + 1].mean(0)
        errs.append(float(np.abs(d2 - (d_real[k] + m * np.eye(len(FEATURE_NAMES))[ji])).max()))
    print(f"analytic-injection check: max error {max(errs):.2e}° "
          f"-> {'PASS' if max(errs) < 1e-9 else 'FAIL'}")

    # ---- assemble out-of-fold predictions
    X_all, y_all, form_all, mag_all = [], [], [], []
    for fi, form in enumerate(forms):
        d_cam, *_ = form_moves(ds[(form, "Frontal")], ds[(form, "Lateral")])
        for k in range(len(d_cam)):
            X_all.append(move_deltas(d_cam[k]))
            y_all.append(0)
            form_all.append(fi)
            mag_all.append(0.0)
            for ji in range(len(FEATURE_NAMES)):
                for m in MAGNITUDES:
                    d_pos = d_cam[k] + m * np.eye(len(FEATURE_NAMES))[ji]
                    X_all.append(move_deltas(d_pos))
                    y_all.append(1)
                    form_all.append(fi)
                    mag_all.append(m)
    X = np.array(X_all)
    y = np.array(y_all)
    form_all = np.array(form_all)
    mag_all = np.array(mag_all)
    print(f"rows: {len(y)} | negatives (camera artifact) {int((y == 0).sum())} | "
          f"positives (injected fault) {int((y == 1).sum())}")

    idx_mag = [ALL_FEATS.index(f) for f in MAGNITUDE_FEATS]
    idx_str = [ALL_FEATS.index(f) for f in STRUCTURE_FEATS]
    variants = {"magnitude_only": idx_mag, "structure_only": idx_str,
                "full": list(range(len(ALL_FEATS)))}
    scores = {}
    coefs = {}
    for name, cols in variants.items():
        p_out = np.zeros(len(y))
        for fi in range(len(forms)):
            tr, te = form_all != fi, form_all == fi
            Xtr, Xte = X[tr][:, cols], X[te][:, cols]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
            lr = LogReg().fit((Xtr - mu) / sd, y[tr])
            p_out[te] = lr.proba((Xte - mu) / sd)
            if name == "full" and fi == 0:
                coefs = {f: round(float(c), 3) for f, c in zip([ALL_FEATS[i] for i in cols], lr.w)}
        scores[name] = p_out
        print(f"{name:<15} pooled AUC {auc(y, p_out):.3f}")

    # ---- operating point vs the shipped rule
    print("\n--- operating point (threshold chosen so TPR at +/-25 deg matches the rule) ---")
    rule = (X[:, idx_mag[0]] > 1.0).astype(float)      # the rule: max exceedance > 1
    print(f"rule (max exceedance > 1): TPR@25 {100*rule[(y == 1) & (mag_all == 25)].mean():.1f}%"
          f" | FPR on camera artifacts {100*rule[y == 0].mean():.1f}%")
    report = {"rule": {"tpr_by_magnitude": {}, "fpr": round(float(rule[y == 0].mean()), 4)}}
    for m in MAGNITUDES:
        report["rule"]["tpr_by_magnitude"][f"{m:.0f}deg"] = round(
            float(rule[(y == 1) & (mag_all == m)].mean()), 4)

    for name in variants:
        p = scores[name]
        # threshold that keeps its own TPR@25 equal to the rule's
        target = float(rule[(y == 1) & (mag_all == 25)].mean())
        s25 = np.sort(p[(y == 1) & (mag_all == 25)])[::-1]
        thr = float(s25[min(int(target * len(s25)), len(s25) - 1)] - 1e-9)
        fpr = float((p[y == 0] > thr).mean())
        entry = {"threshold": round(thr, 3),
                 "fpr_at_matched_tpr": round(fpr, 4),
                 "tpr_by_magnitude": {f"{m:.0f}deg": round(float((p[(y == 1) & (mag_all == m)] > thr).mean()), 4)
                                      for m in MAGNITUDES}}
        report[name] = entry
        print(f"{name:<15} thr {thr:.3f}  FPR {100*fpr:5.1f}%  " +
              "  ".join(f"±{m:.0f}° {100*entry['tpr_by_magnitude'][f'{m:.0f}deg']:.1f}%" for m in MAGNITUDES))

    report["auc"] = {k: round(auc(y, v), 4) for k, v in scores.items()}
    report["coefficients_full"] = coefs
    report["features"] = ALL_FEATS
    report["n_negatives"] = int((y == 0).sum())
    report["n_positives"] = int((y == 1).sum())
    report["split"] = "leave-one-form-out over 8 forms"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "learned_gate.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("\nwrote out/learned_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
