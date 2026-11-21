"""Re-run the camera-angle benchmark in 3D, with the controls that make it honest.

The 2D pipeline measures 10.19 degrees of mean deviation and flags 133 of 156 moves
when a frontal reference is compared with a lateral take of the SAME athlete. Two
model-based fixes failed (learned_gate.py, view_calibration.py), which pointed at
the feature space: a 2D projection discards depth, so every sagittal-plane angle
moves when the camera does.

This module asks the question directly: does MediaPipe's metric 3D
(`pose_world_landmarks`, hip-centred, y-up) reduce cross-view deviation WITHOUT
reducing sensitivity?

That second half is the whole test. A feature set that deviates less because it
responds less to everything is not an improvement, it is a quieter failure. So
three numbers are reported together:

  1. cross-view mean |delta| and flag rate   -- must go DOWN
  2. same-view control                        -- must stay ~0
  3. injected-fault recall at 10/20/25 deg    -- must stay HIGH
  4. per-feature dynamic range (p95-p5)       -- must not collapse

If 1 goes down and 3 or 4 collapses, the answer is no and it is recorded as no.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (ANGLE_TRIPLETS, FEATURE_NAMES, HUMAN, J, JOINTS,
                          TOLERANCE_DEG, dtw, features, load_dataset,
                          score_moves, segment_moves, smooth, _unit)
from poomsae.pose3d import FORMS, load3d, out_path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")
MAGS = (10.0, 20.0, 25.0)
TOL = np.array([TOLERANCE_DEG.get(f, 15.0) for f in FEATURE_NAMES])


def features3d(w: np.ndarray, vis: np.ndarray, min_vis: float = 0.5,
               smooth_w: int = 5) -> np.ndarray:
    """(T,12,3) metre coords -> (T,9) degrees. Missing joints interpolated, never dropped."""
    c = w.astype(float).copy()
    bad = (vis < min_vis) | (np.abs(c).sum(-1) < 1e-9)
    if bad.any():
        for j in range(c.shape[1]):
            ok = ~bad[:, j]
            if ok.sum() >= 2:
                for d in range(3):
                    c[:, j, d] = np.interp(np.arange(len(c)), np.flatnonzero(ok), c[ok, j, d])
    feats = []
    for _, a, b, cc in ANGLE_TRIPLETS:
        u = _unit(c[:, J[a]] - c[:, J[b]])
        v = _unit(c[:, J[cc]] - c[:, J[b]])
        feats.append(np.degrees(np.arccos(np.clip((u * v).sum(-1), -1.0, 1.0))))
    hip_mid = (c[:, J["L_hip"]] + c[:, J["R_hip"]]) / 2
    sho_mid = (c[:, J["L_shoulder"]] + c[:, J["R_shoulder"]]) / 2
    spine = _unit(sho_mid - hip_mid)
    # lean against the world vertical (+y up), not the image vertical
    feats.append(np.degrees(np.arccos(np.clip(spine[:, 1], -1.0, 1.0))))
    return smooth(np.stack(feats, axis=1), smooth_w)


def analyse_f(rf: np.ndarray, sf: np.ndarray, threshold=None):
    """core.analyse() but taking feature matrices, so 2D and 3D share one scorer."""
    cost, path = dtw(rf, sf)
    segs_ref = segment_moves(rf)
    j_of_i = {}
    for i, j in path:
        j_of_i.setdefault(i, j)
    segs_stu = []
    for r0, r1 in segs_ref:
        js = [j_of_i[i] for i in range(r0, r1 + 1) if i in j_of_i] or [0]
        segs_stu.append((min(js), max(js)))
    moves = score_moves(rf, sf, segs_ref, segs_stu, threshold=threshold)
    flags = sum(1 for m in moves if m["flags"])
    return {"cost": round(float(cost), 4), "n_moves": len(moves), "moves": moves,
            "flagged": flags,
            "mean_abs_delta": round(float(np.mean([m["mean_abs_delta"] for m in moves])), 2)}


def dyn_range(F: np.ndarray) -> dict:
    return {f: round(float(np.percentile(F[:, i], 95) - np.percentile(F[:, i], 5)), 1)
            for i, f in enumerate(FEATURE_NAMES)}


def main() -> int:
    # ---------------- 2D baseline, from the same clips through the shipped extractor
    ds = load_dataset(os.path.join(HERE, "data", "Data"))
    base2d = {"cross": [], "same": []}
    for form in FORMS:
        rf = features(ds[(form, "Frontal")])
        lf = features(ds[(form, "Lateral")])
        base2d["same"].append(analyse_f(rf, rf))
        base2d["cross"].append(analyse_f(rf, lf))
    d2 = {"same_mean": round(float(np.mean([r["mean_abs_delta"] for r in base2d["same"]])), 2),
          "same_flags": int(sum(r["flagged"] for r in base2d["same"])),
          "cross_mean": round(float(np.mean([r["mean_abs_delta"] for r in base2d["cross"]])), 2),
          "cross_flags": int(sum(r["flagged"] for r in base2d["cross"])),
          "n_moves": int(sum(r["n_moves"] for r in base2d["cross"]))}

    # ---------------- 3D
    realign = []
    per_form, cross, same, kept = [], [], [], {m: [] for m in MAGS}
    rng3, rng2 = [], []
    for form in FORMS:
        fw, fv = load3d(out_path(form, "Frontal"))
        lw, lv = load3d(out_path(form, "Lateral"))
        F3, L3 = features3d(fw, fv), features3d(lw, lv)
        F2 = features(ds[(form, "Frontal")])
        n = min(len(F3), len(F2), len(fw))
        rng3.append(dyn_range(F3[:n]))
        rng2.append(dyn_range(F2[:n]))
        s = analyse_f(F3, F3)
        c = analyse_f(F3, L3)
        same.append(s)
        cross.append(c)
        # Injected-fault recall. A constant added to one feature over one segment
        # shifts that segment's mean by exactly that constant, so the effect on the
        # per-move delta is analytic: d_new = d_base + m*e_j. Verified against real
        # re-alignment below on a subsample, and previously to 5.7e-14 deg in 2D.
        for k, mv in enumerate(c["moves"]):
            for ji, key in enumerate(FEATURE_NAMES):
                base = mv["deltas"][key]
                for m in MAGS:
                    dnew = base + m
                    kept[m].append(1.0 if abs(dnew) / TOL[ji] > 1.0 else 0.0)
        if len(per_form) < 3:                    # realignment check on a sample
            segs3 = segment_moves(F3)
            rng = np.random.default_rng(len(per_form))
            for _ in range(40):
                k = int(rng.integers(0, min(len(segs3), len(c["moves"]))))
                ji = int(rng.integers(0, len(FEATURE_NAMES)))
                m = float(rng.choice(MAGS))
                L2 = L3.copy()
                s0, s1 = segs3[k]
                L2[s0:s1 + 1, ji] += m
                rr = analyse_f(F3, L2)
                hit_real = any(f["key"] == FEATURE_NAMES[ji]
                               for f in rr["moves"][min(k, len(rr["moves"]) - 1)]["flags"])
                hit_an = abs(c["moves"][k]["deltas"][FEATURE_NAMES[ji]] + m) / TOL[ji] > 1.0
                realign.append(1.0 if hit_real == hit_an else 0.0)
        per_form.append({"form": form, "n_moves": c["n_moves"],
                         "same_view_mean_abs": s["mean_abs_delta"],
                         "cross_view_mean_abs": c["mean_abs_delta"],
                         "cross_view_flags": c["flagged"]})
        print(f"{form:<14} moves {c['n_moves']:>3}  same {s['mean_abs_delta']:>5.2f}deg  "
              f"cross {c['mean_abs_delta']:>5.2f}deg  flags {c['flagged']}/{c['n_moves']}")

    d3 = {"same_mean": round(float(np.mean([r["mean_abs_delta"] for r in same])), 2),
          "same_flags": int(sum(r["flagged"] for r in same)),
          "cross_mean": round(float(np.mean([r["mean_abs_delta"] for r in cross])), 2),
          "cross_flags": int(sum(r["flagged"] for r in cross)),
          "n_moves": int(sum(r["n_moves"] for r in cross))}

    r3 = {f: round(float(np.mean([d[f] for d in rng3])), 1) for f in FEATURE_NAMES}
    r2 = {f: round(float(np.mean([d[f] for d in rng2])), 1) for f in FEATURE_NAMES}

    print("\n--- 2D vs 3D, same clips, same scorer ---")
    print(f"same-view control  2D {d2['same_mean']:>6.2f}deg {d2['same_flags']:>3} flags | "
          f"3D {d3['same_mean']:>6.2f}deg {d3['same_flags']:>3} flags")
    print(f"cross-view         2D {d2['cross_mean']:>6.2f}deg {d2['cross_flags']:>3}/{d2['n_moves']} | "
          f"3D {d3['cross_mean']:>6.2f}deg {d3['cross_flags']:>3}/{d3['n_moves']}")
    print("\nper-feature dynamic range p95-p5 (deg) -- must not collapse if 3D deviates less:")
    for f in FEATURE_NAMES:
        print(f"  {f:<12} 2D {r2[f]:>6.1f}   3D {r3[f]:>6.1f}")
    ratio = float(np.mean([r3[f] / max(r2[f], 1e-6) for f in FEATURE_NAMES]))
    print(f"  mean 3D/2D range ratio {ratio:.2f}")
    print("\ninjected-fault recall, 3D cross-view, one move at a time:")
    rec = {f"{m:.0f}deg": round(float(np.mean(kept[m])), 4) for m in MAGS}
    for m in MAGS:
        print(f"  +/-{m:.0f}deg  {100*rec[f'{m:.0f}deg']:5.1f}%")
    agree = round(float(np.mean(realign)), 4) if realign else None
    print(f"\nanalytic-vs-realignment agreement: {100*agree:.1f}% of {len(realign)} sampled trials"
          if agree is not None else "\nno realignment sample")

    better = d3["cross_mean"] < d2["cross_mean"]
    verdict = ("3D reduces cross-view deviation and keeps sensitivity"
               if better and ratio > 0.9 and rec["20deg"] > 0.9 else
               "3D reduces cross-view deviation but loses sensitivity or range -- not a win"
               if better else "3D does not reduce cross-view deviation -- no benefit")
    print(f"\nVERDICT: {verdict}")

    sig2 = float(np.mean(list(r2.values())))
    sig3 = float(np.mean(list(r3.values())))
    noise2 = d2["cross_mean"] - d2["same_mean"]
    noise3 = d3["cross_mean"] - d3["same_mean"]
    payload = {"protocol": "frontal reference vs lateral take, same athlete, same clips",
               "baseline_2d": d2, "measured_3d": d3, "per_form_3d": per_form,
               "feature_range_2d": r2, "feature_range_3d": r3,
               "range_ratio_3d_over_2d": round(ratio, 3),
               "range_ratio_of_means": round(sig3 / sig2, 3),
               "signal_noise": {"signal_2d": round(sig2, 2), "signal_3d": round(sig3, 2),
                                "noise_2d": round(noise2, 2), "noise_3d": round(noise3, 2),
                                "snr_2d": round(sig2 / max(noise2, 1e-6), 2),
                                "snr_3d": round(sig3 / max(noise3, 1e-6), 2),
                                "note": ("signal = mean per-feature dynamic range p95-p5; "
                                         "noise = cross-view minus same-view deviation. "
                                         "This is the sensitivity verdict, not the "
                                         "injection_recall_3d below.")},
               "injection_recall_3d": rec,
               "injection_protocol_note": (
                   "measured on the CROSS-VIEW pair, where most moves are already flagged "
                   "by the camera artifact, so it is not comparable to the 2D same-view "
                   "injection test (out/injection_test.json). Do not read it as a "
                   "sensitivity win or loss; use signal_noise."),
               "analytic_vs_realignment_agreement": agree,
               "verdict": verdict}
    with open(os.path.join(OUT, "crossview_3d.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("wrote out/crossview_3d.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
