"""Between-athlete variability of the same technique, from data already on disk.

Why this matters: the coach card compares a student against a reference performed by one
person. If the same technique legitimately varies by ~20 deg across 94 athletes, then a 21.5 deg
deviation from that reference is not evidence of a fault — it is within normal human spread.
That is the question the 18 deg elbow tolerance cannot answer on its own, and it needs many
athletes, not more takes of one.

Input: labelled/data/NEW_MP_Data  (8 classes x 94 subject dirs x 30 frames, 132 = 33 MediaPipe
landmarks x (x, y, z, visibility)).
Per subject and class:
  * take the ACTIVE limb = the side whose elbow angle travels furthest through the clip
  * peak frame = the frame of maximum elbow extension (the technique's kime moment)
  * record elbow and knee angle at that frame
Then report: per-class cross-athlete mean / sd / p5-p95 of the peak angle.
sd across athletes is an UPPER bound on legitimate variation and a LOWER bound on how much a
single-performer reference can mislead — it includes technique, body type and stance differences.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.expanduser("~/HERMES/research/taekwondo-ai/labelled/data/NEW_MP_Data")
# MediaPipe pose landmark indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28


def ang(a, b, c):
    """Angle at b, in degrees, from 2D xy coordinates."""
    v1, v2 = a[:2] - b[:2], c[:2] - b[:2]
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def side_angles(clip: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray]:
    sh, el, wr, hip, kn, an = (
        (L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN) if side == "L" else (R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN)
    )
    el_a = np.array([ang(f[sh * 4:sh * 4 + 4], f[el * 4:el * 4 + 4], f[wr * 4:wr * 4 + 4]) for f in clip])
    kn_a = np.array([ang(f[hip * 4:hip * 4 + 4], f[kn * 4:kn * 4 + 4], f[an * 4:an * 4 + 4]) for f in clip])
    return el_a, kn_a


def main() -> int:
    classes = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
    out = {"root": ROOT, "generated": "2026-11-21",
           "definition": "peak-frame joint angle per athlete, active limb, 2D xy",
           "classes": {}, "pooled": {}}
    per_class_elbow, per_class_knee = {}, {}

    for c in classes:
        subs = sorted(d for d in os.listdir(os.path.join(ROOT, c)) if os.path.isdir(os.path.join(ROOT, c, d)))
        elbow_peak, knee_peak, used = [], [], []
        for s in subs:
            files = sorted(glob.glob(os.path.join(ROOT, c, s, "*.npy")), key=lambda x: int(os.path.basename(x)[:-4]))
            if len(files) < 10:
                continue
            clip = np.stack([np.load(f) for f in files])
            best = None
            for side in ("L", "R"):
                el_a, kn_a = side_angles(clip, side)
                if np.all(np.isnan(el_a)):
                    continue
                span = float(np.nanmax(el_a) - np.nanmin(el_a))
                if best is None or span > best[0]:
                    best = (span, el_a, kn_a, side)
            if best is None:
                continue
            span, el_a, kn_a, side = best
            i = int(np.nanargmax(el_a))          # the extension peak = the technique's kime frame
            elbow_peak.append(el_a[i])
            knee_peak.append(kn_a[i])
            used.append(side)

        def stat(v, name):
            v = np.asarray([x for x in v if x == x], dtype=float)
            if v.size < 5:
                return None
            return {"n": int(v.size), "mean": round(float(v.mean()), 1),
                    "sd": round(float(v.std(ddof=1)), 1),
                    "p5": round(float(np.percentile(v, 5)), 1),
                    "p95": round(float(np.percentile(v, 95)), 1),
                    "range": round(float(v.max() - v.min()), 1)}

        out["classes"][c] = {"n_athletes": len(elbow_peak),
                             "elbow_peak": stat(elbow_peak, "elbow"),
                             "knee_peak": stat(knee_peak, "knee"),
                             "active_side_L": used.count("L"), "active_side_R": used.count("R")}
        per_class_elbow[c] = [x for x in elbow_peak if x == x]
        per_class_knee[c] = [x for x in knee_peak if x == x]

    # ---- trajectory variability: resample every clip to 100 phase points, then sd across athletes
    # The peak-angle table above saturates near 180 deg because max-extraction selects the straightest
    # frame in each clip; it therefore cannot measure between-athlete variability. This block can.
    out["trajectory"] = {"definition": "per-class joint-angle trajectory resampled to 100 phase points; "
                                      "sd computed across athletes at each phase point, then averaged",
                         "classes": {}}
    for c in classes:
        subs = sorted(d for d in os.listdir(os.path.join(ROOT, c)) if os.path.isdir(os.path.join(ROOT, c, d)))
        el_curves, kn_curves = [], []
        for s in subs:
            files = sorted(glob.glob(os.path.join(ROOT, c, s, "*.npy")), key=lambda x: int(os.path.basename(x)[:-4]))
            if len(files) < 10:
                continue
            clip = np.stack([np.load(f) for f in files])
            best = None
            for side in ("L", "R"):
                el_a, kn_a = side_angles(clip, side)
                if np.all(np.isnan(el_a)):
                    continue
                span = float(np.nanmax(el_a) - np.nanmin(el_a))
                if best is None or span > best[0]:
                    best = (span, el_a, kn_a)
            if best is None:
                continue
            _, el_a, kn_a = best
            grid = np.linspace(0, len(el_a) - 1, 100)
            el_curves.append(np.interp(grid, np.arange(len(el_a)), el_a))
            kn_curves.append(np.interp(grid, np.arange(len(kn_a)), kn_a))
        if len(el_curves) < 5:
            continue
        E = np.vstack(el_curves)
        K = np.vstack(kn_curves)
        el_sd_phase = np.nanstd(E, axis=0, ddof=1)
        kn_sd_phase = np.nanstd(K, axis=0, ddof=1)
        out["trajectory"]["classes"][c] = {
            "n_athletes": int(E.shape[0]),
            "elbow_traj_sd_mean": round(float(np.nanmean(el_sd_phase)), 1),
            "elbow_traj_sd_max": round(float(np.nanmax(el_sd_phase)), 1),
            "elbow_range_mean": round(float(np.nanmean(np.nanmax(E, 1) - np.nanmin(E, 1))), 1),
            "knee_traj_sd_mean": round(float(np.nanmean(kn_sd_phase)), 1),
            "knee_traj_sd_max": round(float(np.nanmax(kn_sd_phase)), 1),
            "knee_range_mean": round(float(np.nanmean(np.nanmax(K, 1) - np.nanmin(K, 1))), 1),
        }
    tc = out["trajectory"]["classes"]
    if tc:
        out["trajectory"]["pooled"] = {
            "elbow_traj_sd_mean": round(float(np.mean([v["elbow_traj_sd_mean"] for v in tc.values()])), 1),
            "elbow_traj_sd_max": round(float(np.max([v["elbow_traj_sd_max"] for v in tc.values()])), 1),
            "knee_traj_sd_mean": round(float(np.mean([v["knee_traj_sd_mean"] for v in tc.values()])), 1),
            "knee_traj_sd_max": round(float(np.max([v["knee_traj_sd_max"] for v in tc.values()])), 1),
        }

    # pooled WITHIN-class spread: average of per-class sds (does not mix technique means)
    for name, d in (("elbow", per_class_elbow), ("knee", per_class_knee)):
        sds = [float(np.std(v, ddof=1)) for v in d.values() if len(v) > 5]
        out["pooled"][name] = {"mean_within_class_sd": round(float(np.mean(sds)), 1) if sds else None,
                               "max_within_class_sd": round(float(np.max(sds)), 1) if sds else None,
                               "classes": len(sds)}
    json.dump(out, open("/tmp/between_athlete.json", "w"), indent=1)
    print("wrote /tmp/between_athlete.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
