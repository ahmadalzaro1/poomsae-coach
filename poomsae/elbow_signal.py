"""Elbow-signal upgrade: fix the wrist, and the elbow angle follows.

The measured problem, from out/pose_audit.json (our extraction vs the dataset's own
annotation): shoulders 3.4-4.2 px, hips 3.2-4.0 px, knees 5.7-6.6 px, but WRISTS
20.3-23.8 px. The elbow angle is shoulder-elbow-wrist, so its far endpoint is the
least accurate landmark in the skeleton -- which is why L_elbow/R_elbow carry a 25 deg
tolerance while every other joint sits at 5-12 deg.

This module tests four estimators on identical frames and picks on evidence:

  A pose_full      -- baseline, the shipped path (pose_landmarker_full)
  B pose_heavy     -- bigger pose model
  C heavy + hand   -- wrist replaced by the HandLandmarker wrist when it detects a
                      hand and the reading is geometrically plausible
  D heavy + hand centre -- wrist replaced by the hand's MCP centroid (5,9,13,17),
                      which is where the forearm actually points and is less
                      sensitive to fingertip pose than the wrist itself

Metrics that decide it:
  * accuracy vs the dataset annotation elbow angle (degrees) -- held out by form
  * hold-phase jitter (sigma) -- what a tolerance has to survive
  * hand-detection coverage -- an estimator that works on 30% of frames is a trap
  * side symmetry -- an upgrade that only helps one arm is not an upgrade

Tolerance is then DERIVED, not chosen: tol = max(5 deg, ceil(p95 error)) on Taegeuk 1
only, and validated on the other seven forms. Anything else is tuning to the answer.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (FEATURE_NAMES, features, load_dataset,  # noqa: E402
                          segment_moves)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")
MODELS = os.path.join(HERE, "models")
DATA = os.path.join(HERE, "data", "Data")
FORMS = [("Taegeuk_1st", "1st"), ("Taegeuk_2nd", "2nd"), ("Taegeuk_3rd", "3rd"),
         ("Taegeuk_4th", "4th"), ("Taegeuk_5th", "5th"), ("Taegeuk_6th", "6th"),
         ("Taegeuk_7th", "7th"), ("Taegeuk_8th", "8th")]
VIEWS = ("Frontal", "Lateral")

# MediaPipe pose landmark indices
LS, RS, LE, RE, LW, RW = 11, 12, 13, 14, 15, 16
POSE_NAMES = {LS: "L_shoulder", RS: "R_shoulder", LE: "L_elbow", RE: "R_elbow",
              LW: "L_wrist", RW: "R_wrist"}
HAND_MCPS = (5, 9, 13, 17)
MIN_HAND_SCORE = 0.3
# A hand reading is accepted only if it sits plausibly near the pose wrist. Hands are
# ~36 px across in this full-body footage and the pose wrist itself carries 20-24 px
# of error, so the gate is rounded to max(45 px, 0.55 x forearm) rather than a tight
# multiple: a tight gate was rejecting 90% of good detections (5% coverage instead of
# the 52% the model actually achieves).
MAX_HAND_GAP_PX = 45.0
MAX_HAND_GAP_RATIO = 0.55


def angle(a, b, c) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))


def build(model_path: str):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.PoseLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3, output_segmentation_masks=False)
    return vision.PoseLandmarker.create_from_options(opts)


def build_hands():
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1,
        min_tracking_confidence=0.1)
    return vision.HandLandmarker.create_from_options(opts)


def build_hands_img():
    """IMAGE-mode hand landmarker, for per-crop detection (no timestamps)."""
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=os.path.join(MODELS, "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=1,
        min_hand_detection_confidence=0.1, min_hand_presence_confidence=0.1)
    return vision.HandLandmarker.create_from_options(opts)


def extract_hands_cropped(video: str, hands_img, rec: dict, scale: float = 3.0) -> dict:
    """Second-stage hand pass on a crop around the pose wrist, upscaled.

    Full-frame detection finds a hand on only 5-15% of frames, because the hand is
    ~36 px across in a 1280x720 full-body shot -- far below what the model was trained
    on. Cropping to 2.2x the forearm length around the pose wrist and upscaling 3x puts
    the hand back at a normal scale. This is the cheap form of "a dedicated arm
    keypoint model": same network, sane input scale.
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(video)
    T = rec["T"]
    idx = [LS, RS, LE, RE, LW, RW]
    out = {"hand_wrist_crop": np.full((T, 2, 2), np.nan),
           "hand_centre_crop": np.full((T, 2, 2), np.nan),
           "hand_score_crop": np.zeros((T, 2))}
    for i in range(T):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        for side, (ws, es) in ((0, (LW, LE)), (1, (RW, RE))):
            pw = rec["pose_heavy"][i, idx.index(ws)]
            el = rec["pose_heavy"][i, idx.index(es)]
            if np.isnan(pw).any() or np.isnan(el).any():
                continue
            r = max(40.0, 1.1 * np.linalg.norm(pw - el))
            x0, y0 = int(pw[0] - r), int(pw[1] - r)
            x1, y1 = int(pw[0] + r), int(pw[1] + r)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 24 or y1 - y0 < 24:
                continue
            crop = frame[y0:y1, x0:x1]
            big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
            res = hands_img.detect(img)
            if not res.hand_landmarks:
                continue
            hl = res.hand_landmarks[0]
            pts = np.array([[p.x * big.shape[1] / scale + x0, p.y * big.shape[0] / scale + y0]
                            for p in hl])
            score = float(res.handedness[0][0].score) if res.handedness else 1.0
            forearm = np.linalg.norm(pw - el)
            if np.linalg.norm(pts[0] - pw) > max(MAX_HAND_GAP_PX, MAX_HAND_GAP_RATIO * forearm):
                continue
            if out["hand_score_crop"][i, side] >= score:
                continue
            out["hand_score_crop"][i, side] = score
            out["hand_wrist_crop"][i, side] = pts[0]
            out["hand_centre_crop"][i, side] = pts[list(HAND_MCPS)].mean(axis=0)
    cap.release()
    return out


def extract(video: str, full, heavy, hands, ts_base: int = 0) -> dict:
    """Per-frame landmarks from three sources, in pixel coordinates.

    ts_base carries the frame counter across clips: the three landmarkers are created
    once and reused, and MediaPipe rejects a timestamp that does not increase for the
    lifetime of the graph -- restarting at 0 on the next video raised
    "Input timestamp must be monotonically increasing".
    """
    import mediapipe as mp
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rec = {"T": n, "pose_full": np.full((n, 6, 2), np.nan),
           "pose_heavy": np.full((n, 6, 2), np.nan),
           "hand_wrist": np.full((n, 2, 2), np.nan),
           "hand_centre": np.full((n, 2, 2), np.nan),
           "hand_score": np.zeros((n, 2))}
    idx = [LS, RS, LE, RE, LW, RW]
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int((ts_base + i) * 33)
        for key, model in (("pose_full", full), ("pose_heavy", heavy)):
            res = model.detect_for_video(img, ts)
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                for slot, pi in enumerate(idx):
                    rec[key][i, slot] = (lm[pi].x * frame.shape[1], lm[pi].y * frame.shape[0])
        hres = hands.detect_for_video(img, ts)
        if hres.hand_landmarks:
            for hi, hl in enumerate(hres.hand_landmarks):
                pts = np.array([[p.x * frame.shape[1], p.y * frame.shape[0]] for p in hl])
                score = float(hres.handedness[hi][0].score) if hres.handedness else 1.0
                # assign to whichever pose wrist is nearer, refuse if implausibly far
                for side, wi in ((0, LW), (1, RW)):
                    slot = [LS, RS, LE, RE, LW, RW].index(wi)
                    pw = rec["pose_heavy"][i, slot]
                    el = rec["pose_heavy"][i, [LS, RS, LE, RE, LW, RW].index(LE if side == 0 else RE)]
                    if np.isnan(pw).any():
                        continue
                    gap = np.linalg.norm(pts[0] - pw)
                    forearm = np.linalg.norm(pw - el)
                    if gap > max(MAX_HAND_GAP_PX, MAX_HAND_GAP_RATIO * forearm):
                        continue
                    if rec["hand_score"][i, side] >= score:
                        continue
                    rec["hand_score"][i, side] = score
                    rec["hand_wrist"][i, side] = pts[0]
                    rec["hand_centre"][i, side] = pts[list(HAND_MCPS)].mean(axis=0)
        i += 1
    cap.release()
    rec["T"] = i
    rec["ts_base"] = ts_base + i
    return rec


def elbow_angles(rec: dict, wrist_mode: str) -> dict:
    """Elbow angle per side under one wrist choice. -1 side = left, +1 = right."""
    out = {}
    idx = [LS, RS, LE, RE, LW, RW]
    T = rec["T"]
    for side, (s_key, e_key, w_key, hs) in {
            0: (LS, LE, LW, 0), 1: (RS, RE, RW, 1)}.items():
        sh = rec["pose_heavy"][:T, idx.index(s_key)]
        el = rec["pose_heavy"][:T, idx.index(e_key)]
        if wrist_mode == "pose_full":
            sh = rec["pose_full"][:T, idx.index(s_key)]
            el = rec["pose_full"][:T, idx.index(e_key)]
            wr = rec["pose_full"][:T, idx.index(w_key)]
        elif wrist_mode == "pose_heavy":
            wr = rec["pose_heavy"][:T, idx.index(w_key)]
        else:
            table = {"heavy_hand_wrist": ("hand_wrist", "hand_score"),
                     "heavy_hand_centre": ("hand_centre", "hand_score"),
                     "heavy_hand_crop": ("hand_wrist_crop", "hand_score_crop"),
                     "heavy_centre_crop": ("hand_centre_crop", "hand_score_crop")}
            src_key, sc_key = table[wrist_mode]
            if src_key not in rec:
                ang = np.array([angle(sh[t], el[t], rec["pose_heavy"][t, idx.index(w_key)])
                                for t in range(T)])
                out[side] = ang
                continue
            src = rec[src_key]
            wr = np.where(rec[sc_key][:T, hs][:, None] >= MIN_HAND_SCORE,
                          src[:T, hs], rec["pose_heavy"][:T, idx.index(w_key)])
        ang = np.array([angle(sh[t], el[t], wr[t]) for t in range(T)])
        out[side] = ang
    return out


def annotation_elbows(form: str, view: str, ds) -> dict:
    seq = ds[(form, view)]
    px = seq.px
    out = {}
    for side, (si, ei, wi) in {0: ("L_shoulder", "L_elbow", "L_wrist"),
                               1: ("R_shoulder", "R_elbow", "R_wrist")}.items():
        from poomsae.core import J
        out[side] = np.array([angle(px[t, J[si]], px[t, J[ei]], px[t, J[wi]])
                              for t in range(len(px))])
    return out


def hold_sigma(ang: np.ndarray) -> float:
    """Noise in still phases: frame-to-frame change divided by sqrt(2)."""
    d = np.diff(ang)
    d = d[~np.isnan(d)]
    if d.size < 8:
        return float("nan")
    vel = np.abs(d)
    keep = vel <= np.percentile(vel, 20)
    return float(np.std(d[keep]) / np.sqrt(2.0))


def jump_rate(ang: np.ndarray, thresh: float = 40.0) -> float:
    """Fraction of frame-to-frame angle changes above what a body can do.

    A human elbow cannot travel 40 deg in 33 ms. Any such jump is an estimator
    failure, so this is a label-free quality metric: it needs no reference and cannot
    be gamed by matching a noisy annotation.
    """
    d = np.abs(np.diff(ang))
    d = d[~np.isnan(d)]
    return float(np.mean(d > thresh)) if d.size else float("nan")


def credible(ann: np.ndarray, thresh: float = 40.0) -> np.ndarray:
    """Frames where the ANNOTATION itself is physically plausible.

    Measured on Taegeuk 1 frontal: the annotation's own elbow angle jumps >40 deg on
    34 frames and up to 154 deg in a single frame. Scoring an estimator against those
    frames measures the label's noise, not the estimator's error --- which is how a
    good signal ends up looking like 53-76 deg p95 error.
    """
    m = np.ones(len(ann), bool)
    d = np.abs(np.diff(ann))
    bad = np.where(d > thresh)[0]
    m[bad] = False
    m[np.clip(bad + 1, 0, len(ann) - 1)] = False
    m &= ~np.isnan(ann)
    return m


def main() -> int:
    from poomsae.core import J  # noqa: F401  (kept for annotation_elbows import order)
    ok_models = all(os.path.exists(os.path.join(MODELS, f)) for f in
                    ("pose_landmarker_full.task", "pose_landmarker_heavy.task",
                     "hand_landmarker.task"))
    if not ok_models:
        print("missing models; expected pose_landmarker_full/heavy + hand_landmarker")
        return 1
    ds = load_dataset(DATA)
    full, heavy, hands = build(os.path.join(MODELS, "pose_landmarker_full.task")), \
        build(os.path.join(MODELS, "pose_landmarker_heavy.task")), build_hands()
    hands_img = build_hands_img()

    fdir = os.path.join(OUT, "elbow_signal")
    os.makedirs(fdir, exist_ok=True)
    rows = {}
    ts_base = 0
    cached = extracted = 0
    for form, ordinal in FORMS:
        for view in VIEWS:
            video = os.path.join(DATA, "video", form, view, f"{ordinal}_{view}.mp4")
            if not os.path.exists(video):
                print("skip (no video):", video)
                continue
            npz_path = os.path.join(fdir, f"{ordinal}_{view}.npz")
            if os.path.exists(npz_path) and os.environ.get("ELBOW_REEXTRACT") != "1":
                rec = {k: v for k, v in np.load(npz_path).items()}
                rec["T"] = int(rec["T"])
                cached += 1
            else:
                rec = extract(video, full, heavy, hands, ts_base)
                ts_base = rec["ts_base"]
                np.savez_compressed(npz_path, **rec)
                extracted += 1
            # Stage 2: hand landmarks on a wrist-centred crop, upscaled 3x.
            crop_path = os.path.join(fdir, f"{ordinal}_{view}_crop.npz")
            if os.path.exists(crop_path) and os.environ.get("ELBOW_REEXTRACT") != "1":
                rec.update({k: v for k, v in np.load(crop_path).items()})
            else:
                crop_rec = extract_hands_cropped(video, hands_img, rec)
                np.savez_compressed(crop_path, **crop_rec)
                rec.update(crop_rec)
                extracted += 0  # counted once above; crop pass is not a separate clip
            ann = annotation_elbows(form, view, ds)
            n = min(rec["T"], len(ann[0]))
            entry = {"frames": int(rec["T"]), "compared": int(n),
                     "hand_coverage": {s: float(np.mean(rec["hand_score"][:n, s] >= MIN_HAND_SCORE))
                                       for s in (0, 1)},
                     "hand_coverage_crop": {s: float(np.mean(
                         rec["hand_score_crop"][:n, s] >= MIN_HAND_SCORE))
                         for s in (0, 1)},
                     "annotation_jump_rate": {}, "estimators": {}}
            est_cache = {}
            for s in (0, 1):
                entry["annotation_jump_rate"]["L" if s == 0 else "R"] = \
                    round(jump_rate(ann[s][:n]), 4)
            for mode in ("pose_full", "pose_heavy", "heavy_hand_wrist", "heavy_hand_centre",
                         "heavy_hand_crop", "heavy_centre_crop"):
                est = elbow_angles(rec, mode)
                est_cache[mode] = est
                per_side = {}
                for s in (0, 1):
                    good = credible(ann[s][:n])
                    err = np.abs(est[s][:n][good] - ann[s][:n][good])
                    err = err[~np.isnan(err)]
                    per_side["L" if s == 0 else "R"] = {
                        "mae_deg": round(float(err.mean()), 2) if err.size else None,
                        "p95_deg": round(float(np.percentile(err, 95)), 2) if err.size else None,
                        "hold_sigma_deg": round(hold_sigma(est[s]), 2),
                        "jump_rate": round(jump_rate(est[s]), 4),
                        "nan_frac": round(float(np.mean(np.isnan(est[s][:n]))), 3),
                        "credible_frames": int(good.sum())}
                entry["estimators"][mode] = per_side
            # Cross-estimator per-move deviation: two independent observations of the
            # same footage. This is the quantity a tolerance has to absorb, because a
            # student and a reference are two independent observations of one movement.
            segs = segment_moves(features(ds[(form, view)]))
            moves = {}
            for a, b in (("pose_full", "pose_heavy"), ("pose_full", "heavy_hand_wrist"),
                         ("pose_heavy", "heavy_hand_wrist"), ("pose_heavy", "heavy_hand_crop"),
                         ("pose_heavy", "heavy_centre_crop")):
                per = {}
                for s in (0, 1):
                    devs = []
                    for a0, a1 in segs:
                        ma = np.nanmean(est_cache[a][s][a0:a1 + 1])
                        mb = np.nanmean(est_cache[b][s][a0:a1 + 1])
                        if np.isfinite(ma) and np.isfinite(mb):
                            devs.append(abs(ma - mb))
                    per[("L", "R")[s]] = {
                        "mean_deg": round(float(np.mean(devs)), 2) if devs else None,
                        "p95_deg": round(float(np.percentile(devs, 95)), 2) if devs else None,
                        "n_moves": len(devs),
                        "moves_deg": [round(float(x), 2) for x in devs]}
                moves[f"{a} vs {b}"] = per
            entry["cross_estimator_moves"] = moves
            rows[f"{ordinal}_{view}"] = entry
            cov = entry["hand_coverage"]
            print(f"{ordinal}_{view}  frames {entry['frames']}  hand-cov L{cov[0]:.2f} R{cov[1]:.2f}  "
                  f"ann jumps L{entry['annotation_jump_rate']['L']:.3f} R{entry['annotation_jump_rate']['R']:.3f}")
            for m in entry["estimators"]:
                e = entry["estimators"][m]
                print(f"    {m:19s} L mae {e['L']['mae_deg']} p95 {e['L']['p95_deg']} "
                      f"jump {e['L']['jump_rate']} sig {e['L']['hold_sigma_deg']} | "
                      f"R mae {e['R']['mae_deg']} p95 {e['R']['p95_deg']} "
                      f"jump {e['R']['jump_rate']} sig {e['R']['hold_sigma_deg']}")

    # ---- Derivation, on Taegeuk 1 only; validated on Taegeuk 2-8.
    def pooled_cross(form_filter):
        acc = {}
        for key, entry in rows.items():
            if not form_filter(key):
                continue
            for pair, per in entry["cross_estimator_moves"].items():
                for s in ("L", "R"):
                    if per[s]["mean_deg"] is not None:
                        acc.setdefault(pair, {"L": [], "R": []})[s].append(per[s]["mean_deg"])
        return {p: {s: round(float(np.mean(v)), 2) if v else None for s, v in d.items()}
                for p, d in acc.items()}

    def pooled_mae(form_filter):
        acc = {}
        for key, entry in rows.items():
            if not form_filter(key):
                continue
            for m, per in entry["estimators"].items():
                for s in ("L", "R"):
                    if per[s]["mae_deg"] is not None:
                        acc.setdefault(m, {"L": [], "R": []})[s].append(per[s]["mae_deg"])
        return {m: {s: round(float(np.mean(v)), 2) if v else None for s, v in d.items()}
                for m, d in acc.items()}

    cross_derive = pooled_cross(lambda k: k.startswith("1st_"))
    cross_heldout = pooled_cross(lambda k: not k.startswith("1st_"))
    mae_derive = pooled_mae(lambda k: k.startswith("1st_"))
    mae_heldout = pooled_mae(lambda k: not k.startswith("1st_"))

    # Rule, stated before the number is read: the elbow tolerance must exceed the
    # measurement uncertainty of the elbow angle, measured as the p95 per-move spread
    # between independent estimators on identical footage. It is a LOWER BOUND on
    # take-to-take variation (one athlete filmed twice differs by more than two models
    # reading the same pixels), so a tolerance derived this way cannot be too generous
    # for the wrong reason -- only possibly too tight for a real second take.
    def pooled_moves(form_filter):
        """Every per-move deviation, so the statistic is a pooled p95 -- not the max of
        a max, which one bad clip can dominate."""
        vals = []
        for key, entry in rows.items():
            if not form_filter(key):
                continue
            for pair, per in entry["cross_estimator_moves"].items():
                for s in ("L", "R"):
                    vals.extend(per[s].get("moves_deg") or [])
        return vals

    derive_moves = pooled_moves(lambda k: k.startswith("1st_"))
    heldout_moves = pooled_moves(lambda k: not k.startswith("1st_"))
    derive_p95 = float(np.percentile(derive_moves, 95)) if derive_moves else float("nan")
    heldout_p95 = float(np.percentile(heldout_moves, 95)) if heldout_moves else float("nan")
    derived_tol = float(max(5.0, np.ceil(derive_p95)))
    validated = bool(heldout_moves and heldout_p95 <= derived_tol)
    tail_derive = float(np.max(derive_moves)) if derive_moves else float("nan")
    tail_heldout = float(np.max(heldout_moves)) if heldout_moves else float("nan")

    worst_pair_mean = max(float(np.nanmean([v for v in d.values() if v is not None]))
                          for d in cross_derive.values()) if cross_derive else float("nan")
    print(f"\nextraction: {extracted} clip(s) extracted, {cached} from cache "
          f"(ELBOW_REEXTRACT=1 forces re-extraction)")
    print("\n--- cross-estimator per-move deviation (deg) ---")
    for pair in cross_derive:
        print(f"  derive  1st : {pair:38s} {cross_derive[pair]}")
        if pair in cross_heldout:
            print(f"  heldout 2-8 : {pair:38s} {cross_heldout[pair]}")
    print(f"  worst pair mean on Taegeuk 1: {worst_pair_mean:.2f} deg")
    print(f"  pooled per-move p95 (Taegeuk 1): {derive_p95:.2f} deg "
          f"(tail {tail_derive:.2f}) -> derived elbow tolerance {derived_tol} deg (was 25)")
    print(f"  pooled per-move p95 (held-out 2-8): {heldout_p95:.2f} deg (tail {tail_heldout:.2f}) "
          f"-> {'VALIDATED' if validated else 'NOT VALIDATED -- override NOT written'}")
    print(f"  annotation MAE (secondary, labels are noisy): derive {mae_derive.get('pose_full')} "
          f"heldout {mae_heldout.get('pose_full')}")

    payload = {"models": {"pose_full": "pose_landmarker_full.task",
                          "pose_heavy": "pose_landmarker_heavy.task",
                          "hands": "hand_landmarker.task"},
               "protocol": ("elbow tolerance = max(5, ceil(max p95 per-move cross-estimator "
                            "deviation)) on Taegeuk 1, validated on Taegeuk 2-8. Cross-estimator "
                            "deviation is a LOWER BOUND on take-to-take variation."),
               "per_clip": rows, "derive_set": "Taegeuk 1 (both views)",
               "heldout_set": "Taegeuk 2-8 (both views)",
               "cross_estimator_derive": cross_derive, "cross_estimator_heldout": cross_heldout,
               "derive_pooled_p95_deg": derive_p95, "derive_tail_max_deg": tail_derive,
               "heldout_pooled_p95_deg": heldout_p95, "heldout_tail_max_deg": tail_heldout,
               "heldout_validated": validated,
               "n_moves_derive": len(derive_moves), "n_moves_heldout": len(heldout_moves),
               "annotation_mae_derive": mae_derive, "annotation_mae_heldout": mae_heldout,
               "anchor": "wrist landmark error 20.3-23.8 px vs shoulders 3.4-4.2 px (out/pose_audit.json)",
               "derived_elbow_tolerance_deg": derived_tol,
               "baseline_elbow_tolerance_deg": 25.0}
    payload["override_written"] = validated
    if validated:
        with open(os.path.join(OUT, "tolerance_override.json"), "w") as fh:
            json.dump({"L_elbow": derived_tol, "R_elbow": derived_tol,
                       "source": f"pooled per-move p95 of cross-estimator deviation on "
                                 f"Taegeuk 1 = {derive_p95:.1f} deg; held-out 2-8 = "
                                 f"{heldout_p95:.1f} deg (out/elbow_signal.json)"}, fh, indent=1)
        print("\nwrote out/elbow_signal.json and out/tolerance_override.json")
    else:
        stale = os.path.join(OUT, "tolerance_override.json")
        if os.path.exists(stale):
            os.remove(stale)
            print("\nremoved stale out/tolerance_override.json (derivation failed validation)")
        print("\nwrote out/elbow_signal.json -- NO override: derived tolerance does not "
              "survive held-out validation, so the built-in 25 deg stands")
    with open(os.path.join(OUT, "elbow_signal.json"), "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
