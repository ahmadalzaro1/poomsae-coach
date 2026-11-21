"""Video -> pose -> feedback, end to end.

Runs MediaPipe PoseLandmarker on a raw .mp4 from the dataset, writes a CSV in
the dataset's own schema, then measures how close OUR extracted features are to
the published annotations (an independent pose estimator is the honest check on
annotation quality), and finally scores that video take against the annotation
of another view.
"""
from __future__ import annotations  # py3.9: allow `int | None` in signatures

import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import FEATURE_NAMES, JOINTS, analyse, features, load_csv
from poomsae.run_naming import attach_names

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "pose_landmarker_full.task")
OUT = os.path.join(HERE, "out")

MP_IDX = {
    "L_shoulder": 11, "R_shoulder": 12, "L_elbow": 13, "R_elbow": 14,
    "L_wrist": 15, "R_wrist": 16, "L_hip": 23, "R_hip": 24,
    "L_knee": 25, "R_knee": 26, "L_ankle": 27, "R_ankle": 28,
}


def extract(video_path: str, out_csv: str, max_frames: int | None = None):
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    rows, t0, idx = [], time.time(), 0
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(idx * 1000 / fps)
            res = lm.detect_for_video(img, ts)
            rec = {"frame_index": idx, "time_ms": round(idx * 1000 / fps, 1)}
            if res.pose_landmarks:
                pl = res.pose_landmarks[0]
                for name in JOINTS:
                    p = pl[MP_IDX[name]]
                    rec[f"{name}_x"] = round(p.x, 6)
                    rec[f"{name}_y"] = round(p.y, 6)
                    rec[f"{name}_score"] = round(float(getattr(p, "visibility", 0.0) or 0.0), 4)
            else:
                for name in JOINTS:
                    rec[f"{name}_x"] = rec[f"{name}_y"] = 0.0
                    rec[f"{name}_score"] = 0.0
            rows.append(rec)
            idx += 1
            if max_frames and idx >= max_frames:
                break
    cap.release()
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    el = time.time() - t0
    print(f"extracted {len(rows)} frames from {os.path.basename(video_path)} "
          f"({W}x{H} @{fps:.0f}fps) in {el:.1f}s = {1000*el/max(len(rows),1):.1f} ms/frame")
    return out_csv, W, H, len(rows), el


def compare_to_annotations(ours: str, theirs: str, W, H):
    a = load_csv(ours, W, H)
    b = load_csv(theirs, W, H)
    n = min(a.n, b.n)
    fa, fb = features(a), features(b)
    n = min(len(fa), len(fb))
    diff = np.abs(fa[:n] - fb[:n])
    per_feat = diff.mean(0)
    # landmark-level pixel error, in pixels of a 1080p frame
    px_err = np.linalg.norm(a.px[:n] - b.px[:n], axis=-1)
    return {
        "frames_compared": int(n),
        "feature_mae_deg": {f: round(float(v), 1) for f, v in zip(FEATURE_NAMES, per_feat)},
        "feature_mae_mean_deg": round(float(per_feat.mean()), 1),
        "landmark_mae_px": round(float(px_err.mean()), 1),
        "landmark_p90_px": round(float(np.percentile(px_err, 90)), 1),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(HERE, "data", "Data")
    video = os.path.join(base, "video", "Taegeuk_1st", "Frontal", "1st_Frontal.mp4")
    annot = os.path.join(base, "annotations", "Taegeuk_1st", "Frontal", "1st_Frontal.csv")
    ours = os.path.join(OUT, "pose_taegeuk1_frontal.csv")

    _, W, H, n, el = extract(video, ours)
    cmp = compare_to_annotations(ours, annot, W, H)
    print("\n--- our pose vs published annotation (same video) ---")
    print(json.dumps({k: v for k, v in cmp.items() if k != "feature_mae_deg"}, indent=1))
    print("per-feature MAE:", cmp["feature_mae_deg"])

    # Two cards: same view (the realistic coaching case) and cross-view
    # (the camera-angle caveat). Both are printed and written.
    front = load_csv(os.path.join(base, "annotations", "Taegeuk_1st", "Frontal", "1st_Frontal.csv"))
    lateral = load_csv(os.path.join(base, "annotations", "Taegeuk_1st", "Lateral", "1st_Lateral.csv"))
    stu = load_csv(ours, W, H)
    for tag, ref in (("sameview", front), ("crossview", lateral)):
        res = analyse(ref, stu, threshold=None)
        attach_names(res["moves"], stu)
        card = {k: v for k, v in res.items() if k not in ("features_ref", "features_stu", "path")}
        card["source_video"] = os.path.basename(video)
        card["extraction_ms_per_frame"] = round(1000 * el / n, 1)
        card["pose_vs_annotation"] = cmp
        json.dump(card, open(os.path.join(OUT, f"card_{tag}.json"), "w"), indent=1)
        print(f"\n--- card [{tag}] {ref.view} reference vs our video pose ---")
        print(f"frames {res['frames']}  moves {res['n_moves']}  mean|Δ| {res['mean_abs_delta']}°  "
              f"dtw/frame {res['dtw_cost_per_frame']}  flagged {res['flagged_moves']}/{res['n_moves']}")
        for m in res["moves"][:8]:
            top = m["flags"][0]["msg"] if m["flags"] else "within tolerance"
            print(f"  move {m['move']:>2} t={m['t']:>5.2f}s  mean|Δ| {m['mean_abs_delta']:>5.1f}°  -> {top}")


if __name__ == "__main__":
    main()
