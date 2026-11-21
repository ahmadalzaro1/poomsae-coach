"""Extract MediaPipe world landmarks for every clip in the dual-view dataset.

Why this exists: the 2D pipeline's biggest measured weakness is camera side. A
frontal reference compared against a lateral take of the SAME athlete deviates by
10.2 degrees on average and flags 133 of 156 moves, because a 2D projection throws
away depth and every sagittal-plane angle is azimuth-dependent. Two modelling
attempts to remove that error both failed (see `learned_gate.py`,
`view_calibration.py`), which points at the feature space rather than the model.

MediaPipe also returns `pose_world_landmarks`: metric 3D coordinates, hip-centred
and gravity-aligned, which should be far less sensitive to camera azimuth. This
module extracts them so the same benchmark can be re-run in 3D and the claim
tested instead of assumed.

Writes one CSV per clip: out/pose3d/<form>_<view>.csv, with both the image-space
x/y/visibility and the world-space x/y/z per joint, so the two feature sets can be
compared on identical frames.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import JOINTS
from poomsae.pose_video import MP_IDX

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(HERE, "models", "pose_landmarker_full.task")
DATA = os.path.join(HERE, "data", "Data")
OUTDIR = os.path.join(HERE, "out", "pose3d")
FORMS = [f"Taegeuk_{i}{s}" for i, s in
         [(1, "st"), (2, "nd"), (3, "rd"), (4, "th"), (5, "th"),
          (6, "th"), (7, "th"), (8, "th")]]
ORD = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8}


def video_path(form: str, view: str) -> str:
    ordinal = form.split("_")[1]                 # "1st".."8th"
    return os.path.join(DATA, "video", form, view, f"{ordinal}_{view}.mp4")


def out_path(form: str, view: str) -> str:
    return os.path.join(OUTDIR, f"{form}_{view}.csv")


def extract(path: str, dst: str, max_frames: int | None = None) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3)
    fields = ["frame_index", "time_ms"]
    for name in JOINTS:
        fields += [f"{name}_x", f"{name}_y", f"{name}_vis",
                   f"{name}_wx", f"{name}_wy", f"{name}_wz"]
    rows, t0, idx, no_pose, wvis = [], time.time(), 0, 0, []
    with vision.PoseLandmarker.create_from_options(opts) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(idx * 1000 / fps))
            rec = {"frame_index": idx, "time_ms": round(idx * 1000 / fps, 1)}
            world = getattr(res, "pose_world_landmarks", None)
            if res.pose_landmarks and world:
                pl, wl = res.pose_landmarks[0], world[0]
                for name in JOINTS:
                    pi, p = MP_IDX[name], pl[MP_IDX[name]]
                    w = wl[pi]
                    rec[f"{name}_x"] = round(p.x, 6)
                    rec[f"{name}_y"] = round(p.y, 6)
                    rec[f"{name}_vis"] = round(float(getattr(p, "visibility", 0.0) or 0.0), 4)
                    rec[f"{name}_wx"] = round(w.x, 6)
                    rec[f"{name}_wy"] = round(w.y, 6)
                    rec[f"{name}_wz"] = round(w.z, 6)
                wvis.append(float(np.mean([float(getattr(pl[MP_IDX[n]], "visibility", 0.0) or 0.0)
                                           for n in JOINTS])))
            else:
                no_pose += 1
                for name in JOINTS:
                    for k in ("x", "y", "vis", "wx", "wy", "wz"):
                        rec[f"{name}_{k}"] = 0.0
            rows.append(rec)
            idx += 1
            if max_frames and idx >= max_frames:
                break
    cap.release()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    el = time.time() - t0
    return {"frames": len(rows), "seconds": round(el, 1),
            "ms_per_frame": round(1000 * el / max(len(rows), 1), 1),
            "frames_without_pose": no_pose,
            "mean_visibility": round(float(np.mean(wvis)), 3) if wvis else 0.0}


def load3d(path: str):
    """-> (T,12,3) world coords (metres), (T,12) visibility."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    T = len(rows)
    w = np.zeros((T, len(JOINTS), 3))
    vis = np.zeros((T, len(JOINTS)))
    for t, r in enumerate(rows):
        for j, name in enumerate(JOINTS):
            w[t, j] = (float(r[f"{name}_wx"]), float(r[f"{name}_wy"]), float(r[f"{name}_wz"]))
            vis[t, j] = float(r[f"{name}_vis"])
    return w, vis


def main() -> int:
    force = "--force" in sys.argv
    os.makedirs(OUTDIR, exist_ok=True)
    total_f, total_s = 0, 0.0
    for form in FORMS:
        for view in ("Frontal", "Lateral"):
            src, dst = video_path(form, view), out_path(form, view)
            if os.path.exists(dst) and not force:
                print(f"skip (exists) {form}/{view}")
                continue
            if not os.path.exists(src):
                print(f"MISSING video {src}")
                continue
            info = extract(src, dst)
            total_f += info["frames"]
            total_s += info["seconds"]
            print(f"{form:<12}{view:<9} frames {info['frames']:>5}  "
                  f"{info['ms_per_frame']:>5.1f} ms/frame  "
                  f"no-pose {info['frames_without_pose']:>3}  "
                  f"vis {info['mean_visibility']:.2f}")
    if total_s:
        print(f"\ntotal {total_f} frames in {total_s:.0f}s "
              f"({1000*total_s/max(total_f,1):.1f} ms/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
