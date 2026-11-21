"""Render the poomsae analysis onto the raw video, as a shareable MP4.

Why this exists: a coach cannot read a JSON file and will not scrub a widget.
The demo has to be a video they can drop into WhatsApp.

What is drawn, and where each number comes from:
  * skeleton  -- MediaPipe landmarks we extracted ourselves (out/pose_*.csv)
  * colours   -- per-joint |delta| / tolerance for the move being performed,
                 read from the coach card (out/card_*.json), not recomputed here
  * HUD text  -- the same card: technique name, mean |delta|, the worst flag

So the video cannot disagree with the card; both read the same measurement.
Encoding goes through ffmpeg (libx264, faststart) because OpenCV's own writer
produces files that many chat players refuse to seek.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")

BONES = [
    ("L_shoulder", "R_shoulder"), ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
    ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"),
    ("L_shoulder", "L_hip"), ("R_shoulder", "R_hip"), ("L_hip", "R_hip"),
    ("L_hip", "L_knee"), ("L_knee", "L_ankle"), ("R_hip", "R_knee"), ("R_knee", "R_ankle"),
]
JOINTS = ["L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
          "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]

# feature (angle) -> the joints whose colour that angle should drive
FEATURE_JOINTS = {
    "L_elbow": ["L_elbow"], "R_elbow": ["R_elbow"],
    "L_shoulder": ["L_shoulder"], "R_shoulder": ["R_shoulder"],
    "L_hip": ["L_hip"], "R_hip": ["R_hip"],
    "L_knee": ["L_knee"], "R_knee": ["R_knee"],
    "torso_tilt": ["L_shoulder", "R_shoulder"],
}
RED, AMBER, GREEN, WHITE = (60, 60, 235), (36, 165, 245), (120, 220, 120), (245, 245, 245)
PANEL = (28, 28, 32)


def load_landmarks(path):
    """value in 0..1 -> list of {joint: (x, y)} indexed by frame."""
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            i = int(r["frame_index"])
            rows[i] = {j: (float(r[f"{j}_x"]), float(r[f"{j}_y"])) for j in JOINTS}
    return rows


def move_frames(card, n_frames):
    """Attach a [start, end) frame range to each move; ranges tile the clip."""
    ms = sorted(card["moves"], key=lambda m: m["t"])
    for k, m in enumerate(ms):
        start = int(round(m["t"] * 30))
        end = int(round(ms[k + 1]["t"] * 30)) if k + 1 < len(ms) else n_frames
        m["_range"] = (start, max(start + 1, end))
    return ms


def resolved_gates(_sr):
    """feature/joint -> k * this athlete's own scale for that cell, i.e. the gate actually applied.

    effective_threshold_deg is max(tolerance, k*floor) - a LOWER BOUND on the gate. Printing that
    number as 'gate' while the adjudication uses k*scale is how a video ends up disagreeing with its
    own card, which is the exact failure this repo exists to avoid.
    """
    k = float(_sr.get("gate_k") or 2.5)
    scale = _sr.get("scale_deg") or {}
    out = {}
    for cell, sd in scale.items():
        out[cell] = k * float(sd)
        for feat in FEATURE_JOINTS.get(cell, []):
            out[feat] = max(out.get(feat, 0.0), k * float(sd))
    return out


def severity(move, gates=None):
    """joint -> |delta| / gate, using the card's own per-joint numbers.

    The gate is the effective threshold from the self-reference layer (max(tolerance, k*floor)),
    not a private table in here: a renderer that carries its own tolerances can colour a joint
    'close' while the card calls it a fault, and the video is what a coach believes.
    """
    tol = dict(gates or {})
    fallback = float(max(tol.values())) if tol else 15.0
    sev = {j: 0.0 for j in JOINTS}
    for feat, val in (move.get("deltas") or {}).items():
        r = abs(val) / float(tol.get(feat, fallback))
        for j in FEATURE_JOINTS.get(feat, []):
            sev[j] = max(sev[j], r)
    return sev


def band(r):
    return RED if r > 1.0 else (AMBER if r > 0.75 else GREEN)


def bar(img, x0, y0, x1, y1, frac, colour):
    x1 = int(x0 + (x1 - x0) * max(0.0, min(1.0, frac)))
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)


def overlay(img, pts, sev, W, H):
    P = {j: (int(pts[j][0] * W), int(pts[j][1] * H)) for j in JOINTS if j in pts}
    for a, b in BONES:
        if a in P and b in P:
            r = max(sev.get(a, 0), sev.get(b, 0))
            cv2.line(img, P[a], P[b], band(r), 4 if r > 1 else 3, cv2.LINE_AA)
    for j, (x, y) in P.items():
        r = sev.get(j, 0)
        cv2.circle(img, (x, y), 6, band(r), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 6, (20, 20, 20), 1, cv2.LINE_AA)
        if r > 1.0:
            cv2.circle(img, (x, y), 13, RED, 2, cv2.LINE_AA)


def render(card_path, out_path, title, subtitle, warn=None, progress_tag=""):
    card = json.load(open(card_path))
    video = os.path.join(HERE, "data", "Data", "video", "Taegeuk_1st", "Frontal",
                         card["source_video"])
    lm_path = os.path.join(OUT, "pose_taegeuk1_frontal.csv")
    land = load_landmarks(lm_path)

    cap = cv2.VideoCapture(video)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(land)
    moves = move_frames(card, n_frames)
    # thresholds come from the card, never from a table in this file
    _sr = card.get("self_reference") or {}
    gates = _sr.get("effective_threshold_deg") or {}
    resolved = resolved_gates(_sr) or dict(gates)
    meas = _sr.get("tolerance_deg") or {}
    n_faults = (_sr.get("verdict_counts") or {}).get("fault")
    W, H = 1280, 720
    panel_y = 556

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", "30", "-i", "-",
         "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    flagged_total = sum(1 for m in moves if m["flags"])
    f = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            img = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            mv = next((m for m in moves if m["_range"][0] <= f < m["_range"][1]), moves[-1])
            sev = severity(mv, resolved)
            pts = land.get(f) or land.get(min(land, key=lambda k: abs(k - f)))
            if pts:
                overlay(img, pts, sev, W, H)

            # ---- top strip
            cv2.rectangle(img, (0, 0), (W, 46), (18, 18, 22), -1)
            cv2.putText(img, title, (18, 31), cv2.FONT_HERSHEY_DUPLEX, 0.72, WHITE, 1, cv2.LINE_AA)
            tag = f"move {mv['move']} / {len(moves)}"
            tw = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0]
            cv2.putText(img, tag, (W - tw - 18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (200, 200, 210), 1, cv2.LINE_AA)

            # ---- bottom panel
            cv2.rectangle(img, (0, panel_y), (W, H), PANEL, -1)
            cv2.line(img, (0, panel_y), (W, panel_y), (70, 70, 78), 1)

            tech = mv.get("technique") or "unnamed section"
            cv2.putText(img, tech, (18, panel_y + 40), cv2.FONT_HERSHEY_DUPLEX, 0.86,
                        WHITE, 1, cv2.LINE_AA)
            stats = f"mean |diff| {mv['mean_abs_delta']:.1f} deg"
            cv2.putText(img, stats, (18, panel_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (185, 185, 195), 1, cv2.LINE_AA)

            if mv["flags"]:
                fl = mv["flags"][0]
                verdict = ((mv.get("attribution") or {}).get("per_joint") or {}).get(fl["key"], {}).get("verdict")
                txt = f"{fl['joint']}: {abs(fl['delta']):.0f} deg {fl['msg'].split(' ', 1)[1]}"
                if verdict and verdict != "fault":
                    txt += f"  ({verdict})"
                cv2.putText(img, txt, (18, panel_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            RED if verdict == "fault" else AMBER, 2, cv2.LINE_AA)
                cv2.putText(img, f"tol {meas.get(fl['key'], fl['tol']):.0f} / floor {gates.get(fl['key'], fl['tol']):.0f} / gate {resolved.get(fl['key'], gates.get(fl['key'], fl['tol'])):.0f} deg", (18, panel_y + 126),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 180), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "within tolerance", (18, panel_y + 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREEN, 2, cv2.LINE_AA)

            # right side of panel: running tally + legend
            tally = f"{flagged_total} of {len(moves)} moves outside measurement tol"
            if n_faults is not None:
                tally += f" | {n_faults} cells past tolerance, offset, scale AND band"
            tw = cv2.getTextSize(tally, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
            cv2.putText(img, tally, (W - tw - 18, panel_y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (185, 185, 195), 1, cv2.LINE_AA)
            for k, (lab, col) in enumerate((("within", GREEN), ("close", AMBER), ("outside", RED))):
                x = W - 18 - tw + k * 92
                cv2.circle(img, (x, panel_y + 66), 5, col, -1, cv2.LINE_AA)
                cv2.putText(img, lab, (x + 10, panel_y + 71), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (170, 170, 180), 1, cv2.LINE_AA)

            cv2.putText(img, subtitle, (18, panel_y + 152), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (140, 140, 150), 1, cv2.LINE_AA)
            if warn:
                cv2.putText(img, warn, (18, panel_y + 132), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, AMBER, 1, cv2.LINE_AA)
                cv2.rectangle(img, (2, 2), (W - 3, H - 3), AMBER, 3)

            bar(img, 0, H - 4, W, H, f / (n_frames - 1), (200, 140, 60))
            if progress_tag:
                cv2.putText(img, progress_tag, (W - 150, panel_y + 152),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, (140, 140, 150), 1, cv2.LINE_AA)

            ff.stdin.write(img.tobytes())
            f += 1
    finally:
        cap.release()
        ff.stdin.close()
        ff.wait()

    mb = os.path.getsize(out_path) / 1e6
    print(f"{os.path.basename(out_path)}: {f} frames, {f/30:.1f}s, {mb:.1f} MB, "
          f"{flagged_total}/{len(moves)} moves flagged")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="card_sameview.json")
    ap.add_argument("--out", default="demo_sameview.mp4")
    ap.add_argument("--warn", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    same = a.card == "card_sameview.json"
    render(os.path.join(OUT, a.card), os.path.join(OUT, a.out),
           title="Taegeuk 1 Jang  -  joint-angle check",
           subtitle="skeleton: MediaPipe on the raw clip  |  reference: published annotation, "
                    "same camera side  |  12 body joints, angles only",
           warn=a.warn, progress_tag=a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
