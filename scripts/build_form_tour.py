"""Apply the same mask and the same measurement to every clip in the dataset.

The repo's cards exist for Taegeuk 1 Jang only, because the self-referenced gate
needs >= MIN_MOVES repeats of one cell by one athlete and that is measured on
that form. This script answers a narrower, honest question for the other seven:
*does the mask and the measurement travel?* For each form it

  1. extracts 12 body landmarks from the raw clip with MediaPipe (the same mask
     the demo video draws), writing a CSV in the dataset's own schema;
  2. measures our angles against the published annotation of the same clip, so
     the reader sees estimator agreement rather than a claim about the athlete;
  3. renders a 5-second excerpt around the form's worst frame, coloured by
     |delta| / measured tolerance for the single worst joint.

It deliberately publishes **no verdict** for these forms: a verdict needs the
per-athlete gate, and this release has one athlete's gate (Taegeuk 1). A page
that showed "fault" here would be inventing the thing the repo exists to stop.

Outputs
  out/pose_form<n>_frontal.csv     our landmarks, dataset schema
  out/form_tour_<n>.json           the numbers behind the clip
  out/form_tour_<n>.mp4            1280x720 @30 fps, 5 s excerpt
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from poomsae.core import FEATURE_NAMES, JOINTS, features, load_csv  # noqa: E402
from poomsae.pose_video import extract  # noqa: E402

OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data", "Data")

FORM_NAME = {
    1: "Taegeuk 1 Jang", 2: "Taegeuk 2 Jang", 3: "Taegeuk 3 Jang", 4: "Taegeuk 4 Jang",
    5: "Taegeuk 5 Jang", 6: "Taegeuk 6 Jang", 7: "Taegeuk 7 Jang", 8: "Taegeuk 8 Jang",
}
ORD = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th", 6: "6th", 7: "7th", 8: "8th"}

# measured tolerance per cell family: the same numbers the cards use
def tolerance_of(feature: str) -> float:
    f = feature.lower()
    if "elbow" in f:
        return 18.0
    if "knee" in f:
        return 10.0
    if "hip" in f:
        return 10.0
    if "shoulder" in f:
        return 12.0
    return 5.0  # torso tilt


TOL = np.array([tolerance_of(n) for n in FEATURE_NAMES], dtype=float)

W, H = 1280, 720
FPS = 30
EXCERPT_S = 5.0

GREEN = (120, 220, 120)
AMBER = (40, 190, 250)
RED = (70, 70, 235)
GREY = (150, 150, 150)
INK = (235, 235, 235)
DIM = (150, 150, 150)
PANEL = (26, 26, 30)

# parent-child edges for drawing the mask
EDGES = [("L_shoulder", "R_shoulder"), ("L_shoulder", "L_elbow"), ("L_elbow", "L_wrist"),
         ("R_shoulder", "R_elbow"), ("R_elbow", "R_wrist"), ("L_shoulder", "L_hip"),
         ("R_shoulder", "R_hip"), ("L_hip", "R_hip"), ("L_hip", "L_knee"),
         ("L_knee", "L_ankle"), ("R_hip", "R_knee"), ("R_knee", "R_ankle")]


def load_mask(path):
    """landmark CSV -> {frame: {joint: (x, y)}} in 0..1."""
    import csv
    rows = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[int(r["frame_index"])] = {
                j: (float(r[f"{j}_x"]), float(r[f"{j}_y"])) for j in JOINTS}
    return rows


def deltas_for(n: int):
    """(delta[T,9], agreement dict) for form n: our features minus the annotation's."""
    form = ORD[n]
    video = os.path.join(DATA, "video", f"Taegeuk_{form}", "Frontal", f"{form}_Frontal.mp4")
    annot = os.path.join(DATA, "annotations", f"Taegeuk_{form}", "Frontal", f"{form}_Frontal.csv")
    # form 1's landmarks are already produced by step 11 of the pipeline; reuse them rather than
    # shipping a second CSV of the same clip under a different name
    shared = os.path.join(OUT, "pose_taegeuk1_frontal.csv")
    ours = shared if (n == 1 and os.path.exists(shared)) else os.path.join(OUT, f"pose_form{n}_frontal.csv")
    for p in (video, annot):
        if not os.path.exists(p):
            sys.exit(f"build_form_tour: missing {p}")

    if not os.path.exists(ours):
        _, vw, vh, nfr, el = extract(video, ours)
        print(f"  extracted {nfr} frames in {el:.1f}s ({1000*el/nfr:.1f} ms/frame)")
    else:
        vw, vh = 1920.0, 1080.0
        print("  reusing existing landmark CSV")

    ref, stu = load_csv(annot), load_csv(ours, vw, vh)
    fr_ref, fr_stu = features(ref), features(stu)
    T = min(len(fr_ref), len(fr_stu))
    delta = fr_stu[:T] - fr_ref[:T]
    err = np.abs(delta)
    keep = err.max(axis=1) < GROSS_DEG
    err_k = err[keep] if keep.any() else err
    agreement = {
        "frames_compared": int(T),
        "frames_scored": int(keep.sum()),
        "frames_gross": int((~keep).sum()),
        "p50_abs_delta_deg": round(float(np.percentile(err_k, 50)), 1),
        "p95_abs_delta_deg": round(float(np.percentile(err_k, 95)), 1),
        "max_abs_delta_deg": round(float(err.max()), 1),
        "per_feature_p95_deg": {f: round(float(np.percentile(err_k[:, i], 95)), 1)
                                for i, f in enumerate(FEATURE_NAMES)},
    }
    return video, ours, delta, agreement


GROSS_DEG = 60.0   # a 2D pose estimator does not miss an elbow by 60 deg; a foreshortened
                   # arm, or a locked joint, does. Frames past this are reported, not scored.


def worst_window(ratio, span):
    """index of the `span`-frame window with the highest mean worst-joint ratio."""
    if len(ratio) <= span:
        return 0
    c = np.convolve(ratio, np.ones(span) / span, mode="valid")
    return int(np.argmax(c))


def render(n, video, lm_path, delta, out_mp4):
    absd = np.abs(delta)
    gross = absd.max(axis=1) >= GROSS_DEG            # (T,) frames to report, not to score
    ratio = absd / TOL                               # (T, 9) per-frame per-cell severity
    per_frame = ratio.max(axis=1)
    span = min(int(EXCERPT_S * FPS), len(per_frame))
    start = worst_window(per_frame, span)
    stop = start + span
    worst_cell = int(np.argmax(ratio[start:stop].mean(axis=0)))

    land = load_mask(lm_path)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", out_mp4],
        stdin=subprocess.PIPE)

    assert ff.stdin is not None
    name = FORM_NAME[n]
    for i in range(start, stop):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (W, H))
        pts = land.get(i, {})

        # --- the mask
        for a, b in EDGES:
            if a in pts and b in pts:
                cv2.line(frame,
                         (int(pts[a][0] * W), int(pts[a][1] * H)),
                         (int(pts[b][0] * W), int(pts[b][1] * H)), GREEN, 2, cv2.LINE_AA)
        for j, (x, y) in pts.items():
            cv2.circle(frame, (int(x * W), int(y * H)), 3, GREEN, -1, cv2.LINE_AA)

        # --- panel
        top = np.zeros((132, W, 3), np.uint8)
        top[:] = PANEL
        cv2.putText(top, name, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, INK, 2, cv2.LINE_AA)
        cell = FEATURE_NAMES[worst_cell]
        d = float(delta[i, worst_cell])
        tol = float(TOL[worst_cell])
        sev = abs(d) / tol
        if gross[i]:
            col = GREY
            line = f"{cell}: {d:+.1f} deg - gross pose failure, excluded from the summary"
        else:
            col = GREEN if sev <= 1.0 else (AMBER if sev <= 1.6 else RED)
            line = f"{cell}: {d:+.1f} deg vs annotation  (tol {tol:.0f})"
        cv2.putText(top, line, (18, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        cv2.putText(top, f"{name}  -  mask and measurement, no verdict for this form",
                    (18, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1, cv2.LINE_AA)

        strip = np.zeros((56, W, 3), np.uint8)
        strip[:] = PANEL
        cv2.putText(strip,
                    "measurement only: this release publishes a self-referenced gate for "
                    "Taegeuk 1 only, so no cell here is called a fault",
                    (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, DIM, 1, cv2.LINE_AA)
        cv2.putText(strip, f"t {(i-start)/FPS:4.1f}s / {span/FPS:.1f}s   frame {i}",
                    (18, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1, cv2.LINE_AA)

        out = np.vstack([top, frame[:H - top.shape[0] - strip.shape[0]], strip])
        ff.stdin.write(out.tobytes())

    cap.release()
    ff.stdin.close()
    ff.wait()
    return {"window_start_frame": int(start), "window_s": EXCERPT_S,
            "window_worst_cell": FEATURE_NAMES[worst_cell],
            "window_mean_ratio": round(float(ratio[start:stop, worst_cell].mean()), 2),
            "gross_frames_total": int(gross.sum()),
            "gross_pct": round(float(gross.mean() * 100), 1),
            "gross_frames_in_window": int(gross[start:stop].sum())}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    only = [int(x) for x in sys.argv[1:]] or list(range(1, 9))
    summary_path = os.path.join(OUT, "form_tour.json")
    prior = {}
    if os.path.exists(summary_path):
        try:
            prior = {r["form"]: r for r in json.load(open(summary_path))}
        except (ValueError, KeyError):
            prior = {}
    for n in only:
        print(f"[{n}/8] {FORM_NAME[n]}")
        video, lm, delta, agreement = deltas_for(n)
        clip = os.path.join(OUT, f"form_tour_{n}.mp4")
        win = render(n, video, lm, delta, clip)
        rec = {"form": n, "name": FORM_NAME[n], "source_video": os.path.basename(video),
               "agreement": agreement, **win,
               "frames_in_clip": int(min(len(delta), EXCERPT_S * FPS))}
        json.dump(rec, open(os.path.join(OUT, f"form_tour_{n}.json"), "w"), indent=1)
        r = np.abs(delta) / TOL
        surv = np.abs(delta).max(axis=1) < GROSS_DEG
        pct = float(np.mean(r[surv].max(axis=1) > 1.0) * 100) if surv.any() else 0.0
        rec["pct_frames_some_cell_over_tol"] = round(pct, 1)
        print(f"  agreement p50 {agreement['p50_abs_delta_deg']}° p95 "
              f"{agreement['p95_abs_delta_deg']}°  |  worst window "
              f"{win['window_start_frame']/FPS:.1f}s -> {win['window_worst_cell']} "
              f"ratio {win['window_mean_ratio']}  |  {pct:.0f}% of frames have a cell over tol")
        prior[n] = rec
    summary = [prior[k] for k in sorted(prior)]
    json.dump(summary, open(summary_path, "w"), indent=1)
    print(f"\nwrote out/form_tour.json ({len(summary)} forms) + {len(only)} clips this pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
