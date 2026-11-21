"""Taekwondo poomsae form-checker — core library.

Pipeline: pose keypoints -> hip-centred/torso-scaled normalisation -> joint angles
-> Dynamic Time Warping alignment to a reference performance -> per-move
joint-angle diffs -> plain-language coaching feedback.

Design notes
------------
* Coordinates in the shipped CSVs are normalised to [0,1] of image width/height.
  Angles must be computed in PIXEL space, otherwise the 16:9 aspect ratio
  silently distorts every angle. We scale by the real video dimensions.
* Features are joint ANGLES (+ torso tilt), not raw positions: angles are
  (mostly) translation- and scale-free, which is what lets a frontal reference
  be compared to a lateral take at all. Raw x/y would encode camera geometry.
* DTW is standard O(n*m) with an optional Sakoe-Chiba band. Reference and
  student are z-scored with the *reference* statistics so a constant per-view
  bias cannot inflate the match.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

JOINTS = [
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]
J = {name: i for i, name in enumerate(JOINTS)}

# interior angle at the middle joint of (a, b, c)
ANGLE_TRIPLETS = [
    ("L_elbow", "L_shoulder", "L_elbow", "L_wrist"),
    ("R_elbow", "R_shoulder", "R_elbow", "R_wrist"),
    ("L_knee", "L_hip", "L_knee", "L_ankle"),
    ("R_knee", "R_hip", "R_knee", "R_ankle"),
    ("L_hip", "L_shoulder", "L_hip", "L_knee"),
    ("R_hip", "R_shoulder", "R_hip", "R_knee"),
    ("L_shoulder", "L_hip", "L_shoulder", "L_elbow"),
    ("R_shoulder", "R_hip", "R_shoulder", "R_elbow"),
]
FEATURE_NAMES = [t[0] for t in ANGLE_TRIPLETS] + ["torso_tilt"]

HUMAN = {
    "L_elbow": "left elbow", "R_elbow": "right elbow",
    "L_knee": "left knee", "R_knee": "right knee",
    "L_hip": "left hip", "R_hip": "right hip",
    "L_shoulder": "left shoulder", "R_shoulder": "right shoulder",
    "torso_tilt": "torso lean",
}

# Tolerances are NOT arbitrary: they are ~1.5x the measured inter-estimator
# disagreement on the same clip (MediaPipe vs the published annotation):
# legs 1.6 deg, shoulder 5 deg, elbows 16.5 deg, torso 0.5 deg.
# Uniform tolerances would either bury real leg errors or spam arm noise.
TOLERANCE_DEG = {
    "L_knee": 10.0, "R_knee": 10.0,
    "L_hip": 10.0, "R_hip": 10.0,
    "L_shoulder": 12.0, "R_shoulder": 12.0,
    "L_elbow": 25.0, "R_elbow": 25.0,
    "torso_tilt": 5.0,
}
BASELINE_TOLERANCE_DEG = dict(TOLERANCE_DEG)

# A tolerance is a product decision, so an override is allowed but never silent: it
# must come from a measured artifact on disk and is reported by name everywhere the
# tolerances are used. Written by poomsae/elbow_signal.py.
TOLERANCE_SOURCE = "built-in defaults"
_OVERRIDE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "out", "tolerance_override.json")
if os.path.exists(_OVERRIDE_PATH):
    # A half-written override must never take down the pipeline: an interrupted write
    # leaves a 0-byte file, and json.load raises before any step can run. Treat an
    # unreadable or empty override as absent and say so.
    try:
        with open(_OVERRIDE_PATH) as _fh:
            _ov = json.load(_fh)
        _applied = {k: float(v) for k, v in _ov.items() if k in TOLERANCE_DEG}
        if not _applied:
            raise ValueError("no recognised tolerance keys")
    except Exception as _exc:  # noqa: BLE001 - any corruption is the same case
        print(f"[core] ignoring unreadable tolerance override ({_exc}); "
              f"using built-in defaults")
    else:
        _src = _ov.get("source", "unstated source")
        TOLERANCE_DEG.update(_applied)
        TOLERANCE_SOURCE = f"{os.path.relpath(_OVERRIDE_PATH)} ({_src})"
        print(f"[core] tolerance override active: {TOLERANCE_SOURCE}")


# ---------------------------------------------------------------- loading
@dataclass
class Sequence:
    name: str
    form: str
    view: str
    px: np.ndarray          # (T, 12, 2) pixel coordinates
    scores: np.ndarray      # (T, 12) landmark confidence 0..1
    times_ms: np.ndarray    # (T,)
    width: int = 1920
    height: int = 1080

    @property
    def n(self) -> int:
        return self.px.shape[0]


def side_of(v: str | None) -> str | None:
    """Map any view-ish string to Frontal/Lateral. Sides arrive as directory names
    ("Frontal"), dataset filenames ("1st_Frontal") and our own pose CSVs
    ("pose_taegeuk1_frontal"), so match anywhere in the token, not just at the start."""
    s = (v or "").strip().lower()
    if "front" in s:
        return "Frontal"
    if "lateral" in s or "side" in s:
        return "Lateral"
    return None


def load_csv(path: str, width: float = 1920.0, height: float = 1080.0):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    T = len(rows)
    px = np.full((T, 12, 2), np.nan, dtype=float)
    sc = np.full((T, 12), np.nan, dtype=float)
    times = np.zeros(T, dtype=float)
    for t, r in enumerate(rows):
        times[t] = float(r["time_ms"])
        for name, i in J.items():
            x, y = float(r[f"{name}_x"]), float(r[f"{name}_y"])
            s = float(r[f"{name}_score"])
            px[t, i, 0] = x * width
            px[t, i, 1] = y * height
            sc[t, i] = s
    parts = os.path.basename(path).replace(".csv", "").split("_")
    detected = next((side_of(p) for p in parts if side_of(p)), None) \
        or side_of(os.path.basename(os.path.dirname(path)))
    return Sequence(name=os.path.basename(path), form=parts[0], view=detected or "?",
                    px=px, scores=sc, times_ms=times, width=width, height=height)


def load_dataset(root: str) -> dict[tuple[str, str], Sequence]:
    """root/Data/annotations/Taegeuk_1st/Frontal/1st_Frontal.csv -> {(form, view): Sequence}"""
    out = {}
    for form_dir in sorted(os.listdir(os.path.join(root, "annotations"))):
        for view in ("Frontal", "Lateral"):
            d = os.path.join(root, "annotations", form_dir, view)
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".csv"):
                    seq = load_csv(os.path.join(d, fn))
                    seq.form, seq.view = form_dir, view
                    out[(form_dir, view)] = seq
    return out


# ------------------------------------------------------------ features
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-6)


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle ABC in degrees for arrays (T,2)."""
    u = _unit(a - b)
    v = _unit(c - b)
    cos = np.clip((u * v).sum(-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def smooth(x: np.ndarray, w: int = 5) -> np.ndarray:
    if w <= 1 or x.shape[0] < w:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, ((pad, pad),) + ((0, 0),) * (x.ndim - 1), mode="edge")
    if x.ndim == 1:
        return np.convolve(xp, k, mode="valid")[: x.shape[0]]
    return np.stack([np.convolve(xp[:, i], k, mode="valid")[: x.shape[0]]
                     for i in range(x.shape[1])], axis=1)


def features(seq: Sequence, min_score: float = 0.5, smooth_w: int = 5) -> np.ndarray:
    """(T,9) angle features in degrees. Frames with low-confidence joints are
    bridged by linear interpolation rather than dropped, so time stays aligned."""
    px = seq.px.copy()
    bad = (seq.scores < min_score) | np.isnan(px).any(-1)
    if bad.any():
        for j in range(12):
            col = px[:, j, :]
            ok = ~bad[:, j]
            if ok.sum() >= 2:
                for d in range(2):
                    col[:, d] = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok, d])
            px[:, j, :] = col
    feats = []
    for _, a, b, c in ANGLE_TRIPLETS:
        feats.append(angle_at(px[:, J[a]], px[:, J[b]], px[:, J[c]]))
    # torso tilt: spine vector (hip-mid -> shoulder-mid) vs image vertical
    hip_mid = (px[:, J["L_hip"]] + px[:, J["R_hip"]]) / 2
    sho_mid = (px[:, J["L_shoulder"]] + px[:, J["R_shoulder"]]) / 2
    spine = _unit(sho_mid - hip_mid)
    tilt = np.degrees(np.arccos(np.clip(spine[:, 1], -1, 1)))  # 0 = upright (y down)
    feats.append(tilt)
    F = np.stack(feats, axis=1)
    return smooth(F, smooth_w)


# ---------------------------------------------------------------- DTW
def dtw(ref: np.ndarray, stu: np.ndarray, band: int | None = None):
    """Returns (cost_normalised, path[(i,j)...]) aligning ref -> stu."""
    mu, sd = ref.mean(0), ref.std(0)
    sd[sd < 1e-6] = 1.0
    r = (ref - mu) / sd
    s = (stu - mu) / sd
    n, m = len(r), len(s)
    if band is None:
        band = max(60, int(0.2 * max(n, m)))
    INF = float("inf")
    D = np.full((n + 1, m + 1), INF)
    D[0, 0] = 0.0
    B = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        ri = r[i - 1]
        for j in range(lo, hi + 1):
            c = float(np.linalg.norm(ri - s[j - 1]))
            a, b, d = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
            best = a
            arg = 0
            if b < best:
                best, arg = b, 1
            if d < best:
                best, arg = d, 2
            D[i, j] = c + best
            B[i - 1, j - 1] = arg
    # backtrack from (n,m); if unreachable due to band, fall back to full-budget grid end
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        if i == 1:
            j -= 1
        elif j == 1:
            i -= 1
        else:
            arg = B[i - 1, j - 1]
            if arg == 0:
                i, j = i - 1, j - 1
            elif arg == 1:
                i -= 1
            else:
                j -= 1
        path.append((i - 1, j - 1))
    path.reverse()
    cost = D[n, m] / max(len(path), 1)
    return float(cost), path


# --------------------------------------------------------- segmentation
def segment_moves(feats: np.ndarray, min_gap: int = 12, pause_pct: float = 20.0):
    """Split a performance into 'moves' at motion pauses.

    Speed = ||d(features)/dt||; a move boundary is a local minimum of speed that
    sits under the pause_pct percentile of the speed distribution.
    """
    v = np.linalg.norm(np.diff(feats, axis=0), axis=1)
    v = smooth(v, 7)
    thr = np.percentile(v, pause_pct)
    cand = np.flatnonzero(v < thr)
    if len(cand) == 0:
        return [(0, len(feats) - 1)]
    groups, cur = [], [cand[0]]
    for idx in cand[1:]:
        if idx - cur[-1] <= 3:
            cur.append(idx)
        else:
            groups.append(cur)
            cur = [idx]
    groups.append(cur)
    centres = []
    for g in groups:
        seg = v[g[0]: g[-1] + 1]
        centres.append(int(g[0] + int(np.argmin(seg))))
    # enforce minimum gap
    kept = []
    for c in centres:
        if not kept or c - kept[-1] >= min_gap:
            kept.append(c)
    bounds = [0] + [c for c in kept if 0 < c < len(feats) - 1] + [len(feats) - 1]
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


# ------------------------------------------------------------- feedback
def describe(delta: float) -> str:
    if delta > 0:
        return f"{abs(delta):.0f}° more bent than reference"
    return f"{abs(delta):.0f}° straighter than reference"


def score_moves(ref_feats, stu_feats, segs_ref, segs_stu, threshold=15.0):
    """Per-move joint-angle comparison.

    threshold=None -> per-joint tolerances from TOLERANCE_DEG (recommended);
    a float applies one uniform tolerance to every joint (used by the
    injection test, which needs a fixed, known sensitivity).
    """
    tol = np.array([TOLERANCE_DEG.get(f, threshold or 15.0) for f in FEATURE_NAMES])
    if threshold is not None:
        tol = np.full(len(FEATURE_NAMES), float(threshold))
    out = []
    for k, ((r0, r1), (s0, s1)) in enumerate(zip(segs_ref, segs_stu)):
        r = ref_feats[r0:r1 + 1].mean(0)
        s = stu_feats[s0:s1 + 1].mean(0)
        d = s - r
        exceed = np.abs(d) / tol
        order = np.argsort(-exceed)
        flags = [(FEATURE_NAMES[i], float(d[i])) for i in order
                 if exceed[i] > 1.0][:3]
        out.append({
            "move": k + 1,
            "t": round(float(r0 / 30.0), 2),
            "mean_abs_delta": round(float(np.abs(d).mean()), 1),
            "max_abs_delta": round(float(np.abs(d).max()), 1),
            "max_exceedance": round(float(exceed.max()), 2),
            "flags": [{"joint": HUMAN[n], "key": n, "delta": round(v, 1),
                       "tol": float(tol[FEATURE_NAMES.index(n)]),
                       "msg": describe(v)} for n, v in flags],
            "deltas": {FEATURE_NAMES[i]: round(float(d[i]), 1) for i in range(len(d))},
        })
    return out


def analyse(ref: Sequence, stu: Sequence, threshold=None, band=None):
    rf, sf = features(ref), features(stu)
    cost, path = dtw(rf, sf, band=band)
    segs_ref = segment_moves(rf)
    # map each reference segment to the student frames DTW matched it to
    j_of_i = {}
    for i, j in path:
        j_of_i.setdefault(i, j)
    segs_stu = []
    for r0, r1 in segs_ref:
        js = [j_of_i[i] for i in range(r0, r1 + 1) if i in j_of_i] or [0]
        segs_stu.append((min(js), max(js)))
    moves = score_moves(rf, sf, segs_ref, segs_stu, threshold=threshold)
    # Camera side is a property of the comparison, not a footnote. Two modelling
    # attempts to remove the resulting deviation both failed (learned_gate.py,
    # view_calibration.py) and a third in 3D failed too, so the honest contract is
    # to mark the comparison rather than let a caller quote a number that is mostly
    # camera geometry. Views arrive inconsistently cased (dataset dirs vs our own
    # CSV filenames), so normalise before comparing.
    def _side(v):
        return side_of(v)

    sr, ss = _side(ref.view), _side(stu.view)
    if sr and ss:
        cross_side = sr != ss
        comparison = ("opposite-side (camera geometry dominates)" if cross_side
                      else "same-side")
    else:
        cross_side = False
        comparison = "view unknown (camera side not declared) - do not quote this as same-side"
    return {
        "ref": f"{ref.form}/{ref.view}", "stu": f"{stu.form}/{stu.view}",
        "cross_side": bool(cross_side),
        "comparison": comparison,
        "frames": [ref.n, stu.n],
        "dtw_cost_per_frame": round(cost, 4),
        "path_len": len(path),
        "n_moves": len(moves),
        "mean_abs_delta": round(float(np.mean([m["mean_abs_delta"] for m in moves])), 1),
        "flagged_moves": sum(1 for m in moves if m["flags"]),
        "moves": moves,
        "features_ref": rf, "features_stu": sf, "path": path,
    }


def dump(result: dict, path: str):
    slim = {k: v for k, v in result.items() if not k.startswith("features") and k != "path"}
    with open(path, "w") as fh:
        json.dump(slim, fh, indent=1)
