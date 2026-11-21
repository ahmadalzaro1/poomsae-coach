"""Per-joint uncertainty band on the coach card, plus Jev as a release policy.

Two separate additions, deliberately kept separate in the output:

1. BAND (deterministic). The card already reports one deviation per joint. We run the
   same video through a second pose model, score it against the same reference, and
   report how far the deviation moves when the estimator is swapped:
   ``band_deg[joint] = |delta_full - delta_heavy|``. That is a *lower bound* on the real
   uncertainty (two models reading identical pixels cannot differ by more than two takes
   of one athlete would), and it is labelled as such everywhere. It costs one extra pose
   pass per submission and needs no new data.

2. RELEASE (advisory). For each finding the card would print, Jev is asked whether that
   sentence should be shown, softened, or withheld, given the joint, the deviation, the
   band, the tolerance and the camera side. Its answer is written to a separate field and
   can never remove a flag: the flag is a measurement fact ("this joint is beyond
   tolerance"), the release is a wording decision ("should the athlete be told, and how").
   If the key is absent or the API misbehaves, release is reported as unavailable and the
   card is unchanged - the deterministic path never depends on it.

Everything is written back into out/card_*.json, so make_card.py, the widget, the overlay
renderer and the demo page all read one artifact and cannot disagree.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from poomsae.core import JOINTS, TOLERANCE_DEG, analyse, load_csv  # noqa: E402
from poomsae.jev_gate import KEY, ask, validate  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")
CACHE = os.path.join(OUT, "uncertainty")
RELEASE_CACHE = os.path.join(OUT, "release_cache.json")
MODELS = {"full": os.path.join(HERE, "models", "pose_landmarker_full.task"),
          "heavy": os.path.join(HERE, "models", "pose_landmarker_heavy.task")}
MP_IDX = {
    "L_shoulder": 11, "R_shoulder": 12, "L_elbow": 13, "R_elbow": 14,
    "L_wrist": 15, "R_wrist": 16, "L_hip": 23, "R_hip": 24,
    "L_knee": 25, "R_knee": 26, "L_ankle": 27, "R_ankle": 28,
}


def extract_two(video: str, tag: str, rebuild: bool = False):
    """Run BOTH pose models over the same frames; cache (px, visibility) per model."""
    os.makedirs(CACHE, exist_ok=True)
    paths = {k: os.path.join(CACHE, f"{tag}_{k}.npz") for k in MODELS}
    if not rebuild and all(os.path.exists(p) for p in paths.values()):
        z = {k: np.load(p) for k, p in paths.items()}
        meta = json.load(open(os.path.join(CACHE, f"{tag}_meta.json")))
        return z, meta

    cap = cv2.VideoCapture(video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    px = {k: np.full((n_total, 12, 2), np.nan) for k in MODELS}
    sc = {k: np.full((n_total, 12), np.nan) for k in MODELS}
    idx = [MP_IDX[j] for j in JOINTS]

    markers = {}
    for k in MODELS:
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODELS[k]),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        markers[k] = vision.PoseLandmarker.create_from_options(opts)

    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * 1000 / fps)
            for k, marker in markers.items():
                res = marker.detect_for_video(img, ts)
                if res.pose_landmarks:
                    lm = res.pose_landmarks[0]
                    for slot, pi in enumerate(idx):
                        px[k][i, slot] = (lm[pi].x * W, lm[pi].y * H)
                        sc[k][i, slot] = getattr(lm[pi], "visibility", 1.0)
            i += 1
    finally:
        for marker in markers.values():
            marker.close()

    for k in MODELS:
        px[k], sc[k] = px[k][:i], sc[k][:i]
        np.savez_compressed(paths[k], px=px[k], sc=sc[k])
    meta = {"video": os.path.basename(video), "frames": i, "width": W, "height": H, "fps": fps}
    json.dump(meta, open(os.path.join(CACHE, f"{tag}_meta.json"), "w"))
    print(f"  extracted {i} frames with {len(MODELS)} models -> {CACHE}/{tag}_*.npz")
    return {k: {"px": px[k], "sc": sc[k]} for k in MODELS}, meta


def sequences(z, meta, form: str, view: str):
    """One Sequence per estimator, same video, same timeline."""
    out = {}
    for k, v in z.items():
        s = type("S", (), {})()
        s.name, s.form, s.view = f"{k}", form, view
        s.px = v["px"].copy()
        s.scores = v["sc"].copy()
        s.times_ms = np.arange(len(v["px"])) * (1000.0 / meta["fps"])
        s.width, s.height = meta["width"], meta["height"]
        s.n = len(v["px"])
        out[k] = s
    return out


def band_for(ref, seqs, tol_source=None):
    """Score the same reference against each estimator; report the swap sensitivity."""
    res = {k: analyse(ref, s, threshold=None) for k, s in seqs.items()}
    base, alt = res["full"], res["heavy"]
    bands = []
    for m0, m1 in zip(base["moves"], alt["moves"]):
        if m0["move"] != m1["move"] or abs(m0["t"] - m1["t"]) > 0.5:
            bands.append(None)          # segmentation disagreed: refuse to invent a band
            continue
        bands.append({j: round(abs(float(m0["deltas"].get(j, 0.0)) - float(m1["deltas"].get(j, 0.0))), 1)
                      for j in m0["deltas"]})
    return res, bands


def jev_release(card, moves, bands, tol_by_joint):
    """Ask Jev about each finding. Advice only: never removes or changes a flag."""
    if not KEY:
        return {"available": False, "reason": "no key in keychain"}
    try:
        cache = json.load(open(RELEASE_CACHE))
    except Exception:
        cache = {}
    jobs = []
    for mv, band in zip(moves, bands):
        for f in mv["flags"]:
            key = f.get("key") or ""
            jobs.append((mv, f, band, key))

    def run(job):
        mv, f, band, key = job
        b = None if band is None else band.get(key)
        tol = float(tol_by_joint.get(key, f.get("tol") or 15.0))
        payload = {"model": "jev-latest",
                   "state": {"task": "taekwondo poomsae coach card",
                             "form": card.get("ref", "").split("/")[0].replace("Taegeuk_", "Taegeuk "),
                             "reference_side_relative_to_athlete":
                                 "opposite" if card.get("cross_side") else "same",
                             "comparison_note": card.get("comparison", ""),
                             "camera": "single 1280x720-1080p camera, monocular 2D pose estimation",
                             "moves_in_form": card.get("n_moves"),
                             "finding": {"move_index": mv["move"], "joint": f.get("joint"),
                                         "deviation_deg": round(float(f.get("delta") or 0.0), 1),
                                         "tolerance_deg": tol,
                                         "estimator_swap_band_deg": b},
                             "known_limitation":
                                 "The band is the difference between two pose models reading the same "
                                 "frames, so it is a lower bound on measurement uncertainty. When the "
                                 "reference footwork is filmed from the opposite side of the body, a "
                                 "large deviation is produced by camera geometry rather than by the "
                                 "athlete."},
                   "questions": {
                       "release": {"type": "choice",
                                   "instructions": "Should this correction be shown to the athlete, "
                                                   "and how?",
                                   "criteria": {
                                       "show": "Attributable to the athlete and precise enough to state "
                                               "plainly.",
                                       "soften": "Worth saying, but state it as direction only - the "
                                                 "band is too wide for the stated number.",
                                       "withhold": "Do not show: camera geometry or measurement limits "
                                                   "explain it."}},
                       "attributable": {"type": "noul",
                                        "instructions": "The supplied evidence explicitly supports "
                                                        "attributing this deviation to the athlete."}}}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        if key in cache:                 # identical state must give an identical answer
            return {"ms": 0, "err": None, "v": cache[key], "cached": True, "key": key}
        body, ms, err = ask(payload)
        v = validate((body or {}).get("answers", {}),
                     ["release", "attributable"], {"release": "choice", "attributable": "noul"})
        if v:
            cache[key] = v
        return {"ms": ms, "err": err, "v": v, "cached": False, "key": key}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        outs = list(ex.map(run, jobs))

    n_ok = 0
    for (mv, f, band, key), o in zip(jobs, outs):
        v = o["v"]
        if not v:
            f["release"] = "policy_unavailable"
            f["release_error"] = o["err"]
            continue
        n_ok += 1
        f["release"] = v["release"]["choice"]
        f["release_probs"] = {k: round(float(x), 3) for k, x in v["release"]["probs"].items()}
        f["attributable"] = round(float(v["attributable"]), 3)
        f["release_ms"] = o["ms"]
        f["release_cached"] = bool(o.get("cached"))
    json.dump(cache, open(RELEASE_CACHE, "w"), indent=0)
    n_cached = sum(1 for o in outs if o.get("cached"))
    counts = {c: sum(1 for (mv, f, b, k), o in zip(jobs, outs)
                     if o["v"] and f.get("release") == c) for c in ("show", "soften", "withhold")}
    return {"available": True, "findings": len(jobs), "answered": n_ok, "counts": counts,
            "model": "jev-latest", "cached_decisions": n_cached,
            "determinism": ("decisions are cached by state hash: the same finding always gets the same "
                            "release, so re-running the pipeline cannot change a card"),
            "note": ("Release is a wording decision and never removes a flag: the flag states that the "
                     "joint is beyond tolerance, release decides what the athlete is told.")}


def main() -> int:
    base = os.path.join(HERE, "data", "Data")
    video = os.path.join(base, "video", "Taegeuk_1st", "Frontal", "1st_Frontal.mp4")
    res_dir = os.path.join(HERE, "data", "Data", "annotations", "Taegeuk_1st")
    refs = {"sameview": ("Frontal", load_csv(os.path.join(res_dir, "Frontal", "1st_Frontal.csv"))),
            "crossview": ("Lateral", load_csv(os.path.join(res_dir, "Lateral", "1st_Lateral.csv")))}

    print("stage 1: extract both estimators on the student video")
    z, meta = extract_two(video, "taegeuk1_frontal")
    seqs = sequences(z, meta, "Taegeuk_1st", "Frontal")

    print("\nstage 2: score each estimator against each reference")
    summary = {"generated": "2026-11-21", "video": meta["video"], "frames": meta["frames"],
               "band_note": ("|delta_full - delta_heavy| on identical frames: a LOWER BOUND on "
                             "measurement uncertainty, not two takes of one athlete"),
               "tolerance_source": None, "cards": {}}
    for tag, (view_name, ref) in refs.items():
        res, bands = band_for(ref, seqs)
        card_path = os.path.join(OUT, f"card_{tag}.json")
        card = json.load(open(card_path)) if os.path.exists(card_path) else {
            k: v for k, v in res["full"].items() if k not in ("features_ref", "features_stu", "path")}
        # attach band per move and per flag
        per_joint = {}
        for mv, band in zip(card["moves"], bands):
            mv["band_deg"] = band
            mv["band_max_deg"] = None if band is None else round(max(band.values()), 1)
            mv["band_note"] = "lower bound: two pose models on the same frames"
            for f in mv["flags"]:
                f["band_deg"] = None if band is None else band.get(f.get("key"))
            if band:
                for j, val in band.items():
                    per_joint.setdefault(j, []).append(val)
        card["uncertainty"] = {
            "band_note": summary["band_note"],
            "per_joint_band_p50_deg": {j: round(float(np.percentile(v, 50)), 1) for j, v in per_joint.items()},
            "per_joint_band_p95_deg": {j: round(float(np.percentile(v, 95)), 1) for j, v in per_joint.items()},
            "moves_with_band": sum(1 for b in bands if b),
            "moves_total": len(bands),
            "paired_moves_disagreed": sum(1 for b in bands if b is None),
        }
        print(f"\n--- [{tag}] band, per joint p50 / p95 ---")
        for j in card["uncertainty"]["per_joint_band_p50_deg"]:
            print(f"   {j:<12} {card['uncertainty']['per_joint_band_p50_deg'][j]:>5.1f}° / "
                  f"{card['uncertainty']['per_joint_band_p95_deg'][j]:>5.1f}°")

        card["release_policy"] = jev_release(card, card["moves"], bands, TOLERANCE_DEG)
        if card["release_policy"].get("available"):
            print(f"   release: {card['release_policy']['counts']} "
                  f"({card['release_policy']['answered']}/{card['release_policy']['findings']} answered)")
        else:
            print(f"   release unavailable: {card['release_policy'].get('reason')}")
        json.dump(card, open(card_path, "w"), indent=1)
        summary["cards"][tag] = {"uncertainty": card["uncertainty"],
                                 "release_policy": {k: v for k, v in card["release_policy"].items()
                                                    if k != "note"}}

    json.dump(summary, open(os.path.join(OUT, "uncertainty.json"), "w"), indent=1)
    print("\nwrote out/uncertainty.json and updated both cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
