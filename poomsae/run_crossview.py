"""Cross-view generalisation benchmark.

Question the literature does not answer with numbers: if the reference
performance is recorded frontally, how badly does the scorer mis-rate the SAME
performance recorded laterally? That is the camera-angle fragility every coach
with a phone hits on day one.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import load_dataset, analyse, FEATURE_NAMES, segment_moves, features

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Data")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset(ROOT)
    forms = sorted({f for f, _ in ds})
    rows = []
    t0 = time.time()
    for form in forms:
        front, lat = ds[(form, "Frontal")], ds[(form, "Lateral")]
        # control: same view, same take -> must be ~0 deviation
        ctrl = analyse(front, front, threshold=15.0)
        cv = analyse(front, lat, threshold=None)          # per-joint tolerances
        cv_u = analyse(front, lat, threshold=15.0)        # uniform 15 deg
        seg_f, seg_l = segment_moves(features(front)), segment_moves(features(lat))
        rows.append({
            "form": form,
            "frames": [front.n, lat.n],
            "moves_detected": [len(seg_f), len(seg_l)],
            "control_mean_abs_delta": ctrl["mean_abs_delta"],
            "control_flags": ctrl["flagged_moves"],
            "crossview_mean_abs_delta": cv["mean_abs_delta"],
            "crossview_max_abs_delta": max(m["max_abs_delta"] for m in cv["moves"]),
            "crossview_dtw": cv["dtw_cost_per_frame"],
            "crossview_flags_perjoint_tol": cv["flagged_moves"],
            "crossview_flags_uniform15": cv_u["flagged_moves"],
            "crossview_n_moves": cv["n_moves"],
        })
        print(f"{form:<13} frames {front.n:>4}/{lat.n:<4} moves {len(seg_f):>2}/{len(seg_l):<2} "
              f"control Δ {ctrl['mean_abs_delta']:>5.1f}° (flags {ctrl['flagged_moves']:>2}) | "
              f"cross-view Δ {cv['mean_abs_delta']:>5.1f}° max {rows[-1]['crossview_max_abs_delta']:>5.1f}° "
              f"DTW {cv['dtw_cost_per_frame']:.3f} flags {cv['flagged_moves']}/{cv['n_moves']}"
              f" (uniform-15°: {cv_u['flagged_moves']})")
        if form == forms[0]:
            json.dump({k: v for k, v in cv.items() if k not in ("features_ref", "features_stu", "path")},
                      open(os.path.join(OUT, "crossview_example.json"), "w"), indent=1)

    dur = time.time() - t0
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "Taekwondo Poomsae Taegeuk 1-8, dual-view, CC-BY 10.17632/bjy7vr4xkt.1",
        "forms": rows,
        "aggregate": {
            "mean_control_delta_deg": round(float(np.mean([r["control_mean_abs_delta"] for r in rows])), 2),
            "mean_crossview_delta_deg": round(float(np.mean([r["crossview_mean_abs_delta"] for r in rows])), 2),
            "control_flags_total": int(sum(r["control_flags"] for r in rows)),
            "crossview_flags_uniform15_total": int(sum(r["crossview_flags_uniform15"] for r in rows)),
            "crossview_flags_perjoint_tol_total": int(sum(r["crossview_flags_perjoint_tol"] for r in rows)),
            "moves_total": int(sum(r["crossview_n_moves"] for r in rows)),
            "runtime_s": round(dur, 1),
        },
    }
    # per-feature cross-view deviation, pooled
    per_feat = []
    for form in forms:
        cv = analyse(ds[(form, "Frontal")], ds[(form, "Lateral")])
        per_feat.append(np.mean([[abs(m["deltas"][f]) for f in FEATURE_NAMES] for m in cv["moves"]], axis=0))
    per_feat = np.mean(per_feat, axis=0)
    summary["per_feature_crossview_delta_deg"] = {f: round(float(v), 1) for f, v in zip(FEATURE_NAMES, per_feat)}
    json.dump(summary, open(os.path.join(OUT, "crossview_benchmark.json"), "w"), indent=1)
    print("\n--- aggregate ---")
    print(json.dumps(summary["aggregate"], indent=1))
    print("\n--- per-feature cross-view disagreement (mean |Δ| in degrees) ---")
    for f, v in summary["per_feature_crossview_delta_deg"].items():
        print(f"  {f:<12} {v:>5.1f}°")


if __name__ == "__main__":
    main()
