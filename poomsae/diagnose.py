"""Audit: where does OUR pose disagree with the published annotation, and does
the scorer behave sanely on a same-view comparison?"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import FEATURE_NAMES, JOINTS, analyse, load_csv

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = os.path.join(HERE, "data", "Data")
ours = load_csv(os.path.join(HERE, "out", "pose_taegeuk1_frontal.csv"))
annot = load_csv(os.path.join(base, "annotations", "Taegeuk_1st", "Frontal", "1st_Frontal.csv"))
n = min(ours.n, annot.n)
err = np.linalg.norm(ours.px[:n] - annot.px[:n], axis=-1)
print("per-joint landmark error, pixels @1080p (n=%d frames)" % n)
for i, name in enumerate(JOINTS):
    print(f"  {name:<12} mean {err[:,i].mean():>6.1f}  p90 {np.percentile(err[:,i],90):>6.1f}")

print("\nconfidence: ours vs published")
for i, name in enumerate(JOINTS):
    print(f"  {name:<12} ours {ours.scores[:,i].mean():.2f}  theirs {annot.scores[:,i].mean():.2f}")

same = analyse(annot, ours, threshold=None)          # per-joint tolerances (the shipped config)
uniform = analyse(annot, ours, threshold=15.0)       # for reference
print(f"\nSAME-VIEW card (annotation frontal as reference, our video pose as student):")
print(f"  moves {same['n_moves']}  mean|Δ| {same['mean_abs_delta']}°  "
      f"flagged {same['flagged_moves']}/{same['n_moves']} (per-joint tol)  "
      f"| {uniform['flagged_moves']} at uniform 15°")
for m in same["moves"][:6]:
    top = m["flags"][0]["msg"] if m["flags"] else "within tolerance"
    print(f"  move {m['move']:>2} t={m['t']:>5.2f}s mean|Δ| {m['mean_abs_delta']:>5.1f}° -> {top}")

json.dump({"per_joint_px_mae": {j: round(float(err[:,i].mean()),1) for i,j in enumerate(JOINTS)},
           "same_view_mean_delta": same["mean_abs_delta"],
           "same_view_flags_perjoint": same["flagged_moves"],
           "same_view_flags_uniform15": uniform["flagged_moves"],
           "same_view_n_moves": same["n_moves"],
           "landmark_mae_px": round(float(err.mean()), 1),
           "landmark_p90_px": round(float(np.percentile(err, 90)), 1)},
          open(os.path.join(HERE, "out", "pose_audit.json"), "w"), indent=1)
