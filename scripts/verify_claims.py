#!/usr/bin/env python3
"""Re-derive every number the README quotes, from the committed JSON.

A cited number the reader cannot re-run is an assertion, not evidence. This script
reads only files in this repo (no dataset, no models, no network) and exits non-zero
if a single quoted claim has drifted from the artifacts.

    python3 scripts/verify_claims.py          # check
    python3 scripts/verify_claims.py --show   # print the values it checks
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

CHECKS: list[tuple[str, bool, str]] = []


def load(name):
    with open(os.path.join(OUT, name)) as fh:
        return json.load(fh)


def check(label, got, expected, tol=0.051):
    ok = got is not None and abs(float(got) - float(expected)) <= tol
    CHECKS.append((label, ok, f"{got} vs README {expected}"))
    return ok


def check_eq(label, got, expected):
    ok = got == expected
    CHECKS.append((label, ok, f"{got!r} vs README {expected!r}"))
    return ok


def main(show=False):
    sr = load("self_reference.json")
    bench = sr["benchmark"]
    same = sr["cards"]["sameview"]
    cross = sr["cards"]["crossview"]
    ec = load("eight_class.json")
    ba = load("between_athlete.json")
    rep = load("repeatability.json")
    un = load("uncertainty.json")

    # --- the gate itself -----------------------------------------------------
    check("gate_k", sr["gate_k"], 2.5)
    check("R_elbow effective threshold (deg)", bench["effective_threshold_deg"]["R_elbow"], 21.0)
    check("L_elbow effective threshold (deg)", bench["effective_threshold_deg"]["L_elbow"], 18.1)
    check("elbow measurement tolerance (deg)", sr["tolerance_deg"]["R_elbow"], 18.0)
    check_eq("cells (moves x joints)", bench["cells_with_band"], 171)

    # --- clean arm: the published floor -------------------------------------
    clean = bench["clean"]
    check("clean-arm candidate faults", clean["faults"], 3)
    check("clean-arm fault share of cells (%)", 100.0 * clean["faults"] / 171, 1.75, tol=0.02)
    check_eq("clean-arm old-gate flags", clean["old_gate_flags"], 9)

    # --- style arm: a habit must not become a fault -------------------------
    style = bench["style_only"]
    check_eq("style-arm faults beyond clean baseline", style["faults_beyond_baseline"], 0)
    check_eq("style-arm old-gate flags", style["old_gate_flags"], 39)
    check("style-arm recovered R_elbow offset (deg)", style["offset_recovery_deg"]["R_elbow"], 20.4)
    check("style-arm recovered L_knee offset (deg)", style["offset_recovery_deg"]["L_knee"], -15.0)

    # --- zero-base sweep: refusals and detections ---------------------------
    sw = bench["sweep_zero_base"]
    check("0.5x effective threshold, detection", sw["0.5x"]["detection_rate_judgeable"], 0.0, tol=0.001)
    check("0.9x effective threshold, detection", sw["0.9x"]["detection_rate_judgeable"], 0.0, tol=0.001)
    check("1.05x effective threshold, detection of judgeable cells",
          sw["1.05x"]["detection_rate_judgeable"], 1.0)
    check("1.05x all-trial detection (band-blind count as misses)",
          sw["1.05x"]["detection_rate"], 0.9298, tol=0.001)
    check_eq("1.05x band-blind cells", sw["1.05x"]["band_blind"], 12)

    # --- measured-base sweep: the artefact base consumes headroom -----------
    mb = bench["sweep_measured_base"]
    check("1.05x on a measured base, judgeable detection", mb["1.05x"]["detection_rate_judgeable"], 0.6038, tol=0.001)
    check("1.6x on a measured base, judgeable detection", mb["1.6x"]["detection_rate_judgeable"], 0.8788, tol=0.001)

    # --- the two cards -------------------------------------------------------
    check_eq("sameview candidate faults", same["verdict_counts"]["fault"], 3)
    check_eq("sameview own-offset cells", same["verdict_counts"]["style"], 0)
    check_eq("sameview card flags", same["card_flag_count"], 9)
    check_eq("crossview candidate faults", cross["verdict_counts"]["fault"], 11)
    check_eq("crossview own-offset cells", cross["verdict_counts"]["style"], 9)
    check_eq("crossview raw cells over tolerance", cross["old_gate_flags"], 50)
    for name, card in (("sameview", same), ("crossview", cross)):
        check_eq(f"{name} flags byte-identical", card["flags_unchanged"], True)

    # --- the two measurements behind the fix --------------------------------
    groups = rep["cards"]["sameview"]["pairs"]["elbow"]["groups"]
    for tech, want_sd in (("arae makki", 6.5), ("momtong jireugi", 8.0)):
        check(f"repeat sd, {tech} (deg)", groups[tech]["sd"], want_sd)
    check_eq("repeat n, arae makki", groups["arae makki"]["n"], 6)
    traj = ba["trajectory"]["pooled"]
    for key, want in (("elbow_traj_sd_mean", 32.7), ("elbow_traj_sd_max", 68.5),
                      ("knee_traj_sd_mean", 4.7), ("knee_traj_sd_max", 11.3)):
        if key in traj:
            check(f"cross-athlete {key}", traj[key], want, tol=0.11)

    # --- the labelled-dataset result and its four controls -------------------
    check("8-class accuracy (5-fold, player-grouped)", ec["eight_class"]["mean"], 0.9801, tol=0.001)
    check("control: 1-nearest-neighbour", ec["controls"]["one_nearest_neighbour"]["mean"], 0.9735, tol=0.001)
    check("control: single frame only", ec["controls"]["single_frame_only"]["mean"], 0.9258, tol=0.001)
    check("control: shuffled labels", ec["shuffled_control"]["mean"], 0.1276, tol=0.001)
    check("our DTW protocol, binary per movement", ec["comparison"]["our_dtw_binary"], 0.757, tol=0.001)
    check("paper's own 8-class figure", ec["comparison"]["paper_8class"], 0.61, tol=0.001)

    # --- the two headline faults, re-derived end to end ----------------------
    card = load("card_sameview.json")
    off = same["offsets_deg"]
    gate = same["effective_threshold_deg"]
    headline = {6: ("R_elbow", 21.5, 0.4, 21.1, 21.0), 13: ("L_elbow", 22.1, 0.1, 22.0, 18.1)}
    for move_no, (key, delta, offset, residual, g) in headline.items():
        mv = next(m for m in card["moves"] if m["move"] == move_no)
        flag = next(f for f in mv["flags"] if f["key"] == key)
        check(f"move {move_no} measured delta ({key}, deg)", flag["delta"], delta, tol=0.01)
        check(f"move {move_no} own offset ({key}, deg)", off[key], offset, tol=0.01)
        resid = flag["delta"] - off[key]
        check(f"move {move_no} residual ({key}, deg)", resid, residual, tol=0.02)
        check(f"move {move_no} gate ({key}, deg)", gate[key], g, tol=0.01)
        check_eq(f"move {move_no} clears its gate", abs(resid) > gate[key], True)

    # --- camera geometry: the cross-view mean the README quotes --------------
    check("crossview mean deviation (deg)", load("card_crossview.json")["mean_abs_delta"], 12.5, tol=0.01)

    # --- 3D refusal ----------------------------------------------------------
    c3 = load("crossview_3d.json")
    check("3D/2D deviation range ratio", c3["range_ratio_3d_over_2d"], 0.867, tol=0.001)
    snr = c3["signal_noise"]
    check("SNR 2D", snr["snr_2d"], 8.65, tol=0.01)
    check("SNR 3D", snr["snr_3d"], 6.59, tol=0.01)
    check_eq("3D verdict says no benefit", c3["verdict"].startswith("3D does not"), True)

    # --- view calibration cost ----------------------------------------------
    vc = load("view_calibration.json")["injection_recall_after_calibration"]
    check("calibrated recall, 10 deg injection", vc["10deg"], 0.3889, tol=0.001)
    check("calibrated recall, 20 deg injection", vc["20deg"], 0.9028, tol=0.001)

    # --- the Jev policy engine, blind arm only -------------------------------
    jev = load("jev_gate.json")
    check("Jev blind AUC, withhold vs opposite-side", jev["experiment_a_blind"]["auc_p_withhold_vs_opposite_side"], 0.6125, tol=0.001)
    check("Jev single-reading AUC", jev["experiment_b_single_reading"]["auc_p_reject_vs_gap10"], 0.4781, tol=0.001)

    # --- cards: flag cap and cross-view repeat/camera inflation ---------------
    check_eq("crossview card flags (3-per-move cap)", load("card_crossview.json")["self_reference"]["card_flag_count"], 40)
    check_eq("flags per move cap", same["card_flag_cap_per_move"], 3)
    cross_rep = rep["cards"]["crossview"]["pairs"]["elbow"]["pooled"]
    check("crossview repeat mean (camera-inflated)", cross_rep["mean"], 26.5, tol=0.01)
    check("crossview repeat sd (camera-inflated)", cross_rep["sd"], 14.1, tol=0.01)

    # --- residual-naming and subtle-fault refusal -----------------------------
    check("naming held-out accuracy (calibrated on form 1)", load("naming_forms.json")["heldout_mean_accuracy"], 0.837, tol=0.001)
    sf = load("subtle_faults.json")
    fam = sf["families"]
    check("max jitter sigma (deg)", max(sf["jitter_sigma_deg"].values()), 0.615, tol=0.001)
    check("subtle fault: arm-height tolerance (deg)", fam["arm height (block too high/low)"]["tolerance_deg"], 12.0)
    check("subtle fault: arm-height 50% detection (deg)", fam["arm height (block too high/low)"]["same_side"]["m50_deg"], 15.0)
    check("subtle fault: elbow-extension tolerance (deg)", fam["elbow extension (punch short)"]["tolerance_deg"], 18.0)
    check("subtle fault: elbow-extension 50% detection (deg)", fam["elbow extension (punch short)"]["same_side"]["m50_deg"], 20.0)

    # --- bands and release ---------------------------------------------------
    band = un["cards"]["sameview"]["uncertainty"]
    check_eq("sameview moves with an estimator band", band["moves_with_band"], band["moves_total"])
    rel = un["cards"]["sameview"]["release_policy"]["counts"]
    check_eq("release: show", rel["show"], 2)
    check_eq("release: soften", rel["soften"], 6)
    check_eq("release: withhold", rel["withhold"], 1)

    tour = load("form_tour.json")

    # --- the eight-form tour as *written* in the two docs that quote it: the table must equal the
    # artifact cell for cell, in both layouts, or the doc is retyped rather than derived
    def _readme_rows():
        txt = open(os.path.join(ROOT, "README.md")).read()
        found = {}
        for m in re.finditer(r"\| Taegeuk (\d) Jang \| ([\d,]+) \| ([\d.]+)° \| ([\d.]+)° \| ([\d,]+) \(", txt):
            found[int(m.group(1))] = (int(m.group(2).replace(",", "")), float(m.group(3)),
                                      float(m.group(4)), int(m.group(5).replace(",", "")))
        return found

    def _results_rows():
        txt = open(os.path.join(ROOT, "docs/RESULTS.md")).read()
        found = {}
        for m in re.finditer(r"\| Taegeuk (\d) Jang \| ([\d,]+) \| ([\d,]+) \| ([\d.]+)° \| ([\d.]+)° \|", txt):
            found[int(m.group(1))] = (int(m.group(2).replace(",", "")), float(m.group(4)),
                                      float(m.group(5)), int(m.group(3).replace(",", "")))
        return found

    for path, rows in (("README.md", _readme_rows()), ("docs/RESULTS.md", _results_rows())):
        check_eq(f"{path}: all eight tour rows present", len(rows), 8)
        drift = [n for n, r in ((r["form"], r) for r in tour)
                 if rows.get(n) != (r["agreement"]["frames_compared"],
                                    r["agreement"]["p50_abs_delta_deg"],
                                    r["agreement"]["p95_abs_delta_deg"],
                                    r["agreement"]["frames_gross"])]
        check_eq(f"{path}: tour rows equal out/form_tour.json (drifted: {drift or 'none'})", drift, [])

    # --- the eight-form tour: a measurement that travels, and the frames it drops
    tour = load("form_tour.json")
    check_eq("tour covers 8 forms", len(tour), 8)
    check("form 1 |d| p50 (deg)", tour[0]["agreement"]["p50_abs_delta_deg"], 1.1, tol=0.05)
    check("form 1 |d| p95, gross frames excluded (deg)", tour[0]["agreement"]["p95_abs_delta_deg"], 19.9, tol=0.05)
    check("worst p95 of the eight forms (deg)",
          max(r["agreement"]["p95_abs_delta_deg"] for r in tour), 28.3, tol=0.05)
    check_eq("frames compared, all eight forms",
             sum(r["agreement"]["frames_compared"] for r in tour), 8664)
    check_eq("gross pose-failure frames, all eight forms",
             sum(r["agreement"]["frames_gross"] for r in tour), 2170)

    # --- verdict words: the prose must match the artifact, not the other way round
    def verdict_of(move_no, joint):
        mv = [m for m in card["moves"] if m["move"] == move_no][0]
        return mv["attribution"]["per_joint"][joint]["verdict"]

    check_eq("move 6 right elbow verdict", verdict_of(6, "R_elbow"), "noise")
    check_eq("move 13 left elbow verdict", verdict_of(13, "L_elbow"), "noise")
    check_eq("same-side fault cells",
             sorted((mv["move"], j) for mv in card["moves"]
                    for j, v in mv["attribution"]["per_joint"].items() if v["verdict"] == "fault"),
             [(15, "L_shoulder"), (15, "R_elbow"), (16, "L_shoulder")])
    sr_cards = load("self_reference.json")
    check("resolved gate, right elbow (2.5 x own scale)",
          2.5 * sr_cards["cards"]["sameview"]["scale_deg"]["R_elbow"], 28.2, tol=0.05)
    check("resolved gate, left elbow (2.5 x own scale)",
          2.5 * sr_cards["cards"]["sameview"]["scale_deg"]["L_elbow"], 27.4, tol=0.05)

    # --- learned gate: the refusal, from its own artifact ---------------------
    lg = load("learned_gate.json")
    check("learned gate AUC (best arm, full feature set)", lg["auc"]["full"], 0.7066, tol=0.001)
    check("learned gate FPR at matched TPR", lg["full"]["fpr_at_matched_tpr"], 0.9167, tol=0.001)
    check("learned gate rule FPR", lg["rule"]["fpr"], 0.8846, tol=0.001)

    if show:
        for label, ok, detail in CHECKS:
            print(f"{'OK  ' if ok else 'FAIL'} {label:52s} {detail}")

    bad = [c for c in CHECKS if not c[1]]
    print(f"\n{len(CHECKS) - len(bad)}/{len(CHECKS)} documented claims re-derived from committed artifacts")
    if bad:
        print("\nDRIFT (fix the README or the artifact, not this check):")
        for label, _, detail in bad:
            print(f"  - {label}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(show="--show" in sys.argv))
