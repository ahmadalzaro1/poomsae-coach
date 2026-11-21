"""Self-referenced elbow gate: separate an athlete's *style* from their *faults*.

The defect this fixes
---------------------
The gate compares an athlete to a reference performed by someone else and flags every joint
whose deviation exceeds a fixed tolerance (elbow 18 deg). Measured on 94 athletes
(`poomsae/between_athlete.py`), the same technique varies across athletes with sd 32.7 deg in the
elbow and 4.7 deg in the knee. So an 18 deg elbow gate is 2.3x the *repeat* spread but only
0.55x the *cross-athlete* spread: it fires on ordinary human variation. The number was right for
measurement and wrong for attribution.

The fix
-------
Per joint, over an athlete's own performance:
    offset   = median(delta)          -> systematic style/build difference. NOT a fault.
    residual = delta - offset         -> this move versus this athlete's own habit.
    scale    = max(1.4826*MAD(residual), floor)
               floor = max(this pipeline's measured estimator disagreement p50,
                           measured repeat spread where we have it for that joint)
    verdict  = "fault"  if |residual| > tolerance AND |residual| > K*scale
               "style"  if |offset|   > tolerance
               "noise"  otherwise

Both conditions are required: tolerance keeps the finding physically significant, K*scale keeps it
surprising *for this athlete*. K is reported by name (GATE_K) the way TOLERANCE_SOURCE is, because
a constant that decides a coach's card must be visible.

What is deliberately NOT changed
--------------------------------
`flags` stay exactly as they were: deterministic measurement facts. This layer only classifies
them. The release layer (Jev) still cannot remove a flag; neither can this.

Certification arms (known truth, so the gate can be wrong in a way we can see)
------------------------------------------------------------------------------
  clean        no injection                        -> flags are measurement artefacts; disclosed
  style        +20 R_elbow, -15 L_knee everywhere  -> must produce ZERO faults, offset recovered
  fault        style + 25 deg at one cell          -> must find that cell, low collateral
  subtle       style + 12 deg at one cell          -> must refuse (below tolerance)
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "out")

GATE_K = 2.5
MIN_MOVES = 6
JOINTS = ["L_elbow", "R_elbow", "L_knee", "R_knee", "L_hip", "R_hip",
          "L_shoulder", "R_shoulder", "torso_tilt"]


def tolerances() -> tuple[dict, str]:
    """Per-joint tolerances and their provenance, from the same source the cards use."""
    try:
        from . import core as C
        return dict(C.TOLERANCE_DEG), str(getattr(C, "TOLERANCE_SOURCE", "core defaults"))
    except Exception:                                     # pragma: no cover - import guard
        p = os.path.join(OUT, "tolerance_override.json")
        base = {j: (18.0 if "elbow" in j else 10.0 if ("knee" in j or "hip" in j)
                    else 12.0 if "shoulder" in j else 5.0) for j in JOINTS}
        if os.path.exists(p):
            ov = json.load(open(p))
            base.update({k: float(v) for k, v in ov.items() if k in base})
            return base, f"{os.path.relpath(p)} ({ov.get('source', 'no source field')})"
        return base, "built-in fallback (core import failed)"


def floors() -> tuple[dict, dict]:
    """Per-joint scale floor = the larger of two measured quantities. No invented constants."""
    band, rep, src = {}, {}, {}
    upath = os.path.join(OUT, "uncertainty.json")
    if os.path.exists(upath):
        u = json.load(open(upath))
        card = u.get("cards", {}).get("sameview", {}).get("uncertainty", {})
        band = {k: float(v) for k, v in card.get("per_joint_band_p50_deg", {}).items()}
        src["estimator_band_p50"] = "out/uncertainty.json"
    rpath = os.path.join(OUT, "repeatability.json")
    if os.path.exists(rpath):
        r = json.load(open(rpath))
        pairs = r.get("cards", {}).get("sameview", {}).get("pairs", {})
        for pair, keys in (("elbow", ("L_elbow", "R_elbow")), ("knee", ("L_knee", "R_knee"))):
            g = pairs.get(pair, {}).get("groups", {})
            # prefer groups with enough instances; a pooled sd driven by n=2 pairs is not evidence
            sds = [v["sd"] for v in g.values() if v.get("n", 0) >= 6 and v.get("sd") == v.get("sd")]
            val = float(np.median(sds)) if sds else None
            if val is None:
                pooled = pairs.get(pair, {}).get("pooled", {}).get("sd")
                val = float(pooled) if pooled else None
            if val:
                for k in keys:
                    rep[k] = val
                src[f"repeat_sd_{pair}"] = "out/repeatability.json"
        t = pairs.get("torso_tilt", {}).get("pooled", {}).get("sd")
        if t:
            rep["torso_tilt"] = float(t)
            src["repeat_sd_torso"] = "out/repeatability.json"
    floor = {j: max(band.get(j, 0.0), rep.get(j, 0.0)) for j in JOINTS}
    return floor, src


def attribute(deltas: np.ndarray, tol: dict, floor: dict) -> dict:
    """deltas: (n_moves, n_joints) matrix. Returns offsets, scales, residuals, verdicts."""
    n, m = deltas.shape
    off = np.zeros(m)
    scale = np.zeros(m)
    baseline = []
    for c in range(m):
        col = deltas[:, c]
        col = col[~np.isnan(col)]
        j = JOINTS[c]
        if col.size >= MIN_MOVES:
            off[c] = float(np.median(col))
            mad = float(np.median(np.abs(col - off[c])))
            scale[c] = max(1.4826 * mad, floor.get(j, 0.0))
            baseline.append("estimated")
        else:
            off[c] = 0.0
            scale[c] = max(floor.get(j, 0.0), 1e-9)
            baseline.append("insufficient_moves")
    resid = deltas - off[None, :]
    z = resid / scale[None, :]
    return {"offset": off, "scale": scale, "residual": resid, "z": z, "baseline": baseline}


BAND_CONDITION = True   # added after the clean arm showed 3 of 5 candidates inside their own band
BAND_SOURCE = "out/uncertainty.json per-move band (|delta_full - delta_heavy| on identical frames)"


def classify(resid_row: np.ndarray, z_row: np.ndarray, off_vec: np.ndarray,
             band_row: np.ndarray | None, tol: dict, k: float = GATE_K) -> list[str]:
    """off_vec is PER JOINT (one value each), not per move; resid_row and z_row are per move.

    Three conditions, all required for "fault":
      1. |residual| > tolerance                -> physically significant
      2. |residual| > K * scale                -> surprising for THIS athlete
      3. |residual| > band (when band known)   -> larger than the estimators' own disagreement
    """
    out = []
    for c, j in enumerate(JOINTS):
        r = abs(float(resid_row[c]))
        o = abs(float(off_vec[c]))
        t = float(tol.get(j, 15.0))
        band = None if band_row is None else band_row[c]
        band_ok = True
        if BAND_CONDITION and band is not None and band == band:
            band_ok = r > float(band)
        if r > t and abs(float(z_row[c])) > k and band_ok:
            out.append("fault")
        elif o > t:
            out.append("style")
        else:
            out.append("noise")
    return out


def counts(verdicts: list[list[str]]) -> dict:
    flat = [v for row in verdicts for v in row]
    return {k: flat.count(k) for k in ("fault", "style", "noise")}


def matrix(card: dict) -> np.ndarray:
    return np.array([[float(mv["deltas"].get(j, np.nan)) for j in JOINTS] for mv in card["moves"]])


def band_matrix(card: dict) -> np.ndarray:
    """Per-cell estimator band; NaN where the band is unavailable (then condition 3 is skipped)."""
    rows = []
    for mv in card["moves"]:
        bd = mv.get("band_deg")
        bd = bd if isinstance(bd, dict) else {}
        rows.append([float(bd[j]) if isinstance(bd.get(j), (int, float)) else np.nan for j in JOINTS])
    return np.array(rows, dtype=float)


def old_flags(D: np.ndarray, tol: dict) -> int:
    return int(sum(np.nansum(np.abs(D[:, c]) > float(tol.get(j, 15.0))) for c, j in enumerate(JOINTS)))


def fingerprint(card: dict) -> list:
    """Everything the measurement layer produced, in order. Attribution is excluded by construction:
    it lives under a different key, so equality here proves the flags were not touched."""
    return [(mv.get("move"), f.get("joint"), f.get("delta"), f.get("tol"), f.get("msg"),
             f.get("band_deg"), f.get("release"))
            for mv in card.get("moves", []) for f in (mv.get("flags") or [])]


def annotate_card(card: dict, tol: dict, floor: dict) -> dict:
    D = matrix(card)
    B = band_matrix(card)
    a = attribute(D, tol, floor)
    ver = [classify(a["residual"][i], a["z"][i], a["offset"], B[i], tol) for i in range(D.shape[0])]
    for i, mv in enumerate(card["moves"]):
        mv["attribution"] = {
            "offset_deg": {j: round(float(a["offset"][c]), 1) for c, j in enumerate(JOINTS)
                           if a["baseline"][c] == "estimated"},
            "per_joint": {j: {"residual": round(float(a["residual"][i][c]), 1),
                              "z": round(float(a["z"][i][c]), 1),
                              "band": (None if B[i][c] != B[i][c] else round(float(B[i][c]), 1)),
                              "verdict": ver[i][c]} for c, j in enumerate(JOINTS)},
            "faults": [j for c, j in enumerate(JOINTS) if ver[i][c] == "fault"],
            "style_joints": [j for c, j in enumerate(JOINTS) if ver[i][c] == "style"],
        }
    cnt = counts(ver)
    card["self_reference"] = {
        "note": "flags stay measurement facts; this layer classifies them. offset = this athlete's "
                "systematic difference from the reference (style, not fault). A cell is a fault "
                "only when |residual| > tolerance AND |residual| > GATE_K*scale AND "
                "|residual| > that cell's estimator band.",
        "gate_k": GATE_K,
        "min_moves": MIN_MOVES,
        "band_condition": BAND_CONDITION,
        "band_source": BAND_SOURCE if BAND_CONDITION else None,
        "n_moves": int(D.shape[0]),
        "offsets_deg": {j: round(float(a["offset"][c]), 1) for c, j in enumerate(JOINTS)},
        "scale_deg": {j: round(float(a["scale"][c]), 2) for c, j in enumerate(JOINTS)},
        "floor_deg": {j: round(float(floor.get(j, 0.0)), 2) for j in JOINTS},
        "baseline": {j: a["baseline"][c] for c, j in enumerate(JOINTS)},
        "verdict_counts": cnt,
        "tolerance_deg": {j: float(tol.get(j, 15.0)) for j in JOINTS},
        "effective_threshold_deg": {j: round(max(float(tol.get(j, 15.0)),
                                                 GATE_K * float(floor.get(j, 0.0))), 1)
                                    for j in JOINTS},
        "moves_with_faults": int(sum(1 for row in ver if "fault" in row)),
        "card_flag_count": int(sum(len(mv.get("flags", [])) for mv in card["moves"])),
        "card_flag_cap_per_move": 3,
        "old_gate_flags": old_flags(D, tol),
        "old_gate_flags_note": ("raw cells with |delta| > tolerance; the card itself mints at most 3 "
                                "per move by exceedance, so this number is larger than card_flag_count")
    }
    return card


def benchmark(card: dict, tol: dict, floor: dict) -> dict:
    """Known-truth arms. Old gate = |delta| > tolerance. New gate = classified 'fault'."""
    D0 = matrix(card)
    B = band_matrix(card)
    n, m = D0.shape
    ci = {j: c for c, j in enumerate(JOINTS)}
    arms = {}

    def faults_of(D: np.ndarray) -> set[tuple[int, str]]:
        a = attribute(D, tol, floor)
        ver = [classify(a["residual"][i], a["z"][i], a["offset"], B[i], tol) for i in range(D.shape[0])]
        return {(i, JOINTS[c]) for i in range(D.shape[0]) for c in range(m) if ver[i][c] == "fault"}, ver

    def run(D: np.ndarray) -> dict:
        f, ver = faults_of(D)
        cnt = counts(ver)
        return {"old_gate_flags": old_flags(D, tol), "faults": len(f), "style": cnt["style"],
                "noise": cnt["noise"], "fault_cells": [[i + 1, j] for i, j in sorted(f)]}

    # BLIND-FIRST: the clean arm is measured and reported before any threshold constant is touched.
    arms["clean"] = run(D0.copy())
    baseline = set(map(tuple, [[i - 1, j] for i, j in arms["clean"]["fault_cells"]]))
    arms["clean"]["note"] = ("no injected fault exists in this harness: student and reference are the "
                             "same recording, so every clean-arm fault is an annotation/pose "
                             "artefact. This is the detector's floor, published rather than removed.")

    style = D0.copy()
    style[:, ci["R_elbow"]] += 20.0
    style[:, ci["L_knee"]] -= 15.0
    s = run(style)
    a = attribute(style, tol, floor)
    s["offset_recovery_deg"] = {"R_elbow": round(float(a["offset"][ci["R_elbow"]]), 1),
                               "L_knee": round(float(a["offset"][ci["L_knee"]]), 1),
                               "injected_deg": {"R_elbow": 20.0, "L_knee": -15.0}}
    s["faults_beyond_baseline"] = len(set(map(tuple, [[i - 1, j] for i, j in s["fault_cells"]])) - baseline)
    s["note"] = ("a systematic 20 deg elbow / -15 deg knee difference is the athlete's habit. The old "
                 "gate flags it on every move; this layer must produce no finding beyond the clean floor.")
    arms["style_only"] = s

    # Effective threshold = the larger of the physical tolerance and K*floor: the number a coach can
    # be told. On a zero base this is exactly what the gate will catch.
    arms["effective_threshold_deg"] = {
        j: round(max(float(tol.get(j, 15.0)), GATE_K * float(floor.get(j, 0.0))), 1) for j in JOINTS}

    def sweep(base: np.ndarray, label: str) -> dict:
        """Magnitude expressed as a multiple of each cell's effective threshold, so per-joint
        tolerances cannot masquerade as detection differences. Blind cells (band >= injection,
        i.e. the two estimators disagree by more than the fault) are counted, not hidden."""
        eff = {j: max(float(tol.get(j, 15.0)), GATE_K * float(floor.get(j, 0.0))) for j in JOINTS}
        out = {}
        for mult in (0.5, 0.9, 1.05, 1.3, 1.6):
            trials = judgeable = det_all = det_judge = blind = 0
            for i in range(n):
                for c in range(m):
                    j = JOINTS[c]
                    mag = mult * eff[j]
                    band = B[i][c]
                    is_judge = not (band == band) or mag > float(band)
                    D = base.copy()
                    D[i, c] += mag
                    f, _ = faults_of(D)
                    cell = (i, j)
                    ok = cell in f and (cell not in baseline if label == "measured" else True)
                    trials += 1
                    judgeable += 1 if is_judge else 0
                    blind += 0 if is_judge else 1
                    det_all += 1 if ok else 0
                    det_judge += 1 if (ok and is_judge) else 0
            out[f"{mult}x"] = {
                "trials": trials, "judgeable": judgeable, "band_blind": blind,
                "detection_rate": round(det_all / trials, 4),
                "detection_rate_judgeable": round(det_judge / judgeable, 4) if judgeable else None}
        out["note"] = ("zero base isolates gate arithmetic: the injected multiple IS the only signal, so "
                       "any detection below 1.0x would be a defect. band_blind cells cannot be judged at "
                       "all because the two estimators disagree by more than the fault size."
                       if label == "zero" else
                       "measured base: detection counts only findings the clean arm did not already have.")
        return out

    arms["cells_with_band"] = int(np.sum(~np.isnan(B)))
    arms["sweep_zero_base"] = sweep(np.zeros_like(D0), "zero")
    arms["sweep_measured_base"] = sweep(D0, "measured")
    return arms


def main() -> int:
    tol, tol_src = tolerances()
    floor, floor_src = floors()
    res = {"generated": "2026-11-21", "gate_k": GATE_K, "min_moves": MIN_MOVES,
           "tolerance_deg": tol, "tolerance_source": tol_src,
           "floor_deg": floor, "floor_source": floor_src, "cards": {}}

    for tag in ("sameview", "crossview"):
        path = os.path.join(OUT, f"card_{tag}.json")
        card = json.load(open(path))
        before = card.get("self_reference", {})
        fp = fingerprint(card)
        card = annotate_card(card, tol, floor)
        # the claim "flags unchanged" is checked, not asserted
        card["self_reference"]["flags_unchanged"] = (fp == fingerprint(card))
        json.dump(card, open(path, "w"), indent=1)
        res["cards"][tag] = card["self_reference"]
        res["cards"][tag]["idempotent"] = (before.get("verdict_counts") ==
                                           card["self_reference"]["verdict_counts"]) if before else None
        sr = card["self_reference"]
        print(f"[{tag}] card flags {sr['card_flag_count']} unchanged={sr['flags_unchanged']} -> "
              f"faults {sr['verdict_counts']['fault']}, "
              f"style {sr['verdict_counts']['style']}, "
              f"noise {sr['verdict_counts']['noise']}")

    same = json.load(open(os.path.join(OUT, "card_sameview.json")))
    bench = benchmark(same, tol, floor)
    res["benchmark"] = bench
    print("benchmark:", json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'fault_cells'}
                                    if isinstance(v, dict) else v)
                                    for k, v in bench.items()}, indent=1))

    json.dump(res, open(os.path.join(OUT, "self_reference.json"), "w"), indent=1)
    print("wrote out/self_reference.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
