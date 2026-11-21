"""Jev as a release gate for coach feedback — measured on the real card findings.

The product question: a coach card can contain a finding that is really camera geometry or
measurement instability rather than athlete error. Today that is handled by a deterministic
10-degree agreement gate (out/elbow_gate.json) which costs a second pose model per submission.
Jev is a typed decision layer that answers in ~220 ms for $42/M tokens, so the question is
whether it can arbitrate *release* instead of a second sensor — and whether its judgement
matches ground truth we already know by construction.

Two experiments, both on measured data, both using ground truth we control:

  A. Card findings (out/card_*.json). Ground truth by construction: a finding produced by
     comparing the student against an OPPOSITE-side reference is camera geometry, not athlete
     error (16/16 moves deviate, mean 12.5 deg). A SAME-side finding is attributable to the
     athlete (6/19 moves). Question: does Jev's release decision suppress opposite-side
     findings? Reported as AUC of P(withhold) on the cross-side label, plus the policy
     confusion at a fixed threshold.

  B. Elbow moves (out/elbow_gate.json, 598 moves with two independent readings). Ground truth:
     the measured disagreement between the two readings. Jev is given BOTH readings and asked
     whether the move's reading is trustworthy. Measures whether a language-shaped decision
     layer tracks a numeric measurement instability signal.

Neither experiment feeds Jev a secret, and neither lets Jev authorise anything: it is asked for
advice on phrasing, the number itself stays with the scorer. Runs are checkpointed to
out/jev_gate_raw.jsonl so a rerun can be audited without paying twice.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poomsae.core import TOLERANCE_DEG  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
URL = "https://api.typesafe.ai/v1/systemone"
RAW = os.path.join(OUT, "jev_gate_raw.jsonl")
KEY = subprocess.run(["security", "find-generic-password", "-a", "syntaxstudio",
                      "-s", "ai.typesafe.api", "-w"],
                     capture_output=True, text=True).stdout.strip()


def ask(payload, timeout=30):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=data, method="POST",
                                headers={"Authorization": f"Bearer {KEY}",
                                         "Content-Type": "application/json"})
    t = time.time()
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read())
            return body, int((time.time() - t) * 1000), None
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return None, int((time.time() - t) * 1000), repr(exc)[:160]
            time.sleep(0.6)


def validate(ans, names, kinds):
    """Defensive read: exact keys, declared types, finite bounded probabilities."""
    if not isinstance(ans, dict):
        return None
    out = {}
    for n in names:
        a = ans.get(n)
        if not isinstance(a, dict) or a.get("type") != kinds[n]:
            return None
        if kinds[n] == "noul":
            p = a.get("noul")
            if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
                return None
            out[n] = float(p)
        elif kinds[n] == "choice":
            probs = a.get("probabilities")
            if not isinstance(probs, dict) or not probs:
                return None
            if any(not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0)
                   for v in probs.values()):
                return None
            out[n] = {"choice": a.get("choice"), "probs": {k: float(v) for k, v in probs.items()},
                      "confidence": float(a.get("confidence", 0.0))}
        else:
            return None
    return out


def auc(scores, labels):
    """Mann-Whitney AUC with tie handling."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += (p > neg).sum() + 0.5 * (p == neg).sum()
    return float(wins / (len(pos) * len(neg)))


def experiment_a(pool, blind=False):
    cards = {}
    for k in ("sameview", "crossview"):
        p = os.path.join(OUT, f"card_{k}.json")
        if os.path.exists(p):
            with open(p) as fh:
                cards[k] = json.load(fh)
    findings = []
    for k, c in cards.items():
        n = c.get("n_moves") or 0
        for mv in c.get("moves", []):
            for f in (mv.get("flags") or []):
                findings.append({
                    "clip": k, "move": mv.get("move"), "n_moves": n,
                    "joint": f.get("joint"), "delta": float(f.get("delta") or 0.0),
                    "text": f.get("msg") or "",
                    "cross_side": bool(c.get("cross_side")),
                    "reference": c.get("comparison") or "",
                })
    print(f"A{' (blind, no side label in state)' if blind else ''}: {len(findings)} real findings "
          f"({sum(1 for f in findings if f['cross_side'])} opposite-side / "
          f"{sum(1 for f in findings if not f['cross_side'])} same-side)")

    def run(f):
        tol = float(TOLERANCE_DEG.get(
            {"torso": "L_torso", "knee": "L_knee", "hip": "L_hip",
             "shoulder": "L_shoulder", "elbow": "L_elbow"}.get(f["joint"], "L_elbow"), 15.0))
        payload = {"model": "jev-latest",
                   "state": ({"task": "taekwondo poomsae coach card",
                              "form": "Taegeuk 1", "moves_in_form": f["n_moves"],
                              "finding": {"move_index": f["move"], "joint": f["joint"],
                                          "deviation_deg": round(f["delta"], 1),
                                          "tolerance_deg": tol}}
                             if blind else
                             {"task": "taekwondo poomsae coach card",
                              "reference_side_relative_to_athlete":
                                  "opposite" if f["cross_side"] else "same",
                              "reference_note": f["reference"],
                              "camera": "single camera, 1280x720, 30 fps",
                              "form": "Taegeuk 1", "moves_in_form": f["n_moves"],
                              "finding": {"move_index": f["move"], "joint": f["joint"],
                                          "deviation_deg": round(f["delta"], 1),
                                          "tolerance_deg": tol},
                              "known_limitation":
                                  "When the reference footwork is filmed from the opposite side of the "
                                  "body, a large joint deviation is produced by camera geometry rather "
                                  "than by the athlete."}),
                   "questions": {
                       "release": {"type": "choice",
                                   "instructions": "Should this correction be shown to the athlete as an "
                                                   "athlete error?",
                                   "criteria": {
                                       "show": "The deviation is attributable to the athlete and is large "
                                               "enough to state plainly.",
                                       "soften": "Worth mentioning, but must not be attributed to the "
                                                 "athlete as a fault.",
                                       "withhold": "Must not be shown: camera geometry or measurement "
                                                   "limits explain it."}},
                       "attributable": {"type": "noul",
                                        "instructions": "The supplied evidence explicitly supports "
                                                        "attributing this deviation to the athlete.",
                                        "criteria": {"true": "The deviation is caused by the athlete's "
                                                             "technique.",
                                                     "false": "Camera side or measurement limits explain "
                                                              "the deviation."}}}}
        body, ms, err = ask(payload)
        with open(RAW, "a") as fh:
            fh.write(json.dumps({"exp": "A", "id": f"{f['clip']}:{f['move']}:{f['joint']}",
                                 "model": (body or {}).get("model"),
                                 "usage": (body or {}).get("usage"),
                                 "answers": (body or {}).get("answers"), "ms": ms,
                                 "err": err}) + "\n")
        v = validate((body or {}).get("answers", {}),
                     ["release", "attributable"], {"release": "choice", "attributable": "noul"})
        return {**f, "ms": ms, "err": err, "v": v,
                "p_withhold": None if not v else v["release"]["probs"].get("withhold", 0.0),
                "p_show": None if not v else v["release"]["probs"].get("show", 0.0),
                "p_attr": None if not v else v["attributable"],
                "choice": None if not v else v["release"]["choice"]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(run, findings))

    ok = [r for r in rows if r["v"]]
    print(f"   valid {len(ok)}/{len(rows)} | errors {sum(1 for r in rows if r['err'])}")
    if not ok:
        return {"n": 0}
    y = [1 if r["cross_side"] else 0 for r in ok]
    a_with = auc([r["p_withhold"] for r in ok], y)
    a_attr = auc([1.0 - r["p_attr"] for r in ok], y)
    print(f"   AUC P(withhold) vs opposite-side label : {a_with:.3f}")
    print(f"   AUC (1-P(attributable)) same label     : {a_attr:.3f}")
    # policy: withhold or soften anything not attributable
    pol = [0 if r["choice"] == "withhold" else 1 for r in ok]
    tp = sum(1 for r, p in zip(ok, pol) if r["cross_side"] and p == 0)
    fn = sum(1 for r, p in zip(ok, pol) if r["cross_side"] and p == 1)
    tn = sum(1 for r, p in zip(ok, pol) if not r["cross_side"] and p == 1)
    fp = sum(1 for r, p in zip(ok, pol) if not r["cross_side"] and p == 0)
    print(f"   policy 'withhold wins': opposite-side suppressed {tp}/{tp+fn}, "
          f"same-side wrongly suppressed {fp}/{fp+tn}")
    lat = np.array([r["ms"] for r in ok], float)
    print(f"   latency median {np.median(lat):.0f} ms, p95 {np.percentile(lat, 95):.0f} ms")
    return {"n": len(ok), "n_opposite": sum(y), "n_same": len(y) - sum(y),
            "auc_p_withhold_vs_opposite_side": round(a_with, 4),
            "auc_1m_attributable_vs_opposite_side": round(a_attr, 4),
            "policy_withhold": {"opposite_suppressed": tp, "opposite_shown": fn,
                                "same_kept": tn, "same_suppressed": fp},
            "latency_ms": {"median": round(float(np.median(lat)), 1),
                           "p95": round(float(np.percentile(lat, 95)), 1)},
            "by_release_choice": {c: sum(1 for r in ok if r["choice"] == c)
                                  for c in ("show", "soften", "withhold")},
            "rows": [{k: r[k] for k in ("clip", "move", "joint", "delta", "cross_side",
                                        "choice", "p_show", "p_withhold", "p_attr")}
                     for r in ok]}


def experiment_b(pool, n=60, seed=11, pair=True):
    p = os.path.join(OUT, "elbow_gate.json")
    if not os.path.exists(p):
        return {"n": 0}
    with open(p) as fh:
        g = json.load(fh)
    rows = []
    for clip, c in g["per_move"].items():
        for mv in c["per_move"]:
            rows.append({"clip": clip, "form": c["form"], "view": c["view"],
                         "move": mv["move"], "side": mv["side"],
                         "gap": mv["disagree_deg"], "jitter": mv["jitter_deg"],
                         "full": mv["full_deg"], "heavy": mv["heavy_deg"]})
    rng = random.Random(seed)
    # stratified by measured disagreement so the sample spans the tail
    strata = [(0, 3), (3, 8), (8, 15), (15, 25), (25, 1e9)]
    sample = []
    for lo, hi in strata:
        s = [r for r in rows if lo <= r["gap"] < hi]
        sample += rng.sample(s, min(n // len(strata), len(s)))
    print(f"B{' (single reading, no second model)' if not pair else ''}: {len(sample)} elbow moves, gap range {min(r['gap'] for r in sample):.1f}-"
          f"{max(r['gap'] for r in sample):.1f} deg")

    def run(r):
        payload = {"model": "jev-latest",
                   "state": {"task": "taekwondo elbow joint measurement",
                             "joint": "elbow extension (shoulder-elbow-wrist angle)",
                             "camera": "single 1280x720 camera, monocular 2D pose estimation",
                             "tolerance_deg": float(TOLERANCE_DEG["L_elbow"]),
                             **({"two_independent_readings_deg": {"model_A": r["full"],
                                                                  "model_B": r["heavy"]}}
                                if pair else {"reading_deg": r["full"]}),
                             "within_hold_jitter_deg": r["jitter"],
                             "known_limitation":
                                 "Elbow angle depends on a wrist landmark that is small and often "
                                 "occluded; two pose models routinely read the same frames differently."},
                   "questions": {
                       "trust": {"type": "choice",
                                 "instructions": "Is this elbow reading precise enough to coach from?",
                                 "criteria": {"trust": "Both readings agree closely enough that the "
                                                       "angle is usable.",
                                              "qualify": "Usable only as a direction, not a magnitude.",
                                              "reject": "The disagreement makes the reading unusable."}},
                       "stable": {"type": "noul",
                                  "instructions": "The two readings are close enough to support a "
                                                  "degree-level correction."}}}
        body, ms, err = ask(payload)
        with open(RAW, "a") as fh:
            fh.write(json.dumps({"exp": "B", "id": f"{r['clip']}:{r['move']}:{r['side']}",
                                 "model": (body or {}).get("model"),
                                 "usage": (body or {}).get("usage"),
                                 "answers": (body or {}).get("answers"), "ms": ms,
                                 "err": err}) + "\n")
        v = validate((body or {}).get("answers", {}),
                     ["trust", "stable"], {"trust": "choice", "stable": "noul"})
        return {**r, "v": v, "ms": ms,
                "p_reject": None if not v else v["trust"]["probs"].get("reject", 0.0),
                "p_trust": None if not v else v["trust"]["probs"].get("trust", 0.0),
                "p_stable": None if not v else v["stable"]}

    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(run, sample))
    ok = [r for r in res if r["v"]]
    print(f"   valid {len(ok)}/{len(res)}")
    if not ok:
        return {"n": 0}
    gap = np.array([r["gap"] for r in ok], float)
    rej = np.array([r["p_reject"] for r in ok], float)
    stb = np.array([r["p_stable"] for r in ok], float)
    lab = (gap > 10.0).astype(int)
    print(f"   AUC P(reject) vs gap>10 deg  : {auc(rej, lab):.3f}  "
          f"(gate at 10 deg emits {1 - lab.mean():.1%})")
    print(f"   AUC (1-P(stable)) same label : {auc(1 - stb, lab):.3f}")
    rho = float(np.corrcoef(np.argsort(np.argsort(rej)), np.argsort(np.argsort(gap)))[0, 1])
    print(f"   Spearman(P(reject), gap)     : {rho:.3f}")
    return {"n": len(ok), "gap_range_deg": [round(float(gap.min()), 1), round(float(gap.max()), 1)],
            "auc_p_reject_vs_gap10": round(auc(rej, lab), 4),
            "auc_1m_stable_vs_gap10": round(auc(1 - stb, lab), 4),
            "spearman": round(rho, 4),
            "share_flagged_unusable": round(float((rej > 0.5).mean()), 4),
            "latency_ms_median": round(float(np.median([r["ms"] for r in ok])), 1),
            "rows": [{k: r[k] for k in ("clip", "move", "side", "gap", "jitter",
                                        "p_reject", "p_stable")} for r in ok]}


def main() -> int:
    if not KEY:
        print("no key in keychain")
        return 1
    open(RAW, "w").close()
    t0 = time.time()
    a = experiment_a(None, blind=False)
    a_blind = experiment_a(None, blind=True)
    b = experiment_b(None, pair=True)
    b_single = experiment_b(None, pair=False)
    wall = round(time.time() - t0, 1)

    usage = {"input": 0, "output": 0, "requests": 0}
    for line in open(RAW):
        rec = json.loads(line)
        u = rec.get("usage") or {}
        usage["input"] += int(u.get("input_tokens") or 0)
        usage["output"] += int(u.get("output_tokens") or 0)
        usage["requests"] += 1
    cost = usage["input"] / 1e6 * 0.042
    print(f"\nusage: {usage['requests']} requests, {usage['input']} input tokens, "
          f"{usage['output']} output tokens -> ${cost:.5f} at $42/M | wall {wall}s | "
          f"model {json.loads(open(RAW).readline()).get('model')}")

    payload = {"generated": "2026-11-21", "endpoint": URL, "model": "jev-latest",
               "experiment_a_card_findings": a, "experiment_a_blind": a_blind,
               "experiment_b_elbow_moves": b, "experiment_b_single_reading": b_single,
               "usage": {**usage, "cost_usd": round(cost, 6)}, "wall_s": wall,
               "note": ("Jev is asked for advice on release, never authorisation: the number and "
                        "the tolerance stay with the scorer."),
               "leak_disclosure": ("The first arm of A named the answer in state (the side label plus a "
                                   "sentence explaining that opposite-side filming produces large "
                                   "deviations) and the first arm of B supplied both readings before "
                                   "asking whether they agree, so those AUCs measure faithfulness to "
                                   "supplied state and arithmetic, not insight. The blind and "
                                   "single-reading arms are the honest arms.")}
    with open(os.path.join(OUT, "jev_gate.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("wrote out/jev_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
