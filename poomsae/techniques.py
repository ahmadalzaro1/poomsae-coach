"""Named-technique classification for poomsae segments.

Why this module exists
----------------------
The scoring pipeline can say "move 7: your right elbow is 25 degrees more bent
than the reference" but not "move 7 low block". A coach reads the second one.
Naming is therefore a separate, explicit layer.

Taxonomy
--------
TUHAD (Lee & Jung, Sensors 20(17):4871, 2020) defines eight taekwondo unit
techniques, A1-A8, which together cover the hand techniques of Taegeuk 1-3:

    A1 arae makki      low block
    A2 eolgul makki    high block
    A3 sonnal makki    knife-hand block
    A4 momtong jireugi middle punch
    A5 eolgul jireugi  high punch
    A6 sonnal mokchigi knife-hand neck strike
    A7 dubeon jireugi  double punch
    A8 abchagi         front kick

One technique required by Taegeuk Il Jang is NOT in TUHAD: momtong an makki
(inward middle block), labelled "AN" here and flagged as an extension rather
than silently folded into an A-class.

Honest limitation
-----------------
The dual-view dataset annotates 12 joints: shoulders, elbows, wrists, hips,
knees, ankles. There are no hand landmarks. Techniques that differ mainly in
hand shape (sonnal makki vs eolgul makki) or in strike target on the same
height (sonnal mokchigi vs an makki) are therefore only weakly separable by
geometry. This module classifies them anyway, exposes its evidence and its
margin, and the validation run reports how often it is actually right rather
than assuming the rule set works.

Coordinates
-----------
Pixels, y increasing downward (image convention). All height comparisons are
divided by torso length (shoulder-midpoint to hip-midpoint) so they are
independent of camera distance and of the athlete's size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import J, Sequence, angle_at

# label -> (korean, english, in_tuhad)
NAMES = {
    "A1": ("arae makki", "low block", True),
    "A2": ("eolgul makki", "high block", True),
    "A3": ("sonnal makki", "knife-hand block", True),
    "A4": ("momtong jireugi", "middle punch", True),
    "A5": ("eolgul jireugi", "high punch", True),
    "A6": ("sonnal mokchigi", "knife-hand neck strike", True),
    "A7": ("dubeon jireugi", "double punch", True),
    "A8": ("abchagi", "front kick", True),
    "AN": ("momtong an makki", "inward middle block", False),
    "JUMBI": ("joonbi seogi", "ready stance", False),
}


def label_name(lab: str) -> str:
    k, e, _ = NAMES.get(lab, (lab, lab, False))
    return f"{k} ({e})" if k != e else e


@dataclass
class Cluster:
    """Descriptor of the body at one instant, in torso-normalised units."""
    torso: float          # shoulder-mid -> hip-mid distance, px
    # arm: elbow angle (deg), wrist height vs shoulder and hip (torso units,
    # positive = HIGHER than the reference joint), reach from shoulder,
    # lateral offset from body midline (torso units, signed: + = athlete's left)
    elbow: dict
    wrist_above_shoulder: dict
    wrist_above_hip: dict
    reach: dict
    lateral: dict
    knee: dict
    # legs
    foot_above_knee: dict   # positive = foot raised above the knee (kick)
    stance_width: float


def _clean(seq: Sequence, min_score: float = 0.5) -> np.ndarray:
    px = seq.px.copy()
    bad = (seq.scores < min_score) | np.isnan(px).any(-1)
    for j in range(12):
        ok = ~bad[:, j]
        if ok.sum() >= 2:
            for d in range(2):
                px[:, j, d] = np.interp(np.arange(len(px)), np.flatnonzero(ok),
                                        px[ok, j, d])
    return px


def describe(seq: Sequence, lo: int, hi: int) -> Cluster:
    """Median descriptor over frames [lo, hi] (a move's hold/apex window)."""
    px = _clean(seq)[lo:hi + 1]
    hip_mid = (px[:, J["L_hip"]] + px[:, J["R_hip"]]) / 2
    sho_mid = (px[:, J["L_shoulder"]] + px[:, J["R_shoulder"]]) / 2
    torso = np.median(np.linalg.norm(sho_mid - hip_mid, axis=1))
    torso = float(max(torso, 1e-3))
    mid_x = np.median(hip_mid[:, 0])

    elbow, was, wah, reach, lateral = {}, {}, {}, {}, {}
    for side in ("L", "R"):
        e = angle_at(px[:, J[f"{side}_shoulder"]], px[:, J[f"{side}_elbow"]],
                     px[:, J[f"{side}_wrist"]])
        elbow[side] = float(np.median(e))
        w = px[:, J[f"{side}_wrist"]]
        s = px[:, J[f"{side}_shoulder"]]
        h = hip_mid
        was[side] = float(np.median((s[:, 1] - w[:, 1]) / torso))
        wah[side] = float(np.median((h[:, 1] - w[:, 1]) / torso))
        reach[side] = float(np.median(np.linalg.norm(w - s, axis=1)) / torso)
        lateral[side] = float(np.median(w[:, 0] - mid_x) / torso)

    knee, foot = {}, {}
    for side in ("L", "R"):
        k = angle_at(px[:, J[f"{side}_hip"]], px[:, J[f"{side}_knee"]],
                     px[:, J[f"{side}_ankle"]])
        knee[side] = float(np.median(k))
        foot[side] = float(np.median((px[:, J[f"{side}_knee"], 1] -
                                      px[:, J[f"{side}_ankle"], 1]) / torso))
    sw = float(np.median(np.abs(px[:, J["L_ankle"], 0] -
                                px[:, J["R_ankle"], 0])) / torso)
    return Cluster(torso, elbow, was, wah, reach, lateral, knee, foot, sw)


# --------------------------------------------------------------- classifier
# Decision thresholds. Each is a stated, checkable claim about taekwondo
# geometry, in torso-normalised units or degrees.
TH = {
    "kick_peak": 0.25,              # foot lifted 25% of torso above its knee
    "arm_straight_deg": 150.0,      # elbow extended => strike, not block
    # Height bands, in torso units of wrist height ABOVE hip level. Measured on
    # the Taegeuk 1 annotations rather than assumed: low blocks hold at
    # +0.19/-0.02/+0.23, middle punches at +0.74..+0.93, high blocks at +1.82.
    "low_max": 0.45,                # below this = low section (arae)
    "high_min": 1.15,               # above this = high section (eolgul)
    "knife_elbow": (95.0, 145.0),   # knife-hand proxy: bent, but not tucked
    "rest_elbow_deg": 150.0,        # both arms bent => not a technique
    "rest_band": 0.15,              # both wrists within this of hip level
}


def kick_peak(seq: Sequence, lo: int, hi: int) -> float:
    """Largest foot-above-knee height over a window, in torso units.

    A front kick is airborne for ~7-10 frames (measured on the Taegeuk 1
    annotations: peak +0.95 torso, baseline p95 -0.36). Taking the median over a
    segment erases it, so the kick test must use the peak, not the typical pose.
    """
    px = _clean(seq)[lo:hi + 1]
    hip_mid = (px[:, J["L_hip"]] + px[:, J["R_hip"]]) / 2
    sho_mid = (px[:, J["L_shoulder"]] + px[:, J["R_shoulder"]]) / 2
    torso = float(np.median(np.linalg.norm(sho_mid - hip_mid, axis=1)))
    torso = max(torso, 1e-3)
    best = -9.9
    for s in "LR":
        foot = (px[:, J[f"{s}_knee"], 1] - px[:, J[f"{s}_ankle"], 1]) / torso
        best = max(best, float(foot.max()))
    return best


def classify(c: Cluster, peak: float = -9.9) -> dict:
    """Return {label, name, confidence, margin, evidence}.

    Taekwondo logic, not arbitrary thresholds: the TARGET HEIGHT decides the
    technique family -- low section (arae), middle section (momtong), high
    section (eolgul) -- and the ELBOW decides block versus strike. The height
    bands were measured on the published annotations, not guessed:

        low blocks hold at   0.19 / -0.02 / 0.23 torso above hip level
        middle punches hold at 0.74 .. 0.93
        high blocks hold at   1.82

    so the section boundaries are 0.45 and 1.15 torso units.

    Known-weak branches, stated rather than hidden: sonnal makki vs eolgul
    makki differ mainly in hand shape (no hand landmarks here), and sonnal
    mokchigi vs an makki share the same height and arm bend. Both come back
    with low confidence.
    """
    ev = {"kick_peak": round(peak, 2),
          "elbow": {s: round(c.elbow[s], 0) for s in "LR"},
          "wrist_above_hip": {s: round(c.wrist_above_hip[s], 2) for s in "LR"},
          "reach": {s: round(c.reach[s], 2) for s in "LR"},
          "lateral": {s: round(c.lateral[s], 2) for s in "LR"}}

    # 1. kick — unambiguous when it is airborne at all
    if peak > TH["kick_peak"]:
        return _out("A8", "high", 1.0, ev, f"foot raised {peak:+.2f} torso above the knee")

    straight = {s: c.elbow[s] > TH["arm_straight_deg"] for s in "LR"}
    hip = {s: c.wrist_above_hip[s] for s in "LR"}

    # 2. ready stance / non-technique: hands folded at the chest (the recovery
    # posture that ends a form) or both hands at hip level with both arms bent.
    if all(c.elbow[s] < 60 for s in "LR"):
        return _out("JUMBI", "medium", 0.6, ev,
                    "both elbows folded at the chest - recovery posture, no technique held")
    if (all(abs(hip[s]) < TH["rest_band"] for s in "LR")
            and all(c.elbow[s] < 120 for s in "LR")):
        return _out("JUMBI", "medium", 0.7, ev,
                    "both hands at hip level, both elbows bent - no technique held")

    # 3. double punch: both arms straight, both reaching forward at chest height
    if (all(straight.values()) and all(c.reach[s] >= 0.60 for s in "LR")
            and all(hip[k] >= TH["low_max"] for k in "LR")
            and abs(hip["L"] - hip["R"]) < 0.4):
        return _out("A7", "medium", 0.8, ev, "both elbows extended and reaching at chest height")

    # Acting arm. A strike is a straight arm, so a single straight arm IS the
    # acting arm -- reach is not usable for this: from a frontal camera a punch
    # aimed at the lens foreshortens to reach 0.44 while the bent counter-arm
    # sitting at the waist measures farther. When no arm is straight (blocks),
    # take the raised arm, because the counter-arm rests low.
    if sum(straight.values()) == 1:
        s = "L" if straight["L"] else "R"
    else:
        s = max("LR", key=lambda k: hip[k])
    ev["active_arm"] = s
    h = hip[s]

    if h >= TH["high_min"]:
        if straight[s]:
            return _out("A5", "high", 1.0, ev,
                        f"{s} wrist {h:+.2f} torso above hip, elbow straight")
        lo_e, hi_e = TH["knife_elbow"]
        if lo_e <= c.elbow[s] <= hi_e:
            return _out("A3", "low", 0.35, ev,
                        f"{s} wrist high with bent elbow ({c.elbow[s]:.0f} deg) "
                        f"- knife-hand proxy, no hand landmarks")
        return _out("A2", "high", 0.9, ev,
                    f"{s} wrist {h:+.2f} torso above hip (high section), elbow bent")

    if h >= TH["low_max"]:
        if straight[s]:
            return _out("A4", "high", 1.0, ev,
                        f"{s} wrist {h:+.2f} torso above hip at chest, elbow straight "
                        f"({c.elbow[s]:.0f} deg)")
        if c.lateral[s] < -0.55:
            return _out("A6", "low", 0.3, ev,
                        f"{s} bent arm crossing the midline at chest height "
                        f"- ambiguous with an makki")
        return _out("AN", "medium", 0.6, ev,
                    f"{s} bent arm at chest height ({c.elbow[s]:.0f} deg), "
                    f"{c.lateral[s]:+.2f} torso lateral")

    return _out("A1", "high", 1.0, ev, f"{s} wrist {h:+.2f} torso above hip (low section)")


def _out(label, conf_word, margin, ev, why) -> dict:
    k, e, in_tuhad = NAMES[label]
    return {"label": label, "korean": k, "english": e, "in_tuhad": in_tuhad,
            "confidence": conf_word, "margin": round(float(margin), 2),
            "evidence": ev, "why": why, "name": f"{k} ({e})"}


# ------------------------------------------------------- canonical reference
# Taegeuk Il Jang, 18 counts, from two independent published breakdowns
# (British Taekwondo poomsae rules via taekwondo-ai.com; taekwondonation.com).
# Counts 14 and 16 combine a front kick with a middle punch under one count, so
# the technique list is 20 entries for 18 counts.
TAEGEUK1_COUNTS = [
    (1, "A1", "left low block"), (2, "A4", "right middle punch"),
    (3, "A1", "right low block"), (4, "A4", "left middle punch"),
    (5, "A1", "left low block"), (6, "A4", "right reverse punch"),
    (7, "AN", "left inward middle block"), (8, "A4", "right reverse punch"),
    (9, "AN", "right inward middle block"), (10, "A4", "left reverse punch"),
    (11, "A1", "right low block"), (12, "A4", "left reverse punch"),
    (13, "A2", "left high block"), (14, "A8", "right front kick"),
    (14, "A4", "right middle punch"), (15, "A2", "right high block"),
    (16, "A8", "left front kick"), (16, "A4", "left middle punch"),
    (17, "A1", "left low block"), (18, "A4", "right middle punch + kihap"),
]

# Taegeuk 2 Jang, 18 counts (taekwondo-ai.com count table cross-checked against
# British Taekwondo poomsae rules and the Albanian federation chart: 5 kicks,
# 7 punches, 4 inward blocks, 2 high blocks). Counts 8, 10, 16, 17 and 18 each
# hold a kick + landing + punch, so 23 techniques sit inside 18 counts.
# This form is the HELD-OUT test: the classifier thresholds were tuned on
# Taegeuk 1 and are not adjusted here.
TAEGEUK2_COUNTS = [
    (1, "A1", "left low block"), (2, "A4", "right middle punch"),
    (3, "A1", "right low block"), (4, "A4", "left middle punch"),
    (5, "AN", "right inward middle block"), (6, "AN", "left inward middle block"),
    (7, "A1", "left low block"),
    (8, "A8", "right front kick"), (8, "A5", "right high punch"),
    (9, "A1", "right low block"),
    (10, "A8", "left front kick"), (10, "A5", "left high punch"),
    (11, "A2", "left high block"), (12, "A2", "right high block"),
    (13, "AN", "right inward middle block"), (14, "AN", "left inward middle block"),
    (15, "A1", "left low block"),
    (16, "A8", "right front kick"), (16, "A4", "right middle punch"),
    (17, "A8", "left front kick"), (17, "A4", "left middle punch"),
    (18, "A8", "right front kick"), (18, "A4", "right middle punch + kihap"),
]

FORM_COUNTS = {
    "Taegeuk_1st": TAEGEUK1_COUNTS,   # calibration form
    "Taegeuk_2nd": TAEGEUK2_COUNTS,   # held out: thresholds frozen
}
CALIBRATION_FORM = "Taegeuk_1st"
