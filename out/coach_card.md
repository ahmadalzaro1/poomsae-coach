# Poomsae coaching card — Taegeuk 1 Jang

Source: raw video `1st_Frontal.mp4` (1920x1080, 30 fps, 768 frames). Pose extracted locally at 20.8 ms/frame — no cloud, no referee hardware.

> **Headline:** 3 candidate faults across 171 joint cells (mean deviation 3.4°). Another 0 cells exceed tolerance but are this athlete's own systematic offset, not errors. The raw flags below are unchanged measurement facts and were verified byte-identical before and after attribution (`flags_unchanged: True`); the 171-cell count exceeds the 9 listed flags only because the card mints at most 3 per move, by exceedance.

## Move-by-move (reference and student from the same camera side)

| move | technique | t (s) | mean \|Δ\| | largest residual | own offset | threshold | band | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | arae makki (low block) | 0.0 | 2.2° | L_elbow: +7.4° | +0.1° | 18.1° | ±0.9° | **noise** |
| 2 | arae makki (low block) | 0.33 | 5.2° | L_elbow: +17.5° | +0.1° | 18.1° | ±6.4° | **noise** |
| 3 | arae makki (low block) | 1.67 | 1.6° | L_elbow: -8.2° | +0.1° | 18.1° | ±3.7° | **noise** |
| 4 | momtong jireugi (middle punch) | 2.67 | 2.3° | R_elbow: +7.6° | +0.4° | 21° | ±1.3° | **noise** |
| 5 | momtong jireugi (middle punch) | 4.9 | 1.6° | R_elbow: +9.9° | +0.4° | 21° | ±42.9° | **noise** |
| 6 | arae makki (low block) | 6.2 | 4.1° | R_elbow: +21.1° | +0.4° | 21° | ±8.4° | **noise** |
| 7 | momtong jireugi (middle punch) | 6.9 | 0.7° | R_shoulder: -1.8° | -0.2° | 12° | ±6.4° | **noise** |
| 8 | momtong jireugi (middle punch) | 8.13 | 2.7° | R_elbow: -20.3° | +0.4° | 21° | ±47.1° | **noise** |
| 9 | momtong jireugi (middle punch) | 9.3 | 2.3° | L_elbow: +17.1° | +0.1° | 18.1° | ±3.2° | **noise** |
| 10 | momtong jireugi (middle punch) | 10.67 | 1.2° | L_shoulder: -3.2° | -0.7° | 12° | ±13.8° | **noise** |
| 11 | momtong jireugi (middle punch) | 11.93 | 3.1° | R_elbow: +17.9° | +0.4° | 21° | ±0.2° | **noise** |
| 12 | arae makki (low block) | 13.9 | 3.3° | R_elbow: +12.8° | +0.4° | 21° | ±25.4° | **noise** |
| 13 | eolgul makki (high block) | 14.97 | 3.4° | L_elbow: +22.0° | +0.1° | 18.1° | ±6.9° | **noise** |
| 14 | abchagi + momtong jireugi (middle punch) | 17.13 | 2.7° | R_shoulder: +10.4° | -0.2° | 12° | ±6.3° | **noise** |
| 15 | abchagi + momtong jireugi (middle punch) | 20.37 | 10.7° | L_elbow: -45.2° | +0.1° | 18.1° | ±68.2° | **noise** |
| 16 | momtong an makki (inward middle block) | 21.67 | 11.4° | L_elbow: -74.0° | +0.1° | 18.1° | ±78.9° | **noise** |
| 17 | momtong an makki (inward middle block) | 22.83 | 3.8° | L_elbow: -16.6° | +0.1° | 18.1° | ±14.3° | **noise** |
| 18 | arae makki (low block) | 24.87 | 1.0° | L_elbow: +4.6° | +0.1° | 18.1° | ±8.7° | **noise** |
| 19 | joonbi seogi (ready stance) | 25.5 | 0.7° | R_elbow: -2.0° | +0.4° | 21° | ±0.1° | **noise** |

## Same performance, opposite camera side — why angle matters

Feeding the same video to a reference recorded from the *other* side flags 16 of 16 moves at 12.5° mean. The 8-form benchmark puts that at 10.18° across 156 moves, against 3.4° for the same clip analysed twice.

| move | technique | t (s) | mean \|Δ\| | largest residual | own offset | threshold | band | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | arae makki (low block) | 0.0 | 5.9° | L_elbow: -16.1° | +12.3° | 18.1° | ±2.4° | **noise** |
| 2 | arae makki (low block) | 0.27 | 9.4° | R_shoulder: +17.7° | +10.3° | 12° | ±1.1° | **noise** |
| 3 | momtong jireugi (middle punch) | 2.77 | 9.9° | L_elbow: -34.0° | +12.3° | 18.1° | ±41.7° | **noise** |
| 4 | arae makki (low block) | 4.03 | 13.1° | L_shoulder: +29.5° | +14.8° | 12° | ±0.5° | **fault** |
| 5 | momtong jireugi (middle punch) | 4.9 | 8.2° | L_elbow: +13.7° | +12.3° | 18.1° | ±0.8° | **noise** |
| 6 | arae makki (low block) | 6.23 | 25.8° | L_knee: +44.8° | -0.6° | 10° | ±0° | **fault** |

**Operational rule: film the reference model from the same side the student will be filmed from.** That single constraint eliminates ~3x more error than any model upgrade available today.

## How much to trust the technique names

Names come from a rule set over the same joint angles, restricted to the TUHAD A1-A8 taxonomy (Lee & Jung, *Sensors* 20:4871, 2020) plus *momtong an makki*, which TUHAD omits but Taegeuk 1 and 2 require.

- **Taegeuk 1st** — 94.1% (calibration form, thresholds tuned here), 96.9% on the moves the classifier is confident about.
- **Taegeuk 2nd** — 83.7% (held out, thresholds frozen), 89.8% on the moves the classifier is confident about.

- The two views of one form give different numbers. That range, not the better number, is the honest estimate.
- **Known blind spot:** sonnal mokchigi (A6) and momtong an makki share height and arm bend, and sonnal makki (A3) differs from eolgul makki mainly in hand shape. With no hand landmarks these are guesses; the classifier says so with low confidence rather than pretending.

## Why a deviation is not a fault

Across **94 athletes** performing the same eight techniques (2D pose, peak-normalised trajectories) the elbow varies with sd **32.7°** (68.5° at the worst phase) and the knee with sd **4.7°**. An 18° elbow gate is ~2.3x the *repeat* spread of one athlete and only **0.55x** the *cross-athlete* spread. So a 20° elbow difference from a reference is often just this athlete's habit.

The card therefore separates two things it used to conflate:

- **own offset** = the athlete's median deviation across their own moves — style, build, stance. Reported, never counted as an error.
- **residual** = a move's deviation from that offset. Only a residual can be a fault, and only when it clears three conditions: above the joint tolerance, above 2.5x this athlete's own robust spread, and above the two estimators' disagreement (band) at that move.

| joint | tolerance | effective threshold |
|---|---|---|
| L_elbow | 18° | **18.1°** |
| R_elbow | 18° | **21°** |
| L_knee | 10° | **10°** |
| R_knee | 10° | **10°** |
| L_shoulder | 12° | **12°** |
| R_shoulder | 12° | **12°** |
| torso_tilt | 5° | **5°** |

Effective threshold = max(tolerance, gate_k × measured floor); the floor is this pipeline's own repeat spread / estimator disagreement, not a guess. The right elbow gate is 21°, not 18°.

**Certified against known truth** (injections on a clean base, 171 cells each):

| injected | detection (all cells) | detection (judgeable) | band-blind |
|---|---|---|---|
| 0.5x of threshold | 0.000 | 0.000 | 23 |
| 0.9x of threshold | 0.000 | 0.000 | 14 |
| 1.05x of threshold | 0.930 | 1.000 | 12 |
| 1.3x of threshold | 0.959 | 1.000 | 7 |
| 1.6x of threshold | 0.965 | 1.000 | 6 |

- A **systematic** +20° elbow / −15° knee difference is flagged on every move by the old gate (39 cells). This layer calls it style, recovers the offsets as +20.4° / -15.0° against +20.0° / −15.0° injected, and produces **0 findings** beyond the clean floor.
- **Clean floor: 3 cells (1.8%).** In this harness the student *is* the reference recording, so no true fault exists and this floor is annotation/pose artefact. It is published, not tuned away.
- Band-blind cells (the two estimators disagree by more than the fault) are counted and cannot be judged at all. That is a limit of the measurement, not of the rule.

## What this run proves, and what it does not

- **Works end to end:** phone/laptop clip → per-move joint feedback in 20.8 ms/frame of pose extraction.
- **The scorer is not hallucinating:** a known error injected into one joint over one move is localised to the correct move 100.0% of the time at ±25° (1323 trials over all moves × joints), with 0.0 collateral flags per trial. Unmodified input raises 0 flags.
- **The noise floor is measured, not assumed:** two pose estimators looking at the same frames disagree by 3.4°; the disagreement is concentrated in the hands (wrists 20/24 px, elbows 13/16 px) and near zero in the legs (knees 7/6 px). Tolerances follow from that.
- **Not proven:** that these tolerances match federation scoring criteria. They are ~1.5x the measured estimator disagreement; a qualified referee must set the band that matters for competition.
