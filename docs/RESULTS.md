# Results

Every value here is re-derived from a committed artifact by `python3 scripts/verify_claims.py`
(63/63). Nothing in this file is typed from memory.

## A. The gate, certified on four arms

`out/self_reference.json` → `benchmark`

| arm | raw flags (`|delta| > tolerance`) | self-referenced layer |
|---|---:|---|
| **clean** — student is the reference recording; no fault exists | 9 cells | **3 candidates = 1.75% of 171**, 0 style, 168 within noise |
| **style** — systematic **+20° elbow / −15° knee** | 39 cells | **0 beyond the clean floor**, 37 labelled style, offsets recovered **+20.4° / −15.0°** |
| **zero-base sweep**, 0.5× effective threshold | — | detection **0.000** |
| **zero-base sweep**, 0.9× | — | detection **0.000** |
| **zero-base sweep**, 1.05× | — | **1.000 of judgeable**, 12/171 band-blind, 0.9298 counting band-blind as misses |
| **zero-base sweep**, 1.3× | — | 1.000 of judgeable |
| **zero-base sweep**, 1.6× | — | 1.000 of judgeable |
| **measured-base sweep**, 1.05× | — | 0.6038 of judgeable |
| **measured-base sweep**, 1.6× | — | 0.8788 of judgeable |

Effective thresholds: `R_elbow 21.0`, `L_elbow 18.1`, `L/R knee 10.0`, `L/R hip 10.0`,
`L/R shoulder 12.0`, `torso_tilt 5.0`. `gate_k = 2.5`, `min_moves = 6`.

The clean arm's 3 candidates are the published **floor** of this detector, not a result: the
student *is* the reference, so they can only come from annotation/pose disagreement, and they sit
in moves 15–16 where that disagreement reaches 74°.

## B. The two cards

| | same-side | opposite-side |
|---|---:|---:|
| moves compared | 19 | 16 |
| mean deviation | 3.4° | 12.5° |
| DTW cost / frame | 0.898 | 2.769 |
| measurement flags | 9 | 50 raw cells → 40 card flags (3/move cap) |
| **candidate faults** | **3** | **11** |
| **own-offset cells** | 0 | **9** |
| within noise | 168 | 124 |
| `flags_unchanged` | **true** | **true** |

Headline faults, same-side:

| move | t | technique | cell | delta | own offset | residual | gate | margin |
|---|---:|---|---|---:|---:|---:|---:|---:|
| 6 | 6.2 s | arae makki | R elbow | 21.5° | 0.4° | 21.1° | 21.0° | **+0.1°** |
| 13 | 14.97 s | eolgul makki | L elbow | 22.1° | 0.1° | 22.0° | 18.1° | +3.9° |

## C. Noise floors

| source | file | elbow | knee | torso |
|---|---|---:|---:|---:|
| within-take repeats, n ≥ 6 (sameview) | `out/repeatability.json` | sd 6.5° (n=6), 8.0° (n=7) | 0.7° | 0.3° |
| within-take repeats (crossview) | `out/repeatability.json` | pooled mean 26.5°, sd 14.1° — camera geometry inflates repeats | — | — |
| cross-athlete, 94 players, 100-phase | `out/between_athlete.json` | mean sd 32.7°, max 68.5° | 4.7° / 11.3° | — |
| cross-estimator, per cell | `out/uncertainty.json` | band p50 6.4–8.4° | 1.4–1.5° | 0.3° |

Release policy on the same-side card (`out/uncertainty.json`): **show 2 / soften 6 / withhold 1**
of 19 moves.

## D. The labelled dataset, with four controls

`out/eight_class.json` — 752 expert-labelled performances, 94 players, 8 classes, 5 folds grouped
by player.

| arm | score |
|---|---:|
| **this pipeline, 8-class** | **0.9801** |
| control: 1-nearest-neighbour | 0.9735 |
| control: single frame only | 0.9258 |
| control: shuffled labels | 0.1276 |
| our DTW protocol as a binary per-movement test | 0.757 (AUC 0.787) |
| the dataset paper's own 8-class figure | 0.61 |

Read the first two rows together: **+0.7 points over 1-NN**. This dataset is close to linearly
separable and the honest statement is "our pipeline is a competent baseline here", not "a model
win". The 0.1276 shuffled control is what stops the 0.98 from being self-congratulation.

## E. Refusals, with their numbers

| refusal | number | file |
|---|---|---|
| 3D realignment | deviation range **0.867×** of 2D, SNR **6.59 vs 8.65**, analytic-vs-realignment agreement 0.933 | `out/crossview_3d.json` |
| learned gate | AUC **0.712**, FPR **86.5%** | `out/eight_class.json` |
| view-bias calibration as the fix | calibrated flag rate 0.833 vs 0.885 raw; recall at 10° injection **0.389**, at 20° **0.903** | `out/view_calibration.json` |
| Jev (LLM) as authoriser | blind AUC **0.6125**, single-reading AUC **0.4781**, Spearman −0.031 | `out/jev_gate.json` |
| subtle-fault families | the limit is the tolerance grid, not the measurement (jitter σ ≤ 0.615°): 50% detection at **15°** arm height, **20°** elbow extension — 3° and 2° above each family's own tolerance | `out/subtle_faults.json` |
| naming forms from angles | held-out labelling accuracy **0.837** (calibrated on Taegeuk 1, tested on 2–8) | `out/naming_forms.json` |

## F. Determinism

`out/self_reference.json`, `out/card_sameview.json`, `out/card_crossview.json` are byte-identical
across consecutive runs of step 14–18. Release decisions are cached by SHA-256 of their inputs, so
the LLM advice layer cannot introduce run-to-run drift into a deterministic verdict.
