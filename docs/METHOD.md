# Method

Five stages, then one decision that is the actual contribution.

## 1. Angles that survive a camera move

`poomsae/core.py` extracts 33 MediaPipe landmarks per frame and keeps 9 angles. Every angle is
computed from three landmarks and is therefore invariant to where the athlete stands in frame —
but not to where the *camera* stands:

| feature | definition | invariant to |
|---|---|---|
| `L/R elbow` | angle at the elbow, shoulder–elbow–wrist | image translation and scale |
| `L/R knee` | angle at the knee, hip–knee–ankle | image translation and scale |
| `L/R hip` | thigh vs torso axis | image translation and scale |
| `L/R shoulder` | upper arm vs torso axis | image translation and scale |
| `torso_tilt` | torso axis vs image vertical | image translation and scale |

Two deliberate choices: angles are **hip-centred** (the pelvis is the most stable landmark under
MediaPipe, and it is the body's own origin), and **torso-scaled** (a distance in shoulder-widths,
not in pixels). Neither fixes view geometry — nothing 2D can — which is why stage 5 exists.

Two estimators are run on identical frames: `pose_landmarker_full` and `pose_landmarker_heavy`.
Their disagreement is not an error to minimise, it is the **measurement floor**, and it is measured
per cell (`out/uncertainty.json`). Pooled per-move p95 disagreement on Taegeuk 1 is **17.6°** for
the elbow; on held-out forms 2–8 it is **13.2°**.

## 2. Movement segmentation and alignment

Movements are segmented on the reference timeline, then the student's frames are aligned to the
reference by **DTW** over the 9-D angle series. Two views of the same form give
`dtw_cost_per_frame` 0.90 (same-side) versus 2.77 (opposite-side) — the first signal that camera
geometry, not performance, is the dominant difference in the second case.

Per movement and per joint, `delta = |student_angle - reference_angle|` at the DTW-matched frame.
Nineteen movements × nine joints = **171 cells**.

## 3. The measurement tolerance (and what it is not)

`tolerance = ceil(pooled per-move p95 of cross-estimator disagreement)`, clamped per joint:
elbow **18°**, shoulder 12°, knee/hip 10°, torso 5°.

This number answers exactly one question: *is this difference larger than two independent pose
models arguing about identical pixels?* It was then used, for several iterations, to decide
whether an athlete had made a mistake. That is the bug the repo is about — see §5.

## 4. How much do humans vary? (two independent measurements)

| measurement | script | what it isolates | result |
|---|---|---|---|
| within-take repeats | `poomsae/repeatability.py` | same athlete, same camera, same technique repeated inside one form | elbow sd **6.5–8.0°** (n ≥ 6 groups), knee 0.7°, torso 0.3° |
| cross-athlete | `poomsae/between_athlete.py` | 94 different players, 100-phase normalised trajectory | elbow sd **32.7° mean / 68.5° worst**, knee 4.7° / 11.3° |

The 18° elbow tolerance sits at **2.3× the repeat spread** — defensible as "a real change" — and
at **0.55× the cross-athlete spread** — indefensible as "a mistake". A threshold cannot be both.
So the threshold stayed, and the *comparison* changed.

## 5. Attribution: the one idea

Each athlete is compared to their own form.

```
offset[j]   = median over this athlete's moves of delta[·, j]      # style, build, camera
residual    = delta[move, j] - offset[j]
scale[j]    = max(1.4826 * MAD(residuals for j), floor[j])
floor[j]    = cross-estimator disagreement on that cell,
              raised to the measured repeat spread where a repeat group exists

fault(move, j)  <=>  |residual| > tolerance[j]
                 and |residual| > GATE_K * scale[j]              # GATE_K = 2.5
                 and |residual| > band(move, j)                  # that cell's estimator band
```

Three properties this buys, all measurable:

1. **A uniform habit is not a fault.** In the style arm (+20° elbow, −15° knee injected
   systematically), the raw gate flags 39 cells; the attribution layer reports **zero findings
   beyond the clean arm** and recovers the injected offsets at **+20.4° / −15.0°**.
2. **A camera offset is not a fault either.** In the opposite-side pair, 40 flags become **11
   candidates**, of which 9 are labelled *this athlete's own offset under that geometry*.
3. **Flags are untouched.** The measurement layer still flags 9 cells on the same-side card and
   50 on the cross-side card; the gate adds a verdict beside each flag and never removes one.
   `flags_unchanged` is a SHA-256-class fingerprint over `(move, joint, delta, tolerance, message,
   band, release)` before and after the layer runs — `true` on both cards.

The third condition was **forced by the clean arm**, not designed: with only two conditions,
three of the first five "faults" on a self-comparison were smaller than the two pose models'
disagreement on those exact frames (residual −45.2° inside a ±68.2° band).

Effective threshold, reported next to the tolerance everywhere:

```
effective[j] = max(tolerance[j], GATE_K * floor[j])
```

## 6. Presentation is downstream of the numbers

`make_card.py` writes the coach card. `render_video.py` draws the overlay and **reads its
thresholds from the card's `effective_threshold_deg`** — it carries no table of its own, and the
HUD prints both numbers as `tol 18 / gate 21 deg`. This is not stylistic: an earlier build had a
private `{"R_elbow": 25.0, ...}` table in the renderer, so a 21.5° elbow that the card called a
fault was drawn amber as "close". The video is the artefact a coach actually trusts, so a second
threshold table in presentation code is a correctness bug.

`build_demo_page.py` generates `out/demo.html` with three pills per cell — `fault`,
`own offset`, `within noise` — from the same JSON.

## 7. What was refused

Each refusal is a committed artifact, not a paragraph:

| refused | why | artifact |
|---|---|---|
| learned gate | AUC **0.7066**; at matched TPR the false-positive rate is **91.67%** (rule arm 88.46%): a threshold without a meaning | `out/learned_gate.json` |
| 3D realignment | 0.867× the deviation range, SNR **6.59 vs 8.65** — responds less, so it looks "better" | `out/crossview_3d.json` |
| view-bias calibration as the fix | a fault committed on every move is absorbed into the bias; recall at 10° injection collapses to **0.389** | `out/view_calibration.json` |
| an LLM (Jev) as the gate | blind AUC 0.6125, single-reading AUC 0.4781 — advice only, never authorisation | `out/jev_gate.json` |
| measured-base injection as sensitivity | the artefact base consumes the tolerance: 0.6038 of judgeable cells at 1.05× | `out/self_reference.json` |
