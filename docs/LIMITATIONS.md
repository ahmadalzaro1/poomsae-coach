# Limitations

Written as a list of things that would have to be true for this to be more than it is. Each item
names what would move it.

## 1. The clean-arm floor is an artefact, and I can say which artefact

In the clean arm the student *is* the reference recording. Any non-zero delta there is annotation
error, pose error, or DTW frame selection — never the athlete. The 3 surviving candidates (1.75%)
therefore measure the tool, and they concentrate in moves 15–16 where the dataset's own
annotation and the pose model disagree by up to 74°. **1.75% is the detector's floor, not its
false-positive rate on real athletes.** A real FPR needs a second recording of the same athlete,
which no public dataset in this domain provides (see `docs/DATA.md`).

## 2. No published fault ground truth exists

The only fault labels in this project are synthetic injections. That tests whether a detector
*notices*, never whether a coach would *agree*. The 8-class labelled dataset gives
correct/incorrect labels at the **movement** level with no joint or magnitude, so it cannot
validate a per-cell threshold. Anyone quoting a per-cell accuracy from this dataset is quoting an
untested quantity.

## 3. The measurement tolerance is an estimator-disagreement bound, nothing more

18° for the elbow is the p95 of two pose models arguing on identical frames. It says nothing about
biomechanics, injury risk, or what a master would accept. A different pose stack would produce a
different number, and the fair response is to re-derive it, not to keep 18 as a constant.

## 4. Camera geometry dominates the cross-view case

Opposite-side comparison gives a 12.5° mean deviation and flags every move. The attribution layer
absorbs 9 of 11 into the athlete's own offset, but absorption is not correction: with a large
uniform bias, the offsets *are* the camera, and a real fault smaller than that bias becomes
invisible. `out/view_calibration.json` prices this exactly — recall at a 10° injection falls to
**0.389** once a per-view bias is removed. The overlay renderer also still assumes a frontal
reference for its skeleton drawing, so the cross-view clip is honest about its numbers but not
fully viewpoint-generic in its drawing.

## 5. 3D was tried and did not help

Realigning frontal to lateral reduced the deviation range to 0.867× of 2D and the signal-to-noise
ratio from 8.65 to 6.59. A feature that deviates less because it responds less is not an
improvement. Documented in `out/crossview_3d.json`, including the warning that its injection recall
is not comparable to the 2D test because the base is already contaminated.

## 6. Injection benchmarks are arithmetic-sensitive

On a zero base, detection at 1.05× the effective threshold is 1.000 of judgeable cells. On a
*measured* base it is 0.6038: an existing 8° plus an injected 21° is not a detectable 29°, and a
band-blind cell cannot resolve the difference at all (12 of 171). Any sensitivity claim that does
not state which base it used is uninterpretable — including, at one point, mine.

## 7. Scope of the features

Nine joint angles cannot see: stance width (no foot separation feature), hand shape (the dataset
annotates 12 joints, no hand landmarks), foot pivot, rhythm/tempo/kime, or power. WT poomsae
scoring weights presentation heavily. This is a joint-angle checker with a defensible threshold,
not a scoring system, and the README does not claim otherwise.

## 8. Small sample, single performer per view

One athlete, one form, two camera positions, 19 movements. The 94-athlete dataset contributes
trajectory spread and a classification result, not fault labels. Statistically this is a
demonstration with measured floors; the honest next step is a second athlete performing the same
form twice, which would turn §1 from "artefact" into "false-positive rate".

## 9. On a quarter of all frames, the 2D mask cannot see the arm

The eight-form tour scores **2,170 of 8,664 frames (25.0%)** as gross pose
failures: one arm cell whose two estimators disagree by 60° or more on the same pixels. Excluding
them buys a median of 1.0–1.9° and a p95 of 16.3–28.3°; including them
would have bought a worse-looking tail and no more truth.

This is the honest ceiling of the measurement layer, and it is why the band condition exists in the
attribution layer: on an occluded arm the estimator's own disagreement is larger than any coaching
threshold, so no threshold applied there means anything. It also bounds what a two-camera version of
this could claim until the arm is tracked in 3D rather than inferred in 2D.

Corollary, and it is a rule not a caveat: **verdicts are published for Taegeuk 1 Jang only.** Seven
forms have measurements and no judgement, because the gate needs that athlete's repeats. A tour that
labelled faults on form 5 would be borrowing another athlete's scale and calling it a finding.

## What would change the verdict

| evidence | would upgrade | would downgrade |
|---|---|---|
| second take by the same athlete, same camera | a real clean-arm FPR | a floor above 5% kills the 1.75% claim |
| a coach-labelled fault set with magnitudes | validity, not just detectability | disagreement on the ~21° elbow boundary |
| a second pose stack (e.g. RTMPose) | tolerance re-derivation, cross-stack agreement | a tolerance that moves by >5° |
| a hand-landmark model | hand-shape faults | — |
