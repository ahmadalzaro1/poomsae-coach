#!/usr/bin/env bash
# Full demo run, start to finish. Every artifact in out/ is regenerated.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

[ -x "$PY" ] || { echo "no venv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
[ -d data/Data ] || { echo "no data: run python scripts/fetch_data.py --manual (CC BY 4.0 dataset)"; exit 1; }
[ -f models/pose_landmarker_full.task ] || { echo "no pose model: run python scripts/fetch_models.py"; exit 1; }
[ -f models/pose_landmarker_heavy.task ] || { echo "no heavy pose model: run python scripts/fetch_models.py"; exit 1; }
[ -f models/hand_landmarker.task ] || { echo "no hand model: run python scripts/fetch_models.py"; exit 1; }
mkdir -p out

fail=0
run() { local desc=$1; shift; echo; echo "=== $desc ===";
  if ! "$@" > /tmp/rl_step.log 2>&1; then
    echo "!!! FAILED: $desc"; tail -20 /tmp/rl_step.log; fail=$((fail + 1))
  fi
  grep -v '^INFO\|^W0000\|^I0000' /tmp/rl_step.log || true; }

run "1/23 elbow signal + tolerance derivation" $PY -m poomsae.elbow_signal
run "2/23 selective elbow gate (agreement)"     $PY -m poomsae.elbow_gate
run "3/23 metric 3D extraction, both views"     $PY -m poomsae.pose3d
run "4/23 camera-angle benchmark (2D, 8 forms)"  $PY -m poomsae.run_crossview
run "5/23 same benchmark in 3D + controls"      $PY -m poomsae.run_crossview_3d
run "6/23 learned fault-vs-camera gate"         $PY -m poomsae.learned_gate
run "7/23 view calibration"                     $PY -m poomsae.view_calibration
run "8/23 feedback-engine validation"           $PY -m poomsae.run_injection
run "9/23 subtle-fault benchmark (before/after)" $PY -m poomsae.subtle_faults
run "10/23 technique naming, Taegeuk 1+2"       $PY -m poomsae.run_naming
run "11/23 raw video -> pose -> cards"          $PY -m poomsae.pose_video
run "12/23 per-joint uncertainty band + Jev release" $PY -m poomsae.uncertainty
run "13/23 repeat-observation spread (no second take)" $PY -m poomsae.repeatability
run "14/23 self-referenced gate (style vs fault)" $PY -m poomsae.self_reference
run "15/23 pose audit vs annotations"           $PY -m poomsae.diagnose
run "16/23 arm-gate negative result"            $PY -m poomsae.tune_arm_gate
run "17/23 widget"                              $PY -m poomsae.build_widget
run "18/23 coach card"                          $PY -m poomsae.make_card
run "19/23 demo video, same-side reference"     $PY -m poomsae.render_video --card card_sameview.json --out demo_sameview.mp4
run "20/23 demo video, opposite-side ref"       $PY -m poomsae.render_video --card card_crossview.json --out demo_crossview.mp4 \
    --warn "reference filmed from the OPPOSITE camera side - this deviation is camera angle, not athlete error" \
    --tag "same clip, other reference"

# Needs the external labelled set (752 performances, 94 players); skipped when absent.
if [ -d ../labelled/data/NEW_MP_Data ]; then
  run "21/23 8-class head-to-head, player-disjoint" $PY -m poomsae.run_eight_class
  run "22/23 confound audit on that result"         $PY -m poomsae.check_confounds
  run "23/23 external validation, 94 other players" $PY -m poomsae.validate_labelled
else
  echo; echo "=== 18-21/21 skipped: ../labelled/data/NEW_MP_Data not present (see README) ==="
fi

echo; echo "artifacts:"; ls -1 out/
if [ "$fail" -ne 0 ]; then echo; echo "$fail STEP(S) FAILED - this run is not green"; exit 1; fi
echo; echo "all steps ok - 0 failures"
