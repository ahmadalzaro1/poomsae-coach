# Data — what ships, what does not, and why

Three different data sources appear in this project. Only the artifacts computed from them are
committed, because that is the only part that can be redistributed legally.

## 1. Taegeuk 1–8 dual-view dataset — **CC BY 4.0, not committed**

- **What:** synchronised frontal + lateral recordings of Taegeuk 1st–8th (1920×1080, 30 fps,
  MP4/H.264), with frame-level 12-keypoint MediaPipe annotations (`frame_index`, `time_ms`,
  `<joint>_x`, `<joint>_y`, `<joint>_score`), 16 clips in total.
- **Citation:** Zheng, Q. (2025). *Taekwondo Poomsae (8 Forms) — Front & Side Videos with
  12-Keypoint Pose Annotations (MediaPipe)*. Mendeley Data, V1.
  <https://doi.org/10.17632/bjy7vr4xkt.1>. National Kaohsiung University of Science and Technology.
- **Licence:** CC BY 4.0 per the dataset page. Attribution text required by the licence is in
  [ATTRIBUTION.md](../ATTRIBUTION.md).
- **Not committed because:** 280 MB of video has no place in a git history. Ten derived clips
  in `media/` are committed with attribution: the two demos plus the eight five-second form-tour
  excerpts. Derived artifacts and their hashes are in `out/`.
- **How to get it:** `python scripts/fetch_data.py` (prints the DOI and, if the archive is
  present locally, verifies the expected directory shape), or download manually from the DOI
  page and unpack to `data/Data/video/Taegeuk_{1st..8th}/{Frontal,Lateral}/`.

Expected layout after unpacking — the pipeline reads this and nothing else:

```
data/Data/video/Taegeuk_1st/Frontal/*.mp4
data/Data/video/Taegeuk_1st/Lateral/*.mp4
...                                            (through Taegeuk_8th)
```

## 2. Pose model weights — **not committed, freely downloadable**

| file | size | source |
|---|---:|---|
| `pose_landmarker_full.task` | 9.4 MB | `storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/` |
| `pose_landmarker_heavy.task` | 30.7 MB | `storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/` |
| `hand_landmarker.task` | 7.8 MB | `storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/` (checked by `run_all.sh`, unused by the gate) |

Both are MediaPipe's own published weights. `python scripts/fetch_models.py` downloads them into
`models/` and prints the SHA-256 of what it wrote, so a reader can confirm they have the same
models this report measured. Estimates quoted in this repo were produced with MediaPipe 1.2.3.

## 3. 752 expert-labelled performances, 94 players — **NO LICENCE, not committed, not redistributable**

- **Source:** `michelleuf/Poomsae_Movement_Classifier_Dataset`, the artefact of Fernando,
  Sandaruwan & Athapaththu (2024), *Journal of the National Science Foundation of Sri Lanka*
  52(1), doi 10.4038/jnsfsr.v52i1.11986 — 752 poomsae performances labelled correct/incorrect
  across 8 movements, 94 players.
- **Licence:** the upstream repository contains **no licence file**. Under default copyright that
  means all rights reserved: neither the `.npy` tensors nor a model trained on them may be
  redistributed without written permission.
- **What this repo therefore does:** publishes only the *numbers computed from it* —
  `out/eight_class.json` (accuracy and the four controls) and `out/between_athlete.json`
  (cross-athlete spread, with filesystem paths replaced by `<NEW_MP_Data>`). The tensors
  themselves stay out of git and out of every artefact.
- **Internal layout for reference** (if you have your own copy):
  `NEW_MP_Data/<class>/<player>/0..29.npy`, shape `(94, 8, 30, 132)`.

## Reproducibility note

`scripts/verify_claims.py` re-derives every documented number — the tables in `README.md` and
`docs/RESULTS.md` included — from the committed JSON files with no dataset, no models and no network. The full pipeline (`run_all.sh`, 23 steps) needs datasets 1
and 2 above and fails loudly if they are absent — it never silently writes an empty result.
