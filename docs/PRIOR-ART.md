# Prior art

Searched 2026-11-21 across GitHub (topic and code search: `poomsae`, `taekwondo form`, `poomsae
scoring`, `DTW pose`), arXiv/APSIPA/JICCE literature, and the dataset registries (Mendeley Data,
TUHAD, EnjoyPoomsae, TUAD-3D, NTU RGB+D). The table lists what survives relevance filtering; the
paragraphs after it are the part that matters.

| work | type | method | what it publishes | gap |
|---|---|---|---|---|
| Hoang & Ahn 2023, *An Evaluation Method of Taekwondo Poomsae Performance*, JICCE 21(12) | paper | Rodrigues-vector joint representation + DTW against a motion-captured champion | mean / min / max per-movement cost per performer | **no threshold**, no noise floor, no ground-truth check; cost is never converted into a verdict |
| Ang et al. 2025, *SyncScore*, APSIPA ASC | paper | ViTPose landmarks + SVR regression on inter-person distance | R² = 0.3915 against expert synchronisation scores | measures **synchronisation between two people**, not correctness of one |
| `Doankt/poomsae-score-py` | repo, GPL-3.0 | scoring prototype, 2019 | code | no validation, no data, no threshold |
| `Jiwoo-Jung-AIS/Offline_Poomsae_Trainer` | repo | MMPose + Gradio, 14 movements, offline | runnable demo | no accuracy figure, no threshold provenance |
| `SriMed/sr_eval_tkd_v2` | repo | Detectron2 + VideoPose3D + DTW | course project | no evaluation protocol, no held-out split |
| TUHAD (dataset) | dataset | 10 experts × 8 techniques × 12 reps × 2 separate executions | video + labels of *technique* | no execution-quality labels; data behind a JS-gated share |
| EnjoyPoomsae (dataset) | dataset | 540 forms 1–4 | video, restricted | CC BY-NC, no quality labels |
| this repo | — | 9 angles + DTW + self-referenced three-condition gate | tolerance provenance, four-arm certification, published floor, refusals | single performer per view; synthetic faults only |

## The gap, stated precisely

Every entry above either (a) computes a similarity cost and stops there, or (b) trains a classifier
on labels that say *which* movement was performed, not *whether it was performed well*. Neither
answers the question a coach asks: **"is this difference big enough to be worth mentioning?"**

That question needs three numbers, and I could not find a published work that gives all three:

1. the estimators' own disagreement on identical frames (measurement floor),
2. human variation for the same movement (the reachable range),
3. the point where a difference stops being attributable to the athlete's habit.

The literature's silence on (1) and (3) is why app-store poomsae apps can show a "score" with no
stated error bar, and why the closest academic work reports a cost with no threshold.

## What this repo adds, and what it does not

**Adds:** a measured tolerance with its provenance (`out/uncertainty.json`), a self-referenced
attribution layer whose effective threshold is printed (`21.0°` right elbow), four-arm
certification including the arms where it *refuses* (0.5×/0.9×), a published detector floor
(1.75%), and a machine-checked claims file (`scripts/verify_claims.py`).

**Does not:** beat SyncScore at synchronisation, produce WT scores, or claim SOTA on the 8-class
dataset — where the honest reading is +0.7 points over 1-nearest-neighbour on a near-separable
dataset.

## One precedent worth naming

Poomsae and kata judging has a long tradition of **not publishing the number below which a
difference is not real** — in competition, scores are given to two decimals against a panel
average that no one publishes an error bar for. This repo's contribution is deliberately narrow
and unglamorous: pick one threshold, prove where it came from, show the cases where it fires, and
show the cases where it refuses.
