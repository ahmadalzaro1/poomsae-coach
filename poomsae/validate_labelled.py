"""External validation on a LABELLED dataset of OTHER players.

Source: michelleuf/Poomsae_Movement_Classifier_Dataset (Fernando, Sandaruwan &
Athapaththu, J. Natl. Sci. Foundation Sri Lanka 52(1), 2024,
doi:10.4038/jnsfsr.v52i1.11986). 752 performances = 94 players x 4 movements,
each labelled correct (c) or incorrect (i) by domain experts; 30 frames each,
MediaPipe 33 landmarks per frame.

Question: does THIS pipeline's deviation score separate expert-labelled correct
from incorrect performances on people the dataset's reference never saw?

Protocol per movement m:
  * reference template = mean angle-feature trajectory of the correct class
  * leave-one-out for correct clips (template recomputed without the clip)
  * score every clip by (a) DTW cost/frame and (b) max per-joint exceedance
  * report AUC + best-threshold accuracy

Aspect assumption: coordinates are normalised with an unknown source aspect, so
angles are computed with a 16:9 pixel grid. That is a constant across every clip
in this dataset, so it cannot manufacture a correct/incorrect separation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from poomsae.core import (FEATURE_NAMES, JOINTS, TOLERANCE_DEG, dtw, features,
                          smooth)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "..", "labelled", "data", "NEW_MP_Data")
MP_IDX = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # -> 12 joints
W, H = 1920.0, 1080.0


class Seq:
    """Minimal stand-in for core.Sequence: px + scores arrays."""
    def __init__(self, px, scores):
        self.px, self.scores = px, scores


def load_clip(path):
    files = sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))
    frames = np.stack([np.load(os.path.join(path, f)).reshape(33, 4) for f in files])
    lm = frames[:, MP_IDX, :]                      # (T, 12, 4)
    px = np.stack([lm[:, :, 0] * W, lm[:, :, 1] * H], axis=-1)   # pixel space
    return Seq(px, lm[:, :, 3])


def feats(path):
    return features(load_clip(path), min_score=0.5)


def auc(pos, neg):
    """Mann-Whitney U -> AUC. P(score(correct) < score(incorrect))."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    rp = ranks[:len(pos)].sum()
    return (len(pos) * len(neg) - rp + len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_acc(pos, neg):
    """Accuracy at the threshold that best separates lower=better scores."""
    vals = np.unique(np.concatenate([pos, neg]))
    best = (0.0, 0.0)
    for t in vals:
        acc = ((pos <= t).sum() + (neg > t).sum()) / (len(pos) + len(neg))
        if acc > best[0]:
            best = (acc, float(t))
    return best


def main():
    F, lab = {}, {}
    for cls in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, cls)
        ids = sorted(os.listdir(d), key=int)
        for i in ids:
            F[(cls, i)] = feats(os.path.join(d, i))
            lab[(cls, i)] = cls
        print(f"loaded {cls}: {len(ids)} performances")

    tol = np.array([TOLERANCE_DEG.get(f, 15.0) for f in FEATURE_NAMES])
    rows = []
    for m in "1234":
        cor, inc = f"c{m}", f"i{m}"
        cids = [i for (c, i) in F if c == cor]
        iids = [i for (c, i) in F if c == inc]
        # leave-one-out template for correct clips; full-correct template for incorrect
        full = np.mean(np.stack([F[(cor, i)] for i in cids]), axis=0)
        dt, ex = [], []
        for i in cids:
            tmpl = np.mean(np.stack([F[(cor, j)] for j in cids if j != i]), axis=0)
            cost, _ = dtw(tmpl, F[(cor, i)])
            dt.append(cost / len(F[(cor, i)]))
            ex.append(float((np.abs(F[(cor, i)] - tmpl).max(0) / tol).max()))
        dt_i, ex_i = [], []
        for i in iids:
            cost, _ = dtw(full, F[(inc, i)])
            dt_i.append(cost / len(F[(inc, i)]))
            ex_i.append(float((np.abs(F[(inc, i)] - full).max(0) / tol).max()))

        a_dt, a_ex = auc(dt, dt_i), auc(ex, ex_i)
        acc_dt, thr_dt = best_acc(np.array(dt), np.array(dt_i))
        acc_ex, thr_ex = best_acc(np.array(ex), np.array(ex_i))
        rows.append({"movement": int(m), "n_ok": len(dt), "n_bad": len(dt_i),
                     "dtw_auc": round(a_dt, 3), "dtw_acc": round(acc_dt, 3),
                     "exceed_auc": round(a_ex, 3), "exceed_acc": round(acc_ex, 3)})
        print(f"\nmovement {m}:  correct {len(dt)}  incorrect {len(dt_i)}")
        print(f"  DTW cost/frame      AUC {a_dt:.3f}   best-threshold acc {acc_dt:.3f}")
        print(f"  max joint exceedance AUC {a_ex:.3f}  best-threshold acc {acc_ex:.3f}")
        print(f"  median DTW correct {np.median(dt):.3f} vs incorrect {np.median(dt_i):.3f}")

    mean_auc = float(np.mean([r["dtw_auc"] for r in rows]))
    mean_acc = float(np.mean([r["dtw_acc"] for r in rows]))
    print(f"\nMEAN over 4 movements: DTW AUC {mean_auc:.3f}, accuracy {mean_acc:.3f}")
    import json
    json.dump({"dataset": "michelleuf/Poomsae_Movement_Classifier_Dataset",
               "paper": "doi:10.4038/jnsfsr.v52i1.11986",
               "rows": rows, "mean_dtw_auc": round(mean_auc, 3),
               "mean_dtw_acc": round(mean_acc, 3)},
              open(os.path.join(HERE, "out", "labelled_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
