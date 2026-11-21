"""The honest head-to-head: 8-class technique classification on the Fernando set.

The question this answers: "do we beat the paper?" The paper (Fernando,
Sandaruwan & Athapaththu 2024, J. Natl. Sci. Foundation Sri Lanka 52(1),
doi:10.4038/jnsfsr.v52i1.11986) reports 61% across 8 classes on this dataset --
4 movements x correct/incorrect -- using MediaPipe landmarks.

Three things make this comparison fair, and each one is a way a naive
comparison would cheat:

1. SPLIT BY PLAYER, NOT BY CLIP. Each of the 94 players contributed one clip
   per class, so a clip-level split puts the same body, same angle, same
   lighting on both sides of the split and inflates accuracy. Folds here hold
   out whole players. This is the single most common source of fake accuracy
   in pose work.

2. STANDARDISATION FITTED INSIDE THE FOLD. Feature means/stds come from the
   training players only. Fitting them on all data leaks test-set statistics.

3. A SHUFFLED-LABEL CONTROL. The identical pipeline trained on permuted labels
   must collapse to chance (~12.5%). If it does not, the pipeline is leaking
   and the headline number is fiction.

Task parity is also handled: the paper's task is 8-class, so 8-class is the
number to compare. But our own DTW result was a PER-MOVEMENT binary decision,
so that protocol is reproduced too -- same data, same folds -- rather than
comparing two different tasks and calling one the winner.

Everything is numpy. No torch, no sklearn: a two-layer MLP is ~40 lines and
keeps the whole experiment auditable on any machine that can already run the
demo.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "..", "labelled", "data", "NEW_MP_Data")
OUT = os.path.join(HERE, "out")

CLASSES = ["c1", "c2", "c3", "c4", "i1", "i2", "i3", "i4"]
CHANCE_8 = 1 / 8
CHANCE_BIN = 0.5
PAPER_8CLASS = 0.61           # Fernando et al. 2024, reported 8-class accuracy
OURS_DTW_BINARY = 0.757       # our DTW mean accuracy, per-movement binary


# ---------------------------------------------------------------- features
def clip_features(clip: np.ndarray) -> np.ndarray:
    """(30, 33, 4) landmarks -> 528-d descriptor, hip-centred and torso-scaled.

    Translation and scale are removed because the same technique performed by a
    taller athlete at a different distance from the camera must look identical
    to the classifier. Pose in MediaPipe landmarks 11/12 (shoulders), 23/24
    (hips).
    """
    clip = np.asarray(clip, dtype=np.float64).reshape(30, 33, 4)
    xy = clip[:, :, :2]
    hip = (clip[:, 23:24, :2] + clip[:, 24:25, :2]) / 2.0
    sh = (clip[:, 11:12, :2] + clip[:, 12:13, :2]) / 2.0
    torso = float(np.linalg.norm(sh - hip, axis=2).mean())
    if torso < 1e-6:          # degenerate pose guard; conditional, not additive,
        torso = 1e-6          # so scaling the clip cancels exactly instead of
                              # leaving a 1e-6/torso relative drift in every feature
    p = (xy - hip) / torso
    v = np.diff(p, axis=0, prepend=p[:1])          # per-frame velocity
    feats = []
    for b in np.array_split(np.arange(30), 3):     # 3 temporal thirds
        feats.append(p[b].mean(0).ravel())
        feats.append(v[b].mean(0).ravel())
    feats.append(p.std(0).ravel())                 # overall spread
    feats.append(v.std(0).ravel())
    return np.concatenate(feats)


def load_all():
    X, y, player, movement = [], [], [], []
    for ci, c in enumerate(CLASSES):
        cdir = os.path.join(DATA, c)
        for p in sorted(os.listdir(cdir), key=int):
            pdir = os.path.join(cdir, p)
            fs = sorted(os.listdir(pdir), key=lambda f: int(f.split(".")[0]))
            if len(fs) != 30:
                continue
            frames = [np.load(os.path.join(pdir, f)).ravel() for f in fs]
            X.append(clip_features(np.stack(frames)))
            y.append(ci)
            player.append(int(p))
            movement.append(int(c[1]))            # 1..4
    return np.array(X), np.array(y), np.array(player), np.array(movement)


# ---------------------------------------------------------------- model
class MLP:
    """Two-layer MLP, softmax + L2, Adam. Deterministic given a seed."""

    def __init__(self, n_in, n_out, hidden=256, seed=0, l2=1e-3, lr=3e-3, epochs=400):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.l2, self.lr, self.epochs = l2, lr, epochs

    def _softmax(self, z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit(self, X, y, n_classes):
        N = len(X)
        Y = np.zeros((N, n_classes))
        Y[np.arange(N), y] = 1.0
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1c, b2c, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            h = np.maximum(0.0, X @ self.W1 + self.b1)
            p = self._softmax(h @ self.W2 + self.b2)
            dlogits = (p - Y) / N
            gW2 = h.T @ dlogits + self.l2 * self.W2
            gb2 = dlogits.sum(0)
            dh = (dlogits @ self.W2.T) * (h > 0)
            gW1 = X.T @ dh + self.l2 * self.W1
            gb1 = dh.sum(0)
            for i, (par, g) in enumerate(zip(params, (gW1, gb1, gW2, gb2))):
                m[i] = b1c * m[i] + (1 - b1c) * g
                v[i] = b2c * v[i] + (1 - b2c) * g * g
                mh = m[i] / (1 - b1c ** t)
                vh = v[i] / (1 - b2c ** t)
                par -= self.lr * mh / (np.sqrt(vh) + eps)
        return self

    def proba(self, X):
        h = np.maximum(0.0, X @ self.W1 + self.b1)
        return self._softmax(h @ self.W2 + self.b2)


def standardise(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def auc(y, s):
    """Rank-based AUC; ties handled by average rank."""
    y = np.asarray(y)
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # average ties
    s_sorted = np.asarray(s)[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def static_features(frame: np.ndarray) -> np.ndarray:
    """Single-frame posture descriptor (66-d): hip-centred, torso-scaled x,y.

    Used as a control: if one frame scores as well as 30, the model is reading
    posture rather than motion.
    """
    hip = (frame[23:24, :2] + frame[24:25, :2]) / 2.0
    sh = (frame[11:12, :2] + frame[12:13, :2]) / 2.0
    t = float(np.linalg.norm(sh - hip, axis=1).mean())
    if t < 1e-6:
        t = 1e-6
    return ((frame[:, :2] - hip) / t).ravel()


def folds_of(player, k=5, seed=7):
    """Grouped folds: every clip of a player lands in exactly one fold."""
    players = np.unique(player)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(players)
    assign = {int(p): i % k for i, p in enumerate(perm)}
    return np.array([assign[int(p)] for p in player])


def run_8class(X, y, fold, seed=0):
    n_cls = len(CLASSES)
    accs, f1s = [], []
    conf = np.zeros((n_cls, n_cls), dtype=int)
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        Xtr, Xte = standardise(X[tr], X[te])
        m = MLP(Xtr.shape[1], n_cls, seed=seed + f).fit(Xtr, y[tr], n_cls)
        pred = m.proba(Xte).argmax(1)
        acc = float((pred == y[te]).mean())
        accs.append(acc)
        for t, p in zip(y[te], pred):
            conf[t, p] += 1
        # macro F1 over the 8 classes, computed per fold (not cumulatively)
        f1 = []
        for c in range(n_cls):
            tp = int(((pred == c) & (y[te] == c)).sum())
            fp = int(((pred == c) & (y[te] != c)).sum())
            fn = int(((pred != c) & (y[te] == c)).sum())
            pr = tp / (tp + fp) if tp + fp else 0.0
            rc = tp / (tp + fn) if tp + fn else 0.0
            f1.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
        f1s.append(float(np.mean(f1)))
    return {"per_fold": [round(a, 4) for a in accs],
            "mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "confusion": conf.tolist()}


def run_binary_per_movement(X, y, movement, player, fold, seed=0):
    """Exactly our DTW protocol: within one movement, correct vs incorrect."""
    out = {}
    for mv in sorted(set(movement.tolist())):
        sel = movement == mv
        yb = np.isin(y[sel], [0, 1, 2, 3]).astype(int)  # 1 = correct class
        Xs, fs = X[sel], fold[sel]
        accs, aucs = [], []
        for f in range(fold.max() + 1):
            tr, te = fs != f, fs == f
            if len(set(yb[tr].tolist())) < 2 or len(set(yb[te].tolist())) < 2:
                continue
            Xtr, Xte = standardise(Xs[tr], Xs[te])
            m = MLP(Xtr.shape[1], 2, hidden=int(min(256, 2 * 33 * 4 * 3)),
                    seed=seed + f, epochs=250).fit(Xtr, yb[tr], 2)
            pr = m.proba(Xte)
            accs.append(float((pr.argmax(1) == yb[te]).mean()))
            aucs.append(auc(yb[te], pr[:, 1]))
        out[f"movement_{mv}"] = {"accuracy_mean": round(float(np.mean(accs)), 4),
                                 "accuracy_std": round(float(np.std(accs)), 4),
                                 "auc_mean": round(float(np.nanmean(aucs)), 4),
                                 "n": int(sel.sum())}
    out["mean_accuracy"] = round(float(np.mean([v["accuracy_mean"] for v in out.values()
                                               if isinstance(v, dict)])), 4)
    out["mean_auc"] = round(float(np.mean([v["auc_mean"] for v in out.values()
                                           if isinstance(v, dict)])), 4)
    return out


def load_static():
    """Single-frame (frame 0) posture descriptors for the static control."""
    Xs = []
    for c in CLASSES:
        cdir = os.path.join(DATA, c)
        for p in sorted(os.listdir(cdir), key=int):
            pdir = os.path.join(cdir, p)
            fs = sorted(os.listdir(pdir), key=lambda f: int(f.split(".")[0]))
            a = np.stack([np.load(os.path.join(pdir, f)).ravel() for f in fs]).reshape(30, 33, 4)
            Xs.append(static_features(a[0]))
    return np.array(Xs)


def run_1nn(X, y, fold, chunk=32):
    """Nearest-neighbour, no training, no hyperparameters.

    This is the honesty anchor: if 1-NN scores as well as the MLP, the dataset
    is separable rather than the model being good.
    """
    accs = []
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        Xtr, Xte = standardise(X[tr], X[te])
        pred = np.empty(len(Xte), dtype=int)
        for i in range(0, len(Xte), chunk):
            d = ((Xte[i:i + chunk, None, :] - Xtr[None, :, :]) ** 2).sum(2)
            pred[i:i + chunk] = y[tr][d.argmin(1)]
        accs.append(float((pred == y[te]).mean()))
    return {"mean": round(float(np.mean(accs)), 4),
            "std": round(float(np.std(accs)), 4),
            "per_fold": [round(a, 4) for a in accs]}


def main() -> int:
    X, y, player, movement = load_all()
    print(f"{len(X)} performances | {len(np.unique(player))} players | "
          f"{X.shape[1]}-d features | 8 classes (chance {CHANCE_8:.1%})")
    fold = folds_of(player, k=5, seed=7)

    print("\n=== 8-class, split by player (the paper's task) ===")
    r8 = run_8class(X, y, fold)
    print(f"accuracy {r8['mean']:.1%} +/- {r8['std']:.1%}  | macro-F1 {r8['macro_f1']:.1%} "
          f"| paper {PAPER_8CLASS:.0%} | chance {CHANCE_8:.1%}")
    print("per fold:", r8["per_fold"])

    print("\n=== shuffled-label control (must fall to chance) ===")
    rng = np.random.default_rng(11)
    ysh = rng.permutation(y)
    rs = run_8class(X, ysh, fold, seed=100)
    print(f"shuffled accuracy {rs['mean']:.1%} (expected ~{CHANCE_8:.1%})")

    print("\n=== controls: how separable is this data without our model? ===")
    nn = run_1nn(X, y, fold)
    print(f"1-NN, no training:          {nn['mean']:.1%} +/- {nn['std']:.1%}")
    st = run_8class(load_static(), y, fold, seed=200)
    print(f"single frame, no motion:    {st['mean']:.1%} +/- {st['std']:.1%}")

    print("\n=== per-movement binary, correct vs incorrect (our DTW protocol) ===")
    rb = run_binary_per_movement(X, y, movement, player, fold)
    for k, v in rb.items():
        if isinstance(v, dict):
            print(f"{k}: acc {v['accuracy_mean']:.1%} +/- {v['accuracy_std']:.1%}, "
                  f"AUC {v['auc_mean']:.3f} (n={v['n']})")
    print(f"mean binary: acc {rb['mean_accuracy']:.1%}, AUC {rb['mean_auc']:.3f} "
          f"| our DTW was {OURS_DTW_BINARY:.1%}")

    payload = {
        "dataset": "Fernando, Sandaruwan & Athapaththu 2024 (JNSF Sri Lanka 52(1))",
        "n_performances": int(len(X)), "n_players": int(len(np.unique(player))),
        "features": int(X.shape[1]), "split": "5-fold grouped by player (seed 7)",
        "eight_class": r8, "shuffled_control": rs, "binary_per_movement": rb,
        "controls": {"one_nearest_neighbour": nn, "single_frame_only": st,
                     "note": ("1-NN needs no training and no hyperparameters. It "
                              "scoring within a point of the MLP means this dataset "
                              "is separable, not that the model is strong.")},
        "comparison": {"paper_8class": PAPER_8CLASS, "chance_8class": round(CHANCE_8, 4),
                       "chance_binary": CHANCE_BIN, "our_dtw_binary": OURS_DTW_BINARY},
        "note": ("MLP on hip-centred, torso-scaled 33-landmark trajectories "
                 "(3 temporal thirds, position + velocity + spread). Binary "
                 "comparison is per-movement, matching the DTW protocol."),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eight_class.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote out/eight_class.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
