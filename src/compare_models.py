"""Inter-model statistical comparison on the shared FantasyID test set.

Reads the three frozen per-sample score CSVs and runs, per attack class
(face, text) and overall:

  - DeLong's test comparing two ROC-AUCs on the same samples (primary;
    threshold-independent, matches the ranking finding).
  - McNemar's exact test on paired 0.5-threshold predictions (secondary;
    error-agreement check, reported with the inverted-ranking caveat).

A t-test across the three single runs is deliberately NOT used: with one
run per architecture there is no sampling distribution to test against.

Reads frozen CSVs only. No torch, no model, no GPU. Runs on the login node.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

MODELS = {
    "ResNet50": "scores_test_resnet50.csv",
    "DenseNet121": "scores_test_densenet121.csv",
    "ViT-B/16": "scores_test_vit.csv",
}


# ---- DeLong (fast, Sun & Xu 2014) ----------------------------------------

def _compute_midrank(x):
    J = np.argsort(x)
    z = x[J]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    tout = np.empty(n, dtype=float)
    tout[J] = t
    return tout


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    from scipy.stats import norm
    z = float(z.item())
    p = float(2 * (1 - norm.cdf(z)))
    return z, p


def delong_test(y_true, score_a, score_b):
    """Two-sided DeLong test that AUC_a == AUC_b on the same samples.

    Returns (auc_a, auc_b, z, p). y_true is 1 for positive (forged), 0 for
    bonafide. Higher score = more forged.
    """
    order = (-y_true).argsort()
    label_1_count = int(y_true.sum())
    preds = np.vstack((score_a, score_b))[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)
    z, p = _calc_pvalue(aucs, cov)
    return float(aucs[0]), float(aucs[1]), z, p


# ---- McNemar (exact binomial) --------------------------------------------

def mcnemar_exact(correct_a, correct_b):
    """McNemar exact test on paired boolean correctness arrays.

    b = A right & B wrong; c = A wrong & B right. Under the null the split
    of the b+c discordant pairs is Binomial(p=0.5). Returns (b, c, p).
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, float(p)


# ---- driver ---------------------------------------------------------------

def load_scores(scores_dir):
    data = {}
    for model, fname in MODELS.items():
        df = pd.read_csv(Path(scores_dir) / fname)
        data[model] = df
    return data


def subset_vs_bonafide(df, attack):
    """Rows for one attack type plus all bonafide, as (y_true, p_forged).

    y_true = 1 for the attack (forged), 0 for bonafide. Bonafide rows are
    y_true == 0; the named attack rows are y_true == 1 with that attack_type.
    """
    bona = df[df["y_true"] == 0]
    forged = df[(df["y_true"] == 1) & (df["attack_type"] == attack)]
    sub = pd.concat([bona, forged]).sort_values("idx")
    return sub["y_true"].to_numpy(), sub["p_forged"].to_numpy()


def run_class(data, label, selector):
    print(f"\n=== {label} ===")
    arrays = {}
    for model, df in data.items():
        y, p = selector(df)
        arrays[model] = (y, p)
        auc = roc_auc_score(y, p)
        print(f"{model}: n={len(y)}, AUC={auc:.4f}")

    print("-- DeLong (AUC vs AUC) --")
    for a, b in itertools.combinations(MODELS, 2):
        ya, pa = arrays[a]
        # both models share the same sample set per class, so y is identical
        auc_a, auc_b, z, p = delong_test(ya, pa, arrays[b][1])
        print(f"{a} vs {b}: AUC {auc_a:.4f} vs {auc_b:.4f}, "
              f"z={z:.3f}, p={p:.4f}")

    print("-- McNemar exact (0.5-threshold predictions) --")
    for a, b in itertools.combinations(MODELS, 2):
        ya, pa = arrays[a]
        yb, pb = arrays[b]
        ca = (pa >= 0.5).astype(int) == ya
        cb = (pb >= 0.5).astype(int) == yb
        nb, nc, p = mcnemar_exact(ca, cb)
        print(f"{a} vs {b}: b={nb}, c={nc}, p={p:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="DeLong and McNemar inter-model comparison on the test set."
    )
    parser.add_argument("--scores-dir", default="checkpoints",
                        help="Directory holding scores_test_<model>.csv files.")
    args = parser.parse_args()

    data = load_scores(args.scores_dir)

    run_class(data, "FACE vs bonafide",
              lambda df: subset_vs_bonafide(df, "face"))
    run_class(data, "TEXT vs bonafide",
              lambda df: subset_vs_bonafide(df, "text"))
    run_class(data, "OVERALL (all 1385 samples)",
              lambda df: (df["y_true"].to_numpy(), df["p_forged"].to_numpy()))

    print("\nNote: a t-test across the three runs is not reported; with one "
          "run per architecture there is no sampling distribution.")


if __name__ == "__main__":
    main()
