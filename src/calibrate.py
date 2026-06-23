import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Design (settled 12 Jun, metric resolved 23 Jun):
#   - A frozen calibration partition (~150 images, stratified by attack_type)
#     is carved ONCE from the test set. PARTITION_SEED fixes it.
#   - The remaining images form a fixed reporting set whose composition never
#     changes across calibration-set sizes, so the recovery curve isolates a
#     single variable.
#   - Calibration-set sizes 10/25/50 are drawn stratified from the partition
#     with N_REPEATS seeded draws; results reported as mean +/- std.
#   - A threshold is fit on each calibration draw, then applied to the fixed
#     reporting set. Recovery is measured by balanced accuracy and MCC
#     (PRIMARY: predict-all-forged cannot fake recovery on a 3:1 subset),
#     F1 (SECONDARY: shown precisely because it CAN be inflated by the
#     imbalance, which is itself a finding), and AUC (STATIC reference: a
#     threshold cannot change it, so it anchors the inverted-text story).
# ---------------------------------------------------------------------------

PARTITION_SEED = 42
PARTITION_SIZE = 150
CALIB_SIZES = [10, 25, 50]
N_REPEATS = 20


def load_scores(csv_path):
    """Read a frozen scores CSV (cols: idx, y_true, p_forged, attack_type)."""
    df = pd.read_csv(csv_path)
    expected = {"idx", "y_true", "p_forged", "attack_type"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")
    return df


def stratified_draw(df, n, seed, key="attack_type"):
    """Draw n rows from df, stratified by `key`, with a fixed seed.
    Proportional allocation; remainder assigned by largest fractional part."""
    rng = np.random.default_rng(seed)
    groups = {k: sub.index.to_numpy() for k, sub in df.groupby(key)}
    total = len(df)
    # proportional counts per stratum
    raw = {k: n * len(idx) / total for k, idx in groups.items()}
    floor = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(floor.values())
    # hand out the remainder to the largest fractional parts
    frac_order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in frac_order[:remainder]:
        floor[k] += 1
    picked = []
    for k, idx in groups.items():
        take = min(floor[k], len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    return df.loc[picked]


def carve_partition(df):
    """Carve the frozen calibration partition once, stratified, and return
    (calib_partition, reporting_set). Reporting set never changes."""
    partition = stratified_draw(df, PARTITION_SIZE, PARTITION_SEED)
    reporting = df.drop(index=partition.index)
    return partition, reporting


def fit_threshold_f1(y_true, p_forged):
    """F1-optimal threshold on a calibration draw. Sweeps candidate
    thresholds at the unique scores and returns the one maximising forged-F1."""
    candidates = np.unique(p_forged)
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (p_forged >= t).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def apply_threshold(p_forged, t):
    return (p_forged >= t).astype(int)


def recovery_metrics(y_true, y_pred):
    """Threshold-fair recovery metrics. balanced_accuracy and mcc cannot be
    inflated by predict-all-forged on an imbalanced subset; f1 can, and is
    kept as the cautionary secondary."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def static_auc(y_true, p_forged):
    """Threshold-independent. Calibration cannot move this; it is the
    reference ceiling the recovery metrics climb toward (or cannot)."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p_forged))


def subset_by_type(df, atk):
    """Reporting subset for one attack type: that type's rows plus all
    bonafide rows. Mirrors per_attack_type_auc in evaluate.py so the
    recovery curve and the headline AUC table describe the same comparison."""
    if atk == "none":
        return df
    return df[(df["attack_type"] == atk) | (df["attack_type"] == "none")]


def run_recovery(calib_partition, reporting, atk):
    """For one attack type (or 'global'), sweep calibration-set sizes, draw
    N_REPEATS stratified calibration sets, fit a threshold on each, apply to
    the fixed reporting subset, and return mean/std of each recovery metric
    plus the static AUC of the reporting subset."""
    if atk == "global":
        cp, rp = calib_partition, reporting
    else:
        cp, rp = subset_by_type(calib_partition, atk), subset_by_type(reporting, atk)

    auc = static_auc(rp["y_true"].to_numpy(), rp["p_forged"].to_numpy())
    out = {"static_auc": auc, "n_report": int(len(rp)), "sizes": {}}

    for size in CALIB_SIZES:
        if size > len(cp):
            continue
        runs = []
        for r in range(N_REPEATS):
            draw = stratified_draw(cp, size, seed=1000 * size + r)
            t = fit_threshold_f1(draw["y_true"].to_numpy(),
                                 draw["p_forged"].to_numpy())
            pred = apply_threshold(rp["p_forged"].to_numpy(), t)
            runs.append(recovery_metrics(rp["y_true"].to_numpy(), pred))
        agg = {}
        for m in ("balanced_accuracy", "mcc", "f1"):
            vals = np.array([x[m] for x in runs])
            agg[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
        out["sizes"][size] = agg
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="path to a frozen scores_<split>_<model>.csv")
    ap.add_argument("--model", default=None,
                    help="model label for the output; defaults to filename stem")
    args = ap.parse_args()

    df = load_scores(args.scores)
    model_label = args.model or Path(args.scores).stem
    print(f"Loaded {len(df)} scored samples from {args.scores}")
    print(f"Attack-type counts: {df['attack_type'].value_counts().to_dict()}")

    partition, reporting = carve_partition(df)
    print(f"Calibration partition: {len(partition)} "
          f"(seed {PARTITION_SEED}); reporting set: {len(reporting)}")

    attack_types = [a for a in sorted(df["attack_type"].unique()) if a != "none"]
    stages = ["global"] + attack_types

    results = {"model": model_label, "scores_file": str(args.scores),
               "partition_seed": PARTITION_SEED, "n_repeats": N_REPEATS,
               "stages": {}}
    for atk in stages:
        results["stages"][atk] = run_recovery(partition, reporting, atk)

    # Readable summary
    print("\n=== Recovery curve (mean +/- std over "
          f"{N_REPEATS} draws) ===")
    for atk, d in results["stages"].items():
        auc = d["static_auc"]
        auc_s = f"{auc:.4f}" if auc is not None else "n/a"
        print(f"\n{atk:8s}  static AUC={auc_s}  (n_report={d['n_report']})")
        for size in CALIB_SIZES:
            if size not in d["sizes"]:
                continue
            m = d["sizes"][size]
            print(f"  size {size:3d}  "
                  f"balacc {m['balanced_accuracy']['mean']:.3f}"
                  f"+/-{m['balanced_accuracy']['std']:.3f}  "
                  f"mcc {m['mcc']['mean']:+.3f}+/-{m['mcc']['std']:.3f}  "
                  f"f1 {m['f1']['mean']:.3f}+/-{m['f1']['std']:.3f}")

    out_path = Path(args.scores).parent / f"calibration_{model_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
