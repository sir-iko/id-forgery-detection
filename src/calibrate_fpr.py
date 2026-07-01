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
# Fixed-FPR companion to calibrate.py (SECOND PASS, added Jul 2026).
#
# Rationale:
#   calibrate.py fits an F1-optimal threshold PER STAGE. That is the criterion
#   that most flatters the text class (predict-all-forged inflates F1 on the
#   3:1 subset). This companion adds the criterion the field actually deploys:
#   Zhao et al. (2026) and Korshunov et al. (2025, FantasyID) both set the
#   operational threshold on a held-out set so bonafide FPR = 10%, then report
#   on test. Matching it makes the recovery numbers directly commensurable
#   with both papers.
#
# What differs from calibrate.py:
#   - FPR is defined only on the bonafide class, and bonafide is shared across
#     all attack types, so ONE global threshold is fit per calibration draw on
#     that draw's bonafide rows, then applied to EVERY reporting subset
#     (global, face, text, ...). This is the "one operational threshold,
#     reported per attack type" protocol.
#   - Partition, draws, seeds, sizes, and reporting subsets are IDENTICAL to
#     calibrate.py (same PARTITION_SEED, same stratified_draw, same
#     1000*size + r draw seeds), so the F1 and FPR results describe the same
#     calibration sets and can be compared draw-for-draw.
#
# Caveat carried into the write-up:
#   the partition holds ~30 bonafide images; a stratified draw of size 10
#   contains only 2-3 bonafide, so a 10% FPR cut is estimated from ~0.2-0.3
#   expected false positives and is necessarily coarse. The N_REPEATS mean/std
#   exposes this instability honestly. That instability at small calibration
#   sizes is itself a finding about the "as few as 10 images" claim.
#
# This script does NOT touch calibrate.py or its calibration_<model>.json
# outputs. It writes calibration_fpr_<model>.json alongside them.
# ---------------------------------------------------------------------------

PARTITION_SEED = 42
PARTITION_SIZE = 150
CALIB_SIZES = [10, 25, 50]
N_REPEATS = 20
TARGET_FPR = 0.10


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
    Identical to calibrate.py so draws match draw-for-draw."""
    rng = np.random.default_rng(seed)
    groups = {k: sub.index.to_numpy() for k, sub in df.groupby(key)}
    total = len(df)
    raw = {k: n * len(idx) / total for k, idx in groups.items()}
    floor = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = n - sum(floor.values())
    frac_order = sorted(raw, key=lambda k: raw[k] - floor[k], reverse=True)
    for k in frac_order[:remainder]:
        floor[k] += 1
    picked = []
    for k, idx in groups.items():
        take = min(floor[k], len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    return df.loc[picked]


def carve_partition(df):
    """Carve the frozen calibration partition once, stratified. Identical to
    calibrate.py, so the partition and reporting set match exactly."""
    partition = stratified_draw(df, PARTITION_SIZE, PARTITION_SEED)
    reporting = df.drop(index=partition.index)
    return partition, reporting


def fit_threshold_fixed_fpr(draw, target_fpr=TARGET_FPR):
    """One GLOBAL threshold fit on the bonafide rows of a calibration draw.

    Bonafide are attack_type == 'none' (y_true == 0). The threshold is the
    lowest score t such that at most target_fpr of bonafide are flagged
    (p_forged >= t). Concretely: the (1 - target_fpr) empirical quantile of
    the bonafide scores, using 'higher' interpolation so the realised FPR
    does not exceed the target. Returns the threshold, or None if the draw
    has no bonafide (cannot define FPR)."""
    bona = draw[draw["attack_type"] == "none"]["p_forged"].to_numpy()
    if len(bona) == 0:
        return None
    # (1 - fpr) quantile of bonafide scores; 'higher' keeps realised FPR <= target
    t = float(np.quantile(bona, 1.0 - target_fpr, method="higher"))
    return t


def apply_threshold(p_forged, t):
    return (p_forged >= t).astype(int)


def recovery_metrics(y_true, y_pred):
    """Same metric set as calibrate.py: balanced_accuracy and mcc are the
    threshold-fair primaries; f1 is the cautionary secondary."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def static_auc(y_true, p_forged):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, p_forged))


def subset_by_type(df, atk):
    """Reporting subset for one attack type: that type's rows plus all
    bonafide rows. Identical to calibrate.py."""
    if atk == "none":
        return df
    return df[(df["attack_type"] == atk) | (df["attack_type"] == "none")]


def run_recovery_fixed_fpr(calib_partition, reporting, stages, target_fpr=TARGET_FPR):
    """Sweep calibration-set sizes. For each size and repeat, draw ONCE from
    the full partition, fit ONE fixed-FPR threshold on that draw's bonafide,
    then apply it to EVERY stage's reporting subset. Returns a results dict
    keyed by stage, mirroring calibrate.py's shape so a plotter can read both
    files the same way. Also records the realised bonafide FPR on the full
    reporting set per draw, so the write-up can report how close to 10% the
    operating point actually lands."""
    # static AUC + n_report per stage (threshold-independent, same as F1 pass)
    per_stage = {}
    for atk in stages:
        rp = reporting if atk == "global" else subset_by_type(reporting, atk)
        per_stage[atk] = {
            "static_auc": static_auc(rp["y_true"].to_numpy(),
                                     rp["p_forged"].to_numpy()),
            "n_report": int(len(rp)),
            "sizes": {},
        }

    # bonafide rows of the FULL reporting set, for realised-FPR bookkeeping
    report_bona = reporting[reporting["attack_type"] == "none"]["p_forged"].to_numpy()

    for size in CALIB_SIZES:
        if size > len(calib_partition):
            continue
        # accumulate per-stage metric lists across repeats
        stage_runs = {atk: [] for atk in stages}
        realised_fpr = []
        n_thresh = 0
        for r in range(N_REPEATS):
            draw = stratified_draw(calib_partition, size, seed=1000 * size + r)
            t = fit_threshold_fixed_fpr(draw, target_fpr=target_fpr)
            if t is None:
                continue
            n_thresh += 1
            if len(report_bona) > 0:
                realised_fpr.append(float((report_bona >= t).mean()))
            for atk in stages:
                rp = reporting if atk == "global" else subset_by_type(reporting, atk)
                pred = apply_threshold(rp["p_forged"].to_numpy(), t)
                stage_runs[atk].append(
                    recovery_metrics(rp["y_true"].to_numpy(), pred))
        # aggregate
        for atk in stages:
            runs = stage_runs[atk]
            if not runs:
                continue
            agg = {}
            for m in ("balanced_accuracy", "mcc", "f1"):
                vals = np.array([x[m] for x in runs])
                agg[m] = {"mean": float(vals.mean()),
                          "std": float(vals.std())}
            per_stage[atk]["sizes"][size] = agg
        # store realised FPR bookkeeping on the global stage only
        if realised_fpr:
            per_stage["global"].setdefault("realised_fpr", {})[size] = {
                "mean": float(np.mean(realised_fpr)),
                "std": float(np.std(realised_fpr)),
                "n_valid_draws": n_thresh,
            }
    return per_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="path to a frozen scores_<split>_<model>.csv")
    ap.add_argument("--model", default=None,
                    help="model label for the output; defaults to filename stem")
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR,
                    help="bonafide FPR target for the operational threshold")
    args = ap.parse_args()

    target_fpr = args.target_fpr

    df = load_scores(args.scores)
    model_label = args.model or Path(args.scores).stem
    print(f"Loaded {len(df)} scored samples from {args.scores}")
    print(f"Attack-type counts: {df['attack_type'].value_counts().to_dict()}")
    print(f"Target bonafide FPR: {target_fpr:.2f}")

    partition, reporting = carve_partition(df)
    n_bona_part = int((partition["attack_type"] == "none").sum())
    print(f"Calibration partition: {len(partition)} "
          f"(seed {PARTITION_SEED}); of which bonafide: {n_bona_part}")
    print(f"Reporting set: {len(reporting)}")
    if n_bona_part < 10:
        print(f"NOTE: only {n_bona_part} bonafide in partition; the 10% FPR "
              f"cut at small calibration sizes is coarse (this is reported).")

    attack_types = [a for a in sorted(df["attack_type"].unique()) if a != "none"]
    stages = ["global"] + attack_types

    per_stage = run_recovery_fixed_fpr(partition, reporting, stages,
                                       target_fpr=target_fpr)

    results = {
        "model": model_label,
        "scores_file": str(args.scores),
        "criterion": "fixed_fpr",
        "target_fpr": target_fpr,
        "partition_seed": PARTITION_SEED,
        "n_repeats": N_REPEATS,
        "stages": per_stage,
    }

    print(f"\n=== Fixed-FPR recovery (FPR={target_fpr:.2f}, "
          f"mean +/- std over {N_REPEATS} draws) ===")
    for atk in stages:
        d = per_stage[atk]
        auc = d["static_auc"]
        auc_s = f"{auc:.4f}" if auc is not None else "n/a"
        print(f"\n{atk:8s}  static AUC={auc_s}  (n_report={d['n_report']})")
        for size in CALIB_SIZES:
            if size not in d["sizes"]:
                continue
            m = d["sizes"][size]
            line = (f"  size {size:3d}  "
                    f"balacc {m['balanced_accuracy']['mean']:.3f}"
                    f"+/-{m['balanced_accuracy']['std']:.3f}  "
                    f"mcc {m['mcc']['mean']:+.3f}+/-{m['mcc']['std']:.3f}  "
                    f"f1 {m['f1']['mean']:.3f}+/-{m['f1']['std']:.3f}")
            if atk == "global" and "realised_fpr" in d and size in d["realised_fpr"]:
                rf = d["realised_fpr"][size]
                line += (f"   [realised FPR "
                         f"{rf['mean']:.3f}+/-{rf['std']:.3f}]")
            print(line)

    out_path = Path(args.scores).parent / f"calibration_fpr_{model_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
