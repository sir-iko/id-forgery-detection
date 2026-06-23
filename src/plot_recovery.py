import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Models in a fixed display order, each with a stable colour.
MODEL_ORDER = ["resnet50", "densenet121", "vit"]
MODEL_LABEL = {"resnet50": "ResNet50", "densenet121": "DenseNet121", "vit": "ViT-B/16"}
MODEL_COLOR = {"resnet50": "#1f77b4", "densenet121": "#d62728", "vit": "#2ca02c"}

SIZES = [10, 25, 50]


def load(json_dir, model):
    p = Path(json_dir) / f"calibration_{model}.json"
    with open(p) as f:
        return json.load(f)


def series(stage, metric):
    """Return (sizes_present, means, stds) for a metric in a stage, only at
    calibration-set sizes that actually have data (face has no size 50)."""
    xs, ms, ss = [], [], []
    for s in SIZES:
        key = str(s)
        if key in stage["sizes"]:
            xs.append(s)
            ms.append(stage["sizes"][key][metric]["mean"])
            ss.append(stage["sizes"][key][metric]["std"])
    return np.array(xs), np.array(ms), np.array(ss)


def plot_panel(ax, all_data, attack, metric, show_chance=False):
    """One cell: a single metric for one attack class, three model lines with
    thin capped error bars (std), no shaded bands."""
    for i, model in enumerate(MODEL_ORDER):
        stage = all_data[model]["stages"][attack]
        c = MODEL_COLOR[model]
        x, m, s = series(stage, metric)
        # tiny horizontal offset per model so error bars don't sit on top of
        # each other where the lines coincide (e.g. text balanced accuracy)
        dx = (i - 1) * 0.6
        ax.errorbar(x + dx, m, yerr=s, color=c, lw=1.8, marker="o", ms=5,
                    capsize=3, elinewidth=1.0, capthick=1.0, zorder=3)

    if show_chance:
        ax.axhline(0.5, color="0.45", lw=1.0, ls=":", zorder=0)
        ax.text(SIZES[0] - 2.5, 0.505, "chance", fontsize=8,
                color="0.45", ha="left", va="bottom")

    ax.set_xticks(SIZES)
    ax.set_xlim(SIZES[0] - 4, SIZES[-1] + 4)
    ax.grid(True, axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default="checkpoints",
                    help="directory holding calibration_<model>.json files")
    ap.add_argument("--out", default="results/recovery_curves.png")
    args = ap.parse_args()

    all_data = {m: load(args.json_dir, m) for m in MODEL_ORDER}

    # 2x2: rows = metric (balacc top, F1 bottom), cols = attack (face, text).
    # Share x down each column; share y across each row.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6),
                             sharex="col", sharey="row")
    (ax_bf, ax_bt), (ax_ff, ax_ft) = axes

    # top row: balanced accuracy (chance line shown, the primary metric)
    plot_panel(ax_bf, all_data, "face", "balanced_accuracy", show_chance=True)
    plot_panel(ax_bt, all_data, "text", "balanced_accuracy", show_chance=True)
    # bottom row: F1 (the cautionary metric)
    plot_panel(ax_ff, all_data, "face", "f1")
    plot_panel(ax_ft, all_data, "text", "f1")

    # row y-limits: balacc spans the contrast, F1 sits high
    ax_bf.set_ylim(0.38, 0.88)
    ax_ff.set_ylim(0.45, 0.95)

    # column headers (attack class) and a one-word verdict
    ax_bf.set_title("Face manipulation\n(separable, AUC ~ 0.78-0.92)",
                    fontsize=10.5)
    ax_bt.set_title("Text manipulation\n(inverted, AUC ~ 0.14-0.20)",
                    fontsize=10.5)

    # row labels via y-axis titles
    ax_bf.set_ylabel("Balanced accuracy\n(threshold-fair)", fontsize=10)
    ax_ff.set_ylabel("F1\n(inflated by imbalance)", fontsize=10)
    ax_ff.set_xlabel("Calibration-set size")
    ax_ft.set_xlabel("Calibration-set size")

    # single model legend, placed in the cell with the most empty space
    # (bottom-right F1/text sits high, so put it low there)
    model_handles = [Line2D([0], [0], color=MODEL_COLOR[m], lw=2.0,
                            marker="o", ms=5, label=MODEL_LABEL[m])
                     for m in MODEL_ORDER]
    ax_bt.legend(handles=model_handles, loc="upper left", fontsize=8.5,
                 frameon=False, title="Architecture")

    fig.suptitle(
        "Calibration recovers the face class but not the text class",
        fontsize=13, y=0.99)
    fig.text(0.5, 0.945,
             "Mean +/- std over 20 stratified draws (FantasyID test set). "
             "Top row is threshold-fair and tells the truth; bottom-row F1 "
             "rises for text too, which is the trap.",
             ha="center", fontsize=8.5, color="0.35")
    fig.text(0.5, 0.012,
             "Face has no size-50 point: only 150 face samples exist, so a "
             "stratified draw of 50 cannot be filled.",
             ha="center", fontsize=7.5, color="0.45")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out}")
    print(f"Saved: {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
