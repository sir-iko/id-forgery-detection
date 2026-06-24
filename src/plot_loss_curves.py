"""Plot train-vs-val loss curves for the three baseline runs on shared axes.

Reads per-epoch metrics.csv from each run folder, draws two panels
(train loss, val loss), one coloured line per model, each line ending at
its own run length. Marks the best-checkpoint epoch (argmin val_loss) on
each validation curve.

Reads frozen metrics.csv files only. Does not touch torch or any model.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Canonical run folders, keyed by model. The 8 Jun ResNet smoke run and the
# dead 12 Jun ViT run (lr=1e-4 stall) are deliberately excluded.
RUNS = {
    "ResNet50": "resnet50_baseline_20260609_205558",
    "DenseNet121": "densenet121_baseline_20260611_105734",
    "ViT-B/16": "vit_baseline_20260617_112152",
}

COLOURS = {
    "ResNet50": "#1f77b4",
    "DenseNet121": "#d62728",
    "ViT-B/16": "#2ca02c",
}


def main():
    parser = argparse.ArgumentParser(
        description="Plot train/val loss curves for the three baseline runs."
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory holding the run folders.")
    parser.add_argument("--out", default="results/loss_curves.png",
                        help="Output PNG path (a .pdf sibling is also written).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    print("=== Loading metrics ===")
    data = {}
    for model, folder in RUNS.items():
        csv_path = results_dir / folder / "metrics.csv"
        df = pd.read_csv(csv_path)
        data[model] = df
        best_epoch = int(df["val_loss"].idxmin())
        print(f"{model}: {len(df)} epochs, best val_loss at epoch {best_epoch} "
              f"({df['val_loss'].min():.4f})")

    print("=== Plotting ===")
    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for model, df in data.items():
        colour = COLOURS[model]
        ax_train.plot(df["step"], df["train_loss"], color=colour,
                      label=model, linewidth=1.8)
        ax_val.plot(df["step"], df["val_loss"], color=colour,
                    label=model, linewidth=1.8)
        best = int(df["val_loss"].idxmin())
        ax_val.scatter(df["step"].iloc[best], df["val_loss"].iloc[best],
                       color=colour, s=60, zorder=5, edgecolor="black",
                       linewidth=0.8)

    ax_train.set_title("Training loss")
    ax_val.set_title("Validation loss")
    for ax in (ax_train, ax_val):
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    ax_train.set_ylabel("Loss")
    ax_val.legend(title="Model", frameon=False)

    fig.tight_layout()

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")

    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
