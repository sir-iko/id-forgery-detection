"""Select Grad-CAM exemplar images from the ResNet50 score CSV.

Picks three test images that tell the two-failure-mode story:
  - caught_face : a face forgery the model is most confident is forged
                  (attack=face, y_true=1, highest p_forged) -> model attends
                  to the right region when it works.
  - missed_text : a text forgery the model is most confident is genuine
                  (attack=text, y_true=1, lowest p_forged) -> the inversion
                  at its starkest; model ranks a forgery as genuine.
  - genuine     : a bonafide doc confidently and correctly called genuine
                  (y_true=0, lowest p_forged) -> contrast.

Resolves each row's idx back to a file path via FantasyIDDataset.samples,
and writes a manifest (CSV) the GPU Grad-CAM job reads. Login node only:
reads the frozen score CSV and the dataset index, no torch, no model.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import FantasyIDDataset
from src.transforms import get_transforms


def main():
    parser = argparse.ArgumentParser(
        description="Select Grad-CAM exemplars from the ResNet50 score CSV."
    )
    parser.add_argument("--scores",
                        default="checkpoints/scores_test_resnet50.csv",
                        help="Frozen per-sample score CSV for ResNet50.")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config giving data.root and split, for path lookup.")
    parser.add_argument("--split", default="test",
                        help="Dataset split the scores were computed on.")
    parser.add_argument("--out", default="checkpoints/gradcam_exemplars.csv",
                        help="Manifest the GPU Grad-CAM job will read.")
    args = parser.parse_args()

    print("=== Loading scores ===")
    df = pd.read_csv(args.scores)
    print(f"{len(df)} rows")

    print("=== Selecting exemplars ===")
    face = df[(df["attack_type"] == "face") & (df["y_true"] == 1)]
    text = df[(df["attack_type"] == "text") & (df["y_true"] == 1)]
    genuine = df[df["y_true"] == 0]

    caught_face = face.loc[face["p_forged"].idxmax()]
    missed_text = text.loc[text["p_forged"].idxmin()]
    genuine_row = genuine.loc[genuine["p_forged"].idxmin()]

    picks = {
        "caught_face": caught_face,
        "missed_text": missed_text,
        "genuine": genuine_row,
    }
    for name, row in picks.items():
        print(f"{name}: idx={int(row['idx'])}, "
              f"p_forged={row['p_forged']:.4f}, attack={row['attack_type']}")

    print("=== Resolving paths via dataset ===")
    cfg = load_config(args.config)
    tfm = get_transforms(args.split)
    dataset = FantasyIDDataset(
        root=cfg["data"]["root"], split=args.split, transform=tfm
    )

    rows = []
    for name, row in picks.items():
        idx = int(row["idx"])
        rel_path = dataset.samples[idx][0]
        rows.append({
            "name": name,
            "idx": idx,
            "rel_path": rel_path,
            "p_forged": float(row["p_forged"]),
            "attack_type": row["attack_type"],
            "y_true": int(row["y_true"]),
        })
        print(f"{name}: {rel_path}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
