import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix,
)

from src.data import FantasyIDDataset
from src.models import build_model
from src.transforms import get_transforms


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Run inference and return arrays of true labels, predicted labels,
    and predicted probability of the forged class (label 1)."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(forged)
        preds = logits.argmax(1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def overall_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    # per-class precision/recall/f1 (class 0 = bonafide, 1 = forged)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # AUCs need both classes present in y_true
    both_classes = len(np.unique(y_true)) == 2
    roc = roc_auc_score(y_true, y_prob) if both_classes else None
    pr = average_precision_score(y_true, y_prob) if both_classes else None
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(acc),
        "bonafide": {"precision": float(p[0]), "recall": float(r[0]),
                     "f1": float(f1[0]), "support": int(support[0])},
        "forged": {"precision": float(p[1]), "recall": float(r[1]),
                   "f1": float(f1[1]), "support": int(support[1])},
        "macro": {"precision": float(p_macro), "recall": float(r_macro),
                  "f1": float(f1_macro)},
        "roc_auc": float(roc) if roc is not None else None,
        "pr_auc": float(pr) if pr is not None else None,
        "mcc": float(mcc),
        "confusion_matrix": cm,  # rows = true [bonafide, forged], cols = pred
    }


def per_attack_type(dataset, y_true, y_pred):
    """Recall per attack_type. Each forged sample carries an attack_type;
    bonafide samples have attack_type 'none'. Recall is the meaningful
    per-type number: of the samples of this type, how many were caught."""
    attack_types = [s[2] for s in dataset.samples]
    breakdown = {}
    for atk in sorted(set(attack_types)):
        idx = [i for i, a in enumerate(attack_types) if a == atk]
        if not idx:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        # For forged types, "correct" = predicted forged (1).
        # For 'none' (bonafide), "correct" = predicted bonafide (0).
        if atk == "none":
            correct = int((yp == 0).sum())
        else:
            correct = int((yp == 1).sum())
        breakdown[atk] = {
            "n": len(idx),
            "correct": correct,
            "recall": correct / len(idx),
        }
    return breakdown


def per_attack_type_auc(dataset, y_true, y_prob):
    """Per-attack-type discriminability against the bonafide pool.

    Each forged attack type is single-class (all is_attack=1), so AUC is
    computed by pairing that type's samples with ALL bonafide samples and
    scoring P(forged). Answers whether the model can separate THIS attack
    type from genuine documents. Near or below 0.5 means a representational
    failure that threshold calibration cannot fix."""
    attack_types = np.array([s[2] for s in dataset.samples])
    bona_idx = np.where(attack_types == "none")[0]
    breakdown = {}
    for atk in sorted(set(attack_types.tolist())):
        if atk == "none":
            continue
        pos_idx = np.where(attack_types == atk)[0]
        idx = np.concatenate([bona_idx, pos_idx])
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(set(yt.tolist())) < 2:
            breakdown[atk] = {"n_pos": int(len(pos_idx)),
                              "n_bonafide": int(len(bona_idx)),
                              "roc_auc": None, "pr_auc": None}
            continue
        breakdown[atk] = {
            "n_pos": int(len(pos_idx)),
            "n_bonafide": int(len(bona_idx)),
            "roc_auc": float(roc_auc_score(yt, yp)),
            "pr_auc": float(average_precision_score(yt, yp)),
        }
    return breakdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to a *_best.pt checkpoint")
    ap.add_argument("--split", default="test", choices=["test", "train"],
                    help="which FantasyID split to evaluate on")
    ap.add_argument("--data-root", default=None,
                    help="override data root; defaults to checkpoint config")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override DataLoader workers; 0 avoids macOS segfaults")
    args = ap.parse_args()

    device = select_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch')}, "
          f"val_loss {ckpt.get('val_loss'):.4f}")

    data_root = args.data_root or cfg["data"]["root"]
    tf = get_transforms(args.split, image_size=cfg["data"]["image_size"])
    dataset = FantasyIDDataset(data_root, split=args.split, transform=tf)
    nw = args.num_workers if args.num_workers is not None else cfg["data"]["num_workers"]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=nw)
    print(f"Evaluating on {args.split} split: {len(dataset)} samples")

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    y_true, y_pred, y_prob = collect_predictions(model, loader, device)

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "model": cfg["model"]["name"],
        "overall": overall_metrics(y_true, y_pred, y_prob),
        "per_attack_type": per_attack_type(dataset, y_true, y_pred),
        "per_attack_type_auc": per_attack_type_auc(dataset, y_true, y_prob),
    }

    # Print a readable summary
    o = results["overall"]
    print("\n=== Overall ===")
    print(f"Accuracy: {o['accuracy']:.4f}   MCC: {o['mcc']:.4f}")
    print(f"Forged   P {o['forged']['precision']:.4f}  "
          f"R {o['forged']['recall']:.4f}  F1 {o['forged']['f1']:.4f}")
    print(f"Bonafide P {o['bonafide']['precision']:.4f}  "
          f"R {o['bonafide']['recall']:.4f}  F1 {o['bonafide']['f1']:.4f}")
    if o["roc_auc"] is not None:
        print(f"ROC-AUC: {o['roc_auc']:.4f}   PR-AUC: {o['pr_auc']:.4f}")
    print(f"Confusion (rows=true, cols=pred): {o['confusion_matrix']}")

    print("\n=== Per attack type (recall) ===")
    for atk, d in results["per_attack_type"].items():
        print(f"{atk:12s}  n={d['n']:4d}  recall={d['recall']:.4f}")

    print("\n=== Per attack type (AUC vs bonafide pool) ===")
    for atk, d in results["per_attack_type_auc"].items():
        roc = d["roc_auc"]
        pr = d["pr_auc"]
        roc_s = f"{roc:.4f}" if roc is not None else "n/a"
        pr_s = f"{pr:.4f}" if pr is not None else "n/a"
        print(f"{atk:12s}  n_pos={d['n_pos']:4d}  "
              f"n_bonafide={d['n_bonafide']:4d}  "
              f"ROC-AUC={roc_s}  PR-AUC={pr_s}")

    # Save JSON next to the checkpoint
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
