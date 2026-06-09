import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import FantasyIDDataset, stratified_train_val_split, stratified_group_train_val_split, label_counts
from src.models import build_model
from src.transforms import get_transforms
from src.utils.logging import RunLogger
from src.utils.seed import set_seed


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def class_weights_from_counts(counts, num_classes, device):
    """Inverse-frequency weights, normalised to mean 1.0."""
    total = sum(counts.values())
    weights = []
    for c in range(num_classes):
        n = counts.get(c, 0)
        weights.append(total / (num_classes * n) if n > 0 else 0.0)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.mean()


def run_epoch(model, loader, criterion, device, optimizer=None, max_batches=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, correct, seen = 0.0, 0, 0

    for i, (imgs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += labels.size(0)

    return total_loss / seen, correct / seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap batches per epoch for a quick smoke run")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the epoch count in the config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = select_device()
    print(f"Device: {device}")

    logger = RunLogger(cfg["output"]["dir"], cfg["output"]["run_name"], config=cfg)

    # Data
    tf = get_transforms("train", image_size=cfg["data"]["image_size"])
    full = FantasyIDDataset(cfg["data"]["root"], split="train", transform=tf)
    # Pick splitter by dataset: group-aware (no template leakage) when source

    # lineage is available (synthetic MIDV set), plain stratified otherwise

    # (FantasyID, which carries no base_id).

    has_groups = hasattr(full, "groups") and all(g[0] is not None for g in full.groups)

    if has_groups:

        print("Split: group-aware (base_id present)")

        train_set, val_set = stratified_group_train_val_split(full, val_frac=0.2, seed=cfg["seed"])

    else:

        print("Split: plain stratified (no base_id)")

        train_set, val_set = stratified_train_val_split(full, val_frac=0.2, seed=cfg["seed"])
    print(f"Train: {len(train_set)}  Val: {len(val_set)}")
    print(f"Train label counts: {dict(label_counts(train_set))}")

    train_loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # Model, loss, optimizer
    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    weights = class_weights_from_counts(
        label_counts(train_set), cfg["model"]["num_classes"], device
    )
    print(f"Class weights: {weights.tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    # Train loop with best-checkpoint saving
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    n_epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    for epoch in range(n_epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, max_batches=args.max_batches,
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, device,
            optimizer=None, max_batches=args.max_batches,
        )
        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        logger.log(step=epoch, train_loss=tr_loss, train_acc=tr_acc,
                   val_loss=va_loss, val_acc=va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": va_loss, "config": cfg},
                ckpt_dir / f"{cfg['output']['run_name']}_best.pt",
            )
            print(f"  saved best checkpoint (val loss {va_loss:.4f})")

    logger.close()


if __name__ == "__main__":
    main()
