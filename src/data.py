"""Dataset loader and split helpers.

Two datasets flow through this file:
  * FantasyID: independent real-manipulation test set. Rows are independent
    samples; split by stratified sampling on (label, attack_type).
  * MIDV-2020 synthetic set: multiple forgeries share a source template
    (base_id). Splitting by row would leak template identity across train/val,
    so it is split by stratified GROUP sampling on (base_id, doc_type), which
    keeps every template wholly on one side while preserving class balance.

FantasyID CSVs label is_attack as True/False text; the synthetic manifest writes
0/1 integers. The dataset accepts both.
"""

import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from torch.utils.data import Dataset, Subset


def _parse_is_attack(value):
    """Accept 'true'/'false' (FantasyID) and '1'/'0' (synthetic manifest)."""
    v = str(value).strip().lower()
    if v in ("true", "1"):
        return 1
    if v in ("false", "0"):
        return 0
    raise ValueError(f"unrecognised is_attack value: {value!r}")


class FantasyIDDataset(Dataset):
    """CSV-driven ID-document dataset.

    Each row has (path, is_attack, attack_type). is_attack becomes the binary
    label (0 = bonafide, 1 = forged). attack_type is kept per sample for
    per-attack metric breakdowns. If the CSV also has base_id and doc_type
    (the synthetic manifest does; FantasyID does not), they are captured for
    group-aware splitting; otherwise they are None.

    Args:
        root: path to the folder containing <split>.csv.
        split: csv basename without extension (e.g. 'train', 'test', 'synth').
        transform: optional torchvision transform applied to the PIL image.
    """

    def __init__(self, root, split="train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        csv_path = self.root / f"{split}.csv"
        if not csv_path.exists():
            available = [p.name for p in self.root.glob("*.csv")]
            raise FileNotFoundError(f"Expected {csv_path}. CSVs found: {available}")

        self.samples = []   # (rel_path, label, attack_type)
        self.groups = []    # (base_id, doc_type) or (None, None)
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rel = row["path"]
                label = _parse_is_attack(row["is_attack"])
                self.samples.append((rel, label, row["attack_type"]))
                self.groups.append((row.get("base_id"), row.get("doc_type")))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, _attack_type = self.samples[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stratified_train_val_split(dataset, val_frac=0.2, seed=42):
    """Row-wise stratified split on (label, attack_type).

    For datasets whose rows are independent samples (FantasyID). Do NOT use this
    on the synthetic set: it would leak template identity across the split.
    """
    stratify_keys = [f"{lab}_{atk}" for _, lab, atk in dataset.samples]
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_frac,
        random_state=seed,
        stratify=stratify_keys,
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def stratified_group_train_val_split(dataset, val_frac=0.2, seed=42):
    """Group-aware stratified split for the synthetic set.

    Groups by (base_id, doc_type) so every forgery and bonafide derived from one
    source template stays wholly on one side of the split (no leakage), while
    stratifying on (label, attack_type) to preserve class balance. Implemented
    via StratifiedGroupKFold with n_splits chosen from val_frac (e.g. 0.2 -> 5
    folds, take one as val).

    Requires base_id to be present (synthetic manifest). Raises if it is not.
    """
    if any(g[0] is None for g in dataset.groups):
        raise ValueError(
            "stratified_group_train_val_split needs base_id on every row; "
            "this looks like a FantasyID set. Use stratified_train_val_split."
        )

    groups = [f"{b}_{d}" for (b, d) in dataset.groups]
    y = [f"{lab}_{atk}" for _, lab, atk in dataset.samples]
    X = list(range(len(dataset)))

    n_splits = max(2, round(1 / val_frac))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(X, y, groups))
    return Subset(dataset, list(train_idx)), Subset(dataset, list(val_idx))


def label_counts(dataset_or_subset):
    """Return a Counter of labels in a dataset or a Subset of one."""
    if isinstance(dataset_or_subset, Subset):
        base = dataset_or_subset.dataset
        return Counter(base.samples[i][1] for i in dataset_or_subset.indices)
    return Counter(s[1] for s in dataset_or_subset.samples)
