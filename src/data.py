import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset


class FantasyIDDataset(Dataset):
    """FantasyID dataset loader.

    The dataset's own train.csv / test.csv files are the source of truth.
    Each CSV row has (path, is_attack, attack_type). is_attack becomes the
    binary label (0 = bonafide, 1 = forged). attack_type is kept on each
    sample for later per-attack metric breakdowns.

    Args:
        root: path to the FantasyID folder (the one containing train.csv).
        split: 'train' or 'test'.
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

        self.samples = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rel = row["path"]
                is_attack = row["is_attack"].strip().lower() == "true"
                self.samples.append((rel, 1 if is_attack else 0, row["attack_type"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, _attack_type = self.samples[idx]
        img = Image.open(self.root / rel_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def stratified_train_val_split(dataset, val_frac=0.2, seed=42):
    """Split a FantasyIDDataset into train and validation Subsets.

    Stratifies on (label, attack_type) so both halves preserve the class
    imbalance AND the attack-type composition of the original set.
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


def label_counts(dataset_or_subset):
    """Return a Counter of labels in a FantasyIDDataset or a Subset of one."""
    if isinstance(dataset_or_subset, Subset):
        base = dataset_or_subset.dataset
        return Counter(base.samples[i][1] for i in dataset_or_subset.indices)
    return Counter(s[1] for s in dataset_or_subset.samples)
