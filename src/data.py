import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


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
