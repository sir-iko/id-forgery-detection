from torch.utils.data import Dataset


class ForgeryDataset(Dataset):
    """TODO: implement for MIDV-2020 / DocTamper / FantasyID.

    Keep raw downloads in data/raw and processed tensors in data/processed.
    __getitem__ should return (image_tensor, label) with 0=real, 1=forged.
    """

    def __init__(self, root, split="train", transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []  # TODO: populate with (path, label) pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raise NotImplementedError
