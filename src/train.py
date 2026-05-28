import argparse

import torch

from src.config import load_config
from src.models import build_model
from src.utils.logging import RunLogger
from src.utils.seed import set_seed
# from src.data import ForgeryDataset  # enable once implemented


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, GPUs visible: {torch.cuda.device_count()}")

    logger = RunLogger(cfg["output"]["dir"], cfg["output"]["run_name"], config=cfg)

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    # TODO: build datasets and DataLoaders, then the train/val loop with
    # 5-fold CV, BCE/focal loss, Adam, and early stopping (see proposal).

    # Smoke test: confirm a forward pass runs on the GPU before wiring up data.
    dummy = torch.randn(
        2, 3, cfg["data"]["image_size"], cfg["data"]["image_size"]
    ).to(device)
    out = model(dummy)
    print("Forward pass OK, output shape:", tuple(out.shape))
    logger.log(step=0, smoke_output_mean=float(out.float().mean()))

    logger.close()


if __name__ == "__main__":
    main()
