"""Grad-CAM heatmaps for ResNet50 on the three selected exemplars.

Targets the FORGED class for every image (fixed target), so each heatmap
answers the same question: where does the model look for evidence of
forgery? This holds the question constant across the working case
(caught_face) and the failing case (missed_text), and because missed_text
and genuine are the same base document (text-forged vs clean), their CAMs
form a single-variable comparison.

Loads the ResNet50 checkpoint the same way evaluate.py does (rebuild from
ckpt["config"], load ckpt["model_state"]). Reads the exemplar manifest from
the selector step. GPU job: runs gradients through the real model.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.models import build_model
from src.transforms import get_transforms

# Class index 1 is "forged" (p_forged is softmax prob of class 1).
FORGED_CLASS = 1


def main():
    parser = argparse.ArgumentParser(
        description="Grad-CAM on ResNet50 exemplars, forged-class target."
    )
    parser.add_argument("--checkpoint",
                        default="checkpoints/resnet50_baseline_best.pt")
    parser.add_argument("--manifest",
                        default="checkpoints/gradcam_exemplars.csv")
    parser.add_argument("--data-root", default=None,
                        help="Override data root; defaults to ckpt config.")
    parser.add_argument("--out-dir", default="results/gradcam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuda {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")

    print("=== Loading checkpoint ===")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    print(f"epoch {ckpt.get('epoch')}, val_loss {ckpt.get('val_loss'):.4f}")

    model = build_model(
        cfg["model"]["name"],
        num_classes=cfg["model"]["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    image_size = cfg["data"]["image_size"]
    data_root = Path(args.data_root or cfg["data"]["root"])
    tf = get_transforms("test", image_size=image_size)

    # ResNet50 last conv block.
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(FORGED_CLASS)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Grad-CAM overlays ===")
    manifest = pd.read_csv(args.manifest)
    for _, row in manifest.iterrows():
        name = row["name"]
        rel_path = row["rel_path"]
        img_path = data_root / rel_path

        pil = Image.open(img_path).convert("RGB")

        # Normalised tensor for the model.
        input_tensor = tf(pil).unsqueeze(0).to(device)

        # Unnormalised [0,1] RGB at model resolution for the overlay background.
        rgb = np.array(pil.resize((image_size, image_size))).astype(np.float32) / 255.0

        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

        # Report the model's actual forged probability for the caption.
        with torch.no_grad():
            logits = model(input_tensor)
            p_forged = torch.softmax(logits, dim=1)[0, FORGED_CLASS].item()

        out_path = out_dir / f"gradcam_{name}.png"
        Image.fromarray(overlay).save(out_path)
        print(f"{name}: p_forged={p_forged:.4f} (manifest {row['p_forged']:.4f}), "
              f"saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
