"""Build a contact sheet of synthetic forgeries for visual QA.

Picks a few forged samples from synth.csv, draws the mask outline on each forged
image, and stacks each (annotated image | mask) pair into one PNG you can open in
the OnDemand file browser. Lets you confirm the splice lands in the right field
and the mask matches before scaling generation to all document types.
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="synth output root (has images/, masks/, synth.csv)")
    p.add_argument("--out", default="contact_sheet.png")
    p.add_argument("--n", type=int, default=6, help="number of forged samples")
    p.add_argument("--attack", default="", help="filter to one attack type (text/face/face_text)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=500, help="display width per panel")
    args = p.parse_args()

    root = Path(args.root)
    rows = [r for r in csv.DictReader(open(root / "synth.csv")) if r["is_attack"] == "1"]
    if args.attack:
        rows = [r for r in rows if r["attack_type"] == args.attack]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    rows = rows[:args.n]

    panels = []
    for r in rows:
        img = cv2.imread(str(root / r["path"]))
        mask = cv2.imread(str(root / "masks" / os.path.basename(r["path"])), 0)
        if img is None or mask is None:
            continue
        # outline the tamper region on the image
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        annotated = img.copy()
        cv2.drawContours(annotated, cnts, -1, (0, 255, 0), 3)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        def fit(x):
            h, w = x.shape[:2]
            return cv2.resize(x, (args.width, int(h * args.width / w)))

        a, m = fit(annotated), fit(mask_bgr)
        h = max(a.shape[0], m.shape[0])
        a = cv2.copyMakeBorder(a, 0, h - a.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        m = cv2.copyMakeBorder(m, 0, h - m.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        pair = np.hstack([a, m])
        label = f"{os.path.basename(r['path'])}  [{r['attack_type']}]"
        pair = cv2.copyMakeBorder(pair, 28, 4, 4, 4, cv2.BORDER_CONSTANT)
        cv2.putText(pair, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        panels.append(pair)

    if not panels:
        print("no forged samples matched")
        return
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 0, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    sheet = np.vstack(panels)
    cv2.imwrite(args.out, sheet)
    print(f"wrote {args.out}  ({len(panels)} panels, {sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
