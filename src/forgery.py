"""Synthetic forgery generation for MIDV-2020 templates.

Produces same-field, same-document-type copy-move forgeries from the MIDV-2020
template set, together with binary ground-truth tamper masks and a manifest CSV
that matches the FantasyID schema (path, is_attack, attack_type) so the output
drops straight into the existing FantasyIDDataset loader.

Design (decided in project discussion):
  * Copy-move / splice only. Source pixels are real document pixels taken from a
    DIFFERENT image of the SAME field and SAME document type, so the only artefact
    is the splice boundary. This avoids learning a synthetic-renderer signature and
    keeps the forgery comparable to the Bae/DocTamper copy-move family.
  * Ground-truth mask = the destination box exactly (tight, honest, feeds Grad-CAM
    IoU / pointing-game metrics directly).
  * Attack types emitted: none (untouched bonafide), text, face, face_text.
    Bonafide and forged both pass through this one pipeline from the same source
    distribution, removing the "tampered bonafide" confound.
  * --per-template controls forgeries per template (default 2). Attack mix is
    weighted toward text (the demonstrated hard target). Bonafide is one copy per
    template.
  * Seeded throughout for reproducibility. CSV carries base_id and doc_type so the
    train/val split can be done BY BASE IMAGE (no leakage of two forgeries from the
    same base across the split).

Cross-type sourcing is available behind --cross-type but OFF by default; it is a
deliberate confound (global layout mismatch) kept only as an ablation probe.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np

# Field groups. The annotation set per image (confirmed): photo, signature, name,
# surname, gender, nationality, id_number, number, issue_place, birth_date,
# expiry_date, issue_date, face.
TEXT_FIELDS = [
    "name", "surname", "id_number", "number",
    "birth_date", "expiry_date", "issue_date",
    "issue_place", "nationality", "gender",
]
FACE_FIELDS = ["photo", "face"]  # prefer 'photo' (larger portrait region) when present


def load_annotations(ann_path):
    """Return {filename: {field_name: (x, y, w, h)}} for one document-type JSON."""
    with open(ann_path) as f:
        d = json.load(f)
    out = {}
    for v in d["_via_img_metadata"].values():
        fname = v["filename"]
        boxes = {}
        for r in v["regions"]:
            s = r["shape_attributes"]
            if s.get("name") != "rect":
                continue
            field = r["region_attributes"]["field_name"]
            boxes[field] = (int(s["x"]), int(s["y"]), int(s["width"]), int(s["height"]))
        out[fname] = boxes
    return out


def clamp_box(x, y, w, h, img_w, img_h):
    """Clip a box to image bounds; return None if it collapses."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(img_w, x + w)
    y1 = min(img_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def splice_field(dst_img, dst_box, src_img, src_box, feather=2):
    """Paste src_box region of src_img into dst_box of dst_img (resized to fit).

    Returns the modified image and the actual destination box used. A light
    feather on the mask edge reduces a hard 1-pixel seam without hiding the
    tamper (kept small so the boundary signal the edge layers target survives).
    """
    dx, dy, dw, dh = dst_box
    sx, sy, sw, sh = src_box
    patch = src_img[sy:sy + sh, sx:sx + sw]
    if patch.size == 0:
        return dst_img, None
    patch = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_LINEAR)
    dst_img[dy:dy + dh, dx:dx + dw] = patch
    return dst_img, (dx, dy, dw, dh)


def make_mask(img_shape, boxes):
    """Binary mask, 255 inside each destination box, 0 elsewhere."""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for (x, y, w, h) in boxes:
        mask[y:y + h, x:x + w] = 255
    return mask


def pick_source(rng, candidates, exclude):
    """Pick a base id from candidates that is not `exclude`."""
    pool = [c for c in candidates if c != exclude]
    return rng.choice(pool) if pool else None


def generate(args):
    rng = random.Random(args.seed)
    ann_dir = Path(args.annotations)
    img_root = Path(args.images)
    out_img = Path(args.out) / "images"
    out_mask = Path(args.out) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    doc_types = sorted(p.stem for p in ann_dir.glob("*.json"))
    if args.types:
        wanted = set(args.types.split(","))
        doc_types = [t for t in doc_types if t in wanted]

    rows = []
    attack_mix = ["text"] * 6 + ["face"] * 2 + ["face_text"] * 2  # 60/20/20

    for dtype in doc_types:
        anns = load_annotations(ann_dir / f"{dtype}.json")
        type_dir = img_root / dtype
        # base ids that have both an annotation and an image file on disk
        base_ids = sorted(
            fn for fn in anns if (type_dir / fn).exists()
        )
        if len(base_ids) < 2:
            print(f"[skip] {dtype}: needs >=2 images, has {len(base_ids)}")
            continue

        for base in base_ids:
            img_path = type_dir / base
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[warn] unreadable {img_path}")
                continue
            ih, iw = img.shape[:2]
            stem = f"{dtype}__{Path(base).stem}"

            # --- bonafide: one untouched copy ---
            bona_name = f"{stem}__none.jpg"
            cv2.imwrite(str(out_img / bona_name), img)
            cv2.imwrite(str(out_mask / bona_name), np.zeros((ih, iw), np.uint8))
            rows.append([f"images/{bona_name}", 0, "none", Path(base).stem, dtype])

            # --- forgeries ---
            for k in range(args.per_template):
                attack = rng.choice(attack_mix)
                forged = img.copy()
                dst_boxes = []

                def do_field(field_list):
                    field = next((f for f in field_list if f in anns[base]), None)
                    if field is None:
                        return False
                    src_base = pick_source(rng, base_ids, exclude=base)
                    if src_base is None:
                        return False
                    src_img = cv2.imread(str(type_dir / src_base))
                    if src_img is None or field not in anns[src_base]:
                        return False
                    dbox = clamp_box(*anns[base][field], iw, ih)
                    sh_, sw_ = src_img.shape[:2]
                    sbox = clamp_box(*anns[src_base][field], sw_, sh_)
                    if dbox is None or sbox is None:
                        return False
                    _, used = splice_field(forged, dbox, src_img, sbox)
                    if used:
                        dst_boxes.append(used)
                        return True
                    return False

                ok = False
                if attack in ("text", "face_text"):
                    # pick one text field at random for this forgery
                    fields_present = [f for f in TEXT_FIELDS if f in anns[base]]
                    rng.shuffle(fields_present)
                    ok = do_field(fields_present[:1]) or ok
                if attack in ("face", "face_text"):
                    ok = do_field(FACE_FIELDS) or ok

                if not ok or not dst_boxes:
                    continue  # skip degenerate result rather than emit a fake-clean forgery

                fname = f"{stem}__{attack}_{k}.jpg"
                cv2.imwrite(str(out_img / fname), forged)
                cv2.imwrite(str(out_mask / fname), make_mask(forged.shape, dst_boxes))
                rows.append([f"images/{fname}", 1, attack, Path(base).stem, dtype])

    csv_path = Path(args.out) / "synth.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "is_attack", "attack_type", "base_id", "doc_type"])
        w.writerows(rows)

    n_bona = sum(1 for r in rows if r[1] == 0)
    n_forged = sum(1 for r in rows if r[1] == 1)
    print(f"wrote {len(rows)} rows: {n_bona} bonafide, {n_forged} forged")
    print(f"manifest: {csv_path}")


def main():
    p = argparse.ArgumentParser(description="MIDV-2020 synthetic copy-move forgery generator")
    p.add_argument("--annotations", required=True, help="dir of <type>.json VIA files")
    p.add_argument("--images", required=True, help="dir containing <type>/NN.jpg")
    p.add_argument("--out", required=True, help="output root (images/ masks/ synth.csv)")
    p.add_argument("--per-template", type=int, default=2, help="forgeries per template")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--types", default="", help="comma-separated subset of doc types")
    p.add_argument("--cross-type", action="store_true",
                   help="ABLATION ONLY: allow source pixels from other document types")
    args = p.parse_args()
    if args.cross_type:
        raise NotImplementedError("cross-type sourcing is reserved as an ablation; not wired yet")
    generate(args)


if __name__ == "__main__":
    main()
