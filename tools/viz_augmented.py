"""
Visualize augmented training batches from the D-FINE pipeline.

Runs a few batches through the full train dataloader (including Mosaic,
CopyPasteSmallObjects, AR padding, etc.) and saves annotated images so you
can inspect the black bars and augmentation output before training.

Usage:
    python tools/viz_augmented.py \
        -c configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml \
        --n-batches 8 \
        --out output/viz_augmented
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from src.core import YAMLConfig


PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 198),
]


def draw_boxes(img_tensor, boxes_cxcywh_norm, labels, canvas_h, canvas_w):
    """Draw normalized cxcywh boxes onto a tensor image. Returns PIL Image."""
    img = TF.to_pil_image(img_tensor.clamp(0, 1))
    draw = ImageDraw.Draw(img)
    for box, label in zip(boxes_cxcywh_norm, labels):
        cx, cy, bw, bh = box.tolist()
        x1 = (cx - bw / 2) * canvas_w
        y1 = (cy - bh / 2) * canvas_h
        x2 = (cx + bw / 2) * canvas_w
        y2 = (cy + bh / 2) * canvas_h
        color = PALETTE[int(label) % len(PALETTE)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--n-batches", type=int, default=8,
                        help="Number of batches to visualize")
    parser.add_argument("--out", default="output/viz_augmented",
                        help="Output directory for saved images")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    cfg = YAMLConfig(args.config)
    loader = cfg.train_dataloader
    loader.collate_fn.set_epoch(0)   # epoch=0 → augmentation active
    # Wire dataset for AR-aware mosaic sampling (mirrors what _solver.py does at train time)
    if hasattr(loader.collate_fn, "set_dataset"):
        loader.collate_fn.set_dataset(loader.dataset)

    print(f"Saving {args.n_batches} batches to {args.out}/")

    for batch_idx, (images, targets) in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        B, C, H, W = images.shape
        lb = targets[0].get("letterbox")
        ar_label = ""
        if lb is not None:
            ar_label = f"_ar{W}x{H}"

        for i in range(B):
            img = images[i]
            tgt = targets[i]
            boxes = tgt.get("boxes", torch.zeros(0, 4))
            labels = tgt.get("labels", torch.zeros(0, dtype=torch.long))

            pil = draw_boxes(img, boxes, labels, H, W)
            fname = f"batch{batch_idx:03d}_img{i:02d}{ar_label}.jpg"
            pil.save(os.path.join(args.out, fname))

        print(f"  batch {batch_idx:3d}: shape={tuple(images.shape)}  "
              f"AR={'16:9' if W/H > 1.5 else '4:3' if W/H > 1.2 else '~1:1'}")

    print(f"\nDone. Open {args.out}/ to inspect images.")


if __name__ == "__main__":
    main()
