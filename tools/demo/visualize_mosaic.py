"""
Quick visual check that Mosaic augmentation is working.
Loads one training sample, applies the full transform pipeline (which includes Mosaic),
draws bounding boxes, saves to disk, and optionally logs to W&B.

Usage:
  python tools/demo/visualize_mosaic.py --n 4 --wandb
"""

import argparse
import os
import random
import sys

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.core import YAMLConfig

VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]
COLORS = [
    "#FF3333", "#FF9900", "#FFFF00", "#33FF33", "#00FFFF",
    "#3399FF", "#CC33FF", "#FF66CC", "#99FF33", "#FF6633",
]


def draw_boxes(image_tensor, boxes_cxcywh_norm, labels, score=None):
    """Draw boxes (cxcywh normalized) on a CHW float32 tensor, return PIL image."""
    img = (image_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    H, W = img.shape[:2]

    for i, (box, label) in enumerate(zip(boxes_cxcywh_norm, labels)):
        cx, cy, bw, bh = box.tolist()
        x1 = (cx - bw / 2) * W
        y1 = (cy - bh / 2) * H
        x2 = (cx + bw / 2) * W
        y2 = (cy + bh / 2) * H
        color = COLORS[int(label) % len(COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = VISDRONE_CLASSES[int(label)] if int(label) < len(VISDRONE_CLASSES) else str(label)
        draw.text((x1 + 2, y1 + 2), name, fill=color)

    return pil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml")
    parser.add_argument("--n", type=int, default=4, help="number of samples to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="output/mosaic_viz")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = YAMLConfig(args.config)

    # Force Mosaic p=1.0 so it always fires
    transforms = cfg.yaml_cfg["train_dataloader"]["dataset"]["transforms"]["ops"]
    for op in transforms:
        if op.get("type") == "Mosaic":
            op["p"] = 1.0

    dataset = cfg.train_dataloader.dataset

    if args.wandb:
        import wandb
        wandb.init(project="D-FINE", name="mosaic_viz", job_type="debug")

    paths = []
    for i in range(args.n):
        idx = random.randint(0, len(dataset) - 1)
        img_tensor, target = dataset[idx]

        boxes = target["boxes"]   # cxcywh normalized
        labels = target["labels"]

        pil = draw_boxes(img_tensor, boxes, labels)
        out_path = os.path.join(args.output_dir, f"mosaic_{i:02d}.jpg")
        pil.save(out_path)
        print(f"Saved {out_path}  ({len(labels)} objects, image {img_tensor.shape})")
        paths.append((out_path, pil))

    if args.wandb:
        wandb.log({
            "mosaic_samples": [
                wandb.Image(pil, caption=f"sample_{i}")
                for i, (_, pil) in enumerate(paths)
            ]
        })
        wandb.finish()
        print("Logged to W&B.")
    else:
        print(f"\nImages saved to {args.output_dir}/  — no --wandb flag, skipping upload.")


if __name__ == "__main__":
    main()
