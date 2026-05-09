"""Convert VisDrone2019-DET annotations to COCO format.

VisDrone annotation format (per-image .txt):
  x, y, w, h, score, category, truncation, occlusion
  - (x,y) = top-left corner in pixels
  - score=0 means ignored region — skip these
  - category 0 = unknown — skip these
  - categories 1-10 map to the 10 VisDrone classes

Usage:
  python tools/visdrone2coco.py
Outputs:
  dataset/visdrone/annotations/instances_train.json
  dataset/visdrone/annotations/instances_val.json
"""

import json
import os
from pathlib import Path

from PIL import Image

CATEGORIES = [
    {"id": 0,  "name": "pedestrian"},
    {"id": 1,  "name": "people"},
    {"id": 2,  "name": "bicycle"},
    {"id": 3,  "name": "car"},
    {"id": 4,  "name": "van"},
    {"id": 5,  "name": "truck"},
    {"id": 6,  "name": "tricycle"},
    {"id": 7,  "name": "awning-tricycle"},
    {"id": 8,  "name": "bus"},
    {"id": 9,  "name": "motor"},
]

# Map original VisDrone category IDs (1-10) to 0-indexed (0-9)
VISDRONE_CAT_REMAP = {i: i - 1 for i in range(1, 11)}

BASE = Path("dataset/visdrone")


def convert_split(split: str):
    img_dir = BASE / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = BASE / f"VisDrone2019-DET-{split}" / "annotations"
    out_file = BASE / "annotations" / f"instances_{split}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 1

    img_files = sorted(img_dir.glob("*.jpg"))
    print(f"[{split}] {len(img_files)} images")

    for img_id, img_path in enumerate(img_files):
        w, h = Image.open(img_path).size
        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        for line in ann_path.read_text().strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, bw, bh, score, cat = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            if score == 0 or cat == 0:
                continue
            bw = max(1, bw)
            bh = max(1, bh)
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": VISDRONE_CAT_REMAP[cat],
                "bbox": [x, y, bw, bh],
                "area": float(bw * bh),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    out_file.write_text(json.dumps(coco))
    print(f"[{split}] {len(annotations)} annotations → {out_file}")


if __name__ == "__main__":
    convert_split("train")
    convert_split("val")
