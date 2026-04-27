"""
VisDrone → COCO Format Conversion Script
Step 3 of D-FINE + VisDrone project.

VisDrone annotation format (per-image .txt files, one line per object):
    bbox_left, bbox_top, bbox_width, bbox_height, score, category, truncation, occlusion

    Fields:
    - bbox_left, bbox_top: top-left corner of bounding box (pixels)
    - bbox_width, bbox_height: size of bounding box (pixels)
    - score: 0 = ignored region (skip), 1 = normal object
    - category: 0=ignored, 1=pedestrian, 2=people, 3=bicycle, 4=car,
                5=van, 6=truck, 7=tricycle, 8=awning-tricycle, 9=bus, 10=motor
    - truncation: 0=no, 1=partial (object extends beyond frame edge)
    - occlusion: 0=none, 1=partial, 2=heavy

COCO format (single JSON file per split):
    {
        "images": [{"id": int, "file_name": str, "width": int, "height": int}],
        "annotations": [{"id": int, "image_id": int, "category_id": int,
                          "bbox": [x, y, w, h], "area": float, "iscrowd": 0}],
        "categories": [{"id": int, "name": str}]
    }

    Notes:
    - COCO bbox format is [x_min, y_min, width, height] — same as VisDrone, convenient
    - category_id starts at 1 in COCO (0 is reserved for background)
    - We skip objects where score=0 (ignored/distractor regions)
    - We skip objects where category=0 (ambiguous/unknown)

Usage:
    python tools/visdrone2coco.py \
        --visdrone-root dataset/visdrone \
        --output-dir dataset/visdrone/annotations
"""

import argparse
import json
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

# VisDrone category mapping: index → name
# Index 0 = ignored (skip), indices 1-10 = valid classes
VISDRONE_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

# D-FINE with remap_mscoco_category=False uses raw category_id as class index.
# num_classes=10 means valid indices are 0-9. So we remap VisDrone IDs 1-10 -> 0-9.
VISDRONE_ID_TO_LABEL = {k: k - 1 for k in VISDRONE_CATEGORIES}  # 1->0, 2->1, ..., 10->9

COCO_CATEGORIES = [
    {"id": k - 1, "name": v, "supercategory": "object"}
    for k, v in VISDRONE_CATEGORIES.items()
]


def parse_visdrone_annotation(ann_path: str) -> list[dict]:
    """
    Parse one VisDrone .txt annotation file.
    Returns list of valid objects (skipping ignored regions and unknown categories).

    Why we skip score=0:
        VisDrone marks "distractor regions" with score=0. These are areas the
        annotators explicitly said should NOT be evaluated — e.g., regions where
        objects are too small to identify, or background clutter. Training on these
        would teach the model to detect things it shouldn't.

    Why we skip category=0:
        Category 0 = "ignored/other". No useful training signal.
    """
    objects = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])       # 0=ignored, 1=visible
            category = int(parts[5])   # 0=other, 1-10=class

            # Skip ignored/distractor regions
            if score == 0 or category == 0:
                continue

            # Skip degenerate boxes
            if w <= 0 or h <= 0:
                continue

            objects.append({
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "category": category,
            })
    return objects


def get_image_size(img_path: str) -> tuple[int, int]:
    """Return (width, height) of image without loading full pixel data."""
    with Image.open(img_path) as img:
        return img.size  # (width, height)


def convert_split(split_name: str, visdrone_root: str, output_dir: str) -> dict:
    """
    Convert one VisDrone split to COCO JSON format.

    Returns statistics dict for reporting.
    """
    split_dir = os.path.join(visdrone_root, f"VisDrone2019-DET-{split_name}")
    images_dir = os.path.join(split_dir, "images")
    annotations_dir = os.path.join(split_dir, "annotations")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images not found: {images_dir}")

    image_files = sorted(os.listdir(images_dir))
    image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    print(f"\n[{split_name}] Found {len(image_files)} images")

    coco = {
        "images": [],
        "annotations": [],
        "categories": COCO_CATEGORIES,
    }

    image_id = 0
    annotation_id = 0

    # Per-category counts and box size stats for reporting (keyed by original 1-10 IDs)
    category_counts = {k: 0 for k in VISDRONE_CATEGORIES}
    box_areas = []         # list of (w*h) values relative to image area
    box_sizes_abs = []     # list of max(w,h) in pixels

    for fname in tqdm(image_files, desc=f"Converting {split_name}"):
        img_path = os.path.join(images_dir, fname)
        ann_path = os.path.join(annotations_dir, os.path.splitext(fname)[0] + ".txt")

        # Get image dimensions (needed for COCO and for relative size computation)
        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            print(f"  [WARN] Cannot open {img_path}: {e}")
            continue

        # Record image
        coco["images"].append({
            "id": image_id,
            "file_name": fname,
            "width": img_w,
            "height": img_h,
        })

        # Parse annotations
        if os.path.exists(ann_path):
            objects = parse_visdrone_annotation(ann_path)
            for obj in objects:
                # Clamp box to image boundaries
                # (some VisDrone boxes slightly exceed image edges)
                x = max(0, obj["bbox_x"])
                y = max(0, obj["bbox_y"])
                w = min(obj["bbox_w"], img_w - x)
                h = min(obj["bbox_h"], img_h - y)

                if w <= 0 or h <= 0:
                    continue

                area = float(w * h)
                coco["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": VISDRONE_ID_TO_LABEL[obj["category"]],  # remap 1-10 -> 0-9
                    "bbox": [x, y, w, h],       # COCO format: [x_min, y_min, width, height]
                    "area": area,
                    "iscrowd": 0,               # 0 = individual instance, 1 = crowd region
                })

                category_counts[obj["category"]] += 1  # use original ID for stats
                image_area = img_w * img_h
                box_areas.append(area / image_area)
                box_sizes_abs.append(max(w, h))
                annotation_id += 1

        image_id += 1

    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"instances_{split_name}.json")
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"[{split_name}] Saved {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")

    # Compute size stats
    if box_areas:
        import statistics
        med_rel = statistics.median(box_areas)
        mean_rel = statistics.mean(box_areas)
        # "Small" objects in COCO are <32x32 = <1024 pixels² = <0.04% of 640x640
        # VisDrone images are ~2000x1500, so threshold is different
        pct_tiny = sum(1 for a in box_sizes_abs if a < 32) / len(box_sizes_abs) * 100
        pct_small = sum(1 for a in box_sizes_abs if a < 64) / len(box_sizes_abs) * 100
    else:
        med_rel = mean_rel = pct_tiny = pct_small = 0

    stats = {
        "split": split_name,
        "n_images": len(coco["images"]),
        "n_annotations": len(coco["annotations"]),
        "category_counts": {VISDRONE_CATEGORIES[k]: v for k, v in category_counts.items()},
        "median_box_relative_area": round(med_rel * 100, 4),  # as % of image
        "mean_box_relative_area": round(mean_rel * 100, 4),
        "pct_boxes_under_32px": round(pct_tiny, 1),
        "pct_boxes_under_64px": round(pct_small, 1),
        "output_json": out_path,
    }

    return stats


def print_stats(stats: dict):
    """Print human-readable statistics with interpretation."""
    print(f"\n{'='*60}")
    print(f"Split: {stats['split'].upper()}")
    print(f"{'='*60}")
    print(f"  Images:      {stats['n_images']:,}")
    print(f"  Annotations: {stats['n_annotations']:,}")
    print(f"  Avg boxes/image: {stats['n_annotations'] / max(1, stats['n_images']):.1f}")
    print()
    print("  Class distribution:")
    total = sum(stats['category_counts'].values())
    for cls, cnt in sorted(stats['category_counts'].items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / max(1, total) * 40)
        print(f"    {cls:20s}  {cnt:6,}  ({cnt/total*100:4.1f}%)  {bar}")
    print()
    print("  Object size statistics:")
    print(f"    Median box area:  {stats['median_box_relative_area']:.4f}% of image area")
    print(f"    Mean box area:    {stats['mean_box_relative_area']:.4f}% of image area")
    print(f"    Boxes < 32px:     {stats['pct_boxes_under_32px']:.1f}%")
    print(f"    Boxes < 64px:     {stats['pct_boxes_under_64px']:.1f}%")
    print()
    print("  Interpretation:")
    print(f"    COCO 'small' threshold = 32x32px (1024px²).")
    print(f"    {stats['pct_boxes_under_32px']:.0f}% of VisDrone boxes are under 32px max-side.")
    print(f"    This is why VisDrone is hard — most objects have very few pixels.")
    print(f"    Compare to COCO APS (small object mAP) = 29.4%.")
    print(f"    VisDrone will be even harder due to higher density and aerial viewpoint.")


def main(args):
    print("VisDrone -> COCO Annotation Conversion")
    print(f"Input:  {args.visdrone_root}")
    print(f"Output: {args.output_dir}")

    all_stats = []
    for split in args.splits:
        stats = convert_split(split, args.visdrone_root, args.output_dir)
        print_stats(stats)
        all_stats.append(stats)

    # Save stats JSON for later reference
    stats_path = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to: {stats_path}")

    print("\nAll splits converted. Output structure:")
    for stats in all_stats:
        print(f"  {stats['output_json']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visdrone-root", default="dataset/visdrone")
    parser.add_argument("--output-dir", default="dataset/visdrone/annotations")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    main(args)
