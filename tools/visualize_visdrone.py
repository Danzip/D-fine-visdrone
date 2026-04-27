"""
VisDrone Sample Visualization Script
Draws bounding boxes on 10 random training images with per-image statistics.
Saves annotated images to PROJECT_NOTES/visdrone_samples/
"""

import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

CLASS_COLORS = {
    1: "#FF4444",   # pedestrian — red
    2: "#FF8800",   # people — orange
    3: "#FFFF00",   # bicycle — yellow
    4: "#00FF00",   # car — green
    5: "#00FFFF",   # van — cyan
    6: "#0088FF",   # truck — blue
    7: "#AA44FF",   # tricycle — purple
    8: "#FF44FF",   # awning-tricycle — magenta
    9: "#FF8844",   # bus — salmon
    10: "#88FF44",  # motor — lime
}

def visualize_samples(ann_json: str, images_dir: str, output_dir: str, n: int = 10, seed: int = 42):
    with open(ann_json) as f:
        coco = json.load(f)

    # Build image_id -> annotations lookup
    ann_by_image = {}
    for ann in coco["annotations"]:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    # Pick n random images that have annotations
    images_with_ann = [img for img in coco["images"] if img["id"] in ann_by_image]
    random.seed(seed)
    samples = random.sample(images_with_ann, min(n, len(images_with_ann)))

    os.makedirs(output_dir, exist_ok=True)

    print(f"Visualizing {len(samples)} sample images...")
    print()

    for img_info in samples:
        img_path = os.path.join(images_dir, img_info["file_name"])
        anns = ann_by_image.get(img_info["id"], [])

        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        img_w, img_h = im.size
        img_area = img_w * img_h

        # Draw each bounding box
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cat_id = ann["category_id"]
            color = CLASS_COLORS.get(cat_id, "#FFFFFF")
            draw.rectangle([x, y, x + w, y + h], outline=color, width=1)

        # Per-image stats
        n_boxes = len(anns)
        box_areas = [a["bbox"][2] * a["bbox"][3] for a in anns]
        rel_areas = [a / img_area * 100 for a in box_areas]
        tiny = sum(1 for a in anns if max(a["bbox"][2], a["bbox"][3]) < 32)
        cat_counts = {}
        for ann in anns:
            name = VISDRONE_CLASSES.get(ann["category_id"], "?")
            cat_counts[name] = cat_counts.get(name, 0) + 1

        # Overlay stats text in top-left
        stats_text = (
            f"Objects: {n_boxes} | "
            f"Tiny(<32px): {tiny} ({tiny/max(1,n_boxes)*100:.0f}%) | "
            f"Median rel area: {sorted(rel_areas)[len(rel_areas)//2]:.4f}%"
        )
        # Draw text background
        draw.rectangle([0, 0, img_w, 16], fill="#000000AA")
        draw.text((2, 2), stats_text, fill="#FFFFFF")

        out_name = f"vis_{img_info['file_name']}"
        im.save(os.path.join(output_dir, out_name))

        print(f"  {img_info['file_name']}")
        print(f"    Size: {img_w}x{img_h}  |  Objects: {n_boxes}  |  Tiny: {tiny} ({tiny/max(1,n_boxes)*100:.0f}%)")
        print(f"    Classes: {dict(sorted(cat_counts.items(), key=lambda x: -x[1]))}")
        avg_size = sum(max(a["bbox"][2], a["bbox"][3]) for a in anns) / max(1, n_boxes)
        print(f"    Avg max-side: {avg_size:.1f}px  |  Median rel area: {sorted(rel_areas)[len(rel_areas)//2]:.4f}%")
        print()

    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    visualize_samples(
        ann_json="dataset/visdrone/annotations/instances_train.json",
        images_dir="dataset/visdrone/VisDrone2019-DET-train/images",
        output_dir="PROJECT_NOTES/visdrone_samples",
        n=10,
    )
