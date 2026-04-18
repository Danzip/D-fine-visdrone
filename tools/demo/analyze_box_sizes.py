"""
Experiment D: Object-size distribution analysis.

Loads VisDrone COCO annotations, simulates letterbox resize at multiple
target resolutions, and reports box-size statistics. The goal is to determine:
  - What fraction of objects cross the ~16px "feature-representable" floor
  - How much each resolution step actually moves the distribution
  - Whether higher-resolution training is mechanically justified

Usage:
    python tools/demo/analyze_box_sizes.py \
        --ann dataset/visdrone/annotations/instances_train.json \
        --val dataset/visdrone/annotations/instances_val.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# Resolutions to evaluate. Stored as (width, height).
# Multiples of 32 only (backbone stride requirement).
RESOLUTIONS = [
    (640,  640),   # current training size
    (800,  800),   # modest step up, square
    (960,  544),   # 16:9 near-native, fits 8GB VRAM at batch 4
    (960,  960),   # square high-res
    (1280, 736),   # 16:9 high-res (736/32=23, 1280/32=40 — both integer strides)
    (1280, 1280),  # upper bound, likely OOM at batch>1
]

# Minimum pixel thresholds that matter for detection
THRESHOLDS = [4, 8, 12, 16, 24, 32]


def letterbox_box(bx, by, bw, bh, img_w, img_h, target_w, target_h):
    """
    Apply the same letterbox transform used during training to a bounding box.
    Returns (new_bw, new_bh) in target-image pixel space.
    """
    scale = min(target_w / img_w, target_h / img_h)
    new_bw = bw * scale
    new_bh = bh * scale
    return new_bw, new_bh


def analyze(ann_path: Path, split_name: str, resolutions: list) -> dict:
    with open(ann_path) as f:
        data = json.load(f)

    # Build image_id -> (width, height) lookup
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}

    anns = [a for a in data["annotations"] if a["iscrowd"] == 0]
    print(f"\n{'='*60}")
    print(f"  {split_name}  |  {len(data['images'])} images  |  {len(anns)} annotations")
    print(f"{'='*60}")

    results = {}
    for (tw, th) in resolutions:
        label = f"{tw}×{th}"
        scaled_w, scaled_h = [], []
        for ann in anns:
            iw, ih = img_sizes[ann["image_id"]]
            bx, by, bw, bh = ann["bbox"]
            nbw, nbh = letterbox_box(bx, by, bw, bh, iw, ih, tw, th)
            scaled_w.append(nbw)
            scaled_h.append(nbh)

        scaled_w = np.array(scaled_w)
        scaled_h = np.array(scaled_h)
        areas    = scaled_w * scaled_h
        short_side = np.minimum(scaled_w, scaled_h)
        n = len(scaled_w)

        results[label] = {
            "median_w": np.median(scaled_w),
            "median_h": np.median(scaled_h),
            "p25_w":    np.percentile(scaled_w, 25),
            "p75_w":    np.percentile(scaled_w, 75),
            "median_area": np.median(areas),
            "thresholds": {t: 100 * np.mean(short_side < t) for t in THRESHOLDS},
        }

    # Print comparison table
    header = f"{'Resolution':<14}" + "".join(f"  {f'<{t}px':>7}" for t in THRESHOLDS) + \
             f"  {'med W':>7}  {'med H':>7}  {'med √area':>10}"
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        thr_cols = "".join(f"  {r['thresholds'][t]:>6.1f}%" for t in THRESHOLDS)
        print(
            f"{label:<14}{thr_cols}"
            f"  {r['median_w']:>7.1f}"
            f"  {r['median_h']:>7.1f}"
            f"  {np.sqrt(r['median_area']):>10.1f}"
        )

    return results


def print_delta_table(results_by_res: dict, baseline_label: str = "640×640"):
    """Show how each resolution improves over the 640×640 baseline."""
    base = results_by_res[baseline_label]
    print(f"\n--- Delta vs {baseline_label} (percentage-point reduction in sub-threshold boxes) ---")
    header = f"{'Resolution':<14}" + "".join(f"  {f'<{t}px':>7}" for t in THRESHOLDS)
    print(header)
    print("-" * len(header))
    for label, r in results_by_res.items():
        if label == baseline_label:
            continue
        deltas = "".join(
            f"  {base['thresholds'][t] - r['thresholds'][t]:>6.1f}%" for t in THRESHOLDS
        )
        print(f"{label:<14}{deltas}")


def print_native_stats(ann_path: Path, split_name: str):
    """Report box sizes at native image resolution (no resizing) as reference."""
    with open(ann_path) as f:
        data = json.load(f)
    img_sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]}
    anns = [a for a in data["annotations"] if a["iscrowd"] == 0]
    ws = np.array([a["bbox"][2] for a in anns])
    hs = np.array([a["bbox"][3] for a in anns])
    short = np.minimum(ws, hs)
    print(f"\n--- {split_name} native resolution (no resize) ---")
    print(f"  Median box: {np.median(ws):.1f}w × {np.median(hs):.1f}h px")
    print(f"  P25 short side: {np.percentile(short, 25):.1f}px")
    print(f"  P75 short side: {np.percentile(short, 75):.1f}px")
    for t in THRESHOLDS:
        print(f"  % short side < {t:2d}px: {100*np.mean(short < t):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", required=True, help="Train annotation JSON")
    parser.add_argument("--val", required=True, help="Val annotation JSON")
    args = parser.parse_args()

    print_native_stats(Path(args.ann), "TRAIN")
    print_native_stats(Path(args.val), "VAL")

    train_res = analyze(Path(args.ann), "TRAIN", RESOLUTIONS)
    val_res   = analyze(Path(args.val), "VAL",   RESOLUTIONS)

    print("\n\n=== TRAIN: Gain over 640×640 baseline ===")
    print_delta_table(train_res)

    print("\n\n=== VAL: Gain over 640×640 baseline ===")
    print_delta_table(val_res)

    # Summary interpretation
    print("\n\n=== INTERPRETATION ===")
    base_sub16 = train_res["640×640"]["thresholds"][16]
    for label, r in train_res.items():
        sub16 = r["thresholds"][16]
        delta = base_sub16 - sub16
        med_w = r["median_w"]
        verdict = ""
        if label == "640×640":
            verdict = "<-- current baseline"
        elif delta < 2:
            verdict = "negligible gain — resolution doesn't move the floor"
        elif delta < 6:
            verdict = "small gain — worth trying if VRAM allows"
        elif delta < 12:
            verdict = "meaningful gain — good training resolution candidate"
        else:
            verdict = "large gain — high priority to test"
        print(f"  {label:<14}  sub-16px: {sub16:.1f}%  (delta: {delta:+.1f}pp)  {verdict}")


if __name__ == "__main__":
    main()
