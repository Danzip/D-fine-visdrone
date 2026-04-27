"""
D-FINE COCO Baseline Evaluation Script
Step 2 of D-FINE + VisDrone project.

What this script does:
1. Load D-FINE-S pretrained on COCO
2. Run inference on 5 sample images with per-image timing and detection output
3. Benchmark latency: 100 warmup + 500 timed forward passes
4. Run full COCO mAP evaluation if val2017 images are available

Usage:
    python tools/inference/coco_baseline.py \
        --config configs/dfine/dfine_hgnetv2_s_coco.yml \
        --checkpoint weight/dfine_s_coco.pth \
        --coco-root dataset/coco \
        --device cuda:0 \
        --output-dir PROJECT_NOTES/coco_baseline_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.core import YAMLConfig

# ------------------------------------------------------------------
# COCO 80-class names (contiguous 0-79 after remap_mscoco_category)
# These map to the remapped indices used by D-FINE (remap_mscoco_category=True)
# ------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def build_model(config_path: str, checkpoint_path: str, device: str):
    """Load D-FINE model in deploy mode (BatchNorm fused into Conv)."""
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    cfg.yaml_cfg["HGNetv2"]["pretrained"] = False  # don't re-download backbone weights

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # D-FINE checkpoints store EMA weights under 'ema.module' (better generalisation)
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("[model] Loaded EMA weights (exponential moving average — slightly better than raw)")
    else:
        state = checkpoint["model"]
        print("[model] Loaded raw model weights")

    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            # deploy() fuses BN into Conv — faster inference, no numerical difference
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_sizes):
            out = self.model(images)
            return self.postprocessor(out, orig_sizes)

    model = DeployModel().to(device).eval()
    return model


def preprocess(image_path: str, device: str):
    """Resize to 640x640 and normalise to [0,1] float32. D-FINE does NOT use ImageNet mean/std."""
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    tensor = transform(im).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]], dtype=torch.float32).to(device)
    return im, tensor, orig_size


def run_sample_inference(model, image_paths, device, output_dir, threshold=0.4):
    """
    Run inference on a list of images and save annotated results.
    Reports per-image: detected classes, scores, latency.
    """
    print("\n" + "=" * 60)
    print("SAMPLE INFERENCE (5 images)")
    print("=" * 60)
    print(f"Score threshold: {threshold}  (detections below this are discarded)")
    print()

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, img_path in enumerate(image_paths):
        im, tensor, orig_size = preprocess(img_path, device)

        # Time the full forward pass (preprocess already done — timing model only)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t0 = time.perf_counter()
        with torch.no_grad():
            labels, boxes, scores = model(tensor, orig_size)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        latency_ms = (time.perf_counter() - t0) * 1000

        # Filter by threshold
        mask = scores[0] > threshold
        det_labels = labels[0][mask].cpu().numpy()
        det_boxes = boxes[0][mask].cpu().numpy()
        det_scores = scores[0][mask].cpu().numpy()

        # Draw boxes on image
        draw_img = im.copy()
        draw_obj = ImageDraw.Draw(draw_img)
        colors = ["#FF3333", "#33FF33", "#3333FF", "#FF33FF", "#33FFFF",
                  "#FFAA33", "#AA33FF", "#33FFAA", "#FF3333", "#AAFFAA"]
        for j, (lbl, box, scr) in enumerate(zip(det_labels, det_boxes, det_scores)):
            class_name = COCO_CLASSES[int(lbl)] if int(lbl) < len(COCO_CLASSES) else str(lbl)
            color = colors[int(lbl) % len(colors)]
            draw_obj.rectangle(list(box), outline=color, width=2)
            draw_obj.text(
                (box[0], max(0, box[1] - 12)),
                f"{class_name} {scr:.2f}",
                fill=color,
            )

        # Save annotated image
        out_name = f"sample_{i+1}_{Path(img_path).stem}.jpg"
        draw_img.save(os.path.join(output_dir, out_name))

        # Print summary
        print(f"Image {i+1}: {Path(img_path).name}")
        print(f"  Size: {im.size[0]}x{im.size[1]} px")
        print(f"  Latency: {latency_ms:.1f} ms")
        print(f"  Detections (>{threshold}): {mask.sum().item()}")
        for lbl, scr in sorted(zip(det_labels, det_scores), key=lambda x: -x[1])[:5]:
            class_name = COCO_CLASSES[int(lbl)] if int(lbl) < len(COCO_CLASSES) else str(lbl)
            print(f"    {class_name:20s}  score={scr:.3f}")
            # Score interpretation:
            # 0.9+  = very confident — model has seen this pattern many times
            # 0.7-0.9 = confident
            # 0.4-0.7 = moderate — correct class but ambiguous box or partial occlusion
            # <0.4  = discarded by threshold
        print()

        results.append({
            "image": Path(img_path).name,
            "latency_ms": round(latency_ms, 2),
            "n_detections": int(mask.sum().item()),
            "saved_to": out_name,
        })

    return results


def benchmark_latency(model, device, n_warmup=100, n_timed=500, input_size=(640, 640)):
    """
    Benchmark model latency with proper GPU warmup.

    Why warmup?
    GPU starts cold — first N iterations trigger CUDA kernel JIT compilation,
    memory allocation, and cache warmup. Without warmup, first-N results are
    10x slower and pollute the mean. 100 warmup iterations is standard.

    We time ONLY the model forward pass, not preprocessing, to match
    what is reported in the D-FINE paper (3.49ms for S on A100).
    Our RTX 4060 is slower than A100 — that is expected and fine for our purposes.
    """
    print("=" * 60)
    print(f"LATENCY BENCHMARK  ({n_warmup} warmup + {n_timed} timed passes)")
    print(f"Input resolution: {input_size[0]}x{input_size[1]}")
    print("=" * 60)

    dummy = torch.randn(1, 3, *input_size).to(device)
    orig_size = torch.tensor([[input_size[1], input_size[0]]], dtype=torch.float32).to(device)

    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy, orig_size)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    print("Warmup complete.")

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_timed):
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy, orig_size)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    import statistics
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    p50_ms = sorted(times)[n_timed // 2]
    p95_ms = sorted(times)[int(n_timed * 0.95)]
    fps = 1000.0 / mean_ms

    print(f"\nResults over {n_timed} runs:")
    print(f"  Mean latency : {mean_ms:.2f} ms")
    print(f"  Std dev      : {std_ms:.2f} ms")
    print(f"  P50 latency  : {p50_ms:.2f} ms")
    print(f"  P95 latency  : {p95_ms:.2f} ms")
    print(f"  Throughput   : {fps:.1f} FPS")
    print()
    print("Interpretation:")
    print(f"  {fps:.0f} FPS means D-FINE-S can process {fps:.0f} frames/second on your GPU.")
    print(f"  Real-time video is typically 25-30 FPS — we are at {fps/30:.1f}x real-time.")
    print(f"  Paper reports 3.49ms on A100 (80GB). Your RTX 4060 Laptop is expected to be slower.")
    print()

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "device": device,
        "input_size": list(input_size),
    }


def run_coco_eval(config_path, checkpoint_path, coco_root, device):
    """
    Run the official COCO mAP evaluation using D-FINE's built-in evaluator.
    This is identical to what was used to produce the paper's 48.5 mAP number.
    """
    print("=" * 60)
    print("COCO VAL2017 OFFICIAL EVALUATION")
    print("=" * 60)

    val_images = os.path.join(coco_root, "val2017")
    val_ann = os.path.join(coco_root, "annotations", "instances_val2017.json")

    if not os.path.exists(val_images):
        print(f"[SKIP] COCO val2017 images not found at: {val_images}")
        print("       Download from http://images.cocodataset.org/zips/val2017.zip")
        print("       and extract to dataset/coco/val2017/")
        return None

    if not os.path.exists(val_ann):
        print(f"[SKIP] Annotations not found at: {val_ann}")
        return None

    print(f"Found val2017 images at: {val_images}")
    print(f"Found annotations at: {val_ann}")
    print("Running evaluation on 5000 images...")
    print()

    # Build the full training cfg with dataset paths overridden
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "-c", config_path,
            "-r", checkpoint_path,
            "--test-only",
            "--device", device,
            "-u",
            f"val_dataloader.dataset.img_folder={val_images}",
            f"val_dataloader.dataset.ann_file={val_ann}",
            "HGNetv2.pretrained=False",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("[ERROR]", result.stderr[-1000:])
    return result.stdout


def main(args):
    print("\nD-FINE-S COCO BASELINE")
    print("=" * 60)
    print(f"Config:     {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device:     {args.device}")
    print()

    # 1. Load model
    model = build_model(args.config, args.checkpoint, args.device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] Parameters: {n_params:.1f}M")
    print(f"[model] Deploy mode: Conv+BN fused for faster inference")
    print()

    # 2. Sample inference on 5 images
    coco_val = os.path.join(args.coco_root, "val2017")
    if os.path.exists(coco_val):
        import random
        all_imgs = sorted(os.listdir(coco_val))
        # Use specific diverse image IDs that we know contain varied content
        seed_imgs = [
            "000000001000.jpg",  # wildlife (bear)
            "000000003501.jpg",  # kitchen scene
            "000000007386.jpg",  # street scene
            "000000016228.jpg",  # sports
            "000000020247.jpg",  # indoor
        ]
        sample_imgs = []
        for fname in seed_imgs:
            fpath = os.path.join(coco_val, fname)
            if os.path.exists(fpath):
                sample_imgs.append(fpath)
        # Fill remaining slots if any seed images are missing
        if len(sample_imgs) < 5:
            random.seed(42)
            remaining = [f for f in all_imgs if os.path.join(coco_val, f) not in sample_imgs]
            for f in random.sample(remaining, 5 - len(sample_imgs)):
                sample_imgs.append(os.path.join(coco_val, f))
    else:
        print(f"[WARNING] val2017 not found at {coco_val}")
        print("          Skipping sample inference on COCO images.")
        sample_imgs = []

    sample_results = []
    if sample_imgs:
        sample_results = run_sample_inference(
            model, sample_imgs, args.device, args.output_dir
        )

    # 3. Latency benchmark
    bench = benchmark_latency(model, args.device)

    # 4. Full COCO eval
    coco_eval_output = None
    if not args.skip_eval:
        coco_eval_output = run_coco_eval(
            args.config, args.checkpoint, args.coco_root, args.device
        )

    # 5. Save results JSON
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "model": "D-FINE-S",
        "pretrained_on": "COCO2017",
        "checkpoint": args.checkpoint,
        "device": args.device,
        "sample_inference": sample_results,
        "benchmark": bench,
        "coco_eval_log": coco_eval_output,
    }
    out_json = os.path.join(args.output_dir, "coco_baseline_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    print(f"Annotated images saved to: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D-FINE COCO Baseline")
    parser.add_argument("-c", "--config", default="configs/dfine/dfine_hgnetv2_s_coco.yml")
    parser.add_argument("--checkpoint", default="weight/dfine_s_coco.pth")
    parser.add_argument("--coco-root", default="dataset/coco")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="PROJECT_NOTES/coco_baseline_results")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip full COCO mAP eval (just do inference + benchmark)")
    args = parser.parse_args()
    main(args)
