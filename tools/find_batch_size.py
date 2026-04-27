"""
Find the max safe batch size per resolution for D-FINE training.

For each resolution sz in the multi-scale ladder (768..1280 step 32):
  - Loads random real images from the VisDrone dataset
  - Probes 4:3 canvas  (sz × round(sz*3/4))  using 4:3 images from dataset
  - Probes 16:9 canvas (sz × round(sz*9/16)) using 16:9 images from dataset
  - Runs at the estimated limit and at limit+1 to verify the ceiling
  - Saves both values: {sz: {"4:3": max_batch, "16:9": max_batch}}

Only forward + backward is run — optimizer.step() is deliberately skipped because
optimizer state (Adam m/v) is a fixed overhead independent of batch size / resolution.

Usage:
    python tools/find_batch_size.py \\
        -c configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_mosaic.yml \\
        --img-folder dataset/visdrone/VisDrone2019-DET-train/images \\
        --ann-file   dataset/visdrone/annotations/instances_train.json \\
        --out        output/batch_size_table.json
"""

import argparse
import gc
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
import torch.nn.functional as F

from src.core import YAMLConfig


def reset():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def short_side_32(sz, short_ratio, long_ratio):
    """Short side for longest-side=sz at ratio short_ratio:long_ratio, aligned to 32."""
    s = round(sz * short_ratio / long_ratio)
    return max(32, round(s / 32) * 32)


def load_pool(img_folder, ann_file, n=64):
    """
    Load up to n images per AR class (4:3 and 16:9) from the dataset.
    Returns two lists of (3, H, W) float32 CPU tensors in [0, 1].
    """
    from PIL import Image
    import torchvision.transforms.functional as TF

    with open(ann_file) as f:
        imgs = json.load(f)["images"]
    random.shuffle(imgs)

    pool_43, pool_169 = [], []
    for info in imgs:
        if len(pool_43) >= n and len(pool_169) >= n:
            break
        w, h = info["width"], info["height"]
        ar = w / h
        if abs(ar - 4 / 3) < 0.05:
            if len(pool_43) < n:
                path = os.path.join(img_folder, info["file_name"])
                pool_43.append(TF.to_tensor(Image.open(path).convert("RGB")))
        elif abs(ar - 16 / 9) < 0.05:
            if len(pool_169) < n:
                path = os.path.join(img_folder, info["file_name"])
                pool_169.append(TF.to_tensor(Image.open(path).convert("RGB")))

    print(f"  Loaded {len(pool_43)} × 4:3 images,  {len(pool_169)} × 16:9 images")
    return pool_43, pool_169


def resize_to_canvas(tensor, canvas_h, canvas_w):
    """Resize (AR-preserving) then zero-pad to exactly (canvas_h, canvas_w)."""
    _, h, w = tensor.shape
    scale = min(canvas_h / h, canvas_w / w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    img = F.interpolate(
        tensor.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
    ).squeeze(0)
    pad_r = canvas_w - new_w
    pad_b = canvas_h - new_h
    return F.pad(img, [0, pad_r, 0, pad_b], value=0.0)


def fake_targets(batch_size, canvas_h, canvas_w, device):
    targets = []
    for _ in range(batch_size):
        n = 8
        cx = torch.empty(n).uniform_(0.1, 0.9)
        cy = torch.empty(n).uniform_(0.1, 0.9)
        bw = torch.empty(n).uniform_(0.05, 0.25)
        bh = torch.empty(n).uniform_(0.05, 0.25)
        targets.append({
            "boxes":     torch.stack([cx, cy, bw, bh], dim=1).to(device),
            "labels":    torch.randint(0, 10, (n,)).to(device),
            "orig_size": torch.tensor([canvas_h, canvas_w]).to(device),
        })
    return targets


def measure(model, criterion, device, pool, batch_size, canvas_h, canvas_w, use_amp):
    """
    Run forward+backward with `batch_size` real images resized to canvas_h×canvas_w.
    Returns peak reserved VRAM in MB, or None on OOM.
    """
    reset()
    try:
        chosen = random.choices(pool, k=batch_size)
        imgs = torch.stack([resize_to_canvas(im, canvas_h, canvas_w) for im in chosen]).to(device)
        targets = fake_targets(batch_size, canvas_h, canvas_w, device)

        with torch.autocast(device_type="cuda", enabled=use_amp):
            outputs   = model(imgs, targets=targets)
            loss_dict = criterion(outputs, targets)
            loss      = sum(v for v in loss_dict.values() if v.requires_grad)

        loss.backward()
        return torch.cuda.max_memory_reserved() / 1024 ** 2

    except torch.cuda.OutOfMemoryError:
        return None
    finally:
        model.zero_grad(set_to_none=True)
        reset()


def find_max(model, criterion, device, pool, canvas_h, canvas_w, estimate, use_amp, budget_mb, max_batch):
    """
    Probe at estimate. If it passes probe estimate+1 to confirm the ceiling.
    If estimate fails, step down until one passes.
    Returns (max_safe_batch, peak_mb).
    """
    bs = min(estimate, max_batch)
    while bs >= 1:
        peak = measure(model, criterion, device, pool, bs, canvas_h, canvas_w, use_amp)
        if peak is not None and peak <= budget_mb:
            peak_up = measure(model, criterion, device, pool, bs + 1, canvas_h, canvas_w, use_amp)
            if peak_up is not None and peak_up <= budget_mb:
                return bs + 1, peak_up
            return bs, peak
        bs -= 1
    return 0, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",       required=True)
    parser.add_argument("--img-folder",         default="dataset/visdrone/VisDrone2019-DET-train/images")
    parser.add_argument("--ann-file",           default="dataset/visdrone/annotations/instances_train.json")
    parser.add_argument("--pool-size",          type=int,   default=64,
                        help="Images to pre-load per AR class (4:3 and 16:9)")
    parser.add_argument("--base-size",          type=int,   default=1024)
    parser.add_argument("--max-batch",          type=int,   default=20,
                        help="Hard cap on batch size to probe")
    parser.add_argument("--safety-margin",      type=float, default=0.92,
                        help="Fraction of VRAM to use as ceiling")
    parser.add_argument("--device",            default="cuda:0")
    parser.add_argument("--out",               default="output/batch_size_table.json")
    args = parser.parse_args()

    device     = torch.device(args.device)
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    budget_mb  = total_vram * args.safety_margin
    print(f"GPU : {torch.cuda.get_device_name(device)}")
    print(f"VRAM: {total_vram:.0f} MB   budget ({args.safety_margin:.0%}): {budget_mb:.0f} MB\n")

    cfg       = YAMLConfig(args.config)
    model     = cfg.model.to(device).train()
    criterion = cfg.criterion.to(device)
    use_amp   = getattr(cfg, "use_amp", True)

    print("Loading image pools from dataset …")
    pool_43, pool_169 = load_pool(args.img_folder, args.ann_file, n=args.pool_size)
    if not pool_43:
        raise RuntimeError("No 4:3 images found — check --img-folder / --ann-file paths")
    if not pool_169:
        raise RuntimeError("No 16:9 images found — check --img-folder / --ann-file paths")

    base = args.base_size

    # Calibrate at base_size, 4:3 (most pixels → worst case)
    cal_h = base
    cal_w = short_side_32(base, 3, 4)
    cal_pixels = cal_h * cal_w
    print(f"\nCalibrating at {base}px  4:3 → {cal_h}×{cal_w}  ({cal_pixels:,} px) …")
    v1 = measure(model, criterion, device, pool_43, 1, cal_h, cal_w, use_amp)
    v2 = measure(model, criterion, device, pool_43, 2, cal_h, cal_w, use_amp)
    if v1 is None or v2 is None:
        raise RuntimeError("OOM at B=1 or B=2 during calibration — GPU too small.")
    fixed_mb   = max(0.0, 2 * v1 - v2)
    per_sample = v2 - v1
    print(f"  B=1: {v1:.0f} MB   B=2: {v2:.0f} MB")
    print(f"  Fixed overhead: {fixed_mb:.0f} MB   Per-sample at base: {per_sample:.0f} MB\n")

    # Scale ladder: 768..1280 step 32  (matches generate_scales(base=1024))
    scale_repeat = (base - int(base * 0.75 / 32) * 32) // 32
    scales_low  = [int(base * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales_high = [int(base * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    scales      = sorted(set(scales_low + [base] + scales_high))
    print(f"Scale ladder: {scales}\n")

    hdr = (f"{'sz':>5}  {'4:3 canvas':>11}  {'16:9 canvas':>12}  "
           f"{'est43':>6}  {'max43':>6}  {'MB43':>6}  "
           f"{'est169':>7}  {'max169':>7}  {'MB169':>7}")
    print(hdr)
    print("-" * len(hdr))

    table = {}
    for sz in scales:
        h43,  w43  = sz, short_side_32(sz, 3, 4)
        h169, w169 = sz, short_side_32(sz, 9, 16)
        px43  = h43 * w43
        px169 = h169 * w169

        est43  = max(1, min(args.max_batch, int((budget_mb - fixed_mb) / (per_sample * px43  / cal_pixels))))
        est169 = max(1, min(args.max_batch, int((budget_mb - fixed_mb) / (per_sample * px169 / cal_pixels))))

        max43,  peak43  = find_max(model, criterion, device, pool_43,  h43,  w43,  est43,  use_amp, budget_mb, args.max_batch)
        max169, peak169 = find_max(model, criterion, device, pool_169, h169, w169, est169, use_amp, budget_mb, args.max_batch)

        table[sz] = {"4:3": max43, "16:9": max169}
        mark = " ←" if sz == base else ""
        print(f"{sz:>5}  {h43}×{w43:<4}       {h169}×{w169:<4}   "
              f"{est43:>6}  {max43:>6}  {peak43:>6.0f}  "
              f"{est169:>7}  {max169:>7}  {peak169:>7.0f}{mark}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'sz':>5}  {'4:3 canvas':>12}  {'16:9 canvas':>13}  {'4:3':>5}  {'16:9':>6}")
    print("-" * 46)
    for sz, entry in sorted(table.items()):
        h43,  w43  = sz, short_side_32(sz, 3, 4)
        h169, w169 = sz, short_side_32(sz, 9, 16)
        mark = "  ← base" if sz == base else ""
        print(f"{sz:>5}  {h43}×{w43:<5}        {h169}×{w169:<5}       {entry['4:3']:>5}  {entry['16:9']:>6}{mark}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in table.items()}, f, indent=2)
    print(f"\nSaved to: {args.out}")
    print("Use batch_size_table_path in your config to apply this table.")


if __name__ == "__main__":
    main()
