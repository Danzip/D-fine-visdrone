"""
TTA (Test-Time Augmentation) inference for D-FINE.

Runs inference at multiple scales and with horizontal flip, then merges
predictions per image using Weighted Box Fusion (WBF).

Scales: [768, 1024, 1280]  — same range as training multi-scale
Augmentations: original + horizontal flip  → 6 passes per image

WBF is preferable to NMS for TTA because it re-weights boxes by confidence
rather than discarding all but the top prediction.

Usage — evaluate AP on VisDrone val set:
  python tools/inference/tta_inf.py \
    -c configs/dfine/dfine_hgnetv2_s_visdrone_ms1280_cont.yml \
    -r output/dfine_hgnetv2_s_visdrone_ms1280_cont/best_stg1.pth \
    --ann dataset/visdrone/annotations/instances_val.json \
    --img-dir dataset/visdrone/VisDrone2019-DET-val/images \
    --device cuda:0
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ensemble_boxes import weighted_boxes_fusion
from src.core import YAMLConfig


def load_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=False)

    class _Deploy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    model = _Deploy().to(device)
    model.eval()
    return model


def infer_one(model, image_tensor, orig_wh, device):
    """Run model on a single pre-processed image tensor. Returns (labels, boxes_xyxy, scores)."""
    tensor = image_tensor.unsqueeze(0).to(device)
    orig_size = torch.tensor([orig_wh], dtype=torch.float32).to(device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


def preprocess(pil_image, size):
    """Letterbox resize to square, return float32 tensor [3,H,W] in [0,1]."""
    resized = pil_image.resize((size, size), Image.BILINEAR)
    tensor = TF.to_tensor(resized)  # [3,H,W], float32 [0,1]
    return tensor


def run_tta(model, pil_image, scales, device, score_thr=0.01):
    """
    Run TTA: original + hflip at each scale.
    Returns merged (labels, boxes_xyxy, scores) after WBF.
    """
    w, h = pil_image.size
    pil_hflip = TF.hflip(pil_image)

    all_boxes, all_scores, all_labels = [], [], []

    for size in scales:
        for flipped, img in enumerate([pil_image, pil_hflip]):
            tensor = preprocess(img, size)
            labels, boxes, scores = infer_one(model, tensor, [w, h], device)

            mask = scores > score_thr
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            if flipped:
                # Un-flip boxes: x1'= w - x2, x2'= w - x1
                boxes = boxes.clone()
                boxes[:, 0], boxes[:, 2] = w - boxes[:, 2].clone(), w - boxes[:, 0].clone()

            # Normalize to [0,1] for WBF
            boxes_norm = boxes.clone().float()
            boxes_norm[:, [0, 2]] /= w
            boxes_norm[:, [1, 3]] /= h
            boxes_norm = boxes_norm.clamp(0, 1)

            all_boxes.append(boxes_norm.numpy().tolist())
            all_scores.append(scores.numpy().tolist())
            all_labels.append(labels.int().numpy().tolist())

    if not any(all_scores):
        return torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)

    boxes_wbf, scores_wbf, labels_wbf = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=0.55, skip_box_thr=0.01,
    )

    # De-normalize
    boxes_wbf = torch.tensor(boxes_wbf, dtype=torch.float32)
    scores_wbf = torch.tensor(scores_wbf, dtype=torch.float32)
    labels_wbf = torch.tensor(labels_wbf, dtype=torch.long)
    boxes_wbf[:, [0, 2]] *= w
    boxes_wbf[:, [1, 3]] *= h

    return labels_wbf, boxes_wbf, scores_wbf


def evaluate_coco(model, ann_json, img_dir, scales, device, score_thr=0.01, conf_thr=0.3):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(ann_json) as f:
        images = json.load(f)["images"]

    print(f"TTA eval on {len(images)} images, scales={scales} + hflip ({len(scales)*2} passes/img)")
    all_results = []

    for i, img_info in enumerate(images):
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        pil = Image.open(img_path).convert("RGB")
        labels, boxes, scores = run_tta(model, pil, scales, device, score_thr)

        mask = scores >= conf_thr
        for cat, box, sc in zip(labels[mask], boxes[mask], scores[mask]):
            x1, y1, x2, y2 = box.tolist()
            all_results.append({
                "image_id": img_info["id"],
                "category_id": int(cat),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(sc),
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(images)}]")

    pred_path = "output/tta_val_predictions.json"
    os.makedirs("output", exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(all_results, f)

    coco_gt = COCO(ann_json)
    coco_dt = coco_gt.loadRes(pred_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    names = ["AP50:95", "AP50", "AP75", "AP-small", "AP-medium", "AP-large"]
    print("\n=== TTA Results ===")
    for name, val in zip(names, coco_eval.stats[:6]):
        print(f"  {name:12s}: {val:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",   required=True)
    parser.add_argument("-r", "--resume",   required=True)
    parser.add_argument("-d", "--device",   default="cuda:0")
    parser.add_argument("--ann",            required=True)
    parser.add_argument("--img-dir",        required=True)
    parser.add_argument("--scales",         type=int, nargs="+", default=[768, 1024, 1280])
    parser.add_argument("--iou-thr",        type=float, default=0.55)
    parser.add_argument("--score-thr",      type=float, default=0.01,
                        help="min score to pass into WBF (pre-merge filter)")
    parser.add_argument("--conf-thr",       type=float, default=0.3,
                        help="min score to include in COCO results (post-merge filter)")
    args = parser.parse_args()

    model = load_model(args.config, args.resume, args.device)

    evaluate_coco(
        model, args.ann, args.img_dir,
        scales=args.scales,
        device=args.device,
        score_thr=args.score_thr,
        conf_thr=args.conf_thr,
    )
