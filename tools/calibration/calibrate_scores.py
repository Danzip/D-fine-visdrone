"""
R4: per-class score calibration (eval-only, no retraining).

Rationale: DFINEPostProcessor picks the global top-500 detections per image by
flattening (query, class) scores and taking torch.topk over the flattened
tensor (see src/zoo/dfine/postprocessor.py). This means classes compete
directly on raw confidence for the maxDets=500 budget -- a systematically
under-confident class (e.g. rare classes like bicycle/tricycle) can lose
slots to a well-calibrated, more common class even where its localization is
correct. A monotonic per-class rescaling can't change within-class AP (COCO
AP is rank-based), but it CAN change which detections survive the
cross-class top-500 cut, and thus can change overall/per-class AP here.

Method:
  1. Fit: run the model once over a subset of the TRAIN split (eval-mode
     transforms, no augmentation). For each GT box, find the max-scoring
     same-class prediction with IoU >= 0.5 ("matched TP"); record its raw
     sigmoid score. Take the per-class median TP score.
  2. Convert medians to logit space; set target = the least-under-confident
     class's logit. bias_c = max(0, target_logit - logit(median_c)) -- i.e.
     only BOOST under-confident classes, never suppress.
  3. Apply the frozen bias vector to pred_logits (before sigmoid/topk) via a
     thin wrapper around the real postprocessor, and run full COCO eval on
     the VAL split with bias=0 (baseline) and bias=fitted (calibrated) to
     report the actual AP/AP50 delta.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch
import torchvision

from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS
from src.solver.det_engine import evaluate as det_evaluate
from src.solver.validator import scale_boxes

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


class BiasedPostprocessor(torch.nn.Module):
    """Wraps a DFINEPostProcessor, adding a frozen per-class logit bias
    to pred_logits before the real postprocessor does sigmoid+topk."""

    def __init__(self, inner, bias):
        super().__init__()
        self.inner = inner
        self.register_buffer("bias", bias.clone())

    @property
    def remap_mscoco_category(self):
        return self.inner.remap_mscoco_category

    def forward(self, outputs, orig_target_sizes):
        outputs = dict(outputs)
        outputs["pred_logits"] = outputs["pred_logits"] + self.bias.view(1, 1, -1)
        return self.inner(outputs, orig_target_sizes)


def build_solver(config, checkpoint, device, img_folder=None, ann_file=None):
    overrides = {"resume": checkpoint, "device": device}
    if img_folder:
        overrides = yaml_utils.merge_dict(
            overrides, yaml_utils.parse_cli([f"val_dataloader.dataset.img_folder={img_folder}"])
        )
    if ann_file:
        overrides = yaml_utils.merge_dict(
            overrides, yaml_utils.parse_cli([f"val_dataloader.dataset.ann_file={ann_file}"])
        )
    cfg = YAMLConfig(config, **overrides)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    solver.eval()
    return solver


@torch.no_grad()
def fit_bias(solver, num_classes, max_images=None):
    model = solver.ema.module if solver.ema else solver.model
    model.eval()
    device = next(model.parameters()).device
    tp_scores = [[] for _ in range(num_classes)]
    n_seen = 0
    for samples, targets in solver.val_dataloader:
        samples = samples.to(device)
        targets = [
            {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()}
            for t in targets
        ]
        outputs = model(samples)
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        scores = logits.sigmoid()
        orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        bbox_pred = bbox_pred * orig_sizes.repeat(1, 2).unsqueeze(1)

        for i, t in enumerate(targets):
            if t["boxes"].numel() == 0:
                continue
            gt_boxes = scale_boxes(
                t["boxes"],
                (t["orig_size"][1], t["orig_size"][0]),
                (samples[i].shape[-1], samples[i].shape[-2]),
            )
            gt_labels = t["labels"]
            ious = torchvision.ops.box_iou(bbox_pred[i], gt_boxes)  # [num_queries, num_gt]
            for g in range(gt_boxes.shape[0]):
                c = int(gt_labels[g].item())
                mask = ious[:, g] >= 0.5
                if mask.any():
                    best = scores[i, mask, c].max().item()
                    tp_scores[c].append(best)
        n_seen += samples.shape[0]
        if max_images and n_seen >= max_images:
            break
    return tp_scores, n_seen


def compute_bias(tp_scores, num_classes):
    medians = []
    for c in range(num_classes):
        if len(tp_scores[c]) == 0:
            medians.append(None)
            continue
        s = sorted(tp_scores[c])
        medians.append(s[len(s) // 2])

    valid = [m for m in medians if m is not None]
    target = max(valid)
    target_logit = math.log(target / (1 - target))

    bias = []
    for c in range(num_classes):
        if medians[c] is None:
            bias.append(0.0)
            continue
        m = min(max(medians[c], 1e-4), 1 - 1e-4)
        logit_m = math.log(m / (1 - m))
        bias.append(max(0.0, target_logit - logit_m))
    return bias, medians


def per_class_ap(coco_eval, num_classes):
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    out = {}
    for k in range(num_classes):
        p_all = precision[:, :, k, 0, 2]
        ap = float(p_all[p_all > -1].mean()) if (p_all > -1).any() else float("nan")
        p50 = precision[0, :, k, 0, 2]
        ap50 = float(p50[p50 > -1].mean()) if (p50 > -1).any() else float("nan")
        out[CLASS_NAMES[k]] = {"AP": ap, "AP50": ap50}
    return out


def run_full_eval(solver, bias_vec, num_classes):
    device = next((solver.ema.module if solver.ema else solver.model).parameters()).device
    model = solver.ema.module if solver.ema else solver.model
    bias_t = torch.tensor(bias_vec, device=device, dtype=torch.float32)
    biased_pp = BiasedPostprocessor(solver.postprocessor, bias_t).to(device)
    solver.evaluator.cleanup()
    stats, coco_evaluator = det_evaluate(
        model, solver.criterion, biased_pp, solver.val_dataloader,
        solver.evaluator, device, epoch=-1, use_wandb=False,
    )
    overall = coco_evaluator.coco_eval["bbox"].stats
    per_class = per_class_ap(coco_evaluator.coco_eval["bbox"], num_classes)
    return overall, per_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-img-folder", default="dataset/visdrone/VisDrone2019-DET-train/images")
    p.add_argument("--train-ann-file", default="dataset/visdrone/annotations/instances_train.json")
    p.add_argument("--max-fit-images", type=int, default=2000)
    p.add_argument("--out", default="output/calibration/r4_results.json")
    args = p.parse_args()

    num_classes = len(CLASS_NAMES)

    print("=== [1/3] Fitting bias on TRAIN split (eval-mode transforms) ===")
    fit_solver = build_solver(
        args.config, args.checkpoint, args.device,
        img_folder=args.train_img_folder, ann_file=args.train_ann_file,
    )
    tp_scores, n_seen = fit_bias(fit_solver, num_classes, max_images=args.max_fit_images)
    bias_vec, medians = compute_bias(tp_scores, num_classes)
    print(f"(fit over {n_seen} train images)")
    for c in range(num_classes):
        med = f"{medians[c]:.4f}" if medians[c] is not None else "n/a"
        print(f"  {CLASS_NAMES[c]:16s} n_tp={len(tp_scores[c]):5d} median_score={med:>8s} bias={bias_vec[c]:+.3f}")

    del fit_solver
    torch.cuda.empty_cache()

    print("\n=== [2/3] Baseline eval on VAL split (bias=0) ===")
    val_solver = build_solver(args.config, args.checkpoint, args.device)
    base_overall, base_per_class = run_full_eval(val_solver, [0.0] * num_classes, num_classes)

    print("\n=== [3/3] Calibrated eval on VAL split (fitted bias) ===")
    cal_overall, cal_per_class = run_full_eval(val_solver, bias_vec, num_classes)

    result = {
        "bias_vec": bias_vec,
        "medians_train_subset": medians,
        "n_fit_images": n_seen,
        "baseline_overall_stats": base_overall.tolist(),
        "calibrated_overall_stats": cal_overall.tolist(),
        "baseline_per_class": base_per_class,
        "calibrated_per_class": cal_per_class,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== SUMMARY ===")
    print(f"Overall AP:   {base_overall[0]:.4f} -> {cal_overall[0]:.4f}  (delta {cal_overall[0]-base_overall[0]:+.4f})")
    print(f"Overall AP50: {base_overall[1]:.4f} -> {cal_overall[1]:.4f}  (delta {cal_overall[1]-base_overall[1]:+.4f})")
    print("\nPer-class AP50 (baseline -> calibrated):")
    for c in CLASS_NAMES:
        b = base_per_class[c]["AP50"]
        a = cal_per_class[c]["AP50"]
        print(f"  {c:16s} {b:.4f} -> {a:.4f}  ({a-b:+.4f})")
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()
