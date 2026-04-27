# SOTA Gap Analysis: D-FINE-S vs DroneScan-YOLO

## Current state (2026-04-22)
- D-FINE-S (ours, ep131): 29.7% AP50:95, 10M params
- DroneScan-YOLO (2026): 35.6% AP50:95, ~10M params (YOLOv8s backbone)
- Gap: ~6 AP points

## What DroneScan-YOLO does differently

### Architecture
- **P2 detection head (stride 4)** — adds an extra FPN level for 8×8px objects. Without it,
  a 16px object produces a 2×2 feature at stride 8; with P2 it's 4×4. This is the biggest
  single architectural change.
- **MSFD head**: explicit multi-scale feature distillation at the P2 branch.
- **RPA-Block**: dynamic filter pruning (15-20% sparsity, cosine similarity) — keeps params low
  while adding the P2 branch.

### Loss
- **SAL-NWD loss**: replaces CIoU with Normalized Wasserstein Distance for tiny boxes.
  CIoU gradient → 0 when boxes don't overlap; NWD models boxes as 2D Gaussians so gradients
  are smooth even for non-overlapping 8px objects. Size-adaptive weighting w_i = 1/(A_i + ε)
  amplifies small-object gradients further.
- Biggest reported gains: bicycle AP50 +187%, pedestrian non-detection -40%.

### Training
- Mosaic p=1.0, copy-paste p=0.3, scale jitter ±0.9 — nothing exotic.
- 1280×1280 input (we already do this).

### Inference
- conf_threshold=0.010 (vs 0.5 default), iou_threshold=0.4 — aggressive NMS tuning.
  Prevents suppression of legitimately overlapping tiny objects in dense scenes.
  Grid search gave +0.010 mAP.
- TTA hurt performance on VisDrone (same finding as our ablation).

## What's transferable to D-FINE

| Idea | Effort | Expected gain | Notes |
|------|--------|--------------|-------|
| P2 detection head | High | +1–3 AP | Requires new FPN level in HybridEncoder + extra decoder query set |
| SAL-NWD loss (replace/supplement FGL) | Medium | +1–2 AP | D-FINE uses FGL (fine-grained localization); NWD could replace or supplement for small objects |
| NMS threshold tuning (conf=0.01, iou=0.4) | Low | +0.5–1 AP | No retraining, just eval change |
| Size-adaptive loss weighting | Medium | +0.5–1 AP | Weight FGL/VFL loss by 1/(box_area + ε) |
| Mosaic p=1.0 (we use p=0.5) | Low | minor | Worth trying after current run |

## Recommended next steps (after current mosaic run completes)

1. **NMS tuning** (free, no retraining) — run eval with conf=0.01, iou=0.4 and compare.
2. **Size-adaptive loss weighting** — modify dfine_criterion.py to weight VFL + FGL losses
   by inverse box area for small objects.
3. **P2 head** — longer-term; requires architecture changes to HybridEncoder and DFINETransformer.
4. **SAL-NWD** — could be added as an auxiliary loss alongside existing FGL.

## Reference
- DroneScan-YOLO paper: https://arxiv.org/html/2604.13278
