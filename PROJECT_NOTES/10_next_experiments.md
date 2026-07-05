> **Superseded (2026-07-05):** current best is now AP=0.3226
> (`output/runpod_results/polish2_last.pth`, see `00_progress.md` Step 25) and
> the live experiment plan/tracker is `11_ablation_study_runpod.md`. This file
> is kept for historical context on ar_aware/p2_640/msfd_1024's original design
> — ar_aware finished (0.318, below its 0.321 starting point, see
> `11_ablation_study_runpod.md` §1 W-AR), p2_640 was never launched, and
> msfd_1024 became the flagship P2ConvHead run (Step 24) before the
> architecture line was shelved (Step 26).

# Next Experiments Plan (updated 2026-05-10)

## Current Best

AP=0.321 — `output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt.pth`
NWD matcher (cost_nwd=2, nwd_constant=0.5) + SAL sqrt (1/sqrt(area)).

---

## Running / Ready

| Exp | Status | Config | Notes |
|-----|--------|--------|-------|
| ar_aware | 🔄 RunPod ep63+ | experiments/ar_aware/ | Rectangular training, 736×1280 / 960×1280 |
| p2_640 | ✅ Ready | experiments/p2_640/ | 4-level transformer at 640×640 |
| msfd_1024 | ✅ Ready | experiments/msfd_1024/ | P2ConvHead (conv-only) at 1024×1024 |

---

## Experiment Design

### ar_aware

VisDrone is ≈50% 16:9 (1280×720–736) and ≈50% 4:3 (1400×1050, 2000×1500).
Square 1024×1024 training wastes pixels on letterbox bars and compresses objects.

Key idea: ARBucketBatchSampler groups images by AR into two buckets.
ARLetterboxCollateFunction resizes each batch to its canonical canvas:
- 16:9 → 736×1280 (h×w)
- 4:3  → 960×1280 (h×w)

No Mosaic/IoUCrop (both destroy AR). Only PhotometricDistort + HFlip + CopyPaste.

### p2_640 — standard DETR P2

Adds backbone stage-0 (64ch, stride-4) as a 4th feature level to the full
DFINETransformer. At 640×640, P2 = 160×160 = 25,600 tokens — manageable.
Tests whether genuine P2 attention improves small-object detection.

Config changes only (no new code):
- `return_idx: [0,1,2,3]`
- `HybridEncoder in_channels: [64,256,512,1024]`
- `DFINETransformer num_levels: 4, feat_strides: [4,8,16,32]`
- `num_points: [2,3,6,3]` — fewer points at P2

### msfd_1024 — YOLOv8-style lightweight P2

Keeps transformer at 3 levels (P3/P4/P5 at 1024px). P2 = 256×256 is handled
by a lightweight conv head (no cross-attention, no 65K-token problem).

P2ConvHead architecture:
  DWBlock(256→128) → DWBlock(128→128)
  ├── cls branch: DWBlock → Conv1×1 → [B, HW, num_classes]
  └── reg branch: DWBlock → Conv1×1 → [B, HW, 4]  (normalized cxcywh)

Loss (FCOS-style TAL):
  For each GT box: all anchors with center inside box are candidates.
  Best candidate = highest IoU with predicted box.
  Loss: VFL (soft target = IoU) + L1 + GIoU for positives.

Postprocessor merges P2 predictions (65K positions) with transformer predictions
(300 queries) before topk-500 NMS selection.

---

## Why we dropped the Dual-Convergence LR schedule

The `10_next_experiments.md` originally proposed a dual-convergence schedule:
old weights warm up 1e-8→2e-5 while new P2 weights decay 1e-4→2e-5, meeting at ep50.

Decision: keep it simple. Both p2_640 and msfd_1024 use the standard
50-ep warmup + 60-ep cosine schedule. The new params (P2 projection + P2ConvHead)
start with zero LR during warmup and gradually increase — the long warmup handles the
new parameter initialization without special scheduling.

If the new params fail to converge, the dual-convergence schedule can be revisited.

---

## Why 1/area (linear SAL) doesn't work

nwd_sal_linear was killed at ep35 (AP 0.321 → 0.315).

1/area amplifies a 4×4px box gradient by 625× vs a 64×64 box.
Even with 50-ep warmup, this creates label-noise explosions on the tiniest boxes.
sqrt(1/area) gives 25× amplification — validated and works.

DroneScan-YOLO uses 1/area but starts with SAL applied only after 20 warmup epochs
at LR<1e-5. Their regime is safer. Replicating it would need a custom lr/loss schedule.

---

## Checkpoints

| File | AP | Description |
|------|----|-------------|
| output/dfine_hgnetv2_s_visdrone/best_stg1.pth | 0.231 | 640px baseline |
| output/dfine_hgnetv2_s_visdrone_mosaic_resume/best_stg2.pth | 0.316 | Mosaic 1024px ep92 |
| output/dfine_hgnetv2_s_visdrone_nwd/best_stg1_dfine_s_visdrone_nwd_sqrt.pth | **0.321** | NWD+sqrt, current best |
| output/ar_aware/last.pth | 0.316 (ep60) | ar_aware in progress |

---

## Cloud

- Provider: RunPod
- GPU: RTX A5000 (24GB VRAM, $0.27/hr) — confirmed working
- Repo: github.com/Danzip/D-fine-visdrone (git pull before starting)
- Dataset: /workspace/D-fine-visdrone/data/visdrone/ (pre-uploaded)
