# Next Experiments Plan (2026-05-09)

## Context

Current best: AP=0.321 from NWD-sqrt run (`best_dfine_s_visdrone_nwd_sqrt_v2.pth`).
Cloud: moving to RunPod RTX A5000 (24GB VRAM, $0.27/hr). AWS quota denied twice.

---

## Loss Understanding (clarified 2026-05-09)

### What the NWD run introduced

**1. NWD in Hungarian matcher** (`cost_nwd: 2`, `nwd_constant: 0.5`)
- NOT a training loss — affects only the matching step
- Converts boxes to 2D Gaussians N(cx, cy, w/2, h/2), computes Wasserstein distance
- `nwd_cost = 1 - exp(-wasserstein / 0.5)`
- For tiny boxes: size-relative matching cost → better assignment than GIoU alone
- GIoU on a 4px box with 2px error ≈ 0 overlap → often unmatched; NWD handles this correctly

**2. Size-adaptive loss weighting** (`size_adaptive: True`, sqrt version)
- After matching: multiply L1 + GIoU + FGL losses by `1/sqrt(area)`, normalized to mean=1
- Amplifies gradients for small objects (max 25× for 4px boxes vs large)
- Linear `1/area` version (625× amplification) caused AP regression (BUG-035)
- Sqrt is the compromise that works

**3. maxDets=500 in eval** — eval fix only, not training

---

## Literature Findings (2026-05-09)

### DroneScan-YOLO (arXiv 2604.13278, Apr 2025) — ablation on VisDrone
- SAL-NWD = Size-Adaptive Loss (1/area, linear) + NWD combined
- SAL-NWD alone: **+9.8 AP50:95** over YOLOv8s baseline
- MSFD (Multi-Scale Feature Distillation): **+10.3 AP50:95** — lightweight P2 detection head
- RPA-Block: redundancy-aware filter pruning during training (dynamic, not structural)
- All three combined: DroneScan-YOLO at AP=0.356 (current SOTA)

### NWD as regression loss (IECA-YOLOv7 ablation)
- NWD alone as regression loss: **+0.8 AP50:95** on drone detection benchmark
- This is where the ~0.8 number the user recalled came from

### HF-D-FINE (ScienceDirect, Nov 2025) — directly relevant
- Uses D-FINE-S on VisDrone with "Outer-SNWD" loss: Shape-IoU + NWD + aspect ratio penalty
- **+3.2 AP** over vanilla D-FINE-S on VisDrone
- Confirms NWD as a regression loss term helps D-FINE specifically

### Shape-IoU (arXiv 2312.17663)
- Replaces CIoU's fixed aspect ratio penalty with scale-aware directional penalty
- For a tall narrow box: penalizes height errors more than width errors
- Generally better than CIoU for non-square objects and tiny objects

---

## Why Our P2 Branch Exploded VRAM

Our P2 branch added P2 as a full 4th level to the DFINETransformer:
- P5 at 1024px: 32×32 = 1,024 positions
- P4: 64×64 = 4,096 positions
- P3: 128×128 = 16,384 positions
- **P2: 256×256 = 65,536 positions (3× more than P3+P4+P5 combined)**

MSDeformableAttention samples across ALL levels → 4× VRAM. The SFEM module
itself (~100K params, depthwise) is negligible — the transformer is the problem.

## MSFD-Style Fix (Planned)

Instead of adding P2 as a 4th transformer level:
1. Extract P2 features (64ch stride-4) from backbone
2. Apply lightweight depthwise separable convs + SE attention on P2
3. Downsample P2 → P3 resolution and fuse into P3
4. Transformer still sees [P3_enhanced, P4, P5] — same 3 levels, zero VRAM increase

P2 small-object information flows through enriched P3. No transformer change needed.
Estimated overhead: ~200K params, negligible VRAM.

---

## Planned Experiments

### Experiment A: MSFD P2 @ 640×640 + SAL(1/area) + NWD matcher
- **Start:** `best_stg1.pth` (AP=0.231, 640px simple augmentation)
- **Why stg1:** smaller distribution shift for new P2 layers; stg1 trained without mosaic
- **LR schedule:** dual-convergence (see below)
- **Resolution:** 640×640 fixed
- **Loss:** `1/area` size-adaptive (linear, not sqrt), NWD in matcher

### Experiment B: MSFD P2 @ 1024×1024 + SAL(1/area) + NWD matcher
- **Start:** `best_dfine_s_visdrone_nwd_sqrt_v2.pth` (AP=0.321)
- **Why this ckpt:** best weights available; already adapted to NWD matcher
- **LR schedule:** dual-convergence (see below)
- **Resolution:** 1024×1024 fixed
- **Loss:** `1/area` size-adaptive (linear), NWD in matcher

---

## Dual-Convergence LR Schedule (for P2 experiments)

Two param groups converge to the same LR over phase 1, then decay together:

```
Phase 1 (epochs 0–50):
  Old weights (backbone + encoder + existing decoder):
    1e-8 → 2e-5  (linear warmup, rising)
  New P2 MSFD weights:
    1e-4 → 2e-5  (cosine decay, falling)
  Both land at 2e-5 at epoch 50.

Phase 2 (epochs 50–110):
  All weights together:
    2e-5 → 1e-7  (cosine decay)
```

**Why small LR for old weights:** NWD matcher weights are strongly adapted;
high LR would destroy that. Start at 1e-8 to protect them.

**Why 1/area is safe here:** At LR=1e-8, even 625× amplification for a 4px box
gives effective LR = 6e-6 — controlled. As LR warms slowly, model adapts to the
1/area landscape before full amplification kicks in.

**Why 1/area instead of sqrt:** DroneScan-YOLO uses 1/area and gains +9.8 AP.
Our sqrt version only gained ~0.5 AP. The linear version exploded previously because
of high starting LR (2.5e-5 × 625 = 0.016 effective LR → gradient chaos).
Slow warmup from 1e-8 eliminates that risk.

---

## Checkpoints Summary

| File | AP | Description |
|------|-----|-------------|
| `output/dfine_hgnetv2_s_visdrone/best_stg1.pth` | 0.231 | 640px, 72 epochs |
| `output/dfine_hgnetv2_s_visdrone_mosaic_resume/best_stg2.pth` | 0.316 | Mosaic 1024px, ep92 |
| `output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth` | **0.321** | NWD+sqrt, best ever |

---

## Cloud Setup

- **Provider:** RunPod (runpod.io)
- **GPU:** RTX A5000 (24GB VRAM, $0.27/hr)
- **Cost estimate:** ~$7–10 total for both experiments sequentially
- **AWS status:** quota denied twice (us-east-1), new request submitted (case 177831472500962, 6 vCPUs) — not worth pursuing further
