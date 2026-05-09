# Experiment: NWD matcher + SAL sqrt(1/area)

## Status: COMPLETE — peak AP=0.321

## What this tests
Baseline for all future experiments. Introduces NWD in the Hungarian matcher
and sqrt(1/area) size-adaptive loss weighting. No architecture change.

## Changes vs mosaic baseline (AP=0.316)
- **Matcher:** NWD cost added (`cost_nwd: 2`, `nwd_constant: 0.5`)
- **Loss:** `size_adaptive: True`, weight = `1/sqrt(area)`, normalized to mean=1.
  Applied to: L1 + GIoU + FGL losses after matching.
- **Eval:** maxDets=500 (was 100)

## Starting checkpoint
`output/dfine_hgnetv2_s_visdrone_mosaic_resume/best_stg2.pth` — AP=0.316

## LR schedule: CosineAnnealingLR (no warmup)
```
Epoch   0:  5e-5   (start, no warmup)
Epoch 160:  1e-7   (cosine decay complete, T_max=160)
Backbone:   2.5e-5 throughout (0.5× global)
```
Note: model stalled for ~80 epochs (LR too high for new loss landscape).
Only started improving once cosine decay brought LR below ~1.5e-5.
This is why all subsequent experiments use a warmup from 1e-8 instead.

## Results
| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| 109   | **0.321** | ~0.230 | Peak — best checkpoint |

## Checkpoint
`output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth`

## Config / branch
Config: `configs/dfine/dfine_hgnetv2_s_visdrone_nwd.yml`
Branch: `experiment/nwd-sqrt`
