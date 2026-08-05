# Experiment: NWD matcher + SAL linear (1/area)

## Status: ABANDONED — killed early, regressed vs baseline

## What this tests
Same architecture as nwd_sal_sqrt. Only change: switch size-adaptive weighting
from `1/sqrt(area)` to `1/area` (linear). DroneScan-YOLO uses linear and gains
+9.8 AP. Our sqrt version gained only ~0.5 AP. Linear is more aggressive but
is made safe by starting at very low LR.

## Changes vs nwd_sal_sqrt
- **Loss:** `size_adaptive: True`, weight = `1/area` (not sqrt). Max amplification
  625× on a 4×4px box vs large objects. Safe here because starting LR is 1e-8:
  effective LR on tiny boxes = 1e-8 × 625 = 6.25e-6 (controlled).
  As LR warms slowly the model adapts before full amplification kicks in.
- Everything else identical: NWD matcher, same eval settings.

## Starting checkpoint
`output/dfine_hgnetv2_s_visdrone_nwd/best_dfine_s_visdrone_nwd_sqrt_v2.pth` — AP=0.321

## LR schedule: WarmupCosineHoldLR (single param group — no new layers)
```
Epoch   0:  1e-8   (start — protect NWD-adapted weights)
Epoch  50:  2e-5   (peak after linear warmup)
Epoch 110:  1e-7   (cosine decay complete)
Epoch 110+: 1e-7   (hold)
```
Backbone LR = 0.5× global LR throughout.

## Files
- `config.yml` — TBD
- `watchdog.sh` — TBD

## Results
| Epoch | AP50:95 | AP-small | Notes |
|-------|---------|----------|-------|
| 35 (killed) | 0.315 | - | regressed from the 0.321 starting point — 1/area linear SAL amplifies a 4×4px box gradient 625× vs a 64×64 box (sqrt's 25×), causing label-noise explosions on the tiniest boxes; see `PROJECT_NOTES/10_next_experiments.md` |
